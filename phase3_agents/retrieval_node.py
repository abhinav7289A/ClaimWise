"""The retrieval agent — Phase 2's pipeline wrapped as a graph node.

**What this solves.** Phases 1 and 2 retrieved one way: embed the question,
search the whole collection, rerank, hand back the top 5. That is correct for a
question answered by one passage in one document. It is measurably wrong for a
question that compares two policies, and this module exists because we measured
it rather than assumed it.

**The measurement (2026-08-20, `agent_tasks.jsonl`, 12 comparison tasks).**

| | chunk-policy | parent-docs |
|---|---|---|
| any ground-truth ref in top 5 | 0.750 | 0.750 |
| **every required document covered at top 5** | 0.250 | 0.333 |
| every required document covered at **top 20** | 0.417 | 0.500 |

Read the third row first. Twenty is the *entire* candidate pool at rerank depth
20 — so for half the comparison tasks the second policy's page is never fetched
at all. A better reranker cannot promote a passage that was never retrieved, and
a deeper cut cannot reach past the end of the list. The failure is in the
*fetch*, which means the fix has to be in the fetch.

**Why one global search loses a document.** The search returns the globally
nearest N chunks. Nothing in that procedure reserves room for a second document,
so when one policy's wording happens to match the question a little better —
which for near-duplicate insurance boilerplate is close to a coin toss — it can
take every slot. The answer then looks confident and is half-wrong: it compares
one policy against nothing. Six tasks in the run above did exactly that.

**The fix: per-document retrieval on the comparison route.** Enumerate the user's
documents, run one filtered search *inside each*, rerank within each, and reserve
slots per document when merging. Every document is then guaranteed to contribute
its best evidence, and the cross-encoder decides the order among them. This is
what a search team calls *stratified* or *quota* retrieval, and it is the
standard answer whenever recall must be spread across groups rather than pooled.

**Why not simply retrieve more and stratify afterwards.** That was the first
design, and the @20 row above kills it: the missing pages are not deeper in the
pool, they are outside it. Post-hoc grouping can only redistribute what the
fetch already found.

**The routes that are not comparison keep the global path**, unchanged, because
nothing measured says otherwise — one change, one eval.

**`out_of_scope` still retrieves.** It would be cheaper to refuse without
searching, but the router's out-of-scope recall is 0.333 on the golden set, so
refusing blind would discard genuine questions on a misroute. Retrieving and
letting the confidence gate decide is the measured composition (15/15 on those
golden items). Note the caveat recorded the same day: on the independent task
set the gate caught only 4 of 10 negatives, so this composition is weaker than
D-23 suggested and is the reason the gate threshold is still open.

Usage:
    python -m phase3_agents.retrieval_node --help
    python -m phase3_agents.retrieval_node --self-test
    python -m phase3_agents.retrieval_node --question "..." --route comparison
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from phase1_rag.embed_index import collection_name_for
from phase1_rag.rag_chain import RetrievedChunk, build_search_filter, retrieve
from phase2_advanced.document_resolver import Resolution, resolve_documents, scoped_filter
from phase2_advanced.parent_docs import ParentStore, load_parent_store
from phase2_advanced.rerank import CrossEncoderReranker, build_reranker

LOGGER = logging.getLogger("claimwise.retrieval_node")


@dataclass(frozen=True)
class DocumentRef:
    """One document available to a user.

    Attributes:
        doc_id: Content hash of the source PDF — the filter key.
        filename: Source PDF filename, for logging and for the trace.
        insurer: e.g. "starhealth".
        policy_type: e.g. "health".
        doc_label: Free-text label.
    """

    doc_id: str
    filename: str
    insurer: str
    policy_type: str
    doc_label: str


@dataclass
class RetrievalResources:
    """Everything the node needs, loaded once and reused across questions.

    Built once because the cost is all in the constructors: bge-small and
    bge-reranker-base take seconds to load and hundreds of megabytes to hold.
    A node that rebuilt them per question would spend more time loading models
    than retrieving, and on a server would do it per request.

    Attributes:
        client: An open Qdrant client.
        collection_name: Collection to search.
        embedder: Query embedding model, matching the index.
        reranker: Cross-encoder applied after retrieval. Required, not optional
            — the confidence gate reads `retrieved[0].score` and P-14 measured
            the bi-encoder score as unusable for that purpose (0.6687 on
            unanswerable questions, inside the range genuine ones occupy).
        parents: Parent store when the collection has a parent tier.
        settings: Resolved retrieval settings.
        scope_documents: When True, each question is resolved to a document
            subset (Phase 2 technique 5) before retrieval. Off by default so
            every number recorded before 2026-09-01 reproduces byte-identically.
        document_cache: Per-user document lists, filled on first use.
    """

    client: QdrantClient
    collection_name: str
    embedder: SentenceTransformer
    reranker: CrossEncoderReranker
    parents: ParentStore | None
    settings: dict[str, Any]
    scope_documents: bool = False
    document_cache: dict[str, list[DocumentRef]] = field(default_factory=dict)

    def close(self) -> None:
        """Release the Qdrant client. Safe to call more than once."""
        try:
            self.client.close()
        except Exception:  # noqa: BLE001 - closing twice must never mask a real error
            LOGGER.debug("Qdrant client already closed")


def resolve_settings(config: dict[str, Any], pipeline: str | None = None) -> dict[str, Any]:
    """Resolve the retrieval settings for the node.

    Args:
        config: Parsed `config.yaml`.
        pipeline: "chunk_policy" or "parent_docs". Defaults to the configured
            `retrieval_agent.pipeline`.

    Returns:
        Resolved settings.

    Raises:
        ValueError: If `pipeline` is not one of the two known pipelines.
    """
    pipeline = pipeline or cfg_get(config, "retrieval_agent.pipeline", "chunk_policy")
    if pipeline not in {"chunk_policy", "parent_docs"}:
        raise ValueError(
            f"Unknown pipeline {pipeline!r}. Expected 'chunk_policy' (D-19, adopted on "
            "the golden set) or 'parent_docs' (better on agent_tasks, 2026-08-20)."
        )

    if pipeline == "chunk_policy":
        collection_prefix = cfg_get(config, "chunk_policy.collection_prefix", "claimwise_mx")
        parents_path = Path(
            cfg_get(config, "chunk_policy.parents_path", "data/processed/mixed_parents.jsonl")
        )
    else:
        collection_prefix = cfg_get(config, "parent_docs.collection_prefix", "claimwise_pd")
        parents_path = Path(
            cfg_get(config, "parent_docs.parents_path", "data/processed/parents.jsonl")
        )

    return {
        "pipeline": pipeline,
        "collection_prefix": collection_prefix,
        "parents_path": parents_path,
        "embed_model": cfg_get(config, "embed.model_name"),
        "query_prefix": cfg_get(config, "embed.query_prefix", ""),
        "normalize": cfg_get(config, "embed.normalize", True),
        "device": cfg_get(config, "embed.device", "cpu"),
        "qdrant_path": cfg_get(config, "index.path", "qdrant_storage"),
        "default_user_id": cfg_get(config, "index.default_user_id", "local-dev"),
        # Global route: fetch this many, rerank all of them, keep top_k.
        "candidate_depth": cfg_get(config, "rerank.candidate_depth", 20),
        "top_k": cfg_get(config, "rag.top_k", 5),
        # Comparison route.
        "per_doc_depth": cfg_get(config, "retrieval_agent.per_doc_depth", 10),
        "per_doc_top_k": cfg_get(config, "retrieval_agent.per_doc_top_k", 2),
        "comparison_top_k": cfg_get(config, "retrieval_agent.comparison_top_k", 6),
        "max_documents": cfg_get(config, "retrieval_agent.max_documents", 4),
        "document_relevance_floor": cfg_get(
            config, "retrieval_agent.document_relevance_floor", 0.15
        ),
    }


def build_resources(
    config: dict[str, Any],
    pipeline: str | None = None,
    scope_documents: bool = False,
) -> RetrievalResources:
    """Load the models, open the index, and check the collection exists.

    Args:
        config: Parsed `config.yaml`.
        pipeline: Override `retrieval_agent.pipeline`.
        scope_documents: Enable question-to-document scoping (technique 5).

    Returns:
        Ready-to-use resources. The caller owns them and must call `.close()`.

    Raises:
        FileNotFoundError: If the collection does not exist. Raised loudly rather
            than degrading to an empty result, because an empty result is
            indistinguishable from "nothing matched" and would be silently
            reported as a retrieval failure.
    """
    settings = resolve_settings(config, pipeline)
    embedder = SentenceTransformer(settings["embed_model"], device=settings["device"])
    collection_name = collection_name_for(settings["collection_prefix"], settings["embed_model"])

    client = QdrantClient(path=settings["qdrant_path"])
    if not client.collection_exists(collection_name):
        client.close()
        raise FileNotFoundError(
            f"Collection {collection_name} does not exist. Build it with "
            f"`python -m phase1_rag.embed_index --collection-prefix "
            f"{settings['collection_prefix']}` first."
        )

    parents = load_parent_store(config, parents_path=settings["parents_path"])
    reranker = build_reranker(config)

    LOGGER.info(
        "Retrieval node ready: pipeline=%s collection=%s parents=%d",
        settings["pipeline"],
        collection_name,
        len(parents),
    )
    return RetrievalResources(
        client=client,
        collection_name=collection_name,
        embedder=embedder,
        reranker=reranker,
        parents=parents,
        settings=settings,
        scope_documents=scope_documents,
    )


def document_filter(
    user_id: str,
    doc_id: str,
    insurer: str | None = None,
    policy_type: str | None = None,
) -> models.Filter:
    """Build a filter scoped to one document, inside one user's data.

    Written here rather than by extending `rag_chain.build_search_filter` so the
    Phase 1 and 2 code paths stay byte-identical and every recorded metric stays
    reproducible. `user_id` is still first and still mandatory — narrowing to a
    document must never widen access.

    Args:
        user_id: Owner whose documents may be searched.
        doc_id: The single document to search inside.
        insurer: Optional additional filter.
        policy_type: Optional additional filter.

    Returns:
        A filter requiring the user, the document, and any extra conditions.
    """
    base = build_search_filter(user_id, insurer=insurer, policy_type=policy_type)
    conditions = list(base.must or [])
    conditions.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
    return models.Filter(must=conditions)


def list_documents(resources: RetrievalResources, user_id: str) -> list[DocumentRef]:
    """Enumerate the distinct documents a user has indexed.

    Reads from the index rather than from `data/processed/`, because the index is
    what retrieval actually searches. A document present on disk but never
    embedded would otherwise be handed to the comparison route as a target it can
    never fetch from.

    Cached per user for the lifetime of the resources. A scroll over the whole
    collection is cheap at this corpus size but is not per-question work, and at
    serving time the document list changes only on upload.

    Args:
        resources: Loaded retrieval resources.
        user_id: Owner whose documents to list.

    Returns:
        The user's documents, ordered by filename for stable output.
    """
    if user_id in resources.document_cache:
        return resources.document_cache[user_id]

    seen: dict[str, DocumentRef] = {}
    offset: Any = None
    while True:
        points, offset = resources.client.scroll(
            collection_name=resources.collection_name,
            scroll_filter=build_search_filter(user_id),
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = DocumentRef(
                    doc_id=doc_id,
                    filename=payload.get("filename", ""),
                    insurer=payload.get("insurer", ""),
                    policy_type=payload.get("policy_type", ""),
                    doc_label=payload.get("doc_label", ""),
                )
        if offset is None:
            break

    documents = sorted(seen.values(), key=lambda doc: doc.filename)
    resources.document_cache[user_id] = documents
    LOGGER.info("User %s has %d indexed document(s)", user_id, len(documents))
    return documents


def apply_relevance_floor(
    per_document: dict[str, list[RetrievedChunk]],
    fraction: float,
) -> tuple[dict[str, list[RetrievedChunk]], list[str]]:
    """Drop documents too weakly related to the question to be worth a quota.

    A quota guarantees each document a share of the prompt. That is exactly what
    a comparison needs among documents that *could* answer, and exactly wrong
    among documents that cannot: every slot spent on an irrelevant policy is a
    slot taken from a relevant one.

    Relative rather than absolute. Cross-encoder scores move with question
    phrasing, so no fixed cut point holds across questions; the ratio between
    documents scored on the *same* question is the stable comparison. The
    best-scoring document is always kept, so this can never return nothing.

    Args:
        per_document: doc_id to its reranked chunks, best first.
        fraction: Keep a document when its top score is at least this fraction of
            the best document's top score. 0.0 disables the floor.

    Returns:
        `(kept, dropped_doc_ids)`.
    """
    if not per_document or fraction <= 0:
        return per_document, []

    scores = {
        doc_id: (chunks[0].score if chunks else float("-inf"))
        for doc_id, chunks in per_document.items()
    }
    best = max(scores.values())
    # A negative or zero best means nothing scored meaningfully at all. Scaling a
    # floor off it would invert the test, so the floor stands down and the
    # confidence gate handles it downstream — which is its job, not this one's.
    if best <= 0:
        return per_document, []

    floor = best * fraction
    kept = {doc_id: chunks for doc_id, chunks in per_document.items() if scores[doc_id] >= floor}
    dropped = sorted(doc_id for doc_id in per_document if doc_id not in kept)
    return kept, dropped


def select_per_document(
    per_document: dict[str, list[RetrievedChunk]],
    per_doc_top_k: int,
    total_k: int,
) -> list[RetrievedChunk]:
    """Merge per-document results while guaranteeing each document is represented.

    This is the whole fix, and it is deliberately a pure function so it can be
    tested without Qdrant or a GPU.

    Two passes:

    1. **Reserved.** Round-robin up to `per_doc_top_k` from each document, best
       first. Round-robin rather than "all of document A, then all of B" so that
       a `total_k` too small to satisfy every document still spreads what it has
       instead of filling up on whichever document sorted first.
    2. **Fill.** Any remaining slots go to the best leftovers by score, wherever
       they came from. A comparison question whose answer really does live mostly
       in one policy still gets depth there — the quota is a floor, not a cap.

    The result is sorted by score so `retrieved[0]` is the single most relevant
    passage, which is what the confidence gate reads.

    Args:
        per_document: doc_id to its reranked chunks, best first.
        per_doc_top_k: Slots reserved for each document.
        total_k: Total chunks to return.

    Returns:
        The merged chunks, best first.
    """
    if total_k <= 0:
        return []

    ordered_docs = sorted(
        per_document.items(),
        key=lambda pair: pair[1][0].score if pair[1] else float("-inf"),
        reverse=True,
    )

    selected: list[RetrievedChunk] = []
    taken: dict[str, int] = {doc_id: 0 for doc_id, _ in ordered_docs}
    for round_index in range(max(0, per_doc_top_k)):
        for doc_id, chunks in ordered_docs:
            if len(selected) >= total_k:
                break
            if round_index < len(chunks):
                selected.append(chunks[round_index])
                taken[doc_id] += 1
        if len(selected) >= total_k:
            break

    if len(selected) < total_k:
        leftovers = [
            chunk
            for doc_id, chunks in ordered_docs
            for chunk in chunks[taken[doc_id] :]
        ]
        leftovers.sort(key=lambda chunk: chunk.score, reverse=True)
        selected.extend(leftovers[: total_k - len(selected)])

    selected.sort(key=lambda chunk: chunk.score, reverse=True)
    return selected


def retrieve_global(
    resources: RetrievalResources,
    question: str,
    user_id: str,
    insurer: str | None = None,
    policy_type: str | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the Phase 2 way: one search over everything, rerank, cut.

    Unchanged from the pipeline every Phase 2 metric was measured on — dense
    fetch at `candidate_depth`, cross-encoder reorder, parent expansion after
    reranking (D-18).

    Args:
        resources: Loaded retrieval resources.
        question: The user's question.
        user_id: Security boundary.
        insurer: Optional metadata filter.
        policy_type: Optional metadata filter.
        top_k: Chunks to return. Defaults to `rag.top_k`.

    Returns:
        Chunks best first, carrying cross-encoder scores.
    """
    settings = resources.settings
    top_k = top_k or settings["top_k"]

    chunks = retrieve(
        resources.client,
        collection_name=resources.collection_name,
        embedder=resources.embedder,
        question=question,
        top_k=settings["candidate_depth"],
        query_prefix=settings["query_prefix"],
        normalize=settings["normalize"],
        search_filter=build_search_filter(user_id, insurer=insurer, policy_type=policy_type),
    )
    # Stage order is load-bearing and copied from `rag_chain.answer_question`
    # (D-18): rerank the whole pool, expand the winners, THEN truncate. Reranking
    # with a top_k here instead would truncate before expansion, and since
    # expansion deduplicates, the prompt could end up with fewer than top_k
    # passages — the served pipeline would then differ from the evaluated one.
    chunks = resources.reranker.rerank(question, chunks)
    if resources.parents is not None:
        chunks = resources.parents.expand(chunks)
    return chunks[:top_k]


def resolve_scope(
    resources: RetrievalResources,
    question: str,
    user_id: str,
) -> Resolution | None:
    """Resolve a question to a document subset, or None if scoping is off.

    Thin wrapper so exactly one place decides whether scoping applies. Returns
    None — not an empty `Resolution` — when the feature is disabled, so a caller
    can tell "scoping is off" from "scoping ran and narrowed nothing"; the two
    take the same retrieval path but must be logged differently.

    Args:
        resources: Loaded retrieval resources.
        question: The user's question.
        user_id: Owner whose documents are candidates.

    Returns:
        The resolution, or None when `resources.scope_documents` is False.
    """
    if not resources.scope_documents:
        return None
    resolution = resolve_documents(question, list_documents(resources, user_id))
    LOGGER.info(
        "Scope: %d document(s) via %s%s",
        len(resolution.doc_ids),
        resolution.strength,
        " (ambiguous)" if resolution.ambiguous else "",
    )
    return resolution


def retrieve_scoped(
    resources: RetrievalResources,
    question: str,
    user_id: str,
    doc_ids: Sequence[str],
    insurer: str | None = None,
    policy_type: str | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """`retrieve_global`, restricted to a resolved subset of documents.

    Deliberately a separate function rather than a parameter on
    `retrieve_global`. Every Phase 2 number was measured on that function, and a
    new branch inside it would make "reproduces the recorded metric" depend on an
    argument default — the kind of coupling that turns a regression into an
    argument about which run was which. Stage order is copied verbatim from it
    (rerank the pool, expand, then truncate) so the only difference between the
    two is the filter.

    Args:
        resources: Loaded retrieval resources.
        question: The user's question.
        user_id: Security boundary. Still applied first — narrowing to a
            document subset must never widen access.
        doc_ids: Documents to search inside. Must be non-empty; callers with an
            empty resolution should be calling `retrieve_global`.
        insurer: Optional metadata filter.
        policy_type: Optional metadata filter.
        top_k: Chunks to return. Defaults to `rag.top_k`.

    Returns:
        Chunks best first, carrying cross-encoder scores.

    Raises:
        ValueError: If `doc_ids` is empty — silently searching everything would
            report a scoped run's numbers under an unscoped pipeline.
    """
    if not doc_ids:
        raise ValueError(
            "retrieve_scoped called with no doc_ids. An empty resolution means "
            "'search everything', which is retrieve_global — call that instead."
        )

    settings = resources.settings
    top_k = top_k or settings["top_k"]

    # Built here rather than by calling `scoped_filter`, which starts from a
    # bare `build_search_filter(user_id)` and so cannot carry the insurer /
    # policy_type conditions this path also honours. Same MatchAny clause, added
    # on top of the full base filter instead of a partial one.
    conditions = list(
        build_search_filter(user_id, insurer=insurer, policy_type=policy_type).must or []
    )
    conditions.append(
        models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))
    )

    chunks = retrieve(
        resources.client,
        collection_name=resources.collection_name,
        embedder=resources.embedder,
        question=question,
        top_k=settings["candidate_depth"],
        query_prefix=settings["query_prefix"],
        normalize=settings["normalize"],
        search_filter=models.Filter(must=conditions),
    )
    chunks = resources.reranker.rerank(question, chunks)
    if resources.parents is not None:
        chunks = resources.parents.expand(chunks)
    return chunks[:top_k]


def retrieve_per_document(
    resources: RetrievalResources,
    question: str,
    user_id: str,
    insurer: str | None = None,
    policy_type: str | None = None,
    total_k: int | None = None,
    doc_ids: Sequence[str] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve inside every document separately, then merge with quotas.

    One search per document, each reranked on its own. Reranking within a
    document rather than over the pooled candidates is what makes the merged
    scores mean the same thing — the cross-encoder scores a (question, passage)
    pair independently, so scores are comparable across documents as long as
    every document was scored against the same question.

    **`doc_ids` is the 2026-09-01 change.** Without it this fans out to every
    document the user owns and then uses the relevance floor to *guess* which of
    them the question was about. When the resolver has already read the answer
    out of the question — "compare my Star plan against my SBI plan" names both
    — that guess is strictly worse than the fact. Passing the resolved subset
    replaces guessing with knowing, and shrinks the fan-out, so each surviving
    document gets a larger share of `comparison_top_k`.

    Args:
        resources: Loaded retrieval resources.
        question: The user's question.
        user_id: Security boundary.
        insurer: Optional metadata filter.
        policy_type: Optional metadata filter.
        total_k: Chunks to return. Defaults to `retrieval_agent.comparison_top_k`.
        doc_ids: Restrict the fan-out to these documents. None (the default)
            keeps the measured 2026-08-20 behaviour of comparing everything.

    Returns:
        Chunks best first, with each document guaranteed its quota.
    """
    settings = resources.settings
    total_k = total_k or settings["comparison_top_k"]

    documents = list_documents(resources, user_id)
    if policy_type:
        documents = [doc for doc in documents if doc.policy_type == policy_type]
    if insurer:
        documents = [doc for doc in documents if doc.insurer == insurer]
    if doc_ids:
        wanted = set(doc_ids)
        # Intersect rather than trust. A doc_id that is not in this user's list
        # is either stale or someone else's; either way it must not become a
        # search target, so the user's own list stays the authority.
        scoped = [doc for doc in documents if doc.doc_id in wanted]
        if scoped:
            documents = scoped
        else:
            LOGGER.warning(
                "Resolved doc_ids matched none of user %s's %d document(s) — "
                "comparing all of them instead.",
                user_id,
                len(documents),
            )

    if not documents:
        LOGGER.warning("No indexed documents for user %s — nothing to compare", user_id)
        return []

    # A comparison over many documents costs one search and one cross-encoder
    # pass each, so the cap is a latency guard. Ordering by filename keeps the
    # truncation deterministic rather than dependent on scroll order.
    if len(documents) > settings["max_documents"]:
        LOGGER.warning(
            "User %s has %d documents; comparing the first %d. Raise "
            "retrieval_agent.max_documents to widen this.",
            user_id,
            len(documents),
            settings["max_documents"],
        )
        documents = documents[: settings["max_documents"]]

    per_document: dict[str, list[RetrievedChunk]] = {}
    for doc in documents:
        candidates = retrieve(
            resources.client,
            collection_name=resources.collection_name,
            embedder=resources.embedder,
            question=question,
            top_k=settings["per_doc_depth"],
            query_prefix=settings["query_prefix"],
            normalize=settings["normalize"],
            search_filter=document_filter(
                user_id, doc.doc_id, insurer=insurer, policy_type=policy_type
            ),
        )
        if candidates:
            per_document[doc.doc_id] = resources.reranker.rerank(question, candidates)

    # Relevance floor BEFORE the quota, because the quota's cost is paid in
    # slots: an irrelevant document does not merely add noise, it evicts a
    # relevant document's next-best passage.
    #
    # ...unless the caller named the documents. The floor exists to guess which
    # of the user's policies a comparison is about; when the question said so
    # outright, dropping one of the two named documents is not noise control, it
    # is discarding half the comparison. Evidence: the floor was tuned on a
    # question that fanned out to all four documents (config.yaml), a situation
    # scoping removes rather than mitigates.
    floor = settings["document_relevance_floor"]
    if doc_ids and len(documents) > 1:
        floor = 0.0
    per_document, dropped = apply_relevance_floor(per_document, floor)
    if dropped:
        names = {doc.doc_id: doc.filename for doc in documents}
        LOGGER.info(
            "Dropped %d document(s) below the relevance floor: %s",
            len(dropped),
            ", ".join(names.get(doc_id, doc_id) for doc_id in dropped),
        )

    # Quotas are applied to the CHILDREN, then the winners are expanded. The
    # other order would let one parent block absorb two documents' quota slots
    # into a single deduplicated entry and quietly undo the guarantee. Expansion
    # can still shrink the list — that is deduplication working, not evidence
    # lost, since a dropped child is already inside a parent that survived.
    merged = select_per_document(per_document, settings["per_doc_top_k"], total_k)
    if resources.parents is not None:
        merged = resources.parents.expand(merged)
    return merged


def retrieve_node(state: dict[str, Any], resources: RetrievalResources) -> dict[str, Any]:
    """Apply retrieval as a graph node.

    Returns a partial state update rather than a mutated state, which is the
    shape LangGraph merges — the same contract `gate_node` already follows.

    The route decides the strategy, and that is the only thing the route decides
    here. Confidence is written by the gate, not by this node, so there is
    exactly one place that turns a score into a verdict.

    Args:
        state: The current `AgentState`. Reads `question`, `user_id`, `route`,
            and the optional metadata filters.
        resources: Loaded retrieval resources.

    Returns:
        Keys to merge into the state.
    """
    question = state.get("rewritten_question") or state["question"]
    user_id = state.get("user_id") or resources.settings["default_user_id"]
    route = state.get("route", "lookup")

    started = time.perf_counter()
    resolution = resolve_scope(resources, question, user_id)
    scoped_ids = resolution.doc_ids if resolution else ()

    if route == "comparison":
        # Scoping feeds the strata; it does not replace them. Even with a
        # resolution the fan-out and the quota still run — the resolution only
        # says *which* documents get a quota.
        chunks = retrieve_per_document(
            resources,
            question,
            user_id,
            insurer=state.get("insurer"),
            policy_type=state.get("policy_type"),
            doc_ids=scoped_ids or None,
        )
        node_name = "retrieve_per_document"
    elif scoped_ids:
        chunks = retrieve_scoped(
            resources,
            question,
            user_id,
            scoped_ids,
            insurer=state.get("insurer"),
            policy_type=state.get("policy_type"),
        )
        node_name = "retrieve_scoped"
    else:
        chunks = retrieve_global(
            resources,
            question,
            user_id,
            insurer=state.get("insurer"),
            policy_type=state.get("policy_type"),
        )
        node_name = "retrieve_global"
    elapsed = time.perf_counter() - started

    documents_reached = sorted({chunk.doc_id for chunk in chunks})
    LOGGER.info(
        "%s: %d chunk(s) from %d document(s) in %.2fs",
        node_name,
        len(chunks),
        len(documents_reached),
        elapsed,
    )

    update: dict[str, Any] = {
        "retrieved": chunks,
        "trace": [*state.get("trace", []), node_name],
    }

    # Recorded on the state so the eval and the served trace can report the
    # resolution mix — how often scoping fired, and on what signal — without
    # re-running the resolver and risking a different answer than the one the
    # retrieval actually used.
    if resolution is not None:
        update["scope"] = {
            "doc_ids": list(resolution.doc_ids),
            "strength": resolution.strength,
            "matched_on": list(resolution.matched_on),
            "ambiguous": resolution.ambiguous,
        }

    # An ambiguous resolution means one signal matched several documents and the
    # user probably meant exactly one. Retrieval widens rather than guesses, and
    # says so, because a silently-widened search that lands on the wrong policy
    # is the failure this whole layer exists to prevent.
    if resolution is not None and resolution.ambiguous and route != "comparison":
        update["assumptions"] = [
            *state.get("assumptions", []),
            f"Question matched {len(resolution.doc_ids)} of your documents "
            f"({', '.join(resolution.matched_on)}); all of them were searched.",
        ]

    # A comparison that reached one document is answerable only in part. Recorded
    # as an assumption rather than silently accepted, because the measured
    # failure mode is a confident half-answer, and an unqualified one is worse
    # than a qualified one.
    if route == "comparison" and len(documents_reached) < 2:
        update["assumptions"] = [
            *state.get("assumptions", []),
            "Comparison requested but evidence was found in only "
            f"{len(documents_reached)} document — the comparison is incomplete.",
        ]

    return update


# --- Self-test ---------------------------------------------------------------


def _fake_chunk(doc_id: str, page: int, score: float) -> RetrievedChunk:
    """Build a minimal chunk for the pure-function tests.

    Args:
        doc_id: Document id.
        page: Page number.
        score: Relevance score.

    Returns:
        A `RetrievedChunk` with only the fields the merge logic reads.
    """
    return RetrievedChunk(
        chunk_id=f"{doc_id}_p{page}",
        doc_id=doc_id,
        filename=f"{doc_id}.pdf",
        insurer="test",
        policy_type="health",
        doc_label="test",
        page=page,
        text="",
        score=score,
    )


def _self_test() -> list[tuple[str, bool, str]]:
    """Exercise the merge logic without Qdrant, models or network.

    Covers the quota behaviour that the whole module exists for, so a regression
    in it fails here in milliseconds rather than in a 4-minute eval run.

    Returns:
        One (name, passed, detail) per check.
    """
    results: list[tuple[str, bool, str]] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        """Record one comparison."""
        passed = actual == expected
        results.append((name, passed, f"expected {expected!r}, got {actual!r}"))

    # The failure this module was built to fix: document A outscores document B
    # at every rank, so a global top-5 by score returns only A.
    strong = [_fake_chunk("A", page, score) for page, score in ((1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6))]
    weak = [_fake_chunk("B", page, score) for page, score in ((1, 0.5), (2, 0.4), (3, 0.3))]

    global_style = sorted(strong + weak, key=lambda c: c.score, reverse=True)[:4]
    check(
        "global top-4 reaches only one document (the defect)",
        sorted({c.doc_id for c in global_style}),
        ["A"],
    )

    merged = select_per_document({"A": strong, "B": weak}, per_doc_top_k=2, total_k=4)
    check("quota merge reaches both documents", sorted({c.doc_id for c in merged}), ["A", "B"])
    check("quota merge honours total_k", len(merged), 4)
    check("quota merge returns best first", [c.score for c in merged], [0.9, 0.8, 0.5, 0.4])
    check(
        "each document gets exactly its quota when slots allow",
        sorted(c.doc_id for c in merged),
        ["A", "A", "B", "B"],
    )

    # Fill pass: quota is a floor, not a cap. With 2 reserved each and 5 slots,
    # the spare slot goes to the best leftover, which is A's third chunk.
    filled = select_per_document({"A": strong, "B": weak}, per_doc_top_k=2, total_k=5)
    check("spare slots go to the best leftover", sorted(c.doc_id for c in filled), ["A", "A", "A", "B", "B"])

    # Over-subscribed: fewer slots than documents times quota. Round-robin must
    # still spread rather than filling on the first document.
    tight = select_per_document({"A": strong, "B": weak}, per_doc_top_k=2, total_k=2)
    check("over-subscribed merge still spreads", sorted(c.doc_id for c in tight), ["A", "B"])

    # A document that returned nothing must not break the merge or claim a slot.
    with_empty = select_per_document({"A": strong, "B": weak, "C": []}, per_doc_top_k=2, total_k=4)
    check("empty document is skipped", sorted({c.doc_id for c in with_empty}), ["A", "B"])

    check("total_k of zero returns nothing", select_per_document({"A": strong}, 2, 0), [])
    check("no documents returns nothing", select_per_document({}, 2, 5), [])

    # Single document: the merge must degrade to plain top-k, so the comparison
    # route is safe on a user who has uploaded only one policy.
    single = select_per_document({"A": strong}, per_doc_top_k=2, total_k=3)
    check("single document degrades to top-k", [c.score for c in single], [0.9, 0.8, 0.7])

    # The relevance floor, reproducing the 2026-08-20 live failure: a health
    # comparison fanned out to a life and a home policy, which took 2 of 6 slots.
    live = {
        "star": [_fake_chunk("star", 31, 0.91)],
        "sbih": [_fake_chunk("sbih", 20, 0.62)],
        "life": [_fake_chunk("life", 8, 0.04)],
        "home": [_fake_chunk("home", 9, 0.02)],
    }
    kept, dropped = apply_relevance_floor(live, fraction=0.15)
    check("floor keeps the relevant documents", sorted(kept), ["sbih", "star"])
    check("floor drops the irrelevant ones", dropped, ["home", "life"])

    check(
        "floor of zero disables the check",
        sorted(apply_relevance_floor(live, fraction=0.0)[0]),
        ["home", "life", "sbih", "star"],
    )
    # The best document must survive any fraction, or a comparison could be
    # emptied by its own filter.
    check(
        "best document always survives",
        sorted(apply_relevance_floor(live, fraction=1.0)[0]),
        ["star"],
    )
    check("floor on empty input returns empty", apply_relevance_floor({}, 0.15), ({}, []))

    # Uniformly weak scores mean the floor has no signal to work with; refusing
    # is the gate's job, not this function's.
    weak = {"a": [_fake_chunk("a", 1, 0.004)], "b": [_fake_chunk("b", 1, 0.003)]}
    check("weak-but-comparable documents both survive", sorted(apply_relevance_floor(weak, 0.15)[0]), ["a", "b"])

    # The floor frees slots, and the freed slots are the point: with life and
    # home dropped, six slots across two documents is three each — enough depth
    # for the rank-3 chunk that was being evicted.
    star_deep = [_fake_chunk("star", page, score) for page, score in ((31, 0.91), (30, 0.70), (32, 0.50))]
    sbih_deep = [_fake_chunk("sbih", page, score) for page, score in ((21, 0.62), (12, 0.40), (20, 0.35))]
    before, _ = apply_relevance_floor({**live, "star": star_deep, "sbih": sbih_deep}, 0.0)
    after, _ = apply_relevance_floor({**live, "star": star_deep, "sbih": sbih_deep}, 0.15)
    check(
        "without the floor the rank-3 page is evicted",
        20 in [c.page for c in select_per_document(before, per_doc_top_k=2, total_k=6)],
        False,
    )
    # Same quota as the live config (2), so this is the real behaviour change
    # and not an artifact of the test picking a friendlier number: with only two
    # documents left, the fill pass hands the spare slots back to them.
    check(
        "with the floor the rank-3 page survives",
        20 in [c.page for c in select_per_document(after, per_doc_top_k=2, total_k=6)],
        True,
    )

    # The filter must never widen access: user_id survives document scoping.
    scoped = document_filter("user-1", "doc-abc")
    keys = sorted(condition.key for condition in scoped.must)
    check("document filter keeps the user boundary", keys, ["doc_id", "user_id"])

    # --- 2026-09-01: scoping feeds the strata -------------------------------
    # The multi-document filter must keep the boundary too, and must use MatchAny
    # rather than one MatchValue per document — the latter is an AND, which
    # matches nothing, and is the silent way this feature could return zero
    # chunks and be read as "the documents have no relevant pages".
    multi = scoped_filter("user-1", ["doc-a", "doc-b"])
    check(
        "scoped filter keeps the user boundary",
        sorted(condition.key for condition in multi.must),
        ["doc_id", "user_id"],
    )
    doc_condition = next(c for c in multi.must if c.key == "doc_id")
    check(
        "scoped filter matches ANY of the documents",
        sorted(doc_condition.match.any),
        ["doc-a", "doc-b"],
    )

    # `retrieve_scoped` must refuse an empty subset rather than quietly becoming
    # `retrieve_global`. The whole A/B depends on knowing which one ran.
    try:
        retrieve_scoped(None, "q", "user-1", [])  # type: ignore[arg-type]
        check("empty scope is refused, not widened", "no error", "ValueError")
    except ValueError:
        check("empty scope is refused, not widened", "ValueError", "ValueError")
    except Exception as exc:  # noqa: BLE001 - any other error means the guard is not first
        check("empty scope is refused, not widened", type(exc).__name__, "ValueError")

    # The reason scoping should move multi-doc coverage: with four documents the
    # quota gives each of them 6/4 slots and one loses its second page; with the
    # question resolved to the two it named, each gets three.
    four = {
        "star": [_fake_chunk("star", page, score) for page, score in ((31, 0.91), (30, 0.60))],
        "sbih": [_fake_chunk("sbih", page, score) for page, score in ((20, 0.55), (21, 0.30))],
        "life": [_fake_chunk("life", 8, 0.50), _fake_chunk("life", 9, 0.45)],
        "home": [_fake_chunk("home", 3, 0.48), _fake_chunk("home", 4, 0.44)],
    }
    unscoped_pages = {c.doc_id for c in select_per_document(four, per_doc_top_k=2, total_k=6)}
    check(
        "unscoped, the quota is spread across all four documents",
        sorted(unscoped_pages),
        ["home", "life", "sbih", "star"],
    )
    two = {doc_id: four[doc_id] for doc_id in ("star", "sbih")}
    check(
        "scoped, both named documents keep their pages",
        sorted({(c.doc_id, c.page) for c in select_per_document(two, per_doc_top_k=2, total_k=6)}),
        [("sbih", 20), ("sbih", 21), ("star", 30), ("star", 31)],
    )

    return results


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase3_agents.retrieval_node",
        description="Phase 2 retrieval wrapped as a graph node, with per-document comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the merge-logic checks. No index, no models, no network, no cost.",
    )
    parser.add_argument("--question", default=None, help="Retrieve for one question.")
    parser.add_argument(
        "--route",
        default="lookup",
        choices=["lookup", "calculation", "comparison", "out_of_scope"],
        help="Which retrieval strategy to use.",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        choices=["chunk_policy", "parent_docs"],
        help="Override retrieval_agent.pipeline.",
    )
    parser.add_argument("--user-id", default=None, help="Override index.default_user_id.")
    parser.add_argument(
        "--policy-type",
        default=None,
        help="Restrict retrieval to one policy type, e.g. health. Diagnostic: isolates "
        "whether a missing page is absent from a document's own top candidates.",
    )
    parser.add_argument("--insurer", default=None, help="Restrict retrieval to one insurer.")
    parser.add_argument(
        "--per-doc-depth",
        type=int,
        default=None,
        help="Override retrieval_agent.per_doc_depth for one run.",
    )
    parser.add_argument(
        "--scope-documents",
        action="store_true",
        help="Resolve the question to a document subset before retrieving "
        "(Phase 2 technique 5). Off by default so prior numbers reproduce.",
    )
    parser.add_argument("--list-documents", action="store_true", help="List the user's documents.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the retrieval node from the command line.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 on a failed self-test or bad usage.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.self_test:
        results = _self_test()
        for name, passed, detail in results:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
            if not passed:
                print(f"        {detail}")
        failed = sum(1 for _, passed, _ in results if not passed)
        print(f"\n{len(results) - failed}/{len(results)} checks passed")
        return 1 if failed else 0

    if not args.question and not args.list_documents:
        print("Nothing to do. Pass --self-test, --list-documents, or --question.")
        return 1

    config = load_config(args.config)
    resources = build_resources(
        config, pipeline=args.pipeline, scope_documents=args.scope_documents
    )
    user_id = args.user_id or resources.settings["default_user_id"]
    if args.per_doc_depth:
        resources.settings["per_doc_depth"] = args.per_doc_depth

    try:
        if args.list_documents:
            documents = list_documents(resources, user_id)
            print(f"\n{len(documents)} document(s) for user {user_id}:")
            for doc in documents:
                print(f"  {doc.doc_id}  {doc.policy_type:<8} {doc.filename}")

        if args.question:
            state = {
                "question": args.question,
                "user_id": user_id,
                "route": args.route,
                "policy_type": args.policy_type,
                "insurer": args.insurer,
            }
            update = retrieve_node(state, resources)
            chunks = update["retrieved"]

            print(f"\nquestion : {args.question}")
            print(f"route    : {args.route}  ->  {update['trace'][-1]}")
            print(f"pipeline : {resources.settings['pipeline']} / {resources.collection_name}")
            if (scope := update.get("scope")) is not None:
                print(
                    f"scope    : {len(scope['doc_ids'])} doc(s) via {scope['strength']}"
                    f"{' (ambiguous)' if scope['ambiguous'] else ''}"
                    f"  {', '.join(scope['matched_on']) or '-'}"
                )
            print(f"returned : {len(chunks)} chunk(s) from "
                  f"{len({c.doc_id for c in chunks})} document(s)\n")
            for rank, chunk in enumerate(chunks, start=1):
                snippet = " ".join(chunk.text.split())[:110]
                print(
                    f"  {rank}. [{chunk.score:>8.4f}] {chunk.insurer:<12} "
                    f"p.{chunk.page:<4} {snippet}"
                )
            for assumption in update.get("assumptions", []):
                print(f"\n  ! {assumption}")
    finally:
        resources.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
