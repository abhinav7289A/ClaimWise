"""Decide which of a user's documents a question is about, before retrieving.

**The problem this solves, measured rather than assumed.** Phase 4's calculation
slice generated questions that named their own policy in plain text — *"Under my
HDFC ERGO Optima Secure health policy, my hospital bill is Rs.1,50,000..."* — and
the retriever surfaced that policy's citing page **19 times out of 180 (10.6%)**.
The document name was already in the query. Rewriting the query cannot help when
the query already contains the answer to "which document is this"; the retriever
simply is not using it. Multi-document `docs-covered@5` of 0.250 (Phase 3) is the
same failure seen from the other side, and it is what left Phase 3's comparison
route below its exit criterion.

**Why embeddings cannot fix this on their own.** Dense vectors encode *topic*,
not *identity*. Four of the ten indexed documents are HDFC ERGO health policies
whose wording is near-identical, so which one ranks first is decided by noise.
No reranker recovers identity the embedding never carried. Scoping by metadata
before the search is the standard answer — this is CLAUDE.md's Phase 2 technique
5, and `retrieval_node.document_filter` has been sitting there waiting for a
caller to tell it *which* document.

**Why deterministic matching and not an LLM classifier.** It is free, it adds no
latency to a 2-5s query budget, it is unit-testable, and it cannot hallucinate a
document the user does not own. D-20 rejected query rewriting partly because a
generation call per query moved the eval loop from seconds to over an hour; that
objection does not apply here, which is the main reason this is worth trying
first.

**The safety rule, and it is the whole design.** When the question is ambiguous,
**widen rather than guess**. Filtering to the wrong document guarantees a failed
answer; filtering to four candidates instead of ten is still most of the win.
Resolving nothing returns an empty set, and the caller falls back to today's
global search — so a question this cannot parse is never worse off than it is now.

**Multiple documents are a feature, not an ambiguity.** "Compare my Star Health
and SBI policies" resolves to two documents deliberately, which is exactly what
`retrieve_per_document` needs and exactly what the comparison route has been
missing.

Usage:
    python -m phase2_advanced.document_resolver --help
    python -m phase2_advanced.document_resolver --self-test
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable

# Insurer slugs are written without separators ("hdfcergo"), so they cannot be
# split algorithmically into the words a customer types. This table maps each to
# the forms people actually write. A slug not listed here still matches on itself,
# so an unknown insurer degrades to "works if the user types the slug" rather
# than breaking — add a line when a new insurer joins the corpus.
INSURER_ALIASES: dict[str, tuple[str, ...]] = {
    "bajajallianz": ("bajaj allianz", "bajaj"),
    "hdfcergo": ("hdfc ergo", "hdfc"),
    "iciciprulife": ("icici pru life", "icici pru", "icici"),
    "nivabupa": ("niva bupa", "niva", "max bupa"),
    "sbigeneral": ("sbi general", "sbi"),
    "starhealth": ("star health", "star"),
}

# What people call each policy type. Deliberately generous on motor, because
# "bike"/"car" is how the question actually arrives and never appears in the
# document's own metadata.
POLICY_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "health": ("health", "medical", "mediclaim", "hospital", "hospitalisation", "illness"),
    "home": ("home", "house", "household", "property", "contents"),
    "life": ("life", "term plan", "ulip", "endowment", "maturity"),
    "motor": ("motor", "bike", "motorcycle", "two wheeler", "two-wheeler", "scooter",
              "car", "vehicle", "own damage"),
}


@dataclass(frozen=True)
class IndexedDocument:
    """One document available to a user, as read from the index payload.

    Structurally identical to `retrieval_node.DocumentRef`, and deliberately not
    imported from it: that lives in the Phase 3 layer and is built from
    `RetrievalResources`, which the retrieval eval never constructs. The resolver
    reads its inputs by attribute, so either type works wherever documents are
    accepted.

    Attributes:
        doc_id: Content hash of the source PDF — the filter key.
        filename: Source PDF filename.
        insurer: e.g. "starhealth".
        policy_type: e.g. "health".
        doc_label: Free-text plan label, e.g. "optima-secure".
    """

    doc_id: str
    filename: str
    insurer: str
    policy_type: str
    doc_label: str


@dataclass(frozen=True)
class Resolution:
    """Which documents a question is scoped to, and how confidently.

    Attributes:
        doc_ids: Documents to search. Empty means "could not tell — search all".
        matched_on: Human-readable signals that fired, for the trace and logs.
        strength: "plan" | "insurer" | "policy_type" | "sole_document" | "none",
            strongest first. The caller may choose to act only on strong matches.
        ambiguous: True when one signal matched several documents and the user
            probably meant exactly one of them — a prompt-for-clarification cue,
            not an error. False when several documents were named on purpose.
    """

    doc_ids: tuple[str, ...]
    matched_on: tuple[str, ...]
    strength: str
    ambiguous: bool

    def __bool__(self) -> bool:
        """True when the resolver narrowed anything at all."""
        return bool(self.doc_ids)


def normalise(text: str) -> str:
    """Lower-case and flatten punctuation so aliases match how people type.

    Hyphens become spaces so "two-wheeler" and "two wheeler" are one token, and
    "optima-secure" from a filename matches "Optima Secure" in a question.

    Args:
        text: Raw question or label.

    Returns:
        Lower-case text with punctuation reduced to single spaces.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def mentions(haystack: str, needle: str) -> bool:
    """Test whether a phrase appears as whole words.

    Word boundaries matter more than they look: "star" must not match "start",
    and "sbi" must not match inside a longer token.

    Args:
        haystack: Normalised question text.
        needle: Normalised phrase to look for.

    Returns:
        True if the phrase appears on word boundaries.
    """
    if not needle:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def insurer_phrases(slug: str) -> tuple[str, ...]:
    """Every way a user might write one insurer's name.

    Args:
        slug: The insurer slug from document metadata.

    Returns:
        Normalised phrases, including the slug itself as a fallback.
    """
    return tuple(normalise(phrase) for phrase in (*INSURER_ALIASES.get(slug, ()), slug) if phrase)


def resolve_documents(question: str, documents: Iterable[Any]) -> Resolution:
    """Work out which of the user's documents a question refers to.

    Signals are tried strongest first and the first tier that fires wins, because
    a plan name is decisive and a policy type is barely evidence at all. Mixing
    tiers would let a weak signal dilute a strong one.

    Args:
        question: The user's question.
        documents: `DocumentRef`-shaped objects with `doc_id`, `insurer`,
            `policy_type` and `doc_label`.

    Returns:
        A `Resolution`. Empty `doc_ids` means the caller should search everything,
        which is today's behaviour and therefore never a regression.
    """
    docs = list(documents)
    if not docs:
        return Resolution((), (), "none", False)

    # One document means there is nothing to disambiguate, whatever was asked.
    if len(docs) == 1:
        return Resolution((docs[0].doc_id,), ("sole document",), "sole_document", False)

    text = normalise(question)

    plan_hits: list[Any] = []
    insurer_hits: list[Any] = []
    type_hits: list[Any] = []
    signals: set[str] = set()

    for doc in docs:
        label = normalise(getattr(doc, "doc_label", ""))
        # A plan label of one short word ("alpha") is too weak to scope on alone;
        # requiring a longer label avoids scoping the whole query on a coincidence.
        if label and len(label) >= 5 and mentions(text, label):
            plan_hits.append(doc)
            signals.add(f"plan:{label}")

        slug = getattr(doc, "insurer", "")
        matched = next((phrase for phrase in insurer_phrases(slug) if mentions(text, phrase)), None)
        if matched:
            insurer_hits.append(doc)
            signals.add(f"insurer:{slug}")

        policy_type = getattr(doc, "policy_type", "")
        aliases = POLICY_TYPE_ALIASES.get(policy_type, (policy_type,))
        if any(mentions(text, normalise(alias)) for alias in aliases if alias):
            type_hits.append(doc)
            signals.add(f"type:{policy_type}")

    def distinct(prefix: str) -> int:
        """How many different signals of one kind the question carried."""
        return sum(1 for signal in signals if signal.startswith(prefix))

    if plan_hits:
        # Naming a plan is decisive. Several plans named means a comparison.
        return Resolution(
            tuple(doc.doc_id for doc in plan_hits),
            tuple(sorted(signal for signal in signals if signal.startswith("plan:"))),
            "plan",
            ambiguous=len(plan_hits) > 1 and distinct("plan:") == 1,
        )

    if insurer_hits:
        # An insurer plus a policy type narrows further — "my SBI health policy"
        # picks one of two SBI documents. Only narrow if something survives.
        narrowed = [doc for doc in insurer_hits if doc in type_hits] or insurer_hits
        return Resolution(
            tuple(doc.doc_id for doc in narrowed),
            tuple(sorted(signal for signal in signals if not signal.startswith("plan:"))),
            "insurer",
            ambiguous=len(narrowed) > 1 and distinct("insurer:") == 1,
        )

    if type_hits and len(type_hits) < len(docs):
        # Weakest tier, and only worth acting on when it actually excludes
        # something. "Is dental covered?" matching all seven health policies has
        # narrowed nothing and should not be dressed up as a resolution.
        return Resolution(
            tuple(doc.doc_id for doc in type_hits),
            tuple(sorted(signal for signal in signals if signal.startswith("type:"))),
            "policy_type",
            ambiguous=len(type_hits) > 1 and distinct("type:") == 1,
        )

    return Resolution((), (), "none", False)


# --- Index side --------------------------------------------------------------


def documents_from_index(
    client: Any,
    collection_name: str,
    user_id: str,
) -> list[IndexedDocument]:
    """List the distinct documents one user has indexed.

    The raw-client twin of `retrieval_node.list_documents`, which needs a
    `RetrievalResources` the retrieval eval does not build. Scrolls payloads
    rather than searching, because this is a metadata question with no query.

    Args:
        client: An open Qdrant client.
        collection_name: Collection to scroll.
        user_id: Security boundary — never widen this to see other users' docs.

    Returns:
        One entry per distinct `doc_id`, sorted by filename for stable output.
    """
    from phase1_rag.rag_chain import build_search_filter

    seen: dict[str, IndexedDocument] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
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
                seen[doc_id] = IndexedDocument(
                    doc_id=doc_id,
                    filename=payload.get("filename", ""),
                    insurer=payload.get("insurer", ""),
                    policy_type=payload.get("policy_type", ""),
                    doc_label=payload.get("doc_label", ""),
                )
        if offset is None:
            break
    return sorted(seen.values(), key=lambda doc: doc.filename)


def scoped_filter(user_id: str, doc_ids: Iterable[str]) -> Any:
    """Build a search filter restricted to a set of documents, inside one user.

    `retrieval_node.document_filter` scopes to exactly one document; a comparison
    question legitimately needs several, so this takes a set and uses `MatchAny`.
    The `user_id` condition is preserved rather than replaced — narrowing to a
    document must never widen access.

    Args:
        user_id: Security boundary.
        doc_ids: Documents the question resolved to.

    Returns:
        A Qdrant filter requiring the user and any of the given documents.
    """
    from qdrant_client.http import models

    from phase1_rag.rag_chain import build_search_filter

    base = build_search_filter(user_id)
    conditions = list(base.must or [])
    conditions.append(
        models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))
    )
    return models.Filter(must=conditions)


# --- Self-test ---------------------------------------------------------------


def _corpus() -> list[IndexedDocument]:
    """The ten indexed documents, as of D-29."""
    rows = [
        ("2c3eee38a579", "iciciprulife", "life", "prusmart"),
        ("bcab78fd45a4", "bajajallianz", "motor", "commercial-vehicle"),
        ("8bc1e390905f", "hdfcergo", "health", "easy-health"),
        ("4a1c8e53b46c", "hdfcergo", "health", "koti-suraksha"),
        ("0058f205f552", "hdfcergo", "health", "optima-restore"),
        ("195eb0499db9", "hdfcergo", "health", "optima-secure"),
        ("bcbf6c18c13b", "nivabupa", "health", "health-assurance"),
        ("5b8f496626be", "sbigeneral", "health", "alpha"),
        ("478aa61fb6e3", "sbigeneral", "home", "house-insurance"),
        ("b1dbe8fb7864", "starhealth", "health", "comprehensive"),
    ]
    return [
        IndexedDocument(doc_id, f"{insurer}__{ptype}__{label}.pdf", insurer, ptype, label)
        for doc_id, insurer, ptype, label in rows
    ]


def _self_test() -> list[tuple[str, bool, str]]:
    """Exercise every tier and both failure directions. No index, no cost.

    Returns:
        One (name, passed, detail) per check.
    """
    results: list[tuple[str, bool, str]] = []
    docs = _corpus()

    def check(name: str, actual: Any, expected: Any) -> None:
        """Record one comparison."""
        results.append((name, actual == expected, f"expected {expected!r}, got {actual!r}"))

    # Tier 1: a named plan is decisive, even among four same-insurer policies.
    r = resolve_documents(
        "Under my HDFC ERGO Optima Secure health policy, is dental treatment covered?", docs
    )
    check("plan name resolves to one doc", r.doc_ids, ("195eb0499db9",))
    check("plan match reports strength", r.strength, "plan")
    check("single plan is not ambiguous", r.ambiguous, False)

    # Tier 2: insurer alone, where the insurer owns exactly one document.
    r = resolve_documents("Does my Star Health policy cover cataract surgery?", docs)
    check("insurer with one doc resolves", r.doc_ids, ("b1dbe8fb7864",))
    check("insurer match reports strength", r.strength, "insurer")

    # Tier 2, ambiguous: HDFC owns four. Widen rather than guess.
    r = resolve_documents("What is the waiting period on my HDFC policy?", docs)
    check("insurer with four docs returns all four", len(r.doc_ids), 4)
    check("and flags itself ambiguous", r.ambiguous, True)

    # Tier 2 narrowed by policy type: SBI owns one health and one home policy.
    r = resolve_documents("What does my SBI health policy pay for a hospital stay?", docs)
    check("insurer plus type narrows to one", r.doc_ids, ("5b8f496626be",))

    # Comparison: two insurers named on purpose is not ambiguity.
    r = resolve_documents("Compare the room rent limits on my Star Health and Niva Bupa plans", docs)
    check("two insurers resolve to two docs", len(r.doc_ids), 2)
    check("deliberate multi-doc is not ambiguous", r.ambiguous, False)

    # Tier 3: the vocabulary a bike question actually arrives in.
    r = resolve_documents("My bike got stolen last week — am I covered?", docs)
    check("bike resolves to the motor policy", r.doc_ids, ("bcab78fd45a4",))
    check("policy type reports strength", r.strength, "policy_type")

    # The safety rule: no signal means search everything, exactly as today.
    r = resolve_documents("Is dental treatment covered?", docs)
    check("unscoped question resolves nothing", r.doc_ids, ())
    check("and is falsy for the caller", bool(r), False)

    # A type that matches every document has narrowed nothing and must not
    # pretend otherwise — seven of ten are health policies.
    r = resolve_documents("What are my hospital benefits?", docs)
    check("type matching most docs still narrows to those", len(r.doc_ids), 7)

    # Word boundaries: "star" must not fire inside another word.
    r = resolve_documents("When does my cover start after I buy it?", docs)
    check("'start' does not match insurer 'star'", r.doc_ids, ())

    # One document means nothing to disambiguate.
    r = resolve_documents("anything at all", docs[:1])
    check("sole document always resolves", r.strength, "sole_document")

    # No documents must not raise.
    r = resolve_documents("anything", [])
    check("empty corpus resolves nothing", r.doc_ids, ())

    return results


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase2_advanced.document_resolver",
        description="Resolve which documents a question is about. No models, no cost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true", help="Run the checks and exit.")
    parser.add_argument("--question", default="", help="Resolve one question against the corpus.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the self-test or resolve a single question.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    if args.question:
        docs = _corpus()
        resolution = resolve_documents(args.question, docs)
        by_id = {doc.doc_id: doc for doc in docs}
        print(f"question : {args.question}")
        print(f"strength : {resolution.strength}   ambiguous: {resolution.ambiguous}")
        print(f"matched  : {', '.join(resolution.matched_on) or '(nothing)'}")
        print(f"documents: {len(resolution.doc_ids)} of {len(docs)}")
        for doc_id in resolution.doc_ids:
            print(f"  - {by_id[doc_id].filename}")
        return 0

    if args.self_test:
        results = _self_test()
        for name, passed, detail in results:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}" + ("" if passed else f"  — {detail}"))
        failures = sum(1 for _, passed, _ in results if not passed)
        print(f"\n{len(results) - failures}/{len(results)} checks passed")
        return 1 if failures else 0

    build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
