"""Deterministic retrieval metrics — zero LLM calls, zero cost.

These numbers answer one question: *did the pipeline find the right page?* No
judge model is needed, because the answer is already known — it is in the golden
set's `ground_truth_pages`. That is why `build_eval_set.py` treats that field as
load-bearing.

**Why this file exists separately from RAGAS.** The two evaluation tiers have
different cadences. This one is free and instant, so it runs after every single
change — which is exactly what Phase 2 needs, since Phase 2 is entirely about
retrieval and re-evaluates after each technique. RAGAS costs real money and runs
at phase boundaries. Bundling them would mean paying for the expensive tier
every time you wanted the cheap one.

**The metrics, and why more than one.**

* **hit@k** — was a correct page anywhere in the top k? Blunt, widely quoted,
  and it hides a lot.
* **MRR** — *where* was it? Rank 1 scores 1.0, rank 5 scores 0.2. This matters
  because a Phase 2 reranker's entire job is moving correct pages upward, an
  improvement hit@5 alone would report as nothing at all.
* **Page-level vs exact-chunk hit** — page-level counts any chunk from the right
  page, matching how citations work. Exact-chunk demands the source chunk. The
  gap between them says whether retrieval finds the right *region* but the wrong
  *passage*, which points at chunk size rather than embedding quality.
* **Negatives** have no page to find, so we report mean top-1 similarity
  instead. A high score on an unanswerable question means refusal rests entirely
  on the model's judgment rather than being backed by weak retrieval.

Usage:
    python -m evals.retrieval_metrics --help
    python -m evals.retrieval_metrics --allow-unverified
    python -m evals.retrieval_metrics
    python -m evals.retrieval_metrics --baseline evals/results/retrieval_xxx.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from phase1_rag.embed_index import collection_name_for
from phase1_rag.rag_chain import build_search_filter, retrieve
from phase2_advanced.hybrid import BM25Index, build_bm25_index, hybrid_retrieve
from phase2_advanced.parent_docs import ParentStore, load_parent_store
from phase2_advanced.rerank import CrossEncoderReranker, build_reranker

LOGGER = logging.getLogger("claimwise.retrieval_metrics")


@dataclass
class ItemResult:
    """Retrieval outcome for one golden item.

    Attributes:
        item_id: Golden set id, e.g. "g-007".
        question_type: "lookup", "calculation" or "negative".
        policy_type: "health", "home", "life", or "" for negatives.
        ground_truth_pages: Pages that contain the answer.
        ground_truth_refs: Ground truth as "doc_id:page" — the correct unit of
            comparison, since page numbers repeat across documents.
        retrieved_pages: Pages returned, in rank order, duplicates kept.
        retrieved_refs: Retrieved results as "doc_id:page", in rank order.
        retrieved_chunk_ids: Chunk ids returned, in rank order.
        source_chunk_id: The chunk the question was written from.
        top_score: Similarity of the rank-1 result.
        page_rank: 1-based rank of the first correct document-and-page. The
            metric that counts.
        loose_page_rank: 1-based rank of the first matching page number
            *ignoring document*. Kept only to quantify how much the naive
            page-only comparison over-reports.
        chunk_rank: 1-based rank of the exact source chunk, or None.
        latency_ms: Embed-and-search time, plus reranking when enabled.
        rerank_ms: Time spent in the cross-encoder, 0 when disabled. Reported
            separately because Phase 2's delta table pairs every quality gain
            with its latency cost.
    """

    item_id: str
    question_type: str
    policy_type: str
    ground_truth_pages: list[int]
    ground_truth_refs: list[str] = field(default_factory=list)
    retrieved_pages: list[int] = field(default_factory=list)
    retrieved_refs: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    source_chunk_id: str = ""
    top_score: float = 0.0
    page_rank: int | None = None
    loose_page_rank: int | None = None
    chunk_rank: int | None = None
    latency_ms: float = 0.0
    rerank_ms: float = 0.0
    # Candidates surviving parent expansion. Expansion deduplicates, so N
    # children can collapse to far fewer parents — shrinking the pool the
    # reranker sees and confounding a comparison against a depth-20 dense run.
    # Recorded so that shrinkage is visible rather than inferred.
    candidates_after_expansion: int = 0


def load_golden(path: Path, verified_only: bool) -> list[dict[str, Any]]:
    """Read the golden set from JSONL.

    Args:
        path: Path to `golden.jsonl`.
        verified_only: Keep only items a human has marked `verified: true`.

    Returns:
        The selected golden items.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Golden set not found: {path.resolve()}. "
            "Run `python -m phase1_rag.build_eval_set` first."
        )
    with path.open("r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]

    if not verified_only:
        return items

    verified = [item for item in items if item.get("verified")]
    skipped = len(items) - len(verified)
    if skipped:
        LOGGER.warning(
            "Skipping %d unverified item(s). Review data/eval/review.md and set "
            '"verified": true, or pass --allow-unverified for provisional numbers.',
            skipped,
        )
    return verified


def first_rank(values: list[Any], targets: set[Any]) -> int | None:
    """Find the 1-based position of the first value present in `targets`.

    Args:
        values: Ranked results.
        targets: Acceptable values.

    Returns:
        The 1-based rank, or None if no value matched.
    """
    for index, value in enumerate(values, start=1):
        if value in targets:
            return index
    return None


def evaluate_item(
    item: dict[str, Any],
    client: QdrantClient,
    collection_name: str,
    embedder: SentenceTransformer,
    settings: dict[str, Any],
    reranker: CrossEncoderReranker | None = None,
    bm25: BM25Index | None = None,
    parents: ParentStore | None = None,
) -> ItemResult:
    """Retrieve for one golden item and record where the correct page landed.

    Deliberately does not generate an answer. This tier measures retrieval only,
    which is what makes it free.

    When a reranker is supplied, the *entire* candidate list is reordered rather
    than truncated, so hit@k stays computable at every k and the before/after
    comparison is exact — same questions, same candidates, different ordering.

    Args:
        item: A golden set item.
        client: An open Qdrant client.
        collection_name: Collection to search.
        embedder: Query embedding model, matching the index.
        settings: Resolved evaluation settings.
        reranker: Optional cross-encoder applied after retrieval.
        bm25: Optional lexical index. When present, retrieval is dense + BM25
            fused by RRF instead of dense alone.

    Returns:
        The item's retrieval outcome.
    """
    started = time.perf_counter()
    if bm25 is not None:
        chunks = hybrid_retrieve(
            client,
            collection_name=collection_name,
            embedder=embedder,
            bm25=bm25,
            question=item["question"],
            settings=settings,
        )
    else:
        chunks = retrieve(
            client,
            collection_name=collection_name,
            embedder=embedder,
            question=item["question"],
            top_k=settings["retrieval_top_k"],
            query_prefix=settings["query_prefix"],
            normalize=settings["normalize"],
            search_filter=build_search_filter(settings["user_id"]),
        )

    # Rerank the CHILDREN, then expand the winners. The first version did the
    # reverse, on the reasoning that the cross-encoder should read what the
    # generator reads. Measurement killed that argument twice over:
    #
    #   1. bge-reranker-base has a hard 512-token window (model_max_length=512).
    #      Median parent is 1,780 chars ~ 445 tokens; with the query prepended
    #      the upper half of parents were truncated, so the cross-encoder never
    #      saw their endings — precisely where a qualifying clause tends to sit.
    #   2. Expansion deduplicates, so 20 children collapsed to a median of 15
    #      parents. The reranker got a smaller pool than the run it was being
    #      compared against, confounding the comparison on top of degrading it.
    #
    # Reranking children first fixes both: 400-char children are ~100 tokens and
    # fit comfortably, the pool stays the size it was asked to be, and the
    # generator still receives whole parents. Ranking precision and context
    # completeness turn out to want different stages, not the same one.
    rerank_ms = 0.0
    if reranker is not None:
        rerank_started = time.perf_counter()
        chunks = reranker.rerank(item["question"], chunks)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000.0

    # Expand the whole reordered list rather than truncating first, so hit@k
    # stays computable at every k.
    if parents is not None:
        chunks = parents.expand(chunks)
    candidates_after_expansion = len(chunks)
    latency_ms = (time.perf_counter() - started) * 1000.0

    pages = [chunk.page for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    source_chunk_id = item.get("source_chunk_id", "")

    ground_truth_pages = set(item.get("ground_truth_pages") or [])
    # Page numbers repeat across documents — our two health policies span 47 and
    # 30 pages, so their ranges almost entirely overlap. Comparing page numbers
    # alone scores a chunk from the wrong insurer as a hit.
    ground_truth_doc = item.get("ground_truth_doc_id", "")
    ground_truth_refs = (
        {f"{ground_truth_doc}:{page}" for page in ground_truth_pages}
        if ground_truth_doc
        else set()
    )
    retrieved_refs = [f"{chunk.doc_id}:{chunk.page}" for chunk in chunks]

    return ItemResult(
        item_id=item["id"],
        question_type=item.get("question_type", "lookup"),
        policy_type=item.get("policy_type", ""),
        ground_truth_pages=sorted(ground_truth_pages),
        ground_truth_refs=sorted(ground_truth_refs),
        retrieved_pages=pages,
        retrieved_refs=retrieved_refs,
        retrieved_chunk_ids=chunk_ids,
        source_chunk_id=source_chunk_id,
        top_score=chunks[0].score if chunks else 0.0,
        page_rank=first_rank(retrieved_refs, ground_truth_refs) if ground_truth_refs else None,
        loose_page_rank=first_rank(pages, ground_truth_pages) if ground_truth_pages else None,
        chunk_rank=first_rank(chunk_ids, {source_chunk_id}) if source_chunk_id else None,
        latency_ms=round(latency_ms, 2),
        rerank_ms=round(rerank_ms, 2),
        candidates_after_expansion=candidates_after_expansion,
    )


def hit_rate(results: list[ItemResult], k: int, mode: str = "page") -> float:
    """Fraction of items whose correct result appeared within the top k.

    Args:
        results: Positive-item results.
        k: Cutoff.
        mode: "page" for document-scoped page hits (the metric that counts),
            "chunk" for exact source-chunk hits, or "loose" for page-number
            matches ignoring which document they came from.

    Returns:
        A value between 0.0 and 1.0, or 0.0 for an empty input.
    """
    if not results:
        return 0.0
    attribute = {"page": "page_rank", "chunk": "chunk_rank", "loose": "loose_page_rank"}[mode]
    hits = sum(
        1
        for result in results
        if (rank := getattr(result, attribute)) is not None and rank <= k
    )
    return round(hits / len(results), 4)


def mean_reciprocal_rank(results: list[ItemResult], k: int) -> float:
    """Mean of 1/rank for the first correct page, 0 when not found within k.

    Args:
        results: Positive-item results.
        k: Cutoff beyond which a result counts as a miss.

    Returns:
        MRR between 0.0 and 1.0.
    """
    if not results:
        return 0.0
    total = sum(
        1.0 / result.page_rank
        for result in results
        if result.page_rank is not None and result.page_rank <= k
    )
    return round(total / len(results), 4)


def page_recall(results: list[ItemResult], k: int) -> float:
    """Mean fraction of ground-truth pages present in the top k.

    Identical to hit@k when every item has one ground-truth page, but diverges
    once multi-hop questions with several source pages are added.

    Args:
        results: Positive-item results.
        k: Cutoff.

    Returns:
        Mean recall between 0.0 and 1.0.
    """
    if not results:
        return 0.0
    recalls = []
    for result in results:
        if not result.ground_truth_refs:
            continue
        found = set(result.retrieved_refs[:k]) & set(result.ground_truth_refs)
        recalls.append(len(found) / len(result.ground_truth_refs))
    return round(statistics.fmean(recalls), 4) if recalls else 0.0


def compute_metrics(results: list[ItemResult], k_values: list[int]) -> dict[str, Any]:
    """Aggregate per-item results into the reported metric set.

    Args:
        results: All item results, positives and negatives.
        k_values: Cutoffs to report.

    Returns:
        A nested summary dict, also written to disk for later comparison.
    """
    positives = [r for r in results if r.question_type != "negative"]
    negatives = [r for r in results if r.question_type == "negative"]
    max_k = max(k_values) if k_values else 10

    by_type: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[ItemResult]] = defaultdict(list)
    for result in positives:
        grouped[result.policy_type or "unknown"].append(result)
    for policy_type, group in sorted(grouped.items()):
        by_type[policy_type] = {
            "items": len(group),
            "hit_at_5_page": hit_rate(group, 5),
            "mrr": mean_reciprocal_rank(group, max_k),
        }

    latencies = [r.latency_ms for r in results] or [0.0]
    latencies_sorted = sorted(latencies)
    p95_index = min(len(latencies_sorted) - 1, int(round(0.95 * (len(latencies_sorted) - 1))))

    missed = [r.item_id for r in positives if r.page_rank is None]

    return {
        "positives": len(positives),
        "negatives": len(negatives),
        "hit_at_k_page": {str(k): hit_rate(positives, k, "page") for k in k_values},
        "hit_at_k_chunk": {str(k): hit_rate(positives, k, "chunk") for k in k_values},
        "hit_at_k_loose": {str(k): hit_rate(positives, k, "loose") for k in k_values},
        "mrr": mean_reciprocal_rank(positives, max_k),
        "page_recall_at_5": page_recall(positives, 5),
        "latency_ms": {
            "median": round(statistics.median(latencies), 2),
            "p95": round(latencies_sorted[p95_index], 2),
        },
        "by_policy_type": by_type,
        "negatives_mean_top_score": (
            round(statistics.fmean([r.top_score for r in negatives]), 4) if negatives else None
        ),
        "missed_item_ids": missed,
    }


def print_report(summary: dict[str, Any], k_values: list[int], baseline: dict[str, Any] | None) -> None:
    """Print the metric report, with deltas when a baseline is supplied.

    Args:
        summary: Metrics from this run.
        k_values: Cutoffs reported.
        baseline: A previous run's summary, or None.
    """
    print("\n=== RETRIEVAL METRICS (deterministic, 0 LLM calls) ===")
    print(f"positives evaluated : {summary['positives']}")
    print(f"negatives evaluated : {summary['negatives']}")

    header = "               " + "".join(f"{'@' + str(k):>9}" for k in k_values)
    print(f"\n{header}")
    for label, key in (
        ("doc+page     ", "hit_at_k_page"),
        ("exact chunk  ", "hit_at_k_chunk"),
        ("page only*   ", "hit_at_k_loose"),
    ):
        row = "".join(f"{summary[key][str(k)]:>9.3f}" for k in k_values)
        print(f"{label}{row}")
    inflation = summary["hit_at_k_loose"]["5"] - summary["hit_at_k_page"]["5"]
    print(
        f"  * page-number match ignoring document — over-reports by "
        f"{inflation:+.3f} at @5. Not a real metric; shown to keep it honest."
    )

    print(f"\nMRR                 : {summary['mrr']:.4f}")
    print(f"page recall@5       : {summary['page_recall_at_5']:.4f}")
    print(f"latency median/p95  : {summary['latency_ms']['median']} / {summary['latency_ms']['p95']} ms")

    if summary["by_policy_type"]:
        print("\nby policy type (page hit@5 / MRR):")
        for policy_type, stats in summary["by_policy_type"].items():
            print(
                f"  {policy_type:<8}: {stats['hit_at_5_page']:.3f} / "
                f"{stats['mrr']:.3f}  ({stats['items']} items)"
            )

    if summary["negatives_mean_top_score"] is not None:
        print(
            f"\nnegatives mean top-1 score : {summary['negatives_mean_top_score']:.4f} "
            "(lower is better)"
        )

    if summary["missed_item_ids"]:
        shown = ", ".join(summary["missed_item_ids"][:15])
        more = "" if len(summary["missed_item_ids"]) <= 15 else " ..."
        print(f"\ncomplete misses ({len(summary['missed_item_ids'])}): {shown}{more}")

    if baseline:
        print("\n=== DELTA vs BASELINE ===")
        for k in k_values:
            before = baseline.get("hit_at_k_page", {}).get(str(k))
            after = summary["hit_at_k_page"][str(k)]
            if before is not None:
                print(f"  page hit@{k:<3}: {before:.3f} -> {after:.3f}  ({after - before:+.3f})")
        before_mrr = baseline.get("mrr")
        if before_mrr is not None:
            print(
                f"  MRR       : {before_mrr:.3f} -> {summary['mrr']:.3f}  "
                f"({summary['mrr'] - before_mrr:+.3f})"
            )


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evals.retrieval_metrics",
        description="Deterministic retrieval metrics over the golden set. No LLM calls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--golden", type=Path, default=None, help="Golden set JSONL.")
    parser.add_argument("--embed-model", default=None, help="Override embed.model_name.")
    parser.add_argument("--top-k", type=int, default=None, help="Override eval.retrieval_top_k.")
    parser.add_argument(
        "--k-values",
        default=None,
        help="Comma-separated cutoffs to report, e.g. 1,5,10,20,50. Overrides config.",
    )
    parser.add_argument("--user-id", default=None, help="Override index.default_user_id.")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Include items not yet human-verified. Numbers are provisional.",
    )
    parser.add_argument(
        "--baseline", type=Path, default=None, help="Previous results JSON, to print deltas."
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Apply cross-encoder reranking (Phase 2)."
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=None,
        help="Candidates to retrieve before reranking. Overrides --top-k when --rerank is set.",
    )
    parser.add_argument("--rerank-model", default=None, help="Override rerank.model_name.")
    parser.add_argument(
        "--hybrid", action="store_true", help="Fuse BM25 with dense retrieval via RRF (Phase 2)."
    )
    parser.add_argument(
        "--parent-docs",
        action="store_true",
        help="Retrieve small children and expand to parent blocks (Phase 2 technique 3).",
    )
    parser.add_argument(
        "--chunk-policy",
        action="store_true",
        help=(
            "Density-selected per-document chunking (Phase 2 technique 5). Implies parent "
            "expansion: dense documents produce children, sparse ones flat chunks that pass "
            "through untouched."
        ),
    )
    parser.add_argument(
        "--collection-prefix",
        default=None,
        help="Override index.collection_prefix. Implied by --parent-docs.",
    )
    # Sweepable from the CLI for the same reason rerank depth is: fusion is a
    # displacement trade — every lexical candidate admitted to a fixed-size pool
    # evicts a dense one — so the contribution widths are the parameter that
    # decides whether hybrid helps, and sweeping them costs nothing.
    parser.add_argument(
        "--dense-top-k", type=int, default=None, help="Override hybrid.dense_top_k."
    )
    parser.add_argument(
        "--lexical-top-k", type=int, default=None, help="Override hybrid.lexical_top_k."
    )
    parser.add_argument("--rrf-k", type=int, default=None, help="Override hybrid.rrf_k.")
    parser.add_argument("--tag", default="", help="Label recorded in the results file.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def resolve_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge config values with CLI overrides.

    Args:
        config: Parsed `config.yaml`.
        args: Parsed command-line arguments.

    Returns:
        Resolved settings for this run.
    """
    # With reranking on, the candidate pool is what gets retrieved and the
    # cross-encoder reorders all of it — so rerank depth *is* retrieval depth.
    rerank_depth = args.rerank_depth or cfg_get(config, "rerank.candidate_depth", 30)
    return {
        "rerank": bool(args.rerank),
        "rerank_depth": rerank_depth if args.rerank else None,
        "hybrid": bool(args.hybrid),
        # Both flags need parent expansion; they differ only in which corpus and
        # collection they point at. --chunk-policy wins if both are passed.
        "parent_docs": bool(args.parent_docs or args.chunk_policy),
        "chunk_policy": bool(args.chunk_policy),
        # Each implies its own collection: the parent tier only exists in the
        # matching one, and pointing at the Phase 1 collection would retrieve
        # chunks with no parent_id and silently degrade to plain dense search.
        "collection_prefix_override": (
            args.collection_prefix
            or (
                cfg_get(config, "chunk_policy.collection_prefix", "claimwise_mx")
                if args.chunk_policy
                else cfg_get(config, "parent_docs.collection_prefix", "claimwise_pd")
                if args.parent_docs
                else None
            )
        ),
        "parents_path": (
            Path(cfg_get(config, "chunk_policy.parents_path", "data/processed/mixed_parents.jsonl"))
            if args.chunk_policy
            else None
        ),
        # Hybrid fuses down to candidate_depth, which the reranker then narrows.
        "candidate_depth": rerank_depth,
        "dense_top_k": args.dense_top_k or cfg_get(config, "hybrid.dense_top_k", 30),
        "lexical_top_k": args.lexical_top_k or cfg_get(config, "hybrid.lexical_top_k", 30),
        "rrf_k": args.rrf_k or cfg_get(config, "hybrid.rrf_k", 60),
        "retrieval_top_k": (
            rerank_depth if args.rerank else (args.top_k or cfg_get(config, "eval.retrieval_top_k", 10))
        ),
        "k_values": (
            [int(value) for value in args.k_values.split(",")]
            if args.k_values
            else cfg_get(config, "eval.k_values", [1, 3, 5, 10])
        ),
        "embed_model": args.embed_model or cfg_get(config, "embed.model_name"),
        "query_prefix": cfg_get(config, "embed.query_prefix", ""),
        "normalize": cfg_get(config, "embed.normalize", True),
        "device": cfg_get(config, "embed.device", "cpu"),
        "qdrant_path": cfg_get(config, "index.path", "qdrant_storage"),
        "collection_prefix": cfg_get(config, "index.collection_prefix", "claimwise"),
        "user_id": args.user_id or cfg_get(config, "index.default_user_id", "local-dev"),
    }


def main(argv: list[str] | None = None) -> int:
    """Evaluate retrieval over the golden set.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if there was nothing to evaluate.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    settings = resolve_settings(config, args)

    eval_dir = Path(cfg_get(config, "paths.eval_dir", "data/eval"))
    golden_path = args.golden or eval_dir / cfg_get(config, "eval.output_filename", "golden.jsonl")
    results_dir = Path(cfg_get(config, "eval.results_dir", "evals/results"))

    items = load_golden(golden_path, verified_only=not args.allow_unverified)
    if not items:
        LOGGER.error(
            "No items to evaluate. Verify items in %s, or pass --allow-unverified.",
            golden_path,
        )
        return 1
    if args.allow_unverified:
        LOGGER.warning(
            "PROVISIONAL RUN — including unverified items. These numbers must not "
            "be recorded in METRICS.md as the Phase 1 baseline."
        )

    # Negatives have no ground-truth page, so every retrieval metric is
    # undefined without positives. Reporting 0.000 across the board would look
    # like catastrophic retrieval failure rather than an empty measurement —
    # the same "summary that looks like a result" trap as the eval-set bug.
    positive_count = sum(1 for item in items if item.get("question_type") != "negative")
    if positive_count == 0:
        LOGGER.error(
            "No positive items to evaluate — %d negatives carry no ground-truth "
            "page, so hit@k and MRR are undefined, not zero. Either verify items "
            "in %s, or re-run with --allow-unverified for provisional numbers.",
            len(items),
            golden_path,
        )
        return 1

    embedder = SentenceTransformer(settings["embed_model"], device=settings["device"])
    collection_name = collection_name_for(
        settings["collection_prefix_override"] or settings["collection_prefix"],
        settings["embed_model"],
    )
    bm25 = build_bm25_index(config) if settings["hybrid"] else None
    reranker = build_reranker(config, model_name=args.rerank_model) if settings["rerank"] else None
    parents = (
        load_parent_store(config, parents_path=settings["parents_path"])
        if settings["parent_docs"]
        else None
    )
    if parents:
        LOGGER.info(
            "Parent-document retrieval: %d parents, collection %s", len(parents), collection_name
        )
        LOGGER.warning(
            "exact-chunk hit@k is NOT comparable to a Phase 1 run — the golden set's "
            "source_chunk_id refers to 1,000-char chunks that no longer exist. Compare "
            "doc+page only."
        )
    if bm25:
        LOGGER.info(
            "Hybrid retrieval: dense top-%d + BM25 top-%d, RRF k=%d, fused to %d",
            settings["dense_top_k"],
            settings["lexical_top_k"],
            settings["rrf_k"],
            settings["candidate_depth"],
        )
    if reranker:
        LOGGER.info(
            "Reranking enabled: %s over %d candidates",
            reranker.model_name,
            settings["rerank_depth"],
        )

    client = QdrantClient(path=settings["qdrant_path"])
    try:
        if not client.collection_exists(collection_name):
            LOGGER.error("Collection %s does not exist. Run embed_index first.", collection_name)
            return 1
        results = [
            evaluate_item(
                item, client, collection_name, embedder, settings, reranker, bm25, parents
            )
            for item in tqdm(items, desc="Retrieving", unit="q")
        ]
    finally:
        client.close()

    summary = compute_metrics(results, settings["k_values"])
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tag": args.tag,
            "provisional": bool(args.allow_unverified),
            "parent_docs": settings["parent_docs"],
            "chunk_policy": settings["chunk_policy"],
            "candidates_after_expansion_median": (
                round(statistics.median([r.candidates_after_expansion for r in results]), 1)
                if settings["parent_docs"]
                else None
            ),
            # Recorded so a results file can never be silently compared against
            # one built on a different chunking strategy.
            "exact_chunk_comparable": not settings["parent_docs"],
            "rerank": settings["rerank"],
            "rerank_depth": settings["rerank_depth"],
            "hybrid": settings["hybrid"],
            "hybrid_config": (
                {
                    "dense_top_k": settings["dense_top_k"],
                    "lexical_top_k": settings["lexical_top_k"],
                    "rrf_k": settings["rrf_k"],
                }
                if settings["hybrid"]
                else None
            ),
            "rerank_model": (reranker.model_name if reranker else None),
            "rerank_ms_median": (
                round(statistics.median([r.rerank_ms for r in results]), 2) if reranker else 0.0
            ),
            "embed_model": settings["embed_model"],
            "collection": collection_name,
            "retrieval_top_k": settings["retrieval_top_k"],
            "golden_set": golden_path.as_posix(),
        }
    )

    baseline = None
    if args.baseline:
        if args.baseline.is_file():
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        else:
            LOGGER.warning("Baseline file not found: %s", args.baseline)

    print_report(summary, settings["k_values"], baseline)

    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{args.tag}" if args.tag else ""
    # Provisional runs are marked in the filename, not just inside the JSON, so
    # a file tagged "baseline" can never quietly contain unverified numbers.
    prefix = "provisional_" if args.allow_unverified else ""
    results_path = results_dir / f"{prefix}retrieval_{stamp}{suffix}.json"
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    per_item_path = results_dir / f"{prefix}retrieval_{stamp}{suffix}_items.jsonl"
    with per_item_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")

    print(f"\nresults : {results_path}")
    print(f"per-item: {per_item_path}")
    print("\nPass --baseline <that results file> on the next run to see deltas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
