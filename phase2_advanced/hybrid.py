"""Hybrid retrieval: BM25 + dense, fused by Reciprocal Rank Fusion.

**Why reranking alone was not enough.** A cross-encoder reorders the candidate
pool; it cannot add to it. At depth 20 that pool has recall 0.894, so the nine
remaining misses are structurally invisible to reranking. Raising pool recall is
the only route to them — and it compounds, because better candidates in means
better reranked output out.

**Why lexical retrieval specifically.** Dense and lexical search fail in
opposite directions. Embeddings capture meaning but blur exact tokens: P-11 was
exactly this, where "co-payment" appeared verbatim in the target chunk and dense
search still missed it because a stray age in the query moved the vector. BM25
is the mirror image — it nails rare exact terms (`AYUSH`, `UIN`, `Section
II.23`) and understands nothing at all. Insurance questions mix everyday
phrasing with policy-specific terminology in one sentence, so both matter.

**Why fuse by rank rather than score.** Cosine similarity lives in roughly
0.4-0.9; BM25 is unbounded and depends on corpus statistics. There is no
principled way to put them on one scale. Reciprocal Rank Fusion sidesteps the
problem entirely by discarding magnitudes and combining positions:

    score(chunk) = sum over retrievers of 1 / (k + rank)

A chunk ranked 1st by either retriever scores highly; a chunk ranked
respectably by *both* can outrank one that only a single retriever loved. The
constant k (default 60) flattens the curve so the top few ranks don't dominate
completely.

**Security note for Phase 5.** Dense retrieval filters `user_id` inside Qdrant's
index traversal. This BM25 index is built from the local chunks file and has no
such boundary — acceptable while there is one local user, but a production
deployment must build the lexical index per user or filter by owned `doc_id`
before scoring. `allowed_doc_ids` exists for that.

Usage:
    python -m phase2_advanced.hybrid --help
    python -m phase2_advanced.hybrid -q "What is the co-payment at age 65?"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from phase1_rag.embed_index import collection_name_for
from phase1_rag.rag_chain import RetrievedChunk, build_search_filter, retrieve

LOGGER = logging.getLogger("claimwise.hybrid")

# Lowercase alphanumeric runs. "Co-Payment" becomes ["co", "payment"] and the
# query is tokenised identically, so the split is consistent on both sides.
# Digits are kept: "36 months" and "₹5 L" carry real signal in policy text.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into BM25 terms.

    Args:
        text: Any text.

    Returns:
        Lowercased alphanumeric tokens, in order, duplicates kept — BM25 uses
        term frequency, so repetition is signal rather than noise.
    """
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """An in-memory BM25 index over chunk records."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        """Build the index.

        Args:
            chunks: Decoded records from `chunks.jsonl`.

        Raises:
            ValueError: If no chunks were supplied.
        """
        if not chunks:
            raise ValueError("Cannot build a BM25 index from zero chunks.")
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
        LOGGER.info("BM25 index built over %d chunks", len(chunks))

    def search(
        self,
        query: str,
        top_k: int,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Return the highest-scoring chunks for a query.

        Args:
            query: The user's question.
            top_k: How many chunks to return.
            allowed_doc_ids: Restrict to these documents. None allows all.
                This is the hook a per-user deployment must use.

        Returns:
            (chunk record, BM25 score) pairs, best first, zero-score chunks
            dropped — a chunk sharing no terms with the query is not a result.
        """
        scores = self._bm25.get_scores(tokenize(query))

        ranked = sorted(
            (
                (chunk, float(score))
                for chunk, score in zip(self.chunks, scores)
                if score > 0.0
                and (allowed_doc_ids is None or chunk["doc_id"] in allowed_doc_ids)
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]


def load_chunk_records(path: Path) -> list[dict[str, Any]]:
    """Read chunk records from JSONL.

    Args:
        path: Path to `chunks.jsonl`.

    Returns:
        Decoded chunk records.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Chunks file not found: {path.resolve()}. Run `python -m phase1_rag.chunk` first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_bm25_index(config: dict[str, Any], chunks_path: Path | None = None) -> BM25Index:
    """Construct the configured BM25 index.

    Args:
        config: Parsed `config.yaml`.
        chunks_path: Override the configured chunks path.

    Returns:
        A ready-to-search index.
    """
    path = chunks_path or Path(
        cfg_get(config, "hybrid.chunks_path", "data/processed/chunks.jsonl")
    )
    return BM25Index(load_chunk_records(path))


def chunk_record_to_retrieved(record: dict[str, Any], score: float) -> RetrievedChunk:
    """Convert a raw chunk record into the pipeline's retrieval type.

    Args:
        record: A decoded chunk record.
        score: Score to attach.

    Returns:
        A `RetrievedChunk` matching what dense retrieval produces.
    """
    return RetrievedChunk(
        chunk_id=record["chunk_id"],
        doc_id=record["doc_id"],
        filename=record["filename"],
        insurer=record["insurer"],
        policy_type=record["policy_type"],
        doc_label=record["doc_label"],
        page=record["page"],
        text=record["text"],
        score=round(score, 4),
    )


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], rrf_k: int, top_k: int
) -> list[RetrievedChunk]:
    """Fuse several ranked lists into one by position.

    Args:
        rankings: Ranked lists from each retriever, best first.
        rrf_k: The RRF constant. Larger values flatten the weighting so early
            ranks dominate less.
        top_k: How many fused results to return.

    Returns:
        Chunks sorted by fused score, best first. `score` now holds the RRF
        value — comparable within this stage only, never against cosine or BM25.
    """
    fused_scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, RetrievedChunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            fused_scores[chunk.chunk_id] += 1.0 / (rrf_k + rank)
            # Keep the first instance seen; the text and metadata are identical
            # whichever retriever produced it.
            by_id.setdefault(chunk.chunk_id, chunk)

    ordered = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [
        RetrievedChunk(
            chunk_id=by_id[chunk_id].chunk_id,
            doc_id=by_id[chunk_id].doc_id,
            filename=by_id[chunk_id].filename,
            insurer=by_id[chunk_id].insurer,
            policy_type=by_id[chunk_id].policy_type,
            doc_label=by_id[chunk_id].doc_label,
            page=by_id[chunk_id].page,
            text=by_id[chunk_id].text,
            score=round(score, 6),
        )
        for chunk_id, score in ordered[:top_k]
    ]


def hybrid_retrieve(
    client: QdrantClient,
    collection_name: str,
    embedder: SentenceTransformer,
    bm25: BM25Index,
    question: str,
    settings: dict[str, Any],
    insurer: str | None = None,
    policy_type: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve densely and lexically, then fuse.

    Args:
        client: An open Qdrant client.
        collection_name: Collection to search.
        embedder: Query embedding model, matching the index.
        bm25: The lexical index.
        question: The user's question.
        settings: Resolved settings; needs query_prefix, normalize, user_id,
            dense_top_k, lexical_top_k, rrf_k and candidate_depth.
        insurer: Optional metadata filter, applied to both retrievers.
        policy_type: Optional metadata filter, applied to both retrievers.

    Returns:
        Fused candidates, best first, ready for the reranker.
    """
    dense = retrieve(
        client,
        collection_name=collection_name,
        embedder=embedder,
        question=question,
        top_k=settings["dense_top_k"],
        query_prefix=settings["query_prefix"],
        normalize=settings["normalize"],
        search_filter=build_search_filter(settings["user_id"], insurer, policy_type),
    )

    # Qdrant applies insurer/policy_type inside its traversal; BM25 has no such
    # mechanism, so the same restriction is applied here by hand.
    allowed_doc_ids: set[str] | None = None
    if insurer or policy_type:
        allowed_doc_ids = {
            record["doc_id"]
            for record in bm25.chunks
            if (not insurer or record["insurer"] == insurer)
            and (not policy_type or record["policy_type"] == policy_type)
        }

    lexical = [
        chunk_record_to_retrieved(record, score)
        for record, score in bm25.search(
            question, top_k=settings["lexical_top_k"], allowed_doc_ids=allowed_doc_ids
        )
    ]

    return reciprocal_rank_fusion(
        [dense, lexical], rrf_k=settings["rrf_k"], top_k=settings["candidate_depth"]
    )


def resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Collect the settings hybrid retrieval needs.

    Args:
        config: Parsed `config.yaml`.

    Returns:
        A settings dict.
    """
    return {
        "dense_top_k": cfg_get(config, "hybrid.dense_top_k", 30),
        "lexical_top_k": cfg_get(config, "hybrid.lexical_top_k", 30),
        "rrf_k": cfg_get(config, "hybrid.rrf_k", 60),
        "candidate_depth": cfg_get(config, "rerank.candidate_depth", 20),
        "query_prefix": cfg_get(config, "embed.query_prefix", ""),
        "normalize": cfg_get(config, "embed.normalize", True),
        "user_id": cfg_get(config, "index.default_user_id", "local-dev"),
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the demonstration CLI.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase2_advanced.hybrid",
        description="Show dense, lexical and fused rankings side by side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-q", "--question", required=True)
    parser.add_argument("--show", type=int, default=8, help="Rows to print per list.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print the dense, lexical and fused rankings for one question.

    Seeing which retriever found what is the fastest way to tell whether fusion
    is contributing anything — if both lists are identical, hybrid search is
    pure overhead on this corpus.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if nothing was retrieved.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    settings = resolve_settings(config)
    embed_model = cfg_get(config, "embed.model_name")
    collection_name = collection_name_for(
        cfg_get(config, "index.collection_prefix", "claimwise"), embed_model
    )

    embedder = SentenceTransformer(embed_model, device=cfg_get(config, "embed.device", "cpu"))
    build_started = time.perf_counter()
    bm25 = build_bm25_index(config)
    build_ms = (time.perf_counter() - build_started) * 1000

    client = QdrantClient(path=cfg_get(config, "index.path", "qdrant_storage"))
    try:
        if not client.collection_exists(collection_name):
            LOGGER.error("Collection %s does not exist. Run embed_index first.", collection_name)
            return 1

        dense = retrieve(
            client,
            collection_name=collection_name,
            embedder=embedder,
            question=args.question,
            top_k=settings["dense_top_k"],
            query_prefix=settings["query_prefix"],
            normalize=settings["normalize"],
            search_filter=build_search_filter(settings["user_id"]),
        )
        fused = hybrid_retrieve(
            client, collection_name, embedder, bm25, args.question, settings
        )
    finally:
        client.close()

    lexical = [
        chunk_record_to_retrieved(record, score)
        for record, score in bm25.search(args.question, top_k=settings["lexical_top_k"])
    ]

    if not dense and not lexical:
        LOGGER.error("Neither retriever returned anything.")
        return 1

    print(f"\nQ: {args.question}")
    print(f"BM25 index built in {build_ms:.0f} ms over {len(bm25.chunks)} chunks")

    def show(title: str, chunks: list[RetrievedChunk]) -> None:
        print(f"\n=== {title} ===")
        for rank, chunk in enumerate(chunks[: args.show], start=1):
            snippet = " ".join(chunk.text.split())[:80]
            print(f"{rank:>2}. {chunk.score:>9.4f}  p.{chunk.page:<4} [{chunk.insurer}] {snippet}...")

    show("DENSE (cosine)", dense)
    show("LEXICAL (BM25)", lexical)
    show("FUSED (RRF)", fused)

    dense_ids = {chunk.chunk_id for chunk in dense[:5]}
    lexical_ids = {chunk.chunk_id for chunk in lexical[:5]}
    fused_ids = {chunk.chunk_id for chunk in fused[:5]}
    print(
        f"\ntop-5 overlap dense/lexical : {len(dense_ids & lexical_ids)}/5"
        f"  |  fused top-5 not in dense top-5 : {len(fused_ids - dense_ids)}"
    )
    print("Zero unique contributions would mean fusion is pure overhead here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
