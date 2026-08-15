"""Cross-encoder reranking — Phase 2, technique 1.

**The problem with the Phase 1 baseline.** A bi-encoder embeds the query and each
chunk *independently*, then compares vectors. That independence is what makes
search fast — every chunk vector is computed once at upload and reused forever —
but it means the model never sees the query and the chunk together. By the time
they meet, each has been frozen into 384 numbers. It cannot notice that "age 65"
satisfies a clause about "61 years and above", because that judgement requires
reading both at once.

**What a cross-encoder does differently.** It concatenates query and passage into
one input and runs full attention across the pair, reading the question *while*
reading the text. Far better at relevance, and roughly 100x more expensive per
pair — it cannot precompute anything, so every (query, chunk) pair is a forward
pass. That is exactly why it belongs in a second stage: the bi-encoder cheaply
narrows 653 chunks to ~30, and the cross-encoder expensively reorders those 30.

**Why this is the right first technique, from measurement rather than intuition.**
The Phase 1 depth run recorded hit@5 = 0.694 against recall@50 = 0.953. For 95%
of questions the correct page is *already retrieved* and merely ranked too low to
survive the top-5 cut. Only 4 of 85 questions are genuinely unreachable.
Reranking's entire job is to close that 26-point gap; it cannot invent results,
so its ceiling is precisely the recall of the candidate set it is given.

Usage:
    python -m phase2_advanced.rerank --help
    python -m phase2_advanced.rerank -q "What is the co-payment at age 65?"
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from phase1_rag.embed_index import collection_name_for
from phase1_rag.rag_chain import RetrievedChunk, build_search_filter, retrieve

LOGGER = logging.getLogger("claimwise.rerank")


class CrossEncoderReranker:
    """Reorders retrieved chunks by cross-encoder relevance score."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        """Load the cross-encoder.

        The first call downloads the model (~1.1GB for bge-reranker-base) and
        caches it under ~/.cache/huggingface.

        Args:
            model_name: HuggingFace cross-encoder id.
            device: "cpu" or "cuda".
            batch_size: Pairs scored per forward pass.
            max_length: Combined query+passage tokens the model will read.
                Passed explicitly rather than left to the tokenizer default
                because it silently caps what this stage can see:
                bge-reranker-base is `model_max_length=512`, and anything longer
                is truncated with no warning. Phase 2's parent-document run lost
                4.7 points of hit@5 to exactly this — 1,780-character parents
                clipped mid-block, so the cross-encoder scored their openings and
                never read their endings. Stated here so the limit is a visible
                constraint that shapes chunk sizing, not a hidden one.
        """
        LOGGER.info(
            "Loading cross-encoder %s on %s (max_length=%d tokens)", model_name, device, max_length
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Score every (query, chunk) pair and return the chunks reordered.

        Each returned chunk carries the cross-encoder score in `score`, so
        downstream code reads relevance the same way regardless of which stage
        produced it. Note the scales differ: bi-encoder cosine sits in roughly
        0.4-0.9, while bge-reranker emits unbounded logits. Scores are therefore
        comparable *within* a stage, never across stages.

        Args:
            query: The user's question.
            chunks: Candidates from the bi-encoder, any order.
            top_k: Return only this many. None returns all, reordered — which is
                what the eval harness wants, so hit@k stays computable at every k.

        Returns:
            Chunks sorted by descending cross-encoder score.
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        rescored = [
            dataclasses.replace(chunk, score=round(float(score), 4))
            for chunk, score in zip(chunks, scores)
        ]
        rescored.sort(key=lambda chunk: chunk.score, reverse=True)
        return rescored[:top_k] if top_k else rescored


def build_reranker(config: dict[str, Any], model_name: str | None = None) -> CrossEncoderReranker:
    """Construct the configured reranker.

    Args:
        config: Parsed `config.yaml`.
        model_name: Override `rerank.model_name`.

    Returns:
        A ready-to-use reranker.
    """
    return CrossEncoderReranker(
        model_name=model_name or cfg_get(config, "rerank.model_name", "BAAI/bge-reranker-base"),
        device=cfg_get(config, "rerank.device", "cpu"),
        batch_size=cfg_get(config, "rerank.batch_size", 16),
        max_length=cfg_get(config, "rerank.max_length", 512),
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the demonstration CLI.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase2_advanced.rerank",
        description="Show how cross-encoder reranking reorders retrieved chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-q", "--question", required=True, help="Question to retrieve for.")
    parser.add_argument("--depth", type=int, default=None, help="Override rerank.candidate_depth.")
    parser.add_argument("--show", type=int, default=8, help="How many rows to print.")
    parser.add_argument("--model", default=None, help="Override rerank.model_name.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Retrieve, rerank, and print the before/after ordering side by side.

    Runs one query so the effect is visible and verifiable before committing to
    a full evaluation — the same self-verifying pattern as `embed_index.py`,
    which is what caught the ligature corruption in Phase 1.

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
    depth = args.depth or cfg_get(config, "rerank.candidate_depth", 30)
    embed_model = cfg_get(config, "embed.model_name")
    collection_name = collection_name_for(
        cfg_get(config, "index.collection_prefix", "claimwise"), embed_model
    )

    embedder = SentenceTransformer(embed_model, device=cfg_get(config, "embed.device", "cpu"))
    reranker = build_reranker(config, model_name=args.model)

    client = QdrantClient(path=cfg_get(config, "index.path", "qdrant_storage"))
    try:
        if not client.collection_exists(collection_name):
            LOGGER.error("Collection %s does not exist. Run embed_index first.", collection_name)
            return 1

        retrieve_started = time.perf_counter()
        candidates = retrieve(
            client,
            collection_name=collection_name,
            embedder=embedder,
            question=args.question,
            top_k=depth,
            query_prefix=cfg_get(config, "embed.query_prefix", ""),
            normalize=cfg_get(config, "embed.normalize", True),
            search_filter=build_search_filter(
                cfg_get(config, "index.default_user_id", "local-dev")
            ),
        )
        retrieve_ms = (time.perf_counter() - retrieve_started) * 1000
    finally:
        client.close()

    if not candidates:
        LOGGER.error("No chunks retrieved.")
        return 1

    rerank_started = time.perf_counter()
    reranked = reranker.rerank(args.question, candidates)
    rerank_ms = (time.perf_counter() - rerank_started) * 1000

    before = {chunk.chunk_id: rank for rank, chunk in enumerate(candidates, start=1)}

    print(f"\nQ: {args.question}")
    print(f"candidates: {len(candidates)}  |  retrieve {retrieve_ms:.0f} ms  |  rerank {rerank_ms:.0f} ms")

    print("\n=== BEFORE (bi-encoder cosine) ===")
    for rank, chunk in enumerate(candidates[: args.show], start=1):
        snippet = " ".join(chunk.text.split())[:90]
        print(f"{rank:>2}. {chunk.score:>7.4f}  p.{chunk.page:<4} [{chunk.insurer}] {snippet}...")

    print("\n=== AFTER (cross-encoder) ===")
    for rank, chunk in enumerate(reranked[: args.show], start=1):
        snippet = " ".join(chunk.text.split())[:90]
        moved = before[chunk.chunk_id] - rank
        arrow = f"+{moved}" if moved > 0 else (str(moved) if moved < 0 else " =")
        print(
            f"{rank:>2}. {chunk.score:>8.3f}  p.{chunk.page:<4} "
            f"[{chunk.insurer}] {arrow:>4}  {snippet}..."
        )

    promoted = sum(1 for chunk in reranked[:5] if before[chunk.chunk_id] > 5)
    print(
        f"\n{promoted} of the new top-5 came from outside the original top-5. "
        "Zero would mean reranking changed nothing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
