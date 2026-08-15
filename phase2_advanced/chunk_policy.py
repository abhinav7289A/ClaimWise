"""Per-document chunking policy, selected by measured page density.

**The finding this exists to exploit.** Three independent measurements said the
same thing. Hybrid search gave the life policy +9.5 while costing health 4.7
(D-17). Parent-document retrieval gave life +14.3 while costing home 14.3
(D-18). P-17 predicted both from one observation: the life policy packs 11,639
characters into a page against 2,602-5,935 for everything else.

**Optimal chunk size is a property of the document, not of the corpus.** A
single global setting is correct for the typical document and badly wrong for
the outlier. Phase 1 chose 1,000 characters and left roughly 14 points of hit@5
on the table for the densest policy in the corpus.

**Why density and not document name.** Writing `life -> parent_child` into
config would produce the same numbers and teach nothing: it memorises the eval
set. Density is a property of the *document*, measurable at ingest time before
any question has been asked, and it generalises — a new dense policy from a
different insurer gets the right strategy without anyone editing a mapping. The
threshold is the honest part of the claim and is stated in config where it can
be argued with.

**Why this is a build-time policy and not a query-time router.** Everything is
indexed into one collection, so retrieval needs no routing logic, no second
search, and no reconciling scores across separately-built indexes — the same
incomparable-scales problem that forced rank-based fusion in D-17. A router can
be wrong at query time; a build-time policy cannot, because there is no
query-time decision. It also matches how this would really ship: an insurer's
ingestion pipeline picks a chunking policy per document class at upload, and the
serving path stays uniform.

Mixing granularities in one collection works because parent expansion is
opt-in per chunk: dense documents produce children carrying `parent_id`, sparse
documents produce flat chunks with none, and `ParentStore.expand()` passes
anything without a parent through untouched.

**This is a hypothesis under test, not a result.** The projection from D-18 is
0.847, computed as arithmetic over per-type numbers rather than measured. The
strategy split was also chosen using the same 85-item eval set, which is
selection on test data — with 21 life items, part of that +14.3 may be noise.
Run it before believing it.

Usage:
    python -m phase2_advanced.chunk_policy --help
    python -m phase2_advanced.chunk_policy --stats     # decisions only
    python -m phase2_advanced.chunk_policy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from phase1_rag.chunk import build_splitter, chunk_page
from phase2_advanced.parent_docs import build_two_tier, load_pages, write_jsonl

LOGGER = logging.getLogger("claimwise.chunk_policy")

STRATEGY_PARENT_CHILD = "parent_child"
STRATEGY_FLAT = "flat"


@dataclass
class DocumentProfile:
    """What is measured about a document before choosing how to chunk it.

    Attributes:
        doc_id: Content hash of the source PDF.
        filename: Source PDF filename.
        policy_type: e.g. "health". Recorded for reporting only — it is
            deliberately NOT an input to the decision.
        pages: Pages kept after ingestion.
        total_chars: Characters across those pages.
        chars_per_page: The decision variable.
        strategy: The chunking strategy selected for this document.
    """

    doc_id: str
    filename: str
    policy_type: str
    pages: int
    total_chars: int
    chars_per_page: float
    strategy: str


def profile_documents(
    pages: list[dict[str, Any]], density_threshold: int
) -> list[DocumentProfile]:
    """Measure each document's page density and select its chunking strategy.

    Args:
        pages: Page records from `ingest.py`.
        density_threshold: Characters per page at or above which a document is
            treated as dense and gets the parent/child treatment.

    Returns:
        One profile per document, ordered by descending density so the
        outliers — the documents this policy exists for — read first.

    Raises:
        ValueError: If no pages were supplied.
    """
    if not pages:
        raise ValueError("Cannot profile documents from zero pages.")

    totals: dict[str, dict[str, Any]] = {}
    for page in pages:
        entry = totals.setdefault(
            page["doc_id"],
            {
                "filename": page["filename"],
                "policy_type": page["policy_type"],
                "pages": 0,
                "chars": 0,
            },
        )
        entry["pages"] += 1
        entry["chars"] += len(page["text"])

    profiles = []
    for doc_id, entry in totals.items():
        density = entry["chars"] / entry["pages"]
        profiles.append(
            DocumentProfile(
                doc_id=doc_id,
                filename=entry["filename"],
                policy_type=entry["policy_type"],
                pages=entry["pages"],
                total_chars=entry["chars"],
                chars_per_page=round(density, 1),
                strategy=(
                    STRATEGY_PARENT_CHILD
                    if density >= density_threshold
                    else STRATEGY_FLAT
                ),
            )
        )

    return sorted(profiles, key=lambda profile: profile.chars_per_page, reverse=True)


def build_mixed_corpus(
    pages: list[dict[str, Any]],
    profiles: list[DocumentProfile],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chunk every document according to its selected strategy.

    Args:
        pages: Page records from `ingest.py`.
        profiles: Per-document decisions from `profile_documents`.
        settings: Resolved chunking settings for both strategies.

    Returns:
        `(chunks, parents)`. Chunks from both strategies live in one list and
        are indexed into one collection; only those from dense documents carry
        `parent_id`. Parents exist only for dense documents.
    """
    strategy_by_doc = {profile.doc_id: profile.strategy for profile in profiles}
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    flat_splitter = build_splitter(settings["flat_size"], settings["flat_overlap"])

    dense_pages = [p for p in pages if strategy_by_doc[p["doc_id"]] == STRATEGY_PARENT_CHILD]
    sparse_pages = [p for p in pages if strategy_by_doc[p["doc_id"]] == STRATEGY_FLAT]

    chunks: list[dict[str, Any]] = []
    parents: list[dict[str, Any]] = []

    if dense_pages:
        parents, children = build_two_tier(
            dense_pages,
            parent_size=settings["parent_size"],
            parent_overlap=settings["parent_overlap"],
            child_size=settings["child_size"],
            child_overlap=settings["child_overlap"],
        )
        # The same minimum-length filter chunk.py applies. parent_docs.py omits
        # it, which is why its first build produced 6-character children — text
        # too short to carry meaning still occupies a retrieval slot and embeds
        # to noise.
        kept = [
            child for child in children if len(child["text"].strip()) >= settings["min_chunk_chars"]
        ]
        dropped = len(children) - len(kept)
        if dropped:
            LOGGER.info(
                "Dropped %d children under %d chars", dropped, settings["min_chunk_chars"]
            )
        for child in kept:
            child["strategy"] = STRATEGY_PARENT_CHILD
        chunks.extend(kept)

    for page in sparse_pages:
        for record in chunk_page(
            page,
            splitter=flat_splitter,
            min_chunk_chars=settings["min_chunk_chars"],
            chunk_size=settings["flat_size"],
            chunk_overlap=settings["flat_overlap"],
            created_at=created_at,
        ):
            chunk = asdict(record)
            chunk["strategy"] = STRATEGY_FLAT
            # Explicit None rather than absent: `ParentStore.expand()` reads
            # this field to decide whether a chunk expands, and an absent key
            # would rely on getattr defaults doing the right thing silently.
            chunk["parent_id"] = None
            chunks.append(chunk)

    return chunks, parents


def resolve_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge config values with CLI overrides.

    Args:
        config: Parsed `config.yaml`.
        args: Parsed command-line arguments.

    Returns:
        Resolved settings for this run.
    """
    return {
        "density_threshold": args.density_threshold
        or cfg_get(config, "chunk_policy.density_threshold", 8000),
        "child_size": cfg_get(config, "parent_docs.child_size", 400),
        "child_overlap": cfg_get(config, "parent_docs.child_overlap", 50),
        "parent_size": cfg_get(config, "parent_docs.parent_size", 2000),
        "parent_overlap": cfg_get(config, "parent_docs.parent_overlap", 0),
        "flat_size": cfg_get(config, "chunk.chunk_size", 1000),
        "flat_overlap": cfg_get(config, "chunk.chunk_overlap", 150),
        "min_chunk_chars": cfg_get(config, "chunk.min_chunk_chars", 50),
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase2_advanced.chunk_policy",
        description="Chunk each document by its measured page density (technique 5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--density-threshold",
        type=int,
        default=None,
        help="Chars/page at or above which a document gets parent/child chunking.",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Print the decisions without writing files."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Profile every document, chunk by policy, and write one mixed corpus.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if the policy selected one strategy
        for every document, which means the threshold is doing nothing.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    settings = resolve_settings(config, args)
    processed_dir = Path(cfg_get(config, "paths.processed_dir", "data/processed"))
    pages_path = processed_dir / cfg_get(config, "ingest.output_filename", "pages.jsonl")

    pages = load_pages(pages_path)
    profiles = profile_documents(pages, settings["density_threshold"])

    print(f"\ndensity threshold   : {settings['density_threshold']} chars/page\n")
    print(f"{'document':<42} {'type':<8} {'pages':>6} {'chars/page':>11}  strategy")
    for profile in profiles:
        print(
            f"{profile.filename[:42]:<42} {profile.policy_type:<8} {profile.pages:>6} "
            f"{profile.chars_per_page:>11.1f}  {profile.strategy}"
        )

    selected = {profile.strategy for profile in profiles}
    if len(selected) == 1:
        LOGGER.error(
            "Every document selected %s — the threshold is not discriminating. "
            "This policy only earns its complexity when documents differ.",
            selected.pop(),
        )
        return 1

    chunks, parents = build_mixed_corpus(pages, profiles, settings)

    by_strategy: dict[str, int] = {}
    for chunk in chunks:
        by_strategy[chunk["strategy"]] = by_strategy.get(chunk["strategy"], 0) + 1

    print(f"\nchunks out          : {len(chunks)}")
    for strategy, count in sorted(by_strategy.items()):
        print(f"  {strategy:<14}: {count}")
    print(f"parents out         : {len(parents)}")

    if args.stats:
        print("\n--stats: nothing written.")
        return 0

    write_jsonl(chunks, processed_dir / cfg_get(
        config, "chunk_policy.chunks_filename", "mixed_chunks.jsonl"
    ))
    write_jsonl(parents, processed_dir / cfg_get(
        config, "chunk_policy.parents_filename", "mixed_parents.jsonl"
    ))

    decisions_path = processed_dir / "chunk_policy.meta.json"
    decisions_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "density_threshold": settings["density_threshold"],
                "settings": settings,
                "documents": [asdict(profile) for profile in profiles],
                "chunks_by_strategy": by_strategy,
                "parents": len(parents),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"decisions           : {decisions_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
