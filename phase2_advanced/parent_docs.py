"""Parent-document retrieval: index small children, return large parents.

**The problem.** Chunk size is one knob serving two masters that want opposite
things.

*Retrieval wants small chunks.* A 400-character passage about one waiting period
embeds to a vector that points squarely at that topic. A 2,000-character passage
covering five topics embeds to the mush-average of all five and is strongly
"about" none of them — so it ranks mediocre for every one of the five questions
it could have answered.

*Generation wants large chunks.* A clause reading "subject to the sub-limits
specified in Table B" is worthless without Table B. Insurance wording is full of
cross-references, conditions attached two sentences later, and exclusions that
qualify the entitlement above them.

Phase 1 split the difference at 1,000 characters and lost a little of both.

**The fix production systems use.** Decouple the two units: embed fine-grained
*children* for search precision, but when a child is retrieved, hand the reranker
and the generator the larger *parent* block it came from. LangChain calls this
`ParentDocumentRetriever`. It is near-universal in RAG over contracts, policies
and regulatory filings, where the atomic unit of *meaning* is much larger than
the atomic unit of *search*.

**Why this technique and why now.** D-17 established that the reranker's
precision — not pool recall — is the binding constraint: hybrid search failed
because it could only buy recall by widening the pool, which is exactly what
degrades a cross-encoder. This attacks the same constraint from the opposite
side. It does not add candidates; it makes each candidate sharper going in and
more complete coming out, leaving pool size untouched.

It is also a falsifiable test of P-17. The life policy scores 0.619 against
0.86/0.90 for the others, and the standing hypothesis is that its 11,639
characters per page pack so many distinct clauses into each 1,000-character
chunk that no chunk is strongly about anything. If that is right, 400-character
children should sharpen that document specifically. If life does not move, the
hypothesis is wrong and the likelier culprit is the column scrambling recorded
in P-9 — which this technique cannot fix, and which would redirect the rest of
Phase 2.

**The page invariant is preserved.** Parents never span a page break, so every
child still inherits exactly one page number and citations stay verifiable. That
was the Phase 1 rule and it survives here for the same reason: a citation the
reader cannot check is worse than no citation.

Usage:
    python -m phase2_advanced.parent_docs --help
    python -m phase2_advanced.parent_docs                    # build both tiers
    python -m phase2_advanced.parent_docs --stats            # inspect, write nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config

LOGGER = logging.getLogger("claimwise.parent_docs")

# Carried unchanged from every page record onto both tiers. These are the
# provenance fields that become a citation, so they must survive both splits.
INHERITED_FIELDS = ("doc_id", "filename", "insurer", "policy_type", "doc_label", "page")


@dataclass
class TierStats:
    """Size distribution for one tier, used to sanity-check a build.

    Attributes:
        count: Number of records in the tier.
        min_chars: Shortest record.
        median_chars: Median record length.
        mean_chars: Mean record length.
        max_chars: Longest record — must not exceed the configured budget.
        over_budget: Records above the embedding model's input limit.
    """

    count: int
    min_chars: int
    median_chars: float
    mean_chars: float
    max_chars: int
    over_budget: int


def summarise(texts: list[str], budget: int) -> TierStats:
    """Describe a tier's size distribution.

    Args:
        texts: Every record's text in the tier.
        budget: Character limit above which embedding would silently truncate.

    Returns:
        The tier's statistics.

    Raises:
        ValueError: If the tier is empty, which always indicates a build fault.
    """
    if not texts:
        raise ValueError("Cannot summarise an empty tier.")
    lengths = [len(text) for text in texts]
    return TierStats(
        count=len(lengths),
        min_chars=min(lengths),
        median_chars=round(statistics.median(lengths), 1),
        mean_chars=round(statistics.fmean(lengths), 1),
        max_chars=max(lengths),
        over_budget=sum(1 for length in lengths if length > budget),
    )


def load_pages(path: Path) -> list[dict[str, Any]]:
    """Read page records produced by `ingest.py`.

    Args:
        path: Path to `pages.jsonl`.

    Returns:
        Decoded page records.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Pages file not found: {path.resolve()}. Run `python -m phase1_rag.ingest` first."
        )
    with path.open("r", encoding="utf-8") as handle:
        pages = [json.loads(line) for line in handle if line.strip()]
    if not pages:
        raise ValueError(f"{path} contains no pages.")
    return pages


def inherit(source: dict[str, Any]) -> dict[str, Any]:
    """Copy the provenance fields that both tiers must carry.

    Args:
        source: A page or parent record.

    Returns:
        Just the inherited fields.

    Raises:
        KeyError: If a provenance field is missing. Raised loudly because a
            chunk without a page number produces an uncitable answer, and
            discovering that at generation time is far more expensive.
    """
    missing = [field for field in INHERITED_FIELDS if field not in source]
    if missing:
        raise KeyError(f"Record is missing provenance fields {missing}: {source.get('doc_id')!r}")
    return {field: source[field] for field in INHERITED_FIELDS}


def build_two_tier(
    pages: Iterable[dict[str, Any]],
    parent_size: int,
    parent_overlap: int,
    child_size: int,
    child_overlap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split every page into parents, then each parent into children.

    The nesting order matters and is not interchangeable with splitting the page
    twice independently. Children are cut *within* a parent's boundaries, which
    is what guarantees every child has exactly one parent containing it whole.
    Splitting both tiers straight from the page would leave children straddling
    two parents, and the expansion step would then have to choose one and
    silently drop context.

    Parent overlap defaults to zero. Overlap earns its place at the child tier,
    where it stops a clause being cut mid-sentence, but parents are returned
    *whole* into the prompt — overlapping them would feed the generator the same
    sentences twice and spend context window on duplication.

    Args:
        pages: Page records from `ingest.py`.
        parent_size: Target characters per parent.
        parent_overlap: Characters shared between adjacent parents.
        child_size: Target characters per child.
        child_overlap: Characters shared between adjacent children.

    Returns:
        `(parents, children)`. Every child carries `parent_id`; every parent
        carries `child_count`.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size, chunk_overlap=parent_overlap
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size, chunk_overlap=child_overlap
    )

    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []

    for page in pages:
        provenance = inherit(page)
        for parent_index, parent_text in enumerate(parent_splitter.split_text(page["text"])):
            # Deterministic IDs, exactly as in chunk.py: re-running produces
            # identical ids so the vector store is upserted, never duplicated.
            parent_id = f"{provenance['doc_id']}_p{provenance['page']}_P{parent_index}"
            child_texts = child_splitter.split_text(parent_text)

            parents.append(
                {
                    **provenance,
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                    "text": parent_text,
                    "child_count": len(child_texts),
                }
            )

            for child_index, child_text in enumerate(child_texts):
                children.append(
                    {
                        **provenance,
                        # Named `chunk_id`, not `child_id`: embed_index and the
                        # whole retrieval path already speak that field, and a
                        # rename would ripple through code this technique has no
                        # business touching.
                        "chunk_id": f"{parent_id}_c{child_index}",
                        "chunk_index": child_index,
                        "parent_id": parent_id,
                        "text": child_text,
                    }
                )

    return parents, children


class ParentStore:
    """Query-time lookup from a retrieved child back to its parent block."""

    def __init__(self, parents: list[dict[str, Any]]) -> None:
        """Index parents by id.

        Args:
            parents: Decoded records from `parents.jsonl`.

        Raises:
            ValueError: If no parents were supplied.
        """
        if not parents:
            raise ValueError("Cannot build a ParentStore from zero parents.")
        self._by_id = {parent["parent_id"]: parent for parent in parents}
        LOGGER.info("Parent store loaded: %d parents", len(self._by_id))

    def __len__(self) -> int:
        """Return the number of parents held."""
        return len(self._by_id)

    def expand(self, chunks: list[Any], top_k: int | None = None) -> list[Any]:
        """Replace retrieved children with their parent blocks, order preserved.

        Deduplication is the whole point and the reason this cannot be a simple
        map. Small children mean several hits often share one parent — that is
        the technique working, since agreement between neighbouring children is
        evidence the parent is relevant. But emitting that parent three times
        would waste prompt budget and let one block crowd out the rest of the
        context. First occurrence wins, so the best-ranked child decides where
        its parent lands.

        A child whose parent is missing from the store is passed through
        unchanged rather than dropped. Losing a correct retrieval to a stale
        parent file would be an invisible recall regression; a slightly smaller
        context window is the cheaper failure.

        Args:
            chunks: Retrieved children, best first. Each needs `chunk_id`,
                `parent_id` and `text`; anything matching that shape works, which
                keeps this module independent of the retrieval dataclass.
            top_k: Cap on parents returned. None returns all.

        Returns:
            Parent-expanded chunks, best first, one entry per distinct parent.
        """
        seen: set[str] = set()
        expanded: list[Any] = []

        for chunk in chunks:
            parent_id = getattr(chunk, "parent_id", None)
            parent = self._by_id.get(parent_id) if parent_id else None

            if parent is None:
                if parent_id:
                    LOGGER.warning(
                        "Parent %s not found for child %s — passing the child through. "
                        "Rebuild parents.jsonl if this is frequent.",
                        parent_id,
                        getattr(chunk, "chunk_id", "?"),
                    )
                expanded.append(chunk)
                continue

            if parent_id in seen:
                continue
            seen.add(parent_id)

            # Swap the text, keep everything else — score and rank belong to the
            # child that earned the hit, and the reranker downstream reads them.
            expanded.append(replace_text(chunk, parent["text"]))
            if top_k and len(expanded) >= top_k:
                break

        return expanded


def replace_text(chunk: Any, text: str) -> Any:
    """Return a copy of a retrieved chunk carrying different text.

    Args:
        chunk: A retrieval result. Dataclasses are copied via `dataclasses.replace`;
            anything else falls back to mutating a shallow copy.
        text: The replacement text.

    Returns:
        A chunk of the same type with `text` swapped.
    """
    import copy
    import dataclasses

    if dataclasses.is_dataclass(chunk) and not isinstance(chunk, type):
        return dataclasses.replace(chunk, text=text)
    duplicate = copy.copy(chunk)
    duplicate.text = text
    return duplicate


def load_parent_store(config: dict[str, Any], parents_path: Path | None = None) -> ParentStore:
    """Construct the configured parent store.

    Args:
        config: Parsed `config.yaml`.
        parents_path: Override the configured parents path.

    Returns:
        A ready-to-use store.

    Raises:
        FileNotFoundError: If the parents file has not been built.
    """
    path = parents_path or Path(
        cfg_get(config, "parent_docs.parents_path", "data/processed/parents.jsonl")
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Parents file not found: {path.resolve()}. "
            "Run `python -m phase2_advanced.parent_docs` first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return ParentStore([json.loads(line) for line in handle if line.strip()])


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as JSON Lines.

    Args:
        records: Records to write.
        path: Destination file; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote %d records to %s", len(records), path)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase2_advanced.parent_docs",
        description="Build the two-tier parent/child chunk sets for technique 3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--parent-size", type=int, default=None, help="Override parent size.")
    parser.add_argument("--child-size", type=int, default=None, help="Override child size.")
    parser.add_argument(
        "--stats", action="store_true", help="Report the split without writing any files."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build both tiers and report their size distributions.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if a tier would truncate at
        embedding time.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    processed_dir = Path(cfg_get(config, "paths.processed_dir", "data/processed"))
    pages_path = processed_dir / cfg_get(config, "ingest.output_filename", "pages.jsonl")

    parent_size = args.parent_size or cfg_get(config, "parent_docs.parent_size", 2000)
    child_size = args.child_size or cfg_get(config, "parent_docs.child_size", 400)
    child_overlap = cfg_get(config, "parent_docs.child_overlap", 50)
    parent_overlap = cfg_get(config, "parent_docs.parent_overlap", 0)
    embed_budget = cfg_get(config, "parent_docs.embed_budget_chars", 2000)

    pages = load_pages(pages_path)
    parents, children = build_two_tier(
        pages, parent_size, parent_overlap, child_size, child_overlap
    )

    parent_stats = summarise([parent["text"] for parent in parents], embed_budget)
    child_stats = summarise([child["text"] for child in children], embed_budget)

    print(f"\npages in            : {len(pages)}")
    print(f"parents out         : {parent_stats.count}  (target {parent_size} chars)")
    print(f"children out        : {child_stats.count}  (target {child_size} chars)")
    print(f"children per parent : {child_stats.count / parent_stats.count:.2f}")
    print("\n                     min / median / mean / max")
    print(
        f"parent chars       : {parent_stats.min_chars} / {parent_stats.median_chars} / "
        f"{parent_stats.mean_chars} / {parent_stats.max_chars}"
    )
    print(
        f"child chars        : {child_stats.min_chars} / {child_stats.median_chars} / "
        f"{child_stats.mean_chars} / {child_stats.max_chars}"
    )

    # Only children are embedded, so only children can be silently truncated.
    if child_stats.over_budget:
        LOGGER.error(
            "%d children exceed the %d-char embedding budget — they would be silently "
            "truncated at index time.",
            child_stats.over_budget,
            embed_budget,
        )
        return 1

    per_type: dict[str, int] = {}
    for child in children:
        per_type[child["policy_type"]] = per_type.get(child["policy_type"], 0) + 1
    print("\nchildren by policy type:")
    for policy_type, count in sorted(per_type.items()):
        print(f"  {policy_type:<8}: {count}")

    if args.stats:
        print("\n--stats: nothing written.")
        return 0

    write_jsonl(parents, processed_dir / cfg_get(
        config, "parent_docs.parents_filename", "parents.jsonl"
    ))
    write_jsonl(children, processed_dir / cfg_get(
        config, "parent_docs.children_filename", "children.jsonl"
    ))
    print("\nNext: index the CHILDREN into their own collection, leaving the")
    print("Phase 1 collection intact so the baseline stays reproducible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
