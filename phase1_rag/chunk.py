"""Split cleaned pages into overlapping chunks, one idea per vector.

An embedding is a single vector, and a vector can only carry so much meaning.
Our corpus averages ~5,800 characters per page, which is roughly 1,400 tokens —
about three times what bge-small can even read, and far more than one vector can
represent usefully. Embed a whole page and the "waiting period for pre-existing
diseases" signal is averaged together with room rent, co-pay and AYUSH text
until the page matches none of those queries strongly.

**Splitting strategy.** `RecursiveCharacterTextSplitter` tries a list of
separators in order — paragraph break, line break, space, then bare character —
and only falls back to a harsher one when the gentler one leaves a piece over
budget. In practice that means chunks end at real boundaries almost always, and
mid-word cuts happen only inside an unbroken run of text longer than the budget.

**Chunks never span a page break.** Each page is split independently so every
chunk inherits one exact page number. That is what makes a citation verifiable:
"page 34" has to actually be page 34. The cost is that a clause continuing onto
the next page gets divided, which Phase 2's parent-document retrieval is the
proper fix for — not a bigger chunk size here.

Usage:
    python -m phase1_rag.chunk --help
    python -m phase1_rag.chunk
    python -m phase1_rag.chunk --chunk-size 600 --overlap 100
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config

LOGGER = logging.getLogger("claimwise.chunk")

# bge-small truncates input at 512 tokens. English averages ~4 characters per
# token, so anything past ~2000 characters would be silently dropped at
# embedding time. Used only to warn — never to reshape chunks behind your back.
EMBEDDING_CHAR_BUDGET = 2000


@dataclass(frozen=True)
class ChunkRecord:
    """One embeddable chunk of one page.

    Attributes:
        chunk_id: Stable, human-readable id — `{doc_id}_p{page}_c{index}`.
            Deterministic, so re-running chunking produces identical ids and
            the vector store can be upserted rather than rebuilt.
        doc_id: Content hash of the source PDF, carried from ingestion.
        filename: Source PDF filename, for human debugging.
        insurer: e.g. "sbigeneral" — Phase 2 filters retrieval on this.
        policy_type: e.g. "health".
        doc_label: Free-text document label, e.g. "alpha".
        page: 1-indexed PDF page this chunk came from. Becomes the citation.
        chunk_index: Position of this chunk within its page, starting at 0.
        text: The chunk text.
        char_count: Length of `text`.
        chunk_size: Configured target size, recorded so a metric can be traced
            back to the exact chunking that produced it.
        chunk_overlap: Configured overlap.
        created_at: UTC ISO-8601 timestamp of the run.
    """

    chunk_id: str
    doc_id: str
    filename: str
    insurer: str
    policy_type: str
    doc_label: str
    page: int
    chunk_index: int
    text: str
    char_count: int
    chunk_size: int
    chunk_overlap: int
    created_at: str


def load_pages(pages_path: Path) -> Iterator[dict[str, Any]]:
    """Stream page records from a JSONL file.

    Streaming rather than loading the list at once, so corpus growth never
    becomes a memory problem.

    Args:
        pages_path: Path to `pages.jsonl` produced by `ingest.py`.

    Yields:
        One decoded page record per line.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is not valid JSON.
    """
    if not pages_path.is_file():
        raise FileNotFoundError(
            f"Pages file not found: {pages_path.resolve()}. "
            "Run `python -m phase1_rag.ingest` first."
        )

    with pages_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSON at {pages_path}:{line_number} — {error}"
                ) from error


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Construct the recursive character splitter.

    Separators are tried in order and the splitter descends to the next one only
    when a piece is still over budget:
      "\\n\\n" paragraph → "\\n" line → " " word → "" bare character.

    Args:
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters shared between consecutive chunks.

    Returns:
        A configured splitter.

    Raises:
        ValueError: If overlap is not smaller than chunk size, which would make
            the splitter unable to advance.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})."
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        # Length in characters, not tokens: our budget is expressed in
        # characters and a token-based length would drag in a tokenizer that
        # is not the one bge-small actually uses.
        length_function=len,
        keep_separator=False,
    )


def chunk_page(
    page: dict[str, Any],
    splitter: RecursiveCharacterTextSplitter,
    min_chunk_chars: int,
    chunk_size: int,
    chunk_overlap: int,
    created_at: str,
) -> list[ChunkRecord]:
    """Split one page record into chunk records.

    Args:
        page: A decoded page record from `pages.jsonl`.
        splitter: The configured recursive splitter.
        min_chunk_chars: Drop chunks shorter than this.
        chunk_size: Recorded on each chunk for provenance.
        chunk_overlap: Recorded on each chunk for provenance.
        created_at: UTC ISO-8601 timestamp stamped onto every chunk.

    Returns:
        Chunk records for this page, in reading order.
    """
    pieces = splitter.split_text(page["text"])

    records: list[ChunkRecord] = []
    for piece in pieces:
        text = piece.strip()
        if len(text) < min_chunk_chars:
            continue
        # chunk_index counts kept chunks, so ids stay contiguous per page.
        index = len(records)
        records.append(
            ChunkRecord(
                chunk_id=f"{page['doc_id']}_p{page['page']}_c{index}",
                doc_id=page["doc_id"],
                filename=page["filename"],
                insurer=page["insurer"],
                policy_type=page["policy_type"],
                doc_label=page["doc_label"],
                page=page["page"],
                chunk_index=index,
                text=text,
                char_count=len(text),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                created_at=created_at,
            )
        )
    return records


def summarize_chunks(chunks: list[ChunkRecord]) -> dict[str, Any]:
    """Compute corpus statistics for the metrics log.

    Args:
        chunks: All chunk records produced by the run.

    Returns:
        A summary dict with overall size statistics and a per-document
        breakdown. Empty-safe.
    """
    if not chunks:
        return {"total_chunks": 0, "per_document": []}

    lengths = [chunk.char_count for chunk in chunks]
    per_document: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"chunks": 0, "chars": 0, "pages": set()}
    )
    for chunk in chunks:
        entry = per_document[chunk.filename]
        entry["chunks"] += 1
        entry["chars"] += chunk.char_count
        entry["pages"].add(chunk.page)

    return {
        "total_chunks": len(chunks),
        "min_chars": min(lengths),
        "median_chars": int(statistics.median(lengths)),
        "mean_chars": round(statistics.fmean(lengths)),
        "max_chars": max(lengths),
        "over_embedding_budget": sum(1 for n in lengths if n > EMBEDDING_CHAR_BUDGET),
        "per_document": [
            {
                "filename": filename,
                "chunks": entry["chunks"],
                "pages": len(entry["pages"]),
                "mean_chars": round(entry["chars"] / entry["chunks"]),
            }
            for filename, entry in per_document.items()
        ],
    }


def write_jsonl(chunks: list[ChunkRecord], output_path: Path) -> None:
    """Write chunk records to a JSONL file, one JSON object per line.

    Args:
        chunks: Chunk records to write.
        output_path: Destination path; parent directories are created.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser. Defaults are `None` so `config.yaml`
        supplies values unless explicitly overridden.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase1_rag.chunk",
        description="Split cleaned pages into overlapping, page-scoped chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml."
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="Input pages JSONL. Default: from config."
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output chunks JSONL. Default: from config."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="Override chunk.chunk_size (characters)."
    )
    parser.add_argument(
        "--overlap", type=int, default=None, help="Override chunk.chunk_overlap (characters)."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def resolve_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge config file values with CLI overrides.

    Args:
        config: Parsed `config.yaml`.
        args: Parsed command-line arguments.

    Returns:
        The fully resolved `chunk` settings, which are logged with the run.
    """
    return {
        "chunk_size": (
            args.chunk_size
            if args.chunk_size is not None
            else cfg_get(config, "chunk.chunk_size", 1000)
        ),
        "chunk_overlap": (
            args.overlap
            if args.overlap is not None
            else cfg_get(config, "chunk.chunk_overlap", 150)
        ),
        "min_chunk_chars": cfg_get(config, "chunk.min_chunk_chars", 50),
    }


def main(argv: list[str] | None = None) -> int:
    """Run chunking end to end.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if no chunks were produced.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    settings = resolve_settings(config, args)

    processed_dir = Path(cfg_get(config, "paths.processed_dir", "data/processed"))
    input_path = args.input or processed_dir / cfg_get(config, "chunk.input_filename", "pages.jsonl")
    output_path = args.output or processed_dir / cfg_get(
        config, "chunk.output_filename", "chunks.jsonl"
    )

    splitter = build_splitter(settings["chunk_size"], settings["chunk_overlap"])
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    pages = list(load_pages(input_path))
    LOGGER.info(
        "Chunking %d page(s) at size=%d overlap=%d",
        len(pages),
        settings["chunk_size"],
        settings["chunk_overlap"],
    )

    all_chunks: list[ChunkRecord] = []
    for page in tqdm(pages, desc="Chunking", unit="page"):
        all_chunks.extend(
            chunk_page(
                page,
                splitter=splitter,
                min_chunk_chars=settings["min_chunk_chars"],
                chunk_size=settings["chunk_size"],
                chunk_overlap=settings["chunk_overlap"],
                created_at=created_at,
            )
        )

    write_jsonl(all_chunks, output_path)

    stats = summarize_chunks(all_chunks)
    summary = {
        "created_at": created_at,
        "input_path": input_path.as_posix(),
        "output_path": output_path.as_posix(),
        "settings": settings,
        "pages_in": len(pages),
        **stats,
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== CHUNK SUMMARY ===")
    print(f"pages in           : {len(pages)}")
    print(f"chunks out         : {stats['total_chunks']}")
    if all_chunks:
        print(
            f"chars per chunk    : min {stats['min_chars']} / "
            f"median {stats['median_chars']} / mean {stats['mean_chars']} / "
            f"max {stats['max_chars']}"
        )
        print(f"over {EMBEDDING_CHAR_BUDGET}-char budget : {stats['over_embedding_budget']}")
    print(f"output             : {output_path}")
    print(f"run metadata       : {meta_path}")
    for entry in stats["per_document"]:
        print(
            f"  - {entry['filename']}: {entry['chunks']} chunks "
            f"across {entry['pages']} pages, mean {entry['mean_chars']} chars"
        )

    if not all_chunks:
        LOGGER.error("No chunks were produced. Is %s empty?", input_path)
        return 1
    if stats["over_embedding_budget"]:
        # Loud, but not fatal: these chunks will be truncated by the embedder.
        LOGGER.warning(
            "%d chunk(s) exceed %d characters and will be truncated at embedding time.",
            stats["over_embedding_budget"],
            EMBEDDING_CHAR_BUDGET,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
