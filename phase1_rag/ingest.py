"""Parse policy PDFs into one cleaned, page-scoped JSONL record per page.

This is the first tier of the RAG pipeline and it sets the ceiling for
everything after it: a fact mangled here can never be retrieved later, no
matter how good the reranker is.

Two things matter more than raw text quality:

1. **Page provenance.** Every record carries its 1-indexed PDF page number, so
   an answer can cite "page 34" and a human can verify it. In insurance an
   unverifiable answer is worthless. The page number survives chunking,
   embedding, retrieval, and lands in the frontend citation chip.
2. **Boilerplate removal.** Policy PDFs stamp the insurer name, UIN, and
   "Page 12 of 88" onto every page. Left in, that text dominates the embedding
   of every short chunk and makes unrelated pages look similar to each other.

No LLM is involved here and none should be: this step runs on CPU at upload
time and only makes the document *searchable*.

Usage:
    python -m phase1_rag.ingest --help
    python -m phase1_rag.ingest
    python -m phase1_rag.ingest --input data/raw/starhealth__health__comprehensive.pdf
    python -m phase1_rag.ingest --mode blocks --output data/processed/pages_blocks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pymupdf  # PyMuPDF >= 1.24.3 exposes this modern module name
except ImportError:  # older PyMuPDF releases only ship the legacy `fitz` name
    import fitz as pymupdf

from tqdm import tqdm

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config

LOGGER = logging.getLogger("claimwise.ingest")

# Filenames are expected as `insurer__policytype__label.pdf`, e.g.
# `starhealth__health__comprehensive-2024.pdf`. This gives us the metadata that
# Phase 2 filters on and Phase 3 compares across, with no sidecar manifest to
# keep in sync. Files that don't follow it still ingest, with a loud warning.
FILENAME_PATTERN = re.compile(r"^(?P<insurer>[^_]+)__(?P<policy_type>[^_]+)__(?P<label>.+)$")

UNKNOWN = "unknown"


@dataclass(frozen=True)
class PageRecord:
    """One cleaned page of one policy document.

    Attributes:
        doc_id: Stable 12-char hash of the file's bytes. Re-ingesting the same
            PDF yields the same id, so downstream steps are idempotent.
        filename: Original filename, for human debugging.
        source_path: Path the file was read from, relative to the repo root.
        insurer: Parsed from the filename, e.g. "starhealth".
        policy_type: Parsed from the filename, e.g. "health".
        doc_label: Free-text remainder of the filename, e.g. "comprehensive-2024".
        page: 1-indexed PDF page number — what a PDF viewer displays, which may
            differ from any page number printed on the page itself.
        text: Cleaned page text.
        char_count: Length of `text`, used for quick corpus statistics.
        extraction_mode: "text" or "blocks" — recorded so a metrics entry can be
            traced back to how the corpus was parsed.
        ingested_at: UTC ISO-8601 timestamp of the run.
    """

    doc_id: str
    filename: str
    source_path: str
    insurer: str
    policy_type: str
    doc_label: str
    page: int
    text: str
    char_count: int
    extraction_mode: str
    ingested_at: str


def compute_doc_id(pdf_path: Path) -> str:
    """Hash a file's contents into a short, stable document id.

    Content-based rather than name-based so that renaming a file does not
    create a duplicate document in the vector store.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        The first 12 hex characters of the file's SHA-256 digest.
    """
    digest = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def parse_filename_metadata(pdf_path: Path) -> tuple[str, str, str]:
    """Extract insurer / policy type / label from a filename.

    Args:
        pdf_path: Path whose stem is expected to look like
            `insurer__policytype__label`.

    Returns:
        A (insurer, policy_type, doc_label) tuple. Missing parts fall back to
        "unknown" with a warning rather than an exception, so one badly named
        file cannot block a whole ingestion run.
    """
    stem = pdf_path.stem
    match = FILENAME_PATTERN.match(stem)
    if match is None:
        LOGGER.warning(
            "Filename %r does not follow 'insurer__policytype__label.pdf'; "
            "metadata filtering in Phase 2 will not work for this document.",
            pdf_path.name,
        )
        return UNKNOWN, UNKNOWN, stem.lower()
    return (
        match.group("insurer").strip().lower(),
        match.group("policy_type").strip().lower(),
        match.group("label").strip().lower(),
    )


def extract_page_texts(document: "pymupdf.Document", extraction_mode: str) -> list[str]:
    """Pull raw text out of every page of an open PDF.

    Args:
        document: An open PyMuPDF document.
        extraction_mode: "text" for plain reading-order extraction, or "blocks"
            for positioned text blocks sorted top-to-bottom then left-to-right.

    Returns:
        Raw (uncleaned) text, one string per page, in page order.

    Raises:
        ValueError: If `extraction_mode` is not a supported mode.
    """
    if extraction_mode == "text":
        return [page.get_text("text") for page in document]

    if extraction_mode == "blocks":
        pages: list[str] = []
        for page in document:
            # Each block is (x0, y0, x1, y1, text, block_no, block_type);
            # block_type 0 is text, 1 is an image.
            blocks = [block for block in page.get_text("blocks") if block[6] == 0]
            # Round y to the nearest 3pt so that words on the same visual line
            # are not reordered by sub-pixel baseline differences.
            blocks.sort(key=lambda block: (round(block[1] / 3), block[0]))
            pages.append("\n\n".join(block[4].strip() for block in blocks if block[4].strip()))
        return pages

    raise ValueError(
        f"Unsupported extraction_mode {extraction_mode!r}; expected 'text' or 'blocks'."
    )


def _normalize_line_for_counting(line: str) -> str:
    """Reduce a line to a comparable form for boilerplate detection.

    Digits become '#' so that "Page 12 of 88" and "Page 13 of 88" are counted as
    the same recurring line.

    Args:
        line: A single line of page text.

    Returns:
        A lowercased, whitespace-collapsed, digit-masked version of the line.
    """
    return re.sub(r"\d+", "#", " ".join(line.split()).lower())


def find_boilerplate_lines(
    page_texts: list[str], threshold: float, edge_lines: int
) -> set[str]:
    """Identify recurring header/footer lines across a document.

    Only the first and last `edge_lines` lines of each page are candidates, so
    that body text which happens to repeat (e.g. a common clause heading) is
    left in place.

    Args:
        page_texts: Raw text for every page of one document.
        threshold: Fraction of pages a line must appear on to count as
            boilerplate, between 0 and 1.
        edge_lines: How many lines at the top and bottom of a page to consider.

    Returns:
        Normalized line forms judged to be boilerplate. Empty if the document is
        too short (under 5 pages) to infer a pattern reliably.
    """
    if len(page_texts) < 5:
        return set()

    counts: Counter[str] = Counter()
    for text in page_texts:
        lines = [line for line in text.splitlines() if line.strip()]
        edges = lines[:edge_lines] + lines[-edge_lines:]
        # A line repeated twice on one page still counts once.
        counts.update({_normalize_line_for_counting(line) for line in edges})

    minimum_pages = max(2, int(threshold * len(page_texts)))
    return {line for line, count in counts.items() if count >= minimum_pages and line}


def clean_page_text(
    raw_text: str,
    boilerplate: set[str],
    edge_lines: int,
    dehyphenate: bool,
    normalize_unicode: bool,
) -> str:
    """Normalise one page of extracted text.

    Args:
        raw_text: Text as returned by `extract_page_texts`.
        boilerplate: Normalized lines to strip when they appear at a page edge.
        edge_lines: How many lines at each edge are eligible for stripping.
        dehyphenate: Rejoin words broken across a line break.
        normalize_unicode: Apply NFKC normalisation.

    Returns:
        Cleaned text with collapsed whitespace and preserved paragraph breaks.
    """
    text = raw_text

    if normalize_unicode:
        # NFKC folds ligatures ("ﬁ" -> "fi") and exotic spaces into ASCII
        # equivalents, so a query and a chunk are embedded from the same
        # characters. The rupee sign is left untouched by NFKC.
        text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")

    if dehyphenate:
        # "hospi-\ntalisation" -> "hospitalisation"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    lines = text.split("\n")
    if boilerplate:
        keep: list[str] = []
        last_index = len(lines) - 1
        for index, line in enumerate(lines):
            at_edge = index < edge_lines or index > last_index - edge_lines
            if at_edge and _normalize_line_for_counting(line) in boilerplate:
                continue
            keep.append(line)
        lines = keep

    # Collapse runs of spaces/tabs and drop trailing whitespace per line.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    text = "\n".join(lines)
    # Three or more newlines collapse to a paragraph break; single breaks stay.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ingest_pdf(pdf_path: Path, settings: dict[str, Any], ingested_at: str) -> list[PageRecord]:
    """Parse and clean a single PDF into page records.

    Args:
        pdf_path: Path to the PDF file.
        settings: The `ingest` section of the resolved configuration.
        ingested_at: UTC ISO-8601 timestamp stamped onto every record.

    Returns:
        One `PageRecord` per page that survived the minimum-length filter.

    Raises:
        RuntimeError: If the PDF is password protected.
    """
    document = pymupdf.open(pdf_path)
    try:
        if document.needs_pass:
            raise RuntimeError(
                f"{pdf_path.name} is password protected. Decrypt it before ingesting."
            )

        extraction_mode = settings["extraction_mode"]
        raw_pages = extract_page_texts(document, extraction_mode)
    finally:
        document.close()

    boilerplate: set[str] = set()
    if settings["strip_repeated_lines"]:
        boilerplate = find_boilerplate_lines(
            raw_pages,
            threshold=settings["repeated_line_threshold"],
            edge_lines=settings["repeated_line_edge_lines"],
        )
        LOGGER.info("%s: %d boilerplate line(s) detected", pdf_path.name, len(boilerplate))

    doc_id = compute_doc_id(pdf_path)
    insurer, policy_type, doc_label = parse_filename_metadata(pdf_path)

    records: list[PageRecord] = []
    skipped_pages: list[int] = []
    for page_index, raw_text in enumerate(raw_pages):
        cleaned = clean_page_text(
            raw_text,
            boilerplate=boilerplate,
            edge_lines=settings["repeated_line_edge_lines"],
            dehyphenate=settings["dehyphenate"],
            normalize_unicode=settings["normalize_unicode"],
        )
        page_number = page_index + 1
        if len(cleaned) < settings["min_chars_per_page"]:
            skipped_pages.append(page_number)
            continue

        records.append(
            PageRecord(
                doc_id=doc_id,
                filename=pdf_path.name,
                source_path=pdf_path.as_posix(),
                insurer=insurer,
                policy_type=policy_type,
                doc_label=doc_label,
                page=page_number,
                text=cleaned,
                char_count=len(cleaned),
                extraction_mode=extraction_mode,
                ingested_at=ingested_at,
            )
        )

    if skipped_pages:
        # Logged, never silent: a large skip count usually means a scanned PDF
        # that needs OCR, not a document full of blank pages.
        LOGGER.info(
            "%s: skipped %d page(s) under %d chars (pages: %s)",
            pdf_path.name,
            len(skipped_pages),
            settings["min_chars_per_page"],
            ", ".join(str(page) for page in skipped_pages[:20])
            + (" ..." if len(skipped_pages) > 20 else ""),
        )
    return records


def discover_pdfs(input_path: Path) -> list[Path]:
    """List the PDFs to ingest from a file or directory path.

    Args:
        input_path: A single `.pdf` file, or a directory searched recursively.

    Returns:
        Sorted list of PDF paths, so runs are deterministic.

    Raises:
        FileNotFoundError: If the path does not exist or contains no PDFs.
    """
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path.resolve()}")

    pdfs = sorted(path for path in input_path.rglob("*.pdf") if path.is_file())
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found under {input_path.resolve()}. "
            "Place policy PDFs there named 'insurer__policytype__label.pdf'."
        )
    return pdfs


def write_jsonl(records: list[PageRecord], output_path: Path) -> None:
    """Write page records to a JSONL file, one JSON object per line.

    JSONL rather than a single JSON array so the file streams line-by-line in
    later steps without loading the whole corpus into memory.

    Args:
        records: Page records to write.
        output_path: Destination path; parent directories are created.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser. All defaults are `None` so that
        `config.yaml` supplies values unless explicitly overridden.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase1_rag.ingest",
        description="Parse policy PDFs into cleaned, page-scoped JSONL records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml."
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="PDF file or directory. Default: paths.raw_dir."
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output JSONL path. Default: derived from config."
    )
    parser.add_argument(
        "--mode", choices=["text", "blocks"], default=None, help="Override ingest.extraction_mode."
    )
    parser.add_argument(
        "--min-chars", type=int, default=None, help="Override ingest.min_chars_per_page."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Ingest only the first N PDFs (smoke test)."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def resolve_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge config file values with CLI overrides.

    CLI arguments win when provided. The merged result is what gets logged with
    the run, so a recorded metric can always be reproduced.

    Args:
        config: Parsed `config.yaml`.
        args: Parsed command-line arguments.

    Returns:
        The fully resolved `ingest` settings.
    """
    settings = {
        "extraction_mode": args.mode or cfg_get(config, "ingest.extraction_mode", "text"),
        "min_chars_per_page": (
            args.min_chars
            if args.min_chars is not None
            else cfg_get(config, "ingest.min_chars_per_page", 40)
        ),
        "strip_repeated_lines": cfg_get(config, "ingest.strip_repeated_lines", True),
        "repeated_line_threshold": cfg_get(config, "ingest.repeated_line_threshold", 0.6),
        "repeated_line_edge_lines": cfg_get(config, "ingest.repeated_line_edge_lines", 3),
        "dehyphenate": cfg_get(config, "ingest.dehyphenate", True),
        "normalize_unicode": cfg_get(config, "ingest.normalize_unicode", True),
    }
    return settings


def main(argv: list[str] | None = None) -> int:
    """Run ingestion end to end.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on full success, 1 if any document failed.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    settings = resolve_settings(config, args)

    input_path = args.input or Path(cfg_get(config, "paths.raw_dir", "data/raw"))
    output_path = args.output or (
        Path(cfg_get(config, "paths.processed_dir", "data/processed"))
        / cfg_get(config, "ingest.output_filename", "pages.jsonl")
    )

    pdfs = discover_pdfs(input_path)
    if args.limit is not None:
        pdfs = pdfs[: args.limit]

    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LOGGER.info("Ingesting %d PDF(s) from %s (mode=%s)", len(pdfs), input_path, settings["extraction_mode"])

    all_records: list[PageRecord] = []
    per_document: list[dict[str, Any]] = []
    failures: list[str] = []

    for pdf_path in tqdm(pdfs, desc="Ingesting", unit="pdf"):
        try:
            records = ingest_pdf(pdf_path, settings, ingested_at)
        except Exception as error:  # noqa: BLE001 — report every failure, then exit non-zero
            LOGGER.error("FAILED %s: %s", pdf_path.name, error)
            failures.append(pdf_path.name)
            continue

        all_records.extend(records)
        per_document.append(
            {
                "filename": pdf_path.name,
                "doc_id": records[0].doc_id if records else None,
                "insurer": records[0].insurer if records else UNKNOWN,
                "policy_type": records[0].policy_type if records else UNKNOWN,
                "pages_kept": len(records),
                "chars": sum(record.char_count for record in records),
            }
        )

    write_jsonl(all_records, output_path)

    total_chars = sum(record.char_count for record in all_records)
    mean_chars = round(total_chars / len(all_records)) if all_records else 0
    summary = {
        "ingested_at": ingested_at,
        "input_path": input_path.as_posix(),
        "output_path": output_path.as_posix(),
        "settings": settings,
        "documents_ingested": len(per_document),
        "documents_failed": failures,
        "pages_kept": len(all_records),
        "total_chars": total_chars,
        "mean_chars_per_page": mean_chars,
        "per_document": per_document,
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== INGEST SUMMARY ===")
    print(f"documents ingested : {len(per_document)}")
    print(f"pages kept         : {len(all_records)}")
    print(f"mean chars/page    : {mean_chars}")
    print(f"output             : {output_path}")
    print(f"run metadata       : {meta_path}")
    for entry in per_document:
        print(
            f"  - {entry['filename']}: {entry['pages_kept']} pages, "
            f"{entry['chars']} chars, insurer={entry['insurer']}"
        )

    if failures:
        print(f"\nFAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    if not all_records:
        LOGGER.error("No pages were ingested. Check that the PDFs contain extractable text.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
