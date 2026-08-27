"""Verify the citation integrity of a generated RAFT slice. No models, no cost.

**Why this is a script and not an inline command.** The check it performs is the
one that caught the calc-v1 defect — 3 of 5 rows citing a page that belonged to a
different insurer's policy — and a check that important should be re-runnable,
reviewable, and quotable in METRICS, not retyped as a shell one-liner every time.

**What "correct" means for a RAFT row.** The answer cites a page. That page must
belong to the *document the question is about*, and a chunk from that document
and page must actually be present in the row's context. Page numbers alone are
not enough: across ten documents they collide constantly, and a row citing p.26
of one insurer while carrying p.26 of another teaches the exact cross-document
hallucination this dataset exists to train against.

Exit code is 1 if any hard check fails, so a bad slice cannot pass unnoticed.

Usage:
    python -m phase4_finetune.audit_calc_rows --help
    python -m phase4_finetune.audit_calc_rows
    python -m phase4_finetune.audit_calc_rows --path data/train/raft_X_calc-v3.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_PATTERN = "data/train/raft_*_calc-*.jsonl"


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL slice into memory.

    Args:
        path: The .jsonl file to read.

    Returns:
        One dict per row, in file order.

    Raises:
        FileNotFoundError: If the file does not exist — a silent empty result
            would read as "all checks passed", which is the worst failure here.
    """
    if not path.exists():
        raise FileNotFoundError(f"No such slice: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_doc_id(row: dict[str, Any]) -> str:
    """Recover the document a row was generated from.

    Chunk ids are `{doc_id}_p{page}_c{n}` (or `..._P{parent}_c{n}` under the
    parent-child strategy), so the document is the leading field. Read from the
    id rather than stored separately because that is what the generator wrote.

    Args:
        row: One RAFT row.

    Returns:
        The source document id, or "" if the row carries no source chunk.
    """
    return str(row.get("source_chunk_id", "")).split("_")[0]


def oracle_rank(row: dict[str, Any]) -> int | None:
    """Find the 1-indexed position of the oracle context.

    Args:
        row: One RAFT row.

    Returns:
        The rank, or None if no context is flagged as the oracle.
    """
    for rank, context in enumerate(row.get("contexts", []), start=1):
        if context.get("is_oracle"):
            return rank
    return None


def cites_own_document(row: dict[str, Any]) -> bool:
    """Check the cited page belongs to the row's own source document.

    The rule the generator enforces at write time, re-checked here independently
    so a bug in the generator cannot also hide itself in the audit.

    Args:
        row: One RAFT row.

    Returns:
        True if some context matches both the source document and a cited page.
    """
    doc_id = source_doc_id(row)
    cited = set(row.get("cited_pages") or [])
    if not cited:
        return False
    return any(
        str(context.get("doc_id", "")) == doc_id and context.get("page") in cited
        for context in row.get("contexts", [])
    )


def audit(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Run every check and describe the result.

    Args:
        rows: The slice to audit.

    Returns:
        `(passed, report_lines)`. `passed` is False if any hard check failed.
    """
    lines: list[str] = []
    passed = True

    cross_doc = [row for row in rows if not cites_own_document(row)]
    no_oracle = [row for row in rows if oracle_rank(row) is None]

    lines.append(f"rows                        : {len(rows)}")
    lines.append(f"cross-document citations    : {len(cross_doc)}  (must be 0)")
    lines.append(f"rows with no oracle flagged : {len(no_oracle)}  (must be 0)")

    if cross_doc:
        passed = False
        lines.append("")
        lines.append("  FAIL — first offenders:")
        for row in cross_doc[:5]:
            lines.append(
                f"    {row.get('example_id')}: source doc {source_doc_id(row)}, "
                f"cited {row.get('cited_pages')}, "
                f"contexts {[(c.get('doc_id'), c.get('page')) for c in row.get('contexts', [])]}"
            )
    if no_oracle:
        passed = False
        lines.append("")
        lines.append("  FAIL — rows with no oracle context:")
        for row in no_oracle[:5]:
            lines.append(f"    {row.get('example_id')}")

    ranks = Counter(rank for row in rows if (rank := oracle_rank(row)) is not None)
    lines.append("")
    lines.append("oracle rank distribution    : " + (
        ", ".join(f"rank {rank}: {count}" for rank, count in sorted(ranks.items())) or "none"
    ))
    # An oracle pinned at rank 1 teaches position instead of reading. This is a
    # warning, not a failure: a small slice can land lopsided by chance.
    if len(ranks) == 1 and rows:
        lines.append("  WARNING — every oracle sits at the same rank.")

    insurers = Counter(
        context.get("insurer", "unknown")
        for row in rows
        for context in row.get("contexts", [])
        if context.get("is_oracle")
    )
    lines.append("oracle insurers             : " + (
        ", ".join(f"{name}: {count}" for name, count in insurers.most_common()) or "none"
    ))

    slices = Counter(row.get("slice_name", "unknown") for row in rows)
    lines.append("slices                      : " + ", ".join(
        f"{name}: {count}" for name, count in slices.most_common()
    ))

    return passed, lines


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase4_finetune.audit_calc_rows",
        description="Verify citation integrity of a RAFT slice. No models, no cost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Slice to audit. Defaults to the newest file matching --pattern.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob used to find the newest slice when --path is omitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Audit one slice and report.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 if every hard check passed, 1 otherwise.
    """
    args = build_parser().parse_args(argv)

    path = args.path
    if path is None:
        matches = sorted(Path().glob(args.pattern))
        if not matches:
            print(f"No slice matched {args.pattern!r}. Pass --path explicitly.")
            return 1
        path = matches[-1]

    rows = load_rows(path)
    passed, lines = audit(rows)

    print(f"=== CITATION AUDIT: {path} ===")
    for line in lines:
        print(line)
    print()
    print("RESULT: " + ("PASS — slice is trainable" if passed else "FAIL — do not train on this"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
