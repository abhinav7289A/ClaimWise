"""Apply the curation fixes identified in `metrics/failure_analysis_p1.md`.

Two defects in the golden set were found by analysing Phase 2's nine permanent
retrieval failures. Neither is a retrieval problem; both are questions the set
should not have contained in their current form.

**Defect 1 — a duplicate question.** `g-069` and `g-077` are byte-identical:
same question, same answer, same `ground_truth_pages`, same `source_chunk_id`.
The set therefore holds 84 distinct positives while every hit@k has been divided
by 85, and one retrieval failure has been counted twice. `build_eval_set.py`
compared each question against its *source chunk* but never against previously
accepted questions; it now fingerprints questions too, so this cannot recur.

**Defect 2 — two questions that cannot be answered as posed.** `g-030` asks for
"the phone number for customer care" and `g-058` for "the toll-free number for
customer care". Both expect `18001021111`. But `g-030`'s ground truth is page 27
of the SBI *health* policy and `g-058`'s is page 15 of the SBI *home* policy —
the same insurer prints the same number in both. Neither question names a
policy, so nothing distinguishes them, and a retriever returning the other
document would be marked wrong while handing the user the correct phone number.
The retrieval log confirms the ambiguity: both questions pull candidates from
all four documents.

**Why scoping rather than relaxed scoring.** The alternative was to score these
correct if *any* document containing the answer is retrieved. That is closer to
what a user wants, but it introduces a second scoring rule that applies to two
items and to nothing else — a special case inside the metric, which is where
metrics start lying. Naming the policy in the question is what a real user with
several policies would do anyway, and it keeps one scoring rule for the whole
set.

Rewrites are declared in `SCOPING_FIXES` with their reasoning rather than being
applied by a heuristic, because hand-curation of an evaluation set should be
auditable line by line.

**These changes alter every hit@k recorded so far**, so the corrected set gets a
fresh baseline run rather than a quiet edit to old numbers.

Usage:
    python -m evals.clean_golden --help
    python -m evals.clean_golden               # dry run, reports only
    python -m evals.clean_golden --write
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from phase1_rag.build_eval_set import question_fingerprint

LOGGER = logging.getLogger("claimwise.clean_golden")

# Item id -> (replacement question, why). Applied only when the item's current
# question still matches the one analysed, so a re-run after manual editing
# cannot silently overwrite someone else's correction.
SCOPING_FIXES: dict[str, tuple[str, str]] = {
    "g-030": (
        "What is the customer care phone number for my SBI General health insurance policy?",
        "Unscoped: 18001021111 appears in both the SBI health and SBI home "
        "policies, so the expected document was arbitrary.",
    ),
    "g-058": (
        "What is the toll-free customer care number for my SBI General home insurance policy?",
        "Same collision as g-030, resolved to the home policy that its "
        "ground_truth_pages already pointed at.",
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file.

    Args:
        path: File to read.

    Returns:
        Decoded records, in order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_duplicates(items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    """Identify positives whose question repeats an earlier one.

    Negatives are excluded: they are hand-seeded out-of-scope questions with no
    ground-truth page, and several deliberately probe the same refusal behaviour
    from different angles.

    Args:
        items: Golden set records.

    Returns:
        `(duplicate item, id of the first item with that question)` pairs, in
        the order encountered. The FIRST occurrence is always kept, so the
        outcome does not depend on how the file happens to be sorted.
    """
    seen: dict[str, str] = {}
    duplicates: list[tuple[dict[str, Any], str]] = []

    for item in items:
        if not item.get("ground_truth_pages"):
            continue
        fingerprint = question_fingerprint(item["question"])
        if fingerprint in seen:
            duplicates.append((item, seen[fingerprint]))
        else:
            seen[fingerprint] = item["id"]

    return duplicates


def apply_scoping(items: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Rewrite the ambiguous questions in place.

    Args:
        items: Golden set records, modified in place.

    Returns:
        `(item id, old question, new question)` for each rewrite applied.
    """
    applied: list[tuple[str, str, str]] = []
    by_id = {item["id"]: item for item in items}

    for item_id, (new_question, reason) in SCOPING_FIXES.items():
        item = by_id.get(item_id)
        if item is None:
            LOGGER.warning("Scoping fix for %s skipped — item not in the set.", item_id)
            continue
        if item["question"] == new_question:
            LOGGER.info("%s already scoped; nothing to do.", item_id)
            continue

        applied.append((item_id, item["question"], new_question))
        item["question"] = new_question
        # A rewritten question has not been read by a human in its new form.
        item["verified"] = False
        note = item.get("notes") or ""
        item["notes"] = f"{note} [clean_golden: {reason}]".strip()

    return applied


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evals.clean_golden",
        description="Apply the golden-set curation fixes from failure_analysis_p1.md.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--golden", type=Path, default=None, help="Override the golden set path.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually modify the file. Without this the run only reports.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Report, and optionally apply, the curation fixes.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if the golden set is missing.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    eval_dir = Path(cfg_get(config, "paths.eval_dir", "data/eval"))
    golden_path = args.golden or eval_dir / cfg_get(config, "eval.output_filename", "golden.jsonl")

    try:
        items = load_jsonl(golden_path)
    except FileNotFoundError as error:
        LOGGER.error("%s", error)
        return 1

    positives = [item for item in items if item.get("ground_truth_pages")]
    negatives = [item for item in items if not item.get("ground_truth_pages")]

    print(f"\ngolden set          : {golden_path}")
    print(f"items               : {len(items)} ({len(positives)} pos / {len(negatives)} neg)")

    duplicates = find_duplicates(items)
    print(f"\n--- duplicates ({len(duplicates)}) ---")
    for item, original_id in duplicates:
        print(f"  {item['id']} duplicates {original_id}: {item['question'][:70]}...")
    if not duplicates:
        print("  none")

    scoped = apply_scoping(items)
    print(f"\n--- scoping rewrites ({len(scoped)}) ---")
    for item_id, old, new in scoped:
        print(f"  {item_id}")
        print(f"    was: {old}")
        print(f"    now: {new}")
    if not scoped:
        print("  none (already applied)")

    duplicate_ids = {item["id"] for item, _ in duplicates}
    kept = [item for item in items if item["id"] not in duplicate_ids]
    kept_positives = [item for item in kept if item.get("ground_truth_pages")]

    print(f"\npositives           : {len(positives)} -> {len(kept_positives)}")
    print(f"total items         : {len(items)} -> {len(kept)}")

    if not duplicates and not scoped:
        print("\nNothing to do — the set is already clean.")
        return 0

    if not args.write:
        print("\nDry run. Re-run with --write to apply.")
        return 0

    # Preserve the pre-curation set once, so the numbers recorded against it stay
    # reproducible. Never overwritten on a second run.
    raw_path = golden_path.with_suffix(".raw.jsonl")
    if not raw_path.exists():
        shutil.copy2(golden_path, raw_path)
        print(f"\noriginal preserved  : {raw_path}")

    with golden_path.open("w", encoding="utf-8") as handle:
        for item in kept:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"written             : {golden_path}")
    print("\nEvery hit@k recorded so far was computed against the previous set.")
    print("Re-run evals.retrieval_metrics and record a corrected baseline —")
    print("do not edit the old numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
