"""Merge the RAFT slices into a training-ready chat dataset. No models, no cost.

**Why the prompt is rendered here and not on Modal.** The training text must be
byte-identical to what the served pipeline builds, or the model learns a format
it will never be shown at inference. The obvious way to guarantee that is to call
the serving pipeline's own `build_prompt`. But importing `phase1_rag` inside the
Modal training image would drag qdrant-client, sentence-transformers and the
whole retrieval stack into a container that only needs to read text. So the
rendering happens locally, where those imports are already paid for, and Modal
receives finished `messages` rows and needs no ClaimWise code at all.

**Why `--inputs` is required rather than globbed.** `data/train/` still holds
calc-v1 and calc-v2, both of which failed the citation audit. A glob would
silently sweep them into the training set. Naming the files is the whole
safeguard.

**Why the split is stratified by slice.** The set is deliberately unbalanced —
mostly lookup, with smaller negative, over-refusal and calculation slices. A
random split would hand the validation set an arbitrary slice mix, so a
validation loss could move because the mix shifted rather than because the model
changed. Stratifying keeps train and validation comparable.

**Contamination, checked twice.** Every row is re-fingerprinted against the
golden and agent-task sets here, even though both generators already did it. The
check is nearly free and the failure it prevents — training on the questions the
Phase 4 benchmark is scored with — is unrecoverable and invisible afterwards.

Usage:
    python -m phase4_finetune.build_train_split --help
    python -m phase4_finetune.build_train_split \\
        --inputs data/train/raft_A_v1.jsonl data/train/raft_B_calc-v3.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from phase1_rag.build_eval_set import question_fingerprint
from phase1_rag.rag_chain import SYSTEM_PROMPT, build_prompt
from phase4_finetune.gen_dataset import load_holdout_fingerprints

DEFAULT_OUTPUT_DIR = Path("data/train")


class _PromptChunk:
    """Adapt a stored context dict to what `format_passages` reads.

    The generators persist `filename` but not `doc_label` or `policy_type`, both
    of which appear in the passage header. They are recovered from the filename,
    which `ingest.py` writes as `{insurer}__{policy_type}__{doc_label}.pdf` — a
    derivation, not a guess, so the header matches what serving would render.
    """

    def __init__(self, context: dict[str, Any]) -> None:
        """Expose the passage-header fields as attributes.

        Args:
            context: One stored context record from a RAFT row.
        """
        stem = str(context.get("filename", "")).removesuffix(".pdf")
        parts = stem.split("__")
        self.insurer = str(context.get("insurer", "")) or (parts[0] if parts else "")
        self.policy_type = parts[1] if len(parts) > 1 else ""
        self.doc_label = parts[2] if len(parts) > 2 else ""
        self.page = int(context.get("page", 0) or 0)
        self.text = str(context.get("text", ""))


def to_messages(row: dict[str, Any], system: str) -> dict[str, Any]:
    """Render one RAFT row as a chat-format training example.

    Args:
        row: A RAFT row carrying `question`, `contexts` and `answer`.
        system: The rendered system prompt, refusal text already substituted.

    Returns:
        A `{"messages": [...]}` record, the format TRL and Unsloth consume via
        the tokenizer's chat template.
    """
    chunks = [_PromptChunk(context) for context in row.get("contexts", [])]
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": build_prompt(row["question"], chunks)},
            {"role": "assistant", "content": row["answer"]},
        ],
        "slice_name": row.get("slice_name", "unknown"),
        "example_id": row.get("example_id", ""),
    }


def load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read every named slice, reporting what came from where.

    Args:
        paths: The slice files to merge, in order.

    Returns:
        `(rows, per_file_counts)`.

    Raises:
        FileNotFoundError: If any named slice is missing. A missing input must
            never be skipped silently — the split would look successful and be
            short a whole slice.
    """
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"No such slice: {path}")
        with path.open(encoding="utf-8") as handle:
            loaded = [json.loads(line) for line in handle if line.strip()]
        counts[path.name] = len(loaded)
        rows.extend(loaded)
    return rows, counts


def dedupe_and_screen(
    rows: list[dict[str, Any]],
    holdout: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop duplicates, holdout collisions and structurally unusable rows.

    Args:
        rows: Merged rows from every slice.
        holdout: Fingerprints that must never be trained on.

    Returns:
        `(kept, stats)`.
    """
    stats = {"duplicate": 0, "holdout": 0, "empty_context": 0, "empty_answer": 0}
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []

    for row in rows:
        if not row.get("contexts"):
            stats["empty_context"] += 1
            continue
        if not str(row.get("answer", "")).strip():
            stats["empty_answer"] += 1
            continue

        fingerprint = row.get("fingerprint") or question_fingerprint(row.get("question", ""))
        if fingerprint in holdout:
            stats["holdout"] += 1
            continue
        if fingerprint in seen:
            stats["duplicate"] += 1
            continue

        seen.add(fingerprint)
        kept.append(row)

    return kept, stats


def stratified_split(
    rows: list[dict[str, Any]],
    val_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into train and validation, preserving the slice mix in both.

    Args:
        rows: Screened rows.
        val_fraction: Share of each slice to hold out for validation.
        rng: Seeded RNG, so the split reproduces exactly.

    Returns:
        `(train, val)`.
    """
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_slice[row.get("slice_name", "unknown")].append(row)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for slice_name in sorted(by_slice):
        bucket = sorted(by_slice[slice_name], key=lambda item: item.get("example_id", ""))
        rng.shuffle(bucket)
        # At least one validation row per slice, so every slice is actually
        # measured — but never the whole slice, which would leave nothing to
        # train on for a small one.
        take = min(max(1, round(len(bucket) * val_fraction)), max(len(bucket) - 1, 0))
        val.extend(bucket[:take])
        train.extend(bucket[take:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def compose_report(name: str, rows: list[dict[str, Any]]) -> str:
    """Describe a split's slice mix on one line.

    Args:
        name: Split name, for the label.
        rows: The split's rows.

    Returns:
        A printable summary line.
    """
    counts = Counter(row.get("slice_name", "unknown") for row in rows)
    mix = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
    return f"{name:6}: {len(rows):5}  ({mix})"


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase4_finetune.build_train_split",
        description="Merge RAFT slices into train/val chat datasets. No models, no cost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Slice files to merge. Named explicitly so a failed slice cannot be swept in.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="Share of each slice held out."
    )
    parser.add_argument("--tag", default="sft-v1", help="Label in the output filenames.")
    parser.add_argument("--seed", type=int, default=3407)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Merge, screen, split and write.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    rng = random.Random(args.seed)

    system = SYSTEM_PROMPT.format(
        refusal_text=cfg_get(
            config,
            "rag.refusal_text",
            "That isn't covered in the policy documents you've uploaded.",
        )
    )

    rows, per_file = load_rows(args.inputs)
    print("=== INPUTS ===")
    for name, count in per_file.items():
        print(f"  {name}: {count}")
    print(f"  merged: {len(rows)}\n")

    kept, stats = dedupe_and_screen(rows, load_holdout_fingerprints())
    print("=== SCREENING ===")
    for key, value in stats.items():
        print(f"  dropped_{key:14}: {value}")
    print(f"  kept{'':16}: {len(kept)}\n")

    if stats["holdout"]:
        print("REFUSING TO WRITE: rows collided with the golden/agent holdout.")
        print("Training on them would make the Phase 4 benchmark meaningless.")
        return 1

    train, val = stratified_split(kept, args.val_fraction, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    written: dict[str, str] = {}
    for name, split in (("train", train), ("val", val)):
        path = args.output_dir / f"sft_{stamp}_{args.tag}_{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in split:
                handle.write(json.dumps(to_messages(row, system), ensure_ascii=False) + "\n")
        written[name] = str(path)

    print("=== SPLIT ===")
    print(compose_report("train", train))
    print(compose_report("val", val))

    # A character count is not a token count, but it is the cheapest early
    # warning that a row will blow the training sequence length.
    longest = max(
        (len(build_prompt(row["question"], [_PromptChunk(c) for c in row["contexts"]])) for row in kept),
        default=0,
    )
    print(f"\nlongest user message : {longest} chars (~{longest // 4} tokens, rough)")

    meta_path = args.output_dir / f"sft_{stamp}_{args.tag}.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "inputs": {name: count for name, count in per_file.items()},
                "screening": stats,
                "seed": args.seed,
                "val_fraction": args.val_fraction,
                "train_rows": len(train),
                "val_rows": len(val),
                "train_mix": dict(Counter(r.get("slice_name") for r in train)),
                "val_mix": dict(Counter(r.get("slice_name") for r in val)),
                "longest_user_message_chars": longest,
                "outputs": written,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote: {written['train']}")
    print(f"wrote: {written['val']}")
    print(f"meta : {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
