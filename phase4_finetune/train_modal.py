"""LoRA fine-tune of Qwen3.5-4B on Modal, using plain transformers.Trainer.

**bf16 LoRA, not QLoRA (D-31).** CLAUDE.md's written plan says QLoRA. Unsloth's
own Qwen3.5 documentation advises against 4-bit training on these models, and the
compat probe confirmed bf16 LoRA works at 9.08 GB peak on an L4. Plan overridden
by evidence; this comment is the record.

**Why not TRL's SFTTrainer (P-21).** Four launches died inside it. The last one
proved the cause: the SFTConfig was correct — the log showed
`config eos/pad -> '<|im_end|>' / '<|vision_pad|>'` — and TRL still raised
`eos_token ('<EOS_TOKEN>') is not found in the vocabulary`. That placeholder is
injected inside Unsloth's patched `SFTTrainer.__init__`, downstream of anything
the caller can set, and it does not resolve against Qwen3.5's `TokenizersBackend`
processor. So TRL is out of the path entirely. `transformers.Trainer` is a stable,
unpatched API, and the one thing SFTTrainer was giving us for free — masking loss
to the assistant turn — is twenty lines we write ourselves and can actually verify.

**Why the masking is done by length, not by string search.** `train_on_responses_only`
finds turn boundaries by matching template markers, which breaks silently when a
template changes. Here the prompt is rendered twice — once without the answer, once
with — and everything up to the prompt's token length becomes -100. No markers, no
matching, and the run prints how many label positions survived so a masking bug is
visible in the first ten lines rather than after an hour of training.

**Why the vision tower stays frozen.** The dataset is text-only. LoRA on the
language side leaves image understanding exactly as the base model shipped it,
which keeps bill and claim-form photos working in Phase 5 for free.

**Cost control.** Nothing launches without `--yes`. `--smoke` runs ten steps so the
full-run estimate rests on a measured seconds-per-step. Checkpoints live on a
Volume and the run resumes from the newest, so an interruption costs minutes.

Usage:
    modal run phase4_finetune/train_modal.py                 # estimate only
    modal run phase4_finetune/train_modal.py --smoke --yes
    modal run phase4_finetune/train_modal.py --yes --push-repo you/claimwise-qwen35-4b
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "claimwise-train"

# The model the compat probe returned GO on.
MODEL_NAME = "unsloth/Qwen3.5-4B"

# L4 (24 GB). The probe measured 9.08 GB peak at seq 2048, almost all weights
# (4.57B x 2 bytes). Adapter optimiser states add ~0.4 GB and activations at
# seq 4096 with gradient checkpointing are modest — expect 13-16 GB.
GPU = "L4"

TIMEOUT_SECONDS = 5400

# Modal's published L4 rate at the time of writing. VERIFY at modal.com/pricing
# and override with --gpu-rate if it moved; a stale rate makes the estimate a lie.
L4_USD_PER_HOUR = 0.80

# Hard rail. CLAUDE.md caps Modal at $30 and Phase 4.5 reserves $5 that is never
# spent, so a single run over this is a config bug, not an expensive experiment.
MAX_RUN_USD = 5.00

REMOTE_DATA_DIR = "/data"
CHECKPOINT_DIR = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Do NOT pin torch (P-20): pinning it against transformers cost two false
    # FALLBACK verdicts. Unsloth declares a mutually-consistent stack.
    .pip_install("unsloth", "unsloth_zoo", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("data/train", REMOTE_DATA_DIR)
)

hf_cache = modal.Volume.from_name("claimwise-hf-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name("claimwise-checkpoints", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache, CHECKPOINT_DIR: checkpoints},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def train(
    train_file: str,
    val_file: str,
    run_name: str,
    epochs: float,
    max_steps: int,
    seq_len: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    push_repo: str,
    seed: int,
) -> dict[str, Any]:
    """Fine-tune the adapter and, unless smoke-testing, push the merged model.

    Args:
        train_file: Basename of the training JSONL in the mounted data dir.
        val_file: Basename of the validation JSONL.
        run_name: Checkpoint subdirectory; reuse to resume.
        epochs: Epochs to train; ignored when `max_steps` is positive.
        max_steps: Hard step cap. Positive means smoke mode — no push.
        seq_len: Max sequence length.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        learning_rate: Peak LR.
        push_repo: HF repo id for the merged model. Empty means do not push.
        seed: Seeded everywhere.

    Returns:
        A record of the run: losses, timings, versions, and where it wrote.
    """
    import os
    import time

    import torch
    import transformers
    from datasets import load_dataset
    from transformers import Trainer, TrainingArguments
    from unsloth import FastVisionModel

    started = time.perf_counter()
    smoke = max_steps > 0
    out_dir = os.path.join(CHECKPOINT_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    transformers.set_seed(seed)

    # --- Model ------------------------------------------------------------
    #
    # Identical to the probe's GO path. FastVisionModel.from_pretrained returns
    # a PROCESSOR, not a tokenizer; the tokenizer hangs off it.
    model, processor = FastVisionModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=seq_len,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    tokenizer = getattr(processor, "tokenizer", processor)

    model = FastVisionModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vision_trainable = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and ("visual" in name or "vision" in name)
    )
    print(f"trainable params: {trainable:,}  (vision: {vision_trainable:,}, must be 0)")
    if vision_trainable:
        raise RuntimeError(
            f"Vision tower is trainable ({vision_trainable:,} params). Text-only "
            "fine-tuning would drift the image path — refusing to train."
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    print(f"eos={tokenizer.eos_token!r} ({tokenizer.eos_token_id})  pad_id={pad_id}")

    # --- Data -------------------------------------------------------------
    data_files = {
        "train": os.path.join(REMOTE_DATA_DIR, train_file),
        "val": os.path.join(REMOTE_DATA_DIR, val_file),
    }
    raw = load_dataset("json", data_files=data_files)
    truncated = {"n": 0}

    def encode(row: dict[str, Any]) -> dict[str, list[int]]:
        """Tokenise one row and mask the prompt out of the labels.

        The prompt is rendered twice — without the assistant turn to measure how
        long it is, then with it for the real sequence. Everything before the
        answer becomes -100, so loss is computed on the answer alone.

        Args:
            row: A record carrying `messages`.

        Returns:
            `input_ids` and `labels` of equal length.
        """
        messages = row["messages"]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if len(input_ids) > seq_len:
            truncated["n"] += 1
            input_ids = input_ids[:seq_len]

        boundary = min(len(prompt_ids), len(input_ids))
        labels = [-100] * boundary + input_ids[boundary:]
        return {"input_ids": input_ids, "labels": labels}

    dataset = raw.map(encode, remove_columns=raw["train"].column_names)
    print(f"train rows: {len(dataset['train'])}, val rows: {len(dataset['val'])}")
    print(f"rows truncated at seq_len={seq_len}: {truncated['n']}")

    # A masking bug is invisible until the model is useless, so prove it now: the
    # first row must have some masked positions and some unmasked ones.
    sample = dataset["train"][0]
    supervised = sum(1 for label in sample["labels"] if label != -100)
    print(
        f"masking check — row 0: {len(sample['labels'])} tokens, "
        f"{supervised} supervised, {len(sample['labels']) - supervised} masked"
    )
    if supervised == 0 or supervised == len(sample["labels"]):
        raise RuntimeError(
            "Label masking failed: every token is masked or none are. Training "
            "would teach the wrong thing — refusing to continue."
        )

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad a batch to its longest sequence.

        Args:
            features: Encoded rows.

        Returns:
            Batched tensors: input_ids, attention_mask, labels.
        """
        longest = max(len(item["input_ids"]) for item in features)
        input_ids, attention, labels = [], [], []
        for item in features:
            ids = list(item["input_ids"])
            lab = list(item["labels"])
            padding = longest - len(ids)
            input_ids.append(ids + [pad_id] * padding)
            attention.append([1] * len(ids) + [0] * padding)
            # -100 on the padding too, or the model is trained to emit pad.
            labels.append(lab + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # --- Trainer ----------------------------------------------------------
    #
    # transformers.Trainer, not TRL. TrainingArguments is an unpatched dataclass,
    # so the field names are whatever this transformers version really declares —
    # the one rename that matters is evaluation_strategy -> eval_strategy.
    import dataclasses

    fields = {field.name for field in dataclasses.fields(TrainingArguments)}
    total_steps = (
        max_steps
        if smoke
        else max(round(len(dataset["train"]) * epochs / max(batch_size * grad_accum, 1)), 1)
    )

    args_kwargs: dict[str, Any] = {
        "output_dir": out_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": 1 if smoke else epochs,
        "max_steps": max_steps if smoke else -1,
        "learning_rate": learning_rate,
        "warmup_steps": max(5, round(total_steps * 0.05)),
        "lr_scheduler_type": "linear",
        "logging_steps": 5,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "bf16": True,
        "fp16": False,
        "seed": seed,
        "report_to": "none",
        "save_strategy": "steps",
        "save_steps": 25,
        "save_total_limit": 2,
        "eval_steps": 50,
        "remove_unused_columns": False,
    }
    strategy = "no" if smoke else "steps"
    if "eval_strategy" in fields:
        args_kwargs["eval_strategy"] = strategy
    elif "evaluation_strategy" in fields:
        args_kwargs["evaluation_strategy"] = strategy

    unknown = [key for key in args_kwargs if key not in fields]
    for key in unknown:
        print(f"WARNING: TrainingArguments has no {key!r} — dropping it.")
        args_kwargs.pop(key)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**args_kwargs),
        train_dataset=dataset["train"],
        eval_dataset=None if smoke else dataset["val"],
        data_collator=collate,
    )

    resume = os.path.isdir(out_dir) and any(
        name.startswith("checkpoint-") for name in os.listdir(out_dir)
    )
    print(f"resuming from checkpoint: {resume}")

    stats = trainer.train(resume_from_checkpoint=resume)
    checkpoints.commit()

    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "model": MODEL_NAME,
        "run_name": run_name,
        "smoke": smoke,
        # Nothing in the image is pinned, so the resolved stack is part of the
        # run's identity — without it "reproduce this run" has no target.
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "trainable_params": trainable,
        "train_rows": len(dataset["train"]),
        "val_rows": len(dataset["val"]),
        "rows_truncated": truncated["n"],
        "steps": int(stats.global_step),
        "train_loss": round(float(stats.training_loss), 4),
        # `elapsed` starts at function entry, so it includes the multi-minute
        # model load. Dividing that by steps overstates per-step time by 3x and
        # would inflate every downstream cost estimate. The Trainer's own
        # train_runtime covers the training loop alone, which is what scales
        # with step count; the load is a fixed cost counted as overhead instead.
        "seconds_per_step": round(
            float(stats.metrics.get("train_runtime", elapsed)) / max(int(stats.global_step), 1),
            2,
        ),
        "train_runtime_s": round(float(stats.metrics.get("train_runtime", 0.0)), 1),
        "load_overhead_s": round(elapsed - float(stats.metrics.get("train_runtime", 0.0)), 1),
        "elapsed_s": round(elapsed, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }

    if not smoke:
        metrics = trainer.evaluate()
        result["eval_loss"] = round(float(metrics.get("eval_loss", float("nan"))), 4)

    if smoke or not push_repo:
        print("smoke run or no repo given — not pushing")
        result["pushed"] = None
    else:
        # merged_16bit, not the adapter alone: Phase 5 serves this through plain
        # transformers on ZeroGPU, which should not need Unsloth at inference.
        model.push_to_hub_merged(
            push_repo,
            processor,
            save_method="merged_16bit",
            token=os.environ["HF_TOKEN"],
        )
        result["pushed"] = push_repo
        print(f"pushed merged model to {push_repo}")

    print(json.dumps(result, indent=2))
    return result


def estimate(
    rows: int,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    seconds_per_step: float,
    gpu_rate: float,
    overhead_minutes: float,
) -> dict[str, float]:
    """Work out what a run should cost before committing to it.

    Args:
        rows: Training rows.
        epochs: Epochs planned.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        seconds_per_step: Measured by --smoke, or the conservative default.
        gpu_rate: USD per GPU-hour.
        overhead_minutes: Load, merge and push time, which is billed too.

    Returns:
        Steps, minutes and dollars.
    """
    effective_batch = max(batch_size * grad_accum, 1)
    steps = max(round(rows * epochs / effective_batch), 1)
    minutes = (steps * seconds_per_step) / 60.0 + overhead_minutes
    return {
        "steps": float(steps),
        "minutes": round(minutes, 1),
        "usd": round(minutes / 60.0 * gpu_rate, 3),
    }


def newest(pattern: str) -> str:
    """Find the most recent local file matching a glob.

    Args:
        pattern: Glob relative to the repo root.

    Returns:
        The basename, since the container sees the directory mounted flat.

    Raises:
        FileNotFoundError: If nothing matches.
    """
    matches = sorted(Path().glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched {pattern!r}. Run build_train_split first.")
    return matches[-1].name


@app.local_entrypoint()
def main(
    train_file: str = "",
    val_file: str = "",
    run_name: str = "sft-v1",
    epochs: float = 2.0,
    seq_len: int = 4096,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    push_repo: str = "",
    seed: int = 3407,
    seconds_per_step: float = 6.0,
    gpu_rate: float = L4_USD_PER_HOUR,
    smoke: bool = False,
    yes: bool = False,
) -> None:
    """Estimate the cost, then launch only if explicitly told to.

    Args:
        train_file: Training JSONL basename. Defaults to the newest.
        val_file: Validation JSONL basename. Defaults to the newest.
        run_name: Checkpoint directory; reuse to resume an interrupted run.
        epochs: Epochs for the full run.
        seq_len: Max sequence length. 4096 because the split measured a longest
            user message of 10,652 chars — near 3,000 tokens once system prompt
            and answer are added, which 2048 would silently truncate.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        learning_rate: Peak LR.
        push_repo: HF repo for the merged model.
        seed: Reproducibility.
        seconds_per_step: Used for the estimate. Replace with what --smoke measures.
        gpu_rate: USD per GPU-hour; override if Modal's pricing moved.
        smoke: Ten steps to measure s/step. Cheap, and never pushes.
        yes: Required to actually launch. Without it this only estimates.
    """
    train_file = train_file or newest("data/train/sft_*_train.jsonl")
    val_file = val_file or newest("data/train/sft_*_val.jsonl")
    rows = sum(1 for _ in Path("data/train", train_file).open(encoding="utf-8"))

    plan = estimate(
        rows=10 * batch_size * grad_accum if smoke else rows,
        epochs=1 if smoke else epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        seconds_per_step=seconds_per_step,
        gpu_rate=gpu_rate,
        overhead_minutes=6.0 if smoke else 12.0,
    )

    print("=== PLANNED RUN ===")
    print(f"  model            : {MODEL_NAME} on {GPU}")
    print(f"  train file       : {train_file}  ({rows} rows)")
    print(f"  val file         : {val_file}")
    print(f"  mode             : {'SMOKE (10 steps, no push)' if smoke else 'FULL'}")
    print(f"  epochs           : {1 if smoke else epochs}")
    print(f"  seq_len          : {seq_len}")
    print(f"  effective batch  : {batch_size * grad_accum}")
    print(f"  steps            : {int(plan['steps'])}")
    print(f"  s/step assumed   : {seconds_per_step}")
    print(f"  est. wall time   : {plan['minutes']} min")
    print(f"  est. cost        : ${plan['usd']}  (at ${gpu_rate}/GPU-hour)")
    print(f"  push to          : {push_repo or '(not pushing)'}")

    if plan["usd"] > MAX_RUN_USD:
        print(f"\nREFUSING: ${plan['usd']} exceeds the ${MAX_RUN_USD} per-run rail.")
        return

    if not yes:
        print("\nEstimate only. Re-run with --yes to launch.")
        return

    if not smoke and not push_repo:
        print("\nREFUSING: a full run with no --push-repo would train and discard the model.")
        return

    result = train.remote(
        train_file=train_file,
        val_file=val_file,
        run_name=run_name,
        epochs=epochs,
        max_steps=10 if smoke else -1,
        seq_len=seq_len,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        push_repo="" if smoke else push_repo,
        seed=seed,
    )

    out = Path("evals/results")
    out.mkdir(parents=True, exist_ok=True)
    name = f"train_{run_name}{'_smoke' if smoke else ''}.json"
    (out / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out / name}")
    print(f"measured s/step: {result['seconds_per_step']}  peak VRAM: {result['peak_vram_gb']} GB")
