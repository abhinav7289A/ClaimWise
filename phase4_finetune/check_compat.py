"""Probe whether Qwen3.5-4B can actually be LoRA fine-tuned, before spending on it.

**Why this file exists at all.** CLAUDE.md picks Qwen3.5-4B because users attach
photos of bills and claim forms, so the generator has to be natively multimodal.
The catch is that Qwen3.5 is not a plain transformer: it combines Gated DeltaNet
layers with a sparse Mixture-of-Experts, and fine-tuning libraries support new
architectures weeks or months after the weights ship. A LoRA adapter needs
defined injection points; if the library has no mapping for this architecture,
`get_peft_model` either raises or — worse — silently attaches to nothing and
trains a no-op.

Discovering that during a paid training run, with a dataset already generated,
is the expensive path. This is the cheap one: load the model, attach an adapter,
take three optimiser steps on throwaway data, save, and report. Total spend is
cents. In engineering terms it is a **spike** — buy information about the risky
part before committing to the plan built around it.

**Why bf16 LoRA and not QLoRA.** CLAUDE.md's Phase 4 plan says QLoRA. Unsloth's
own Qwen3.5 documentation now advises against it: *"It is not recommended to do
QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense, due to
higher than normal quantization differences."* So this probes bf16 LoRA, which
needs ~10GB for the 4B, and reports peak VRAM so the real run's GPU is chosen
from a measurement rather than a guess.

**Why every check is reported separately.** A single try/except around the whole
probe tells you "it failed". Seven independent checks tell you *which capability
is missing*, which is what decides go-vs-fallback. A model that loads but cannot
save an adapter is a different problem from one that will not load at all.

**Why the vision layers are frozen.** The RAFT dataset is text-only by budget
decision. Freezing the vision tower means the LoRA touches only language layers,
which is what preserves the base model's image understanding — the whole reason
this model was chosen. Check 5 verifies the freeze actually took, because a
config flag that is silently ignored would let vision weights drift during
training and regress the image path with no error anywhere.

Usage — this runs on Modal, not locally:
    modal run phase4_finetune/check_compat.py
    modal run phase4_finetune/check_compat.py --model unsloth/Qwen3-4B
    modal run phase4_finetune/check_compat.py --gpu A10G
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "claimwise-check-compat"

# The Phase 4 model decision from CLAUDE.md §5. Unsloth republishes the official
# weights with its own patches applied; falling back to "Qwen/Qwen3.5-4B" works
# but loses Unsloth's kernels, so start with theirs.
PRIMARY_MODEL = "unsloth/Qwen3.5-4B"

# The documented fallback: text-only, plain dense architecture, long-supported by
# every LoRA library. Vision would then be handled by the un-fine-tuned base
# model or a NIM free vision endpoint.
FALLBACK_MODEL = "unsloth/Qwen3-4B"

# L4 is the cheapest Modal GPU with enough headroom: 24GB against the ~10GB
# bf16 LoRA needs. A10G is ~40% dearer and faster, which matters for a real
# training run and not for a three-step probe.
DEFAULT_GPU = "L4"

# Per-second Modal rates, USD, used only for the estimate printed before launch.
# Kept here rather than in config.yaml because this file is the only thing in the
# project that spends Modal credits so far.
GPU_USD_PER_SECOND = {
    "L4": 0.000222,
    "A10G": 0.000306,
    "A100": 0.000583,
    "A100-80GB": 0.000694,
}

# Generous enough that a cold image build plus an 8GB weight download cannot trip
# it, low enough that a hung job cannot quietly drain credits.
TIMEOUT_SECONDS = 1800

RESULTS_DIR = Path("evals/results")

# **Do not pin torch here.** The first probe (2026-08-24) pinned torch==2.6.0
# alongside transformers>=4.57.0 and failed at `from unsloth import ...` after
# 8.3s with `module 'torch.utils._pytree' has no attribute ...` — the classic
# signature of a transformers/torch skew. It cost $0.002 and told us nothing
# about Qwen3.5, because the import dies before any model is touched.
#
# Unsloth declares a mutually-consistent torch/transformers/trl/peft set of its
# own. Letting it resolve them is more robust than three hand-picked pins that
# have to stay in step with each other across releases. On Linux the PyPI torch
# wheel bundles CUDA, so no devel base image is needed.
#
# hf_transfer is a meaningful cost lever, not a nicety: it parallelises the ~8GB
# weight download, which is most of a cold run's billed seconds.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("unsloth", "unsloth_zoo", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Weights persist between runs, so a second probe skips the multi-GB download.
# That is the difference between a ~$0.20 first run and a ~$0.07 repeat.
hf_cache = modal.Volume.from_name("claimwise-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache},
    # Qwen weights are not gated today, but a token costs nothing to attach and
    # avoids a rerun if that changes. Create it with:
    #   modal secret create huggingface HF_TOKEN=hf_...
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def probe(model_name: str, max_seq_length: int = 2048) -> dict[str, Any]:
    """Run the seven compatibility checks on one model and report each outcome.

    Never raises for a compatibility failure. A check that fails records its
    error and the remaining checks are skipped, because they all depend on the
    model having loaded — but the result still comes back as data, so the caller
    can write a decision record instead of reading a traceback.

    Args:
        model_name: Hugging Face repo id to probe.
        max_seq_length: Sequence length to load with. 2048 is enough to hold a
            RAFT example (question + ~4 chunks + answer) without inflating the
            probe's memory use.

    Returns:
        A record with per-check outcomes, peak VRAM, and a go/fallback verdict.
    """
    import time

    import torch

    os.environ.setdefault("HF_HOME", "/cache/huggingface")

    checks: list[dict[str, Any]] = []
    started = time.perf_counter()

    def record(name: str, ok: bool, detail: str) -> None:
        """Append one check outcome."""
        checks.append({"check": name, "ok": ok, "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}: {detail}", flush=True)

    result: dict[str, Any] = {
        "model": model_name,
        "gpu": os.environ.get("MODAL_GPU", DEFAULT_GPU),
        "checks": checks,
        "peak_vram_gb": None,
        "verdict": "fallback",
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    # --- Check 0: report the resolved dependency versions -------------------
    #
    # Printed BEFORE the unsloth import, because the 2026-08-24 probe died on
    # that import and left us with no way to tell which versions had actually
    # been installed. A skew is obvious from these four lines and invisible
    # without them.
    versions: dict[str, str] = {"torch": torch.__version__}
    for module_name in ("transformers", "peft", "trl", "unsloth"):
        try:
            versions[module_name] = __import__(module_name).__version__
        except Exception as error:  # noqa: BLE001
            versions[module_name] = f"UNAVAILABLE ({type(error).__name__})"
    result["versions"] = versions
    record("dependency versions", True, ", ".join(f"{k}={v}" for k, v in versions.items()))

    # --- Check 1: the library imports at all -------------------------------
    #
    # Unsloth pins tightly to torch and transformers versions. An import error
    # here is a dependency problem in the image, not a model problem, and the
    # two need different fixes.
    try:
        from unsloth import FastLanguageModel, FastVisionModel

        record("unsloth imports", True, "FastLanguageModel and FastVisionModel available")
    except Exception as error:  # noqa: BLE001 - reporting is the point
        record("unsloth imports", False, f"{type(error).__name__}: {error}")
        result["verdict"] = "fallback"
        result["elapsed_s"] = round(time.perf_counter() - started, 1)
        return result

    # --- Check 2: the weights load in bf16 ---------------------------------
    #
    # Qwen3.5 is a unified vision-language model, so FastVisionModel is the
    # correct loader even when only the language side will be trained. Trying
    # FastLanguageModel first and falling back tells us which one this repo is,
    # which the training script then has to match.
    model = None
    tokenizer = None
    loader_used = None
    for loader_name, loader in (("FastVisionModel", FastVisionModel), ("FastLanguageModel", FastLanguageModel)):
        try:
            model, tokenizer = loader.from_pretrained(
                model_name,
                max_seq_length=max_seq_length,
                # bf16, NOT 4-bit. Unsloth advises against QLoRA on Qwen3.5 due
                # to larger-than-usual quantisation error, dense or MoE alike.
                load_in_4bit=False,
                dtype=torch.bfloat16,
            )
            loader_used = loader_name
            record("model loads (bf16)", True, f"loaded via {loader_name}")
            break
        except Exception as error:  # noqa: BLE001
            record(f"model loads via {loader_name}", False, f"{type(error).__name__}: {error}")

    if model is None:
        result["elapsed_s"] = round(time.perf_counter() - started, 1)
        return result

    result["loader"] = loader_used

    # --- Check 3: a chat template exists -----------------------------------
    #
    # RAFT examples are supervised chat turns. Without a template the training
    # script has to hand-roll the format, and a mismatch between training and
    # serving formats is a silent quality killer that no metric names directly.
    try:
        template = getattr(tokenizer, "chat_template", None)
        if template:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Is knee surgery covered?"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            record("chat template", True, f"{len(template)} chars, renders {len(rendered)} chars")
        else:
            record("chat template", False, "tokenizer has no chat_template")
    except Exception as error:  # noqa: BLE001
        record("chat template", False, f"{type(error).__name__}: {error}")

    # --- Check 4: a LoRA adapter attaches ----------------------------------
    #
    # The check that actually decides go-vs-fallback. Target modules are the
    # standard attention and MLP projections; on a MoE architecture the expert
    # layers may be named differently, which is exactly what would fail here.
    try:
        peft_kwargs: dict[str, Any] = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "random_state": 3407,
        }
        if loader_used == "FastVisionModel":
            # Text-only LoRA: the vision tower stays frozen so image
            # understanding survives the fine-tune untouched.
            peft_kwargs.update(
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
            )
            model = FastVisionModel.get_peft_model(model, **peft_kwargs)
        else:
            peft_kwargs["target_modules"] = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
            model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        # A LoRA that attaches to nothing reports zero trainable parameters and
        # would otherwise "train" happily for an hour, producing an adapter that
        # changes no output at all.
        record(
            "LoRA attaches",
            trainable > 0,
            f"{trainable:,} trainable of {total:,} ({100 * trainable / max(total, 1):.3f}%)",
        )
        result["trainable_params"] = trainable
        result["total_params"] = total
    except Exception as error:  # noqa: BLE001
        record("LoRA attaches", False, f"{type(error).__name__}: {error}")
        result["elapsed_s"] = round(time.perf_counter() - started, 1)
        return result

    # --- Check 5: the vision tower is genuinely frozen ---------------------
    #
    # `finetune_vision_layers=False` is a request. This verifies it was honoured,
    # because a silently ignored flag would let vision weights drift during
    # training and regress the image path with nothing raising anywhere.
    if loader_used == "FastVisionModel":
        try:
            vision_trainable = sum(
                p.numel()
                for name, p in model.named_parameters()
                if p.requires_grad and ("visual" in name or "vision" in name)
            )
            record(
                "vision tower frozen",
                vision_trainable == 0,
                f"{vision_trainable:,} trainable vision params (want 0)",
            )
        except Exception as error:  # noqa: BLE001
            record("vision tower frozen", False, f"{type(error).__name__}: {error}")

    # --- Check 6: forward + backward + optimiser step ----------------------
    #
    # Attaching an adapter and actually training through it are different
    # things. Gradient checkpointing on an unusual architecture is a common
    # place for the backward pass to fail while the forward pass looks fine.
    try:
        torch.cuda.reset_peak_memory_stats()
        model.train()
        optimiser = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        # `FastVisionModel.from_pretrained` returns a PROCESSOR, not a tokenizer.
        # Its `__call__` signature is (images=..., text=...), so a positional
        # list of strings is read as image sources — which is exactly how the
        # 2026-08-24 probe failed:
        #   "Incorrect image source... Got Question: Is knee surgery covered?"
        # The processor wraps the text tokenizer at `.tokenizer`; on a plain
        # language model there is no wrapper and the object is already one.
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token

        batch = text_tokenizer(
            ["Question: Is knee surgery covered?\nAnswer: Not in the given context."] * 2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to("cuda")
        batch["labels"] = batch["input_ids"].clone()

        losses = []
        for _ in range(3):
            optimiser.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimiser.step()
            losses.append(round(float(loss.item()), 4))

        record("3 training steps", True, f"losses {losses}")
        result["losses"] = losses
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except Exception as error:  # noqa: BLE001
        record("3 training steps", False, f"{type(error).__name__}: {error}")
        result["elapsed_s"] = round(time.perf_counter() - started, 1)
        return result

    # --- Check 7: the adapter saves ----------------------------------------
    #
    # The artifact Phase 4 delivers. A run that trains for an hour and cannot
    # write its adapter has produced nothing.
    try:
        save_dir = "/cache/compat_probe_adapter"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        files = sorted(os.listdir(save_dir))
        record("adapter saves", bool(files), f"{len(files)} files: {files[:4]}")
    except Exception as error:  # noqa: BLE001
        record("adapter saves", False, f"{type(error).__name__}: {error}")

    result["verdict"] = "go" if all(check["ok"] for check in checks) else "fallback"
    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(model: str = PRIMARY_MODEL, gpu: str = DEFAULT_GPU, seq_len: int = 2048) -> None:
    """Print a cost estimate, run the probe, and write the decision record.

    Args:
        model: Hugging Face repo id to probe.
        gpu: Modal GPU name, used for the cost estimate only — the function's
            own decorator fixes the hardware, so change `DEFAULT_GPU` to run on
            different silicon.
        seq_len: Max sequence length to load with.
    """
    rate = GPU_USD_PER_SECOND.get(gpu, GPU_USD_PER_SECOND[DEFAULT_GPU])

    # CLAUDE.md §3: every Modal script prints an estimated cost before spending.
    print("=== COST ESTIMATE (before launch) ===")
    print(f"model            : {model}")
    print(f"gpu              : {gpu} @ ${rate:.6f}/s")
    print(f"timeout ceiling  : {TIMEOUT_SECONDS}s -> ${TIMEOUT_SECONDS * rate:.2f} worst case")
    print(f"expected runtime : ~600s cold (image build + weight download) -> ~${600 * rate:.2f}")
    print(f"                   ~180s warm (weights cached on volume)      -> ~${180 * rate:.2f}")
    print("Modal budget     : $30.00 hard cap, $5.00 reserved for Phase 5\n")

    print(f"=== PROBING {model} ===", flush=True)
    result = probe.remote(model, seq_len)

    print("\n=== RESULT ===")
    print(f"verdict          : {result['verdict'].upper()}")
    print(f"loader           : {result.get('loader')}")
    print(f"trainable params : {result.get('trainable_params')}")
    print(f"peak VRAM        : {result.get('peak_vram_gb')} GB")
    print(f"elapsed          : {result.get('elapsed_s')}s -> ~${result.get('elapsed_s', 0) * rate:.3f}")

    failed = [check["check"] for check in result["checks"] if not check["ok"]]
    if failed:
        print(f"\nfailed checks    : {failed}")
        print(f"\nNext: re-run against the documented fallback.\n"
              f"  modal run phase4_finetune/check_compat.py --model {FALLBACK_MODEL}")
    else:
        print("\nAll checks passed. Qwen3.5-4B is trainable with bf16 LoRA.")
        print(f"Size the real run from peak VRAM: {result.get('peak_vram_gb')} GB at "
              f"seq_len={seq_len}, before batch size and longer sequences.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = model.split("/")[-1].replace(".", "_")
    out_path = RESULTS_DIR / f"compat_{stamp}_{slug}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nrecord           : {out_path}")
