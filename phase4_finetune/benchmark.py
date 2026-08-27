"""Compare generators through the identical retrieval pipeline. Phase 4 exit criterion.

**The comparison that matters is a generator swap, nothing else.** Retrieval runs
once, locally, against the FROZEN `claimwise_mx__` 4-document collection the golden
set was written against. Every model then answers from byte-identical prompts. Any
difference in the scores below is the generator and only the generator — which is
the whole point, and is why retrieval is not re-run per model.

**Why the scoring is deterministic and not RAGAS.** Every metric here is computed
by rule: does the cited page exist in the passages, is it the right page, do the
figures in the reference answer appear in the output, did the model refuse when it
should have. No judge model, so it costs nothing, cannot drift between runs, and
cannot be accused of grading its own homework. RAGAS stays available for a later
subset run; the OpenRouter balance (~$1.33) is reserved for Phase 4.5 preference
generation, where a judge is structurally required and here it is not.

**Why the API arm is deferred.** CLAUDE.md asks for fine-tuned vs base vs a large
API model. The exit criterion is "fine-tuned clearly >= base", which is settled by
the two local arms at GPU cost alone. `--models` takes any number of HF repo ids,
so a third arm is one flag when there is budget for it.

**Contamination.** The golden set was held out of training by fingerprint, verified
`dropped_holdout: 0` in both generators and again in the split. These 92 questions
have never been trained on, which is what makes this benchmark meaningful.

Usage:
    modal run phase4_finetune/benchmark.py --help
    modal run phase4_finetune/benchmark.py --limit 10
    modal run phase4_finetune/benchmark.py \\
        --models unsloth/Qwen3.5-4B,AbhiCommits/claimwise-qwen35-4b
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "claimwise-benchmark"
GPU = "L4"
TIMEOUT_SECONDS = 3600

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Same unpinned stack as training (P-20): let it resolve its own torch.
    .pip_install("unsloth", "unsloth_zoo", "hf_transfer", "accelerate")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("claimwise-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME, image=image)

CITATION_RE = re.compile(r"\[p\.(\d+)\]", re.IGNORECASE)
# Rupee amounts, percentages and plain figures, however the document writes them.
FIGURE_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@app.function(
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def generate_batch(
    model_name: str,
    chats: list[list[dict[str, str]]],
    max_new_tokens: int,
) -> list[str]:
    """Generate one answer per chat with a single model, greedily.

    Greedy decoding, not sampling: a benchmark that moves between runs cannot
    settle a "clearly better than" question, and Phase 3 already lost time to a
    non-deterministic generator at temperature 0.

    Args:
        model_name: HF repo id to load.
        chats: One message list per question, already rendered upstream.
        max_new_tokens: Generation cap.

    Returns:
        One answer string per chat, in the same order.
    """
    import os

    import torch
    from transformers import AutoProcessor

    # Qwen3.5 is a vision-language model, and the right auto class moved between
    # transformers versions. Try the current name first and report what worked,
    # rather than hard-coding one and discovering it on a billed GPU.
    model = None
    loader_used = ""
    for class_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            import transformers

            auto_class = getattr(transformers, class_name)
            model = auto_class.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="cuda",
                token=os.environ["HF_TOKEN"],
            )
            loader_used = class_name
            break
        except Exception as error:  # noqa: BLE001 - try the next class
            print(f"{class_name} failed: {type(error).__name__}: {error}")
    if model is None:
        raise RuntimeError(f"No auto class could load {model_name}.")
    print(f"loaded {model_name} with {loader_used}")

    processor = AutoProcessor.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
    tokenizer = getattr(processor, "tokenizer", processor)
    model.eval()

    answers: list[str] = []
    # Per-item progress, flushed. Generation is sequential and each answer takes
    # 15-20s, so a batch-level counter makes a working run look hung.
    import time

    started = time.perf_counter()
    for index, messages in enumerate(chats):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Slice off the prompt so only the completion is scored.
        completion = generated[0][inputs["input_ids"].shape[1]:]
        answers.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
        elapsed = time.perf_counter() - started
        print(
            f"  {index + 1}/{len(chats)}  "
            f"{len(completion)} new tokens  "
            f"{elapsed / (index + 1):.1f}s/item  "
            f"eta {(len(chats) - index - 1) * elapsed / (index + 1) / 60:.1f} min",
            flush=True,
        )

    return answers


def cited_pages(text: str) -> set[int]:
    """Pull every [p.N] citation out of an answer.

    Args:
        text: The generated answer.

    Returns:
        The set of cited page numbers.
    """
    return {int(match) for match in CITATION_RE.findall(text)}


def figures(text: str) -> set[str]:
    """Normalise every number in a string for comparison.

    Commas are stripped so "2,40,000" and "240000" compare equal — the point is
    whether the *value* survived, not how it was punctuated.

    Args:
        text: Any answer text.

    Returns:
        Normalised numeric strings.
    """
    found = set()
    for raw in FIGURE_RE.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        if cleaned and cleaned not in {"0"}:
            found.add(cleaned.lstrip("0") or "0")
    return found


def score_one(
    item: dict[str, Any],
    answer: str,
    context_pages: set[int],
    refusal_text: str,
) -> dict[str, Any]:
    """Score one generated answer against its golden record.

    Args:
        item: The golden row.
        answer: What the model produced.
        context_pages: Pages present in the retrieved passages.
        refusal_text: The configured refusal sentence.

    Returns:
        Per-item flags, all booleans or None where not applicable.
    """
    refused = refusal_text.lower()[:40] in answer.lower()
    is_negative = not item.get("ground_truth_pages")
    citations = cited_pages(answer)
    truth_pages = set(item.get("ground_truth_pages") or [])

    result: dict[str, Any] = {
        "id": item.get("id"),
        "question_type": item.get("question_type"),
        "is_negative": is_negative,
        "refused": refused,
        "answer": answer,
        "cited": sorted(citations),
    }

    if is_negative:
        # The only correct behaviour on a negative is to refuse.
        result["correct_refusal"] = refused
        return result

    # A hallucinated citation is a page the model cited that was never in front
    # of it. This is the single behaviour RAFT is meant to fix.
    result["hallucinated_citation"] = bool(citations - context_pages)
    result["cited_anything"] = bool(citations)
    result["cited_truth_page"] = bool(citations & truth_pages)
    result["over_refused"] = refused
    # Did the reference answer's figures survive into the output? Targets the
    # Phase 3 defect directly: a generator handed 240,000 writing 240,0000.
    reference = figures(item.get("answer", ""))
    result["reference_figures"] = len(reference)
    result["figures_preserved"] = bool(reference) and reference <= figures(answer)
    return result


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item scores into the numbers that go in METRICS.

    Args:
        rows: Scored items for one model.

    Returns:
        Rates, each with the denominator it was computed over.
    """
    negatives = [row for row in rows if row["is_negative"]]
    positives = [row for row in rows if not row["is_negative"]]
    with_figures = [row for row in positives if row.get("reference_figures")]
    # A model that never cites cannot cite wrongly, so a hallucination rate over
    # *all* positives rewards silence and flatters a weak generator. The base
    # model scored 0.0 on the 10-question dry run purely because it cited nothing
    # in 8 of 10 answers. Conditioning on rows that actually cited is the honest
    # denominator; the unconditioned rate is kept alongside it, not replaced.
    cited_something = [row for row in positives if row.get("cited_anything")]

    def rate(subset: list[dict[str, Any]], key: str) -> float | None:
        """Share of a subset where `key` is true."""
        if not subset:
            return None
        return round(sum(1 for row in subset if row.get(key)) / len(subset), 4)

    return {
        "n_total": len(rows),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "refusal_accuracy_on_negatives": rate(negatives, "correct_refusal"),
        "over_refusal_on_positives": rate(positives, "over_refused"),
        "citation_present": rate(positives, "cited_anything"),
        "hallucinated_citation_rate": rate(positives, "hallucinated_citation"),
        "hallucinated_when_cited": rate(cited_something, "hallucinated_citation"),
        "n_cited_something": len(cited_something),
        "cited_correct_page": rate(positives, "cited_truth_page"),
        "figures_preserved": rate(with_figures, "figures_preserved"),
        "n_scored_for_figures": len(with_figures),
    }


@app.local_entrypoint()
def main(
    models: str = "unsloth/Qwen3.5-4B,AbhiCommits/claimwise-qwen35-4b",
    config: str = "config.yaml",
    golden: str = "data/eval/golden.jsonl",
    collection_prefix: str = "claimwise_mx",
    limit: int = 0,
    max_new_tokens: int = 320,
    tag: str = "phase4",
) -> None:
    """Retrieve once locally, generate per model on GPU, score locally.

    Args:
        models: Comma-separated HF repo ids. Base first, fine-tuned second.
        config: Config path for retrieval settings.
        golden: The held-out eval set.
        collection_prefix: FROZEN 4-document collection the golden set matches.
            Do not point this at the 10-document training collection.
        limit: Score only the first N questions. 0 means all.
        max_new_tokens: Generation cap per answer.
        tag: Label in the output filename.
    """
    # Imported here, not at module scope: this module is also imported inside the
    # Modal container, where ClaimWise packages do not exist.
    from common.config import cfg_get, load_config
    from phase1_rag.rag_chain import SYSTEM_PROMPT, build_prompt
    from phase3_agents.retrieval_node import build_resources, retrieve_global

    repos = [name.strip() for name in models.split(",") if name.strip()]
    items = [json.loads(line) for line in Path(golden).open(encoding="utf-8") if line.strip()]
    if limit:
        items = items[:limit]

    cfg = load_config(Path(config))
    cfg.setdefault("chunk_policy", {})["collection_prefix"] = collection_prefix
    refusal_text = cfg_get(
        cfg, "rag.refusal_text", "That isn't covered in the policy documents you've uploaded."
    )
    system = SYSTEM_PROMPT.format(refusal_text=refusal_text)

    print(f"=== RETRIEVAL (once, shared by every model) ===")
    print(f"  collection : {collection_prefix}  (frozen 4-doc set)")
    print(f"  questions  : {len(items)}")

    resources = build_resources(cfg)
    settings = dict(resources.settings)
    chats: list[list[dict[str, str]]] = []
    context_pages: list[set[int]] = []
    try:
        for item in items:
            chunks = retrieve_global(
                resources,
                item["question"],
                user_id=settings["default_user_id"],
                top_k=settings["top_k"],
            )
            context_pages.append({int(getattr(chunk, "page", 0) or 0) for chunk in chunks})
            chats.append(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": build_prompt(item["question"], chunks)},
                ]
            )
    finally:
        resources.close()
    print("  retrieval complete\n")

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection_prefix": collection_prefix,
        "golden": golden,
        "n_questions": len(items),
        "max_new_tokens": max_new_tokens,
        "models": {},
    }
    per_item: dict[str, list[dict[str, Any]]] = {}

    for repo in repos:
        print(f"=== GENERATING: {repo} ===")
        answers = generate_batch.remote(
            model_name=repo, chats=chats, max_new_tokens=max_new_tokens
        )
        scored = [
            score_one(item, answer, pages, refusal_text)
            for item, answer, pages in zip(items, answers, context_pages)
        ]
        per_item[repo] = scored
        report["models"][repo] = summarise(scored)
        print(json.dumps(report["models"][repo], indent=2))

    out_dir = Path("evals/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"benchmark_{stamp}_{tag}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    items_path = out_dir / f"benchmark_{stamp}_{tag}_items.jsonl"
    with items_path.open("w", encoding="utf-8") as handle:
        for repo, scored in per_item.items():
            for row in scored:
                handle.write(json.dumps({"model": repo, **row}, ensure_ascii=False) + "\n")

    # The delta table is the deliverable: absolutes tell you little, the change
    # between generators through an identical pipeline tells you everything.
    print("\n=== GENERATOR SWAP ===")
    keys = [
        "citation_present",
        "hallucinated_when_cited",
        "cited_correct_page",
        "figures_preserved",
        "refusal_accuracy_on_negatives",
        "over_refusal_on_positives",
    ]
    header = "metric".ljust(34) + "".join(repo.split("/")[-1][:20].rjust(22) for repo in repos)
    print(header)
    for key in keys:
        row = key.ljust(34)
        for repo in repos:
            value = report["models"][repo].get(key)
            row += ("n/a" if value is None else f"{value:.4f}").rjust(22)
        print(row)

    print(f"\nwrote {report_path}")
    print(f"wrote {items_path}")
