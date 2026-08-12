"""Generation-quality evaluation: the paid tier.

Runs every golden question through the full RAG chain, then measures the answers
two ways.

**Free, and for this product more important than either paid metric:**

* **citation validity** — every `[p.N]` checked against the pages actually
  retrieved. A fabricated citation is the failure that destroys trust fastest,
  because it looks authoritative.
* **false refusal rate** — positives answered with the refusal sentence. A
  system that declines when the answer is in front of it is useless in a
  quieter way than one that invents things, and it passes review because
  refusing *looks* like caution.
* **refusal accuracy** — negatives correctly declined.
* latency, tokens, and estimated spend.

**Paid, via RAGAS:**

* **faithfulness** — are the answer's claims supported by the retrieved context?
* **answer relevancy** — does the answer actually address the question?

**Why only two RAGAS metrics.** `context_precision` costs one LLM call *per
retrieved chunk*, which is what makes a full RAGAS run ~1,000 calls for 100
questions. But context recall and precision are already computed exactly, for
free, in `evals/retrieval_metrics.py` from the golden set's ground-truth pages.
Paying a judge to estimate a number we know precisely is waste, so this file
runs only the metrics that genuinely require judgement — roughly 4 calls per
question instead of 10.

Usage:
    python -m evals.run_ragas --help
    python -m evals.run_ragas --skip-ragas --limit 5     # free metrics only
    python -m evals.run_ragas --allow-unverified
    python -m evals.run_ragas
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from common.generator import build_generator
from phase1_rag.embed_index import collection_name_for
from phase1_rag.rag_chain import answer_question
from phase1_rag.rag_chain import resolve_settings as resolve_rag_settings

LOGGER = logging.getLogger("claimwise.run_ragas")


@dataclass
class AnsweredItem:
    """One golden question, answered, with everything needed to score it.

    Attributes:
        item_id: Golden set id.
        question: The question asked.
        question_type: "lookup", "calculation" or "negative".
        policy_type: Source policy type, "" for negatives.
        expected_answer: Ground-truth answer from the golden set.
        answer: What the pipeline actually produced.
        contexts: Retrieved chunk texts, in rank order — the evidence RAGAS
            judges faithfulness against.
        retrieved_pages: Pages retrieved, in rank order.
        had_evidence: Whether the ground-truth page was actually in the context
            the model saw. Without this, "refused" cannot be interpreted: the
            same behaviour is a defect when evidence was present and correct
            when it was absent.
        cited_pages: Pages the model cited.
        invalid_citations: Cited pages that were never retrieved.
        refused: Whether the refusal sentence was emitted.
        should_refuse: True for negatives.
        latency_seconds: End-to-end time.
        prompt_tokens: Input tokens billed.
        completion_tokens: Output tokens billed.
    """

    item_id: str
    question: str
    question_type: str
    policy_type: str
    expected_answer: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    retrieved_pages: list[int] = field(default_factory=list)
    had_evidence: bool = False
    cited_pages: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    refused: bool = False
    should_refuse: bool = False
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def load_golden(path: Path, verified_only: bool) -> list[dict[str, Any]]:
    """Read golden items from JSONL.

    Args:
        path: Path to `golden.jsonl`.
        verified_only: Keep only human-verified items.

    Returns:
        The selected items.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Golden set not found: {path.resolve()}. Run build_eval_set first."
        )
    with path.open("r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]

    if not verified_only:
        return items
    verified = [item for item in items if item.get("verified")]
    if len(verified) < len(items):
        LOGGER.warning(
            "Skipping %d unverified item(s). Pass --allow-unverified for "
            "provisional numbers.",
            len(items) - len(verified),
        )
    return verified


def free_metrics(answered: list[AnsweredItem]) -> dict[str, Any]:
    """Compute the deterministic answer-quality metrics.

    None of these needs a judge, and for an insurance assistant they matter more
    than the judged ones: a fabricated citation or a false refusal is a product
    failure regardless of how fluent the prose is.

    Args:
        answered: All answered items.

    Returns:
        A metrics dict.
    """
    positives = [item for item in answered if not item.should_refuse]
    negatives = [item for item in answered if item.should_refuse]

    # Split positives by whether the ground-truth page reached the context.
    # Refusing is a defect only in the first group and correct in the second;
    # reporting one combined "false refusal rate" conflates them and reads as
    # far worse (or better) than reality.
    with_evidence = [item for item in positives if item.had_evidence]
    without_evidence = [item for item in positives if not item.had_evidence]

    answered_positives = [item for item in positives if not item.refused]
    with_citations = [item for item in answered_positives if item.cited_pages]
    clean_citations = [item for item in with_citations if not item.invalid_citations]

    latencies = [item.latency_seconds for item in answered] or [0.0]
    tokens_in = sum(item.prompt_tokens for item in answered)
    tokens_out = sum(item.completion_tokens for item in answered)
    latencies_sorted = sorted(latencies)
    p95_index = min(len(latencies_sorted) - 1, int(round(0.95 * (len(latencies_sorted) - 1))))

    return {
        "positives": len(positives),
        "negatives": len(negatives),
        # Of positives that produced an answer, how many cited at least once.
        "citation_coverage": round(len(with_citations) / max(1, len(answered_positives)), 4),
        # Of those that cited, how many cited only real retrieved pages.
        "citation_validity": round(len(clean_citations) / max(1, len(with_citations)), 4),
        "fabricated_citation_items": [
            item.item_id for item in with_citations if item.invalid_citations
        ],
        "positives_with_evidence": len(with_evidence),
        "positives_without_evidence": len(without_evidence),
        # THE defect: refused while the ground-truth page was in context.
        "false_refusal_rate": round(
            sum(1 for item in with_evidence if item.refused) / max(1, len(with_evidence)), 4
        ),
        "false_refusal_items": [item.item_id for item in with_evidence if item.refused],
        # Correct behaviour: declined because retrieval gave it nothing to use.
        "correct_refusal_rate": round(
            sum(1 for item in without_evidence if item.refused) / max(1, len(without_evidence)), 4
        ),
        # Answered with no ground-truth evidence in context. Some are genuine
        # hallucinations; others are correct answers found on a page the golden
        # set didn't record, since policy documents repeat content. Needs
        # inspection rather than a verdict.
        "ungrounded_answer_rate": round(
            sum(1 for item in without_evidence if not item.refused)
            / max(1, len(without_evidence)),
            4,
        ),
        "ungrounded_answer_items": [
            item.item_id for item in without_evidence if not item.refused
        ],
        # Correctly declined the unanswerable.
        "refusal_accuracy": round(
            sum(1 for item in negatives if item.refused) / max(1, len(negatives)), 4
        ),
        "missed_refusal_items": [item.item_id for item in negatives if not item.refused],
        "latency_seconds": {
            "median": round(statistics.median(latencies), 3),
            "p95": round(latencies_sorted[p95_index], 3),
        },
        "tokens": {
            "prompt": tokens_in,
            "completion": tokens_out,
            "total": tokens_in + tokens_out,
            "mean_per_query": round((tokens_in + tokens_out) / max(1, len(answered))),
        },
    }


def run_ragas_metrics(
    answered: list[AnsweredItem], config: dict[str, Any], provider: str | None, model: str | None
) -> dict[str, Any]:
    """Score faithfulness and answer relevancy with RAGAS.

    Only positives that produced a non-refusal answer are scored — faithfulness
    of a refusal is meaningless, and including refusals would drag the average
    around for reasons unrelated to answer quality.

    Args:
        answered: All answered items.
        config: Parsed `config.yaml`.
        provider: Provider override for the judge.
        model: Model override for the judge.

    Returns:
        A dict of metric name to score, or an "error" key if RAGAS failed.
    """
    scorable = [
        item for item in answered
        if not item.should_refuse and not item.refused and item.contexts
    ]
    if not scorable:
        return {"error": "no scorable items"}

    try:
        from datasets import Dataset
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as error:
        return {"error": f"ragas imports failed: {error}"}

    provider_name = provider or cfg_get(config, "generator.provider", "hf")
    provider_config = cfg_get(config, f"generator.providers.{provider_name}", {})
    import os

    api_key = os.getenv(provider_config.get("api_key_env", ""), "")
    if not api_key:
        return {"error": f"missing API key for provider {provider_name!r}"}

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=model or provider_config["model"],
            base_url=provider_config["base_url"],
            api_key=api_key,
            temperature=0.0,
            timeout=cfg_get(config, "generator.timeout_seconds", 60),
        )
    )
    # answer_relevancy embeds generated questions; reuse the pipeline's own
    # embedding model rather than pulling in a second one.
    embedder = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=cfg_get(config, "embed.model_name"))
    )

    dataset = Dataset.from_dict(
        {
            "question": [item.question for item in scorable],
            "answer": [item.answer for item in scorable],
            "contexts": [item.contexts for item in scorable],
            "ground_truth": [item.expected_answer for item in scorable],
        }
    )

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=judge,
            embeddings=embedder,
        )
    except Exception as error:  # noqa: BLE001 — surface any RAGAS failure, don't crash the run
        return {"error": f"ragas evaluate failed: {type(error).__name__}: {error}"}

    scores = {"scored_items": len(scorable)}
    for key, value in dict(result).items():
        try:
            scores[key] = round(float(value), 4)
        except (TypeError, ValueError):
            scores[key] = value
    return scores


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evals.run_ragas",
        description="Generation-quality evaluation: free answer metrics plus RAGAS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--provider", default=None, help="Override generator.provider.")
    parser.add_argument("--model", default=None, help="Override the generator model.")
    parser.add_argument("--judge-provider", default=None, help="Provider for the RAGAS judge.")
    parser.add_argument("--judge-model", default=None, help="Model for the RAGAS judge.")
    parser.add_argument("--top-k", type=int, default=None, help="Override rag.top_k.")
    parser.add_argument("--limit", type=int, default=None, help="Answer only N items.")
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Answer questions and report the free metrics only. Much cheaper.",
    )
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--tag", default="", help="Label recorded in the results file.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Answer the golden set and score the answers.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if there was nothing to evaluate.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    rag_settings = resolve_rag_settings(config, argparse.Namespace(
        top_k=args.top_k, embed_model=None, user_id=None,
    ))

    eval_dir = Path(cfg_get(config, "paths.eval_dir", "data/eval"))
    golden_path = args.golden or eval_dir / cfg_get(config, "eval.output_filename", "golden.jsonl")
    results_dir = Path(cfg_get(config, "eval.results_dir", "evals/results"))

    items = load_golden(golden_path, verified_only=not args.allow_unverified)
    if args.limit:
        items = items[: args.limit]
    if not items:
        LOGGER.error("No items to evaluate. Verify the golden set or pass --allow-unverified.")
        return 1
    if args.allow_unverified:
        LOGGER.warning("PROVISIONAL RUN — unverified items included.")

    embedder = SentenceTransformer(rag_settings["embed_model"], device=rag_settings["device"])
    generator = build_generator(config, provider=args.provider, model=args.model)
    collection_name = collection_name_for(
        rag_settings["collection_prefix"], rag_settings["embed_model"]
    )

    answered: list[AnsweredItem] = []
    client = QdrantClient(path=rag_settings["qdrant_path"])
    try:
        if not client.collection_exists(collection_name):
            LOGGER.error("Collection %s does not exist. Run embed_index first.", collection_name)
            return 1

        for item in tqdm(items, desc="Answering", unit="q"):
            try:
                result = answer_question(
                    item["question"],
                    client=client,
                    collection_name=collection_name,
                    embedder=embedder,
                    generator=generator,
                    settings=rag_settings,
                )
            except RuntimeError as error:
                LOGGER.error("Item %s failed: %s", item["id"], error)
                continue

            # Compare on doc_id:page — page numbers repeat across documents, so
            # a page-only match credits evidence that came from another policy.
            ground_truth_doc = item.get("ground_truth_doc_id", "")
            ground_truth_refs = (
                {f"{ground_truth_doc}:{page}" for page in item.get("ground_truth_pages") or []}
                if ground_truth_doc
                else set()
            )
            retrieved_refs = {f"{chunk.doc_id}:{chunk.page}" for chunk in result.retrieved}

            answered.append(
                AnsweredItem(
                    item_id=item["id"],
                    question=item["question"],
                    question_type=item.get("question_type", "lookup"),
                    policy_type=item.get("policy_type", ""),
                    expected_answer=item.get("answer", ""),
                    answer=result.answer,
                    contexts=[chunk.text for chunk in result.retrieved],
                    retrieved_pages=[chunk.page for chunk in result.retrieved],
                    had_evidence=bool(ground_truth_refs & retrieved_refs),
                    cited_pages=result.cited_pages,
                    invalid_citations=result.invalid_citations,
                    refused=result.refused,
                    should_refuse=item.get("question_type") == "negative",
                    latency_seconds=result.total_seconds,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
            )
    finally:
        client.close()

    if not answered:
        LOGGER.error("No questions were answered successfully.")
        return 1

    metrics = free_metrics(answered)
    usd_per_million = cfg_get(config, "eval.estimated_usd_per_million_tokens", 0.7)
    metrics["estimated_usd"] = round(metrics["tokens"]["total"] / 1_000_000 * usd_per_million, 4)

    ragas_scores = (
        {"skipped": True}
        if args.skip_ragas
        else run_ragas_metrics(answered, config, args.judge_provider, args.judge_model)
    )

    print("\n=== FREE ANSWER METRICS (0 extra LLM calls) ===")
    print(f"answered            : {len(answered)} ({metrics['positives']} pos / {metrics['negatives']} neg)")
    print(f"citation coverage   : {metrics['citation_coverage']:.3f}")
    print(f"citation validity   : {metrics['citation_validity']:.3f}")
    print(
        f"\npositives split     : {metrics['positives_with_evidence']} with evidence "
        f"/ {metrics['positives_without_evidence']} without"
    )
    print(
        f"false refusal rate  : {metrics['false_refusal_rate']:.3f}  "
        "(refused WITH evidence — the defect)"
    )
    print(
        f"correct refusal rate: {metrics['correct_refusal_rate']:.3f}  "
        "(refused WITHOUT evidence — correct)"
    )
    print(
        f"ungrounded answers  : {metrics['ungrounded_answer_rate']:.3f}  "
        "(answered WITHOUT evidence — inspect)"
    )
    print(f"refusal accuracy    : {metrics['refusal_accuracy']:.3f}  (negatives)")
    print(f"latency median/p95  : {metrics['latency_seconds']['median']} / {metrics['latency_seconds']['p95']} s")
    print(f"tokens per query    : {metrics['tokens']['mean_per_query']}")
    print(f"spend this run      : ~${metrics['estimated_usd']}")

    if metrics["fabricated_citation_items"]:
        print(f"\nFABRICATED CITATIONS: {', '.join(metrics['fabricated_citation_items'])}")
    if metrics["false_refusal_items"]:
        shown = ", ".join(metrics["false_refusal_items"][:15])
        print(f"false refusals      : {shown}")
    if metrics["missed_refusal_items"]:
        print(f"MISSED REFUSALS     : {', '.join(metrics['missed_refusal_items'])}")

    print("\n=== RAGAS (judged) ===")
    if ragas_scores.get("skipped"):
        print("skipped (--skip-ragas)")
    elif "error" in ragas_scores:
        print(f"FAILED: {ragas_scores['error']}")
        print("Free metrics above are still valid.")
    else:
        for key, value in ragas_scores.items():
            print(f"{key:<20}: {value}")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": args.tag,
        "provisional": bool(args.allow_unverified),
        "generator": {"provider": generator.provider, "model": generator.model},
        "top_k": rag_settings["top_k"],
        "free_metrics": metrics,
        "ragas": ragas_scores,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "provisional_" if args.allow_unverified else ""
    suffix = f"_{args.tag}" if args.tag else ""
    results_path = results_dir / f"{prefix}generation_{stamp}{suffix}.json"
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    per_item_path = results_dir / f"{prefix}generation_{stamp}{suffix}_items.jsonl"
    with per_item_path.open("w", encoding="utf-8") as handle:
        for item in answered:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    print(f"\nresults : {results_path}")
    print(f"per-item: {per_item_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
