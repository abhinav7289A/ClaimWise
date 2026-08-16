"""Full-context stuffing vs RAG — the Phase 2 blog experiment.

**The claim being tested.** "Long context windows make RAG obsolete — just put
the whole document in the prompt." With a 262K-token window and a corpus of
491,986 characters (~123K tokens), the entire ClaimWise corpus genuinely fits.
So the question is not *can* you, it is *should* you — and that deserves
measurement rather than assertion in either direction.

**What this measures.** The same golden questions through a deliberately naive
pipeline: no retrieval, no reranking, no chunking. Every document, whole, in one
prompt, with the same system contract and the same citation rules the RAG path
uses. Only the retrieval stage differs, so the comparison isolates it.

**Four axes, because "accuracy" alone hides the trade:**

- **Citation validity** — does the model still cite real pages when it can see
  every page? This is where stuffing is expected to struggle: with 102 pages in
  context there is no retrieved set to check a citation against, and "lost in
  the middle" degradation is well documented for long contexts.
- **Tokens per query** — the honest cost axis. RAG sends ~1,500 tokens; stuffing
  sends ~123,000. That is ~80x per question, every question, forever.
- **Latency** — time-to-answer scales with prompt length. An assistant that
  takes a minute per question is a different product.
- **Cost** — the multiplier that decides whether the architecture survives
  contact with a finance team.

**The honest counter-argument, stated up front.** Stuffing has one real
advantage this experiment should surface rather than hide: it cannot suffer a
retrieval miss. Every one of the nine questions Phase 2 never retrieved
correctly is *visible* to a stuffing model. If it answers those correctly, that
is a genuine win for the approach and belongs in the write-up.

**Cost control is mandatory.** A full 100-question run at ~123K tokens each is
~12.3M input tokens. This module prints an estimate and refuses to run the full
set without `--yes`, following the same discipline CLAUDE.md imposes on Modal
training scripts. Default is a 10-question sample, which is enough to establish
the shape of the trade.

Usage:
    python -m phase2_advanced.full_context --help
    python -m phase2_advanced.full_context --estimate-only
    python -m phase2_advanced.full_context --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from common.generator import build_generator
from phase1_rag.rag_chain import CITATION_PATTERN, SYSTEM_PROMPT
from phase2_advanced.parent_docs import load_pages, load_parent_store

LOGGER = logging.getLogger("claimwise.full_context")

# Rough and deliberately stated rather than hidden: English averages ~4 chars
# per token. Used only for the pre-flight estimate, never for a reported metric
# — reported token counts come from the provider's own accounting.
CHARS_PER_TOKEN = 4


@dataclass
class StuffedAnswer:
    """One question answered from the full corpus.

    Attributes:
        item_id: Golden set id.
        question: The question asked.
        answer: The generated answer.
        cited_pages: Pages the model cited.
        ground_truth_pages: Pages the golden set expects.
        cited_correctly: Whether any cited page matches the expected set.
        refused: Whether the model emitted the exact refusal sentence.
        should_refuse: Whether refusal was the correct behaviour.
        prompt_tokens: Input tokens billed.
        completion_tokens: Output tokens billed.
        latency_seconds: Wall-clock time for the call.
    """

    item_id: str
    question: str
    answer: str
    cited_pages: list[int] = field(default_factory=list)
    ground_truth_pages: list[int] = field(default_factory=list)
    cited_correctly: bool = False
    refused: bool = False
    should_refuse: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0


def build_corpus_prompt(pages: list[dict[str, Any]]) -> str:
    """Assemble every page into one prompt block.

    Pages are labelled exactly as the RAG path labels retrieved passages, so the
    model faces an identical citation task and any difference in citation
    behaviour is attributable to context length rather than prompt format.

    Args:
        pages: Page records from `ingest.py`.

    Returns:
        The full corpus as one labelled string.
    """
    return "\n\n".join(
        f"[{page['insurer']} {page['policy_type']} — p.{page['page']}]\n{page['text']}"
        for page in sorted(pages, key=lambda p: (p["filename"], p["page"]))
    )


def load_golden(path: Path, verified_only: bool) -> list[dict[str, Any]]:
    """Read the golden evaluation set.

    Args:
        path: Path to `golden.jsonl`.
        verified_only: Keep only human-verified items.

    Returns:
        Golden items.

    Raises:
        FileNotFoundError: If the golden set does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Golden set not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    return [item for item in items if item.get("verified")] if verified_only else items


def estimate_cost(
    corpus_chars: int,
    question_count: int,
    usd_per_million_input: float,
    usd_per_million_output: float,
    expected_output_tokens: int = 150,
) -> dict[str, Any]:
    """Estimate the spend before any call is made.

    Input and output are priced separately because they differ by 2x on the
    configured provider and, for this workload, are wildly asymmetric: ~124,000
    tokens in against ~150 out. A blended rate applied to a blended total would
    be wrong in both directions at once.

    Args:
        corpus_chars: Length of the assembled corpus prompt.
        question_count: Questions to be asked.
        usd_per_million_input: Provider's input price per million tokens.
        usd_per_million_output: Provider's output price per million tokens.
        expected_output_tokens: Assumed answer length. Deliberately generous —
            an estimate that under-predicts spend is worse than one that
            over-predicts it.

    Returns:
        The estimate, for printing and for recording alongside results.
    """
    tokens_per_query = corpus_chars // CHARS_PER_TOKEN
    total_input = tokens_per_query * question_count
    total_output = expected_output_tokens * question_count
    input_usd = total_input / 1_000_000 * usd_per_million_input
    output_usd = total_output / 1_000_000 * usd_per_million_output
    return {
        "corpus_chars": corpus_chars,
        "estimated_tokens_per_query": tokens_per_query,
        "questions": question_count,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_input_usd": round(input_usd, 4),
        "estimated_output_usd": round(output_usd, 4),
        "estimated_usd": round(input_usd + output_usd, 4),
    }


def answer_from_corpus(
    item: dict[str, Any], corpus: str, generator: Any, refusal_text: str
) -> StuffedAnswer:
    """Answer one question with the entire corpus in the prompt.

    Args:
        item: A golden set item.
        corpus: The assembled full-corpus block.
        generator: The swappable generator.
        refusal_text: Exact refusal sentence, for countable refusals.

    Returns:
        The answer and its accounting.
    """
    prompt = f"{corpus}\n\n---\n\nQuestion: {item['question']}"

    started = time.perf_counter()
    result = generator.generate(prompt, system=SYSTEM_PROMPT)
    latency = time.perf_counter() - started

    # `verify_citations` cannot be reused here: it checks cited pages against
    # the *retrieved* set, and stuffing has no retrieved set — every page is in
    # context, so no citation is ever "fabricated" in that sense. That asymmetry
    # is itself part of the finding: the free citation-validity signal the RAG
    # path gets for nothing is unavailable to a stuffing architecture, so its
    # citations can only be checked against ground truth.
    cited = [int(match) for match in CITATION_PATTERN.findall(result.text)]
    expected = list(item.get("ground_truth_pages") or [])
    return StuffedAnswer(
        item_id=item["id"],
        question=item["question"],
        answer=result.text,
        cited_pages=cited,
        ground_truth_pages=expected,
        cited_correctly=bool(set(cited) & set(expected)),
        refused=refusal_text.lower() in result.text.lower(),
        should_refuse=not expected,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_seconds=round(latency, 3),
    )


def answer_via_rag(
    items: list[dict[str, Any]], config: dict[str, Any], generator: Any, refusal_text: str
) -> list[StuffedAnswer]:
    """Answer the same questions through the configured RAG pipeline.

    Exists so the comparison is honest. `cited_correctly` — did the model cite a
    page the golden set expects — has no counterpart in `run_ragas.py`, which
    measures citation *validity* (were cited pages among those retrieved). Those
    are different questions, and quoting stuffing's 0.30 against the RAG
    pipeline's hit@5 would compare a generation metric with a retrieval one.

    Returning the same dataclass means `summarise()` computes both sides with
    the same code, so the two columns cannot drift apart. The generator is
    shared too, which removes the other confound: stuffing on a paid endpoint
    against RAG on a rate-limited free tier measures the provider's queue, not
    the architecture.

    Args:
        items: The golden items to answer.
        config: Parsed `config.yaml`.
        generator: The same generator used for stuffing.
        refusal_text: Exact refusal sentence.

    Returns:
        One record per item, in the same shape as the stuffed answers.
    """
    import argparse as _argparse

    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    from phase1_rag.embed_index import collection_name_for
    from phase1_rag.rag_chain import answer_question
    from phase1_rag.rag_chain import resolve_settings as resolve_rag_settings

    settings = resolve_rag_settings(
        config, _argparse.Namespace(top_k=None, embed_model=None, user_id=None)
    )
    embedder = SentenceTransformer(settings["embed_model"], device=settings["device"])
    collection_name = collection_name_for(settings["collection_prefix"], settings["embed_model"])

    reranker = None
    if settings["rerank"]:
        from phase2_advanced.rerank import build_reranker

        reranker = build_reranker(config)

    parents = None
    if settings["chunk_policy"]:
        parents = load_parent_store(config, parents_path=Path(settings["parents_path"]))

    LOGGER.info("RAG baseline: collection %s, rerank=%s", collection_name, bool(reranker))

    answers: list[StuffedAnswer] = []
    client = QdrantClient(path=settings["qdrant_path"])
    try:
        for item in tqdm(items, desc="RAG", unit="q"):
            started = time.perf_counter()
            try:
                result = answer_question(
                    item["question"],
                    client=client,
                    collection_name=collection_name,
                    embedder=embedder,
                    generator=generator,
                    settings=settings,
                    reranker=reranker,
                    parents=parents,
                )
            except RuntimeError as error:
                LOGGER.error("Item %s failed: %s", item["id"], error)
                continue

            expected = list(item.get("ground_truth_pages") or [])
            answers.append(
                StuffedAnswer(
                    item_id=item["id"],
                    question=item["question"],
                    answer=result.answer,
                    cited_pages=result.cited_pages,
                    ground_truth_pages=expected,
                    cited_correctly=bool(set(result.cited_pages) & set(expected)),
                    refused=result.refused,
                    should_refuse=not expected,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    latency_seconds=round(time.perf_counter() - started, 3),
                )
            )
    finally:
        client.close()
    return answers


def summarise(answers: list[StuffedAnswer]) -> dict[str, Any]:
    """Aggregate the comparison metrics.

    Args:
        answers: Every answered item.

    Returns:
        Metrics directly comparable with `run_ragas.py`'s free tier.

    Raises:
        ValueError: If nothing was answered.
    """
    if not answers:
        raise ValueError("No answers to summarise.")

    positives = [a for a in answers if not a.should_refuse]
    negatives = [a for a in answers if a.should_refuse]
    latencies = [a.latency_seconds for a in answers]
    prompt_tokens = sum(a.prompt_tokens for a in answers)

    return {
        "answered": len(answers),
        "positives": len(positives),
        "negatives": len(negatives),
        # The headline comparison. Named to match run_ragas so the two files
        # line up in the blog table without translation.
        "cited_correctly_rate": (
            round(sum(a.cited_correctly for a in positives) / len(positives), 4)
            if positives
            else None
        ),
        "false_refusal_rate": (
            round(sum(a.refused for a in positives) / len(positives), 4) if positives else None
        ),
        "refusal_accuracy": (
            round(sum(a.refused for a in negatives) / len(negatives), 4) if negatives else None
        ),
        "tokens": {
            "prompt": prompt_tokens,
            "completion": sum(a.completion_tokens for a in answers),
            "mean_prompt_per_query": round(prompt_tokens / len(answers)),
        },
        "latency_seconds": {
            "median": round(statistics.median(latencies), 3),
            "p95": round(max(latencies) if len(latencies) < 20 else
                         statistics.quantiles(latencies, n=20)[18], 3),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase2_advanced.full_context",
        description="Answer golden questions with the whole corpus in the prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--provider", default=None, help="Override generator.provider.")
    parser.add_argument(
        "--model", default=None, help="Long-context model id. Must fit the whole corpus."
    )
    parser.add_argument("--limit", type=int, default=10, help="Questions to ask.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help=(
            "Per-request timeout. Defaults far above generator.timeout_seconds (60) because "
            "prefill scales with input length and this prompt is ~83x a RAG prompt."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Kept low: re-sending a 124K-token prompt pays the whole prefill again.",
    )
    parser.add_argument(
        "--estimate-only", action="store_true", help="Print the cost estimate and stop."
    )
    parser.add_argument(
        "--rag-baseline",
        action="store_true",
        help=(
            "Also answer the same questions through the configured RAG pipeline, using the "
            "same generator, and print both columns side by side. Adds ~1.5K tokens per "
            "question — negligible next to stuffing's ~121K."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required to run more than the default sample. Guards against a large spend.",
    )
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--tag", default="", help="Label recorded in the results file.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the stuffing experiment and record the comparison.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if the run was declined or blocked.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    processed_dir = Path(cfg_get(config, "paths.processed_dir", "data/processed"))
    pages = load_pages(processed_dir / cfg_get(config, "ingest.output_filename", "pages.jsonl"))
    corpus = build_corpus_prompt(pages)

    eval_dir = Path(cfg_get(config, "paths.eval_dir", "data/eval"))
    golden_path = eval_dir / cfg_get(config, "eval.output_filename", "golden.jsonl")
    items = load_golden(golden_path, verified_only=not args.allow_unverified)
    if args.limit:
        items = items[: args.limit]
    if not items:
        LOGGER.error("No items to evaluate. Pass --allow-unverified for a provisional run.")
        return 1

    # Price the provider that will actually serve the run, not a blended
    # average across providers with a 10x spread between them.
    provider_name = args.provider or cfg_get(config, "generator.provider", "nim")
    provider_config = cfg_get(config, f"generator.providers.{provider_name}", {})
    price_in = provider_config.get("usd_per_million_input", 0.0)
    price_out = provider_config.get("usd_per_million_output", 0.0)
    estimate = estimate_cost(len(corpus), len(items), price_in, price_out)
    estimate["provider"] = provider_name
    estimate["model"] = args.model or provider_config.get("model", "?")

    print("\n=== FULL-CONTEXT STUFFING — PRE-FLIGHT ===")
    print(f"provider/model        : {provider_name}/{estimate['model']}")
    print(f"price per 1M tokens   : ${price_in} in / ${price_out} out")
    print(f"corpus                : {estimate['corpus_chars']:,} chars "
          f"(~{estimate['estimated_tokens_per_query']:,} tokens)")
    print(f"questions             : {estimate['questions']}")
    print(f"estimated input       : {estimate['estimated_input_tokens']:,} tokens "
          f"= ${estimate['estimated_input_usd']}")
    print(f"estimated output      : {estimate['estimated_output_tokens']:,} tokens "
          f"= ${estimate['estimated_output_usd']}")
    print(f"ESTIMATED COST        : ~${estimate['estimated_usd']}")
    if price_in == 0.0:
        print("  (free tier — costs time and rate limits, not money)")
    print(f"RAG sends ~1,475 tokens/query — this is "
          f"~{estimate['estimated_tokens_per_query'] // 1475}x that.")

    if args.estimate_only:
        return 0

    # The guard exists because the failure mode is silent and expensive: a
    # mistyped --limit spends real money before anything is printed.
    if len(items) > 10 and not args.yes:
        LOGGER.error(
            "Refusing to run %d questions without --yes. Estimated cost ~$%s.",
            len(items),
            estimate["estimated_usd"],
        )
        return 1

    generator = build_generator(
        config,
        provider=args.provider,
        model=args.model,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    refusal_text = cfg_get(config, "rag.refusal_text", "")
    print(f"\ngenerator             : {generator.provider}/{generator.model}")
    print(f"timeout               : {args.timeout}s (config default is "
          f"{cfg_get(config, 'generator.timeout_seconds', 60)}s, sized for ~1.5K-token prompts)\n")

    answers: list[StuffedAnswer] = []
    for item in tqdm(items, desc="Stuffing", unit="q"):
        try:
            answers.append(answer_from_corpus(item, corpus, generator, refusal_text))
        except RuntimeError as error:
            # A context-length rejection is a result, not a crash: it is the
            # experiment's answer for models whose window is too small.
            LOGGER.error("Item %s failed: %s", item["id"], error)

    if not answers:
        LOGGER.error("Every call failed — most likely the model's context window is too small.")
        return 1

    metrics = summarise(answers)
    metrics["estimate"] = estimate
    # Billed from the provider's own token accounting, not our 4-chars-per-token
    # approximation. This is the figure that goes in the budget ledger.
    metrics["actual_usd"] = round(
        metrics["tokens"]["prompt"] / 1_000_000 * price_in
        + metrics["tokens"]["completion"] / 1_000_000 * price_out,
        4,
    )

    print("\n=== FULL-CONTEXT RESULTS ===")
    print(f"answered              : {metrics['answered']} "
          f"({metrics['positives']} pos / {metrics['negatives']} neg)")
    print(f"cited correctly       : {metrics['cited_correctly_rate']}")
    print(f"false refusal rate    : {metrics['false_refusal_rate']}")
    print(f"refusal accuracy      : {metrics['refusal_accuracy']}")
    print(f"prompt tokens/query   : {metrics['tokens']['mean_prompt_per_query']:,}")
    print(f"latency median/p95    : {metrics['latency_seconds']['median']} / "
          f"{metrics['latency_seconds']['p95']} s")
    print(f"actual spend          : ~${metrics['actual_usd']}")

    rag_metrics: dict[str, Any] | None = None
    if args.rag_baseline:
        # Guarded, and the guard is not defensive programming for its own sake.
        # The expensive work is already done and paid for by this point. Three
        # separate times in this project a failure *after* a paid call has
        # discarded results that cost real money or an hour of runtime (P-18,
        # twice here). The comparison column is a nice-to-have; the stuffing
        # numbers above are the thing that was bought.
        rag_answers = []
        try:
            rag_answers = answer_via_rag(items, config, generator, refusal_text)
        except Exception as error:  # noqa: BLE001 — never lose paid results to a free extra
            LOGGER.error(
                "RAG baseline failed (%s: %s). Stuffing results below are unaffected "
                "and will still be written.",
                type(error).__name__,
                error,
            )
        if rag_answers:
            rag_metrics = summarise(rag_answers)
            rag_metrics["actual_usd"] = round(
                rag_metrics["tokens"]["prompt"] / 1_000_000 * price_in
                + rag_metrics["tokens"]["completion"] / 1_000_000 * price_out,
                4,
            )

            stuffed_tokens = metrics["tokens"]["mean_prompt_per_query"]
            rag_tokens = rag_metrics["tokens"]["mean_prompt_per_query"]
            print("\n=== STUFFING vs RAG — same questions, same generator ===")
            print(f"{'metric':<24} {'stuffing':>14} {'RAG':>14}")
            print(f"{'cited correctly':<24} {str(metrics['cited_correctly_rate']):>14} "
                  f"{str(rag_metrics['cited_correctly_rate']):>14}")
            print(f"{'prompt tokens/query':<24} {stuffed_tokens:>14,} {rag_tokens:>14,}")
            print(f"{'median latency (s)':<24} "
                  f"{metrics['latency_seconds']['median']:>14} "
                  f"{rag_metrics['latency_seconds']['median']:>14}")
            print(f"{'spend (10 q)':<24} {('$' + str(metrics['actual_usd'])):>14} "
                  f"{('$' + str(rag_metrics['actual_usd'])):>14}")
            if rag_tokens:
                print(f"\nStuffing costs {stuffed_tokens / rag_tokens:.0f}x the tokens per query.")
    else:
        print("\nRe-run with --rag-baseline for a like-for-like comparison; without it,")
        print("these numbers have no counterpart computed the same way.")

    results_dir = Path(cfg_get(config, "eval.results_dir", "evals/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{args.tag}" if args.tag else ""
    prefix = "provisional_" if args.allow_unverified else ""
    results_path = results_dir / f"{prefix}fullcontext_{stamp}{suffix}.json"
    results_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tag": args.tag,
                "provisional": bool(args.allow_unverified),
                "generator": {"provider": generator.provider, "model": generator.model},
                "metrics": metrics,
                "rag_baseline": rag_metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    items_path = results_dir / f"{prefix}fullcontext_{stamp}{suffix}_items.jsonl"
    with items_path.open("w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write(json.dumps(asdict(answer), ensure_ascii=False) + "\n")

    print(f"\nresults : {results_path}")
    print(f"per-item: {items_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
