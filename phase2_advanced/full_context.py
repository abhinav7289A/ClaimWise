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
from phase2_advanced.parent_docs import load_pages

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
    corpus_chars: int, question_count: int, usd_per_million: float
) -> dict[str, Any]:
    """Estimate the spend before any call is made.

    Args:
        corpus_chars: Length of the assembled corpus prompt.
        question_count: Questions to be asked.
        usd_per_million: Configured price per million tokens.

    Returns:
        The estimate, for printing and for recording alongside results.
    """
    tokens_per_query = corpus_chars // CHARS_PER_TOKEN
    total_tokens = tokens_per_query * question_count
    return {
        "corpus_chars": corpus_chars,
        "estimated_tokens_per_query": tokens_per_query,
        "questions": question_count,
        "estimated_total_tokens": total_tokens,
        "estimated_usd": round(total_tokens / 1_000_000 * usd_per_million, 4),
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
        "--estimate-only", action="store_true", help="Print the cost estimate and stop."
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

    usd_per_million = cfg_get(config, "eval.estimated_usd_per_million_tokens", 0.7)
    estimate = estimate_cost(len(corpus), len(items), usd_per_million)

    print("\n=== FULL-CONTEXT STUFFING — PRE-FLIGHT ===")
    print(f"corpus                : {estimate['corpus_chars']:,} chars "
          f"(~{estimate['estimated_tokens_per_query']:,} tokens)")
    print(f"questions             : {estimate['questions']}")
    print(f"estimated total tokens: {estimate['estimated_total_tokens']:,}")
    print(f"estimated cost        : ~${estimate['estimated_usd']}")
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

    generator = build_generator(config, provider=args.provider, model=args.model)
    refusal_text = cfg_get(config, "rag.refusal_text", "")
    print(f"\ngenerator             : {generator.provider}/{generator.model}\n")

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
    metrics["actual_usd"] = round(
        metrics["tokens"]["prompt"] / 1_000_000 * usd_per_million, 4
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
    print("\nCompare against the RAG pipeline's run_ragas.py numbers on the")
    print("same questions. Tokens and latency are the axes that decide this.")

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
