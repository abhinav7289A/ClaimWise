"""Route a question by nearest labelled exemplar. Zero LLM calls.

**What the router decides.** Not "which pipeline runs" but "what runs *after*
retrieval". CLAUDE.md's motivating question — *"Is my knee surgery covered after
18 months and what will I get back on a ₹2.4L bill?"* — is lookup **and**
calculation, so four mutually exclusive pipelines would be the wrong shape. You
cannot compute a settlement without first retrieving the co-payment rate and its
page, which means `calculation` structurally contains `lookup`. The routes are
therefore additive: retrieval always runs except for `out_of_scope`, and the
route selects which optional stages follow it.

**Why exemplars and not an LLM call.** Routing is a four-way decision with strong
semantic signal, not a subtle judgement. An LLM classifier would tax every single
question with a network round trip — on the current free tier that is seconds,
sometimes tens of seconds — and introduce a non-deterministic failure point
*before* retrieval has even started. Embedding the question against a handful of
labelled examples reuses the bge-small model already loaded for retrieval, costs
~30 ms, and is deterministic.

The decisive argument is measurability. This can be evaluated for nothing, which
is the same property that let Phase 2 reject hybrid search on evidence rather
than intuition. If the accuracy number turns out to be insufficient, that number
is itself the justification for paying for an LLM classifier.

**Why `out_of_scope` is an exemplar class and not a similarity threshold.** The
tempting design is "if nothing scores above X, it is out of scope". P-14 already
measured why that fails: bi-encoder cosine on the 15 hand-seeded unanswerable
questions averaged **0.6687**, sitting squarely inside the range genuine
questions occupy. There is no threshold that separates them. Out-of-scope
questions are recognised by resembling *other out-of-scope questions*, not by
being far from everything.

That is a different signal from the confidence gate, which uses the
**cross-encoder** score (0.0985 on the same negatives, cleanly separated) and
runs after retrieval. Routing asks "what kind of question is this"; the gate asks
"did we actually find the answer". Both are needed and neither substitutes.

**Why exemplars live in this module and not `config.yaml`.** They are not
tunables — they are the semantic definition of each route, closer to code than
configuration, and unreadable as nested YAML. The genuine tunable, the ambiguity
margin, does live in config.

Usage:
    python -m phase3_agents.router --help
    python -m phase3_agents.router -q "What do I get back on a 2.4 lakh bill?"
    python -m phase3_agents.router --eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from phase3_agents.state import Route

LOGGER = logging.getLogger("claimwise.router")

# Labelled examples defining each route. Written to cover the *phrasings a user
# would actually type*, including the vague ones — "will they pay for" is far
# more common than "what is the reimbursement amount".
#
# `out_of_scope` deliberately samples other insurance lines (motor, travel,
# marine, crop) rather than obvious nonsense. A question about car insurance is
# the realistic failure: it is plausible, insurance-shaped, and absent from a
# health/home/life corpus. Weather questions route themselves.
EXEMPLARS: dict[Route, list[str]] = {
    "lookup": [
        "What is the waiting period for pre-existing diseases?",
        "Is maternity covered under this policy?",
        "What are the exclusions for dental treatment?",
        "Does the policy cover AYUSH treatment?",
        "What is the room rent limit?",
        "How do I file a claim?",
        "What documents are needed for reimbursement?",
        "Is day care surgery covered?",
        "What is the grace period for renewal?",
        "Are ambulance charges covered?",
    ],
    "calculation": [
        "What will I get back on a 2.4 lakh hospital bill?",
        "How much will the insurer pay for my surgery?",
        "If my bill is 3 lakhs, how much do I have to pay myself?",
        "What is my out-of-pocket cost after co-payment?",
        "How much is deducted if I take a room above my limit?",
        "Calculate my reimbursement for a 50000 rupee claim",
        "What amount is payable after the deductible?",
        "How much of my 1.5 lakh claim will be settled?",
    ],
    "comparison": [
        "Which of my policies is better for knee surgery?",
        "Compare the room rent limits across my policies",
        "Which policy has a shorter waiting period?",
        "Should I claim under my health or my home policy?",
        "What is the difference between these two policies?",
        "Which of my plans covers maternity better?",
    ],
    "out_of_scope": [
        "What is the premium for car insurance?",
        "Does my travel insurance cover flight cancellation?",
        "How do I insure my crops against drought?",
        "What is marine cargo insurance?",
        "What is the weather forecast for tomorrow?",
        "Can you help me file my income tax return?",
        "What is the stock price of this insurance company?",
        "How do I get a driving licence?",
    ],
}


@dataclass
class RouteDecision:
    """One routing decision, with the evidence behind it.

    Attributes:
        route: The chosen destination.
        score: Cosine similarity to the nearest exemplar of that route.
        margin: Gap to the best-scoring alternative route. A small margin means
            the question sat between two routes and the decision was close.
        nearest_exemplar: The example it matched, so a misroute can be diagnosed
            by reading one line instead of re-running the model.
        ambiguous: Whether `margin` fell below the configured threshold.
        reason: Human-readable summary, written into `AgentState.route_reason`.
    """

    route: Route
    score: float
    margin: float
    nearest_exemplar: str
    ambiguous: bool
    reason: str


class ExemplarRouter:
    """Routes questions by nearest labelled exemplar."""

    def __init__(
        self,
        embedder: SentenceTransformer,
        exemplars: dict[Route, list[str]] | None = None,
        ambiguity_margin: float = 0.05,
    ) -> None:
        """Embed the exemplars once.

        Args:
            embedder: The sentence-transformers model. Passed in rather than
                constructed so the graph shares one instance with retrieval —
                loading bge-small twice would double memory for no benefit.
            exemplars: Route to example questions. Defaults to `EXEMPLARS`.
            ambiguity_margin: Below this gap between the top two routes, the
                decision is flagged ambiguous.

        Raises:
            ValueError: If any route has no exemplars.
        """
        self.exemplars = exemplars or EXEMPLARS
        self.ambiguity_margin = ambiguity_margin

        empty = [route for route, items in self.exemplars.items() if not items]
        if empty:
            raise ValueError(f"Routes with no exemplars: {empty}")

        self._routes: list[Route] = []
        self._texts: list[str] = []
        for route, items in self.exemplars.items():
            for text in items:
                self._routes.append(route)
                self._texts.append(text)

        # No BGE query-instruction prefix here, and that is deliberate. The
        # prefix exists to bridge the query-to-document asymmetry those models
        # are trained with. Both sides of this comparison are questions, so the
        # asymmetry does not apply; adding the same instruction to both would
        # only push every vector in one shared direction.
        self._matrix = embedder.encode(
            self._texts, normalize_embeddings=True, show_progress_bar=False
        )
        self._embedder = embedder
        LOGGER.info(
            "Router ready: %d exemplars across %d routes",
            len(self._texts),
            len(self.exemplars),
        )

    def route(self, question: str) -> RouteDecision:
        """Classify one question.

        Scores each route by its *best* matching exemplar rather than its mean.
        A route is defined by the union of the phrasings it covers, not by their
        average — averaging would penalise a route for having one broad example
        and reward routes whose exemplars are all alike.

        Args:
            question: The user's question.

        Returns:
            The decision and the evidence for it.
        """
        vector = self._embedder.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )[0]
        similarities = self._matrix @ vector

        best_per_route: dict[Route, tuple[float, str]] = {}
        for route, text, score in zip(self._routes, self._texts, similarities):
            score = float(score)
            if route not in best_per_route or score > best_per_route[route][0]:
                best_per_route[route] = (score, text)

        ranked = sorted(best_per_route.items(), key=lambda pair: pair[1][0], reverse=True)
        (winner, (top_score, exemplar)) = ranked[0]
        runner_up_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
        margin = top_score - runner_up_score
        ambiguous = margin < self.ambiguity_margin

        reason = (
            f"nearest exemplar {exemplar!r} (score {top_score:.3f}, "
            f"margin {margin:.3f} over {ranked[1][0] if len(ranked) > 1 else 'none'})"
        )
        if ambiguous:
            reason += " — AMBIGUOUS, margin below threshold"

        return RouteDecision(
            route=winner,
            score=round(top_score, 4),
            margin=round(margin, 4),
            nearest_exemplar=exemplar,
            ambiguous=ambiguous,
            reason=reason,
        )


def build_router(config: dict[str, Any], embedder: SentenceTransformer) -> ExemplarRouter:
    """Construct the configured router.

    Args:
        config: Parsed `config.yaml`.
        embedder: Shared embedding model.

    Returns:
        A ready-to-use router.
    """
    return ExemplarRouter(
        embedder=embedder,
        ambiguity_margin=cfg_get(config, "router.ambiguity_margin", 0.05),
    )


def evaluate(router: ExemplarRouter, golden_path: Path) -> dict[str, Any]:
    """Measure routing accuracy against the golden set's labels.

    **This measures 2 of the 4 routes, and says so.** `golden.jsonl` contains 77
    items labelled `lookup`, 15 labelled `negative`, and 8 positives carrying no
    `question_type` at all. There are no `calculation` or `comparison` labels
    until the Phase 3 50-task agent set exists, so this reports what it can and
    is explicit about the rest rather than implying full coverage.

    Args:
        router: The router under test.
        golden_path: Path to `golden.jsonl`.

    Returns:
        Per-class counts, accuracy, and the misroutes for inspection.

    Raises:
        FileNotFoundError: If the golden set is missing.
    """
    if not golden_path.is_file():
        raise FileNotFoundError(f"Golden set not found: {golden_path.resolve()}")

    with golden_path.open("r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]

    correct = 0
    scored = 0
    unlabelled = 0
    confusion: dict[str, dict[str, int]] = {}
    misroutes: list[dict[str, Any]] = []
    ambiguous = 0
    # Split accuracy by whether the router thought it knew. If confident
    # decisions are reliable, this survives as a cheap pre-filter with an LLM
    # handling only the ambiguous remainder; if they are not, the mechanism is
    # unusable at any confidence and must be replaced outright.
    confident_correct = 0
    confident_total = 0
    ambiguous_correct = 0

    for item in items:
        question_type = item.get("question_type")
        if question_type == "negative":
            expected: Route = "out_of_scope"
        elif question_type == "lookup":
            expected = "lookup"
        else:
            unlabelled += 1
            continue

        decision = router.route(item["question"])
        scored += 1
        ambiguous += decision.ambiguous
        confusion.setdefault(expected, {}).setdefault(decision.route, 0)
        confusion[expected][decision.route] += 1

        hit = decision.route == expected
        if decision.ambiguous:
            ambiguous_correct += hit
        else:
            confident_total += 1
            confident_correct += hit

        if hit:
            correct += 1
        else:
            misroutes.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expected": expected,
                    "got": decision.route,
                    "score": decision.score,
                    "margin": decision.margin,
                    "nearest_exemplar": decision.nearest_exemplar,
                }
            )

    # The number any classifier must beat: always predict the majority class.
    # Quoting accuracy without it is how a classifier worse than a constant gets
    # mistaken for a working one.
    majority = max(
        sum(counts.values()) for counts in confusion.values()
    ) if confusion else 0

    return {
        "scored": scored,
        "unlabelled_skipped": unlabelled,
        "accuracy": round(correct / scored, 4) if scored else None,
        "majority_class_baseline": round(majority / scored, 4) if scored else None,
        "confident_accuracy": (
            round(confident_correct / confident_total, 4) if confident_total else None
        ),
        "confident_decisions": confident_total,
        "ambiguous_accuracy": (
            round(ambiguous_correct / ambiguous, 4) if ambiguous else None
        ),
        "ambiguous_decisions": ambiguous,
        "confusion": confusion,
        "misroutes": misroutes,
        "routes_covered": ["lookup", "out_of_scope"],
        "routes_unmeasured": ["calculation", "comparison"],
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase3_agents.router",
        description="Route a question by nearest labelled exemplar (no LLM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-q", "--question", default=None, help="Route one question.")
    parser.add_argument(
        "--eval", action="store_true", help="Measure accuracy against the golden set."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Route a question or evaluate the router.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if nothing was asked of it.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.question and not args.eval:
        LOGGER.error("Nothing to do. Pass --question or --eval.")
        return 1

    config = load_config(args.config)
    embedder = SentenceTransformer(
        cfg_get(config, "embed.model_name"), device=cfg_get(config, "embed.device", "cpu")
    )
    router = build_router(config, embedder)

    if args.question:
        decision = router.route(args.question)
        print(f"\nQ: {args.question}")
        print(f"route            : {decision.route}")
        print(f"score            : {decision.score}")
        print(f"margin           : {decision.margin}")
        print(f"nearest exemplar : {decision.nearest_exemplar!r}")
        print(f"ambiguous        : {decision.ambiguous}")

    if args.eval:
        eval_dir = Path(cfg_get(config, "paths.eval_dir", "data/eval"))
        golden_path = eval_dir / cfg_get(config, "eval.output_filename", "golden.jsonl")
        report = evaluate(router, golden_path)

        print("\n=== ROUTER ACCURACY ===")
        print(f"scored             : {report['scored']}")
        print(f"unlabelled skipped : {report['unlabelled_skipped']}")
        print(f"accuracy           : {report['accuracy']}")
        print(f"majority baseline  : {report['majority_class_baseline']}  <- must beat this")
        print(
            f"confident          : {report['confident_accuracy']} "
            f"over {report['confident_decisions']} decisions"
        )
        print(
            f"ambiguous          : {report['ambiguous_accuracy']} "
            f"over {report['ambiguous_decisions']} decisions"
        )
        print(f"routes measured    : {report['routes_covered']}")
        print(f"routes UNMEASURED  : {report['routes_unmeasured']}  <- need the 50-task set")

        print("\nconfusion (expected -> got):")
        for expected, got in sorted(report["confusion"].items()):
            for route, count in sorted(got.items(), key=lambda pair: -pair[1]):
                marker = "  " if route == expected else "X "
                print(f"  {marker}{expected:<14} -> {route:<14} {count}")

        if report["misroutes"]:
            print(f"\nmisroutes ({len(report['misroutes'])}):")
            for miss in report["misroutes"][:15]:
                print(f"  {miss['id']} expected {miss['expected']}, got {miss['got']} "
                      f"(margin {miss['margin']})")
                print(f"    {miss['question'][:88]}")
                print(f"    matched: {miss['nearest_exemplar'][:88]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
