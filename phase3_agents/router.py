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

# Labelled examples defining each route.
#
# REVISED 2026-08-18 (D-25). The first set scored lookup recall 0.325 on the
# golden set and 0.500 on the agent set. Diagnosis came from the misroutes'
# `nearest_exemplar` field rather than from reading individual questions: three
# exemplars accounted for 36 of the 52 stolen golden lookups. Each revision below
# fixes a *stated defect in an exemplar*, not a specific failing test item —
# fitting to the failures would tune the router to the eval set rather than to
# the routing problem.
#
# The four principles, each derivable from the exemplar text alone:
#
# P1. LOOKUP MUST COVER THE CORPUS. The original ten were all health questions
#     while the corpus is health, home and life. Home and life lookups had no
#     representative at all and fell to whichever other route was nearest.
#
# P2. COMPARISON EXEMPLARS CARRY COMPARATIVE STRUCTURE, NOT DOMAIN NOUNS.
#     "Should I claim under my health or my home policy?" stole 19 golden
#     lookups by itself: its comparative frame is buried under "claim", "health"
#     and "home policy", which appear in nearly every insurance question. The
#     replacements keep "which of", "compare", "difference between", "both", and
#     drop the domain vocabulary. The procedure name in "knee surgery" went for
#     the same reason — it pulled two calculation questions across.
#
# P3. CALCULATION EXEMPLARS MUST CONTAIN THE USER'S OWN FIGURES. "How much will
#     the insurer pay for my surgery?" stole 10: it is exactly the shape of a
#     *limit lookup* ("how much does the policy pay for an air ambulance?"). The
#     real discriminator is not the words "how much" but whether the user has
#     supplied numbers of their own, so every exemplar here now carries some.
#
# P4. OUT-OF-SCOPE EXEMPLARS LEAD WITH THE OUT-OF-SCOPE DOMAIN. "Does my travel
#     insurance cover flight cancellation?" stole 7, because "does my X cover Y"
#     is the single commonest lookup phrasing in existence and the only
#     out-of-scope signal in it is one adjective. Reworded noun-forward.
#
#     P4 WAS FIRST APPLIED TOO BROADLY, AND THE MEASUREMENT CAUGHT IT. The
#     original exemplars were near-duplicates of `agent_tasks.jsonl`'s
#     out_of_scope items — "What is the premium for car insurance?" against
#     `t-041` "What premium would I pay to insure my Honda City?" — because the
#     same author wrote both, two days apart. The first revision responded by
#     replacing the whole line-set (motor, travel, marine, crop) with disjoint
#     ones (fire, pet, mobile, cyber). Out-of-scope recall then collapsed:
#     golden 0.533 -> 0.067, agent 0.900 -> 0.300.
#
#     The reason is that golden's 15 hand-seeded negatives ARE those lines —
#     four travel, three motor, three marine/cargo, one crop — and golden is not
#     contaminated, having been written in Phase 1 before this module existed.
#     The removed exemplars were carrying real signal; only their overlap with
#     the *agent* set was accidental. So the lines are restored, because "an
#     insurance line we do not hold" is the correct feature for this route, and
#     the contamination is reported as a caveat on the agent-set number instead
#     of being hidden by crippling the exemplars. Golden is the trustworthy
#     measurement for this route; the agent set is not.
#
# One deliberate imbalance, stated because it is a thumb on the scale: a route is
# scored by its single best-matching exemplar, so a route with more exemplars has
# more chances to win. `lookup` is the largest (22 against 8-10) because it is
# the majority class in real traffic and the de-facto fallback — but that choice
# helps it, and any lookup improvement must be read with that in mind.
EXEMPLARS: dict[Route, list[str]] = {
    "lookup": [
        # Health — what the corpus is mostly made of.
        "What is the waiting period for pre-existing diseases?",
        "Is maternity covered under this policy?",
        "What are the exclusions for dental treatment?",
        "Does the policy cover AYUSH treatment?",
        "What is the room rent limit under my plan?",
        "Is day care surgery covered?",
        "What is the grace period for renewal?",
        "Are ambulance charges covered?",
        # A LIMIT lookup. Phrased with "how much" on purpose: the router must
        # learn that "how much does the POLICY pay for X" is reading a stated
        # figure out of the wording, while "how much will I get on MY bill" is
        # arithmetic. Without this, P3's boundary has only one side.
        "How much does the policy pay for an air ambulance?",
        "What is the co-payment percentage in this policy?",
        "Does the policy pay for treatment taken at home?",
        "How long must I wait before a specific illness is covered?",
        # Claims process — procedural lookups, not facts about cover.
        "How do I file a claim?",
        "What documents are needed for reimbursement?",
        "Which number do I call to intimate a claim?",
        "How many days do I have to inform the insurer after admission?",
        # Home (P1) — previously unrepresented.
        "Does my home policy cover theft of contents?",
        "What depreciation applies to a damaged appliance?",
        "Am I covered if a pipe bursts and damages my flat?",
        "Do I need to tell the insurer before renovating my house?",
        # Life (P1) — previously unrepresented.
        "Can I withdraw money from my ULIP before maturity?",
        "What happens to my policy if I stop paying premiums?",
    ],
    "calculation": [
        # P3: every one states figures the user brought, not figures the policy
        # states. That is the feature separating this route from a limit lookup.
        "What will I get back on a 2.4 lakh hospital bill?",
        "If my bill is 3 lakhs, how much do I have to pay myself?",
        "My bill is 1.5 lakh and there is a 10% co-payment — what is payable?",
        "How much is deducted if my room cost 6000 a day against a 4000 limit?",
        "Calculate my reimbursement for a 50000 rupee claim",
        "My bill came to 4.5 lakhs. What amount will the insurer settle?",
        "What is my out-of-pocket cost on a 2 lakh claim after co-payment?",
        "After a 25000 deductible, how much of my 1 lakh bill is paid?",
        "What is payable on a 3 lakh claim if a 1 lakh sub-limit applies?",
        "How much of my 1.5 lakh claim will be settled?",
    ],
    "comparison": [
        # P2, refined after over-applying it. The defect was ONE exemplar whose
        # only content was high-frequency nouns ("claim", "health", "home
        # policy"), which made it a generic insurance attractor. Stripping domain
        # content from ALL of them made the route vague instead and cost it 0.917
        # -> 0.750. These keep an explicit comparative marker — "which of",
        # "compare", "difference", "both", "between" — AND a concrete policy
        # feature to be compared.
        "Which of my two policies has a shorter waiting period?",
        "Compare the room rent limits across my policies",
        "What is the difference between these two policies?",
        "Which of my plans covers maternity better?",
        "Do both my policies apply the same co-payment?",
        "Between my two insurers, which one pays more for the same treatment?",
        "Is the pre-existing disease wait the same on both my policies?",
        "Which policy gives the better no-claim bonus?",
        "Do both policies deduct proportionately for a higher room category?",
        "Which of my policies should I claim under for this?",
    ],
    "out_of_scope": [
        # P4: the out-of-scope signal leads, and the insurance LINES are the
        # load-bearing half — a question about motor or marine cover is the
        # realistic failure, being plausible and insurance-shaped, where a
        # weather question routes itself. Only the travel exemplar is reworded,
        # noun-forward, since it was the one acting as a "does my X cover Y"
        # attractor.
        "What is the premium for car insurance?",
        "Travel insurance for an overseas trip — what does it include?",
        "How do I insure my crops against drought?",
        "What is marine cargo insurance?",
        "What is professional indemnity insurance for a consultancy?",
        "What is workmen's compensation insurance?",
        # REMOVED 2026-08-19 — do not re-add a commercial-property exemplar here
        # without first checking the home policy for the words you plan to use.
        #
        # It was added as "What does a fire insurance policy for a shop cover?",
        # which was categorically wrong: it led with **fire**, a peril the SBI
        # home policy actively covers, and pulled `g-051` (air conditioner fire,
        # home p.6) into a refusal. Reworded to "What does a shopkeeper's policy
        # cover for my business premises?" to lead with the commercial signal
        # instead — which fixed g-051 and immediately took `g-045` in its place
        # ("is my business covered if it's interrupted...", home p.16, which is
        # business-interruption wording). Aggregate accuracy did not move: 0.6522
        # before and after, one victim swapped for another.
        #
        # The rule both versions broke: AN OUT-OF-SCOPE EXEMPLAR MUST NAME
        # SOMETHING THE CORPUS DOES NOT HOLD. The home policy holds both fire
        # perils and business-interruption clauses, so neither phrasing was a
        # clean signal. Commercial cover is still represented on this route by
        # the professional-indemnity and workmen's-compensation exemplars, which
        # name lines the corpus genuinely lacks.
        "Is pet insurance available for my dog?",
        # Outside insurance entirely.
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


def load_labelled(path: Path) -> list[dict[str, Any]]:
    """Read route labels from either eval set.

    Two files carry routing labels and they spell them differently. `golden.jsonl`
    predates the router and encodes the label as `question_type`, where
    `"negative"` means out-of-scope and 8 positives carry no label at all.
    `agent_tasks.jsonl` was built for this purpose and states `route` outright.

    Normalising here rather than in `evaluate` keeps one scoring path for both
    sets, so a number from the agent set and a number from the golden set are
    produced by identical code and differ only in their inputs.

    Args:
        path: Path to `golden.jsonl` or `agent_tasks.jsonl`.

    Returns:
        Items as `{"id", "question", "expected"}`, skipping unlabelled ones.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Labelled set not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    labelled: list[dict[str, Any]] = []
    for row in rows:
        if "route" in row:
            expected = row["route"]
        elif row.get("question_type") == "negative":
            expected = "out_of_scope"
        elif row.get("question_type") == "lookup":
            expected = "lookup"
        else:
            continue
        labelled.append({"id": row["id"], "question": row["question"], "expected": expected})
    return labelled


def evaluate(router: ExemplarRouter, golden_path: Path) -> dict[str, Any]:
    """Measure routing accuracy against a labelled set.

    Coverage depends on which set is passed. `golden.jsonl` labels only `lookup`
    and `out_of_scope` — 2 of the 4 routes — because it was built in Phase 1,
    before routing existed. `agent_tasks.jsonl` labels all four and exists
    specifically to close that gap.

    The majority-class baseline is reported either way and it moves sharply
    between the two: 0.837 on the golden set, which is overwhelmingly lookup, and
    0.28 on the agent set, which is deliberately balanced. **An accuracy from one
    set cannot be compared against a baseline from the other.**

    Args:
        router: The router under test.
        golden_path: Path to `golden.jsonl` or `agent_tasks.jsonl`.

    Returns:
        Per-class counts, accuracy, and the misroutes for inspection.

    Raises:
        FileNotFoundError: If the labelled set is missing.
    """
    items = load_labelled(golden_path)
    unlabelled = 0

    correct = 0
    scored = 0
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
        expected: Route = item["expected"]
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
        # Derived from the data rather than hardcoded. The original constants
        # became wrong the moment a second labelled set existed, and a stale
        # "routes_unmeasured" would keep claiming a gap that had been closed.
        "routes_covered": sorted(confusion),
        "routes_unmeasured": sorted(set(EXEMPLARS) - set(confusion)),
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
        "--eval", action="store_true", help="Measure accuracy against a labelled set."
    )
    parser.add_argument(
        "--tasks",
        action="store_true",
        help="Evaluate against data/eval/agent_tasks.jsonl (all 4 routes) instead of "
        "golden.jsonl (lookup and out_of_scope only).",
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
        golden_path = (
            eval_dir / "agent_tasks.jsonl"
            if args.tasks
            else eval_dir / cfg_get(config, "eval.output_filename", "golden.jsonl")
        )
        report = evaluate(router, golden_path)

        print("\n=== ROUTER ACCURACY ===")
        print(f"set                : {golden_path.name}")
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
