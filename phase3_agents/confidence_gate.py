"""Decide whether the pipeline actually found the answer, before generating.

**The defect this exists to fix.** METRICS §1.7 and §2.5 measure the same thing
twice: when retrieval fails, the generator invents an answer instead of refusing.
Phase 1 — 20 of 26 evidence-less positives answered. Phase 2, after reranking
recovered 15 of those items — 10 of the remaining 11. Reranking shrank the
*population* of the defect without touching the defect. No Phase 2 technique
addresses it, because it is not a retrieval problem: retrieval already knows it
found nothing useful, and that knowledge is thrown away at the prompt boundary.

**Why a gate and not a better prompt.** The system prompt already instructs the
model to refuse when the context lacks the answer. It complies 23% of the time.
Asking a hosted free model more firmly is not a control — it is a request. The
gate is a control: a deterministic branch that refuses *without calling the
model at all*, so refusal cannot be overridden by a generator having a bad day.
This is the standard production shape — an insurer would call it a
straight-through-processing threshold, the score above which a claim is decided
automatically and below which it goes to a human.

**Why the cross-encoder score and not the bi-encoder's.** P-14 measured both on
the 15 hand-seeded unanswerable questions:

    bi-encoder cosine  0.6687  — sits inside the range genuine questions occupy
    cross-encoder      0.0985  — 14/15 below 0.22, median 0.017

Cosine similarity is trained to place *related* text nearby, and an unanswerable
insurance question is related to insurance prose. It cannot express "close, but
this passage does not contain your answer". A cross-encoder reads question and
passage jointly and is trained on exactly that judgement, which is why its score
separates where cosine does not. The gate therefore runs **after** reranking,
and its input is the rank-1 reranked score.

**Why this is a different signal from the router's.** The router asks "what kind
of question is this" and must answer before retrieval runs. The gate asks "did
we actually find it" and can only answer after. `out_of_scope` and low
confidence overlap but are not the same set: a perfectly in-scope question about
a clause the corpus does not contain is `lookup` and ungrounded at once. Both
mechanisms are needed and neither substitutes for the other.

**Three verdicts, not two.** CLAUDE.md Phase 3 task 5 is "confidence gate +
human-escalation node", and collapsing them loses the interesting middle. Below
`refuse_below` the evidence is absent and the honest output is the refusal
sentence. Above `escalate_below` the evidence is strong and the answer stands on
its own. Between them the pipeline found something plausible but weak — that
answer is still worth generating, but it is exactly the band a human should see,
and in Phase 5 it is what the monitoring dashboard filters on.

**What is measured here and what is not.** `--sweep` re-analyses a recorded
retrieval run rather than re-running retrieval. That is deliberate: the sweep
cannot drift from the pipeline whose numbers are in METRICS.md, it costs
nothing, and it is exactly reproducible from a file in the repo. It measures the
gate's *decision*, not the generation that follows it — whether refusing at
threshold t would have caught the ungrounded answers, and what it would have
cost in false refusals. The generation-side confirmation needs a paid run and is
a separate, later step.

Usage:
    python -m phase3_agents.confidence_gate --help
    python -m phase3_agents.confidence_gate --self-test
    python -m phase3_agents.confidence_gate --sweep
    python -m phase3_agents.confidence_gate --sweep --items evals/results/<run>_items.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config

LOGGER = logging.getLogger("claimwise.confidence_gate")

# What the gate decided to do with the retrieved evidence.
Verdict = Literal["answer", "escalate", "refuse"]

# Thresholds swept when no explicit grid is given. Dense below 0.30 because that
# is where P-14 put the decision boundary, and the shape of the trade-off there
# is what the sweep exists to reveal; coarse above it, where the only question
# is how fast false refusals grow.
DEFAULT_GRID: tuple[float, ...] = (
    0.00, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
)

# The band the decision actually lives in, once measured. The negatives cluster
# below 0.04 and the lowest-scoring genuine hit sits at 0.0414, so the whole
# usable range is narrower than the default grid's first step.
FINE_GRID: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.08, 0.10,
)


@dataclass(frozen=True)
class GateDecision:
    """One gate decision, with the evidence behind it.

    Attributes:
        verdict: What to do next — generate, generate-and-flag, or refuse.
        confidence: The rank-1 cross-encoder score the decision was made on.
            0.0 when nothing was retrieved at all.
        refuse_below: The threshold in force, recorded so a decision in a log can
            be re-explained months later without knowing which config produced it.
        escalate_below: The upper threshold in force.
        reason: Human-readable summary, written into `AgentState.route_reason`'s
            sibling fields and surfaced by the Phase 5 dashboard.
    """

    verdict: Verdict
    confidence: float
    refuse_below: float
    escalate_below: float
    reason: str

    @property
    def refused(self) -> bool:
        """Whether this decision short-circuits generation."""
        return self.verdict == "refuse"

    @property
    def escalate(self) -> bool:
        """Whether this answer should be flagged for a human."""
        return self.verdict == "escalate"


class ConfidenceGate:
    """Turns a rank-1 relevance score into a refuse / escalate / answer verdict."""

    def __init__(self, refuse_below: float, escalate_below: float) -> None:
        """Configure the two thresholds.

        Args:
            refuse_below: Confidence under which the pipeline refuses without
                calling the generator.
            escalate_below: Confidence under which an answer is generated but
                flagged for human review.

        Raises:
            ValueError: If the thresholds are inverted. An escalation band that
                sits below the refusal line is unreachable, and a silently
                unreachable branch is worse than a crash — it would look like a
                working escalation path that never fires.
        """
        if escalate_below < refuse_below:
            raise ValueError(
                f"escalate_below ({escalate_below}) must be >= refuse_below "
                f"({refuse_below}); otherwise the escalation band is empty."
            )
        self.refuse_below = refuse_below
        self.escalate_below = escalate_below

    def decide(self, retrieved: list[Any]) -> GateDecision:
        """Judge whether the retrieved evidence supports answering.

        Reads `score` off the rank-1 item via `getattr`, so this works on
        `RetrievedChunk` without importing `phase1_rag` — the gate stays a
        contract over "something ranked with a score", which is also what lets
        the sweep replay recorded runs through the identical code path.

        The caller is responsible for passing a *reranked* list. Handing it raw
        bi-encoder output would not raise; it would quietly gate on the one
        score P-14 measured as unusable, which is why every call site in this
        project passes the cross-encoder's output.

        Args:
            retrieved: Retrieved chunks, best first, after reranking.

        Returns:
            The verdict and the evidence for it.
        """
        if not retrieved:
            return GateDecision(
                verdict="refuse",
                confidence=0.0,
                refuse_below=self.refuse_below,
                escalate_below=self.escalate_below,
                reason="nothing retrieved — refusing without generating",
            )

        confidence = float(getattr(retrieved[0], "score", 0.0))
        return self._verdict_for(confidence)

    def decide_on_score(self, confidence: float) -> GateDecision:
        """Judge from a bare score, for replaying recorded runs.

        Args:
            confidence: A rank-1 cross-encoder score.

        Returns:
            The verdict and the evidence for it.
        """
        return self._verdict_for(float(confidence))

    def _verdict_for(self, confidence: float) -> GateDecision:
        """Apply the two thresholds to one score.

        Args:
            confidence: A rank-1 cross-encoder score.

        Returns:
            The verdict and the evidence for it.
        """
        if confidence < self.refuse_below:
            verdict: Verdict = "refuse"
            reason = (
                f"top-1 relevance {confidence:.4f} below refusal threshold "
                f"{self.refuse_below} — no passage supports an answer"
            )
        elif confidence < self.escalate_below:
            verdict = "escalate"
            reason = (
                f"top-1 relevance {confidence:.4f} in the weak band "
                f"[{self.refuse_below}, {self.escalate_below}) — answering, "
                f"flagged for review"
            )
        else:
            verdict = "answer"
            reason = (
                f"top-1 relevance {confidence:.4f} at or above "
                f"{self.escalate_below} — evidence is strong"
            )

        return GateDecision(
            verdict=verdict,
            confidence=round(confidence, 4),
            refuse_below=self.refuse_below,
            escalate_below=self.escalate_below,
            reason=reason,
        )


def build_gate(config: dict[str, Any]) -> ConfidenceGate:
    """Construct the configured gate.

    Args:
        config: Parsed `config.yaml`.

    Returns:
        A ready-to-use gate.
    """
    return ConfidenceGate(
        refuse_below=cfg_get(config, "confidence_gate.refuse_below", 0.20),
        escalate_below=cfg_get(config, "confidence_gate.escalate_below", 0.50),
    )


# Routes whose answer does not come from the passages alone. On these the gate
# downgrades a refusal to an escalation instead of ending the run.
#
# **Why this exists.** The gate scores top-1 cross-encoder relevance, which
# measures how much a passage *looks like* the question. A calculation question
# is mostly rupee figures and reads as unlike policy prose, so it scores low even
# when the right page was fetched. Measured 2026-08-22 on the 14-task
# calculation subset: t-015, t-020 and t-024 died here at `steps: 3` with the
# calculator never invoked, and **t-024 had `grounded: true` and
# `docs_covered: true`** — the evidence was retrieved and the gate discarded it.
#
# A calculation's figure is produced by deterministic Python from terms that may
# come from the question itself (t-018 needs no retrieval at all), so "no passage
# looks like this question" is not evidence that the question is unanswerable.
# The refusal is still reported as an escalation, so nothing is hidden.
#
# This deliberately does NOT change lookup or out_of_scope, which is where
# D-23's 15/15 refusal accuracy was measured.
TOOL_BACKED_ROUTES = frozenset({"calculation", "comparison"})


def gate_node(
    state: dict[str, Any],
    gate: ConfidenceGate,
    refusal_text: str,
    tool_backed_routes: frozenset[str] = TOOL_BACKED_ROUTES,
) -> dict[str, Any]:
    """Apply the gate as a graph node.

    Returns a partial state update rather than a mutated state, which is the
    shape LangGraph merges — writing it this way now means the node drops into
    the graph in Phase 3 task 3 without a rewrite.

    Args:
        state: The current `AgentState`.
        gate: The configured gate.
        refusal_text: The exact refusal sentence from config. Fixed wording is
            what makes refusals countable without an LLM judge.
        tool_backed_routes: Routes on which a refusal is downgraded to an
            escalation, because a deterministic tool can still answer.

    Returns:
        Keys to merge into the state.
    """
    retrieved = state.get("retrieved") or []
    decision = gate.decide(retrieved)
    update: dict[str, Any] = {
        "confidence": decision.confidence,
        "trace": [*state.get("trace", []), "confidence_gate"],
    }

    # Nothing retrieved at all is still a refusal on every route: a tool needs
    # terms, and with no passages there is neither evidence nor a fallback.
    downgrade = (
        decision.refused
        and bool(retrieved)
        and state.get("route") in tool_backed_routes
    )

    if decision.refused and not downgrade:
        update["answer"] = refusal_text
        update["refused"] = True
        update["escalate"] = True
        update["escalation_reason"] = decision.reason
    elif downgrade:
        update["escalate"] = True
        update["escalation_reason"] = (
            f"{decision.reason}; continuing anyway because the "
            f"{state.get('route')} route has a deterministic tool"
        )
    elif decision.escalate:
        update["escalate"] = True
        update["escalation_reason"] = decision.reason

    return update


# --- Measurement -------------------------------------------------------------


def expected_action(item: dict[str, Any], evidence_k: int) -> Literal["answer", "refuse"]:
    """The action a perfect gate would take on one recorded golden item.

    Three cases, and the middle one is the whole point of this module:

    - A **negative** is unanswerable by construction. Refusing is correct.
    - A **positive whose correct page reached the prompt** is answerable.
      Refusing it is a false refusal, and false refusals are the price of the
      gate — a product that refuses questions it could have answered is a
      product nobody uses.
    - A **positive whose correct page did NOT reach the prompt** is, at
      inference time, indistinguishable from a negative: the generator is
      holding passages that do not contain the answer. Answering it is the
      hallucination measured in §1.7. Refusing is correct even though the
      question itself was fair.

    Args:
        item: One row from a `*_items.jsonl` retrieval results file.
        evidence_k: How many chunks reach the prompt — `rag.top_k`. A correct
            page retrieved at rank 8 is not evidence if only 5 are prompted.

    Returns:
        "answer" or "refuse".
    """
    if item.get("question_type") == "negative":
        return "refuse"
    page_rank = item.get("page_rank")
    if page_rank is not None and page_rank <= evidence_k:
        return "answer"
    return "refuse"


def score_threshold(
    items: list[dict[str, Any]], threshold: float, evidence_k: int
) -> dict[str, Any]:
    """Measure one refusal threshold against a recorded retrieval run.

    Only `refuse_below` is measured. The escalation band changes no verdict's
    answer/refuse character — it only tags answers for review — so it cannot be
    scored against a golden set that records answerability rather than
    reviewability. Reporting an accuracy for it would be inventing a metric.

    Args:
        items: Rows from a `*_items.jsonl` retrieval results file.
        threshold: The `refuse_below` value under test.
        evidence_k: How many chunks reach the prompt.

    Returns:
        Counts and rates for this threshold.
    """
    gate = ConfidenceGate(refuse_below=threshold, escalate_below=threshold)

    negatives_refused = 0
    negatives_total = 0
    blind_refused = 0          # positives whose evidence never reached the prompt
    blind_total = 0
    grounded_refused = 0       # positives that DID have their evidence
    grounded_total = 0

    for item in items:
        refused = gate.decide_on_score(item.get("top_score", 0.0)).refused
        expected = expected_action(item, evidence_k)

        if item.get("question_type") == "negative":
            negatives_total += 1
            negatives_refused += refused
        elif expected == "refuse":
            blind_total += 1
            blind_refused += refused
        else:
            grounded_total += 1
            grounded_refused += refused

    correct = negatives_refused + blind_refused + (grounded_total - grounded_refused)
    total = negatives_total + blind_total + grounded_total

    def rate(numerator: int, denominator: int) -> float | None:
        """Guard division so an empty class reports None, never a fake 0.0."""
        return round(numerator / denominator, 4) if denominator else None

    return {
        "threshold": threshold,
        # The win: ungrounded answers the gate would have stopped.
        "hallucinations_prevented": blind_refused,
        "hallucinations_possible": blind_total,
        "prevention_rate": rate(blind_refused, blind_total),
        # The price: answerable questions the gate would have refused.
        "false_refusals": grounded_refused,
        "grounded_total": grounded_total,
        "false_refusal_rate": rate(grounded_refused, grounded_total),
        # The existing metric, now produced deterministically instead of by a
        # generator that complied 93% of the time.
        "negatives_refused": negatives_refused,
        "negatives_total": negatives_total,
        "refusal_accuracy": rate(negatives_refused, negatives_total),
        # Net items moved from wrong to right. The single number that says
        # whether this threshold is worth adopting at all.
        "net": blind_refused + negatives_refused - grounded_refused,
        "accuracy": rate(correct, total),
        "scored": total,
    }


def score_distribution(items: list[dict[str, Any]], evidence_k: int) -> dict[str, Any]:
    """Summarise top-1 scores per class, to show whether they separate at all.

    This is the sanity check that must be read *before* any threshold table.
    A sweep will always name a best threshold; only the distributions say
    whether a threshold can work. If grounded positives and negatives overlap
    heavily, no cut point exists and the mechanism should be rejected the way
    the exemplar router was.

    Args:
        items: Rows from a `*_items.jsonl` retrieval results file.
        evidence_k: How many chunks reach the prompt.

    Returns:
        Per-class count, mean, median, min and max of `top_score`.
    """
    buckets: dict[str, list[float]] = {"grounded": [], "blind": [], "negative": []}
    for item in items:
        score = float(item.get("top_score", 0.0))
        if item.get("question_type") == "negative":
            buckets["negative"].append(score)
        elif expected_action(item, evidence_k) == "refuse":
            buckets["blind"].append(score)
        else:
            buckets["grounded"].append(score)

    summary: dict[str, Any] = {}
    for name, scores in buckets.items():
        summary[name] = {
            "n": len(scores),
            "mean": round(mean(scores), 4) if scores else None,
            "median": round(median(scores), 4) if scores else None,
            "min": round(min(scores), 4) if scores else None,
            "max": round(max(scores), 4) if scores else None,
        }
    return summary


def sweep(
    items: list[dict[str, Any]], grid: tuple[float, ...], evidence_k: int
) -> dict[str, Any]:
    """Measure every candidate threshold against a recorded retrieval run.

    Args:
        items: Rows from a `*_items.jsonl` retrieval results file.
        grid: Thresholds to test.
        evidence_k: How many chunks reach the prompt.

    Returns:
        The distributions, the per-threshold rows, and the baseline to beat.
    """
    rows = [score_threshold(items, threshold, evidence_k) for threshold in grid]

    # The number any gate must beat: never refuse anything, which is what the
    # pipeline does today. Quoting gate accuracy without it is how a mechanism
    # that is worse than doing nothing gets adopted (see D-22).
    always_answer = score_threshold(items, threshold=-1.0, evidence_k=evidence_k)

    return {
        "evidence_k": evidence_k,
        "distribution": score_distribution(items, evidence_k),
        "baseline_always_answer": always_answer,
        "thresholds": rows,
    }


def compare_to_generator(
    items: list[dict[str, Any]],
    generation: dict[str, dict[str, Any]],
    grid: tuple[float, ...],
    evidence_k: int,
) -> list[dict[str, Any]]:
    """Measure the gate as an addition to the generator, not as a replacement.

    The sweep's "never refuse" baseline is the right control for the gate in
    isolation, but it is not what the system does today: the generator already
    refuses on its own, imperfectly. The gate runs *before* it, so the two
    compose — the deployed behaviour is "refused if either refuses", and the
    gate's true cost is only the false refusals the generator would not have
    made anyway.

    Measured without spending anything by joining a recorded retrieval run to a
    recorded generation run over the same items.

    Args:
        items: Rows from a retrieval `*_items.jsonl`.
        generation: Generation rows keyed by `item_id`.
        grid: Thresholds to test.
        evidence_k: How many chunks reach the prompt.

    Returns:
        One row per threshold, over the items present in both runs.
    """
    joined = [item for item in items if item.get("item_id") in generation]
    rows: list[dict[str, Any]] = []

    for threshold in grid:
        gate = ConfidenceGate(refuse_below=threshold, escalate_below=threshold)
        negatives = [0, 0, 0]      # generator-only, combined, total
        blind = [0, 0, 0]
        new_false_refusals = 0
        grounded_total = 0
        generator_calls_saved = 0

        for item in joined:
            refused_by_gate = gate.decide_on_score(item.get("top_score", 0.0)).refused
            refused_by_generator = bool(generation[item["item_id"]].get("refused"))
            generator_calls_saved += refused_by_gate

            if item.get("question_type") == "negative":
                negatives[0] += refused_by_generator
                negatives[1] += refused_by_generator or refused_by_gate
                negatives[2] += 1
            elif expected_action(item, evidence_k) == "refuse":
                blind[0] += refused_by_generator
                blind[1] += refused_by_generator or refused_by_gate
                blind[2] += 1
            else:
                grounded_total += 1
                # Only counted when the gate refuses something the generator
                # would have answered. A question the generator was going to
                # refuse anyway is not a cost the gate introduced.
                new_false_refusals += refused_by_gate and not refused_by_generator

        rows.append(
            {
                "threshold": threshold,
                "negatives_generator_only": negatives[0],
                "negatives_combined": negatives[1],
                "negatives_total": negatives[2],
                "blind_generator_only": blind[0],
                "blind_combined": blind[1],
                "blind_total": blind[2],
                "new_false_refusals": new_false_refusals,
                "grounded_total": grounded_total,
                "generator_calls_saved": generator_calls_saved,
                "joined_items": len(joined),
            }
        )
    return rows


def load_generation_items(path: Path) -> dict[str, dict[str, Any]]:
    """Read a generation results items file, keyed by item id.

    Args:
        path: Path to a generation `*_items.jsonl`.

    Returns:
        Generation rows by `item_id`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Generation items file not found: {path.resolve()}. "
            "Run `python -m evals.run_ragas` first, or omit --vs-generator."
        )
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["item_id"]: row for row in rows}


def load_items(path: Path) -> list[dict[str, Any]]:
    """Read a retrieval results items file.

    Args:
        path: Path to a `*_items.jsonl` produced by `evals.retrieval_metrics`.

    Returns:
        The recorded per-item retrieval outcomes.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the file predates `top_score`, which the gate cannot work
            without. Failing loudly beats gating every item on a default 0.0.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Retrieval items file not found: {path.resolve()}. "
            "Run `python -m evals.retrieval_metrics` first."
        )
    with path.open("r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]

    missing = [item.get("item_id", "?") for item in items if "top_score" not in item]
    if missing:
        raise KeyError(
            f"{len(missing)} item(s) have no 'top_score' (first: {missing[0]}). "
            "This file predates the confidence gate; re-run evals.retrieval_metrics."
        )
    return items


def latest_items_file(results_dir: Path, pattern: str) -> Path:
    """Find the most recent retrieval items file.

    Sorts on the **timestamp parsed out of the filename**, not on the filename
    itself. Sorting lexically looks equivalent and is not: results carry two
    prefixes in this repo (`retrieval_...` and `provisional_retrieval_...`), and
    `"r" > "p"`, so a plain sort returns the oldest run as the newest. That is
    not hypothetical — it silently fed this module the Phase 1 dense baseline on
    its first run, and the swept "result" was a measurement of bi-encoder cosine
    that the gate is specifically designed not to use.

    File mtime would fix the ordering and break something else: a cloned repo
    has every mtime set to checkout time. The timestamp in the name is the only
    field that travels with the data.

    Args:
        results_dir: Directory holding eval results.
        pattern: Glob for candidate files.

    Returns:
        The matching file with the newest embedded timestamp.

    Raises:
        FileNotFoundError: If nothing matches, or nothing matching carries a
            parseable timestamp.
    """
    stamp = re.compile(r"(\d{8}T\d{6}Z)")
    dated: list[tuple[str, Path]] = []
    for candidate in results_dir.glob(pattern):
        found = stamp.search(candidate.name)
        if found:
            dated.append((found.group(1), candidate))
        else:
            LOGGER.warning("Ignoring %s — no timestamp in the filename.", candidate.name)

    if not dated:
        raise FileNotFoundError(
            f"No timestamped file matching {pattern!r} in {results_dir.resolve()}. "
            "Run `python -m evals.retrieval_metrics` first, or pass --items."
        )
    return max(dated)[1]


def load_run_summary(items_path: Path) -> dict[str, Any]:
    """Read the summary JSON sitting beside an items file.

    Args:
        items_path: Path to a `*_items.jsonl`.

    Returns:
        The parsed summary, or an empty dict if there is none.
    """
    summary_path = items_path.with_name(items_path.name.replace("_items.jsonl", ".json"))
    if not summary_path.is_file():
        LOGGER.warning("No summary beside %s — cannot verify the pipeline.", items_path.name)
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_reranked(summary: dict[str, Any], items_path: Path) -> None:
    """Refuse to sweep a run whose scores are not cross-encoder scores.

    The gate's entire premise is P-14's measurement that bi-encoder cosine does
    not separate answerable from unanswerable questions while the cross-encoder
    does. `top_score` is populated either way, so a non-reranked run sweeps
    perfectly happily and produces a table of nonsense that looks exactly like a
    real result. This check is the difference between the two.

    Args:
        summary: The run summary.
        items_path: The items file, for the error message.

    Raises:
        ValueError: If the run was not reranked.
    """
    if summary.get("rerank") is True:
        return
    raise ValueError(
        f"{items_path.name} was NOT produced with reranking (rerank="
        f"{summary.get('rerank')!r}), so its top_score is bi-encoder cosine, not a "
        "cross-encoder score. P-14 measured cosine at 0.6687 on the unanswerable "
        "questions — inside the range genuine ones occupy — so no threshold over it "
        "can work. Re-run `python -m evals.retrieval_metrics` with reranking, or pass "
        "--allow-unreranked to reproduce that negative result deliberately."
    )


# --- Self-test ---------------------------------------------------------------


def _self_test() -> list[tuple[str, bool, str]]:
    """Exercise the verdict boundaries and the measurement arithmetic.

    Boundary conditions get their own cases because `<` versus `<=` on a
    threshold silently moves every item sitting exactly on it, and the P-14
    negatives cluster near 0.

    Returns:
        One (name, passed, detail) triple per case.
    """
    results: list[tuple[str, bool, str]] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        """Record one comparison."""
        results.append((name, actual == expected, f"got {actual!r}, want {expected!r}"))

    gate = ConfidenceGate(refuse_below=0.20, escalate_below=0.50)

    check("far below refuses", gate.decide_on_score(0.01).verdict, "refuse")
    check("just below refuses", gate.decide_on_score(0.1999).verdict, "refuse")
    # Exactly on the threshold answers: `refuse_below` reads as "refuse strictly
    # below this", so the threshold itself is the first acceptable score.
    check("exactly at refuse_below escalates", gate.decide_on_score(0.20).verdict, "escalate")
    check("mid band escalates", gate.decide_on_score(0.35).verdict, "escalate")
    check("exactly at escalate_below answers", gate.decide_on_score(0.50).verdict, "answer")
    check("high answers", gate.decide_on_score(0.97).verdict, "answer")
    check("negative score refuses", gate.decide_on_score(-4.2).verdict, "refuse")

    check("empty retrieval refuses", gate.decide([]).verdict, "refuse")
    check("empty retrieval confidence 0", gate.decide([]).confidence, 0.0)

    inverted_raised = False
    try:
        ConfidenceGate(refuse_below=0.60, escalate_below=0.10)
    except ValueError:
        inverted_raised = True
    check("inverted thresholds raise", inverted_raised, True)

    # A gate that refuses everything and one that refuses nothing bracket the
    # measurement: prevention and false refusals must be total or zero.
    fixture = [
        {"item_id": "t-1", "question_type": "lookup", "page_rank": 1, "top_score": 0.90},
        {"item_id": "t-2", "question_type": "lookup", "page_rank": 9, "top_score": 0.05},
        {"item_id": "t-3", "question_type": "lookup", "page_rank": None, "top_score": 0.02},
        {"item_id": "t-4", "question_type": "negative", "page_rank": None, "top_score": 0.01},
    ]

    # page_rank 9 is beyond a top-5 prompt, so t-2 is blind despite being found.
    check("rank beyond k counts as blind", expected_action(fixture[1], 5), "refuse")
    check("rank within k counts as grounded", expected_action(fixture[0], 5), "answer")
    check("rank 9 grounded at k=10", expected_action(fixture[1], 10), "answer")

    refuse_all = score_threshold(fixture, threshold=1.0, evidence_k=5)
    check("refuse-all prevents both", refuse_all["hallucinations_prevented"], 2)
    check("refuse-all refuses the negative", refuse_all["negatives_refused"], 1)
    check("refuse-all costs one false refusal", refuse_all["false_refusals"], 1)
    check("refuse-all net", refuse_all["net"], 2)

    answer_all = score_threshold(fixture, threshold=-1.0, evidence_k=5)
    check("answer-all prevents none", answer_all["hallucinations_prevented"], 0)
    check("answer-all costs nothing", answer_all["false_refusals"], 0)
    check("answer-all accuracy is the baseline", answer_all["accuracy"], 0.25)

    tuned = score_threshold(fixture, threshold=0.10, evidence_k=5)
    check("tuned prevents both", tuned["hallucinations_prevented"], 2)
    check("tuned keeps the answerable one", tuned["false_refusals"], 0)
    check("tuned accuracy is perfect", tuned["accuracy"], 1.0)

    distribution = score_distribution(fixture, evidence_k=5)
    check("distribution counts grounded", distribution["grounded"]["n"], 1)
    check("distribution counts blind", distribution["blind"]["n"], 2)
    check("distribution counts negatives", distribution["negative"]["n"], 1)

    # --- Route-aware refusal ------------------------------------------------
    #
    # Measured 2026-08-22: t-015, t-020 and t-024 died at the gate with the
    # calculator never invoked, and t-024 had both `grounded` and `docs_covered`
    # true. The gate scores how much a passage looks like the question, which a
    # rupee-heavy calculation question fails even when the right page was found.

    class _Scored:
        """Minimal stand-in for a reranked chunk."""

        def __init__(self, score: float) -> None:
            self.score = score

    weak = [_Scored(0.001)]
    strong = [_Scored(0.9)]
    node_gate = ConfidenceGate(refuse_below=0.02, escalate_below=0.50)

    lookup = gate_node({"retrieved": weak, "route": "lookup"}, node_gate, "REFUSED")
    check("lookup still refuses on weak evidence", lookup.get("refused"), True)
    check("and answers with the refusal text", lookup.get("answer"), "REFUSED")

    calc = gate_node({"retrieved": weak, "route": "calculation"}, node_gate, "REFUSED")
    check("calculation is not refused at the gate", calc.get("refused"), None)
    check("no refusal text is written", calc.get("answer"), None)
    check("but it is escalated", calc.get("escalate"), True)
    results.append(
        (
            "and the reason names the tool",
            "deterministic tool" in str(calc.get("escalation_reason")),
            str(calc.get("escalation_reason")),
        )
    )

    comparison = gate_node({"retrieved": weak, "route": "comparison"}, node_gate, "REFUSED")
    check("comparison is not refused either", comparison.get("refused"), None)

    # Nothing retrieved is still a refusal everywhere: a tool needs terms, and
    # with no passages there is neither evidence nor a fallback.
    empty = gate_node({"retrieved": [], "route": "calculation"}, node_gate, "REFUSED")
    check("empty retrieval refuses even on a tool route", empty.get("refused"), True)

    # A strong score is unaffected by any of this.
    fine = gate_node({"retrieved": strong, "route": "calculation"}, node_gate, "REFUSED")
    check("strong evidence is not escalated", fine.get("escalate"), None)

    return results


def _print_sweep(report: dict[str, Any]) -> None:
    """Print the sweep as the tables METRICS.md expects.

    Args:
        report: Output of `sweep`.
    """
    distribution = report["distribution"]
    print("\n=== TOP-1 CROSS-ENCODER SCORE BY CLASS ===")
    print("Read this BEFORE the threshold table. If these overlap, no cut exists.\n")
    print(f"{'class':<10} {'n':>4} {'mean':>9} {'median':>9} {'min':>9} {'max':>9}")
    for name in ("grounded", "blind", "negative"):
        stats = distribution[name]
        row = f"{name:<10} {stats['n']:>4}"
        for key in ("mean", "median", "min", "max"):
            value = stats[key]
            row += f" {value:>9.4f}" if value is not None else f" {'—':>9}"
        print(row)
    print("\n  grounded = correct page reached the prompt (answering is right)")
    print("  blind    = positive whose correct page did NOT (answering hallucinates)")
    print("  negative = unanswerable by construction (refusing is right)")

    baseline = report["baseline_always_answer"]
    print("\n=== REFUSAL THRESHOLD SWEEP ===")
    print(
        f"baseline 'never refuse' accuracy: {baseline['accuracy']}  "
        f"<- must beat this, over {baseline['scored']} items\n"
    )
    header = (
        f"{'thresh':>7} {'prevented':>12} {'rate':>7} {'false ref':>11} {'rate':>7} "
        f"{'neg refused':>12} {'net':>5} {'acc':>7}"
    )
    print(header)
    print("-" * len(header))

    best = max(report["thresholds"], key=lambda row: (row["accuracy"] or 0.0))
    for row in report["thresholds"]:
        marker = " <-- best accuracy" if row is best else ""
        print(
            f"{row['threshold']:>7.3f} "
            f"{row['hallucinations_prevented']:>5}/{row['hallucinations_possible']:<6} "
            f"{(row['prevention_rate'] or 0.0):>7.3f} "
            f"{row['false_refusals']:>4}/{row['grounded_total']:<6} "
            f"{(row['false_refusal_rate'] or 0.0):>7.3f} "
            f"{row['negatives_refused']:>5}/{row['negatives_total']:<6} "
            f"{row['net']:>5} "
            f"{(row['accuracy'] or 0.0):>7.4f}{marker}"
        )

    print(
        "\nnet = ungrounded answers stopped + negatives refused - answerable "
        "questions wrongly refused."
    )
    print("A threshold whose net is <= 0 buys nothing and must not be adopted.")


def _print_vs_generator(rows: list[dict[str, Any]], source: Path) -> None:
    """Print the gate measured as an addition to the generator.

    Args:
        rows: Output of `compare_to_generator`.
        source: The generation run joined against, for provenance.
    """
    if not rows:
        return
    first = rows[0]
    print("\n=== GATE COMBINED WITH THE GENERATOR'S OWN REFUSALS ===")
    print(f"joined {first['joined_items']} items against {source.name}")
    print(
        f"generator alone: negatives {first['negatives_generator_only']}/"
        f"{first['negatives_total']}, blind {first['blind_generator_only']}/"
        f"{first['blind_total']}\n"
    )
    header = (
        f"{'thresh':>7} {'negatives':>12} {'blind':>10} {'new false ref':>14} "
        f"{'gen calls saved':>16}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['threshold']:>7.3f} "
            f"{row['negatives_combined']:>5}/{row['negatives_total']:<6} "
            f"{row['blind_combined']:>4}/{row['blind_total']:<5} "
            f"{row['new_false_refusals']:>7}/{row['grounded_total']:<6} "
            f"{row['generator_calls_saved']:>16}"
        )
    print(
        "\n'new false ref' counts only answerable questions the gate refuses that "
        "the generator would have answered — the cost the gate actually introduces."
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase3_agents.confidence_gate",
        description="Refuse, escalate or answer based on rank-1 retrieval confidence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--self-test", action="store_true", help="Run the boundary cases.")
    parser.add_argument(
        "--sweep", action="store_true", help="Sweep refusal thresholds over a recorded run."
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=None,
        help="Retrieval *_items.jsonl to analyse. Defaults to the newest one.",
    )
    parser.add_argument(
        "--evidence-k",
        type=int,
        default=None,
        help="Chunks that reach the prompt. Defaults to rag.top_k from config.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Write the sweep report to this JSON file."
    )
    parser.add_argument(
        "--vs-generator",
        type=Path,
        default=None,
        help="Generation *_items.jsonl to join against, measuring the gate as an "
        "addition to the generator's own refusals rather than as a replacement.",
    )
    parser.add_argument(
        "--fine-grid",
        action="store_true",
        help="Sweep the low band densely. The decision boundary sits near 0 and "
        "the default grid is too coarse to site a threshold in it.",
    )
    parser.add_argument(
        "--allow-unreranked",
        action="store_true",
        help="Sweep a run whose scores are bi-encoder cosine. Reproduces P-14's "
        "negative result on purpose; never a source of an adoptable threshold.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the self-test and/or the threshold sweep.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if any self-test case failed.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.self_test and not args.sweep:
        args.self_test = True

    exit_code = 0

    if args.self_test:
        print("=== CONFIDENCE GATE SELF-TEST ===")
        failures = 0
        for name, passed, detail in _self_test():
            print(f"  {'PASS' if passed else 'FAIL'}  {name}" + ("" if passed else f"  ({detail})"))
            failures += not passed
        total = len(_self_test())
        print(f"\n{total - failures}/{total} passed")
        if failures:
            exit_code = 1

    if args.sweep:
        config = load_config(args.config)
        evidence_k = args.evidence_k or cfg_get(config, "rag.top_k", 5)
        results_dir = Path(cfg_get(config, "eval.results_dir", "evals/results"))
        items_path = args.items or latest_items_file(results_dir, "*retrieval*_items.jsonl")

        LOGGER.info("Analysing %s (evidence_k=%d)", items_path, evidence_k)
        summary = load_run_summary(items_path)
        if not args.allow_unreranked:
            require_reranked(summary, items_path)

        grid = FINE_GRID if args.fine_grid else DEFAULT_GRID
        items = load_items(items_path)
        report = sweep(items, grid, evidence_k)
        report["source_items"] = str(items_path)
        report["source_run"] = {
            "tag": summary.get("tag"),
            "created_at": summary.get("created_at"),
            "collection": summary.get("collection"),
            "rerank": summary.get("rerank"),
            "rerank_depth": summary.get("rerank_depth"),
            "chunk_policy": summary.get("chunk_policy"),
            "golden_set": summary.get("golden_set"),
        }
        report["configured"] = asdict(build_gate(config).decide_on_score(0.0))

        if args.vs_generator:
            generation = load_generation_items(args.vs_generator)
            report["vs_generator"] = compare_to_generator(items, generation, grid, evidence_k)
            report["vs_generator_source"] = str(args.vs_generator)

        _print_sweep(report)
        if "vs_generator" in report:
            _print_vs_generator(report["vs_generator"], args.vs_generator)
        print(f"\nsource: {items_path}")
        print(f"run: tag={summary.get('tag')!r} rerank={summary.get('rerank')!r} "
              f"depth={summary.get('rerank_depth')!r} collection={summary.get('collection')!r}")
        print(
            f"config in force: refuse_below="
            f"{cfg_get(config, 'confidence_gate.refuse_below', 0.20)}, "
            f"escalate_below={cfg_get(config, 'confidence_gate.escalate_below', 0.50)}"
        )

        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
            print(f"wrote {args.out}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
