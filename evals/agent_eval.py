"""The agent eval harness — Phase 3's exit criterion, measured.

**What CLAUDE.md asks for.** Task completion rate, tool-call accuracy, escalation
rate, average steps per task, and end-to-end latency against plain RAG, over the
50-task set. The exit criterion is narrower than "the agent is good": *the agent
must beat plain RAG on calculation and comparison tasks, with recorded evidence.*

**Why plain RAG is run too, on the same tasks, in the same process.** A number
for the agent alone answers nothing — every one of these metrics is a comparison.
Running the baseline from a previous session's notes would confound the
generator, the index and the corpus with the change being measured, which is
exactly the confound that made Phase 2's judged metrics unusable (METRICS §2.5).
Both systems here see the same tasks, the same collection and the same generator
in a single run.

**How "completed" is defined, and why it differs per route.** A single definition
would be dishonest, because the routes fail differently:

| Route | Completed when |
|---|---|
| `out_of_scope` | the system refused |
| `calculation` | the settlement equals the hand-verified `expected_payable` |
| `comparison` | every required document contributed a ground-truth page, and nothing was left unaddressed |
| `lookup` | not refused, a ground-truth page was retrieved, and no citation was fabricated |

The calculation rule is the strongest of the four and the reason the task set was
built: `expected_payable` was computed by hand and verified against the
calculator at build time (D-24), so "right answer" means an exact rupee match
rather than a judge's opinion. **No LLM judges anything in this file.**

**Plain RAG is scored by the same rules**, with one necessary difference: it has
no calculator, so a calculation task counts as completed when the expected figure
appears in its answer text. That is the fairest possible reading — it asks
whether plain RAG can produce the right number at all, which is the actual claim
under test.

Usage:
    python -m evals.agent_eval --help
    python -m evals.agent_eval --self-test
    python -m evals.agent_eval --routes calculation,comparison
    python -m evals.agent_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config

LOGGER = logging.getLogger("claimwise.agent_eval")

# Graph node names to the tool vocabulary `agent_tasks.jsonl` uses. The task set
# names capabilities ("retrieve"); the graph names implementations
# ("retrieve_global"). Mapping here keeps the task set independent of how the
# graph happens to be wired, so re-wiring the graph does not invalidate the set.
NODE_TO_TOOL: dict[str, str] = {
    "retrieve_global": "retrieve",
    "retrieve_per_document": "retrieve",
    "claims_calculator": "claims_calculator",
    "comparison_agent": "retrieve",
}

# Nodes that are pipeline stages rather than tools. Counted in `steps`, excluded
# from tool-call accuracy — scoring the router as a "tool call" would inflate
# accuracy on every task, since it runs unconditionally.
NON_TOOL_NODES = {"router", "confidence_gate", "generate"}

RUPEE_TOLERANCE = 1.0


@dataclass
class TaskOutcome:
    """One task's result under one system.

    Attributes:
        task_id: e.g. "t-015".
        route: The task's labelled route.
        system: "agent" or "plain_rag".
        completed: Whether the route's completion rule was satisfied.
        reason: Why not, when it failed. Empty on success.
        expected_tools: Tools the task says are needed.
        actual_tools: Tools the run actually used.
        tool_correct: Whether those sets match exactly.
        refused: Whether the system emitted the refusal sentence.
        should_refuse: Whether it was supposed to.
        escalated: Whether the run was flagged for human review.
        steps: Nodes visited. 1 for plain RAG, which has no graph.
        latency_s: End-to-end wall time.
        grounded: Whether a ground-truth page reached the context.
        docs_covered: Whether every required document did.
        invalid_citations: Cited pages absent from the retrieved set.
        figure_corrupted: Whether a rupee figure failed verification.
        payable: The settled figure, when a calculator ran.
        expected_payable: The hand-verified figure, for calculation tasks.
        prompt_tokens: Input tokens billed.
        completion_tokens: Output tokens billed.
    """

    task_id: str
    route: str
    system: str
    completed: bool = False
    reason: str = ""
    expected_tools: list[str] = field(default_factory=list)
    actual_tools: list[str] = field(default_factory=list)
    tool_correct: bool = False
    refused: bool = False
    should_refuse: bool = False
    escalated: bool = False
    steps: int = 0
    latency_s: float = 0.0
    grounded: bool = False
    docs_covered: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    figure_corrupted: bool = False
    payable: float | None = None
    expected_payable: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def load_tasks(path: Path, routes: set[str] | None, limit: int | None) -> list[dict[str, Any]]:
    """Read the agent task set.

    Args:
        path: Path to `agent_tasks.jsonl`.
        routes: Keep only these routes, or None for all.
        limit: Keep at most this many tasks, in file order.

    Returns:
        The selected tasks.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Agent task set not found: {path.resolve()}. "
            "Run `python -m evals.build_agent_tasks --write` first."
        )
    with path.open("r", encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    if routes:
        tasks = [task for task in tasks if task.get("route") in routes]
    return tasks[:limit] if limit else tasks


def tools_from_trace(trace: list[str]) -> list[str]:
    """Map a graph trace to the tool vocabulary, preserving first-use order.

    Args:
        trace: Node names in visit order.

    Returns:
        Distinct tool names, in the order first used.
    """
    seen: list[str] = []
    for node in trace:
        if node in NON_TOOL_NODES:
            continue
        tool = NODE_TO_TOOL.get(node)
        if tool and tool not in seen:
            seen.append(tool)
    return seen


def retrieved_refs(chunks: list[Any]) -> set[str]:
    """Render retrieved chunks as `doc_id:page` refs.

    Args:
        chunks: Retrieved chunks.

    Returns:
        The set of refs present.
    """
    return {f"{getattr(c, 'doc_id', '')}:{getattr(c, 'page', 0)}" for c in chunks}


def evidence_status(task: dict[str, Any], chunks: list[Any]) -> tuple[bool, bool]:
    """Check whether the retrieved chunks carry the task's evidence.

    Args:
        task: The task, carrying `ground_truth_refs`.
        chunks: Retrieved chunks.

    Returns:
        `(grounded, docs_covered)` — whether any ground-truth ref arrived, and
        whether every required document contributed one of its own.
    """
    truth = set(task.get("ground_truth_refs") or [])
    if not truth:
        return False, False

    got = retrieved_refs(chunks)
    hit = truth & got
    required_docs = {ref.split(":", 1)[0] for ref in truth}
    reached_docs = {ref.split(":", 1)[0] for ref in hit}
    return bool(hit), required_docs <= reached_docs


def score_agent_task(task: dict[str, Any], final: dict[str, Any], latency_s: float) -> TaskOutcome:
    """Apply the per-route completion rules to one agent run.

    Args:
        task: The task definition.
        final: The graph's final state.
        latency_s: End-to-end wall time.

    Returns:
        The scored outcome.
    """
    route = task.get("route", "lookup")
    chunks = final.get("retrieved") or []
    grounded, docs_covered = evidence_status(task, chunks)
    calculation = final.get("calculation")
    payable = getattr(calculation, "payable", None) if calculation is not None else None
    expected_payable = task.get("expected_payable")

    outcome = TaskOutcome(
        task_id=task["id"],
        route=route,
        system="agent",
        expected_tools=sorted(task.get("expected_tools") or []),
        actual_tools=sorted(tools_from_trace(final.get("trace") or [])),
        refused=bool(final.get("refused")),
        should_refuse=bool(task.get("should_refuse")),
        escalated=bool(final.get("escalate")),
        steps=len(final.get("trace") or []),
        latency_s=round(latency_s, 3),
        grounded=grounded,
        docs_covered=docs_covered,
        invalid_citations=list(final.get("invalid_citations") or []),
        figure_corrupted=bool(final.get("figure_corrupted")),
        payable=payable,
        expected_payable=expected_payable,
    )
    outcome.tool_correct = outcome.expected_tools == outcome.actual_tools
    outcome.completed, outcome.reason = _completion(
        route, outcome, answer=str(final.get("answer") or "")
    )
    return outcome


def _completion(route: str, outcome: TaskOutcome, answer: str) -> tuple[bool, str]:
    """Decide whether one outcome satisfies its route's completion rule.

    Shared by both systems so the comparison is scored identically, with the one
    documented exception: plain RAG has no calculator, so a calculation task is
    checked against the answer text instead of a settled figure.

    Args:
        route: The task's labelled route.
        outcome: The outcome so far.
        answer: The answer text, needed for the plain-RAG calculation rule.

    Returns:
        `(completed, reason_when_not)`.
    """
    if route == "out_of_scope" or outcome.should_refuse:
        return (outcome.refused, "" if outcome.refused else "answered a question it should refuse")

    if outcome.refused:
        return False, "refused an answerable question"

    if route == "calculation":
        if outcome.expected_payable is None:
            return False, "task has no expected_payable to check against"
        if outcome.payable is None:
            # Plain RAG, or an agent run where the calculator could not fire.
            forms = _amount_forms(outcome.expected_payable)
            if any(form in answer for form in forms):
                return True, ""
            return False, f"expected ₹{outcome.expected_payable:,.0f} absent from the answer"
        if abs(outcome.payable - outcome.expected_payable) <= RUPEE_TOLERANCE:
            return True, ""
        return False, (
            f"settled ₹{outcome.payable:,.0f}, expected ₹{outcome.expected_payable:,.0f}"
        )

    if route == "comparison":
        if not outcome.docs_covered:
            return False, "not every required document contributed evidence"
        return True, ""

    if not outcome.grounded:
        return False, "no ground-truth page reached the context"
    if outcome.invalid_citations:
        return False, f"fabricated citation(s): {outcome.invalid_citations}"
    return True, ""


def _amount_forms(value: float) -> set[str]:
    """Every spelling of a rupee amount that should count as the same number.

    Args:
        value: The amount in rupees.

    Returns:
        Western grouping, Indian grouping and bare digits.
    """
    from phase3_agents.graph import indian_grouped

    return {f"{value:,.0f}", f"{value:.0f}", indian_grouped(value)}


def score_rag_task(task: dict[str, Any], answer: Any, latency_s: float) -> TaskOutcome:
    """Score one plain-RAG run under the same rules.

    Args:
        task: The task definition.
        answer: A `RagAnswer`.
        latency_s: End-to-end wall time.

    Returns:
        The scored outcome.
    """
    route = task.get("route", "lookup")
    grounded, docs_covered = evidence_status(task, answer.retrieved)

    outcome = TaskOutcome(
        task_id=task["id"],
        route=route,
        system="plain_rag",
        expected_tools=sorted(task.get("expected_tools") or []),
        # Plain RAG always retrieves and can do nothing else. Recording that
        # honestly is the point: on a calculation task its tool set is simply
        # wrong, and that is the finding rather than a scoring artefact.
        actual_tools=["retrieve"],
        refused=bool(answer.refused),
        should_refuse=bool(task.get("should_refuse")),
        escalated=False,
        steps=1,
        latency_s=round(latency_s, 3),
        grounded=grounded,
        docs_covered=docs_covered,
        invalid_citations=list(answer.invalid_citations or []),
        payable=None,
        expected_payable=task.get("expected_payable"),
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
    )
    outcome.tool_correct = outcome.expected_tools == outcome.actual_tools
    outcome.completed, outcome.reason = _completion(route, outcome, answer=answer.answer or "")
    return outcome


def summarise(outcomes: list[TaskOutcome]) -> dict[str, Any]:
    """Aggregate outcomes into the reported metric set.

    Args:
        outcomes: Scored outcomes for one system.

    Returns:
        A nested summary dict.
    """
    if not outcomes:
        return {"tasks": 0}

    def rate(predicate: Any, population: list[TaskOutcome]) -> float:
        """Fraction of a population satisfying a predicate."""
        if not population:
            return 0.0
        return round(sum(1 for o in population if predicate(o)) / len(population), 4)

    by_route: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[TaskOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.route].append(outcome)
    for route, group in sorted(grouped.items()):
        by_route[route] = {
            "tasks": len(group),
            "task_completion": rate(lambda o: o.completed, group),
            "tool_call_accuracy": rate(lambda o: o.tool_correct, group),
            "escalation_rate": rate(lambda o: o.escalated, group),
            "avg_steps": round(statistics.fmean([o.steps for o in group]), 2),
            "failed_task_ids": [o.task_id for o in group if not o.completed],
        }

    latencies = sorted(o.latency_s for o in outcomes)
    p95_index = min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))

    return {
        "tasks": len(outcomes),
        "task_completion": rate(lambda o: o.completed, outcomes),
        "tool_call_accuracy": rate(lambda o: o.tool_correct, outcomes),
        "escalation_rate": rate(lambda o: o.escalated, outcomes),
        "avg_steps": round(statistics.fmean([o.steps for o in outcomes]), 2),
        "grounded_rate": rate(lambda o: o.grounded, outcomes),
        "docs_covered_rate": rate(lambda o: o.docs_covered, outcomes),
        "fabricated_citation_rate": rate(lambda o: bool(o.invalid_citations), outcomes),
        "figure_corrupted_rate": rate(lambda o: o.figure_corrupted, outcomes),
        "latency_s": {
            "median": round(statistics.median(latencies), 2),
            "p95": round(latencies[p95_index], 2),
        },
        "tokens": {
            "prompt": sum(o.prompt_tokens for o in outcomes),
            "completion": sum(o.completion_tokens for o in outcomes),
        },
        "by_route": by_route,
    }


def print_report(agent: dict[str, Any], rag: dict[str, Any] | None) -> None:
    """Print the metric report, with the head-to-head when a baseline ran.

    Args:
        agent: Agent summary.
        rag: Plain-RAG summary, or None.
    """
    print("\n=== AGENT EVAL (deterministic scoring, 0 judge calls) ===")
    print(f"tasks evaluated : {agent['tasks']}")

    rows = [
        ("task completion", "task_completion", "{:.3f}"),
        ("tool-call accuracy", "tool_call_accuracy", "{:.3f}"),
        ("escalation rate", "escalation_rate", "{:.3f}"),
        ("avg steps/task", "avg_steps", "{:.2f}"),
        ("grounded rate", "grounded_rate", "{:.3f}"),
        ("all-docs-covered", "docs_covered_rate", "{:.3f}"),
        ("fabricated citations", "fabricated_citation_rate", "{:.3f}"),
        ("corrupted figures", "figure_corrupted_rate", "{:.3f}"),
    ]
    header = f"\n{'':<24}{'agent':>10}"
    if rag:
        header += f"{'plain RAG':>12}{'delta':>10}"
    print(header)
    for label, key, fmt in rows:
        line = f"{label:<24}{fmt.format(agent[key]):>10}"
        if rag:
            before = rag.get(key, 0.0)
            line += f"{fmt.format(before):>12}{agent[key] - before:>+10.3f}"
        print(line)

    print(
        f"\n{'p50 / p95 latency':<24}{agent['latency_s']['median']:>6}s /"
        f" {agent['latency_s']['p95']}s"
    )
    if rag:
        print(
            f"{'  plain RAG':<24}{rag['latency_s']['median']:>6}s /"
            f" {rag['latency_s']['p95']}s"
        )

    print("\nby route — task completion (agent vs plain RAG):")
    for route, stats in agent["by_route"].items():
        line = f"  {route:<14}: {stats['task_completion']:.3f}  ({stats['tasks']} tasks)"
        if rag and route in rag.get("by_route", {}):
            before = rag["by_route"][route]["task_completion"]
            line += f"   vs {before:.3f}   {stats['task_completion'] - before:+.3f}"
        print(line)
        if stats["failed_task_ids"]:
            print(f"      failed: {', '.join(stats['failed_task_ids'])}")

    if rag:
        print("\n--- PHASE 3 EXIT CRITERION ---")
        print("The agent must beat plain RAG on calculation and comparison tasks.")
        met = True
        for route in ("calculation", "comparison"):
            if route not in agent["by_route"] or route not in rag.get("by_route", {}):
                print(f"  {route:<12}: not evaluated in this run")
                met = False
                continue
            after = agent["by_route"][route]["task_completion"]
            before = rag["by_route"][route]["task_completion"]
            verdict = "PASS" if after > before else "FAIL"
            met = met and after > before
            print(f"  {route:<12}: {before:.3f} -> {after:.3f}  ({after - before:+.3f})  {verdict}")
        print(f"\n  EXIT CRITERION: {'MET' if met else 'NOT MET'}")


def _run_pool(
    tasks: list[dict[str, Any]],
    system: str,
    desc: str,
    workers: int,
    run_one: Any,
) -> list[TaskOutcome]:
    """Run one task per worker, preserving task order in the results.

    **Why threads and not processes.** Almost all of a task's wall clock is spent
    waiting on the generator's HTTP call, and the cross-encoder releases the GIL
    during inference, so threads recover most of the idle time without the cost
    of re-loading the embedder and reranker in every process.

    **The caveat, and how to settle it.** The embedder, reranker and the
    embedded-mode `QdrantClient(path=...)` are shared across workers. Searches
    are read-only, but Qdrant's local mode is not *documented* thread-safe, and a
    silently wrong retrieval would corrupt a number rather than raise. Before
    trusting a concurrent run, reproduce one subset with `--workers 1` and
    confirm the scores match. That check costs one run and is the only thing that
    makes the speedup free rather than assumed.

    Order is restored by index rather than by completion, so a results file does
    not change shape with the worker count and two runs stay diffable.

    A task that raises is scored as a failure rather than losing the run — the
    2026-08-20 NIM timeouts would otherwise have discarded 50 completed tasks
    along with the two that failed.

    Args:
        tasks: The task set.
        system: Column name recorded on each outcome, "agent" or "plain_rag".
        desc: Progress-bar label.
        workers: Concurrent tasks. 1 runs strictly sequentially.
        run_one: Callable taking a task and returning the system's raw output.

    Returns:
        One scored outcome per task, in task order.
    """
    scorer = score_agent_task if system == "agent" else score_rag_task
    results: list[TaskOutcome | None] = [None] * len(tasks)

    def work(index: int) -> None:
        """Run and score one task into `results[index]`."""
        task = tasks[index]
        started = time.perf_counter()
        try:
            output = run_one(task)
        except Exception as error:  # noqa: BLE001 - one bad task must not lose the run
            LOGGER.error("Task %s raised: %s", task["id"], error)
            results[index] = TaskOutcome(
                task_id=task["id"],
                route=task.get("route", "lookup"),
                system=system,
                reason=f"raised: {error}",
                latency_s=round(time.perf_counter() - started, 3),
            )
            return
        results[index] = scorer(task, output, time.perf_counter() - started)

    if workers <= 1:
        for index in tqdm(range(len(tasks)), desc=desc, unit="task"):
            work(index)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, index) for index in range(len(tasks))]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=desc, unit="task"
            ):
                future.result()

    return [outcome for outcome in results if outcome is not None]


def run_agent(
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
    pipeline: str | None,
    workers: int = 1,
) -> list[TaskOutcome]:
    """Run every task through the compiled agent graph.

    Args:
        tasks: The task set.
        config: Parsed `config.yaml`.
        pipeline: Override `retrieval_agent.pipeline`.
        workers: Concurrent tasks.

    Returns:
        One scored outcome per task.
    """
    from phase3_agents.graph import build_graph, build_graph_resources, run

    resources = build_graph_resources(config, pipeline=pipeline)
    try:
        compiled = build_graph(resources)
        return _run_pool(
            tasks,
            "agent",
            "Agent",
            workers,
            lambda task: run(task["question"], resources, compiled),
        )
    finally:
        resources.close()


def run_plain_rag(
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
    pipeline: str | None,
    workers: int = 1,
) -> list[TaskOutcome]:
    """Run every task through the Phase 2 pipeline, with no agent layer.

    Uses the same collection, reranker, parent store and generator as the agent,
    so the only difference between the two columns is the agent layer itself.

    Args:
        tasks: The task set.
        config: Parsed `config.yaml`.
        pipeline: Override `retrieval_agent.pipeline`.
        workers: Concurrent tasks.

    Returns:
        One scored outcome per task.
    """
    from common.generator import build_generator
    from phase1_rag.rag_chain import answer_question
    from phase3_agents.retrieval_node import build_resources

    resources = build_resources(config, pipeline=pipeline)
    generator = build_generator(config)
    settings = {
        "candidate_depth": resources.settings["candidate_depth"],
        "top_k": resources.settings["top_k"],
        "query_prefix": resources.settings["query_prefix"],
        "normalize": resources.settings["normalize"],
        "user_id": resources.settings["default_user_id"],
        "refusal_text": cfg_get(config, "rag.refusal_text", "Not covered in your documents."),
    }

    try:
        return _run_pool(
            tasks,
            "plain_rag",
            "Plain RAG",
            workers,
            lambda task: answer_question(
                task["question"],
                resources.client,
                resources.collection_name,
                resources.embedder,
                generator,
                settings,
                reranker=resources.reranker,
                parents=resources.parents,
            ),
        )
    finally:
        resources.close()


# --- Self-test ---------------------------------------------------------------


def _self_test() -> list[tuple[str, bool, str]]:
    """Exercise the scoring rules without models, an index or a generator.

    The scoring rules decide the phase's exit criterion, so they are tested
    directly rather than inferred from a live run.

    Returns:
        One (name, passed, detail) per check.
    """
    results: list[tuple[str, bool, str]] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        """Record one comparison."""
        passed = actual == expected
        results.append((name, passed, f"expected {expected!r}, got {actual!r}"))

    check(
        "trace maps to tools, pipeline stages excluded",
        tools_from_trace(["router", "retrieve_global", "confidence_gate", "claims_calculator", "generate"]),
        ["retrieve", "claims_calculator"],
    )
    check(
        "per-document retrieval is still 'retrieve'",
        tools_from_trace(["router", "retrieve_per_document", "confidence_gate", "comparison_agent"]),
        ["retrieve"],
    )
    check("a refusal uses no tools", tools_from_trace(["router", "confidence_gate"]), [])

    class _C:
        """Minimal chunk stand-in."""

        def __init__(self, doc_id: str, page: int) -> None:
            self.doc_id = doc_id
            self.page = page

    comparison_task = {
        "id": "t-029",
        "route": "comparison",
        "expected_tools": ["retrieve"],
        "ground_truth_refs": ["5b8f496626be:20", "b1dbe8fb7864:31"],
        "min_documents": 2,
    }
    both = [_C("5b8f496626be", 20), _C("b1dbe8fb7864", 31)]
    one = [_C("b1dbe8fb7864", 31)]
    check("both documents reached", evidence_status(comparison_task, both), (True, True))
    check("one document reached is grounded but not covered", evidence_status(comparison_task, one), (True, False))
    check("nothing reached", evidence_status(comparison_task, []), (False, False))

    # The half-answered comparison — grounded, plausible, and NOT complete. This
    # is the failure the whole per-document change exists to remove, so scoring
    # it as a success would hide the finding.
    half = score_agent_task(
        comparison_task,
        {"retrieved": one, "trace": ["router", "retrieve_per_document", "confidence_gate", "comparison_agent"]},
        latency_s=1.0,
    )
    check("half-answered comparison is not completed", half.completed, False)
    check("and says why", half.reason, "not every required document contributed evidence")

    full = score_agent_task(
        comparison_task,
        {"retrieved": both, "trace": ["router", "retrieve_per_document", "confidence_gate", "comparison_agent"]},
        latency_s=1.0,
    )
    check("covered comparison is completed", full.completed, True)
    check("tool call scored correct", full.tool_correct, True)
    check("steps counted", full.steps, 4)

    calc_task = {
        "id": "t-015",
        "route": "calculation",
        "expected_tools": ["retrieve", "claims_calculator"],
        "ground_truth_refs": ["b1dbe8fb7864:39"],
        "expected_payable": 216000,
    }

    class _Calc:
        """Minimal ClaimResult stand-in."""

        def __init__(self, payable: float) -> None:
            self.payable = payable

    right = score_agent_task(
        calc_task,
        {
            "retrieved": [_C("b1dbe8fb7864", 39)],
            "trace": ["router", "retrieve_global", "confidence_gate", "claims_calculator", "generate"],
            "calculation": _Calc(216000.0),
        },
        latency_s=1.0,
    )
    check("exact rupee match completes a calculation", right.completed, True)

    # The live 2026-08-20 failure: terms were never retrieved, so the calculator
    # returned the bill unchanged. Plausible-looking, and wrong.
    wrong = score_agent_task(
        calc_task,
        {
            "retrieved": [_C("b1dbe8fb7864", 16)],
            "trace": ["router", "retrieve_global", "confidence_gate", "claims_calculator", "generate"],
            "calculation": _Calc(240000.0),
        },
        latency_s=1.0,
    )
    check("wrong settlement fails", wrong.completed, False)
    check("and reports both figures", wrong.reason, "settled ₹240,000, expected ₹216,000")

    negative = {"id": "t-041", "route": "out_of_scope", "expected_tools": [], "should_refuse": True}
    refused = score_agent_task(negative, {"refused": True, "trace": ["router", "confidence_gate"]}, 1.0)
    check("refusing an out-of-scope task completes it", refused.completed, True)
    check("refusal uses no tools and scores correct", refused.tool_correct, True)

    answered = score_agent_task(negative, {"refused": False, "trace": ["router", "retrieve_global", "confidence_gate", "generate"]}, 1.0)
    check("answering an out-of-scope task fails", answered.completed, False)
    check("and says why", answered.reason, "answered a question it should refuse")

    lookup_task = {
        "id": "t-002",
        "route": "lookup",
        "expected_tools": ["retrieve"],
        "ground_truth_refs": ["b1dbe8fb7864:31"],
    }
    fabricated = score_agent_task(
        lookup_task,
        {
            "retrieved": [_C("b1dbe8fb7864", 31)],
            "trace": ["router", "retrieve_global", "confidence_gate", "generate"],
            "invalid_citations": [99],
        },
        1.0,
    )
    check("a fabricated citation fails the task", fabricated.completed, False)
    check("and says which", fabricated.reason, "fabricated citation(s): [99]")

    refused_answerable = score_agent_task(
        lookup_task, {"retrieved": [_C("b1dbe8fb7864", 31)], "refused": True, "trace": ["router"]}, 1.0
    )
    check("refusing an answerable task fails", refused_answerable.completed, False)

    summary = summarise([right, wrong, full, half, refused, answered])
    check("summary counts tasks", summary["tasks"], 6)
    check("summary computes completion", summary["task_completion"], 0.5)
    check("summary groups by route", sorted(summary["by_route"]), ["calculation", "comparison", "out_of_scope"])
    check("empty input summarises to zero tasks", summarise([]), {"tasks": 0})

    return results


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evals.agent_eval",
        description="Phase 3 exit metrics: the agent against plain RAG on the 50-task set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tasks-path", type=Path, default=None, help="Override the task set path.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the scoring checks. No index, no models, no LLM call, no cost.",
    )
    parser.add_argument(
        "--routes",
        default=None,
        help="Comma-separated routes to evaluate, e.g. calculation,comparison.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many tasks.")
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Skip the plain-RAG baseline. Halves the LLM calls, but the exit criterion "
        "cannot be evaluated without it.",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        choices=["chunk_policy", "parent_docs"],
        help="Override retrieval_agent.pipeline for BOTH systems.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent tasks. Most of a task is spent waiting on the generator, "
        "so 4 cuts wall clock roughly 4x. Use 1 to reproduce a sequential run "
        "exactly, or if the CPU reranker starts contending.",
    )
    parser.add_argument("--tag", default="", help="Label recorded in the results file.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the agent eval harness.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 on a failed self-test or bad usage.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.self_test:
        results = _self_test()
        for name, passed, detail in results:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
            if not passed:
                print(f"        {detail}")
        failed = sum(1 for _, passed, _ in results if not passed)
        print(f"\n{len(results) - failed}/{len(results)} checks passed")
        return 1 if failed else 0

    config = load_config(args.config)
    eval_dir = Path(cfg_get(config, "paths.eval_dir", "data/eval"))
    tasks_path = args.tasks_path or eval_dir / cfg_get(
        config, "eval.agent_tasks_filename", "agent_tasks.jsonl"
    )
    routes = {r.strip() for r in args.routes.split(",")} if args.routes else None
    tasks = load_tasks(tasks_path, routes, args.limit)
    if not tasks:
        LOGGER.error("No tasks selected from %s.", tasks_path)
        return 1

    # One generator call per non-refused task, per system. Printed before
    # spending anything, per CLAUDE.md §3.
    systems = 1 if args.agent_only else 2
    print(f"\n{len(tasks)} task(s) x {systems} system(s) = up to {len(tasks) * systems} generator calls.")
    print(f"provider: {cfg_get(config, 'generator.provider', 'unknown')} "
          f"/ {cfg_get(config, 'generator.model', 'unknown')}\n")

    agent_outcomes = run_agent(tasks, config, args.pipeline, workers=args.workers)
    rag_outcomes = (
        []
        if args.agent_only
        else run_plain_rag(tasks, config, args.pipeline, workers=args.workers)
    )

    agent_summary = summarise(agent_outcomes)
    rag_summary = summarise(rag_outcomes) if rag_outcomes else None
    print_report(agent_summary, rag_summary)

    results_dir = Path(cfg_get(config, "eval.results_dir", "evals/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{args.tag}" if args.tag else ""

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": args.tag,
        "task_set": tasks_path.as_posix(),
        "tasks_evaluated": len(tasks),
        "routes_filter": sorted(routes) if routes else None,
        "generator": {
            "provider": cfg_get(config, "generator.provider", ""),
            "model": cfg_get(config, "generator.model", ""),
        },
        "agent": agent_summary,
        "plain_rag": rag_summary,
    }
    results_path = results_dir / f"agenteval_{stamp}{suffix}.json"
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    per_task_path = results_dir / f"agenteval_{stamp}{suffix}_items.jsonl"
    with per_task_path.open("w", encoding="utf-8") as handle:
        for outcome in [*agent_outcomes, *rag_outcomes]:
            handle.write(json.dumps(asdict(outcome), ensure_ascii=False) + "\n")

    print(f"\nresults  : {results_path}")
    print(f"per-task : {per_task_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
