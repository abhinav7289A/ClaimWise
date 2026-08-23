"""The agent graph — Phase 3, task 3. Where the five nodes become one system.

**What a graph buys that a function call chain does not.** Phases 1 and 2 were a
straight line: retrieve, rerank, prompt, generate. Every question paid for every
stage. Phase 3's nodes are conditional — an out-of-scope question should never
reach the generator, a calculation must reach the calculator, a comparison needs
a different retrieval strategy and a different prompt. Expressing that as nested
`if` statements works exactly until the fourth branch, at which point nobody can
say which paths are reachable or what ran on a given request.

LangGraph makes the control flow a declared object rather than an emergent
property of the code. Three things follow, and the third is why this is worth a
dependency:

1. **The trace is free.** Every node appends its name to `state["trace"]`, so
   "avg steps per task" — a Phase 3 exit metric — is read off the state instead
   of instrumented for.
2. **Short-circuits are explicit.** The gate refusing is an edge to `END`, not a
   flag that later code has to remember to check.
3. **Phase 5 streams it.** The status events the frontend mascot consumes
   (`retrieving`, `calculating`, `generating`) are node boundaries. A graph
   already has those; a call chain has to invent them.

**Why LangGraph and not LangChain's `AgentExecutor` or a hand-rolled loop.** An
`AgentExecutor` gives the model the choice of tool at every step, which is the
wrong shape here — routing is a measured, deterministic classifier (D-25) and
handing that decision to an LLM would discard a component with recorded accuracy
for one with none. A hand-rolled dispatch is honestly fine at four routes; the
dependency earns itself at the point where Phase 5 needs streamed intermediate
state, which is built in here and fiddly by hand.

**The shape:**

    router -> retrieve -> gate --(refused)------------------> END
                            |
                            +--(comparison)--> compare -----> END
                            |
                            +--(calculation)-> calculate --+
                            |                              |
                            +--(otherwise)-----------------+-> generate -> END

**Every node is already built and separately tested.** This module wires them and
adds only one new node — `calculate` — which is itself a thin shell over
`term_extraction` and `claims_calculator`.

Usage:
    python -m phase3_agents.graph --help
    python -m phase3_agents.graph --self-test
    python -m phase3_agents.graph --question "..."
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from common.generator import Generator, build_generator
from phase1_rag.rag_chain import SYSTEM_PROMPT, format_passages, verify_citations
from phase3_agents.claims_calculator import ClaimResult, InvalidClaimInput, settle
from phase3_agents.comparison_agent import comparison_node
from phase3_agents.confidence_gate import ConfidenceGate, build_gate, gate_node
from phase3_agents.retrieval_node import RetrievalResources, build_resources, retrieve_node
from phase3_agents.router import ExemplarRouter, build_router
from phase3_agents.state import AgentState, new_state
from phase3_agents.term_extraction import build_claim_request

try:
    from langgraph.graph import END, StateGraph
except ImportError as error:  # pragma: no cover - dependency guidance
    raise ImportError(
        "langgraph is required for the agent graph. Add it with "
        "`uv add langgraph`, or run the individual nodes directly — every one of "
        "them works without it."
    ) from error

LOGGER = logging.getLogger("claimwise.graph")

# The calculator's figures are handed to the generator as an authoritative block
# it must not recompute. This is the prompt-side half of CLAUDE.md's no-LLM-
# arithmetic rule: the rule is only enforceable if the model is told, in the
# prompt, that the arithmetic is already done and is not its job.
CALCULATION_BLOCK = """
=== SETTLEMENT CALCULATION (already computed — do not recompute) ===
{breakdown}

Present these figures exactly as given. Do not add, subtract or re-derive any \
amount. Explain in words what each deduction is and cite the pages shown."""

# Rupee amounts as they appear in an ANSWER — always symbol- or Rs-prefixed,
# unlike the looser patterns in `term_extraction`, which read policy prose. A
# bare number in an answer is usually a page, a percentage or a month count, and
# treating those as money would flag every correct answer.
# The \b before `rs` is load-bearing: without it, "36 years 5 months" contains
# "rs 5" and would be read as a rupee amount.
ANSWER_AMOUNT_PATTERN = re.compile(r"(?:₹|\brs\.?\s*)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class GraphResources:
    """Everything the compiled graph needs, loaded once.

    Attributes:
        retrieval: Loaded retrieval resources — index, embedder, reranker, parents.
        router: The exemplar router, sharing the retrieval embedder.
        gate: The configured confidence gate.
        generator: The swappable generator.
        refusal_text: The exact refusal sentence from config.
        top_k: Passages placed in a non-comparison prompt.
    """

    retrieval: RetrievalResources
    router: ExemplarRouter
    gate: ConfidenceGate
    generator: Generator
    refusal_text: str
    top_k: int

    def close(self) -> None:
        """Release the Qdrant client."""
        self.retrieval.close()


def build_graph_resources(
    config: dict[str, Any],
    pipeline: str | None = None,
) -> GraphResources:
    """Load every node's dependencies once.

    The router is given the retrieval node's embedder rather than constructing
    its own. bge-small is ~130MB and loading it twice buys nothing — the router
    embeds questions with the identical model the index was built with.

    Args:
        config: Parsed `config.yaml`.
        pipeline: Override `retrieval_agent.pipeline`.

    Returns:
        Ready-to-use resources. The caller owns them and must call `.close()`.
    """
    retrieval = build_resources(config, pipeline=pipeline)
    return GraphResources(
        retrieval=retrieval,
        router=build_router(config, retrieval.embedder),
        gate=build_gate(config),
        generator=build_generator(config),
        refusal_text=cfg_get(config, "rag.refusal_text", "Not covered in your documents."),
        top_k=cfg_get(config, "rag.top_k", 5),
    )


# --- Nodes -------------------------------------------------------------------


def router_node(state: AgentState, router: ExemplarRouter) -> dict[str, Any]:
    """Classify the question into one of the four routes.

    Args:
        state: The current state. Reads `question`.
        router: The configured exemplar router.

    Returns:
        Keys to merge into the state.
    """
    decision = router.route(state["question"])
    LOGGER.info("route=%s (%s)", decision.route, decision.reason)
    return {
        "route": decision.route,
        "route_reason": decision.reason,
        "trace": [*state.get("trace", []), "router"],
    }


def calculator_node(state: AgentState) -> dict[str, Any]:
    """Settle the claim deterministically, when the question supplies enough.

    Fails soft on purpose. A calculation question with no rupee figure in it, or
    one whose policy terms were never retrieved, produces no `calculation` and
    the graph continues to the generator — which answers from the passages like
    any lookup. Raising instead would turn a partially-answerable question into
    an error page.

    Args:
        state: The current state. Reads `question` and `retrieved`.

    Returns:
        Keys to merge into the state.
    """
    trace = [*state.get("trace", []), "claims_calculator"]
    request, conflicts = build_claim_request(state["question"], state.get("retrieved") or [])

    if request is None:
        return {
            "trace": trace,
            "assumptions": [
                *state.get("assumptions", []),
                "No claim amount was found in the question, so no settlement was computed.",
            ],
        }

    try:
        result: ClaimResult = settle(request)
    except InvalidClaimInput as error:
        LOGGER.warning("Calculator rejected the inputs: %s", error)
        return {
            "trace": trace,
            "assumptions": [*state.get("assumptions", []), f"Calculation skipped: {error}"],
        }

    update: dict[str, Any] = {"calculation": result, "trace": trace}
    carried = [*state.get("assumptions", [])]

    # A figure the user gave that contradicts their own policy is worth saying
    # out loud. The policy value was used (see `merge_terms`), but silently
    # substituting it would leave the user unable to tell why the answer differs
    # from the arithmetic they did in their head.
    if conflicts:
        carried.extend(conflicts)
        update["escalate"] = True
        update["escalation_reason"] = (
            f"{len(conflicts)} term(s) in the question disagree with the policy"
        )

    if result.assumptions or conflicts:
        update["assumptions"] = [*carried, *result.assumptions]
    if result.assumptions:
        # A provisional rupee figure is exactly the kind of answer a human should
        # see before the user does.
        update["escalate"] = True
        update["escalation_reason"] = (
            f"settlement computed with {len(result.assumptions)} assumed term(s)"
        )
    return update


def build_answer_prompt(
    question: str,
    chunks: list[Any],
    calculation: ClaimResult | None,
    top_k: int,
) -> str:
    """Assemble the generator prompt, with the settlement block when there is one.

    Args:
        question: The user's question.
        chunks: Retrieved chunks, best first.
        calculation: The calculator's result, when it ran.
        top_k: How many passages to include.

    Returns:
        The complete user message.
    """
    passages = format_passages(chunks[:top_k])
    prompt = f"{passages}\n\nQuestion: {question}\n"
    if calculation is not None:
        prompt += CALCULATION_BLOCK.format(breakdown=calculation.explain())
    else:
        prompt += "\nAnswer using only the passages above, citing pages as [p.N]."
    return prompt


def indian_grouped(value: float) -> str:
    """Format a whole rupee amount with Indian digit grouping.

    Last three digits, then groups of two: 216000 -> "2,16,000". Needed because
    the calculator renders Western grouping (`:,.0f` -> "216,000") while the
    generator, writing for an Indian user, often reformats to the local
    convention. Both spell the same number, and a guard that rejected one of them
    would report a correct answer as corrupted.

    Args:
        value: The amount in rupees.

    Returns:
        The amount with Indian digit grouping, no currency symbol.
    """
    digits = f"{abs(int(round(value))):d}"
    sign = "-" if value < 0 else ""
    if len(digits) <= 3:
        return sign + digits

    head, tail = digits[:-3], digits[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return sign + ",".join([*parts, tail])


def known_amounts(calculation: ClaimResult) -> set[int]:
    """Every rupee value the calculator actually produced.

    Args:
        calculation: The calculator's result.

    Returns:
        Whole-rupee values from the bill, the payable figure, and every step's
        running balance and reduction.
    """
    values = {round(calculation.claimed), round(calculation.payable)}
    for step in calculation.steps:
        values.add(round(step.amount_before))
        values.add(round(step.amount_after))
        values.add(round(step.reduction))
    return values


def verify_computed_figures(answer_text: str, calculation: ClaimResult | None) -> list[str]:
    """Check the answer's rupee amounts against what the calculator computed.

    Measured 2026-08-20: handed ₹240,000 and an explicit instruction not to
    recompute, the model wrote **₹240,0000**. Nothing caught it — citations were
    valid, the calculator's own figure was right, and the answer read normally.
    A wrong rupee amount is the most expensive output this system can produce, so
    it gets the same treatment as a fabricated citation: deterministic and free,
    on every call.

    Two checks, because either alone misses the live failure:

    1. **The payable figure must appear as a whole number.** Bounded by digit and
       comma lookaround, since ``"240,000" in "240,0000"`` is true and a naive
       substring test would pass the very corruption it exists to catch.
    2. **Every rupee amount in the answer must be one the calculator produced.**
       This is what actually catches the live case: the claimed figure was
       written correctly on one line, so check 1 alone was satisfied while
       ₹240,0000 sat on the next line. An amount that appears from nowhere is the
       signal.

    Any spelling of a number is accepted — Western grouping, Indian grouping, or
    bare digits — because the check is about the value surviving, not formatting.

    **Known false positive:** a policy amount quoted from a passage but never
    applied as a step (a sum insured mentioned in passing) has no matching value
    here and will be reported. That is the intended direction of error — the
    consequence is an escalation flag on a kept answer, not a blocked one.

    Args:
        answer_text: The generated answer.
        calculation: The calculator's result, when it ran.

    Returns:
        One description per problem found. Empty when there was no calculation or
        everything checks out.
    """
    if calculation is None:
        return []

    problems: list[str] = []

    if calculation.eligible:
        payable = calculation.payable
        forms = {f"{payable:,.0f}", f"{payable:.0f}", indian_grouped(payable)}
        # Lookaround on digits AND commas: without the comma, "2,16,000" would
        # be found inside "12,16,000".
        found = any(
            re.search(rf"(?<![\d,]){re.escape(form)}(?![\d,])", answer_text) for form in forms
        )
        if not found:
            problems.append(f"computed payable ₹{payable:,.0f} does not appear in the answer")

    known = known_amounts(calculation)
    for match in ANSWER_AMOUNT_PATTERN.finditer(answer_text):
        raw = match.group(1).replace(",", "")
        if not raw or raw.startswith("."):
            continue
        value = round(float(raw))
        if value not in known:
            problems.append(f"₹{value:,} appears in the answer but was never computed")

    return problems


def generate_node(
    state: AgentState,
    generator: Generator,
    refusal_text: str,
    top_k: int,
) -> dict[str, Any]:
    """Generate the final answer and verify its citations.

    Args:
        state: The current state.
        generator: The swappable generator.
        refusal_text: The exact refusal sentence from config.
        top_k: Passages to place in the prompt.

    Returns:
        Keys to merge into the state.
    """
    chunks = state.get("retrieved") or []
    trace = [*state.get("trace", []), "generate"]

    if not chunks:
        return {"answer": refusal_text, "refused": True, "trace": trace}

    prompt = build_answer_prompt(
        state["question"], chunks, state.get("calculation"), top_k
    )
    result = generator.generate(prompt, system=SYSTEM_PROMPT.format(refusal_text=refusal_text))
    cited, invalid = verify_citations(result.text, chunks[:top_k])
    calculation = state.get("calculation")
    refused = refusal_text.lower() in result.text.lower()

    # A refusal cannot contain the payable figure, so verifying one against it
    # reports a corruption on every wrongly-refused calculation. Measured
    # 2026-08-21: 3 of 5 `figure_corrupted` flags on the exit subset were
    # refusals (t-016, t-017, t-023), inflating the rate from 0.077 to 0.192 and
    # hiding the real defect, which is the generator refusing while holding a
    # correct settlement.
    missing_figures = [] if refused else verify_computed_figures(result.text, calculation)

    update: dict[str, Any] = {
        "answer": result.text,
        "cited_pages": cited,
        "invalid_citations": invalid,
        "refused": refused,
        "figure_corrupted": bool(missing_figures),
        "trace": trace,
    }

    # A refusal on a question the calculator could settle is its own failure, and
    # a distinct one from a garbled figure. Surface it rather than letting it
    # score as a corruption.
    if refused and calculation is not None and calculation.eligible:
        update["escalate"] = True
        update["escalation_reason"] = (
            f"generator refused although a settlement of ₹{calculation.payable:,.0f} "
            "was computed from the retrieved passages"
        )

    # The answer is kept rather than replaced. Substituting the calculator's own
    # text would guarantee a correct figure and hide how often the generator
    # fails, and that failure rate is the number Phase 4's RAFT data and Phase
    # 4.5's numeric reward both need. Escalating surfaces it instead.
    if missing_figures:
        LOGGER.error(
            "Generator did not reproduce the computed figure %s — answer escalated.",
            missing_figures[0],
        )
        # `missing_figures[0]` is already a complete sentence carrying its own
        # currency symbol. Prefixing another one produced
        # "₹computed payable ₹600,000 does not appear in the answer".
        update["escalate"] = True
        update["escalation_reason"] = (
            f"{missing_figures[0]} — the stated amount may be wrong"
        )
        update["assumptions"] = [
            *state.get("assumptions", []),
            f"Figure check: {missing_figures[0]}.",
        ]

    return update


# --- Edges -------------------------------------------------------------------


def after_gate(state: AgentState) -> str:
    """Decide where a question goes once the gate has judged the evidence.

    A pure function of the state, so the graph's branching is testable without
    building the graph.

    Args:
        state: The current state. Reads `refused` and `route`.

    Returns:
        The name of the next node, or "end".
    """
    if state.get("refused"):
        return "end"
    if state.get("route") == "comparison":
        return "compare"
    if state.get("route") == "calculation":
        return "calculate"
    return "generate"


def build_graph(resources: GraphResources) -> Any:
    """Wire the nodes into a compiled LangGraph.

    Args:
        resources: Loaded dependencies for every node.

    Returns:
        The compiled graph, ready for `.invoke(state)`.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", lambda state: router_node(state, resources.router))
    graph.add_node("retrieve", lambda state: retrieve_node(state, resources.retrieval))
    graph.add_node(
        "gate", lambda state: gate_node(state, resources.gate, resources.refusal_text)
    )
    graph.add_node("calculate", calculator_node)
    graph.add_node(
        "compare",
        lambda state: comparison_node(state, resources.generator, resources.refusal_text),
    )
    graph.add_node(
        "generate",
        lambda state: generate_node(
            state, resources.generator, resources.refusal_text, resources.top_k
        ),
    )

    graph.set_entry_point("router")
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "gate")
    graph.add_conditional_edges(
        "gate",
        after_gate,
        {"end": END, "compare": "compare", "calculate": "calculate", "generate": "generate"},
    )
    # The calculator never answers on its own — it produces figures the generator
    # then explains. That keeps one place responsible for the user-facing text.
    graph.add_edge("calculate", "generate")
    graph.add_edge("compare", END)
    graph.add_edge("generate", END)

    return graph.compile()


def run(
    question: str,
    resources: GraphResources,
    compiled: Any,
    user_id: str | None = None,
    insurer: str | None = None,
    policy_type: str | None = None,
) -> AgentState:
    """Answer one question end to end.

    Args:
        question: The user's question.
        resources: Loaded dependencies.
        compiled: The compiled graph.
        user_id: Owner of the documents. Defaults to the configured user.
        insurer: Optional metadata filter.
        policy_type: Optional metadata filter.

    Returns:
        The final state, including the answer and the full trace.
    """
    state = new_state(
        question=question,
        user_id=user_id or resources.retrieval.settings["default_user_id"],
        insurer=insurer,
        policy_type=policy_type,
    )
    started = time.perf_counter()
    final: AgentState = compiled.invoke(state)
    LOGGER.info(
        "answered in %.2fs via %s", time.perf_counter() - started, " -> ".join(final.get("trace", []))
    )
    return final


# --- Self-test ---------------------------------------------------------------


def _self_test() -> list[tuple[str, bool, str]]:
    """Exercise the branching and prompt assembly without models or an index.

    The routing decisions are what a graph gets wrong, and they are pure, so they
    are tested directly rather than through a live run.

    Returns:
        One (name, passed, detail) per check.
    """
    results: list[tuple[str, bool, str]] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        """Record one comparison."""
        passed = actual == expected
        results.append((name, passed, f"expected {expected!r}, got {actual!r}"))

    # A refusal must short-circuit regardless of route — including a calculation,
    # where reaching the calculator with no evidence would produce a figure
    # nobody can cite.
    check("refusal ends the run", after_gate({"refused": True, "route": "lookup"}), "end")
    check(
        "refusal beats the calculation route",
        after_gate({"refused": True, "route": "calculation"}),
        "end",
    )
    check("comparison goes to compare", after_gate({"route": "comparison"}), "compare")
    check("calculation goes to calculate", after_gate({"route": "calculation"}), "calculate")
    check("lookup goes to generate", after_gate({"route": "lookup"}), "generate")
    # An out-of-scope question that survived the gate is answerable after all —
    # the router's out-of-scope recall is 0.333 on the golden set, so this path
    # is the recovery for a misroute, not a contradiction.
    check("surviving out_of_scope still generates", after_gate({"route": "out_of_scope"}), "generate")
    check("missing route defaults to generate", after_gate({}), "generate")

    class _Chunk:
        """Minimal chunk stand-in."""

        def __init__(self) -> None:
            self.insurer = "starhealth"
            self.doc_label = "comprehensive"
            self.policy_type = "health"
            self.page = 39
            self.text = "co-payment of 10% of each and every claim"

    plain = build_answer_prompt("Am I covered?", [_Chunk()], None, top_k=5)
    check("plain prompt asks for citations", "citing pages as [p.N]" in plain, True)
    check("plain prompt has no settlement block", "SETTLEMENT CALCULATION" in plain, False)

    result = ClaimResult(payable=216000.0, claimed=240000.0)
    with_calc = build_answer_prompt("What do I get back?", [_Chunk()], result, top_k=5)
    check("calculation prompt carries the block", "SETTLEMENT CALCULATION" in with_calc, True)
    check("calculation prompt forbids recomputing", "do not recompute" in with_calc.lower(), True)
    check("calculation prompt carries the figure", "216,000" in with_calc, True)

    # Indian digit grouping, which the generator tends to use even though the
    # calculator writes Western grouping.
    check("indian grouping of 216000", indian_grouped(216000), "2,16,000")
    check("indian grouping of 240000", indian_grouped(240000), "2,40,000")
    check("indian grouping of 1000000", indian_grouped(1000000), "10,00,000")
    check("short numbers are ungrouped", indian_grouped(500), "500")

    # The exact 2026-08-20 corruption, verbatim. Note the answer ALSO contains a
    # correct "₹240,000" on the line above, which is why the presence check alone
    # passes and the uncomputed-amount check is what catches it.
    corrupted = ClaimResult(payable=240000.0, claimed=240000.0)
    check(
        "the live corruption is caught",
        verify_computed_figures("Claimed: ₹240,000\nPayable: ₹240,0000", corrupted),
        ["₹2,400,000 appears in the answer but was never computed"],
    )
    check(
        "a substring match does not mask the corruption",
        len(verify_computed_figures("Payable: ₹240,0000", corrupted)),
        2,
    )
    check("western grouping passes", verify_computed_figures("You will receive ₹216,000.", result), [])
    check("indian grouping passes", verify_computed_figures("You will receive ₹2,16,000.", result), [])
    check("bare digits pass", verify_computed_figures("You will receive 216000 rupees.", result), [])
    check(
        "an invented deduction is caught",
        verify_computed_figures("₹216,000 after a ₹50,000 deduction.", result),
        ["₹50,000 appears in the answer but was never computed"],
    )
    check(
        "a missing payable figure is caught",
        verify_computed_figures("Your claim has been approved.", result),
        ["computed payable ₹216,000 does not appear in the answer"],
    )
    check("no calculation means nothing to verify", verify_computed_figures("anything", None), [])

    # An ineligible claim has no payable figure to reproduce, so demanding one
    # would flag every correctly-refused claim as corrupted.
    blocked = ClaimResult(
        payable=0.0, claimed=240000.0, eligible=False, rejection_reason="waiting period"
    )
    check(
        "ineligible claim is not checked for a figure",
        verify_computed_figures("This claim is not payable yet.", blocked),
        [],
    )
    # Page numbers and percentages must never be read as money.
    check(
        "citations and percentages are not amounts",
        verify_computed_figures("You get ₹216,000 after the 10% co-pay [p.39].", result),
        [],
    )
    check(
        "'years 5' is not a rupee amount",
        verify_computed_figures("₹216,000. The policy has run 3 years 5 months.", result),
        [],
    )

    return results


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase3_agents.graph",
        description="The Phase 3 agent graph: router, retrieval, gate, calculator, comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the branching checks. No index, no models, no LLM call, no cost.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Answer one question end to end. Costs one LLM call unless the gate refuses.",
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        choices=["chunk_policy", "parent_docs"],
        help="Override retrieval_agent.pipeline.",
    )
    parser.add_argument("--user-id", default=None, help="Override index.default_user_id.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the graph from the command line.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 on a failed self-test or bad usage.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
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

    if not args.question:
        print("Nothing to do. Pass --self-test or --question.")
        return 1

    config = load_config(args.config)
    resources = build_graph_resources(config, pipeline=args.pipeline)
    try:
        compiled = build_graph(resources)
        final = run(args.question, resources, compiled, user_id=args.user_id)
    finally:
        resources.close()

    print(f"\nquestion   : {final['question']}")
    print(f"route      : {final.get('route')} — {final.get('route_reason', '')}")
    print(f"trace      : {' -> '.join(final.get('trace', []))}")
    print(f"confidence : {final.get('confidence')}")
    print(f"refused    : {final.get('refused')}")
    print(f"escalate   : {final.get('escalate')} {final.get('escalation_reason', '')}")
    print(f"\n{final.get('answer', '')}\n")
    print(f"cited      : {final.get('cited_pages')}")
    print(f"invalid    : {final.get('invalid_citations')}")
    for assumption in final.get("assumptions", []):
        print(f"  ! {assumption}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
