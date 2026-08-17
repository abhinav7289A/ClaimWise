"""The shared state that flows through the Phase 3 agent graph.

**Why a state object at all.** Phase 1 and 2 were a straight line: retrieve,
rerank, prompt, generate. A graph is not a line — the router sends a question one
of four ways, the confidence gate can short-circuit to a refusal, and the
comparison path visits retrieval twice. Once control flow branches, every node
needs a single agreed place to read its inputs and write its outputs, or the
nodes end up coupled to each other's signatures and the graph stops being
rearrangeable.

**Why it is defined before any node exists.** This is the contract. Writing it
first is what lets the calculator, the gate and the router be built and tested
independently, in any order, without one of them dictating another's interface.

**Why plain `TypedDict` and no LangGraph import.** LangGraph consumes an
ordinary `TypedDict`, so nothing here needs the library — which means this module
imports in a bare environment, the tools that depend on it stay unit-testable
before any graph is wired, and `phase3_agents` does not pull a dependency in
until the module that genuinely needs it arrives.

**Design rule: every field is either evidence or a decision, and both are
recorded.** `trace` and `assumptions` exist so an answer can be explained after
the fact. An agent that produces a correct number without being able to say
which nodes ran and what it assumed is not auditable, and for a claims figure
that is the difference between a product and a demo.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# The router's four destinations, from CLAUDE.md Phase 3 task 1.
#
# `out_of_scope` is a first-class route rather than a failure mode: the golden
# set carries 15 hand-seeded unanswerable questions precisely because refusing
# well is a product requirement, not an error path.
Route = Literal["lookup", "calculation", "comparison", "out_of_scope"]


class AgentState(TypedDict, total=False):
    """Everything a node may read or write.

    `total=False` because nodes populate this progressively — a state with only
    `question` set is the valid input to the graph, and requiring every key up
    front would force meaningless placeholders.

    Attributes:
        question: The user's question, as asked. Never rewritten in place; a
            rewritten form goes in `rewritten_question` so the original survives
            for logging and for the eval harness.
        rewritten_question: Optional normalised form, if a future node adds one.
        user_id: The security boundary. Every retrieval filters on it, and it is
            carried in state rather than read from config so a server handling
            concurrent users cannot leak across requests.
        insurer: Optional metadata filter.
        policy_type: Optional metadata filter.

        route: Where the router sent this question.
        route_reason: Why. Recorded because router accuracy is a Phase 3 metric,
            and a misroute is far easier to diagnose with the reasoning attached.

        retrieved: Chunks from the Phase 2 pipeline, best first. Typed as `Any`
            to avoid importing `phase1_rag` here — the tools that consume it
            already know its shape, and the state stays a contract rather than a
            dependency hub.
        confidence: Cross-encoder top-1 score. P-14 measured 0.0985 mean on
            unanswerable questions against 0.85-0.99 on genuine hits, which is
            what makes this usable as a gate at ~0.20-0.25.

        calculation: A `ClaimResult` when the calculator ran.
        comparison: Structured output when the comparison agent ran.

        answer: The final text shown to the user.
        cited_pages: Pages cited in `answer`.
        invalid_citations: Cited pages absent from `retrieved` — fabricated, and
            free to detect. The same signal Phase 4.5's reward function targets.
        refused: Whether the answer is the exact configured refusal sentence.

        escalate: Whether this question should go to a human.
        escalation_reason: Why, for the monitoring dashboard in Phase 5.

        trace: Node names in visit order. Drives "avg steps per task" and the
            streamed status events the Phase 6 mascot consumes.
        assumptions: Anything a node had to assume. A claim figure computed with
            an assumed co-payment must say so — an unqualified number is worse
            than a qualified one.
        error: Set when a node fails. Present so the graph can route to a
            graceful response instead of raising into the request handler.
    """

    question: str
    rewritten_question: str
    user_id: str
    insurer: str | None
    policy_type: str | None

    route: Route
    route_reason: str

    retrieved: list[Any]
    confidence: float

    calculation: Any
    comparison: Any

    answer: str
    cited_pages: list[int]
    invalid_citations: list[int]
    refused: bool

    escalate: bool
    escalation_reason: str

    trace: list[str]
    assumptions: list[str]
    error: str


def new_state(
    question: str,
    user_id: str,
    insurer: str | None = None,
    policy_type: str | None = None,
) -> AgentState:
    """Build the initial state for one question.

    A constructor rather than a bare dict literal so the collection fields are
    always present and empty. Nodes can then append without a `setdefault`
    dance, and a missing key means a genuine bug rather than "nobody has written
    to it yet".

    Args:
        question: The user's question.
        user_id: Owner of the documents that may be retrieved.
        insurer: Optional metadata filter.
        policy_type: Optional metadata filter.

    Returns:
        A state with the inputs set and every collection initialised.
    """
    return AgentState(
        question=question,
        user_id=user_id,
        insurer=insurer,
        policy_type=policy_type,
        retrieved=[],
        cited_pages=[],
        invalid_citations=[],
        refused=False,
        escalate=False,
        trace=[],
        assumptions=[],
    )
