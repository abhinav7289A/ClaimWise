"""The policy comparison agent — Phase 3, task 4.

**What this solves.** "Which of my two health policies makes me wait less for
pre-existing conditions?" is not a lookup with a longer answer. It needs one fact
from each of two documents, side by side, and it is wrong in a specific and
dangerous way when only one side arrives: the answer still reads as a
comparison, names one policy, and sounds finished.

**The measured failure.** On the 12 comparison tasks in `agent_tasks.jsonl`
(2026-08-20), a single global top-5 retrieval put *some* correct page in the
prompt for 0.750 of them but covered *every* required document for only 0.250.
That gap is the population of confident half-answers. `retrieval_node.py` fixes
the retrieval half with per-document quotas; this module is the generation half.

**The design rule: structure is computed, prose is generated.** The grouping of
evidence by document, the list of which policies contributed anything, and the
check that the answer actually addresses each of them are all deterministic
Python. The generator is asked only to write the comparison in words. This is the
same division CLAUDE.md fixes for arithmetic — the model is good at language and
unreliable at bookkeeping, so the bookkeeping does not go to the model.

**Why grouped passages instead of the flat Phase 1 block.** A flat list of five
passages from two insurers invites the model to blend them into one policy's
rules. Grouping under an explicit per-policy header makes the boundary a visible
part of the prompt, and makes "the second policy has no evidence here" a thing
the model can see rather than infer from absence.

**Citations stay `[p.N]`.** The temptation is to invent `[Star p.31]` so a
citation names its document. That would fork the grounding contract in
`rag_chain.SYSTEM_PROMPT`, which Phase 4's RAFT dataset is generated against —
one prompt format at training and inference is worth more than prettier
citations. Document attribution is instead verified separately and deterministically
by `verify_document_coverage`.

Usage:
    python -m phase3_agents.comparison_agent --help
    python -m phase3_agents.comparison_agent --self-test
    python -m phase3_agents.comparison_agent --question "..."
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from common.generator import Generator, build_generator
from phase1_rag.rag_chain import CITATION_PATTERN, RetrievedChunk, verify_citations

LOGGER = logging.getLogger("claimwise.comparison_agent")

# Extends the Phase 1 grounding contract rather than replacing it. Rules 1-6 of
# `rag_chain.SYSTEM_PROMPT` still hold; these add what comparison needs and
# nothing else, so the two prompts stay recognisably the same shape.
COMPARISON_SYSTEM_PROMPT = """You are ClaimWise, an assistant that compares the \
user's own insurance policy documents.

Rules you must follow exactly:
1. Answer ONLY from the numbered passages provided. They are the user's actual
   policies. Never use general insurance knowledge.
2. The passages are grouped under a heading per policy. Never attribute a
   passage under one policy to a different one.
3. Cite a page for every factual claim, in the form [p.N], using the page number
   shown on the passage you took the claim from. Name the policy in your own
   words next to the citation, e.g. "Star Health waits 36 months [p.31]".
4. State the position for EACH policy separately before drawing any conclusion.
5. If a policy's passages do not address the question, say so explicitly for that
   policy and do not guess. A partial comparison stated as partial is correct; a
   partial comparison stated as complete is not.
6. Refuse ONLY when NO policy's passages address the question at all. When
   refusing, reply with exactly this sentence and nothing else:
   "{refusal_text}"
7. Quote amounts, percentages, waiting periods and time limits exactly as
   written. Do not compute rupee figures or totals — a separate calculator does
   that.
8. Be concise and factual. No preamble, no disclaimers."""

COMPARISON_PROMPT_TEMPLATE = """{passages}

Question: {question}

Compare the policies above, giving each policy's position separately with page \
citations as [p.N], then answer the question."""


@dataclass
class PolicyEvidence:
    """The evidence one document contributed to a comparison.

    Attributes:
        doc_id: Content hash of the source PDF.
        display_name: What the answer should call this policy, e.g.
            "starhealth health". Built from metadata rather than the filename,
            because the filename is an implementation detail the user never saw.
        insurer: e.g. "starhealth".
        policy_type: e.g. "health".
        chunks: The document's chunks, best first.
    """

    doc_id: str
    display_name: str
    insurer: str
    policy_type: str
    chunks: list[RetrievedChunk] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        """Distinct pages contributed, in rank order."""
        seen: list[int] = []
        for chunk in self.chunks:
            if chunk.page not in seen:
                seen.append(chunk.page)
        return seen

    @property
    def top_score(self) -> float:
        """Best relevance score in this document, or 0.0 when empty."""
        return self.chunks[0].score if self.chunks else 0.0


@dataclass
class ComparisonResult:
    """A comparison answer plus everything needed to evaluate it.

    Attributes:
        question: The question asked.
        answer: The generated comparison.
        policies: Evidence per document, best-scoring document first.
        cited_pages: Pages cited, in order of appearance.
        invalid_citations: Cited pages absent from the retrieved set.
        unaddressed: Policies that contributed evidence but are never named in
            the answer. The comparison-specific defect, measured deterministically.
        complete: True when at least two policies contributed evidence and every
            one of them is addressed.
        refused: Whether the answer is the configured refusal sentence.
        generation_seconds: Time spent in the generator.
        prompt_tokens: Input tokens billed.
        completion_tokens: Output tokens billed.
        provider: Provider that served the generation.
        model: Generator model id.
    """

    question: str
    answer: str
    policies: list[PolicyEvidence] = field(default_factory=list)
    cited_pages: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    unaddressed: list[str] = field(default_factory=list)
    complete: bool = False
    refused: bool = False
    generation_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str = ""
    model: str = ""

    @property
    def documents_compared(self) -> int:
        """How many documents actually contributed evidence."""
        return len(self.policies)


def display_name_for(chunk: RetrievedChunk) -> str:
    """Build the human name for the policy a chunk came from.

    Args:
        chunk: Any chunk from the document.

    Returns:
        A short name, falling back through metadata to the filename so this
        never returns an empty string — an unnamed policy in a comparison is
        worse than an ugly one.
    """
    parts = [part for part in (chunk.insurer, chunk.doc_label) if part]
    if not parts:
        parts = [part for part in (chunk.insurer, chunk.policy_type) if part]
    return " ".join(parts) or chunk.filename or chunk.doc_id


def group_by_document(chunks: list[RetrievedChunk]) -> list[PolicyEvidence]:
    """Group retrieved chunks into one evidence block per document.

    Documents are ordered by their best chunk's score, so the policy the
    retriever was most confident about is presented first. Within a document the
    incoming rank order is preserved — it is already the cross-encoder's.

    Args:
        chunks: Retrieved chunks, best first.

    Returns:
        Evidence per document, strongest document first.
    """
    grouped: dict[str, PolicyEvidence] = {}
    for chunk in chunks:
        if chunk.doc_id not in grouped:
            grouped[chunk.doc_id] = PolicyEvidence(
                doc_id=chunk.doc_id,
                display_name=display_name_for(chunk),
                insurer=chunk.insurer,
                policy_type=chunk.policy_type,
            )
        grouped[chunk.doc_id].chunks.append(chunk)

    return sorted(grouped.values(), key=lambda policy: policy.top_score, reverse=True)


def format_comparison_passages(policies: list[PolicyEvidence]) -> str:
    """Render grouped evidence as per-policy passage blocks.

    Passage numbering runs continuously across policies rather than restarting
    per policy, so "Passage 4" is unambiguous in the model's own reasoning.
    The page number stays in every passage header for the same reason as in
    Phase 1: citing becomes a copy rather than a recall task.

    Args:
        policies: Evidence per document.

    Returns:
        The passage block for the prompt.
    """
    blocks: list[str] = []
    passage_number = 1
    for index, policy in enumerate(policies):
        heading = f"=== POLICY {chr(ord('A') + index)}: {policy.display_name} " \
                  f"({policy.policy_type}) ==="
        lines = [heading]
        for chunk in policy.chunks:
            lines.append(f"[Passage {passage_number} | page {chunk.page}]\n{chunk.text}")
            passage_number += 1
        blocks.append("\n\n".join(lines))
    return "\n\n".join(blocks)


def build_comparison_prompt(question: str, policies: list[PolicyEvidence]) -> str:
    """Assemble the single prompt sent to the generator.

    Args:
        question: The user's question.
        policies: Evidence per document.

    Returns:
        The complete user message.
    """
    return COMPARISON_PROMPT_TEMPLATE.format(
        passages=format_comparison_passages(policies), question=question
    )


def verify_document_coverage(answer_text: str, policies: list[PolicyEvidence]) -> list[str]:
    """Find policies that contributed evidence but are never named in the answer.

    Deterministic and free, so it runs on every comparison — the same bargain as
    `verify_citations`. A policy counts as addressed when any distinctive word
    from its display name appears in the answer, or when one of its own pages is
    cited. Two signals rather than one because the model may name a policy in
    prose the metadata does not predict ("SBI's plan"), or may cite it without
    naming it; requiring both would over-report the defect.

    Args:
        answer_text: The generated comparison.
        policies: Evidence per document.

    Returns:
        Display names of policies the answer never addresses.
    """
    lowered = answer_text.lower()
    cited_pages = {int(match) for match in CITATION_PATTERN.findall(answer_text)}

    unaddressed: list[str] = []
    for policy in policies:
        tokens = [
            token
            for token in policy.display_name.lower().replace("-", " ").split()
            # Words shared by every policy carry no attribution signal, so
            # matching on them would mark every policy as addressed.
            if len(token) > 3 and token not in {"health", "home", "life", "policy", "insurance"}
        ]
        named = any(token in lowered for token in tokens)
        page_cited = bool(set(policy.pages) & cited_pages)
        if not named and not page_cited:
            unaddressed.append(policy.display_name)
    return unaddressed


def compare(
    question: str,
    chunks: list[RetrievedChunk],
    generator: Generator,
    refusal_text: str,
) -> ComparisonResult:
    """Run the comparison: group → prompt → generate → verify.

    Args:
        question: The user's question.
        chunks: Retrieved chunks, ideally from `retrieve_per_document`.
        generator: The swappable generator.
        refusal_text: The exact refusal sentence from config.

    Returns:
        The comparison with its evidence, citation check and coverage check.
    """
    policies = group_by_document(chunks)

    if not policies:
        LOGGER.warning("No chunks to compare for %r — refusing without generating.", question)
        return ComparisonResult(question=question, answer=refusal_text, refused=True)

    prompt = build_comparison_prompt(question, policies)
    system = COMPARISON_SYSTEM_PROMPT.format(refusal_text=refusal_text)

    started = time.perf_counter()
    result = generator.generate(prompt, system=system)
    generation_seconds = time.perf_counter() - started

    cited, invalid = verify_citations(result.text, chunks)
    unaddressed = verify_document_coverage(result.text, policies)
    refused = refusal_text.lower() in result.text.lower()

    if len(policies) < 2:
        LOGGER.warning(
            "Comparison ran against %d document(s) — the answer cannot be a "
            "complete comparison.",
            len(policies),
        )

    return ComparisonResult(
        question=question,
        answer=result.text,
        policies=policies,
        cited_pages=cited,
        invalid_citations=invalid,
        unaddressed=unaddressed,
        complete=len(policies) >= 2 and not unaddressed and not refused,
        refused=refused,
        generation_seconds=round(generation_seconds, 3),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        provider=result.provider,
        model=result.model,
    )


def comparison_node(
    state: dict[str, Any],
    generator: Generator,
    refusal_text: str,
) -> dict[str, Any]:
    """Apply the comparison agent as a graph node.

    Returns a partial state update rather than a mutated state, matching
    `gate_node` and `retrieve_node`.

    Args:
        state: The current `AgentState`. Reads `question` and `retrieved`.
        generator: The swappable generator.
        refusal_text: The exact refusal sentence from config.

    Returns:
        Keys to merge into the state.
    """
    question = state.get("rewritten_question") or state["question"]
    result = compare(question, state.get("retrieved") or [], generator, refusal_text)

    update: dict[str, Any] = {
        "comparison": result,
        "answer": result.answer,
        "cited_pages": result.cited_pages,
        "invalid_citations": result.invalid_citations,
        "refused": result.refused,
        "trace": [*state.get("trace", []), "comparison_agent"],
    }

    assumptions = list(state.get("assumptions", []))
    if result.documents_compared == 1:
        assumptions.append(
            f"Only {result.policies[0].display_name} had relevant passages — "
            "the other policies were searched but contributed no evidence."
        )
    if result.unaddressed:
        assumptions.append(
            "The answer does not address: " + ", ".join(result.unaddressed)
        )
        update["escalate"] = True
        update["escalation_reason"] = (
            f"comparison left {len(result.unaddressed)} policy/policies unaddressed"
        )
    if assumptions:
        update["assumptions"] = assumptions

    return update


# --- Self-test ---------------------------------------------------------------


def _fake_chunk(doc_id: str, insurer: str, page: int, score: float, text: str = "x") -> RetrievedChunk:
    """Build a minimal chunk for the pure-function tests.

    Args:
        doc_id: Document id.
        insurer: Insurer name.
        page: Page number.
        score: Relevance score.
        text: Chunk text.

    Returns:
        A `RetrievedChunk` carrying only the fields these functions read.
    """
    return RetrievedChunk(
        chunk_id=f"{doc_id}_p{page}",
        doc_id=doc_id,
        filename=f"{insurer}.pdf",
        insurer=insurer,
        policy_type="health",
        doc_label="comprehensive",
        page=page,
        text=text,
        score=score,
    )


def _self_test() -> list[tuple[str, bool, str]]:
    """Exercise grouping, prompt assembly and coverage checking.

    No generator, no index, no network, no cost — every function under test is
    pure, which is why they were written that way.

    Returns:
        One (name, passed, detail) per check.
    """
    results: list[tuple[str, bool, str]] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        """Record one comparison."""
        passed = actual == expected
        results.append((name, passed, f"expected {expected!r}, got {actual!r}"))

    star = [_fake_chunk("d1", "starhealth", 31, 0.91, "36 months of continuous coverage")]
    sbi = [_fake_chunk("d2", "sbigeneral", 20, 0.72, "24 months of continuous coverage")]
    mixed = [star[0], sbi[0], _fake_chunk("d1", "starhealth", 32, 0.55)]

    policies = group_by_document(mixed)
    check("grouping splits by document", [p.doc_id for p in policies], ["d1", "d2"])
    check("strongest document first", policies[0].doc_id, "d1")
    check("pages collected in rank order", policies[0].pages, [31, 32])
    check("display name uses metadata", policies[0].display_name, "starhealth comprehensive")

    passages = format_comparison_passages(policies)
    check("one heading per policy", passages.count("=== POLICY"), 2)
    check("policies labelled A and B", ("POLICY A" in passages, "POLICY B" in passages), (True, True))
    check("passage numbering is continuous", "[Passage 3 | page 20]" in passages, True)

    # Coverage: an answer naming only one insurer must flag the other.
    one_sided = "Star Health waits 36 months [p.31]."
    check(
        "one-sided answer flags the missing policy",
        verify_document_coverage(one_sided, policies),
        ["sbigeneral comprehensive"],
    )

    both = "starhealth waits 36 months [p.31] while sbigeneral waits 24 [p.20]."
    check("two-sided answer flags nothing", verify_document_coverage(both, policies), [])

    # A policy cited by page but not named still counts as addressed — the
    # second signal exists precisely for models that write "the other policy".
    by_page_only = "One policy waits 36 months [p.31], the other 24 [p.20]."
    check(
        "citation alone counts as addressing a policy",
        verify_document_coverage(by_page_only, policies),
        [],
    )

    # Shared words must not be treated as attribution, or every policy would
    # look addressed by any answer mentioning insurance.
    generic = "This health insurance policy has a waiting period."
    check(
        "generic words do not count as attribution",
        len(verify_document_coverage(generic, policies)),
        2,
    )

    single = group_by_document(star)
    check("single document groups to one policy", len(single), 1)
    check("empty input groups to nothing", group_by_document([]), [])

    return results


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase3_agents.comparison_agent",
        description="Compare the user's policies side by side, with per-policy citations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the grouping and coverage checks. No index, no models, no LLM call, no cost.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Run a real comparison. Costs one LLM call through the configured provider.",
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
    """Run the comparison agent from the command line.

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

    # Imported here, not at module scope: the comparison logic above is pure and
    # must stay importable without loading torch or opening the index.
    from phase3_agents.retrieval_node import build_resources, retrieve_per_document

    config = load_config(args.config)
    resources = build_resources(config, pipeline=args.pipeline)
    user_id = args.user_id or resources.settings["default_user_id"]
    refusal_text = cfg_get(config, "rag.refusal_text", "Not covered in your documents.")

    try:
        chunks = retrieve_per_document(resources, args.question, user_id)
        generator = build_generator(config)
        result = compare(args.question, chunks, generator, refusal_text)
    finally:
        resources.close()

    print(f"\nquestion  : {result.question}")
    print(f"documents : {result.documents_compared}")
    for policy in result.policies:
        print(f"  - {policy.display_name:<28} pages {policy.pages}")
    print(f"\n{result.answer}\n")
    print(f"cited     : {result.cited_pages}")
    print(f"invalid   : {result.invalid_citations}")
    print(f"unaddress.: {result.unaddressed}")
    print(f"complete  : {result.complete}")
    print(f"tokens    : {result.prompt_tokens} in / {result.completion_tokens} out "
          f"({result.provider} {result.model})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
