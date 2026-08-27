"""Generate calculation RAFT rows with exact labels and zero LLM calls.

**Why this is a separate file from `gen_dataset.py`.** That one asks a model to
write questions and answers, then filters what comes back. This one asks nothing:
it reads real policy terms out of real chunks, runs them through the same
`settle()` the agent uses, and renders the answer deterministically. Different
mechanism, different failure modes, different cost — so a different file.

**Why the labels are better than an LLM's.** The rupee figure comes from
`claims_calculator.settle()`, which is unit-tested to 27 worked examples and is
the same function the served pipeline calls. The page citation comes from the
`PolicyTerm` the extractor read it from. Neither can drift, so every row here is
correct by construction. `gen_dataset.py` rejected 181 of 1,172 candidates on
quality filters; this file's rejection rate for *wrong arithmetic* is structurally
zero.

**What these rows teach, and why the dataset needs them.** Nothing in the 979
lookup/negative/over-refusal rows involves a number being computed. The defect
Phase 3 measured — a generator handed ₹240,000 with an explicit instruction not
to recompute, writing ₹240,0000 — is untouched by prose examples. These rows put
a computed settlement in the context and a correctly-transcribed figure in the
target, which is the only way supervised fine-tuning reaches that behaviour.

**Why the question carries the bill and the policy carries the rate.** That split
mirrors reality: the user knows what they were charged, the document knows what
share the insurer pays. It is also what `term_extraction.merge_terms` implements
(D-26 precedence), so the training rows and the serving path agree about where
each number legitimately comes from.

**Why every question names its policy, and why the wording varies.** The first
run (calc-v1, 2026-08-26) asked about "this policy" and kept 5 rows from 1,316
chunks. Two reasons, both fixed here and both worth not reintroducing. First,
`question_fingerprint` sorts *content words* and discards numbers, so every
co-payment question in the corpus collapsed to a single fingerprint and 48 of 77
candidates were dropped as duplicates — varying the rupee figure cannot fix that,
only varying the wording can, which is what `QUESTION_TEMPLATES` does. Second, an
unqualified "this policy" is unanswerable against a ten-document index and gives
retrieval nothing to steer on; four of the ten documents are HDFC ERGO health
policies, so the question needs the plan label, not just the insurer.

**Why the oracle is matched on (doc_id, page), and injected when missing.**
calc-v1 compared page numbers alone. Page numbers collide across ten documents,
and 3 of the 5 rows it kept cited a page belonging to a different insurer's
policy — the exact cross-document citation this dataset exists to train against.
Matching on both fields fixed that but exposed the real constraint: on calc-v2,
161 of 180 questions did not retrieve the page their own term came from, so
requiring the retriever to supply the oracle threw away 89% of the set. RAFT
builds context as oracle *plus* distractors by construction and never asks a
retriever for the oracle, so a missed page is spliced in and the weakest
distractor dropped. The slot varies with `variant`, because an oracle always at
rank 1 would teach position rather than reading. `oracle_retrieved` versus
`oracle_injected` in the run stats keeps the retrieval failure visible instead
of papering over it — that ratio is a Phase 2 finding and belongs in METRICS.

**Contamination.** Same rule as `gen_dataset.py` — every generated question is
fingerprinted against the golden and agent-task sets and dropped on collision.
The agent set in particular contains hand-written calculation questions, and
training on them would make the Phase 4 benchmark meaningless.

Usage:
    python -m phase4_finetune.gen_calc_examples --help
    python -m phase4_finetune.gen_calc_examples --self-test
    python -m phase4_finetune.gen_calc_examples --dry-run
    python -m phase4_finetune.gen_calc_examples --tag calc-v2
    python -m phase4_finetune.gen_calc_examples --tag calc-v2 --variants 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from phase1_rag.build_eval_set import question_fingerprint
from phase3_agents.claims_calculator import ClaimRequest, ClaimResult, PolicyTerm, settle
from phase3_agents.term_extraction import extract_terms
from phase4_finetune.gen_dataset import (
    DEFAULT_COLLECTION_PREFIX,
    DEFAULT_OUTPUT_DIR,
    RaftExample,
    contexts_from_chunks,
    is_question_worthy,
    load_chunks,
    load_completed,
    load_holdout_fingerprints,
    summarise,
    write_example,
)

LOGGER = logging.getLogger("claimwise.gen_calc_examples")

# Bill amounts in Indian hospitalisation range, written the way a user writes
# them. Sampled with a fixed seed so a re-run reproduces the same dataset.
BILL_AMOUNTS = (75_000, 1_00_000, 1_50_000, 2_40_000, 3_00_000, 4_50_000, 6_00_000)

# Room rents that exceed a typical cap, so the proportionate-deduction rule
# actually fires rather than being recorded as a no-op step.
ROOM_RENTS = (5_000, 6_000, 8_000)

# Policy ages either side of a waiting period, so both the payable and the
# not-yet-payable branch appear in training.
POLICY_AGES = (12, 18, 30, 48, 60)

# Slug to the name a customer would actually write. Four of the ten documents are
# HDFC ERGO health policies, so the insurer alone does not identify a policy —
# every question has to carry the plan label too.
INSURER_NAMES = {
    "bajajallianz": "Bajaj Allianz",
    "hdfcergo": "HDFC ERGO",
    "iciciprulife": "ICICI Pru",
    "nivabupa": "Niva Bupa",
    "sbigeneral": "SBI General",
    "starhealth": "Star Health",
}


def policy_phrase(chunk: dict[str, Any]) -> str:
    """Name the document the way its policyholder would.

    **Why every question must carry this.** The first run generated
    document-agnostic questions ("this policy") against a ten-document index.
    Three consequences, all bad: the question was unanswerable, retrieval had
    nothing to steer on, and `question_fingerprint` — a sorted content-word set
    that discards numbers — collapsed every co-payment question in the corpus to
    one fingerprint, so 48 of 77 candidates were dropped as duplicates.

    Args:
        chunk: A chunk record carrying `insurer`, `doc_label` and `policy_type`.

    Returns:
        A phrase like "HDFC ERGO Optima Secure health", identifying one document.
    """
    insurer = INSURER_NAMES.get(chunk.get("insurer", ""), chunk.get("insurer", "")).strip()
    label = str(chunk.get("doc_label", "")).replace("-", " ").strip().title()
    policy_type = str(chunk.get("policy_type", "")).strip()
    return " ".join(part for part in (insurer, label, policy_type) if part)


def indian_format(amount: float) -> str:
    """Format rupees the way an Indian policy document does.

    2,40,000 rather than 240,000. Used in the question text so the training rows
    look like the questions users actually type.

    Args:
        amount: Rupee amount.

    Returns:
        The amount with Indian digit grouping.
    """
    whole = f"{int(round(amount))}"
    if len(whole) <= 3:
        return whole
    head, tail = whole[:-3], whole[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


# How many differently-worded questions to attempt per (chunk, term) pair.
#
# Varying the rupee figure alone would not help: `question_fingerprint` sorts
# *content words* and discards numbers, so two questions differing only in the
# bill are the same fingerprint. Distinctness has to come from wording, so each
# template carries three phrasings that share no distinctive vocabulary.
VARIANTS_PER_TERM = 3

# One phrasing set per term type. `{policy}` names the document, `{bill}` the
# amount the user was charged, and the policy supplies the rate — the same split
# `term_extraction.merge_terms` implements at serving time (D-26).
QUESTION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "co_pay_percent": (
        "Under my {policy} policy, my hospital bill is Rs.{bill}. "
        "After the co-payment, how much do I actually receive?",
        "I hold the {policy} plan and was charged Rs.{bill} for treatment. "
        "What share must I bear myself under the co-pay clause?",
        "My {policy} cover — the hospital raised an invoice of Rs.{bill}. "
        "Work out the settlement once co-payment is deducted.",
    ),
    "deductible": (
        "Under my {policy} policy, my bill came to Rs.{bill} and a deductible applies. "
        "What is payable?",
        "I have the {policy} plan. Treatment cost Rs.{bill}. "
        "How much does the insurer settle once the deductible is knocked off?",
        "My {policy} cover — an invoice of Rs.{bill} was raised. "
        "Calculate the reimbursement net of the deductible.",
    ),
    "sub_limit": (
        "Under my {policy} policy, my bill for this treatment is Rs.{bill}. "
        "What is the most it will pay for it?",
        "I hold the {policy} plan and was charged Rs.{bill}. "
        "Does the sub-limit restrict what comes back to me?",
        "My {policy} cover — treatment invoiced at Rs.{bill}. "
        "Work out the capped settlement.",
    ),
    "sum_insured": (
        "Under my {policy} policy, my bill is Rs.{bill}. Can it pay all of that?",
        "I hold the {policy} plan and was charged Rs.{bill}. "
        "Does the sum insured stretch far enough?",
        "My {policy} cover — invoice Rs.{bill}. "
        "Calculate what remains payable against the sum insured.",
    ),
    "room_rent_cap_per_day": (
        "Under my {policy} policy, I stayed in a Rs.{rent} a day room "
        "and my bill is Rs.{bill}. What will it pay?",
        "I hold the {policy} plan. My ward cost Rs.{rent} each day "
        "against a total invoice of Rs.{bill}. How much is settled after deduction?",
        "My {policy} cover — accommodation charged at Rs.{rent} daily, "
        "treatment billed Rs.{bill}. Work out the reimbursement.",
    ),
    "waiting_period_months": (
        "Under my {policy} policy, it has been running {age} months "
        "and my bill is Rs.{bill}. Is this claim payable yet?",
        "I bought the {policy} plan {age} months ago and now face a charge "
        "of Rs.{bill}. Has the waiting period elapsed?",
        "My {policy} cover completed {age} months when treatment costing "
        "Rs.{bill} was billed. Can I claim?",
    ),
}


def build_case(
    term_name: str,
    term: PolicyTerm,
    policy: str,
    variant: int = 0,
) -> tuple[str, ClaimRequest] | None:
    """Turn one extracted policy term into a question and a claim to settle.

    Each term type gets a question that *requires* that term to answer. A
    co-payment question with no bill is unanswerable; a room-rent question needs
    both the actual rent and the cap, so the actual is supplied in the question
    and the cap comes from the policy.

    Fully deterministic in `variant` rather than sampled from an RNG: the same
    (term, policy, variant) must yield the same row on a re-run, and the three
    variants must be guaranteed distinct rather than distinct by luck.

    Args:
        term_name: Which term was extracted.
        term: The term, carrying its page.
        policy: The document-identifying phrase from `policy_phrase`.
        variant: Which phrasing to use; strides the amounts so the arithmetic
            varies alongside the wording.

    Returns:
        `(question, request)`, or None for a term type with no template.
    """
    phrasings = QUESTION_TEMPLATES.get(term_name)
    if not phrasings:
        return None
    template = phrasings[variant % len(phrasings)]

    # Stride rather than step, so three variants span the range instead of
    # clustering at the cheap end.
    bill = BILL_AMOUNTS[(variant * 3) % len(BILL_AMOUNTS)]
    bill_text = indian_format(bill)

    if term_name == "co_pay_percent":
        request = ClaimRequest(claimed_amount=bill, co_pay_percent=term)
    elif term_name == "deductible":
        request = ClaimRequest(claimed_amount=bill, deductible=term)
    elif term_name == "sub_limit":
        request = ClaimRequest(claimed_amount=bill, sub_limit=term)
    elif term_name == "sum_insured":
        request = ClaimRequest(claimed_amount=bill, sum_insured=term)
    elif term_name == "room_rent_cap_per_day":
        # The cap must be a real rupee figure for the ratio to mean anything; a
        # zero cap would make the deduction total, which no policy intends.
        if term.value <= 0:
            return None
        above_cap = [rent for rent in ROOM_RENTS if rent > term.value]
        rent = above_cap[variant % len(above_cap)] if above_cap else term.value * 1.5
        return (
            template.format(policy=policy, bill=bill_text, rent=indian_format(rent)),
            ClaimRequest(
                claimed_amount=bill,
                room_rent_per_day=float(rent),
                room_rent_cap_per_day=term,
            ),
        )
    elif term_name == "waiting_period_months":
        # Stride across the boundary so both the payable and the not-yet-payable
        # branch appear rather than one dominating.
        age = POLICY_AGES[(variant * 2) % len(POLICY_AGES)]
        return (
            template.format(policy=policy, bill=bill_text, age=age),
            ClaimRequest(
                claimed_amount=bill,
                policy_age_months=age,
                waiting_period_months=term,
            ),
        )
    else:
        return None

    return template.format(policy=policy, bill=bill_text), request


def render_answer(result: ClaimResult, term: PolicyTerm) -> str:
    """Write the target answer deterministically from the settlement.

    Short, cited, and stating the figure once. Deliberately not
    `ClaimResult.explain()`, which is a multi-line audit breakdown: the model's
    job at serving time is to write the sentence a user reads, and the training
    target has to be that sentence.

    The provisional caveat is included when terms were assumed, because an
    answer that hides its own incompleteness is exactly the behaviour this
    project refuses to train for.

    Args:
        result: The settled claim.
        term: The governing term, for the citation.

    Returns:
        The target answer text.
    """
    citation = term.citation.strip()

    if not result.eligible:
        return (
            f"This claim is not payable yet. {result.rejection_reason}, "
            f"so nothing is payable on a bill of Rs.{indian_format(result.claimed)}."
        )

    reduction = result.claimed - result.payable
    if reduction <= 0:
        body = (
            f"On a bill of Rs.{indian_format(result.claimed)}, {term.label} {citation} "
            f"does not reduce the amount, so Rs.{indian_format(result.payable)} is payable."
        )
    else:
        body = (
            f"{term.label} {citation} applies, so on a bill of "
            f"Rs.{indian_format(result.claimed)} the amount payable is "
            f"Rs.{indian_format(result.payable)} "
            f"(a reduction of Rs.{indian_format(reduction)})."
        )

    if result.assumptions:
        body += (
            " This figure is provisional: the policy terms not found in the passages "
            "above were assumed not to apply."
        )
    return body


def generate(
    chunks: list[dict[str, Any]],
    resources: Any,
    settings: dict[str, Any],
    holdout: set[str],
    seen: set[str],
    out_handle: Any,
    start_index: int,
    max_rows: int | None,
    variants: int = VARIANTS_PER_TERM,
) -> dict[str, int]:
    """Build calculation rows from chunks that contain extractable terms.

    Only a minority of chunks carry a term the calculator can use — most policy
    prose is definitions and exclusions. That is expected: this slice trades
    coverage for exactness.

    Args:
        chunks: Candidate source chunks.
        resources: Loaded `RetrievalResources`.
        settings: Resolved retrieval settings.
        holdout: Fingerprints that must never be trained on.
        seen: Fingerprints already accepted; mutated as rows are written.
        out_handle: Open output file, appended per row.
        start_index: First example number, for resumed runs.
        max_rows: Stop after this many rows, or None for no limit.
        variants: Differently-worded questions to attempt per (chunk, term).

    Returns:
        Counts per outcome.
    """
    from phase3_agents.retrieval_node import retrieve_global

    stats = {
        "chunks_processed": 0,
        "chunks_with_terms": 0,
        "kept": 0,
        # The two together are the retrieval finding: how often the pipeline
        # surfaced the citing page on its own versus how often we had to supply
        # it. calc-v2 measured 19 retrieved against 161 missed.
        "oracle_retrieved": 0,
        "oracle_injected": 0,
        "dropped_duplicate": 0,
        "dropped_holdout": 0,
        "dropped_no_template": 0,
        "dropped_invalid_input": 0,
    }
    index = start_index
    top_k = int(settings["top_k"])

    for chunk in tqdm(chunks, desc="Calculation", unit="chunk"):
        if max_rows is not None and stats["kept"] >= max_rows:
            break
        stats["chunks_processed"] += 1

        # `extract_terms` takes chunk objects, so wrap the dict in the minimal
        # shape it reads via getattr.
        terms = extract_terms([_ChunkView(chunk)])
        if not terms:
            continue
        stats["chunks_with_terms"] += 1

        policy = policy_phrase(chunk)
        source_doc = str(chunk.get("doc_id", ""))

        for term_name, term in terms.items():
            for variant in range(variants):
                if max_rows is not None and stats["kept"] >= max_rows:
                    break

                case = build_case(term_name, term, policy, variant)
                if case is None:
                    stats["dropped_no_template"] += 1
                    continue
                question, request = case

                fingerprint = question_fingerprint(question)
                if fingerprint in holdout:
                    stats["dropped_holdout"] += 1
                    continue
                if fingerprint in seen:
                    stats["dropped_duplicate"] += 1
                    continue

                try:
                    retrieved = retrieve_global(
                        resources,
                        question,
                        user_id=settings["default_user_id"],
                        top_k=settings["top_k"],
                    )
                except Exception as error:  # noqa: BLE001 - one bad row must not lose the run
                    LOGGER.error("Retrieval failed for %r: %s", question[:60], error)
                    continue

                # The context must contain the page the answer cites, or the row
                # teaches the model to cite something it cannot see. This is the
                # same rule `verify_citations` enforces at serving time.
                #
                # Matched on (doc_id, page), never page alone. The first run
                # compared page numbers only, and across ten documents page
                # numbers collide constantly: 3 of the 5 rows it kept cited a
                # page belonging to a *different insurer's* policy — precisely
                # the cross-document citation RAFT exists to train against.
                oracle_id = next(
                    (
                        getattr(candidate, "chunk_id", "")
                        for candidate in retrieved
                        if str(getattr(candidate, "doc_id", "")) == source_doc
                        and int(getattr(candidate, "page", 0) or 0) == term.source_page
                    ),
                    None,
                )
                if oracle_id is None:
                    # Retrieval could not surface the page the term was read
                    # from — which happened to 161 of 180 questions on calc-v2,
                    # and is a finding about retrieval, not a reason to throw the
                    # row away. RAFT builds context as oracle + distractors *by
                    # construction*; it does not ask a retriever to supply the
                    # oracle. So splice the source chunk in and keep the
                    # distractors, which are what teach the model to ignore
                    # another insurer's page.
                    oracle = _OracleChunk(chunk)
                    oracle_id = oracle.chunk_id
                    # Drop the weakest distractor so context size is unchanged.
                    distractors = (
                        list(retrieved[:-1]) if len(retrieved) >= top_k else list(retrieved)
                    )
                    # Vary the slot, or the model learns "the answer is the first
                    # passage" instead of learning to read the passages.
                    slot = variant % (len(distractors) + 1)
                    context_chunks = distractors[:slot] + [oracle] + distractors[slot:]
                    stats["oracle_injected"] += 1
                else:
                    context_chunks = list(retrieved)
                    stats["oracle_retrieved"] += 1

                try:
                    result = settle(request)
                except Exception as error:  # noqa: BLE001
                    LOGGER.error("settle() rejected %r: %s", question[:60], error)
                    stats["dropped_invalid_input"] += 1
                    continue

                example = RaftExample(
                    example_id=f"calc-{index:05d}",
                    question=question,
                    # The oracle is the chunk on the cited page of the source
                    # document — either the retrieved one (which may be a
                    # neighbour on the same page, not the one the term was read
                    # from) or the source chunk spliced in above.
                    contexts=contexts_from_chunks(context_chunks, oracle_id),
                    answer=render_answer(result, term),
                    slice_name="calculation",
                    oracle_present=True,
                    cited_pages=[term.source_page] if term.source_page else [],
                    source_chunk_id=chunk.get("chunk_id"),
                    fingerprint=fingerprint,
                )
                write_example(out_handle, example)
                seen.add(fingerprint)
                stats["kept"] += 1
                index += 1

    return stats


class _ChunkView:
    """Adapt a chunk dict to the attribute access `extract_terms` expects."""

    def __init__(self, chunk: dict[str, Any]) -> None:
        """Expose text and page as attributes.

        Args:
            chunk: A chunk record from `mixed_chunks.jsonl`.
        """
        self.text = chunk.get("text", "")
        self.page = int(chunk.get("page", 0) or 0)


class _OracleChunk:
    """Adapt a chunk dict to the attribute access `contexts_from_chunks` expects.

    Used when the retriever fails to surface the page a term was read from, so
    the chunk can be spliced into the context directly. Separate from
    `_ChunkView` because that one feeds the extractor and needs only text and
    page; this one has to look like a `RetrievedChunk`.

    `score` is 0.0 and deliberately so: this chunk never went through the
    cross-encoder, and inventing a score would put a number in the record that
    no model produced.
    """

    def __init__(self, chunk: dict[str, Any]) -> None:
        """Expose the full retrieved-chunk surface.

        Args:
            chunk: A chunk record from `mixed_chunks.jsonl`.
        """
        self.chunk_id = str(chunk.get("chunk_id", ""))
        self.doc_id = str(chunk.get("doc_id", ""))
        self.filename = str(chunk.get("filename", ""))
        self.insurer = str(chunk.get("insurer", ""))
        self.page = int(chunk.get("page", 0) or 0)
        self.text = str(chunk.get("text", ""))
        self.score = 0.0


# --- Self-test ---------------------------------------------------------------


def _self_test() -> list[tuple[str, bool, str]]:
    """Exercise formatting, templating and answer rendering. No index, no cost.

    Returns:
        One (name, passed, detail) per check.
    """
    results: list[tuple[str, bool, str]] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        """Record one comparison."""
        results.append((name, actual == expected, f"expected {expected!r}, got {actual!r}"))

    check("indian grouping of 240000", indian_format(240000), "2,40,000")
    check("indian grouping of 100000", indian_format(100000), "1,00,000")
    check("indian grouping of 1000000", indian_format(1000000), "10,00,000")
    check("short numbers ungrouped", indian_format(750), "750")

    secure = policy_phrase(
        {"insurer": "hdfcergo", "doc_label": "optima-secure", "policy_type": "health"}
    )
    check("policy phrase names insurer and plan", secure, "HDFC ERGO Optima Secure health")

    co_pay = PolicyTerm(10.0, 39, "Co-payment 10%")
    case = build_case("co_pay_percent", co_pay, secure)
    check("co-pay template builds", case is not None, True)
    question, request = case
    check("question carries a bill", "Rs." in question, True)
    check("question names the policy", secure in question, True)
    check("request carries the term", request.co_pay_percent.value, 10.0)

    # The dedup that collapsed the first run: fingerprints sort content words and
    # discard numbers, so the variants have to differ in *wording*, and two
    # documents sharing a term must not produce one fingerprint between them.
    variant_prints = {
        question_fingerprint(build_case("co_pay_percent", co_pay, secure, v)[0])
        for v in range(VARIANTS_PER_TERM)
    }
    check("variants are distinct fingerprints", len(variant_prints), VARIANTS_PER_TERM)

    star = policy_phrase(
        {"insurer": "starhealth", "doc_label": "comprehensive", "policy_type": "health"}
    )
    check(
        "different policies do not collide",
        question_fingerprint(build_case("co_pay_percent", co_pay, star, 0)[0])
        not in variant_prints,
        True,
    )

    # 240000 * 0.9 = 216000. The arithmetic is settle()'s, not this file's, but
    # the rendering must transcribe it exactly — that is the defect being taught.
    result = settle(ClaimRequest(claimed_amount=240_000, co_pay_percent=co_pay))
    answer = render_answer(result, co_pay)
    check("payable appears in Indian grouping", "2,16,000" in answer, True)
    check("citation appears", "[p.39]" in answer, True)
    results.append(("provisional caveat present", "provisional" in answer, answer[:90]))

    blocked = settle(
        ClaimRequest(
            claimed_amount=240_000,
            waiting_period_months=PolicyTerm(36.0, 31, "PED waiting period"),
            policy_age_months=18,
        )
    )
    blocked_answer = render_answer(blocked, PolicyTerm(36.0, 31, "PED waiting period"))
    check("ineligible claim says not payable", "not payable" in blocked_answer, True)
    results.append(
        ("and states no rupee payout", "2,16,000" not in blocked_answer, blocked_answer[:90])
    )

    # A zero room-rent cap would make the proportionate deduction total, which no
    # policy intends — the template must decline rather than emit it.
    check(
        "zero room cap yields no case",
        build_case("room_rent_cap_per_day", PolicyTerm(0.0, 20, "Room rent"), secure),
        None,
    )
    check(
        "unknown term yields no case",
        build_case("not_a_term", PolicyTerm(1.0, 5, "x"), secure),
        None,
    )

    room = PolicyTerm(4_000.0, 27, "Eligible room rent limit")
    case = build_case("room_rent_cap_per_day", room, secure)
    check("room template builds", case is not None, True)
    _, room_request = case
    results.append(
        (
            "actual rent exceeds the cap",
            room_request.room_rent_per_day > room.value,
            f"{room_request.room_rent_per_day} vs {room.value}",
        )
    )

    view = _ChunkView({"text": "co-payment of 10% of each and every claim", "page": 39})
    extracted = extract_terms([view])
    check("chunk view feeds the extractor", extracted["co_pay_percent"].value, 10.0)

    # The injected oracle has to survive `contexts_from_chunks`, which reads it
    # by attribute — a missing field would serialise as "" into a training row.
    record = {
        "chunk_id": "195eb0499db9_p68_c1",
        "doc_id": "195eb0499db9",
        "filename": "hdfcergo__health__optima-secure.pdf",
        "insurer": "hdfcergo",
        "page": 68,
        "text": "Co-payment of 20% shall apply to each and every claim.",
    }
    serialised = contexts_from_chunks([_OracleChunk(record)], "195eb0499db9_p68_c1")[0]
    check("injected oracle carries its document", serialised["doc_id"], "195eb0499db9")
    check("injected oracle is flagged as oracle", serialised["is_oracle"], True)
    check("injected oracle scores zero", serialised["score"], 0.0)

    # An oracle pinned at rank 1 would teach position instead of reading.
    slots = {variant % 5 for variant in range(VARIANTS_PER_TERM)}
    check("oracle slot varies across variants", len(slots), VARIANTS_PER_TERM)

    return results


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase4_finetune.gen_calc_examples",
        description="Generate calculation RAFT rows with exact labels. No LLM, no cost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the formatting and rendering checks. No index, no models, no cost.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many chunks carry extractable terms, then exit.",
    )
    parser.add_argument("--collection-prefix", default=DEFAULT_COLLECTION_PREFIX)
    parser.add_argument(
        "--chunks-path", type=Path, default=Path("data/processed/mixed_chunks.jsonl")
    )
    parser.add_argument(
        "--max-rows", type=int, default=600, help="Stop after this many rows."
    )
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument(
        "--variants",
        type=int,
        default=VARIANTS_PER_TERM,
        help="Differently-worded questions to attempt per (chunk, term) pair.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tag", default="calc", help="Label in the output filename.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the calculation slice.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.self_test:
        results = _self_test()
        for name, passed, detail in results:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}" + ("" if passed else f"  — {detail}"))
        failures = sum(1 for _, passed, _ in results if not passed)
        print(f"\n{len(results) - failures}/{len(results)} checks passed")
        return 1 if failures else 0

    config = load_config(args.config)

    all_chunks = load_chunks(args.chunks_path)
    worthy = [chunk for chunk in all_chunks if is_question_worthy(chunk)]
    if args.limit_chunks:
        worthy = worthy[: args.limit_chunks]

    if args.dry_run:
        with_terms = sum(1 for chunk in worthy if extract_terms([_ChunkView(chunk)]))
        print("=== DRY RUN (no cost either way — this slice never calls an LLM) ===")
        print(f"chunks total       : {len(all_chunks)}")
        print(f"question-worthy    : {len(worthy)}")
        print(f"carrying a term    : {with_terms}")
        print(f"row ceiling        : {args.max_rows}")
        return 0

    config.setdefault("chunk_policy", {})["collection_prefix"] = args.collection_prefix

    from phase3_agents.retrieval_node import build_resources

    resources = build_resources(config)
    settings = dict(resources.settings)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output or DEFAULT_OUTPUT_DIR / f"raft_{stamp}_{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing, _ = load_completed(out_path)
    holdout = load_holdout_fingerprints()
    seen = {example.fingerprint for example in existing if example.fingerprint}

    print(f"output             : {out_path}")
    print(f"chunks             : {len(worthy)}\n")

    try:
        with out_path.open("a", encoding="utf-8") as handle:
            stats = generate(
                worthy,
                resources,
                settings,
                holdout,
                seen,
                handle,
                start_index=len(existing),
                max_rows=args.max_rows,
                variants=args.variants,
            )
    finally:
        resources.close()

    final, _ = load_completed(out_path)
    summary = summarise(final)

    print("\n=== CALCULATION SLICE ===")
    for key, value in stats.items():
        print(f"{key:24}: {value}")
    print(f"\ntotal rows           : {summary['total']}")
    print(f"by insurer (oracle)  : {summary['by_insurer']}")

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "collection_prefix": args.collection_prefix,
                "seed": args.seed,
                "llm_calls": 0,
                "usd": 0.0,
                "stats": stats,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"metadata             : {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
