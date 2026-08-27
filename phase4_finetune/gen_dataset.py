"""Generate the RAFT training set from our own corpus, retriever and calculator.

**What a RAFT example is.** Not a question-answer pair. Each row is:

    question + [oracle chunk, distractor, distractor, ...] -> grounded, cited answer

The model is being taught a *policy over contexts* — stay inside the passages you
were handed, cite the page you used, decline when the answer is not there. It is
never taught what any policy says. Insurance wordings are revised annually, and a
fact baked into weights goes stale and cannot be cited to a page.

**Why we generate rather than download.** Four public insurance QA datasets were
surveyed (decisions.md D-30). Every one is question-and-answer only, with no
context column, so none can teach grounding at all — training on them would push
insurance facts into the weights, which CLAUDE.md §3 forbids outright.

Generating from our own corpus is not a compromise, it is strictly better:

* **Contexts** come from our own retriever against our own index, so the training
  distribution matches the inference distribution.
* **Distractors** are the chunks the reranker actually returned alongside the
  oracle — near-misses from competing insurers, which are far harder than the
  randomly sampled irrelevant chunks the RAFT paper uses.
* **Negatives cost nothing and are real.** For every generated question we run
  the real retriever. When the source chunk does not come back, that *is* an
  observed "answer not in context" case. No synthesis required.
* **Calculation labels are exact.** `claims_calculator.settle()` produces the
  rupee figure and `term_extraction` supplies the page each term came from, so
  those rows are correct by construction rather than by an LLM's good intentions.

**Why the prompt is imported, not written here.** `SYSTEM_PROMPT` and
`format_passages` come from `phase1_rag/rag_chain.py` — the exact strings used at
serving time. A model fine-tuned on one format and served another degrades in a
way no metric names directly; it simply gets worse and the dataset gets blamed.

**Contamination is the one unrecoverable mistake.** The 92-question golden set
and the 50-task agent set are holdout. If they leak into training, the
fine-tuned-vs-base comparison that justifies the whole phase measures nothing.
Every candidate question is fingerprinted and dropped if it collides with the
golden set or with an already-accepted question.

**Cost discipline.** OpenRouter is the only paid path and ~$1.60 remains. Only
the lookup and comparison slices need a model; calculation, negatives and
over-refusal positives are derived for free. `--dry-run` prints the estimate and
exits without spending, and the run appends to its output file so an interrupted
run resumes instead of paying twice.

Usage:
    python -m phase4_finetune.gen_dataset --help
    python -m phase4_finetune.gen_dataset --self-test
    python -m phase4_finetune.gen_dataset --dry-run
    python -m phase4_finetune.gen_dataset --limit-chunks 5 --tag smoke
    python -m phase4_finetune.gen_dataset --collection-prefix claimwise_train
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from phase1_rag.build_eval_set import question_fingerprint
from phase1_rag.rag_chain import SYSTEM_PROMPT, format_passages, verify_citations

LOGGER = logging.getLogger("claimwise.gen_dataset")

# The expanded 10-document index built for Phase 4. The 4-document
# `claimwise_mx` collection stays frozen so Phase 1-3 numbers remain
# reproducible (D-29), which is why this is a parameter and not a default.
DEFAULT_COLLECTION_PREFIX = "claimwise_train"

DEFAULT_OUTPUT_DIR = Path("data/train")
GOLDEN_PATH = Path("data/eval/golden.jsonl")
AGENT_TASKS_PATH = Path("data/eval/agent_tasks.jsonl")

# Chunks below this are headers, page numbers and addresses; above it they start
# to be two topics at once, which produces questions no single passage answers.
MIN_CHUNK_CHARS = 300
MAX_CHUNK_CHARS = 4000

# A chunk that is mostly digits and punctuation is a premium table or an index.
# Questions generated from those are unanswerable in prose and poison the set.
MIN_ALPHA_RATIO = 0.55

# Boilerplate that repeats across every insurer. These are the same passages
# behind the eight permanent retrieval misses documented in Phase 1 — worth
# excluding as question *sources* while remaining perfectly good distractors.
BOILERPLATE_MARKERS = (
    "grievance redressal",
    "ombudsman",
    "table of contents",
    "toll free",
    "regd. office",
    "registered office",
    "irdai registration",
    "cin:",
)

# An answer longer than this is summarising the whole passage rather than
# answering, which teaches verbosity.
MAX_ANSWER_CHARS = 900

QUESTION_PROMPT = """You are preparing training questions for an insurance policy assistant.

Read the passage below and write {n} DISTINCT questions that this passage alone answers.

Rules:
- Each question must be fully answerable from this passage, with no outside knowledge.
- Write the way a policyholder would ask, in plain language.
- Never refer to "the passage", "the document", "the text" or "above".
- Do not include the answers.
- Vary the phrasing between questions: one direct, one situational.
- Output one question per line. No numbering, no bullets, no preamble.

Passage (from {insurer}, page {page}):
\"\"\"
{text}
\"\"\""""


@dataclass
class RaftExample:
    """One training row: a question, the retrieved context, and the target answer.

    Deliberately stores raw text rather than a rendered chat string. The trainer
    applies the model's chat template, so this file stays valid whether Phase 4
    lands on Qwen3.5-4B or the Qwen3-4B fallback.

    Attributes:
        example_id: Stable identifier, `{slice}-{n:05d}`.
        question: The user-facing question.
        contexts: Retrieved chunks in rank order, each flagged `is_oracle`.
        answer: The target completion.
        slice_name: Which generation strategy produced this row.
        oracle_present: Whether the source chunk survived retrieval.
        cited_pages: Pages the answer cites, after validation.
        source_chunk_id: The chunk the question was written from, when there was
            one. None for calculation rows, which are built from terms.
        fingerprint: Dedupe key from `question_fingerprint`.
    """

    example_id: str
    question: str
    contexts: list[dict[str, Any]]
    answer: str
    slice_name: str
    oracle_present: bool
    cited_pages: list[int] = field(default_factory=list)
    source_chunk_id: str | None = None
    fingerprint: str = ""


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Read the chunk file produced by `chunk_policy`.

    Args:
        path: Path to `mixed_chunks.jsonl`.

    Returns:
        Every chunk as a dict.

    Raises:
        FileNotFoundError: If the chunk file is missing, which means the ingest
            pipeline has not been run for this corpus.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks not found at {path}. Run `python -m phase1_rag.ingest` then "
            f"`python -m phase2_advanced.chunk_policy` first."
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_question_worthy(chunk: dict[str, Any]) -> bool:
    """Decide whether a chunk can source a good question.

    Excludes three kinds of passage that produce unusable questions: too short to
    contain a fact, mostly numeric (premium tables and indexes), or administrative
    boilerplate that repeats across every insurer.

    Rejected chunks are still indexed and still appear as distractors — this
    filter only governs what a question is *written from*.

    Args:
        chunk: A chunk record.

    Returns:
        True if the chunk should source questions.
    """
    text = (chunk.get("text") or "").strip()
    if not MIN_CHUNK_CHARS <= len(text) <= MAX_CHUNK_CHARS:
        return False

    letters = sum(character.isalpha() or character.isspace() for character in text)
    if letters / max(len(text), 1) < MIN_ALPHA_RATIO:
        return False

    lowered = text.lower()
    return not any(marker in lowered for marker in BOILERPLATE_MARKERS)


def parse_questions(raw: str, limit: int) -> list[str]:
    """Pull clean questions out of a model's line-per-question reply.

    Strips numbering and bullets the model adds despite being told not to, and
    keeps only lines that actually end in a question mark — a model that ignores
    the format usually emits a preamble sentence first.

    Args:
        raw: The model's reply.
        limit: Maximum questions to keep.

    Returns:
        Cleaned questions, at most `limit`.
    """
    questions: list[str] = []
    for line in raw.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if cleaned.endswith("?") and len(cleaned) > 15:
            questions.append(cleaned)
        if len(questions) >= limit:
            break
    return questions


def load_holdout_fingerprints() -> set[str]:
    """Fingerprint every holdout question so training can never collide with it.

    **The one unrecoverable mistake in this phase.** If a golden or agent-task
    question reaches training, the fine-tuned-vs-base benchmark measures
    memorisation instead of skill, and there is no way to detect that after the
    fact from the numbers alone.

    Returns:
        Fingerprints of every question in the golden and agent-task sets.
    """
    fingerprints: set[str] = set()
    for path in (GOLDEN_PATH, AGENT_TASKS_PATH):
        if not path.exists():
            LOGGER.warning("Holdout set %s not found — contamination check weakened.", path)
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                question = item.get("question")
                if question:
                    fingerprints.add(question_fingerprint(question))
    LOGGER.info("Loaded %d holdout fingerprints.", len(fingerprints))
    return fingerprints


def load_completed(path: Path) -> tuple[list[RaftExample], set[str]]:
    """Read an existing output file so an interrupted run resumes.

    Generation costs real money, so a crash at row 1,400 must not mean paying for
    those rows twice. The chunk ids already processed are returned so the caller
    can skip them.

    Args:
        path: Output JSONL path.

    Returns:
        `(examples, processed_chunk_ids)`. Empty when the file does not exist.
    """
    if not path.exists():
        return [], set()

    examples: list[RaftExample] = []
    processed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(RaftExample(**row))
            if row.get("source_chunk_id"):
                processed.add(row["source_chunk_id"])

    LOGGER.info("Resuming: %d existing rows, %d chunks already done.", len(examples), len(processed))
    return examples, processed


def contexts_from_chunks(chunks: list[Any], oracle_chunk_id: str | None) -> list[dict[str, Any]]:
    """Serialise retrieved chunks into the record's context list.

    Args:
        chunks: Retrieved chunks, best first.
        oracle_chunk_id: The chunk the question was written from, if any.

    Returns:
        One dict per chunk, in rank order, flagged for which is the oracle.
    """
    return [
        {
            "rank": rank,
            "chunk_id": getattr(chunk, "chunk_id", ""),
            "doc_id": getattr(chunk, "doc_id", ""),
            "filename": getattr(chunk, "filename", ""),
            "insurer": getattr(chunk, "insurer", ""),
            "page": int(getattr(chunk, "page", 0) or 0),
            "text": getattr(chunk, "text", ""),
            "score": round(float(getattr(chunk, "score", 0.0)), 4),
            "is_oracle": getattr(chunk, "chunk_id", "") == oracle_chunk_id,
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]


def validate_answer(
    answer: str,
    chunks: list[Any],
    oracle_page: int | None,
) -> tuple[bool, list[int], str]:
    """Apply the deterministic quality filters to one generated answer.

    No LLM judge anywhere. Four checks, all free, all reusing machinery that
    already guards the serving path:

    1. Non-empty and under the length cap — a long answer is summarising the
       whole passage rather than answering the question.
    2. At least one citation. An uncited answer teaches the model that citing is
       optional, which is the opposite of the intent.
    3. No fabricated citations, via `verify_citations` — every cited page must
       have actually been in the context.
    4. The oracle's page must be among those cited. An answer that cites only a
       distractor got the right words from the wrong passage.

    Args:
        answer: The generated answer.
        chunks: Chunks that were placed in the prompt.
        oracle_page: Page of the source chunk, when there is one.

    Returns:
        `(accepted, cited_pages, reason)`. `reason` is empty when accepted.
    """
    text = answer.strip()
    if not text:
        return False, [], "empty answer"
    if len(text) > MAX_ANSWER_CHARS:
        return False, [], f"answer too long ({len(text)} chars)"

    cited, invalid = verify_citations(text, chunks)
    if not cited:
        return False, [], "no citation"
    if invalid:
        return False, cited, f"fabricated citation(s) {invalid}"
    if oracle_page is not None and oracle_page not in cited:
        return False, cited, f"oracle page {oracle_page} not cited (cited {cited})"

    return True, cited, ""


def drop_category(reason: str) -> str:
    """Map a rejection message to a stable counting key.

    The smoke run on 2026-08-24 rejected 13 of 47 answers and could not say why:
    reasons were logged at DEBUG and only the total was counted. A 28% rejection
    rate is either a prompt problem or an over-strict filter, and those need
    opposite fixes — so the breakdown has to be in the summary, not in a log
    nobody reads.

    Args:
        reason: The message from `validate_answer`.

    Returns:
        One of a small fixed set of categories.
    """
    if reason.startswith("oracle page"):
        return "oracle_page_not_cited"
    if reason.startswith("fabricated"):
        return "fabricated_citation"
    if reason.startswith("no citation"):
        return "no_citation"
    if reason.startswith("answer too long"):
        return "too_long"
    if reason.startswith("empty"):
        return "empty"
    return "other"


def estimate_cost(chunk_count: int, questions_per_chunk: int, config: dict[str, Any]) -> dict[str, float]:
    """Estimate token spend before any of it happens.

    CLAUDE.md §6 requires an estimate before a paid run. The last one in this
    project was 3x low because it ignored how much context each call carries, so
    this counts the passage text explicitly rather than assuming a flat per-call
    size.

    Args:
        chunk_count: Chunks that will source questions.
        questions_per_chunk: Questions requested per chunk.
        config: Parsed `config.yaml`, for per-provider pricing.

    Returns:
        Token counts and a dollar estimate.
    """
    provider = cfg_get(config, "generator.provider", "openrouter")
    usd_in = cfg_get(config, f"generator.providers.{provider}.usd_per_million_input", 0.0)
    usd_out = cfg_get(config, f"generator.providers.{provider}.usd_per_million_output", 0.0)

    # One question-generation call per chunk: ~700 tokens of passage and
    # instructions in, ~120 out for several questions.
    q_in = chunk_count * 700
    q_out = chunk_count * 120

    # One answer call per surviving question, each carrying top_k passages —
    # roughly 1,800 tokens in, ~150 out. Measured from the Phase 3 agent-eval
    # items file, where plain-RAG prompts ran 1,410-2,357 tokens.
    answers = chunk_count * questions_per_chunk
    a_in = answers * 1_800
    a_out = answers * 150

    total_in = q_in + a_in
    total_out = q_out + a_out
    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "usd": round(total_in / 1e6 * usd_in + total_out / 1e6 * usd_out, 4),
    }


def generate_lookup_slice(
    chunks: list[dict[str, Any]],
    resources: Any,
    generator: Any,
    settings: dict[str, Any],
    holdout: set[str],
    seen: set[str],
    questions_per_chunk: int,
    out_handle: Any,
    start_index: int,
    require_oracle_page: bool = True,
) -> dict[str, int]:
    """Generate lookup rows, and harvest natural negatives on the way.

    For each source chunk: ask the model for questions, run each through the real
    retriever, then ask for a grounded answer over whatever came back.

    The negatives are the interesting part and they are free. When the source
    chunk does not survive retrieval, the honest target is the refusal sentence —
    and that is a genuinely observed failure of this retriever on this corpus,
    not a synthesised one. Those rows teach declining, which is the defect P-10
    measures from the other direction.

    Args:
        chunks: Question-worthy source chunks.
        resources: Loaded `RetrievalResources`.
        generator: The swappable generator.
        settings: Retrieval settings.
        holdout: Fingerprints that must never be trained on.
        seen: Fingerprints already accepted; mutated as rows are written.
        questions_per_chunk: Questions to request per chunk.
        out_handle: Open output file, appended to per row so a crash resumes.
        start_index: First example number, for resumed runs.

    Returns:
        Counts per outcome.
    """
    from phase3_agents.retrieval_node import retrieve_global

    refusal_text = cfg_get_refusal(settings)
    stats = {
        "questions": 0,
        "kept_lookup": 0,
        "kept_negative": 0,
        "kept_over_refusal": 0,
        "dropped_duplicate": 0,
        "dropped_holdout": 0,
        "dropped_filter": 0,
        "chunks_processed": 0,
    }
    index = start_index

    for chunk in tqdm(chunks, desc="Lookup", unit="chunk"):
        stats["chunks_processed"] += 1
        try:
            raw = generator.generate(
                QUESTION_PROMPT.format(
                    n=questions_per_chunk,
                    insurer=chunk.get("insurer", "an insurer"),
                    page=chunk.get("page", 0),
                    text=chunk.get("text", "")[:MAX_CHUNK_CHARS],
                )
            ).text
        except Exception as error:  # noqa: BLE001 - one bad chunk must not lose the run
            LOGGER.error("Question generation failed for %s: %s", chunk.get("chunk_id"), error)
            continue

        for question in parse_questions(raw, questions_per_chunk):
            stats["questions"] += 1
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
            except Exception as error:  # noqa: BLE001
                LOGGER.error("Retrieval failed for %r: %s", question[:60], error)
                continue

            if not retrieved:
                continue

            oracle_id = chunk.get("chunk_id")
            oracle_rank = next(
                (
                    rank
                    for rank, candidate in enumerate(retrieved, start=1)
                    if getattr(candidate, "chunk_id", "") == oracle_id
                ),
                None,
            )

            if oracle_rank is None:
                # The retriever did not find the passage this question came from.
                # An observed "answer not in context" case — the target is the
                # refusal, and it costs nothing to produce.
                example = RaftExample(
                    example_id=f"neg-{index:05d}",
                    question=question,
                    contexts=contexts_from_chunks(retrieved, None),
                    answer=refusal_text,
                    slice_name="negative",
                    oracle_present=False,
                    source_chunk_id=oracle_id,
                    fingerprint=fingerprint,
                )
                write_example(out_handle, example)
                seen.add(fingerprint)
                stats["kept_negative"] += 1
                index += 1
                continue

            prompt = f"{format_passages(retrieved)}\n\nQuestion: {question}\n"
            try:
                answer = generator.generate(
                    prompt, system=SYSTEM_PROMPT.format(refusal_text=refusal_text)
                ).text
            except Exception as error:  # noqa: BLE001
                LOGGER.error("Answer generation failed for %r: %s", question[:60], error)
                continue

            oracle_page = int(chunk.get("page", 0) or 0)
            accepted, cited, reason = validate_answer(
                answer, retrieved, None if require_oracle_page is False else oracle_page
            )
            if not accepted:
                LOGGER.debug("Rejected %r: %s", question[:60], reason)
                stats["dropped_filter"] += 1
                category = drop_category(reason)
                stats[f"drop_{category}"] = stats.get(f"drop_{category}", 0) + 1
                continue

            # An oracle that only just survived reranking is a weak-but-sufficient
            # context. Those rows are exactly the P-10 case: the answer IS there,
            # and the model must not refuse. Labelled separately so the mix is
            # auditable rather than assumed.
            slice_name = "over_refusal" if oracle_rank >= 3 else "lookup"
            example = RaftExample(
                example_id=f"{slice_name[:3]}-{index:05d}",
                question=question,
                contexts=contexts_from_chunks(retrieved, oracle_id),
                answer=answer.strip(),
                slice_name=slice_name,
                oracle_present=True,
                cited_pages=cited,
                source_chunk_id=oracle_id,
                fingerprint=fingerprint,
            )
            write_example(out_handle, example)
            seen.add(fingerprint)
            stats["kept_over_refusal" if slice_name == "over_refusal" else "kept_lookup"] += 1
            index += 1

    return stats


def cfg_get_refusal(settings: dict[str, Any]) -> str:
    """Return the exact refusal sentence used at serving time.

    Fixed wording is what makes refusals countable without an LLM judge, so the
    training target must be the same string the pipeline emits.

    Args:
        settings: Resolved retrieval settings.

    Returns:
        The refusal sentence.
    """
    return settings.get("refusal_text") or "That isn't covered in the policy documents you've uploaded."


def write_example(handle: Any, example: RaftExample) -> None:
    """Append one row and flush, so an interrupted run keeps what it paid for.

    Args:
        handle: Open output file.
        example: The row to write.
    """
    handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    handle.flush()


def summarise(examples: list[RaftExample]) -> dict[str, Any]:
    """Describe the finished set, so the mix is inspected rather than assumed.

    Args:
        examples: Every accepted row.

    Returns:
        Counts by slice, by insurer, and the oracle-present ratio.
    """
    by_slice: dict[str, int] = {}
    by_insurer: dict[str, int] = {}
    oracle_present = 0

    for example in examples:
        by_slice[example.slice_name] = by_slice.get(example.slice_name, 0) + 1
        oracle_present += int(example.oracle_present)
        for context in example.contexts:
            if context.get("is_oracle"):
                insurer = context.get("insurer", "unknown")
                by_insurer[insurer] = by_insurer.get(insurer, 0) + 1

    total = len(examples)
    return {
        "total": total,
        "by_slice": dict(sorted(by_slice.items())),
        "by_insurer": dict(sorted(by_insurer.items(), key=lambda item: -item[1])),
        "oracle_present": oracle_present,
        "oracle_present_ratio": round(oracle_present / max(total, 1), 4),
    }


# --- Self-test ---------------------------------------------------------------


class _Chunk:
    """Minimal stand-in for a retrieved chunk, so the test needs no index."""

    def __init__(self, chunk_id: str, page: int, text: str = "", score: float = 0.5) -> None:
        """Store the fields the serialiser and validators read."""
        self.chunk_id = chunk_id
        self.doc_id = "doc"
        self.filename = "f.pdf"
        self.insurer = "starhealth"
        self.policy_type = "health"
        self.page = page
        self.text = text
        self.score = score


def _self_test() -> list[tuple[str, bool, str]]:
    """Exercise the filters and parsers. No index, no models, no LLM call, no cost.

    Returns:
        One (name, passed, detail) per check.
    """
    results: list[tuple[str, bool, str]] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        """Record one comparison."""
        results.append((name, actual == expected, f"expected {expected!r}, got {actual!r}"))

    prose = "The Company shall indemnify hospitalisation expenses. " * 8
    check("prose chunk is question-worthy", is_question_worthy({"text": prose}), True)
    check("short chunk is rejected", is_question_worthy({"text": "Page 4"}), False)
    check(
        "numeric table is rejected",
        is_question_worthy({"text": "1,00,000 2,00,000 5,00,000 " * 40}),
        False,
    )
    check(
        "grievance boilerplate is rejected",
        is_question_worthy({"text": prose + " Grievance Redressal Officer"}),
        False,
    )

    parsed = parse_questions(
        "1. Is knee surgery covered?\n- What is the waiting period?\nHere are questions:\n",
        limit=5,
    )
    check("numbering stripped", parsed[0], "Is knee surgery covered?")
    check("bullets stripped", parsed[1], "What is the waiting period?")
    check("preamble without a ? is dropped", len(parsed), 2)

    # Realistic lengths matter here: the parser drops anything under 15 chars to
    # filter stray fragments, so a limit test built from "A?\nB?\nC?" measures
    # the length guard rather than the limit and reports 0.
    three_questions = (
        "Is knee surgery covered under this policy?\n"
        "What is the waiting period that applies?\n"
        "Are consumables payable on this claim?"
    )
    check("limit is honoured", len(parse_questions(three_questions, limit=2)), 2)
    check("all three parse without a limit", len(parse_questions(three_questions, limit=9)), 3)
    check("a short fragment is dropped", parse_questions("Ok?\n" + three_questions, limit=9)[0],
          "Is knee surgery covered under this policy?")

    chunks = [_Chunk("c1", 39, "co-payment of 10%"), _Chunk("c2", 12, "other")]

    ok, cited, reason = validate_answer("You bear 10% [p.39].", chunks, oracle_page=39)
    check("valid answer accepted", ok, True)
    check("citation captured", cited, [39])

    ok, _, reason = validate_answer("You bear 10%.", chunks, oracle_page=39)
    check("uncited answer rejected", ok, False)
    check("and says why", reason, "no citation")

    ok, _, reason = validate_answer("See [p.99].", chunks, oracle_page=39)
    check("fabricated citation rejected", ok, False)
    results.append(("and names the page", "99" in reason, reason))

    ok, _, reason = validate_answer("Something [p.12].", chunks, oracle_page=39)
    check("citing only a distractor is rejected", ok, False)

    ok, _, reason = validate_answer("x" * 1200 + " [p.39].", chunks, oracle_page=39)
    check("over-long answer rejected", ok, False)

    ok, _, _ = validate_answer("Refused [p.39].", chunks, oracle_page=None)
    check("no oracle page means no oracle check", ok, True)

    serialised = contexts_from_chunks(chunks, "c1")
    check("oracle flagged", serialised[0]["is_oracle"], True)
    check("distractor not flagged", serialised[1]["is_oracle"], False)
    check("rank recorded", serialised[1]["rank"], 2)

    rows = [
        RaftExample("a", "q", serialised, "a", "lookup", True, [39], "c1", "fp1"),
        RaftExample("b", "q", serialised, "a", "negative", False, [], "c2", "fp2"),
    ]
    summary = summarise(rows)
    check("summary counts rows", summary["total"], 2)
    check("summary counts slices", summary["by_slice"], {"lookup": 1, "negative": 1})
    check("summary computes oracle ratio", summary["oracle_present_ratio"], 0.5)

    return results


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase4_finetune.gen_dataset",
        description="Generate RAFT training data from our own corpus, retriever and calculator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the filter checks. No index, no models, no LLM call, no cost.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cost estimate and the source-chunk count, then exit without spending.",
    )
    parser.add_argument(
        "--collection-prefix",
        default=DEFAULT_COLLECTION_PREFIX,
        help="Which index to retrieve against. The 4-document claimwise_mx stays frozen.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/processed/mixed_chunks.jsonl"),
        help="Chunk file that sources the questions.",
    )
    parser.add_argument(
        "--questions-per-chunk", type=int, default=2, help="Questions requested per source chunk."
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Use only the first N question-worthy chunks. Use a small value to smoke-test cheaply.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument("--tag", default="", help="Label in the output filename and metadata.")
    parser.add_argument(
        "--allow-distractor-citation",
        action="store_true",
        help="Accept an answer that cites a retrieved page other than the source chunk's. "
        "The strict default requires the oracle page, on the grounds that an answer "
        "citing only a distractor got the right words from the wrong passage. Relax it "
        "when the drop breakdown shows oracle_page_not_cited dominating, because a "
        "question answerable from a higher-ranked chunk is a real situation, not a defect.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Seed for chunk shuffling.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the dataset.

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
    random.seed(args.seed)

    all_chunks = load_chunks(args.chunks_path)
    worthy = [chunk for chunk in all_chunks if is_question_worthy(chunk)]
    random.shuffle(worthy)
    if args.limit_chunks:
        worthy = worthy[: args.limit_chunks]

    estimate = estimate_cost(len(worthy), args.questions_per_chunk, config)
    provider = cfg_get(config, "generator.provider", "openrouter")

    print("=== COST ESTIMATE (before spending) ===")
    print(f"chunks total       : {len(all_chunks)}")
    print(f"question-worthy    : {len(worthy)} ({100 * len(worthy) / max(len(all_chunks), 1):.1f}%)")
    print(f"questions requested: {len(worthy) * args.questions_per_chunk}")
    print(f"provider           : {provider} / {cfg_get(config, f'generator.providers.{provider}.model', '?')}")
    print(f"tokens             : {estimate['input_tokens']:,} in / {estimate['output_tokens']:,} out")
    print(f"estimated cost     : ${estimate['usd']:.4f}")
    print("OpenRouter budget  : ~$1.60 remaining — the only paid path\n")

    if args.dry_run:
        print("Dry run: nothing spent. Re-run without --dry-run to generate.")
        return 0

    # Point retrieval at the expanded index without disturbing the frozen one.
    config.setdefault("chunk_policy", {})["collection_prefix"] = args.collection_prefix

    from common.generator import build_generator
    from phase3_agents.retrieval_node import build_resources

    resources = build_resources(config)
    settings = dict(resources.settings)
    settings["refusal_text"] = cfg_get(
        config, "rag.refusal_text", "That isn't covered in the policy documents you've uploaded."
    )
    generator = build_generator(config)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = args.output or DEFAULT_OUTPUT_DIR / f"raft_{stamp}{suffix}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing, processed = load_completed(out_path)
    holdout = load_holdout_fingerprints()
    seen = {example.fingerprint for example in existing if example.fingerprint}
    pending = [chunk for chunk in worthy if chunk.get("chunk_id") not in processed]

    print(f"output             : {out_path}")
    print(f"resuming           : {len(existing)} rows, {len(pending)} chunks pending\n")

    try:
        with out_path.open("a", encoding="utf-8") as handle:
            stats = generate_lookup_slice(
                pending,
                resources,
                generator,
                settings,
                holdout,
                seen,
                args.questions_per_chunk,
                handle,
                start_index=len(existing),
                require_oracle_page=not args.allow_distractor_citation,
            )
    finally:
        resources.close()

    final, _ = load_completed(out_path)
    summary = summarise(final)

    print("\n=== DATASET ===")
    for key, value in stats.items():
        print(f"{key:22}: {value}")
    print(f"\ntotal rows           : {summary['total']}")
    print(f"by slice             : {summary['by_slice']}")
    print(f"by insurer (oracle)  : {summary['by_insurer']}")
    print(f"oracle-present ratio : {summary['oracle_present_ratio']}")

    # Yield per source chunk is the number that decides whether a full run is
    # worth its price. The 2026-08-24 smoke returned 34 rows from 50 chunks; at
    # 0.68/chunk the whole corpus yields ~895 rows, which is a very different
    # dataset from the ~2,200 the plan assumed.
    if stats.get("chunks_processed"):
        print(
            f"yield                : {summary['total'] / stats['chunks_processed']:.2f} rows/chunk"
        )

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "collection_prefix": args.collection_prefix,
                "chunks_path": str(args.chunks_path),
                "questions_per_chunk": args.questions_per_chunk,
                "seed": args.seed,
                "generator": {
                    "provider": provider,
                    "model": cfg_get(config, f"generator.providers.{provider}.model", ""),
                },
                "estimate": estimate,
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
