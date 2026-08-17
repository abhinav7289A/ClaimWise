"""Generate golden Q&A pairs for manual verification.

The golden set is ground truth: the fixed question set every future change is
measured against. Its quality caps the quality of every number this project ever
reports, which is why generation here is only the first half — a human verifies
each item before it counts.

**The trap this file exists to avoid.** Ask an LLM to "write a question about
this passage" and it inherits the passage's vocabulary. Given "arthroscopic
procedures of the knee joint shall be payable", it writes "What is the coverage
for arthroscopic procedures of the knee joint?" — which retrieval finds
trivially, because it is nearly the same string. hit@5 comes out at 95%, you
conclude retrieval is solved, and Phase 2's real improvements vanish into noise.
You have measured string matching and called it semantic search.

Real users ask "is my knee surgery covered?". Three defences:

1. **Persona and paraphrase** in the generation prompt — write as a policyholder
   who has never opened the document.
2. **A deterministic leakage filter** — reject a question when too many of its
   content words appear in the source chunk. Free, and it catches the model
   ignoring instruction 1, which it will.
3. **Hand-seeded negatives** — questions the corpus cannot answer, taken from a
   fixed list of out-of-scope domains rather than generated, because an LLM
   asked for an unanswerable question sometimes writes an answerable one.

**The load-bearing output field is `ground_truth_pages`.** It turns hit@5, MRR
and context recall into pure-Python comparisons instead of paid API runs, which
is what keeps Phase 2's per-technique re-evaluations affordable.

Usage:
    python -m phase1_rag.build_eval_set --help
    python -m phase1_rag.build_eval_set --dry-run
    python -m phase1_rag.build_eval_set --limit 5
    python -m phase1_rag.build_eval_set
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from common.generator import Generator, build_generator

LOGGER = logging.getLogger("claimwise.build_eval_set")

# Words carrying no topical signal. Excluded from the leakage measurement so it
# reflects real vocabulary reuse rather than shared grammar.
STOPWORDS = frozenset("""
a an and are as at be been being by can could did do does for from had has have
how i if in into is it its may me might must my no not of on or our shall should
that the their then there these they this those to under upon was we were what
when where which who whom why will with would you your
""".split())

# Questions that reference the document instead of standing alone. A real user
# has no idea what a "passage" is, and such questions cannot be asked in the UI.
META_REFERENCE = re.compile(
    r"\b(accord\w*\s+to|the\s+(passage|excerpt|document|text|section|policy\s+wording)"
    r"|mentioned\s+above|as\s+stated|in\s+this\s+(passage|excerpt|document|section))\b",
    re.IGNORECASE,
)

GENERATION_PROMPT = """Below is an excerpt from a real insurance policy document.

--- EXCERPT ---
{chunk_text}
--- END EXCERPT ---

Write ONE question that a real policyholder would ask, whose answer is fully \
contained in the excerpt above.

Requirements:
- Write as a customer who has NEVER read the document. Use everyday language.
  If the excerpt says "arthroscopic procedures of the knee joint", ask about
  "knee surgery". If it says "Insured Person", say "I" or "my family".
- Do NOT copy distinctive phrases or technical wording from the excerpt.
- Do NOT refer to the document. No "according to the passage", no "in this
  section". The question must stand alone.
- The answer must come entirely from the excerpt. No outside knowledge.
- Keep the answer to one or two sentences, quoting any figures exactly as
  written.

Return ONLY this JSON object and nothing else:
{{"question": "...", "answer": "...", "question_type": "lookup"}}

Use "question_type": "calculation" if answering requires applying a stated \
threshold, percentage or limit to a specific situation. Otherwise "lookup"."""

# Negatives are hand-written, not generated. Our corpus covers health, home and
# life policies only, so these domains are guaranteed absent — which means they
# need no verification, unlike an LLM's guess at what it doesn't know.
NEGATIVE_QUESTIONS = [
    "My car was damaged in a collision last week. How much will I get back?",
    "Does this cover my flight being cancelled while I'm abroad?",
    "What's the excess on my two-wheeler insurance claim?",
    "Is my shipping container covered if the cargo is lost at sea?",
    "Can I claim for crop failure after the monsoon flooded my fields?",
    "What is the third-party liability limit for my motorcycle?",
    "Does this policy pay out if my business is sued for professional negligence?",
    "Am I covered for a ransomware attack on my company's servers?",
    "What's the vet bill limit for my dog's surgery?",
    "Does this cover trip cancellation if I miss my connecting train?",
    "How much do I get if my drone crashes into someone's property?",
    "Is my rental car covered when I travel overseas?",
    "What's the payout if my fishing boat sinks?",
    "Does this include cover for my employees' workplace injuries?",
    "Can I claim for damage to my crops from a hailstorm?",
    "What's the premium for adding my motorbike to this policy?",
    "Does this policy cover legal costs if I'm taken to court over a car accident?",
    "Am I insured for cancelled concert tickets?",
    "What happens if my freight is delayed at customs?",
    "Is satellite launch failure covered under this policy?",
]


@dataclass
class GoldenItem:
    """One evaluation question with its ground truth.

    Attributes:
        id: Stable identifier, e.g. "g-007".
        question: The question as a user would ask it.
        answer: Expected answer, for the LLM-judged generation metrics.
        question_type: "lookup", "calculation" or "negative".
        ground_truth_pages: Pages that contain the answer. **Empty for
            negatives.** This field is what makes retrieval metrics free.
        ground_truth_doc_id: Source document's content hash; empty for negatives.
        filename: Source PDF; empty for negatives.
        insurer: Source insurer; empty for negatives.
        policy_type: "health", "home", "life", or "" for negatives.
        source_chunk_id: Chunk the question was generated from, so a reviewer can
            trace any item back to the exact text it came from.
        vocab_overlap: Fraction of the question's content words also present in
            the source chunk. Lower is better; reported so leakage is visible.
        verified: Set to true by a human after review. `run_ragas.py` refuses to
            evaluate unverified items unless explicitly overridden.
        notes: Free-text field for reviewer comments.
    """

    id: str
    question: str
    answer: str
    question_type: str
    ground_truth_pages: list[int] = field(default_factory=list)
    ground_truth_doc_id: str = ""
    filename: str = ""
    insurer: str = ""
    policy_type: str = ""
    source_chunk_id: str = ""
    vocab_overlap: float = 0.0
    verified: bool = False
    notes: str = ""


def content_words(text: str) -> set[str]:
    """Reduce text to lowercase content words for overlap measurement.

    Args:
        text: Any text.

    Returns:
        Lowercased alphanumeric tokens of 3+ characters, minus stopwords.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) >= 3 and token not in STOPWORDS}


def vocabulary_overlap(question: str, chunk_text: str) -> float:
    """Measure how much of a question's vocabulary is lifted from its source.

    A question sharing most of its content words with the chunk is trivially
    retrievable and would inflate hit@5 — it tests string matching, not
    semantic retrieval.

    Args:
        question: The generated question.
        chunk_text: The chunk it was generated from.

    Returns:
        Fraction of the question's content words present in the chunk, 0.0-1.0.
        Returns 1.0 for a question with no content words, so it gets rejected.
    """
    question_words = content_words(question)
    if not question_words:
        return 1.0
    return len(question_words & content_words(chunk_text)) / len(question_words)


def question_fingerprint(question: str) -> str:
    """Reduce a question to a form that makes duplicates comparable.

    The existing leakage filter compares each question against its *source
    chunk*, never against previously accepted questions. Two different sampled
    chunks from the same page can therefore yield the same paraphrase — which is
    exactly what happened: g-069 and g-077 are byte-identical, so the golden set
    held 84 distinct positives while every hit@k was divided by 85, and one
    retrieval failure was counted twice.

    Deliberately an exact match on the sorted content-word set rather than a
    fuzzy similarity threshold. Fuzzy matching would also collapse questions that
    are genuinely distinct: g-002 ("notice before terms change", 30 days,
    starhealth) and g-012 ("notice if product discontinued", 90 days, SBI) share
    most of their wording and have different answers in different documents.
    Rejecting those would silently shrink coverage. Exact fingerprinting catches
    true duplicates and word-order variants while leaving near-misses alone.

    Args:
        question: The generated question.

    Returns:
        A canonical string; equal fingerprints mean duplicate questions.
    """
    return " ".join(sorted(content_words(question)))


def parse_generation(raw: str) -> dict[str, str] | None:
    """Extract the JSON object from a model response.

    Models wrap JSON in markdown fences or prose despite instructions, so we
    locate the outermost braces rather than trusting the whole string to parse.

    Args:
        raw: The model's response text.

    Returns:
        The parsed object, or None if it could not be read.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Read chunk records from JSONL.

    Args:
        chunks_path: Path to `chunks.jsonl`.

    Returns:
        Decoded chunk records.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path.resolve()}. "
            "Run `python -m phase1_rag.chunk` first."
        )
    with chunks_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def allocate_targets(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split a total across policy types by weight, summing exactly.

    Uses the largest-remainder method so rounding never loses or invents an
    item, and every weighted type receives at least its share.

    Args:
        total: Number of items to allocate.
        weights: Policy type to relative weight.

    Returns:
        Policy type to integer target, summing to `total`.
    """
    if not weights:
        return {}
    weight_sum = sum(weights.values()) or 1.0
    exact = {name: total * weight / weight_sum for name, weight in weights.items()}
    allocated = {name: int(value) for name, value in exact.items()}

    leftover = total - sum(allocated.values())
    by_remainder = sorted(exact, key=lambda name: exact[name] - allocated[name], reverse=True)
    for name in by_remainder[:leftover]:
        allocated[name] += 1
    return allocated


def sample_chunks(
    chunks: list[dict[str, Any]],
    quota_by_type: dict[str, int],
    min_chars: int,
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    """Choose candidate source chunks per policy type, spread across pages.

    Returns candidates grouped by type rather than as one flat list. Flattening
    them was a real bug: the generation loop consumed types in order and hit a
    global stop condition before reaching the last one, so the life policy
    contributed zero questions. Grouping lets each type carry its own target.

    Pages are used at most once where possible, so the eval set covers the
    documents rather than clustering on whichever pages happen to be long.

    Args:
        chunks: All available chunks.
        quota_by_type: How many candidate chunks to draw per policy type.
        min_chars: Skip chunks shorter than this — fragments rarely hold a rule.
        rng: Seeded random source, so the sample is reproducible.

    Returns:
        Candidate chunks keyed by policy type.
    """
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        if chunk["char_count"] >= min_chars:
            by_type[chunk["policy_type"]].append(chunk)

    selected: dict[str, list[dict[str, Any]]] = {}
    for policy_type, quota in quota_by_type.items():
        candidates = by_type.get(policy_type, [])
        if not candidates:
            LOGGER.warning("No chunks available for policy_type=%r", policy_type)
            selected[policy_type] = []
            continue
        rng.shuffle(candidates)

        used_pages: set[tuple[str, int]] = set()
        picked: list[dict[str, Any]] = []
        # First pass: one chunk per page. Second pass: relax if quota unmet.
        for allow_repeat_pages in (False, True):
            for chunk in candidates:
                if len(picked) >= quota:
                    break
                key = (chunk["doc_id"], chunk["page"])
                if not allow_repeat_pages and key in used_pages:
                    continue
                used_pages.add(key)
                picked.append(chunk)
            if len(picked) >= quota:
                break

        if len(picked) < quota:
            LOGGER.warning(
                "policy_type=%r yielded only %d of %d requested chunks",
                policy_type, len(picked), quota,
            )
        selected[policy_type] = picked
    return selected


def generate_item(
    chunk: dict[str, Any], generator: Generator, settings: dict[str, Any]
) -> tuple[GoldenItem | None, str]:
    """Generate one candidate question from one chunk and apply quality filters.

    Args:
        chunk: The source chunk.
        generator: The generator to call.
        settings: Resolved eval settings.

    Returns:
        A tuple of (item or None, rejection reason). Reason is "ok" on success.
    """
    result = generator.generate(GENERATION_PROMPT.format(chunk_text=chunk["text"]))
    parsed = parse_generation(result.text)
    if not parsed:
        return None, "unparseable"

    question = str(parsed.get("question", "")).strip()
    answer = str(parsed.get("answer", "")).strip()
    question_type = str(parsed.get("question_type", "lookup")).strip().lower()

    if len(question) < settings["min_question_chars"] or not answer:
        return None, "too_short"
    if META_REFERENCE.search(question):
        return None, "meta_reference"

    overlap = vocabulary_overlap(question, chunk["text"])
    if overlap > settings["max_vocab_overlap"]:
        return None, "vocab_leakage"

    if question_type not in {"lookup", "calculation"}:
        question_type = "lookup"

    return (
        GoldenItem(
            id="",  # assigned once the full set is assembled
            question=question,
            answer=answer,
            question_type=question_type,
            ground_truth_pages=[chunk["page"]],
            ground_truth_doc_id=chunk["doc_id"],
            filename=chunk["filename"],
            insurer=chunk["insurer"],
            policy_type=chunk["policy_type"],
            source_chunk_id=chunk["chunk_id"],
            vocab_overlap=round(overlap, 3),
        ),
        "ok",
    )


def build_negatives(count: int, refusal_text: str, rng: random.Random) -> list[GoldenItem]:
    """Take hand-written questions the corpus cannot answer.

    Args:
        count: How many to include.
        refusal_text: The expected answer — the configured refusal sentence.
        rng: Seeded random source.

    Returns:
        Negative items with empty `ground_truth_pages`.
    """
    pool = NEGATIVE_QUESTIONS.copy()
    rng.shuffle(pool)
    if count > len(pool):
        LOGGER.warning(
            "Requested %d negatives but only %d are hand-written; using all.",
            count, len(pool),
        )
    return [
        GoldenItem(
            id="",
            question=question,
            answer=refusal_text,
            question_type="negative",
            ground_truth_pages=[],
            verified=True,  # hand-written and out-of-domain by construction
            notes="hand-seeded negative; correct behaviour is refusal",
        )
        for question in pool[:count]
    ]


def write_review_markdown(items: list[GoldenItem], chunk_by_id: dict[str, dict[str, Any]], path: Path) -> None:
    """Write a human-readable review file with each question beside its source.

    Verification is the expensive part of building a golden set, so the reviewer
    should never have to cross-reference two files to check one item.

    Args:
        items: The generated items.
        chunk_by_id: Source chunks keyed by chunk id.
        path: Destination markdown file.
    """
    unverified = sum(1 for item in items if not item.verified)
    lines = [
        "# Golden eval set — manual review",
        "",
        f"{len(items)} items, {unverified} awaiting verification.",
        "",
        "For each item check three things: the question sounds like a real",
        "customer, the answer is correct and complete, and the page is right.",
        "Then edit `golden.jsonl` — fix the text if needed and set",
        '`"verified": true`. `run_ragas.py` skips unverified items.',
        "",
        "---",
        "",
    ]
    for item in items:
        pages = ", ".join(f"p.{page}" for page in item.ground_truth_pages) or "— (negative)"
        source = f"{item.insurer} {item.filename}" if item.filename else "no source document"
        lines += [
            f"## `{item.id}` · {item.question_type} · {source} · {pages}",
            "",
            f"**Q:** {item.question}",
            "",
            f"**A:** {item.answer}",
            "",
            f"*verified: {item.verified} · vocab overlap: {item.vocab_overlap}*",
            "",
        ]
        chunk = chunk_by_id.get(item.source_chunk_id)
        if chunk:
            lines += [
                "<details><summary>source chunk</summary>",
                "",
                "```",
                chunk["text"],
                "```",
                "",
                "</details>",
                "",
            ]
        lines += ["---", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase1_rag.build_eval_set",
        description="Generate golden Q&A pairs for manual verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input", type=Path, default=None, help="Chunks JSONL.")
    parser.add_argument("--output", type=Path, default=None, help="Golden set JSONL.")
    parser.add_argument("--provider", default=None, help="Override generator.provider.")
    parser.add_argument("--model", default=None, help="Override the generator model.")
    parser.add_argument(
        "--target", type=int, default=None, help="Override eval.target_questions."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Generate only N questions (smoke test)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show sampling and cost estimate; call nothing."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def resolve_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge config values with CLI overrides.

    Args:
        config: Parsed `config.yaml`.
        args: Parsed command-line arguments.

    Returns:
        The resolved eval settings.
    """
    return {
        "target": args.target or cfg_get(config, "eval.target_questions", 100),
        "negative_fraction": cfg_get(config, "eval.negative_fraction", 0.15),
        "type_weights": cfg_get(config, "eval.type_weights", {"health": 1.0}),
        "max_vocab_overlap": cfg_get(config, "eval.max_vocab_overlap", 0.6),
        "min_question_chars": cfg_get(config, "eval.min_question_chars", 25),
        "min_source_chunk_chars": cfg_get(config, "eval.min_source_chunk_chars", 300),
        "oversample_factor": cfg_get(config, "eval.oversample_factor", 1.6),
        "usd_per_million": cfg_get(config, "eval.estimated_usd_per_million_tokens", 0.7),
        "refusal_text": cfg_get(config, "rag.refusal_text", ""),
        "seed": cfg_get(config, "project.seed", 42),
    }


def main(argv: list[str] | None = None) -> int:
    """Generate the golden set end to end.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if too few items were produced.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    settings = resolve_settings(config, args)
    rng = random.Random(settings["seed"])

    processed_dir = Path(cfg_get(config, "paths.processed_dir", "data/processed"))
    eval_dir = Path(cfg_get(config, "paths.eval_dir", "data/eval"))
    input_path = args.input or processed_dir / "chunks.jsonl"
    output_path = args.output or eval_dir / cfg_get(config, "eval.output_filename", "golden.jsonl")
    review_path = eval_dir / cfg_get(config, "eval.review_filename", "review.md")

    chunks = load_chunks(input_path)

    target = settings["target"]
    negatives_wanted = int(round(target * settings["negative_fraction"]))
    positives_wanted = target - negatives_wanted
    if args.limit is not None:
        positives_wanted = args.limit
        negatives_wanted = min(negatives_wanted, 2)

    # Each policy type carries its own accepted-item target. Enforcing this
    # globally instead starves whichever type is sampled last whenever the
    # generator performs well and few candidates are rejected.
    positive_targets = allocate_targets(positives_wanted, settings["type_weights"])
    # Oversample candidates per type to absorb quality-filter rejections.
    quota_by_type = {
        policy_type: max(1, int(round(target * settings["oversample_factor"])))
        for policy_type, target in positive_targets.items()
    }
    candidates_by_type = sample_chunks(
        chunks, quota_by_type, settings["min_source_chunk_chars"], rng
    )
    total_candidates = sum(len(group) for group in candidates_by_type.values())

    print("=== PLAN ===")
    print(f"chunks available   : {len(chunks)}")
    print(f"positives wanted   : {positives_wanted}")
    print(f"negatives (seeded) : {negatives_wanted}")
    print(f"max attempts       : {total_candidates}")
    for policy_type in sorted(positive_targets):
        print(
            f"  - {policy_type}: target {positive_targets[policy_type]} questions "
            f"from {len(candidates_by_type.get(policy_type, []))} candidate chunks"
        )
    estimated_tokens = total_candidates * 1400
    print(
        f"est. tokens        : ~{estimated_tokens:,} "
        f"(~${estimated_tokens / 1_000_000 * settings['usd_per_million']:.3f})"
    )

    if args.dry_run:
        print("\nDry run — no API calls made.")
        return 0

    generator = build_generator(config, provider=args.provider, model=args.model)
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    items: list[GoldenItem] = []
    rejections: Counter[str] = Counter()
    accepted_by_type: Counter[str] = Counter()
    # Duplicate detection spans policy types on purpose: the same paraphrase can
    # be generated from boilerplate that appears in several documents.
    seen_questions: dict[str, str] = {}

    with tqdm(total=positives_wanted, desc="Generating", unit="q") as progress:
        for policy_type, target in positive_targets.items():
            for chunk in candidates_by_type.get(policy_type, []):
                if accepted_by_type[policy_type] >= target:
                    break
                try:
                    item, reason = generate_item(chunk, generator, settings)
                except RuntimeError as error:
                    LOGGER.error("Generation failed: %s", error)
                    rejections["api_error"] += 1
                    continue
                if item is None:
                    rejections[reason] += 1
                    continue

                fingerprint = question_fingerprint(item.question)
                if fingerprint in seen_questions:
                    LOGGER.debug(
                        "Duplicate question, already generated as %s: %r",
                        seen_questions[fingerprint],
                        item.question,
                    )
                    rejections["duplicate_question"] += 1
                    continue
                seen_questions[fingerprint] = item.id

                items.append(item)
                accepted_by_type[policy_type] += 1
                progress.update(1)

            shortfall = target - accepted_by_type[policy_type]
            if shortfall > 0:
                # Loud: a missing type means that document is unmeasured.
                LOGGER.warning(
                    "policy_type=%r produced %d of %d target questions "
                    "(%d short) — candidates exhausted.",
                    policy_type, accepted_by_type[policy_type], target, shortfall,
                )

    items.extend(build_negatives(negatives_wanted, settings["refusal_text"], rng))
    for index, item in enumerate(items, start=1):
        item.id = f"g-{index:03d}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    write_review_markdown(items, chunk_by_id, review_path)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": settings,
        "items": len(items),
        "positives": sum(1 for item in items if item.question_type != "negative"),
        "negatives": sum(1 for item in items if item.question_type == "negative"),
        "rejections": dict(rejections),
        "by_policy_type": dict(Counter(item.policy_type for item in items)),
        "mean_vocab_overlap": round(
            sum(i.vocab_overlap for i in items if i.source_chunk_id)
            / max(1, sum(1 for i in items if i.source_chunk_id)),
            3,
        ),
    }
    (eval_dir / "golden.meta.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== EVAL SET SUMMARY ===")
    print(f"items written      : {len(items)}")
    print(f"  positives        : {summary['positives']}")
    print(f"  negatives        : {summary['negatives']}")
    print(f"mean vocab overlap : {summary['mean_vocab_overlap']}")
    print(f"rejections         : {dict(rejections) or 'none'}")
    print(f"by policy type     : {summary['by_policy_type']}")
    print(f"golden set         : {output_path}")
    print(f"review file        : {review_path}")
    print(f"\nNEXT: review {review_path} and set \"verified\": true in {output_path}.")

    # A weighted policy type with zero questions means that document is
    # completely unmeasured — an eval set that looks complete but silently
    # cannot detect any regression in it.
    missing = [name for name, target in positive_targets.items() if target > 0 and not accepted_by_type[name]]
    if missing:
        LOGGER.error(
            "No questions generated for policy type(s) %s. Those documents "
            "would be invisible to every metric. Do not verify this set.",
            ", ".join(sorted(missing)),
        )
        return 1

    if summary["positives"] < positives_wanted * 0.7:
        LOGGER.error(
            "Only %d of %d positives survived filtering. Inspect the rejection "
            "counts before verifying.",
            summary["positives"], positives_wanted,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
