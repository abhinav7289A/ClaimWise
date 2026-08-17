# Failure Analysis — Phase 1 retrieval baseline

Phase 1 exit criterion (CLAUDE.md §5): the ten worst failures, analysed.

Written 2026-08-17, after Phase 2 completed, which turned out to be the right
order. Several of these failures survived *every* Phase 2 technique — dense
retrieval, four hybrid configurations, parent-document retrieval, and the
adopted density-based chunk policy. A failure that nine different pipelines
cannot fix is usually not a pipeline failure, and that is what this document
concludes.

**Method.** Failures are taken from the per-item JSONL of the adopted pipeline
run (`provisional_retrieval_20260815T184853Z_mx-rr20_items.jsonl`) and
cross-referenced against `data/eval/golden.jsonl`. "Complete miss" means the
correct `doc_id:page` never appeared anywhere in the retrieved list, at any
rank. Every claim below is traceable to those two files.

---

## The nine complete misses

Identical across the rerank@20 baseline and the adopted chunk-policy pipeline —
the same nine items, which is itself a finding.

| ID | Question (abbreviated) | Doc | GT page | Vocab overlap |
|---|---|---|---|---|
| g-002 | Notice before terms/price change | starhealth health | 42 | 0.500 |
| g-012 | Notice if product discontinued | sbi health | 23 | 0.429 |
| g-017 | How can I get help with questions | starhealth health | 45 | 0.200 |
| g-030 | Customer care phone number | sbi health | 27 | 0.400 |
| g-041 | If I don't tell the truth when applying | sbi health | 2 | **0.000** |
| g-058 | Toll-free customer care number | sbi home | 15 | 0.333 |
| g-069 | Law for naming a beneficiary | icici life | 5 | 0.111 |
| g-076 | How long to revive a discontinued policy | icici life | 1 | 0.400 |
| g-077 | Law for naming a beneficiary | icici life | 5 | 0.111 |

**All nine are `question_type: lookup`.** None is a calculation or comparison
question. And all nine ask about **administrative and regulatory boilerplate** —
notice periods, contact details, grievance routes, nomination law, revival
windows — rather than the coverage clauses, waiting periods and sub-limits the
product exists to answer. That distribution is not random and is explained
below.

---

## Category 1 — Eval-set defects (3 items)

These are not retrieval failures. They are questions the golden set should not
have contained in their current form.

> **CORRECTION 2026-08-17.** This section said there was *one* duplicate. There
> were **eight** — `g-061`, `g-063`, `g-073`, `g-074`, `g-075`, `g-077`,
> `g-079`, `g-082`. The claim below was generalised from the single pair
> inspected by hand rather than counted. See P-19. The measured impact was still
> negligible (hit@5 0.8706 → 0.8701 after cleaning), but the estimate was right
> by luck, not by method.

### F-1 · g-069 and g-077 are the same question

Byte-identical: same `question`, same `answer`, same `ground_truth_pages: [5]`,
same `source_chunk_id: 2c3eee38a579_p5_c11`.

**Consequence.** There are **84 distinct positives, not 85**. One retrieval
failure is counted twice, so it depresses every hit@k figure in this project by
roughly 1.2 points. The deduplication filter in `build_eval_set.py` checks
vocabulary overlap against the *source chunk*, not against previously accepted
*questions* — two different sampled chunks from the same page can therefore
produce the same paraphrase.

**Fix.** Add question-to-question deduplication to `build_eval_set.py` before
Phase 3 relies on this set.

### F-2 · g-030 and g-058 are unanswerable as posed

| | g-030 | g-058 |
|---|---|---|
| Question | "phone number ... for customer care" | "toll-free number ... for customer care" |
| Expected answer | `18001021111` | `18001021111` |
| Ground truth | `5b8f496626be:27` (sbi **health**) | `478aa61fb6e3:15` (sbi **home**) |

**The same insurer prints the same number in two different policies.** Neither
question names a policy, so nothing in the question distinguishes them. The
"correct" document is arbitrary, and a retriever that returned the other one
would be marked wrong while giving the user the right phone number.

**Confirmed in the retrieval log.** Both questions pull candidates from **all
four documents**. For g-058 the retrieved set includes `478aa61fb6e3` pages 2,
3, 5, 10, 12 and 17 — the correct document, with pages **12 and 17 bracketing
the ground-truth page 15** without hitting it.

**Fix.** Either scope such questions to a document ("in my SBI home policy…"),
or score them as correct if *any* document containing the answer is retrieved.
The second is closer to what a user wants.

---

## Category 2 — Boilerplate ambiguity (4 items)

g-002, g-012, g-017, g-030 · *the structural finding*

Administrative content — grievance procedures, contact details, notice periods,
ombudsman addresses — appears in **every policy document**, phrased almost
identically because it is regulatory text that insurers copy from the same IRDAI
templates. Four documents therefore contain four near-identical passages for
each of these topics.

Two consequences follow, and both are visible in the retrieval log:

1. **Cross-document spray.** For g-030 the top candidates come from all four
   documents (`2c3eee38a579`, `b1dbe8fb7864`, `478aa61fb6e3`, `5b8f496626be`).
   The embedding cannot prefer one, because they genuinely say the same thing.
2. **Within-document near-misses.** g-030 wanted sbi-health page 27 and
   retrieved pages 2, 22, 28 and 28 of that same document. The right
   neighbourhood, the wrong page — because contact blocks recur across pages
   inside a document too.

g-002 and g-012 are the same problem in a different guise: both ask "how much
notice will I get if…", one about changed terms (30 days, starhealth p42) and
one about product discontinuation (90 days, sbi p23). Near-identical questions,
different documents, different answers.

**This explains the whole distribution.** The pipeline fails on the content that
is *least* specific to any one policy, and succeeds on the coverage clauses that
are unique to each. For a product that answers "is my knee surgery covered", the
failures are concentrated in exactly the material that matters least.

---

## Category 3 — Everyday-to-legal vocabulary mismatch (2 items)

### F-3 · g-041 — vocabulary overlap 0.000

> "What happens if I don't tell the truth or leave out important information
> when applying for insurance?"

The document says: *"misrepresentation, mis-description or non-disclosure of any
material fact"*. **Not one content word is shared.** Retrieval returned pages 7,
8, 23, 24, 28, 36, 39 and 40 of the right document; the answer is on page 2.

### F-4 · g-069 / g-077 — vocabulary overlap 0.111

> "What law applies to how I name someone to get my policy money…"

The document says *"nomination"* and *"Section 39 of the Insurance Act, 1938"*.
The question never uses the word "nomination".

**These two are the clearest case for query rewriting**, the technique D-20
deliberately skipped. That decision remains defensible — the eval set
deliberately paraphrases vocabulary away, so a gain would be hard to attribute —
but these are the items to target if it is ever revisited, and pool recall
rather than hit@5 is the metric that would show it working.

---

## Category 4 — The generation-side defect (the tenth failure)

Not a retrieval failure, and by volume the most serious thing in this document.

From `mx-rr20-gen`: retrieval supplies evidence for **74 of 85** positives. On
the 11 where it fails, the model should refuse. It refuses once and **answers
ten times** — a 91% hallucination rate on unsupported questions.

Phase 2 halved the absolute count (20 of 26 → 10 of 11) purely by retrieving
better, but the *behaviour* is unchanged. The model does not know it is
guessing.

**This is already solved in principle and not yet wired.** P-14 measured the
signal: cross-encoder top-1 scores **0.0985** on negatives against **0.85–0.99**
on genuine hits, and 14 of 15 negatives fall below 0.22. A threshold at
0.20–0.25 would catch most of these ten before the generator is ever called.
That is Phase 3's confidence gate, and this analysis says build it early.

---

## What this means for the recorded metrics

**hit@5 = 0.871 is understated.** Of the nine complete misses:

| | Items | |
|---|---|---|
| Eval-set defects | 3 | g-069/g-077 duplicate, g-030 + g-058 ambiguous |
| Genuine retrieval failures | 6 | g-002, g-012, g-017, g-041, g-069/077 (one), g-076 |

Correcting only the duplicate takes the denominator to 84. Treating the two
phone-number questions as satisfied by any document containing the answer would
lift hit@5 further. **The honest statement is that the true figure is somewhat
above 0.871, and the eval set — not the retriever — is the limiting factor on
knowing by how much.**

This is the standard trajectory for a golden dataset: it is only after the
pipeline gets good that the dataset's own defects become the dominant error
term. An evals team at an insurer would call this the point where the golden set
needs its second pass.

---

## Actions

| # | Action | Status |
|---|---|---|
| 1 | Question-to-question dedup in `build_eval_set.py`; remove g-077 | ✅ built · ⏸ **not applied** |
| 2 | Scope the ambiguous contact-detail questions | ✅ built · ⏸ **not applied** |
| 3 | Confidence gate on cross-encoder top-1 at ~0.20–0.25 | **Phase 3** |
| 4 | Re-run retrieval metrics on the cleaned set; record a corrected baseline | with 1–2 |
| 5 | If query rewriting is revisited, target g-041 and g-069 and measure pool recall | optional |

**Actions 1 and 2 are implemented but deliberately not run (2026-08-17).** The
code is `evals/clean_golden.py` plus `question_fingerprint()` in
`build_eval_set.py`; applying it is `--write` and one re-eval.

**Why deferring is defensible.** The duplicate is worth ~1.2 points of hit@k and
changed no decision in Phase 2 — every technique was compared against every
other on the *same* set, so the bias is common to all of them and cancels in the
deltas that drove the decisions. Phase 3 builds its own 50-task agent eval set
rather than reusing this one.

**Triggers that make it necessary.** Run it before any of:

- publishing an absolute hit@k figure outside this repo;
- Phase 4's RAFT dataset generation, which samples from this set and would
  otherwise train on a duplicated question and two unanswerable ones;
- any future eval-set expansion, so the dedup filter is in force before new
  items are generated.

Until then, every absolute number in `METRICS.md` carries a known ~1.2-point
understatement, recorded here rather than silently absorbed.
