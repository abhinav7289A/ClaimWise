# ClaimWise — Decisions & Problem Log

The project's engineering history: every design decision with the options that
were rejected, and every problem hit with how it was diagnosed and what it cost.

**How this differs from the other docs**

| Doc | Answers |
|---|---|
| `workflow.md` | *What does this file do?* — one paragraph per file |
| `explaination.md` | *How does the system work, and what did it measure?* |
| `decisions.md` | *Why is it this way, and what went wrong getting here?* |

**Status legend:** ✅ Resolved · 🔶 Open, deliberately deferred · ❌ Unresolved · ⏳ Pending measurement

---

# Phase 1 — Naive RAG Baseline

## Decisions

### D-1 · Text extraction mode: plain text, made switchable
**Date:** 2026-08-10

**Context.** A PDF has no paragraphs or tables, only positioned glyphs. How much
structure should ingestion try to recover?

| Option | Gains | Costs |
|---|---|---|
| `get_text("text")` | Simplest, fastest, truest naive baseline | Tables collapse; columns interleave |
| `get_text("blocks")` | Paragraph fidelity, free bounding boxes | Slightly more code, still no table structure |
| `pymupdf4llm.to_markdown()` | Real table extraction, headings become `#` | Extra dependency, 5–10× slower, mangles merged cells |

**Decision.** Default to plain `text`, but expose the mode as a config parameter
rather than hardcoding the call.

**Rationale.** Table-aware parsing is a *Phase 2 experiment*. Baking it into the
baseline would permanently forfeit the ability to report "table-aware parsing
bought us N points of context recall" — one of the more interesting numbers this
project can produce. The switch costs about six lines and buys a free experiment.

**Vindicated by.** [P-9](#p-9--column-scrambling-visible-in-extracted-text) —
column scrambling was observed in real output within hours, giving the Phase 2
experiment a concrete, reproducible test case.

---

### D-2 · Corpus: health-weighted, multi-insurer
**Date:** 2026-08-10

**Context.** How many documents, of what type, for a ~100-question golden set?

**Decision.** 3–5 health policy wordings from different insurers.

**What actually happened.** The delivered corpus is **2 health + 1 home + 1
life**. Two health wordings from different insurers is the *minimum* that makes
Phase 3's policy-comparison agent possible, so this is workable — but home and
life will have roughly 25 questions each, which is too thin for per-domain RAGAS
numbers to be stable.

**Consequence.** The eval set should be weighted ~50 health / 25 home / 25 life,
and per-domain metrics outside health must be reported as indicative only.

---

### D-3 · A shared `common/` package
**Date:** 2026-08-10

**Context.** `CLAUDE.md`'s repo structure lists only phase directories, but every
phase needs to load `config.yaml`.

**Options.** Duplicate the loader per phase · have `phase2_advanced` import from
`phase1_rag` · add a top-level `common/`.

**Decision.** Add `common/`, flagged explicitly to the user as a deviation.

**Rationale.** Cross-phase imports create a dependency direction that becomes
painful the moment Phase 1 is refactored. Duplication guarantees drift.

---

### D-4 · Chunking: recursive splitting at 1000/150
**Date:** 2026-08-10

**Context.** Pages average ~4,800 characters — roughly 1,200 tokens, well past
what one vector can represent usefully and past bge-small's 512-token input.

| Option | Trade-off |
|---|---|
| Fixed-size slicing | Simplest and most deterministic; cuts mid-sentence and mid-table |
| **Recursive splitting** | Same size control, breaks land on real boundaries |
| Structure-aware (headings) | Best coherence, but extraction scrambles heading order (see [P-9](#p-9--column-scrambling-visible-in-extracted-text)) |

**Decision.** Recursive, 1,000 characters with 150 overlap (15%).

**Rationale.** Unlike D-1, this consumes no future experiment — none of Phase 2's
planned techniques is "recursive vs fixed splitting". A baseline should be what
a competent team would actually start from, not a deliberately weakened
strawman. 1,000 chars ≈ 250 tokens leaves headroom under the 512-token limit so
nothing is silently truncated; 150 overlap means a clause straddling a boundary
survives intact in one of the two neighbours.

**Revisit trigger.** Eval evidence only, never intuition.

---

### D-5 · Chunks never span a page break
**Date:** 2026-08-10

**Decision.** Split each page independently, so every chunk inherits exactly one
page number.

**Rationale.** Citations must be verifiable. "Page 34" has to actually be page
34, and a chunk spanning pages 34–35 makes that ambiguous.

**Accepted cost.** A clause continuing onto the next page is divided. The correct
fix is Phase 2's parent-document retrieval, **not** a larger chunk size — raising
the chunk size to paper over this would degrade retrieval precision everywhere
else to solve a boundary problem.

---

### D-6 · Embedding model: bge-small, with an A/B deferred
**Date:** 2026-08-10

| Option | Dim | CPU cost | Quality |
|---|---|---|---|
| **bge-small-en-v1.5** | 384 | ~37 s for 653 chunks | baseline |
| bge-base-en-v1.5 | 768 | ~3× | ~1.5 MTEB retrieval points better |

**Decision.** Index with bge-small now; A/B against bge-base once the golden eval
set exists so the comparison uses real questions rather than vibes.

**Rationale.** ~1.5 MTEB points is less than reranking should buy in Phase 2, so
spending the quality budget here first optimises the wrong stage. At 653 chunks a
full re-index costs about a minute, making the decision cheap to revisit — which
is itself the argument for not agonising over it now.

**Enabled by.** [D-9](#d-9--collection-names-encode-the-embedding-model).

---

### D-7 · Qdrant embedded on-disk
**Date:** 2026-08-10

**Options.** In-memory (vanishes on exit) · **embedded on-disk** · Docker server.

**Decision.** Embedded on-disk at `qdrant_storage/`.

**Rationale.** No Docker dependency, persists between runs, and the client API is
identical to server mode — so Phase 5's move to a hosted instance is a config
change rather than a rewrite.

**Known limitation.** Single-process. Two terminals cannot open the store at
once, which produces a confusing "already accessed by another instance" error.

---

### D-8 · `user_id` filtering from day one
**Date:** 2026-08-10

**Decision.** Write `user_id` into every point's payload and filter on it in
every search, even though Phase 1 has exactly one user.

**Rationale.** This is a security boundary, not an optimisation. Two things go
wrong if it's added later: filtering *after* search returns fewer results than
requested, and the day one code path forgets the post-filter, a customer sees
another customer's policy. Qdrant applies the filter during index traversal, so
a chunk you don't own is never scored at all. A boundary retrofitted later gets
missed in exactly one place.

---

### D-9 · Collection names encode the embedding model
**Date:** 2026-08-10

**Decision.** `claimwise__baai_bge_small_en_v1_5` rather than `claimwise`.

**Rationale.** Two embedding models can then be indexed side by side and queried
with the same golden questions, instead of one silently overwriting the other.
This is what makes [D-6](#d-6--embedding-model-bge-small-with-an-ab-deferred)'s
deferred A/B possible at all.

**Related.** Point IDs are `uuid5(chunk_id)` — deterministic, so re-indexing is
an idempotent upsert rather than a duplication.

---

### D-10 · Two-tier evaluation: deterministic retrieval, judged generation
**Date:** 2026-08-11

**Context.** RAGAS is not one call per question. Faithfulness costs 1–2, answer
relevancy 1, context recall 1, and **context precision one call per retrieved
chunk** — 5 more at top-5. With answer generation that is ~9–11 calls per
question, so a 100-question set is **~1,000 calls per evaluation**. Phase 2 needs
one evaluation per technique. See [P-4](#p-4--free-tier-request-budgets-cannot-fund-ragas).

**Decision.** Split evaluation into two tiers.

| Tier | Metrics | Cost | Cadence |
|---|---|---|---|
| Retrieval (deterministic) | hit@5, context recall, MRR, latency | **free** | after every change |
| Generation (LLM-judged) | faithfulness, answer relevancy | ~10 calls/question | phase boundaries |

**Rationale.** Retrieval metrics need no LLM *if the golden set stores the
ground-truth page per question* — they become page-number comparisons in pure
Python. Since Phase 2 is entirely about retrieval, nearly all of its
per-technique re-evaluations become free.

This is not a budget compromise. It is how a cost-aware evals team would build it
regardless, because **a metric that runs in five seconds gets run, and one that
costs a day doesn't.**

**Binding constraint on `build_eval_set.py`:** every golden question MUST carry
its ground-truth page number, or this entire tier collapses.

---

### D-11 · One OpenAI-compatible client for three providers
**Date:** 2026-08-11

**Decision.** A single `OpenAICompatGenerator` with per-provider `base_url`,
`api_key_env` and `model` in config.

**Rationale.** NVIDIA NIM, OpenRouter and HF Inference Providers all speak the
OpenAI chat-completions protocol. "Provider" is therefore configuration, not a
code path — `--provider openrouter` switches mid-evaluation. Phase 5 adds a
fourth entry pointing at the fine-tuned Qwen Space without touching the
interface, which is exactly what makes the Phase 4 generator comparison valid:
identical retrieval feeding every generator means differences are attributable
to the generator alone.

**Role assignment.**

| Provider | Role | Status |
|---|---|---|
| HF Pro | Current default; $2/month | ✅ working |
| OpenRouter | Development, hand-testing; $5 credits | ✅ working |
| NVIDIA NIM | Intended eval workhorse (~40 rpm, no daily cap) | ❌ see [P-5](#p-5--nvidia-nim-requests-time-out) |

---

### D-12 · Citations: inline `[p.N]` plus structured metadata
**Date:** 2026-08-11

**Options.** Inline markers · structured JSON output · a "Sources:" footer.

**Decision.** Inline `[p.N]` in the answer text, *with* the retrieved chunks also
returned as structured data.

**Rationale.** The frontend builds citation chips from the structured metadata,
never by parsing model prose. But the inline markers are what make per-claim
grounding checkable, and they are what Phase 4.5's citation-validity reward
scores. JSON-only output was rejected because JSON-mode reliability varies across
free models and it fights token streaming, which Phase 5's SSE UI needs.

**Consequence.** `rag_chain.py` verifies every emitted `[p.N]` against the pages
actually retrieved, on every call, for free.

---

### D-13 · Temperature 0
**Date:** 2026-08-11

**Decision.** All generation defaults to `temperature: 0.0`.

**Rationale.** A judged metric that moves because of sampling noise cannot be
attributed to a pipeline change, which would make the entire Phase 2 delta table
meaningless.

---

## Problems

### P-1 · Filenames didn't match the metadata convention
**Status:** ✅ Resolved · **Cost:** ~5 minutes

**Symptom.** All four PDFs would have ingested with `insurer=unknown`.

**Cause.** `ingest.py` parses `insurer__type__label.pdf` on **double**
underscores; the downloaded files used single ones.

**Why it mattered.** Silently disables Phase 2's metadata filtering and Phase 3's
policy-comparison agent — both key off those fields, and neither would have
failed loudly.

**Resolution.** Renamed before first ingest. Files that don't match still ingest,
but now emit a warning rather than passing quietly.

---

### P-2 · The Star Health document was a brochure, not a policy wording
**Status:** ✅ Resolved · **Cost:** ~30 minutes + one re-ingest

**Symptom.** 17,078 characters from a 3.86 MB file — one tenth the text of a
comparable policy from a file one fifth the size. 1,423 chars/page.

**Diagnosis.** Density was the tell: 3.86 MB across 12 pages means the bytes are
images, not text. Reading a sample confirmed it — the document itself said
*"kindly refer the Prospectus document ... available in our website download
section."* It was announcing that it wasn't the authoritative source.

**Why it mattered.** Questions would have been evaluated against a document that
doesn't contain their answers. Context recall would have looked like a retrieval
problem when it was a corpus problem, and Phase 2 would have been spent
optimising against a hole.

**Resolution.** Replaced with the real policy wording. **12 pages → 47**, density
1,423 → 2,602 chars/page.

**Lesson.** Check corpus *quality* before measuring pipeline quality. A metric
computed over a broken corpus is worse than no metric, because it looks credible.

---

### P-3 · Mis-decoded f-ligatures across an entire publisher
**Status:** ✅ Resolved · **Cost:** ~1 hour + full pipeline re-run

**Symptom.** A smoke-query result read `beneĤt`, `speciĤed`.

**Diagnosis.** Grepping the Latin Extended-A range across the corpus revealed the
scale and the mapping:

| Corrupt | Real | Observed as |
|---|---|---|
| `ģ` U+0123 | `ff` | staģ, oģer, suģering |
| `Ĥ` U+0124 | `fi` | beneĤt, speciĤed, qualiĤed, beneĤciary |
| `ĥ` U+0125 | `fl` | **ĥoater**, inĥicted, Reĥux |
| `Ħ` U+0126 | `ffi` | OĦce |

**Scope.** 47 of 102 pages — **100% of both SBI documents** (30/30 and 17/17),
**0%** of Star Health and ICICI. A per-publisher font-subset fault.

**Why NFKC didn't catch it.** `Ĥ` is a legitimate Unicode character. The text is
not malformed, merely *wrong*. No normaliser can know that.

**Why it mattered — the part that made this urgent.** The corrupted words are core
insurance vocabulary: *benefit, beneficiary, specified, qualified, **floater**,
certified*. Each became an out-of-vocabulary token, so a chunk about floater
benefits stopped matching a query about floater benefits. And the damage was
**entirely one-sided** — every SBI retrieval score was depressed relative to Star
Health. Any per-insurer comparison would have been measuring font encoding rather
than retrieval quality, and would have been confidently wrong.

**Resolution.** A character-repair map applied before de-hyphenation, plus a
warning when any Latin Extended-A character *survives* repair, so an unknown
publisher mapping surfaces immediately.

**Verification.** Latin Extended-A occurrences **47 → 0**. 66 pages now contain
correctly spelled `benefit`/`specified`/`floater`/`beneficiary`. Character deltas
prove the repair was surgical:

| Document | Before | After | Δ |
|---|---:|---:|---:|
| `sbigeneral__health__alpha` | 174,999 | 175,687 | +688 |
| `sbigeneral__home__house-insurance` | 100,706 | 100,903 | +197 |
| `iciciprulife__life__prusmart` | 93,109 | 93,109 | **0** |

Retrieval changed as a direct result: on the same smoke query the third hit moved
from p.19 @ 0.7517 (reading `beneĤt`) to a *better* chunk, p.21 @ 0.7576.

**Lesson.** The self-verifying smoke query in `embed_index.py` paid for itself on
its first run. A pipeline reporting only counts and timings would have shown four
green stages and a silently poisoned index.

---

### P-4 · Free-tier request budgets cannot fund RAGAS
**Status:** ✅ Resolved architecturally · **Cost:** ~1 hour of planning

**Symptom.** Planning revealed one RAGAS run ≈ 1,000 LLM calls; OpenRouter's free
tier allows 50/day.

**The numbers.**

| Source | Allowance | Full runs supported |
|---|---|---|
| OpenRouter free | 50/day (1,000/day above $10 credits) | 1 per **3 weeks** |
| HF Pro | **$2/month** | ~2 per **month** |
| NVIDIA NIM | ~40/min, no daily cap | 1 per **25–50 min** |

**Resolution.** [D-10](#d-10--two-tier-evaluation-deterministic-retrieval-judged-generation)
— move retrieval metrics off the LLM entirely. The remaining judged metrics fit
in the available budget.

---

### P-5 · NVIDIA NIM requests time out
**Status:** ❌ Unresolved · **Impact:** lost the best eval-throughput option

**Symptom.** Every NIM call times out after 60 s and exhausts all retries. Not an
auth failure — a bad key returns HTTP 401 in under a second.

**Investigation so far.** OpenRouter and HF both work from the same machine,
same code, same imports — so it is not the client, the network stack, or TLS in
general. Narrowed to something NIM-specific: key, region, or model availability.

**Workaround.** HF Pro is the current default; OpenRouter handles hand-testing.

**Why it still matters.** NIM's ~40 rpm with no daily cap is worth more to this
project than the $2/month HF budget. Worth one more attempt before the first full
RAGAS run — verifying the model id against their live catalogue is the obvious
next step.

---

### P-6 · Import hang, misdiagnosed as a OneDrive fault
**Status:** ✅ Resolved · **Cost:** ~20 minutes, one wrong hypothesis

**Symptom.** `import openai` hung mid-file-read inside
`OneDrive\...\.venv\Lib\site-packages\`.

**First hypothesis (wrong as a general explanation).** The project lives inside
OneDrive; Files On-Demand turns synced files into placeholders that must be
fetched on access, and a virtualenv is tens of thousands of tiny files. I
extended this to explain the NIM timeouts too, via a stalled `certifi` read.

**Why it was wrong.** The user pushed back, and the evidence supported them:
OpenRouter subsequently completed the *same* imports from the *same* path and
made a clean HTTPS call. OneDrive stalls are real but **intermittent**; they
don't explain a reproducible, provider-specific timeout.

**Corrected understanding.** The import hang was a one-off rehydration stall.
[P-5](#p-5--nvidia-nim-requests-time-out) is a genuinely separate problem.

**Standing recommendation, unchanged.** Move the project out of OneDrive
eventually — `qdrant_storage/` is a lock-based embedded database and syncing it
mid-write risks quiet corruption. But it is not a blocker.

**Lesson.** A hypothesis that neatly explains two unrelated symptoms at once
deserves *more* suspicion, not less.

---

### P-7 · "$2/day" was actually $2/month
**Status:** ✅ Resolved · **Cost:** would have been a badly wrong eval plan

**Symptom.** Planning proceeded on an assumed HF budget 30× larger than reality.

**Resolution.** Verified against the billing page: *"Remaining compute credits,
including $2.00 from your PRO subscription this billing period"*, period ending
Sep 1.

**Practical consequences.**
- The $2 is a **monthly grant that does not roll over** — unspent credit on the
  last day of the period is wasted, so there is no reason to hoard it.
- Automatic Recharge is **off**, so exhausting it fails calls rather than
  charging a card. No surprise bill.
- **ZeroGPU's 40 min/day is a separate pool** and is the far more valuable half
  of the subscription — it is the Phase 5 serving budget.

---

### P-8 · A reasoning model billed 168 tokens to say "OK"
**Status:** 🔶 Noted, avoided by model choice

**Symptom.** The OpenRouter connectivity check returned `15 in / 168 out` for a
one-word answer.

**Cause.** The configured Qwen model is a reasoning model; those are thinking
tokens, billed but absent from `content`.

**Why it matters.** RAGAS is ~1,000 calls per run. Hundreds of invisible
reasoning tokens per call means output cost dominates — on a judge deciding "is
this statement supported by this context?", a task wanting a decisive, cheap,
non-thinking model.

**Resolution.** Judge and bulk-generation roles should use non-thinking models.
`--model` already overrides per-invocation, so each script can use the tier its
job deserves without code changes.

---

### P-9 · Column scrambling visible in extracted text
**Status:** 🔶 Open — deliberately deferred to Phase 2

**Symptom.** In `text` extraction mode, headings arrive *after* the paragraphs
they label:

```
...co-payment of 10% ... for Insured Persons whose age at entry is 61 years...
Day Care Treatment          <- heading, AFTER its own paragraph
Co-Payment                  <- same
```

**Cause.** Two-column layout. Plain reading-order extraction reads the label
column and the content column as separate streams.

**Why not fixed.** This is [D-1](#d-1--text-extraction-mode-plain-text-made-switchable)
working as intended — `blocks`/markdown extraction is a Phase 2 experiment, and
this is its reproducible test case. Fixing it now would forfeit the measurement.

---

### P-10 · False refusal from an over-strict grounding contract
**Status:** ✅ Resolved and verified · **Cost:** ~45 minutes

**Symptom.** *"What is the co-payment for someone who joins at age 65?"* returned
the refusal sentence — despite the answer being in the corpus.

**Diagnosis.** The clause exists on page 39 of the Star Health wording:
*"co-payment of 10% ... for Insured Persons whose age at the time of entry is 61
years and above."* Retrieval returned page 39 at **rank 3** — so the chunk was in
the model's context.

The decisive test was rephrasing to echo the policy's own wording:

| Question | p.39 retrieved | Outcome |
|---|---|---|
| "...at **age 65**?" | rank 3 | **refused** |
| "...at **61 years or above**?" | rank 2 | answered, cited `[p.39]` |

Identical context, different phrasing, opposite outcome. Retrieval exonerated.

**Root cause.** My own system prompt. Rule 4 read *"Never calculate, estimate,
convert or total anything"* and rule 3 said to refuse *"if the passages do not
contain the answer"*. Together the model concluded that "65" appearing nowhere
meant the answer was absent, and that deciding 65 ≥ 61 was forbidden
computation. Instruction-following, not capability — Llama 3.3 70B can compare
two integers.

**Resolution.** Rule 4 existed to stop co-pay *arithmetic* (₹2.4L × 10%), which
belongs to Phase 3's deterministic calculator. It was too broad. The contract now
separates the two: applying a stated **threshold** to the user's situation is
required; computing **figures** remains forbidden. A new rule states explicitly
that a general rule answers a question about a specific case under it, and
refusal is narrowed to "the passages do not address the question at all".

**Verification.** Identical retrieval `[1, 3, 39, 15, 20]`:

| | Before | After |
|---|---|---|
| Answer | refusal sentence | "10% of each and every claim amount [p.39]" |
| `refused` | True | **False** |
| Citation | none | `p.39`, valid |

Regression check held: the out-of-scope meteor question still refuses in 13
output tokens, so the relaxed contract did not cause over-answering.

**Lesson.** **A false refusal is a failure mode in its own right**, and a
sneakier one than hallucination — refusing *looks* like caution, so it passes
review. Phase 4's RAFT dataset must train both halves: refuse when the context is
silent, **and** answer when it is not.

---

### P-11 · Retrieval instability across question phrasings
**Status:** 🔶 Open — Phase 2 target, with a reproducible test case

**Symptom.** Changing only the age in a question changes which chunks return, and
drops the only clause that can answer it.

| Question | Retrieved pages | Clause p.39 present? |
|---|---|---|
| "...at age 65?" | 1, 3, **39**, 15, 20 | yes |
| "...at age 45?" | 1, 3, 20, 20, 1 | **no** |

**Cause.** Dense retrieval embeds the whole question, so a number carrying no
semantic weight for the policy text still shifts the query vector. Nothing
anchors the search to the literal term "co-payment", which appears verbatim in
the target chunk.

**Secondary finding.** The age-45 result covers only **three distinct pages
across five slots** — duplicate pages consuming context budget without adding
information.

**Why not fixed.** Both are exactly what Phase 2 exists to solve — hybrid BM25
would match "co-payment" lexically regardless of the age; query rewriting would
normalise the phrasings; diversity-aware selection would stop one page taking
multiple slots. Kept as a baseline weakness so the Phase 2 delta table can
*demonstrate* the improvement rather than assert it.

---

### P-12 · PDF hard line-wraps survive into answers
**Status:** 🔶 Open — cosmetic, low priority

**Symptom.** A generated answer read *"Insured Personswhose"*.

**Cause.** Cleaning preserves single newlines, so the PDF's physical line wraps
remain mid-sentence. Harmless for embeddings; visible in user-facing prose.

**Deferred to.** Phase 2's extraction work, alongside
[P-9](#p-9--column-scrambling-visible-in-extracted-text).

### P-13 · Eval set silently omitted an entire document
**Status:** ✅ Resolved · **Cost:** ~20 minutes + one regeneration (~$0.13)

**Symptom.** A clean, successful-looking run: 100 items, 85 positives, 15
negatives, mean vocab overlap 0.377, only 10 rejections. But
`by policy type : {'health': 59, 'home': 26, '': 15}` — **no `life` key at all.**

**Diagnosis.** 59 + 26 + 10 rejections = 95 attempts against 136 allocated
candidates ordered health → home → life. The loop stopped at 85 accepted items,
still inside the home block, never reaching a life chunk.

**Root cause.** Oversampling (1.6×) was applied per type but the stop condition
was global, so surplus from earlier types satisfied the target and starved the
last one. **The better the generator performs, the worse the bug** — with more
rejections it would have reached life and looked fine, which is precisely why
`--limit 5` didn't catch it.

**Why it mattered.** An eval set with no life questions cannot detect *any*
regression in that document, while presenting as a complete golden set. The
intended 50/25/25 split came out 69/31/0.

**Fix.** Per-type accepted-item targets via largest-remainder allocation;
`sample_chunks()` returns candidates grouped by type; generation runs per type
against its own target; a weighted type yielding zero questions is now a hard
failure with an explicit message.

**Lesson.** It was not a crash — it was a summary that looked like success. The
only line that exposed it reported a **distribution** rather than a total. Run
summaries must show the *shape* of what was produced, not just that the run
finished. Same lesson as the self-verifying smoke query in
[P-3](#p-3--mis-decoded-f-ligatures-across-an-entire-publisher).

Full write-up in `explaination.md` §7.6.

### P-14 · Retrieval score is not a usable confidence signal
**Status:** 🔶 Open finding — constrains Phase 3 design

**Measurement.** Across the 15 hand-seeded negatives — questions the corpus
provably cannot answer (car collision, crop failure, marine cargo, satellite
launch) — **mean top-1 similarity is 0.6687**.

**Compare against genuine hits** measured on real questions: 0.8102, 0.7795,
0.7576, 0.6632, 0.6575, 0.6507, 0.6506, 0.6461.

**The unanswerable questions score inside the same band as the answerable ones.**
Dense retrieval always returns its nearest neighbours; "nearest" carries no
information about whether anything relevant exists at all.

**Consequence for Phase 3.** CLAUDE.md's task list includes a "confidence gate +
human-escalation node". The obvious implementation — escalate when the top score
falls below a threshold — **cannot work on this corpus**. There is no threshold
separating the two populations.

Refusal currently works because the *model* reads its context and declines, not
because retrieval signals uncertainty. So the confidence gate must be built from
something else. Candidates worth testing when Phase 3 arrives:

- **Score margin** (top-1 minus top-5) rather than absolute score — a diffuse
  match may separate better than a weak one.
- **Cross-encoder relevance** from the Phase 2 reranker, which scores the
  query–chunk *pair* and is a genuinely different signal.
- **The model's own refusal** as the gate input, which is what already works.

**Why this was nearly free.** Fifteen deterministic queries, no LLM calls. This
is the argument for [D-10](#d-10--two-tier-evaluation-deterministic-retrieval-judged-generation)
in miniature: a measurement cheap enough to run casually surfaced a design
constraint that would otherwise have been discovered by building the wrong thing.

### P-15 · Retrieval metric matched page numbers across documents
**Status:** ✅ Resolved · **Cost:** ~15 minutes + one free re-run

**Symptom.** The first real metrics run produced plausible numbers, but with an
odd shape: `health` scored **worst** (page hit@5 0.674) despite having the most
items and two documents' worth of coverage, while `home` — a single 17-page
document — scored best at 0.762.

**Root cause.** `evaluate_item()` compared **page numbers alone**, ignoring which
document they came from:

```python
ground_truth = set(item.get("ground_truth_pages") or [])
page_rank = first_rank(pages, ground_truth)     # doc_id never consulted
```

`ground_truth_doc_id` was stored on every golden item and never read. Our two
health policies span 47 and 30 pages, so their page ranges almost entirely
overlap — a chunk from page 30 of the SBI policy scored as a hit for a question
whose answer is on page 30 of the Star Health policy.

**Why the numbers still looked reasonable.** The bug *inflates* hit rates, so
nothing looked broken. It was only visible as a distributional oddity: health
should not underperform a single small document. The suspicious result was the
clue, not an error message.

**Fix.** Ground truth and retrieved results are now compared as `doc_id:page`
references. The old page-only calculation is retained as a clearly-labelled
`page only*` row purely to quantify how much it over-reports, so the size of the
error is visible rather than merely corrected.

**Lesson — third instance of the same theme.** After
[P-3](#p-3--mis-decoded-f-ligatures-across-an-entire-publisher) (a silently
corrupted index) and [P-13](#p-13--eval-set-silently-omitted-an-entire-document)
(a silently incomplete eval set), this is a silently wrong *metric*. None of the
three raised an error. Each was caught by looking at the **shape** of an output
rather than its status. The recurring defence is the same: report distributions
and breakdowns, not just totals — and treat a result that is surprising in the
wrong direction as a bug report about the measurement.

### P-16 · "False refusal rate" conflated a defect with correct behaviour
**Status:** ✅ Resolved · **Cost:** ~30 minutes of cross-referencing

**Symptom.** The first generation run reported `false refusal rate: 0.094`,
which looked encouraging. It was meaningless — the metric counted every refused
positive without asking whether the model had anything to refuse *from*.

**Verification.** Joining each refused item to its `page_rank` from the depth
retrieval run:

| Item | page_rank | Verdict |
|---|---:|---|
| **g-008** | **2** | genuine false refusal |
| g-037 / g-080 / g-045 | 6 / 7 / 8 | correct — no evidence in top-5 |
| g-049 / g-054 / g-002 / g-077 | 13 / 13 / 21 / 42 | correct |

**Only 1 of 8 was a defect.** Against the 59 items that actually had evidence in
context, the true false-refusal rate is **1.7%**, not 9.4%. The model's refusal
behaviour is close to correct; the metric was not.

**The inverse problem, which is the real one.** 26 positives lacked the correct
page; only 7 were refused. So **19 questions were answered with no ground-truth
evidence in context**. Two sampled:

- **g-030** — "What is the customer care phone number?" Expected `18001021111`;
  the model answered `1800 102 1111 [p.28, p.10]`. **Correct**, but from pages
  the golden set didn't record. Ground truth is incomplete, not the answer
  wrong. It is also a *bad eval question* — a contact number repeated across a
  document tests nothing about policy comprehension.
- **g-043** — "Will my insurance pay if I get hurt while doing something
  illegal?" Tagged `health`, answered from the **ICICI life policy** ("the Life
  Assured"), with a fabricated `[p.4]` citation. A genuine failure, and direct
  evidence for **cross-document contamination**.

**Fix.** `AnsweredItem` now records `had_evidence` (ground-truth `doc_id:page`
present in the retrieved context), and the metric splits three ways:
`false_refusal_rate` (refused **with** evidence — the defect),
`correct_refusal_rate` (refused **without** — correct), and
`ungrounded_answer_rate` (answered **without** — needs inspection, since it
mixes hallucination with incomplete ground truth).

**Consequences carried forward.**
- **hit@5 = 0.694 probably understates retrieval**, because single-page ground
  truth under-counts content that policy documents repeat across pages.
- **Metadata filtering is not optional.** g-043 shows a health question answered
  from a life policy. Already planned for Phase 2; now evidenced.
- **The golden set contains weak questions.** The sampled verification pass
  should drop items like g-030 rather than merely confirming them.

**Lesson.** A rate is only interpretable if its denominator is the population
the behaviour could apply to. "Refused" is neither good nor bad until you know
whether there was anything to answer from — the same behaviour is a defect in
one condition and correct in the other.

---

# Phase 2 — Advanced RAG

## Decisions

### D-15 · Reranking first, chosen from measurement
**Date:** 2026-08-12

**Context.** CLAUDE.md lists Phase 2's techniques in an order. Which to do first
should come from data.

**Evidence.** Phase 1's depth run: hit@5 **0.694** against recall@50 **0.953**.
For 95% of questions the correct page was already retrieved and merely ranked
too low; only 4 of 85 were unreachable. Reranking's addressable gap was 26
points; hybrid search's was 5.

**Decision.** Cross-encoder reranking (`BAAI/bge-reranker-base`) first.

**Outcome.** hit@5 0.694 → **0.812**. Confirmed.

---

### D-16 · Candidate depth 20, not 50
**Date:** 2026-08-13

**Context.** Depth was left as a swept parameter because evaluating it is free.

**Result — deeper candidates made things worse:**

| Depth | hit@5 | Ceiling | % of ceiling | ms/pair |
|---|---|---|---|---|
| 10 | 0.776 | 0.800 | **97%** | 190 |
| **20** | **0.812** | 0.894 | 91% | 200* |
| 30 | 0.800 | 0.920 | 87% | 206 |
| 50 | 0.788 | 0.953 | 83% | 200 |

\* the depth-20 run recorded 481 ms/pair. I attributed this to transient machine
contention and predicted a clean re-run would show ~200 ms/pair. **Wrong** — the
re-run reproduced it at 510 ms/pair. The real explanation is **thermal
throttling under sustained load**: a single query reranks 20 pairs in 3.6 s
(180 ms/pair) while 100 back-to-back queries average 10.2 s (510 ms/pair). Both
numbers are real and measure different things — the first is user-facing
latency, the second applies only to eval sweeps.

**Why deeper is worse.** Every extra candidate is another chance for the
cross-encoder to promote a wrong chunk above the right one. Its precision
degrades faster than the ceiling rises — efficiency falls monotonically from
97% to 83%. The opposite of "retrieve more, rerank harder" intuition.

**Decision.** Depth 20.

**Why quality and not CPU latency drove this.** CLAUDE.md serves the reranker
via `@spaces.GPU` in Phase 5, where 20 pairs costs tens of milliseconds. Tuning
depth around a laptop CPU would optimise for an environment the product never
runs in.

---

### D-17 · Hybrid search REJECTED — a measured negative result
**Date:** 2026-08-14

**Context.** Reranking hit a structural ceiling: a cross-encoder reorders the
candidate pool but cannot add to it, so at depth 20 the nine remaining misses
were invisible to it. Hybrid retrieval (BM25 + dense, fused by Reciprocal Rank
Fusion) was the obvious way to raise pool recall. It does not work here.

**Result — every configuration lost ground against dense alone:**

| Retrieval | Lexical width | Pool depth | Pool recall | Misses | Recovered | Evicted | Net |
|---|---|---|---|---|---|---|---|
| **dense** | — | 20 | **0.894** | 9 | — | — | — |
| hybrid RRF | 30 | 20 | 0.859 | 12 | 3 | 6 | **−3** |
| hybrid RRF | 10 | 20 | 0.871 | 11 | 3 | 5 | **−2** |
| hybrid RRF | 5 | 20 | 0.871 | 11 | 1 | 3 | **−2** |
| **dense** | — | 30 | **0.906** | 8 | — | — | — |
| hybrid RRF | 30 | 30 | 0.894 | 9 | 3 | 4 | **−1** |

End-to-end with reranking at depth 20, hybrid was an exact wash on hit@5
(0.8118 → 0.8118), slightly better on MRR (0.6158 → 0.6177), worse on hit@10
(0.882 → 0.824) and worse on exact-chunk hit@5 (0.765 → 0.718 — BM25 often
matches a *neighbouring* chunk on the correct page). p95 latency doubled,
11,496 → 23,415 ms.

**Root cause — fusion is a displacement trade, not an addition.** The pool is
fixed-size, so every lexical candidate admitted evicts a dense one. BM25's
recoveries are real and repeatable (g-002, g-012, g-041 in nearly every config;
g-076 at depth 30) but its evictions are equally consistent (g-059, g-067,
g-074, g-078). If fusion could keep both, pool recall at depth 30 would be
~0.94.

It cannot, because **the reranker's precision — not pool recall — is the
binding constraint** (D-16: 97% efficiency at depth 10 falling to 83% at depth
50). Fusion buys recall only by widening the pool, and widening is exactly what
degrades the cross-encoder. The two levers available (narrow BM25, widen the
pool) both converge toward dense-only, and dense-only wins at every depth.

**Why weighted RRF was not attempted.** As the lexical weight approaches zero,
weighted RRF *becomes* dense-only. The measured results are monotone in
eviction pressure (−3 → −2 → −1), so the optimum of that weight sweep sits at
the boundary. It would cost a code change and three more runs to rediscover the
baseline.

**Decision.** `hybrid.enabled` stays `false`. The module is kept, not deleted —
it is a working, documented implementation and the evidence behind this entry.

**What survives.** Two findings, both routed elsewhere rather than discarded:

1. The per-type split is consistent across all four runs — life gains, health
   and home lose (see P-17). That is a *routing* problem, not a fusion-weight
   problem; it belongs to Phase 3's router or technique 5's metadata filtering.
2. g-076 is reachable by BM25 though not densely, which partially resolves the
   false-confidence case left open in P-14.

**Cost.** ~3 hours including the sweep. A negative result that closes off a
plausible direction with evidence is worth recording, not hiding (CLAUDE.md §6).

---

### D-18 · Parent-document retrieval — rejected globally, kept for routing
**Date:** 2026-08-15

**Setup.** 337 parents (2,000 chars) over 1,644 children (400 chars) from the
same 102 pages, children indexed into their own collection so the Phase 1
baseline stayed reproducible.

**The composition bug, and why it is the interesting part.** The first version
expanded children to parents *before* reranking, reasoning that the
cross-encoder should read the same text the generator will. Two measurements
killed that argument:

1. **bge-reranker-base is `model_max_length=512`** and truncates silently past
   it. Median parent is 1,780 chars ≈ 445 tokens; with the query prepended, the
   upper half of parents were cut off. The cross-encoder scored their openings
   and never read their endings — where qualifying clauses tend to sit.
2. **Expansion deduplicates**, so 20 children collapsed to a median of 15
   parents. The reranker got a smaller pool than the run it was compared
   against, confounding the comparison on top of degrading it.

Reranking children first and expanding the winners afterwards recovered 2.3
points, **−4.7 → −2.4**. Ranking precision and context completeness want
different stages, not the same one. `rerank.max_length` is now set explicitly so
the limit is a visible constraint that shapes chunk sizing rather than a silent
one.

**Result vs dense + rerank@20:**

| Metric | Baseline | Parent-docs | Δ |
|---|---|---|---|
| hit@5 (doc+page) | **0.8118** | 0.7882 | −2.4 |
| MRR | 0.6158 | 0.5826 | −3.3 |
| Pool recall @20 | 0.894 | **0.9059** | **+1.2** |
| Complete misses | 9 | **8** | −1 |
| p50 latency | 10,217 ms | **1,476 ms** | **7× faster** |

**Retrieval improved; ranking did not.** Pool recall rose and the technique
reached `g-017` and `g-058` — two of the five items D-17 recorded as unreachable
by dense retrieval *or* any of four hybrid configurations. 400-char children
reach content nothing else in Phase 2 could. The aggregate still fell because
the reranker converts that pool into a top-5 slightly less efficiently (87% vs
90.8%): more fragmentary children give a cross-encoder less to judge.

**Decision.** Not adopted globally. `parent_docs.enabled` stays `false`. Kept
and indexed, because the per-type result below makes it a routing component
rather than a dead end.

**The per-type result, which is the real finding:**

| Policy type | Items | Baseline | Parent-docs | Δ |
|---|---:|---|---|---|
| health | 43 | 0.8605 | 0.8140 | −4.7 |
| home | 21 | **0.9048** | 0.7619 | **−14.3** |
| life | 21 | 0.6190 | **0.7619** | **+14.3** |

An exact mirror, and the third independent measurement of the same shape —
hybrid gave life +9.5 (D-17), parent-docs gives it +14.3, and P-17 predicted
both. **Optimal chunk size is a property of the document, not of the corpus.**
A single global chunk size leaves ~14 points on the table for the densest
document while being correct for the others.

**Carried to technique 5 as a hypothesis, not a result.** Routing life to
parent-docs and everything else to Phase 1 chunks projects to
`(43×0.8605 + 21×0.9048 + 21×0.7619) / 85 = 0.847` — +3.5 over the current best
and +15.3 over the Phase 1 baseline, which would clear the exit criterion. That
number is **arithmetic on measured per-type values, not a measured run**, and
selecting the per-type strategy from the same 85-item eval set is selection on
test data. With 21 life items, part of the +14.3 may be noise. It must be
measured before it is claimed.

---

### D-19 · ADOPTED — density-selected per-document chunking
**Date:** 2026-08-16

**Hypothesis under test.** D-18 projected 0.847 from arithmetic over per-type
numbers. Measured: **0.871**. The projection was conservative.

| Metric | Phase 1 baseline | Best (rerank@20) | **Chunk policy** | Δ vs best |
|---|---|---|---|---|
| hit@5 (doc+page) | 0.694 | 0.8118 | **0.8706** | **+5.9** |
| hit@1 | — | 0.4824 | **0.5294** | +4.7 |
| MRR | — | 0.6158 | **0.6501** | +3.4 |
| p50 latency | ~120 ms | 10,217 ms | **3,427 ms** | **3× faster** |

**Against the Phase 1 baseline: 0.694 → 0.871 = +17.7 points.** Phase 2's exit
criterion (≥15-point context-recall improvement) is met.

**Per type — the whole point of the technique:**

| Policy type | Baseline | rerank@20 | Chunk policy | Δ vs rerank |
|---|---|---|---|---|
| health | 0.674 | 0.8605 | 0.8605 | 0.0 |
| home | 0.762 | 0.9048 | 0.9048 | 0.0 |
| life | 0.667 | 0.6190 | **0.8571** | **+23.8** |

Health and home land on their previous values to four decimal places — mixing
granularities in one collection costs the flat-chunked documents nothing, which
was the main risk. All the gain is life, and it exceeds the +14.3 that
parent-documents alone produced. The extra ~9.5 points is most likely the
`min_chunk_chars` filter this module applies and `parent_docs.py` omitted: that
build produced 6-character children, which embed to noise and still occupy a
retrieval slot.

**Decision.** Adopted. `chunk_policy.enabled: true`.

**What did NOT improve, and it matters.** Complete misses are 9 — *the same nine
items as the rerank@20 baseline*, exactly. Pool recall is unchanged at 0.894.
This technique moved nothing into the pool; it ranked the existing pool far
better. Note the cost: the pure parent-documents run reached `g-017` and `g-058`
(two of D-17's five permanently-unreachable items) because *every* document had
small children. Restricting children to dense documents gives those two back.
Small chunks reach content in sparse documents too — they just cost more
elsewhere than they gain.

**Honest limits on this result — read before quoting it.**

1. **The strategy split was chosen using the same 85-item eval set.** That is
   selection on test data. The number is optimistic and a held-out document is
   the only real test.
2. **The density threshold is validated on n=1.** Exactly one document is above
   8,000 chars/page. The corpus gap is wide (5,935 → 11,639) so 8,000 is not
   delicate, but "dense documents want small chunks" currently rests on a single
   document, and one more dense policy from another insurer would test it
   properly.
3. `exact chunk` reads 0.612 rather than 0.000 here because health and home keep
   Phase 1 chunk ids that still match the golden set; only life's children do
   not. It is not comparable to Phase 1's 0.765 and is not a regression.

---

### D-20 · Query rewriting (technique 4) — deliberately not built
**Date:** 2026-08-16

**Decision.** Skipped. Recorded as a decision rather than an omission, because
a technique CLAUDE.md lists and this project does not ship needs a reason.

**Why.** Three arguments, in descending strength:

1. **The eval set is constructed to defeat it.** Query rewriting mainly fixes
   vocabulary mismatch between question and passage. `build_eval_set.py`
   deliberately paraphrases that overlap away and rejects questions whose
   content words overlap the source chunk beyond a threshold — precisely to stop
   hit@5 measuring string matching. The mechanism rewriting exploits is
   therefore largely *absent by construction*. A gain here would be hard to
   trust and a null result hard to interpret.
2. **It costs LLM calls per query, and the eval loop is already the
   bottleneck.** Every other Phase 2 technique is free and deterministic:
   ~30 seconds for 100 questions. Multi-query adds a generation call before
   every retrieval, which on the current free tier means ~53 s median per query
   (2026-08-15 measurement). It would move Phase 2's inner loop from seconds to
   over an hour.
3. **The exit criterion is already met** — +17.7 points against a ≥15 target
   (D-19).

**The honest counter-argument.** Pool recall has been 0.894 since Phase 1 and
*no technique has moved it*: the same nine items are missed by dense retrieval,
by four hybrid configurations, and by the adopted chunk policy. Query rewriting
is a **recall** technique and would be the first genuine attempt at those nine.
That is a real argument for building it.

It is outweighed by D-17's lesson: a recall technique feeding a
precision-bound reranker made things *worse*, not better, because the pool is
fixed-size and every admission is an eviction. Rewriting would face the same
structure.

**If revisited**, scope it narrowly at those nine items rather than as a general
improvement, and measure pool recall rather than hit@5 — hit@5 cannot show a
recall gain that the reranker then fails to convert.

---

### D-21 · Full-context stuffing loses to RAG on every axis
**Date:** 2026-08-16 · **Cost:** $0.25 of the $2 OpenRouter budget

**The claim tested.** "Long context windows make RAG obsolete — just put the
whole document in the prompt." Worth measuring rather than asserting: at ~124K
tokens against a 1M-token window, the ClaimWise corpus genuinely fits.

**Setup.** Same 10 golden questions, same generator
(`openrouter/deepseek-v4-flash-0731`), so the only difference is the retrieval
stage. Running both through one generator was deliberate — an earlier pairing
compared stuffing on a paid endpoint against RAG on a rate-limited free tier,
which measures the provider's queue rather than the architecture.

| Metric | Stuffing | RAG | Advantage |
|---|---|---|---|
| Cited correctly | 0.20 | **0.60** | **3×** |
| Prompt tokens/query | 120,972 | **1,646** | **73×** |
| p50 latency | 36.5 s | **14.2 s** | 2.6× |
| Spend, 10 questions | $0.0829 | **$0.0017** | **49×** |

**RAG wins on accuracy, tokens, latency and cost simultaneously.** There is no
axis on which stuffing trades favourably here.

**Stuffing fails plausibly, not loudly.** It never once refused. Every question
received a confident answer, and 8 of 10 cited the wrong page while the right
page sat in context. A silent wrong answer with a confident citation is the
worst failure mode an insurance assistant can have.

**The counter-argument that did not survive.** Stuffing cannot suffer a
retrieval miss — every page Phase 2 fails to retrieve is *visible* to it. That
should have been its advantage, and it did not materialise: having the page in
context is not the same as finding it there.

**The structural finding.** RAG gets citation validity for free — every `[p.N]`
checked against the retrieved set, deterministically, on every call. Stuffing
cannot have that metric at all: with every page in context nothing is ever
invalid relative to what was retrieved, so verification requires ground truth
that does not exist at inference time. **The simpler architecture gives up its
own error detection**, and it is the same signal Phase 4.5's GRPO reward
targets.

**Honest limits.** n=10, and stuffing scored 0.30 / 0.40 / 0.20 across three
identical `temperature=0` runs — ±10 points of noise, so the accuracy claim is
"clearly worse", not a precise figure. RAG was measured once. The 73× token
ratio is arithmetic and needs no sample size. RAG's 14.2 s also includes ~6 s of
CPU reranking that becomes ~0.1 s on GPU, so the latency gap is understated.

---

## Problems

### P-14 · RESOLVED — the cross-encoder *is* a usable confidence signal

The open finding recorded in Phase 1 listed cross-encoder relevance as a
candidate replacement for the unusable bi-encoder score. Now measured on the
same 15 negatives:

| Stage | Mean top-1 on negatives | Usable as confidence? |
|---|---|---|
| Bi-encoder cosine | **0.6687** | No — inside the positives' range |
| Cross-encoder | **0.0985** | **Yes** — 14 of 15 below 0.22, median 0.017 |

Genuine hits score 0.85–0.99. Low-scoring *positives* are largely the retrieval
failures — g-017 (0.030), g-058 (0.171), g-041 (0.174) — which is the desired
behaviour: a confidence gate needn't distinguish "no answer exists" from "I
couldn't find it", since both warrant the same action.

**Carried to Phase 3.** The confidence gate should threshold cross-encoder
top-1, around 0.2–0.25 on this evidence. One known false-confidence case
remains: g-076 scores 0.997 while being a complete retrieval miss.

### P-11 · RESOLVED — reranking fixes the phrasing instability

The Phase 1 weakness (identical question, different age, correct clause dropped
from the top-5) is fixed by reranking. Same question, one flag apart:

| | Retrieved pages | Answer |
|---|---|---|
| `--no-rerank` | 1, 3, 20, 20, 1 | *refused* |
| reranked, depth 20 | 3, **39**, 1, 12, 20 | "There is no co-payment specified for someone who joins at age 45 **[p.39]**" |

The clause returns at rank 2, and the answer is correct in the *inverse*
direction — the 10% co-pay applies at entry age 61+, so someone joining at 45
owes nothing. The model applied the threshold rather than echoing it.

The secondary observation from P-11 also resolved: the baseline returned only 3
distinct pages across 5 slots (`1, 3, 20, 20, 1`); reranked, all 5 are distinct.

### P-17 · Reranking regresses the life policy
**Status:** 🔶 Open

Per-type page hit@5 across the sweep:

| Policy type | Baseline | d10 | d20 | d30 | d50 |
|---|---|---|---|---|---|
| health | 0.674 | — | **0.861** | 0.837 | 0.814 |
| home | 0.762 | — | **0.905** | 0.905 | 0.905 |
| life | **0.667** | 0.714 | 0.619 | 0.619 | 0.619 |

Health gains +18.7 and home +14.3, but **life loses 4.8 points** at every depth
≥20. It is the outlier document in every respect: 11,639 chars/page against
~2,600–5,900, dense multi-column fine print, and only 8 pages.

Hypotheses, untested: its chunks span more topics each, so no chunk is strongly
"about" the question; or the cross-encoder's training distribution suits prose
better than dense clause lists.

**Update 2026-08-14 — the lexical hypothesis was right, and it is the only
place hybrid helped.** With BM25 fused in at depth 20 plus reranking:

| Policy type | dense + rerank | hybrid + rerank | Δ |
|---|---|---|---|
| health | 0.8605 | 0.8140 | −4.7 |
| home | 0.9048 | 0.9048 | 0.0 |
| life | 0.6190 | **0.7143** | **+9.5** |

The direction is consistent across all four hybrid configurations swept in
D-17, so this is a property of the document rather than run-to-run noise. It is
not enough to save hybrid globally — the aggregate is a wash and pool recall
falls — but it means the life policy wants a *different retrieval strategy*,
not a different global setting.

**Carried to Phase 3 / technique 5.** Per-policy-type retrieval routing, where
`policy_type=life` fuses lexically and everything else does not. Deliberately
not done inside Phase 2's hybrid experiment: it would mix a routing change into
a fusion measurement and break the one-change-one-eval attribution rule.

**Still open for technique 3.** The first hypothesis — chunks spanning too many
topics at 11,639 chars/page — is precisely what parent-document retrieval
addresses, and remains the more likely root cause of the 0.619 baseline.

---

### P-18 · RESOLVED — RAGAS was silently broken for the whole of Phase 1
**Date:** 2026-08-14 · **Cost:** ~1 hour

**Symptom.** Every RAGAS run since Phase 1 failed with
`ragas imports failed: No module named 'langchain_community.chat_models.vertexai'`.
The full Phase 1 generation eval was then run with `--skip-ragas`, so
faithfulness and answer relevancy were **never computed for any phase** — and
because the failure was caught and reported rather than raised, the gap was
invisible until METRICS.md was backfilled and the cells came up empty.

**Diagnosis.** `ragas/llms/base.py` line 12 is an unconditional top-level
`from langchain_community.chat_models.vertexai import ChatVertexAI`.
langchain-community 0.4 deleted that module — `ChatVertexAI` moved to the
separate `langchain-google-vertexai` package — and ragas 0.4.3 declares
`langchain-community` with **no upper version bound**, so the resolver was free
to pick a version that breaks it. An upstream defect, not a misconfiguration.

**Root cause.** A lockfile guarantees the same versions every run; it does not
guarantee those versions work together. `uv.lock` faithfully reproduced a broken
combination. The discipline that was supposed to make runs reproducible
reproduced the breakage perfectly.

**Fix.** A scoped shim in `run_ragas.py` registering a stub module before the
ragas import. Bounded by measurement rather than hope: ragas references
`langchain_community` in exactly two places, both in that one file, and the
other (`langchain_community.llms.vertexai`) still ships. The placeholder class
is *correct*, not merely convenient — ragas uses the symbol only in `isinstance`
checks deciding whether a judge supports multiple completions, and we never
judge with Vertex AI, so every such check should be False.

Downgrading instead would have pulled langchain-core below 1.0 and taken
`RecursiveCharacterTextSplitter` with it, re-chunking the corpus and
invalidating every retrieval number recorded so far.

**Two further defects found in our own code, both surfaced by the fix:**

1. `dict(result)` was never valid. `EvaluationResult` has `__getitem__` but no
   `keys()`, so `dict()` fell back to sequence iteration, requested `result[0]`,
   and hit a dict keyed by metric name — `KeyError: 0`. The public surface is
   `.scores`, a list of one dict per sample.
2. That aggregation sat **outside** the `try`, so it crashed *after* every judge
   call had been paid for and discarded the free metrics too — breaking the
   module's documented promise that a RAGAS failure never costs them. Anything
   touching the result object now sits behind the same guard as the call that
   produced it.

**Verification, and why it mattered immediately.** Per-metric NaN counters were
added so a mean over survivors could not be mistaken for a mean over the set.
They earned it on the first run: faithfulness reported 0.8420 over an apparent
78 items while 24 judge replies had actually returned NaN via
`OUTPUT_PARSING_FAILURE`. The clean re-run scored 0.8289 with 1 failure. Without
the counter, 0.8420 would have entered METRICS.md as the Phase 1 baseline.

**Carried forward.** The two runs differ by up to 5.1 points on citation
coverage with identical inputs at `temperature=0` — the hosted free model is not
deterministic. **Any generation-side gain below ~2 points is noise** and cannot
be attributed to a technique from a single run. Recorded in METRICS.md §1.5b.

---

Open items inherited from Phase 1:
[P-9](#p-9--column-scrambling-visible-in-extracted-text) ·
[P-11](#p-11--retrieval-instability-across-question-phrasings) ·
[P-12](#p-12--pdf-hard-line-wraps-survive-into-answers) ·
[D-6](#d-6--embedding-model-bge-small-with-an-ab-deferred)'s deferred A/B.

# Phase 3 — Agentic Layer
⏳ Not started.

# Phase 4 — Supervised Fine-tuning
⏳ Not started. [P-10](#p-10--false-refusal-from-an-over-strict-grounding-contract)
is a ready-made training case for the "answer when the context supports it" half
of the RAFT dataset.

# Phase 4.5 — Post-training
⏳ Not started, credit-gated.

# Phase 5 — Deployment
⏳ Not started. [P-5](#p-5--nvidia-nim-requests-time-out) must be settled or
permanently worked around before eval-heavy work resumes.

# Phase 6 — Frontend
⏳ Not started. Design handoff reviewed; note the design package describes a
*claim-filing* assistant while the backend is a *policy Q&A + calculation*
assistant, and it contains no citation-chip component despite Phase 6 requiring
one.
