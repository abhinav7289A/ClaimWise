# ClaimWise — Metrics

Single source of truth for every number in this project (CLAUDE.md §6).

Rules:
- Every entry records **date, git commit hash, config used, and the number**.
- No metric is written here unless it was computed from a real run's output.
- Deltas over absolutes: frame results as "X → Y after change Z".
- Negative results are recorded, not hidden.

**Commit note.** Phase 1 landed at `97545ea`. Everything from the rerank sweep
onward is uncommitted working-tree state, recorded below as `97545ea+wt`. These
rows should be re-stamped with the real hash when Phase 2 is committed.

---

## Phase 1 — Naive RAG Baseline

### 1.1 Corpus (from `python -m phase1_rag.ingest`)

Run C — after both corpus faults were fixed (brochure swapped for policy
wording; publisher-wide f-ligature corruption repaired). See `explaination.md`.

| Date | Commit | Config | Documents | Pages kept | Total chars | Mean chars/page |
|------|--------|--------|-----------|------------|-------------|-----------------|
| 2026-08-11 | `5785c07` | `extraction_mode=text` | 4 | 102 | 491,986 | 4,823 |

Per document — the spread matters, because the outlier drives P-17:

| Document | Type | Pages | Chars | Chars/page |
|---|---|---:|---:|---:|
| `starhealth__health__comprehensive` | health | 47 | 122,287 | 2,602 |
| `sbigeneral__health__alpha` | health | 30 | 175,687 | 5,856 |
| `sbigeneral__home__house-insurance` | home | 17 | 100,903 | 5,935 |
| `iciciprulife__life__prusmart` | life | 8 | 93,109 | **11,639** |

### 1.2 Chunking (from `python -m phase1_rag.chunk`)

| Date | Commit | Chunk size / overlap | Pages in | Chunks | Chars min/median/mean/max | Over 2,000-char budget |
|------|--------|----------------------|----------|--------|---------------------------|------------------------|
| 2026-08-12 | `f22650d` | 1000 / 150, recursive | 102 | 653 | 52 / 945 / 819 / 1000 | 0 |

### 1.3 Index build (from `python -m phase1_rag.embed_index`)

| Date | Commit | Model | Dims | Chunks indexed | Points | Embed time | Throughput | Upsert |
|------|--------|-------|------|----------------|--------|------------|------------|--------|
| 2026-08-12 | `f22650d` | `BAAI/bge-small-en-v1.5` (CPU) | 384 | 653 | 653 | 36.68 s | 17.8 chunks/s | 2.22 s |

### 1.4 Retrieval baseline (from `python -m evals.retrieval_metrics`)

Dense only, no reranking. **This is the number Phase 2 must beat.**

| Date | Commit | Config | hit@5 (page) | hit@5 (chunk) | MRR | Pool recall@20 | Pool recall@50 |
|------|--------|--------|--------------|---------------|-----|----------------|----------------|
| 2026-08-12 | `97545ea` | top-k 5, bge-small | **0.694** | — | — | 0.894 | 0.953 |

### 1.5 Generation baseline (from `python -m evals.run_ragas`)

Free deterministic tier only. Run tag `free2`, 85 positives + 15 negatives.

| Date | Commit | Generator | Citation validity | Citation coverage | False refusal | Refusal accuracy | p50 / p95 latency | Tokens/query | Est. cost |
|------|--------|-----------|-------------------|-------------------|---------------|------------------|-------------------|--------------|-----------|
| 2026-08-12 | `97545ea` | `meta-llama/Llama-3.3-70B-Instruct` (HF) | 0.960 | 0.974 | 0.0169 | 0.9333 | 1.074 s / 5.688 s | 1,475 | $0.1032 |

Fabricated citations: g-043, g-056, g-057. Missed refusal: g-097.

### 1.5b RAGAS judged metrics — resolved 2026-08-14

Blocked since Phase 1 by an upstream defect (P-18): ragas 0.4.3 hard-imports
`langchain_community.chat_models.vertexai`, which langchain-community 0.4
deleted. Unblocked with a scoped compatibility shim, recorded in every results
file as `shims_applied`.

| Date | Commit | Run tag | Judge | Scored | Faithfulness | _failed_ | Answer relevancy | _failed_ |
|------|--------|---------|-------|-------:|--------------|---------:|------------------|---------:|
| 2026-08-14 | `97545ea+wt` | `ragas-baseline` (14:21) | Llama-3.3-70B (HF) | 78 | 0.8420 | **24** | 0.8313 | 1 |
| 2026-08-14 | `97545ea+wt` | **`ragas-baseline` (16:43)** | Llama-3.3-70B (HF) | 78 | **0.8289** | 1 | **0.8384** | 0 |

> **⚠️ Generator changed 2026-08-15 — these rows are superseded, not
> comparable.** The HF Inference balance was exhausted, forcing a move to NIM.
> The intended swap was like-for-like (`meta-llama/Llama-3.3-70B-Instruct` →
> `meta/llama-3.3-70b-instruct`, same weights, different host), but that NIM
> endpoint refused connections. The first working alternative,
> `nvidia/llama-3.3-nemotron-super-49b-v1.5`, turned out to be a reasoning
> model — it spent **248 output tokens answering "OK"** — making it slow,
> rate-limit-prone and expensive. Settled on `google/gemma-4-31b-it`, which
> answers the same prompt in **2 tokens**.
>
> So the generator changed **family and size**, 70B Llama → 31B Gemma, not
> merely host. **Nothing below can be compared against a NIM run.** A fresh
> no-rerank baseline on NIM is required before reranking is measured, or the
> model swap and the pipeline change would be confounded in a single delta.
>
> The judge is also now pinned separately (`openai/gpt-oss-120b`) so it stays
> fixed while generators change — see the `judge:` block in `config.yaml`.

**Pipeline measured: dense retrieval, top-5, NO reranking.** `run_ragas.py`
calls `answer_question()` without a reranker, so it ignores `rerank.enabled:
true` in config. That makes these numbers a true *Phase 1* baseline, which is
what was wanted — but it also means **reranking's effect on generation has
never been measured**, and Phase 2's delta table has no generation-side rows.
`run_ragas.py` must accept a reranker before technique 3 is evaluated.

**The second run is the baseline of record.** The first computed faithfulness
over only 54 of 78 items — 24 judge replies hit
`OUTPUT_PARSING_FAILURE` and returned NaN. The `_failed` counters were added in
the same session precisely so a mean over survivors could not be mistaken for a
mean over the set; they caught a real 24-item hole on their first run.

**⚠️ Run-to-run variance is real and must be respected in attribution.** The two
runs used identical inputs and `temperature=0`, yet differ:

| Metric | Run 1 | Run 2 | Spread |
|---|---|---|---|
| Faithfulness | 0.8420 | 0.8289 | 1.3 pts |
| Answer relevancy | 0.8313 | 0.8384 | 0.7 pts |
| Citation validity | 0.947 | 0.958 | 1.1 pts |
| Citation coverage | 0.974 | 0.923 | **5.1 pts** |
| Fabricated citations | 4 (adds g-075) | 3 | 1 item |

The hosted free model is not deterministic despite `temperature=0`. **A
generation-side change worth fewer than ~2 points is indistinguishable from
noise on a single run** and must not be reported as an improvement.

### 1.6 Phase 1 exit criteria status

| Criterion | Status |
|---|---|
| Baseline retrieval numbers recorded | ✅ |
| Baseline generation numbers recorded | ✅ as of 2026-08-14 (§1.5b) |
| `metrics/failure_analysis_p1.md` — 10 worst failures analysed | ❌ not written |

### 1.7 The dominant generation defect

Of 85 positives, retrieval supplies evidence for 59 and fails on 26. On those 26
the model should refuse. It refuses 6 times and answers 20 times:

| Behaviour | Rate | Count |
|---|---|---|
| Refused without evidence (correct) | 0.231 | 6 / 26 |
| **Answered without evidence (hallucinated)** | **0.769** | **20 / 26** |
| Refused with evidence (false refusal) | 0.017 | 1 / 59 (g-008) |
| Refusal accuracy on negatives | 0.933 | 14 / 15 (misses g-097) |

**When retrieval fails, the generator invents an answer 77% of the time.** This
is a generator defect, not a retrieval one, and no Phase 2 technique addresses
it. It is the explicit target of Phase 3's confidence gate (P-14: threshold
cross-encoder top-1 at ~0.20–0.25) and Phase 4's RAFT negatives.

---

## Phase 2 — Advanced RAG

_One row per technique, so every improvement is attributable._

| Date | Commit | Technique | hit@5 (page) | Δ | MRR | Pool recall | p50 latency | Verdict |
|------|--------|-----------|--------------|---|-----|-------------|-------------|---------|
| 2026-08-12 | `97545ea+wt` | *(baseline: dense, top-5)* | 0.694 | — | — | 0.894 @20 | ~120 ms | — |
| 2026-08-13 | `97545ea+wt` | **1. Cross-encoder rerank** @20 | **0.812** | **+11.8** | 0.6158 | 0.894 @20 | 10,217 ms | ✅ **adopted** |
| 2026-08-14 | `97545ea+wt` | 2. Hybrid BM25 + RRF @20 | 0.812 | **+0.0** | 0.6177 | 0.859 @20 | 10,851 ms | ❌ **rejected (D-17)** |
| 2026-08-15 | `97545ea+wt` | 3. Parent-document retrieval | 0.788 | −2.4 | 0.5826 | **0.906** @20 | 1,476 ms | ⚠️ **rejected globally, kept for routing (D-18)** |
| 2026-08-16 | `97545ea+wt` | **5. Density-based chunk policy** | **0.871** | **+5.9** | **0.6501** | 0.894 @20 | 3,427 ms | ✅ **adopted (D-19)** |

### Phase 2 exit criterion

| | Value |
|---|---|
| Phase 1 baseline hit@5 (doc+page) | 0.694 |
| Best Phase 2 pipeline | **0.871** |
| **Improvement** | **+17.7 points** |
| Target | ≥15 points |
| **Status** | ✅ **met** |

Best pipeline: density-selected chunking (life → 400/2,000 parent-child, others
→ flat 1,000/150) + cross-encoder rerank@20 over children, parents expanded
after reranking.

**Read with D-19's three caveats**: the strategy split was chosen on this eval
set (selection on test data), the density threshold is validated on n=1, and
`exact chunk` is not comparable across chunking strategies.

### 2.1 Reranking — candidate depth sweep (D-16)

| Depth | hit@5 | Pool ceiling | % of ceiling |
|---|---|---|---|
| 10 | 0.776 | 0.800 | **97%** |
| **20** | **0.812** | 0.894 | 91% |
| 30 | 0.800 | 0.906 | 88% |
| 50 | 0.788 | 0.953 | 83% |

Efficiency falls monotonically with depth — the opposite of "retrieve more,
rerank harder". Depth 20 adopted.

> **Correction 2026-08-14.** D-16 originally recorded the depth-30 ceiling as
> 0.920. The `rr30` results file lists 8 complete misses out of 85 positives,
> which is **0.906**. Corrected here; the depth-20 decision is unaffected.

### 2.2 Hybrid search — the negative result (D-17)

Pool recall by configuration, all at matched depth. Recovered/evicted are
item-ID set differences against the dense miss list at the same depth.

| Retrieval | Lexical width | Pool depth | Pool recall | Misses | Recovered | Evicted | Net |
|---|---|---|---|---|---|---|---|
| **dense** | — | 20 | **0.894** | 9 | — | — | — |
| hybrid RRF | 30 | 20 | 0.859 | 12 | 3 | 6 | **−3** |
| hybrid RRF | 10 | 20 | 0.871 | 11 | 3 | 5 | **−2** |
| hybrid RRF | 5 | 20 | 0.871 | 11 | 1 | 3 | **−2** |
| **dense** | — | 30 | **0.906** | 8 | — | — | — |
| hybrid RRF | 30 | 30 | 0.894 | 9 | 3 | 4 | **−1** |

End-to-end with reranking at depth 20: hit@5 0.8118 → 0.8118, MRR 0.6158 →
0.6177, hit@10 0.882 → 0.824, exact-chunk hit@5 0.765 → 0.718, p95 latency
11,496 → 23,415 ms. `hybrid.enabled` stays `false`.

### 2.3 Per-policy-type hit@5 (page)

| Policy type | Items | Baseline | + rerank@20 | + hybrid@20 |
|---|---:|---|---|---|
| health | 43 | 0.674 | **0.8605** | 0.8140 |
| home | 21 | 0.762 | **0.9048** | 0.9048 |
| life | 21 | 0.667 | 0.6190 | **0.7143** |

The life policy is the outlier in every measurement (11,639 chars/page). See
P-17 — carried to technique 3 and to Phase 3 routing.

### 2.4 Confidence signal (P-14)

Mean top-1 score on the 15 hand-seeded negatives:

| Stage | Score | Usable as a confidence gate? |
|---|---|---|
| Bi-encoder cosine | 0.6687 | No — inside the positives' range |
| Cross-encoder | **0.0985** | **Yes** — 14/15 below 0.22, median 0.017 |

Threshold ~0.20–0.25 recommended for Phase 3's confidence gate.

### 2.5 Items still unreachable by any Phase 2 retrieval

`g-017`, `g-030`, `g-058`, `g-069`, `g-077` — missed by dense and by every
hybrid configuration. The target set for technique 3.

---

## Modal credit ledger (Phases 4 / 4.5)

Starting balance: **$30.00** (hard cap). $5.00 is reserved for Phase 5 and never spent on training.

| Date | Run | Estimated cost | Actual cost | Remaining |
|------|-----|----------------|-------------|-----------|
|      | —   | —              | —           | $30.00    |

### 2.5 Generation-side results — the adopted pipeline

Run `mx-rr20-gen`, 2026-08-16. Pipeline: dense top-5 + rerank@20 +
chunk-policy/parent-expansion, collection `claimwise_mx__baai_bge_small_en_v1_5`.
Generator **and** judge `openrouter/deepseek-v4-flash-0731`. Spend $0.1453.

**Attributable — deterministic, no judge involved:**

| Metric | Phase 1 baseline (HF) | Phase 2 (mx-rr20) | Δ |
|---|---|---|---|
| **Positives with evidence** | 59 / 85 | **74 / 85** | **+15 items** |
| **Hallucinations (absolute)** | 20 of 26 | **10 of 11** | **halved** |
| Citation validity | 0.958 | 0.962 | flat |
| Citation coverage | 0.923 | 0.963 | +4.0 |
| False refusals | 1 (g-008) | 3 (g-035, g-074, g-082) | worse |
| Refusal accuracy (negatives) | 0.933 | 0.933 | flat |
| Tokens per query | 1,475 | 2,075 | +41% (parent blocks) |
| p50 latency | 1.6 s | 11.3 s | reranking on CPU |

**+15 positives gained evidence** — hit@5 0.694 → 0.871 arriving generation-side,
and the check that the chunk-policy wiring reaches the served pipeline.

> **`ungrounded answers: 0.909` is a shrinking-denominator artifact, not a
> regression.** It is 10/11 against the baseline's 20/26. Retrieval now fails on
> 11 positives instead of 26, so the same defect over a smaller base reads
> higher while absolute hallucinations halved. **Compare counts, not rates.**
> The defect itself — the model answers rather than refusing when evidence is
> missing — is unchanged and belongs to Phase 3's confidence gate and Phase 4's
> RAFT negatives.

**NOT attributable — do not quote as a Phase 2 delta:**

| Metric | Baseline | This run |
|---|---|---|
| Faithfulness | 0.8289 | 0.7667 (**16 of 81 failed**) |
| Answer relevancy | 0.8384 | 0.7978 |

Three confounds stacked on an incomplete measurement: the generator changed
(Llama-3.3-70B → DeepSeek), the judge changed *and became self-judging*
(DeepSeek grading DeepSeek, so this is an upper bound), and faithfulness was
computed on 65 of 81 items. Per P-18, a mean over survivors is not the mean over
the set. A comparable pair would need one fixed generator and one fixed,
independent judge across both runs.

---

### 2.6 Blog experiment — full-context stuffing vs RAG (technique 6)

Same 10 golden questions, **same generator** (`openrouter/deepseek-v4-flash-0731`),
so the only difference is the retrieval stage. Corpus is 494,874 chars (~124K
tokens) against a 1M-token context window, so the whole thing genuinely fits.

| Metric | Full-context stuffing | RAG pipeline | Advantage |
|---|---|---|---|
| Cited correctly | 0.20 | **0.60** | **3× RAG** |
| Prompt tokens/query | 120,972 | **1,646** | **73× RAG** |
| p50 latency | 36.5 s | **14.2 s** | 2.6× RAG |
| Spend, 10 questions | $0.0829 | **$0.0017** | **49× RAG** |
| False refusal rate | 0.0 | — | — |

**Stuffing does not fail loudly — it fails plausibly.** It never refused. Every
question got a confident answer; 8 of 10 simply cited the wrong page with all
102 pages visible. Classic lost-in-the-middle.

**Stuffing accuracy is unstable.** Three runs at `temperature=0` on identical
inputs: **0.30, 0.40, 0.20**. Latency drifted too: 8.1 → 14.3 → 36.5 s p50.
At n=10 this supports "clearly worse than RAG" and nothing more precise. The
73× token ratio is arithmetic and is the finding that does not depend on sample
size.

**The structural finding, which is the most interesting part.** RAG gets
citation validity *for free* — every `[p.N]` is checked against the retrieved
set deterministically on every call (0.958 in §1.5). Stuffing cannot have that
metric at all: with every page in context, no citation is ever invalid relative
to what was retrieved, so nothing can be checked short of ground truth that does
not exist at inference time. The simpler-looking architecture quietly gives up
its own error detection — and it is the same signal Phase 4.5's GRPO reward was
designed to optimise.

**Caveat:** RAG's 14.2 s includes ~6 s of CPU cross-encoding, which moves to
~0.1 s on `@spaces.GPU` in Phase 5. The latency gap is understated here, not
overstated.

---

## OpenRouter budget ledger — the only paid inference path

Starting balance: **~$2.00**, and it is expected to cover the rest of the
project. Every OpenRouter run prints an estimate before spending. NIM and HF are
free tiers priced at 0.0 in config, so anything with a non-zero cost here went
through OpenRouter.

Model: `deepseek/deepseek-v4-flash-0731` — **$0.0675 in / $0.135 out per 1M**,
1M-token context. Chosen because it is the only configured model whose window
fits the ~124K-token full-corpus prompt, which NIM timed out on.

| Date | Run | Est. cost | Actual cost | Remaining |
|------|-----|-----------|-------------|-----------|
| 2026-08-16 | `stuff10-deepseek` | $0.0836 | $0.0827 | ~$1.917 |
| 2026-08-16 | `stuff10` re-run (crashed after payment) | $0.0836 | $0.0829 | ~$1.834 |
| 2026-08-16 | `stuff10-vs-rag` (stuffing + RAG column) | $0.0836 | $0.0846 | ~$1.750 |
| 2026-08-16 | `mx-rr20-gen` (generation eval, gen + judge) | ~$0.045 | **$0.1453** | **~$1.605** |

> The generation-eval estimate was **3× low**. I sized judge calls at ~2,500
> tokens; RAGAS sends the full retrieved context per claim being verified, so a
> faithfulness judgement costs far more than one question-and-answer pair. Use
> ~$0.15 per 100-question judged run for planning, not $0.045.

**~$0.25 spent, ~$1.75 remaining.** Estimates tracked actuals within ~1%, so the
per-provider pricing added on 2026-08-16 can be trusted for planning.

One run was wasted: the second crashed on a `NameError` *after* the paid calls
completed, discarding the results file. The `--rag-baseline` block is now
guarded so a failure in the free extra can never destroy paid results — the
third instance of that pattern in this project (see P-18).

**Reference points for planning:**

| Workload | Input tokens | Cost |
|---|---|---|
| Full-context stuffing, 10 questions | 1.24 M | **~$0.09** |
| Full-context stuffing, 100 questions | 12.4 M | ~$0.85 |
| A full `run_ragas` generation eval (~100 q) | ~0.15 M | ~$0.02 |

The stuffing experiment is ~40× more expensive per question than the entire RAG
pipeline. Ten questions is enough to establish the shape of the trade; the full
100 would consume nearly half the remaining budget to sharpen a number whose
conclusion is already visible at n=10.

## API spend on free tiers (no money, tracked for the cost comparison)

| Date | Run | Provider | Notional cost |
|------|-----|----------|---------------|
| 2026-08-12 | `free2` generation eval | HF Inference | $0.1032 |
| 2026-08-12 | `ragas-smoke` | HF Inference | $0.0105 |

> These used the old blended `estimated_usd_per_million_tokens: 0.7`, which
> overestimated DeepSeek by ~10×. Pricing is now per-provider in `config.yaml`.
> Treat these two as notional.
