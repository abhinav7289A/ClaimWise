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

## API spend (free tiers, tracked for the cost table)

| Date | Run | Provider | Est. cost |
|------|-----|----------|-----------|
| 2026-08-12 | `free2` generation eval | HF Inference | $0.1032 |
| 2026-08-12 | `ragas-smoke` | HF Inference | $0.0105 |
