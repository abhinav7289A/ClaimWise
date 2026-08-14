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

> **⚠️ RAGAS faithfulness and answer relevancy were never successfully
> computed.** The smoke run failed on
> `No module named 'langchain_community.chat_models.vertexai'`, and the full
> `free2` run was executed with `--skip-ragas`. Phase 1's stated exit criteria
> are therefore **not fully met** on paper. Per CLAUDE.md §6 these cells stay
> empty rather than being filled with plausible values.

| Faithfulness | Answer relevancy |
|--------------|------------------|
| _not computed_ | _not computed_ |

### 1.6 Phase 1 exit criteria status

| Criterion | Status |
|---|---|
| Baseline retrieval numbers recorded | ✅ |
| Baseline generation numbers recorded | ⚠️ free metrics only; RAGAS pair missing |
| `metrics/failure_analysis_p1.md` — 10 worst failures analysed | ❌ not written |

---

## Phase 2 — Advanced RAG

_One row per technique, so every improvement is attributable._

| Date | Commit | Technique | hit@5 (page) | Δ | MRR | Pool recall | p50 latency | Verdict |
|------|--------|-----------|--------------|---|-----|-------------|-------------|---------|
| 2026-08-12 | `97545ea+wt` | *(baseline: dense, top-5)* | 0.694 | — | — | 0.894 @20 | ~120 ms | — |
| 2026-08-13 | `97545ea+wt` | **1. Cross-encoder rerank** @20 | **0.812** | **+11.8** | 0.6158 | 0.894 @20 | 10,217 ms | ✅ **adopted** |
| 2026-08-14 | `97545ea+wt` | 2. Hybrid BM25 + RRF @20 | 0.812 | **+0.0** | 0.6177 | 0.859 @20 | 10,851 ms | ❌ **rejected (D-17)** |

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
