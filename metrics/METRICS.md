# ClaimWise — Metrics

Single source of truth for every number in this project (CLAUDE.md §6).

Rules:
- Every entry records **date, git commit hash, config used, and the number**.
- No metric is written here unless it was computed from a real run's output.
- Deltas over absolutes: frame results as "X → Y after change Z".
- Negative results are recorded, not hidden.

---

## Phase 1 — Naive RAG Baseline

### 1.1 Corpus (from `python -m phase1_rag.ingest`)

| Date | Commit | Config | Documents | Pages kept | Mean chars/page | Notes |
|------|--------|--------|-----------|------------|-----------------|-------|
|      |        | `extraction_mode=text` |  |  |  | |

### 1.2 Chunking

_Populated after `chunk.py`._

| Date | Commit | Chunk size / overlap | Chunks | Mean tokens/chunk |
|------|--------|----------------------|--------|-------------------|
|      |        |                      |        |                   |

### 1.3 Retrieval + RAGAS baseline

_Populated after `run_ragas.py`. This is the number Phase 2 must beat._

| Date | Commit | Faithfulness | Answer relevancy | Context recall | Context precision | hit@5 | p50 latency | p95 latency | Tokens/query |
|------|--------|--------------|------------------|----------------|-------------------|-------|-------------|-------------|--------------|
|      |        |              |                  |                |                   |       |             |             |              |

---

## Phase 2 — Advanced RAG

_One row per technique, so every improvement is attributable._

| Date | Commit | Technique added | Context recall Δ | Faithfulness Δ | Latency cost |
|------|--------|-----------------|------------------|----------------|--------------|
|      |        |                 |                  |                |              |

---

## Modal credit ledger (Phases 4 / 4.5)

Starting balance: **$30.00** (hard cap). $5.00 is reserved for Phase 5 and never spent on training.

| Date | Run | Estimated cost | Actual cost | Remaining |
|------|-----|----------------|-------------|-----------|
|      | —   | —              | —           | $30.00    |
