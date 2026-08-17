# Session handoff — 2026-08-17

Where things stand and what to do next. Delete once Phase 3 is under way.

---

## Uncommitted work

Everything below is in the working tree and **not yet committed**. Last commit
was the Phase 2 batch.

| File | State |
|---|---|
| `phase3_agents/__init__.py` | new |
| `phase3_agents/state.py` | new — the graph contract |
| `phase3_agents/claims_calculator.py` | new — **10/10 self-test passing** |
| `phase3_agents/router.py` | new — **measured and rejected (D-22)** |
| `evals/clean_golden.py` | new — **already run with `--write`** |
| `phase1_rag/build_eval_set.py` | `question_fingerprint()` added |
| `config.yaml` | `router:` block added |
| `data/eval/golden.jsonl` | **CHANGED: 100 → 92 items** |
| `data/eval/golden.raw.jsonl` | the preserved 100-item original |
| `decisions.md` | D-22, P-19 |
| `metrics/METRICS.md` | §2.7 corrected baseline, Phase 3 router row |
| `metrics/failure_analysis_p1.md` | correction banner |
| `workflow.md` | Phase 3 entries |

Suggested commit:

```
git add -A
git commit -m "Phase 3: calculator (10/10), router rejected on measurement; golden set deduplicated"
```

---

## What happened this session

1. **Phase 2 closed.** Exit criterion met: hit@5 0.694 → 0.871, **+17.7 points**.
   Blog written (`docs/blog/phase2.md`), failure analysis written.
2. **Generation eval ran** on the adopted pipeline: positives with evidence
   **59 → 74**, absolute hallucinations **20 → 10**.
3. **Phase 3 started.** State contract and claims calculator built and passing.
4. **Router built, measured, rejected.** 0.359 against a 0.837 majority-class
   baseline.
5. **Golden set cleaned**: 8 duplicates removed (not 1 — see P-19), 2 questions
   scoped. Re-baselined: hit@5 0.8706 → **0.8701**, so nothing material moved.

---

## Current best pipeline

Density-selected chunking (life → 400/2,000 parent-child, others → flat
1,000/150) + cross-encoder rerank@20 over children, parents expanded **after**
reranking. Collection `claimwise_mx__baai_bge_small_en_v1_5`.

| Metric | Value (cleaned 92-item set) |
|---|---|
| hit@5 (doc+page) | **0.8701** |
| MRR | 0.6462 |
| health / home / life | 0.861 / 0.895 / 0.867 |
| p50 latency | ~3.8 s (CPU reranking dominates) |

---

## Next actions, in order

### 1. Replace the router with an LLM classifier (D-22)
The exemplar approach is closed off with evidence. Build an LLM classification
node calling `common/generator.py` (NIM is the configured provider and has still
**never been exercised in Phase 3** — every module so far is zero-LLM).

Measure it with the harness that already exists:
```
uv run python -m phase3_agents.router --eval
```
It must beat **0.837**, not 0.359. Keep the majority-class baseline in the
output — that comparison is what made the first result interpretable.

### 2. Confidence gate
Highest-value remaining item. On the 11 positives where retrieval fails, the
model answers instead of refusing **10 times**. Threshold already measured in
P-14: cross-encoder top-1 at **0.20–0.25** (negatives 0.0985, genuine hits
0.85–0.99). Free to evaluate against the 15 negatives.

### 3. LangGraph assembly
router → retrieval → gate → calculator → generate. First module needing the
`langgraph` dependency.

### 4. The 50-task agent eval set
Blocks the phase exit criterion, and supplies the `calculation` / `comparison`
labels the router eval currently cannot measure (it covers 2 of 4 routes).

### 5. Comparison agent

---

## Open items carried forward

- **Phase 2 judged metrics are unusable.** Generator and judge both changed
  mid-phase and the judge ended up scoring its own output. Faithfulness /
  answer relevancy are recorded but explicitly not comparable. A clean pair
  needs one fixed generator and one fixed independent judge.
- **Two judges rejected**: `gpt-oss-120b` (reasoning tokens break RAGAS's
  parser, 4/4 failures) and `nim/gemma-4-31b-it` (HTTP 500 on every judge call;
  generation-capable ≠ judge-capable).
- **Eight retrieval misses survive everything**: `g-002`, `g-012`, `g-017`,
  `g-030`, `g-041`, `g-058`, `g-069`, `g-076`. All administrative boilerplate
  that repeats across documents.
- **`data/processed/` is gitignored** and must be rebuilt after any clone:
  `ingest` → `chunk_policy` → `embed_index --input mixed_chunks.jsonl
  --collection-prefix claimwise_mx`.

---

## Budget

| | Spent | Remaining |
|---|---|---|
| OpenRouter (only paid path) | ~$0.40 | **~$1.60** |
| Modal (Phases 4 / 4.5) | $0.00 | **$30.00** |

Reference: a judged 100-question generation eval costs **~$0.15**, not the
~$0.045 first estimated — RAGAS sends the full retrieved context per claim.

---

## Working-agreement change

The CLAUDE.md rule "Claude writes code, never executes it" was **lifted by the
user this session**. CLAUDE.md itself has not been edited to reflect that —
worth deciding whether the change is permanent or was session-scoped.
