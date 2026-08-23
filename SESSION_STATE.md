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

## 2026-08-17, later session — confidence gate done

`phase3_agents/confidence_gate.py` built, swept and **adopted at
`refuse_below: 0.02`** (D-23). Self-test 26/26. Cost: nothing — the sweep
re-analyses recorded runs, no LLM call and no retrieval re-run.

- Refusal accuracy **0.933 → 1.000** combined with the generator, **zero** new
  false refusals, 9 generator calls skipped.
- **P-14's 0.20–0.25 was ~10× too high** — at 0.20 it falsely refuses 9 of 67
  answerable questions. The lowest genuine hit scores 0.0414.
- **The gate does NOT fix the §1.7 hallucination defect.** Blind positives
  average 0.5061 vs grounded 0.6837 — the cross-encoder scores topical
  relevance, not answer presence. 1–2 of 10 at any zero-cost threshold. That
  hole now belongs to Phase 4's RAFT negatives, measured rather than assumed.

**Working agreement — REVERTED 2026-08-18 to full write-only.** §1 was briefly
rewritten to split on who *pays*; after three builds ran that way I reverted it.
Claude writes code and never executes it, Execution Card for everything including
free self-tests. CLAUDE.md §1 now carries an amendment-history note, and records
that a verbal lift is not in force until the section is actually edited.

Consequence for the numbers below: the confidence-gate sweep, the agent task-set
build and the router evals were all **run by Claude**, not by me. They are local,
deterministic and free, so they reproduce exactly — but if I want them under the
normal discipline I should re-run them myself and confirm the figures match
before they are treated as mine.

Uncommitted: `phase3_agents/confidence_gate.py` (new), `config.yaml`
(`confidence_gate:` block), `CLAUDE.md` (§1 rewritten), `metrics/METRICS.md`,
`decisions.md` (D-23), `SESSION_STATE.md`, and
`evals/results/gate_sweep_mx-rr20-clean.json`.

---

## Next actions, in order

### 0. Wire the gate into the served pipeline
Built and measured, but nothing calls it yet — `gate_node()` is written in the
LangGraph shape and is still unreferenced. It lands with the graph (item 3).

### 1. ~~Fix the router's `lookup` exemplars, then re-measure~~ ✅ done 2026-08-18 (D-25)
Exemplars only, 32 → 54. **Lookup recall 0.325 → 0.714 on the golden set**
(+38.9), which is the least-biased number available — those 77 questions predate
this module. Overall: agent set 0.780 → **0.820**, golden 0.359 → **0.652**.

Cost-weighted, golden now has **0 harmful misroutes**; the agent set's 2 are
`t-027`/`t-028`, which no tool can settle anyway. Out-of-scope regressed on
golden (0.533 → 0.333) and that is structural — lookup's 22 exemplars outnumber
out_of_scope's 12 under max-per-route scoring — but those misroutes are refused
downstream by the D-23 gate, measured 15/15 on those exact items.

Still worse than "always lookup" on golden (0.652 vs 0.837). That comparison is
not the one that matters: the constant scores **0.000 on `calculation`**, the
route that keeps the calculator firing.

**Do not tune the exemplars again.** Two revisions have been measured against
both sets; a third would be fitting to these 142 items. The next real control is
an independently authored task set.

### 1b. Superseded — the original D-22 plan
The 50-task set exists now, and on it the **unchanged** router scores **0.78
against a 0.28 majority baseline** — D-22's rejection was measured on a set that
is 84% `lookup`, which is the router's one weak class (recall 0.500; comparison
0.917, out_of_scope 0.900, calculation 0.857). Only **3 of 50** misroutes are
harmful.

So the next move is *not* an LLM classifier. It is: broaden the `lookup`
exemplars (14 varied lookups currently match 10 narrow examples), pull "knee
surgery" out of the comparison exemplar that steals `t-019`/`t-020`, then:

```
uv run python -m phase3_agents.router --eval --tasks   # 4 routes, baseline 0.28
uv run python -m phase3_agents.router --eval           # 2 routes, baseline 0.837
```

Report **both**. Neither set alone is honest: the agent set over-represents the
rare routes, the golden set over-represents lookup, and 0.78 carries an
authorship caveat (same person wrote the tasks and the exemplars).

### 1b. If that is not enough — the LLM classifier (D-22)
The exemplar approach is closed off with evidence. Build an LLM classification
node calling `common/generator.py` (NIM is the configured provider and has still
**never been exercised in Phase 3** — every module so far is zero-LLM).

Measure it with the harness that already exists:
```
uv run python -m phase3_agents.router --eval
```
It must beat **0.837**, not 0.359. Keep the majority-class baseline in the
output — that comparison is what made the first result interpretable.

### 2. ~~Confidence gate~~ ✅ done 2026-08-17 — see D-23 above.

### 3. LangGraph assembly
router → retrieval → gate → calculator → generate. First module needing the
`langgraph` dependency.

### 4. ~~The 50-task agent eval set~~ ✅ done 2026-08-18 (D-24)
`data/eval/agent_tasks.jsonl`, built by `evals/build_agent_tasks.py`. 50 tasks,
14/14/12/10 across the four routes, majority baseline **0.28**. Every citation
verified against the corpus and every rupee figure verified against the
calculator at build time; the build refuses to write on any disagreement.

**Still to do on top of it:** the agent eval *harness*. Task completion,
tool-call accuracy, escalation rate and avg steps all need the LangGraph agent
(item 3), so the harness lands with it. What is already measurable and already
measured is router accuracy on all four routes.

**Also not yet done:** running `evals.retrieval_metrics` over `agent_tasks.jsonl`
to find out whether the ground-truth pages are even retrievable. The schemas
differ (`agent_tasks` has no `source_chunk_id` / `policy_type` / 
`ground_truth_doc_id`), so it needs a small loader change. Worth doing before the
graph is wired — if a comparison task's second document never surfaces, that is
a retrieval problem the agent cannot fix.

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
