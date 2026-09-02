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

### 2.7 CORRECTED baseline on the cleaned golden set — 2026-08-17

`evals/clean_golden.py --write` was run. It removed **8 duplicate questions, not
the 1 recorded in `failure_analysis_p1.md`** — that document generalised from the
single pair (`g-069`/`g-077`) that had been inspected by hand; the fingerprint
found the rest. Removed: `g-061`, `g-063`, `g-073`, `g-074`, `g-075`, `g-077`,
`g-079`, `g-082`, each byte-identical to a survivor. Two questions were scoped
(`g-030`, `g-058`). Original preserved at `data/eval/golden.raw.jsonl`.

**Golden set: 100 items (85 pos / 15 neg) → 92 items (77 pos / 15 neg).**

Every metric recorded above §2.7 was computed on the 100-item set, where 8
questions were double-weighted. Re-measured on the cleaned set, same pipeline
(`mx-rr20-clean`):

| Metric | Old set (100) | **Cleaned set (92)** | Δ |
|---|---|---|---|
| hit@5 (doc+page) | 0.8706 | **0.8701** | −0.001 |
| hit@1 | 0.5294 | 0.5195 | −0.010 |
| hit@3 | 0.7059 | 0.7143 | +0.008 |
| hit@10 | 0.8824 | 0.8831 | +0.001 |
| MRR | 0.6501 | 0.6462 | −0.004 |
| health / home / life | 0.861 / 0.905 / 0.857 | 0.861 / 0.895 / 0.867 | — |

**The duplicates were not biasing the result.** Every figure moves by less than
one point, so the Phase 2 conclusions and the technique ranking stand as
recorded. Complete misses drop 9 → 8 (`g-077` was the duplicate).

Numbers above §2.7 are kept as measured rather than restated — they were
correct for the set they ran against, and the delta here is the honest bridge.

### Phase 3 — router

| Date | Component | Result | Verdict |
|---|---|---|---|
| 2026-08-17 | Exemplar router (bge-small, 32 exemplars) | accuracy **0.359** | ❌ **rejected (D-22)** |

| | Accuracy |
|---|---|
| Overall | 0.3587 |
| **Majority-class baseline ("always lookup")** | **0.8370** |
| Router when *confident* (29 decisions) | 0.4138 |
| Router when ambiguous (63 decisions) | 0.3333 |

Worse than a constant, and worse than a constant even on its own confident
subset — so the hybrid "cheap pre-filter with LLM fallback" design is dead too.
Measured on 2 of 4 routes; `calculation` and `comparison` have no labels until
the 50-task set exists.

### Phase 3 — the 50-task agent eval set, 2026-08-18

`python -m evals.build_agent_tasks --write` → `data/eval/agent_tasks.jsonl`.
Authored, not generated (D-24). Every citation machine-verified against
`data/processed/pages.jsonl`; every rupee figure hand-computed and verified
against `phase3_agents.claims_calculator`. Build fails on any disagreement.

| | Count |
|---|---:|
| Total | **50** |
| lookup / calculation / comparison / out_of_scope | 14 / 14 / 12 / 10 |
| **Majority-class baseline** | **0.28** |
| Calculation tasks the calculator can settle today | 12 of 14 |
| Tasks needing tooling that does not exist | 2 (`t-027` depreciation, `t-028` ULIP withdrawal) |
| Tasks whose correct answer is "provisional, term not in the wording" | 2 (`t-025`, `t-039`) |
| Evidence spans star / sbih / home / life | 24 / 19 / 7 / 3 |

The mix is deliberately **not** natural traffic: `lookup` is under-weighted
because 92 lookup items already exist in `golden.jsonl`, and the two routes with
no labels anywhere are over-weighted. **The 0.28 baseline here and the 0.837
baseline on the golden set are not comparable.**

### Phase 3 — router re-measured on all 4 routes (D-24), 2026-08-18

`python -m phase3_agents.router --eval --tasks`. Same router, same code path,
different set — the one D-22 said it could not be measured on.

| Set | Items | Accuracy | Majority baseline | Margin |
|---|---:|---|---|---|
| `golden.jsonl` (2 routes) | 92 | 0.3587 | 0.8370 | **−47.8** |
| **`agent_tasks.jsonl` (4 routes)** | 50 | **0.7800** | **0.2800** | **+50.0** |

Per-route recall, which is the part that matters:

| Route | Recall |
|---|---|
| comparison | 11/12 = **0.917** |
| out_of_scope | 9/10 = **0.900** |
| calculation | 12/14 = **0.857** |
| **lookup** | 7/14 = **0.500** |

**Cost-weighted, only 3 of 50 misroutes are harmful** — `t-019`/`t-020`
(calculation → comparison, so the calculator never runs and an LLM does the
arithmetic) and `t-050` (out_of_scope → calculation). The other 8 are
lookup → calculation/comparison, which cost an unnecessary stage but leave the
answer reachable, because the routes are additive.

**D-22's rejection was an artifact of the eval set, not a property of the
router.** The golden set is 84% lookup, so "always lookup" is nearly unbeatable
there and the router's one weak class is exactly the class that dominates it.
**Read with the authorship caveat in D-24: the same person wrote these 50 tasks
and holds the exemplars, so 0.78 is an upper bound.**

### Phase 3 — router exemplars revised (D-25), 2026-08-18

`python -m phase3_agents.router --eval --tasks` and `--eval`. Exemplars only —
no code change to the routing mechanism. 32 → 54 exemplars
(lookup 10→22, calculation 8→10, comparison 6→10, out_of_scope 8→12).

| | agent_tasks (50) | golden (92) |
|---|---|---|
| Majority baseline | 0.28 | 0.837 |
| Accuracy before | 0.7800 | 0.3587 |
| **Accuracy after** | **0.8200** | **0.6522** |
| Δ | **+4.0** | **+29.3** |

Per-route recall — `orig` → `v1` → **`v2`**:

| Route | agent_tasks | golden |
|---|---|---|
| **lookup** (the target) | 0.500 → 0.857 → **0.714** | 0.325 → 0.727 → **0.714** |
| calculation | 0.857 → 0.857 → **0.857** | — |
| comparison | 0.917 → 0.750 → **0.833** | — |
| out_of_scope | 0.900 → 0.300 → **0.900** | 0.533 → 0.067 → **0.333** |

**Lookup, the class this was aimed at, roughly doubled on the golden set —
0.325 → 0.714, +38.9 points.** That is the least-biased figure available: those
77 lookups were LLM-generated in Phase 1, months before this module existed, and
were not touched today.

> **Provenance, 2026-08-19 — both rows now confirmed by the user.** Golden:
> 0.6522 / lookup 55/77 / out_of_scope 5/15. Agent set: 0.8200, per-route
> 12/14 calculation, 10/12 comparison, 10/14 lookup, 9/10 out_of_scope. Both
> reproduced exactly. The rest of the Phase 3 suite was reproduced in the same
> session: calculator 10/10, gate 26/26, task-set build 50 items with all
> citations and figures verified.
>
> **Exemplar correction — final measured state, 2026-08-20.** The
> commercial-property out_of_scope exemplar went through reword then removal
> (D-25 addendum). Golden set, all three states run by the user:
>
> | State | Accuracy | lookup | out_of_scope |
> |---|---|---|---|
> | v2, `"...fire insurance policy for a shop..."` | 0.6522 | 55/77 | 5/15 |
> | reworded to `"...shopkeeper's policy...business premises"` | 0.6522 | 55/77 | 5/15 |
> | **removed** | **0.6630** | **56/77** | **5/15** |
>
> The reword was exactly neutral — it cleared `g-051` and took `g-045` in its
> place, both home-policy questions. Removal cleared both and cost nothing on
> the negatives, confirming the exemplar was pure liability against this corpus.
> **Golden accuracy of record: 0.6630.** The agent-set row (0.8200) predates the
> removal; none of its out_of_scope items are commercial-property, so it is
> expected to be unchanged, but it has not been re-run.

**`v1` was wrong and the measurement caught it.** The first revision replaced the
out_of_scope insurance lines (motor, travel, marine, crop) with disjoint ones to
break contamination against `agent_tasks.jsonl`. Out-of-scope recall collapsed —
golden 0.533 → 0.067 — because golden's 15 hand-seeded negatives *are* those
lines (4 travel, 3 motor, 3 marine, 1 crop). The contamination was only ever with
the agent set; the exemplars were carrying real signal. Restored in v2.

#### Three caveats, and the third is the one that matters

1. **This is a fitted number.** Exemplars were revised while looking at these two
   sets. The revisions target stated structural defects rather than individual
   failures (D-25's P1–P4), but that is an assurance, not a control.
2. **On the realistic distribution the router still loses to a constant.** Golden
   0.652 against a 0.837 majority baseline. Lookup improved hugely; the overall
   verdict on that set did not change.
3. **Cost-weighted, that comparison is the wrong one.** Routes are additive, so
   misroutes are not equal:

| | agent_tasks | golden |
|---|---|---|
| Harmful (calculation misrouted → calculator never fires, LLM does arithmetic) | **2** | **0** |
| Recoverable (out_of_scope → a retrieving route; the D-23 gate refuses these, 15/15 on these exact items) | 1 | 10 |
| Benign (lookup/comparison → extra stage, answer still reachable) | 6 | 22 |

Both remaining harmful cases are `t-027` and `t-028` — the two tasks flagged
`requires_unimplemented`, which no tool can settle regardless of routing.
`t-019`/`t-020`, harmful before, now route correctly: P2 removed the "knee
surgery" noun from a comparison exemplar.

**The router's value is in the `calculation` route specifically** — 0.857, and the
calculator firing is what keeps CLAUDE.md's no-LLM-arithmetic rule enforceable.
"Always lookup" scores 0.837 on golden and **0.000 on calculation**, a route
golden does not contain. That is what the 50-task set was built to expose.

### Phase 3 — confidence gate (D-23), 2026-08-17

`python -m phase3_agents.confidence_gate --sweep --fine-grid --vs-generator ...`
Offline re-analysis of retrieval run `mx-rr20-clean` (2026-08-17, rerank@20,
`claimwise_mx__baai_bge_small_en_v1_5`, 92 items) joined to generation run
`mx-rr20-gen`. **Zero cost — no LLM call, no retrieval re-run.** Self-test 26/26.

**Top-1 cross-encoder score by class — read this before any threshold:**

| Class | n | Mean | Median | Min | Max |
|---|---:|---|---|---|---|
| grounded (correct page reached the prompt) | 67 | 0.6837 | 0.7728 | **0.0414** | 0.9992 |
| blind (positive, correct page did NOT) | 10 | **0.5061** | 0.5124 | 0.0073 | 0.9983 |
| negative (unanswerable by construction) | 15 | **0.0964** | 0.0174 | 0.0006 | 0.7856 |

**Adopted: `refuse_below: 0.02`.** Measured against the generator's own refusals,
since the gate runs before it and the two compose:

| | Generator alone | + gate @ 0.02 | Δ |
|---|---|---|---|
| Negatives refused | 14 / 15 | **15 / 15** | **+1, perfect** |
| Blind positives refused | 1 / 10 | 1 / 10 | **0** |
| New false refusals | — | **0 / 67** | **none** |
| Generator calls skipped | 0 | **9** | latency + tokens saved |

#### Two findings, and the second one matters more

**1. P-14's recommended 0.20–0.25 was ~10× too high.** It was derived from the
negatives alone and never priced what the same cut does to genuine hits. At 0.20
the gate falsely refuses **9 of 67** answerable questions to gain one extra
catch. The lowest-scoring *genuine* hit (`g-067`) sits at **0.0414**, so the
entire usable band is narrower than the original grid's first step.

**2. The gate does NOT fix the defect it was built for.** §1.7's hallucination
— the generator answers when retrieval failed — is barely addressable by this
signal. Blind positives average **0.5061**, far closer to grounded (0.6837) than
to negatives (0.0964). At every threshold costing zero false refusals the gate
catches **1–2 of 10**; reaching 7 of 10 costs 20–30 false refusals and drives net
negative.

The cross-encoder answers *"is this passage on-topic for this question"*, not
*"does this passage contain the answer"*. Those coincide for an out-of-corpus
question and come apart for an in-scope question whose specific clause was
missed — which is exactly the blind set. **The confidence gate closes the
out-of-scope hole; it does not close the ungrounded-answer hole.** That remains
Phase 4's RAFT negatives, and this is now measured rather than assumed.

The complementarity is nearly exact: the one negative the generator fails to
refuse is **`g-099`, scoring 0.0008** — the second-lowest score in the whole set.
The gate catches at any threshold above 0.001.

**Caveats.** The 0.0414 ceiling is set by a single item, so the margin is the
adoptable finding, not the exact value — 0.02 was chosen over the
accuracy-optimal 0.04 for that reason (0.04 sits 0.0014 from `g-067`). The
generation side of the join is the older 100-item run; 92 items intersect and
only those were scored. Escalation band `[0.02, 0.50)` is a product judgement,
not a tuned number — it flags 15 of 67 answerable questions (22%) for review.

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

### 2.8 Document scoping on the agent retrieval path — 2026-09-02 (D-33)

Commit: `ab8ad3f`. Config: `pipeline=chunk_policy`,
`per_doc_depth=10`, `per_doc_top_k=2`, `comparison_top_k=6`, `max_documents=4`,
`document_relevance_floor=0.15` (stood down when 2+ documents resolve).
Eval set: `data/eval/agent_tasks.jsonl`, 40 pos / 10 neg, **gold route labels,
not the router**. Both runs through `phase3_agents.retrieval_node`. 0 LLM calls,
local CPU, $0.

Command:
`python -m evals.retrieval_metrics --tasks --agent-retrieval [--scope-documents]`

| Metric | scope OFF | scope ON | Δ |
|---|---|---|---|
| doc+page@3 | 0.525 | 0.550 | **+0.025** |
| doc+page@5 | 0.600 | **0.650** | **+0.050** |
| doc+page@10 | 0.625 | 0.650 | +0.025 |
| all docs cov@5 (overall) | 0.475 | **0.500** | +0.025 |
| MRR | 0.4175 | **0.4279** | +0.010 |
| page recall@5 | 0.5375 | **0.5750** | +0.0375 |
| complete misses | 15 | **14** | −1 |
| latency p50 | 10,896.7 ms | **3,595.6 ms** | **−3.03x** |
| latency p95 | 18,924.3 ms | **5,893.2 ms** | **−3.21x** |
| comparison page hit@5 | 0.667 | **0.750** | +0.083 |
| lookup page hit@5 | 0.571 | **0.643** | +0.072 |
| calculation page hit@5 | 0.571 | 0.571 | +0.000 |
| **multi-doc all-docs-cov@5** | **0.250** | **0.250** | **+0.000** |

Resolution mix (50 tasks): `insurer` 13 · `plan` 8 · `policy_type` 7 ·
`policy_type/ambiguous` 5 · `insurer/ambiguous` 1 · `none` 16.

Calculation is byte-identical because those tasks never name a policy — the
resolver correctly no-ops on all of them.

**Verdict: adopted, target metric unmet.** The multi-document exit criterion
stays at 0.250 and Phase 3's comparison route remains a FAIL. Per-item analysis
shows the quota now reaches **both** required documents in 5 of the 6
half-answered tasks — the remaining failure is the correct page ranking below 3
*within its own document*, i.e. fetch depth, not slot allocation. Closed as
future scope; see D-33.

Result files: `evals/results/retrievalagent_20260902T181934Z_agent-scope-off.json`,
`evals/results/retrievalagent_20260902T193229Z_agent-scope-on.json`.

---

## Modal credit ledger (Phases 4 / 4.5)

Starting balance: **$30.00** (hard cap). $5.00 is reserved for Phase 5 and never spent on training.

| Date | Run | Estimated cost | Actual cost | Remaining |
|------|-----|----------------|-------------|-----------|
| — | opening balance | — | — | $30.00 |
| 2026-08-24 | `check_compat` probes (3 runs, 2 false FALLBACK from pinned torch — P-20) | — | **$0.077** | ~$29.92 |
| 2026-08-27 | 4 failed training launches (TRL `SFTTrainer` — P-21) | — | *dashboard* | |
| 2026-08-27 | `sft-v1` smoke, 10 steps, 378.2s elapsed | — | *dashboard* | |
| 2026-08-27 | `sft-v1` full train, 236 steps, 3,279.1s elapsed | $0.95 | *dashboard* | |
| 2026-08-27 | `benchmark --limit 10`, 2 models | — | *dashboard* | |
| 2026-08-27 | `benchmark` full 92, 2 models | ~$0.65 | *dashboard* | |

> **The `dashboard` cells are not yet real numbers and must not be treated as
> such.** Only the $0.077 compat figure was read from Modal billing. Everything
> after it is derived from wall-clock elapsed times at an *unverified* $0.80/L4-hour
> rate, which gives a rough total of **~$1.84 spent, ~$28.16 remaining**. Read the
> actual figures off the Modal dashboard and replace these cells; the rate itself
> also needs checking against modal.com/pricing, since every estimate in this file
> depends on it.

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

---

## Phase 4 — Supervised Fine-tuning (RAFT)

Commit: working tree, uncommitted as of 2026-08-27. Re-stamp when Phase 4 lands.

### 4.1 Calculation slice — three iterations, two real defects

Generated by `phase4_finetune.gen_calc_examples`. Zero LLM calls, $0.00, all three runs.

| Run | Date | kept | Defect found |
|---|---|---:|---|
| `calc-v1` | 2026-08-26 | **5** | Oracle matched on page number alone; **3 of 5 rows cited a page belonging to a different insurer's policy** |
| `calc-v2` | 2026-08-26 | **19** | `(doc_id, page)` match fixed that, but 161 of 180 questions failed to retrieve their own citing page |
| `calc-v3` | 2026-08-26 | **63** | Adopted. Oracle injected when retrieval misses it (RAFT construction) |

`calc-v3` funnel: 1,316 chunks processed → 72 carrying an extractable term →
231 candidates (77 chunk×term pairs × 3 phrasings) → 3 no-template, 165 duplicate
fingerprints → **63 kept**. `oracle_retrieved: 10`, `oracle_injected: 53`, so
**84% of calculation rows carry a hand-placed oracle**. `dropped_holdout: 0`.

**Ceiling is 21.** Unique fingerprints equal kept rows, and 63 ÷ 3 phrasings = 21
distinct (document, term type) pairs in the whole corpus. Health policies only —
the motor and life documents contributed zero calculable terms. More rows would
need more phrasings or more documents, not more chunks.

Citation audit (`phase4_finetune.audit_calc_rows`), `calc-v3`: **0 / 63
cross-document citations, 0 / 63 rows without an oracle**, oracle rank spread
22/22/17/2 across ranks 1–4. Oracle insurers hdfcergo 33, sbigeneral 15,
starhealth 9, nivabupa 6.

### 4.2 Merged SFT dataset

`phase4_finetune.build_train_split`, 2026-08-27. Free.

| | |
|---|---:|
| `raft_20260824T193308Z_v1` | 979 |
| `raft_20260826T215520Z_calc-v3` | 63 |
| merged | **1,042** |
| dropped duplicate / holdout / empty context / empty answer | 0 / **0** / 0 / 0 |
| train | **938** (lookup 543, negative 231, over_refusal 107, calculation 57) |
| val | **104** (lookup 60, negative 26, over_refusal 12, calculation 6) |
| longest user message | 10,652 chars (~2,663 tokens rough) |

**1,042 rows against CLAUDE.md's 3–5k target**, of which calculation is 63 (6%).
Recorded as-is rather than padded — 32.5M trainable parameters learning a format
and a refusal policy did not need 5k rows, and the benchmark below bears that out.

The longest-message figure is what set `seq_len=4096` (D-32); 2048 would have
silently truncated.

### 4.3 Training run — `sft-v1`

`phase4_finetune.train_modal`, 2026-08-27, Modal L4. bf16 LoRA, **not QLoRA** (D-31).
Plain `transformers.Trainer`, not TRL (P-21). Record: `evals/results/train_sft-v1.json`.

| | Smoke (10 steps) | Full run |
|---|---:|---:|
| steps | 10 | **236** |
| train_loss | 0.4771 | **0.2049** |
| eval_loss | — | **0.2150** |
| seconds/step | — | 13.37 |
| train runtime | — | 3,156.3 s (52.6 min) |
| model load overhead | — | 122.8 s |
| peak VRAM | 11.39 GB | **12.49 GB** |
| rows truncated at seq 4096 | 0 | **0** |

trainable 32,464,896 (vision tower 0, enforced by a hard check).
Stack resolved: torch 2.11.0+cu130, transformers 5.5.0 (nothing pinned — P-20).
Pushed merged 16-bit to **`AbhiCommits/claimwise-qwen35-4b`**.

**eval_loss 0.2150 against train_loss 0.2049** — a gap that small at 2 epochs
means the format was learned without memorising the training set.

### 4.4 Generator swap — Phase 4 exit criterion

`phase4_finetune.benchmark`, 2026-08-27. Retrieval run **once**, locally, against
the frozen 4-document `claimwise_mx` collection; both models answered from
byte-identical prompts, so every delta is the generator alone. Scoring is fully
deterministic — no judge model, $0.00 of OpenRouter. 92 golden questions
(77 positive, 15 negative), held out of training by fingerprint.

Record: `evals/results/benchmark_20260827T153421Z_phase4.json`.

| Metric | base `Qwen3.5-4B` | fine-tuned | Delta |
|---|---:|---:|---:|
| cited anything | 0.0649 (5/77) | **0.6104 (47/77)** | **+54.6 pts** |
| cited the correct page | 0.0390 (3/77) | **0.5195 (40/77)** | **+48.1 pts** |
| hallucinated when it cited | 0.0000 (0/5) | **0.0000 (0/47)** | — |
| figures preserved | 0.6364 (21/33) | **0.8182 (27/33)** | **+18.2 pts** |
| correct refusal on negatives | 0.6667 (10/15) | **0.9333 (14/15)** | **+26.7 pts** |
| wrongly refused a positive | 0.0519 (4/77) | 0.1169 (9/77) | **−6.5 pts (worse)** |

**Zero hallucinated citations across 47 citing answers** is the headline: the model
learned to cite *and* stayed inside the passages it was shown. The base model
barely cites at all, which is why its unconditioned hallucination rate was also
0.0 — that metric was replaced mid-run with one conditioned on answers that
actually cited, because a model that never cites cannot cite wrongly.

**The refusal training was close to a wash in absolute terms** and should not be
reported as a clean win: +4 correct refusals on negatives, −5 positives wrongly
refused. It reads as a large gain only because the negative set is 15 questions
against 77 positives. Over-refusal is the safer failure mode for an insurance
assistant, but the honest statement is "gained 4, lost 5".

`figures_preserved` moving 21/33 → 27/33 was mildly surprising given the
calculation slice was only 6% of training.

### 4.5 Phase 4 exit criteria status

| Criterion | Status |
|---|---|
| Fine-tuned model clearly ≥ base | **MET** — decisive on citation behaviour, the core RAFT objective |
| Cost comparison table recorded | **PARTIAL** — GPU costs are elapsed-time estimates; actuals pending from the Modal dashboard |
| Modal spend tracked with remaining balance | **PARTIAL** — see the ledger caveat above |
| 10-image vision sanity suite | **BLOCKED** — no images exist in `data/`. Training-side evidence (0 trainable vision params, hard-checked) proves the weights did not move, not that the image path still works end to end |

---

## Phase 5 — Deployment

### 5.1 Modal inference endpoint — first light, 2026-09-03

Commit: working tree at `ab8ad3f`. Serving `AbhiCommits/claimwise-qwen35-4b`
(Phase 4 merged 16-bit) from `deployment/modal_inference.py` on Modal, L4,
`min_containers=0`, `scaledown_window=120`. transformers + TextIteratorStreamer,
**not** vLLM (CLAUDE.md §4). OpenAI-compatible, so the model swap cost **zero
lines** of change upstream of `common/generator.py`.

| Measure | Value |
|---|---|
| Model class resolved | `AutoModelForImageTextToText` (multimodal — vision tower present) |
| First cold start | **71.4 s** (~55 s of it the one-time ~8 GB weight download) |
| Warm latency, 20-token prompt | **6.50 s** (128 completion tokens) |
| Warm latency, 388-token RAG prompt | **4.23 s** (36 completion tokens) |
| Idle cost | $0.00 — scales to zero |
| L4 rate | **UNVERIFIED** — still the assumed $0.80/hr; read off modal.com/pricing |

Latency is dominated by tokens generated, not prompt length: the 388-token
prompt was *faster* than the 20-token one because it produced 36 tokens instead
of 128. Decode-bound, as expected for a 4B model on an L4.

### 5.2 Grounded-answer check through the production prompt

Two hand-built passages (pp. 31 and 32), the real `SYSTEM_PROMPT` from
`phase1_rag/rag_chain.py` and the real `format_passages` header layout.

Question: *"Is knee surgery covered after 18 months?"* against a passage stating
a 24-month joint-replacement waiting period.

> "No. Knee surgery is subject to a specific waiting period of 24months, so it
> is not covered after 18 months [p.32]."

Correct, correctly cited, no preamble, no chain-of-thought. **n=1 — this is a
smoke test, not a metric.** The real generator comparison is
`phase4_finetune/benchmark.py` over the golden set, and it has not been re-run
against this endpoint.

**Two serving defects found and fixed before this passed**, both in the harness
rather than the model — the P-20/P-21 pattern for a third time:

1. `from __future__ import annotations` plus a function-local `from fastapi
   import Request` made FastAPI unable to resolve the annotation, so **every**
   request returned `{"loc":["query","request"],"msg":"Field required"}`. Fixed
   by importing fastapi at module level. `BUILD` is now reported by `/health` so
   "my fix is wrong" is distinguishable from "the container is stale".
2. Qwen3.5's chat template enables **thinking mode by default**, and the first
   completion opened with *"Here's a thinking process that leads to the suggested
   answer:"* — a train/serve mismatch, since the RAFT set contains no thinking
   blocks. Fixed with `enable_thinking=False`, applied defensively.

Also fixed: `finish_reason` was hardcoded to `"stop"` and reported a completion
truncated at `max_tokens` as complete, which would have written truncated answers
into METRICS as finished ones (§6).

**A first wrong answer was the harness too.** An ad-hoc system prompt saying "if
the answer is not in the context, say it is not covered" produced exactly that
refusal. The production prompt's Rule 3 and Rule 4 exist to prevent it. No model
conclusion should have been drawn from that run, and none was.

### 5.3 Open items

| Item | Status |
|---|---|
| Verified L4 $/hour | **OUTSTANDING** — every Modal cost figure in this file still rests on an assumed $0.80/hr |
| Second cold start (Volume-cached) | **NOT MEASURED** — decides whether the 71.4 s is a one-off |
| Reranker on free-Space CPU | **RISK** — p95 was 5.9 s locally on a faster CPU; the chosen topology gives it no GPU |
| API key rotation before public launch | **OUTSTANDING** |

### 5.4 The generator swap, end to end — 2026-09-03

`python -m phase3_agents.graph --question "..." [--provider modal]`. Same
question, same index, same router, same gate; only `generator.provider` differs.

| | openrouter | modal |
|---|---|---|
| generator | `deepseek/deepseek-v4-flash-0731` | `AbhiCommits/claimwise-qwen35-4b` |
| route | lookup (0.706, margin 0.078) | lookup (0.706, margin 0.078) — identical |
| trace | router → retrieve_global → confidence_gate → generate | identical |
| confidence | 0.8855 | 0.8855 — identical |
| answer | the refusal sentence | the refusal sentence |
| cited / invalid | [] / [] | [] / [] |

**This is the Phase 5 model swap, and it cost zero lines upstream.** One
`config.yaml` provider block plus a `--provider` flag; no change to the graph,
the nodes, the prompt assembly or the citation verifier. The identical route,
trace and confidence are the evidence that only the generator moved — everything
computed before `generate` is byte-identical across the two runs.

**Both refused, and both were RIGHT.** A false refusal was suspected and
investigated; it was not one. `retrieval_node --question "knee joint replacement
waiting period"` returns sum-insured limits (starhealth p.15, 0.7798), general
surgery preconditions (sbigeneral p.21, 0.5875), prostate and gynaecological
exclusions (starhealth p.32, 0.3733) and two "reasonable and customary charges"
definitions. **The corpus contains no knee or joint-replacement waiting period.**
Refusing was the correct behaviour from both generators.

**The finding is in the gate, not the generators.** Rank-1 scored **0.7798** on a
question the corpus cannot answer, and the gate passed it at confidence 0.8855.
P-14 set the threshold at 0.20–0.25 from negatives scoring 0.0985; this clears it
by a factor of three. The gate contributed nothing and the generator caught it.
That is consistent with the caveat already recorded on the task set — 4 of 10
negatives caught — and this is a concrete instance of the same weakness: a
lexically plausible but semantically irrelevant passage scores high on the
cross-encoder.

**n=1. This is a smoke test, not a benchmark.** No quality claim is made about
either generator from it. The generator comparison of record remains
`phase4_finetune/benchmark.py` over the golden set, which has NOT been re-run
against the Modal endpoint.
