# What actually moved the needle in our RAG pipeline (and what didn't)

*Phase 2 of ClaimWise — an assistant for Indian insurance policy documents.
Six techniques, measured one at a time. Two worked, one failed instructively,
one was skipped on purpose, and the most-hyped alternative lost on every axis.*

---

## The setup

ClaimWise answers questions about Indian health, home and life insurance
policies — 60-to-100-page PDFs full of waiting periods, sub-limits, co-payments
and exclusions. Questions look like *"Is my knee surgery covered after 18 months,
and what will I get back on a ₹2.4L bill?"*

The corpus is four real policy wordings: 102 pages, 491,986 characters. The
evaluation set is ~100 golden question-answer pairs, each tagged with the exact
page that answers it, so **hit@k, MRR and page recall are pure Python
comparisons — no LLM judge, no cost, results in 30 seconds.** That property
shaped everything that follows: when evaluation is free, you can afford to be
wrong in public, repeatedly.

Phase 1 shipped a naive baseline: fixed 1,000-character chunks, bge-small
embeddings, top-5 dense retrieval. **hit@5 = 0.694.**

The Phase 2 rule was one change, one re-evaluation, so every delta is
attributable to exactly one thing.

---

## Where to start: let the data pick

Phase 1's depth sweep produced two numbers that decided the whole phase:

| Metric | Value |
|---|---|
| hit@5 | 0.694 |
| recall@50 | 0.953 |

For **95% of questions the correct page was already being retrieved** — just
ranked too low. Only 4 of 85 were genuinely unreachable.

That makes the choice arithmetic rather than taste. Reranking's addressable gap
was 26 points; better recall was worth at most 5. So: cross-encoder reranking
first, and the intuition that "hybrid search is table stakes" waits its turn.

---

## Technique 1: cross-encoder reranking — **0.694 → 0.812**

A bi-encoder embeds question and passage *separately*. That is what makes it
fast — passage vectors are computed once at upload — but the model never sees
the two together. A cross-encoder concatenates them and runs full attention
across the pair. Roughly 100× the cost per pair, nothing precomputable, which is
why it runs as a second stage over ~20 candidates rather than all 653.

**+11.8 points.** The largest single gain in the project.

### The counterintuitive part: deeper candidate pools made it worse

Since evaluation was free, we swept the candidate depth instead of guessing:

| Depth | hit@5 | Pool ceiling | % of ceiling realised |
|---|---|---|---|
| 10 | 0.776 | 0.800 | **97%** |
| **20** | **0.812** | 0.894 | 91% |
| 30 | 0.800 | 0.906 | 88% |
| 50 | 0.788 | 0.953 | 83% |

Feed the reranker more candidates and the *ceiling* rises — but the result
falls. Every extra candidate is another chance to promote a wrong chunk above
the right one, and the cross-encoder's precision degrades faster than the
ceiling climbs. Efficiency falls monotonically, 97% → 83%.

"Retrieve more, rerank harder" is exactly backwards.

### A free bonus: a usable confidence signal

We also measured what each stage scores on the 15 hand-written unanswerable
questions:

| Stage | Mean top-1 score on negatives | Usable as a confidence gate? |
|---|---|---|
| Bi-encoder cosine | 0.6687 | **No** — sits inside the positives' range |
| Cross-encoder | **0.0985** | **Yes** — 14 of 15 below 0.22 |

Genuine hits score 0.85–0.99. The bi-encoder score is famously useless as a
confidence measure and now we have the number; the cross-encoder score is a
usable one, for free, as a by-product of a stage we were adding anyway.

---

## Technique 2: hybrid search — **rejected, 0.812 → 0.812**

Everyone tells you to fuse BM25 with dense retrieval. Dense search blurs exact
tokens; BM25 nails rare terms like `AYUSH` or `Section II.23` and understands
nothing. Fuse by rank (Reciprocal Rank Fusion) because cosine sits in ~0.4–0.9
while BM25 is unbounded — there is no shared scale to normalise onto.

It did not work here. Not "gave a small gain" — actively lost ground at every
configuration we tried:

| Retrieval | Lexical width | Pool depth | Pool recall | Recovered | Evicted | Net |
|---|---|---|---|---|---|---|
| **dense** | — | 20 | **0.894** | — | — | — |
| hybrid | 30 | 20 | 0.859 | 3 | 6 | **−3** |
| hybrid | 10 | 20 | 0.871 | 3 | 5 | **−2** |
| hybrid | 5 | 20 | 0.871 | 1 | 3 | **−2** |
| **dense** | — | 30 | **0.906** | — | — | — |
| hybrid | 30 | 30 | 0.894 | 3 | 4 | **−1** |

### Why: fusion is a displacement trade, not an addition

The candidate pool is **fixed-size**. Every lexical candidate admitted evicts a
dense one. BM25's recoveries were real and repeatable — three specific questions
came back in nearly every configuration — but its evictions were equally
consistent, and there were always more of them.

The deeper reason is the constraint we had already measured in technique 1:
**the reranker's precision, not pool recall, is the binding constraint.** Fusion
can only buy recall by widening the pool, and widening the pool is precisely
what degrades a cross-encoder.

We did not try weighted RRF. As the lexical weight approaches zero, weighted RRF
*becomes* dense-only, and our results are monotone in eviction pressure
(−3 → −2 → −1). The optimum of that sweep sits at the boundary. It would have
cost a code change and three runs to rediscover the baseline.

### But one signal survived

Broken out by document, hybrid search was not uniformly bad:

| Policy type | Baseline | Hybrid | Δ |
|---|---|---|---|
| health | 0.8605 | 0.8140 | −4.7 |
| home | 0.9048 | 0.9048 | 0.0 |
| life | 0.6190 | **0.7143** | **+9.5** |

One document *loved* it. Hold that thought.

---

## Technique 3: parent-document retrieval — and a trap worth knowing

Chunk size serves two masters that want opposite things. **Retrieval wants small
chunks**: a 400-character passage about one waiting period embeds to a focused
vector, while a 2,000-character passage covering five topics embeds to the
mush-average of all five. **Generation wants large chunks**: a clause reading
*"subject to the sub-limits in Table B"* is useless without Table B.

The standard fix decouples them: index small children, return the larger parent
block they came from. We built 337 parents (2,000 chars) over 1,644 children
(400 chars).

First result: **0.812 → 0.788**. Worse.

### The trap: your reranker silently truncates

Two causes, and the first is the one most tutorials walk straight into.

`bge-reranker-base` has `model_max_length: 512`. Our median parent was 1,780
characters — about 445 tokens — and with the query prepended, **the upper half
of every parent was being truncated before the cross-encoder saw it.** It scored
each block's opening and never read its ending, which in insurance wording is
exactly where the qualifying clause lives.

Second cause: expansion deduplicates. Twenty children collapsed to a median of
15 parents, so the reranker got a smaller pool than the run it was being
compared against.

We had put expansion *before* reranking, reasoning that the cross-encoder should
read what the generator reads. That reasoning does not survive a 512-token
window. Reranking the small children first and expanding the winners afterwards
recovered 2.3 points: **−4.7 → −2.4**.

**Ranking precision and context completeness want different stages.**

Still a net loss overall — but pool recall *improved* (0.894 → 0.906), latency
dropped 7× because 400-char children are far cheaper to cross-encode, and the
per-document split repeated:

| Policy type | Baseline | Parent-docs | Δ |
|---|---|---|---|
| health | 0.8605 | 0.8140 | −4.7 |
| home | **0.9048** | 0.7619 | **−14.3** |
| life | 0.6190 | **0.7619** | **+14.3** |

An exact mirror. Same document winning, again.

---

## Technique 5: the thing that actually worked — **0.812 → 0.871**

Three independent measurements had now said the same thing. One document gains
from small chunks; the others lose. That document is the ICICI life policy, and
it is an outlier in one measurable respect:

| Document | Pages | Chars/page |
|---|---|---|
| starhealth health | 47 | 2,602 |
| sbi health | 30 | 5,856 |
| sbi home | 17 | 5,935 |
| **icici life** | **8** | **11,639** |

**Optimal chunk size is a property of the document, not of the corpus.** A
single global setting is correct for the typical document and badly wrong for
the outlier — and ours was leaving ~14 points on the table for the densest
policy we had.

So: measure characters-per-page at ingest, and chunk accordingly. Dense
documents (≥8,000 chars/page) get the 400/2,000 parent-child treatment;
everything else keeps flat 1,000-character chunks. All of it indexed into **one
collection**.

Two design choices worth defending:

**Select on measured density, not document name.** Writing `life →
parent_child` in config would produce identical numbers while memorising the
eval set. Density is a property of the document, measurable before any question
is asked, and it generalises to a new dense policy from any insurer.

**Do it at build time, not query time.** Everything lands in one collection, so
retrieval needs no routing logic, no second search, and no reconciling scores
across separately-built indexes — the same incomparable-scales problem that
forced rank-based fusion earlier. A router can be wrong at query time; a
build-time policy has no query-time decision to get wrong. Mixed granularities
coexist because parent expansion is opt-in per chunk: dense-document chunks
carry a `parent_id`, flat ones don't and pass through untouched.

### Results

| Metric | Phase 1 | Best (rerank) | **Chunk policy** |
|---|---|---|---|
| hit@5 | 0.694 | 0.812 | **0.871** |
| hit@1 | — | 0.482 | **0.529** |
| MRR | — | 0.616 | **0.650** |
| p50 latency | ~120 ms | 10,217 ms | **3,427 ms** |

| Policy type | rerank@20 | Chunk policy | Δ |
|---|---|---|---|
| health | 0.8605 | 0.8605 | 0.0 |
| home | 0.9048 | 0.9048 | 0.0 |
| life | 0.6190 | **0.8571** | **+23.8** |

Health and home land on their previous values to four decimal places — mixing
granularities costs the flat-chunked documents nothing. All the gain is the
outlier document, and it is larger than parent-documents alone delivered.

**+17.7 points against the Phase 1 baseline.**

### Where we are honest about it

The strategy split was chosen using the same eval set it is measured on — that
is selection on test data, and the number is optimistic. The density threshold
is validated on exactly **one** document above it. "Dense documents want small
chunks" currently rests on n=1. One more dense policy from another insurer would
test it properly.

---

## Technique 4: query rewriting — deliberately not built

Worth recording why, since it is on every RAG checklist.

Our eval set is **constructed to defeat it**. The generator deliberately
paraphrases away vocabulary overlap and rejects questions that share too many
content words with their source passage — specifically so hit@5 measures
semantic search rather than string matching. The mechanism query rewriting
exploits is therefore largely absent by construction, so a gain would be hard to
trust and a null result hard to interpret.

There is a real counter-argument. Pool recall has been 0.894 since Phase 1 and
*no technique moved it* — the same nine questions are missed by dense retrieval,
by four hybrid configurations, and by the adopted chunk policy. Query rewriting
is a recall technique and would be the first genuine attempt at those nine.

It loses to the lesson from technique 2: a recall technique feeding a
precision-bound reranker made things worse, not better. If revisited, it should
be scoped narrowly at those nine and measured on **pool recall**, because hit@5
cannot show a recall gain the reranker then fails to convert.

---

## Technique 6: does a 1M-token context window make this obsolete?

The corpus is ~124K tokens. A 1M-token window fits it whole. So the question is
not *can* you skip retrieval — it's *should* you.

Same 10 questions, same generator, the only difference being the retrieval
stage:

| Metric | Full-context stuffing | RAG | Advantage |
|---|---|---|---|
| Cited the correct page | 0.20 | **0.60** | **3×** |
| Prompt tokens/query | 120,972 | **1,646** | **73×** |
| p50 latency | 36.5 s | **14.2 s** | 2.6× |
| Cost, 10 questions | $0.0829 | **$0.0017** | **49×** |

RAG wins on accuracy, tokens, latency and cost simultaneously. There is no axis
on which stuffing trades favourably.

**Stuffing fails plausibly, not loudly.** It never once refused. Every question
got a confident answer, and 8 of 10 cited the wrong page while the right page
sat in context. For an insurance assistant, a confident wrong citation is the
worst available failure mode.

### The structural argument, which outlives the benchmark

RAG gets **citation validity for free**. Every `[p.N]` the model emits is
checked against the pages actually retrieved — deterministically, on every call,
no judge required. That is a live correctness signal in production.

Stuffing cannot have that metric *at all*. With every page in context, no
citation is ever invalid relative to what was retrieved, so verification demands
ground truth that does not exist at inference time. **The simpler-looking
architecture quietly gives up its own error detection.**

Caveats: n=10, and stuffing scored 0.30 / 0.40 / 0.20 across three identical
`temperature=0` runs — ±10 points of noise, so this supports "clearly worse",
not a precise figure. The 73× token ratio is arithmetic and needs no sample
size.

---

## Did it survive contact with generation?

Retrieval metrics are a proxy. The end-to-end run, deterministic metrics only:

| Metric | Phase 1 | Phase 2 |
|---|---|---|
| **Positives with retrieved evidence** | 59 / 85 | **74 / 85** |
| **Hallucinations (absolute)** | 20 of 26 | **10 of 11** |
| Citation validity | 0.958 | 0.962 |
| Tokens per query | 1,475 | 2,075 |

Fifteen more questions now have their answer in context, and absolute
hallucinations halved.

One number needs care. The *rate* of ungrounded answers rose from 0.769 to
0.909 — but that is 10/11 against 20/26. Retrieval now fails on 11 positives
instead of 26, so the same defect over a smaller base reads higher. **Compare
counts, not rates.**

And the defect itself is untouched: when retrieval fails, the model invents an
answer 10 times out of 11. Better retrieval reduced how often it *has* to guess;
it did nothing about the guessing. That is the confidence gate's job — and we
already know the threshold from technique 1's free by-product.

*(Judged metrics — faithfulness, answer relevancy — are omitted deliberately.
Our generator and judge both changed mid-phase when a free tier ran out, and the
judge ended up scoring its own output. Three confounds on one measurement is not
a delta, so we are not reporting one.)*

---

## What we'd tell you to take away

1. **Pick your first technique from a measurement, not a checklist.** hit@5
   0.694 against recall@50 0.953 said "ranking problem" in one line, and the
   biggest win followed directly.
2. **Deeper candidate pools can hurt.** Reranker efficiency fell 97% → 83% as we
   went from 10 candidates to 50.
3. **Hybrid search is not free.** With a fixed-size pool, every lexical
   candidate evicts a dense one. Ours evicted twice what it recovered.
4. **Check your reranker's max token length before building large chunks.** Ours
   was 512, silently truncating 2,000-character parents mid-block, and it cost
   4.7 points before we found it.
5. **Chunk size is a property of the document.** One global setting was leaving
   ~24 points on the table for the densest policy in our corpus.
6. **Long context is not a retrieval replacement** — 73× the tokens for a third
   of the citation accuracy, and it silently discards your ability to verify
   citations at all.
7. **Negative results are results.** Two of the six techniques here were
   rejected, and the reasons taught us more than the wins.

Everything above is reproducible from free, deterministic retrieval metrics that
run in 30 seconds. The total paid inference spend for this entire phase,
including the long-context experiment, was **under $0.50**.

*Next: an agentic layer — a router, a deterministic claims calculator (because
an LLM doing co-pay arithmetic in its head is a bug, not a feature), and the
confidence gate this phase produced the threshold for.*
