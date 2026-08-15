# ClaimWise — Workflow

A running index of every file in this repository and what it does.
Updated whenever a file is added or its responsibility changes.

---

## Root

### `CLAUDE.md`
The project constitution. Defines the non-negotiable workflow (Claude writes code, never executes it; every file ships with an Execution Card; work stops until real run output is pasted back), the teaching contract (Concept Brief before each module, trade-off tables before design decisions), the compute and cost constraints (no paid cloud, $30 Modal hard cap, ZeroGPU for serving only), the six-phase build plan with per-phase metrics and exit criteria, and the metrics-honesty rules. Overrides all default assistant behaviour.

### `workflow.md`
This file. A file-by-file map of the repository so the structure stays legible as it grows across six phases. Answers *"what does this file do?"* in one paragraph each.

### `decisions.md`
The engineering history: every design decision (`D-n`) with the options that were rejected and why, and every problem (`P-n`) with its symptom, diagnosis, root cause, fix and verification — organised by phase and cross-linked. Records what things cost in time, marks each item Resolved / Open-deferred / Unresolved, and keeps deliberately-deferred weaknesses visible so Phase 2 can *demonstrate* improvements against reproducible test cases rather than assert them. Also records wrong turns, including a hypothesis that neatly explained two unrelated symptoms and turned out to be wrong about one of them.

### `explaination.md`
The deep companion to this file: the **what, how and why** of every component, the technical detail behind each model, all measured results and benchmarks, and mermaid flowcharts covering the end-to-end phase pipeline, the Phase 1 build-time internals, and the query-time path. Also records defects found and fixed — the brochure-not-policy-wording corpus fault and the publisher-wide ligature corruption — because how a bug was caught is as instructive as the metric it would have spoiled. Every number in it comes from a real pasted run; unmeasured values are marked pending rather than estimated.

### `pyproject.toml`
Declares the Python project and its dependencies for uv. Configured as a **virtual project** (`package = false`), meaning uv creates and manages `.venv` and installs dependencies but does not build or install ClaimWise itself — modules run from the repo root via `python -m package.module`, so new phase packages become importable without editing this file. Currently pins Phase 1 ingestion dependencies only (pymupdf, pyyaml, tqdm); retrieval dependencies are deliberately deferred to `embed_index.py` so that ~2GB of torch isn't downloaded before a single PDF has been parsed.

### `config.yaml`
The single source of runtime truth. Holds project seed, all directory paths, and every ingestion setting (extraction mode, minimum characters per page, boilerplate-stripping thresholds, de-hyphenation and Unicode flags, output filename). Scripts read settings from here rather than hardcoding them, and CLI flags override these values at run time. Secrets never live here — they belong in `.env`.

### `.gitignore`
Keeps raw policy PDFs, processed artifacts, run logs, the virtual environment, vector-store data, model checkpoints, and `.env` out of version control. Deliberately does **not** ignore `data/eval/` — the golden evaluation set is versioned, because a metric is meaningless without the exact question set that produced it.

---

## `common/` — shared across all phases

### `common/__init__.py`
Marks `common` as a Python package. Exists so that any phase can import shared utilities without depending on another phase's package.

### `common/config.py`
Loads `config.yaml` into a plain dict and reads nested values by dotted key (`cfg_get(cfg, "ingest.extraction_mode")`). Fails loudly with a clear message if the config file is missing or isn't a YAML mapping, rather than silently falling back to defaults — a silently-defaulted config produces metrics that cannot be reproduced. Every phase reads its settings through this one module so a run's behaviour is fully described by one versioned file.

### `common/generator.py`
The swappable generator slot — every LLM call in the project goes through it, and nothing else may talk to a provider directly. Exposes `generate(prompt, system) -> GenerationResult`, returning the answer plus token counts and latency so "tokens per query" is measurable without extra plumbing. One `OpenAICompatGenerator` class covers NVIDIA NIM, OpenRouter and Hugging Face Inference Providers, because all three speak the OpenAI chat-completions protocol and differ only in base URL, key and model name; Phase 5 adds a fourth entry pointing at the fine-tuned Qwen Space without changing the interface. Temperature defaults to 0 so evaluation is reproducible — a judged metric that drifts with sampling noise cannot be attributed to a pipeline change. Includes a sliding-window client-side throttle (NIM allows ~40 req/min and a full RAGAS run is ~1,000 calls) and exponential-backoff retries for 429/5xx only, never for 400/401 where retrying just burns quota. Missing API keys fail at construction, not 400 calls into an eval run. Runnable standalone as a connectivity check.

### `.env.example`
Template for the three API keys (`NVIDIA_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`) with the URL to obtain each. Copy to `.env`, which is gitignored and never committed.

---

## `phase1_rag/` — naive RAG baseline

### `phase1_rag/__init__.py`
Marks `phase1_rag` as a Python package so its modules are runnable as `python -m phase1_rag.<module>`.

### `phase1_rag/ingest.py`
Reads every PDF in `data/raw/`, extracts text page by page, cleans it (Unicode NFKC normalisation, de-hyphenation of words broken across line breaks, removal of repeated headers/footers like "Page 12 of 88"), and writes one JSON record per page to `data/processed/pages.jsonl`. Each record keeps its 1-indexed PDF page number plus insurer and policy type parsed from the filename — the provenance that later becomes a citation. It also writes a `pages.meta.json` sidecar recording the exact settings used, so any metric traced from this corpus is reproducible.

Supports two extraction modes selectable from config: `text` (plain reading order, the honest Phase 1 baseline) and `blocks` (positioned text blocks sorted top-to-bottom then left-to-right). The mode is a switch rather than a hardcoded call specifically so Phase 2 can flip it and measure the retrieval delta attributably. Document IDs are content hashes, so re-ingesting the same PDF is idempotent and renaming a file does not create a duplicate. Failures are per-document: a broken PDF is logged and the run continues, but the process exits non-zero so no failure passes unnoticed.

### `phase1_rag/chunk.py`
Reads `data/processed/pages.jsonl` and splits each page into overlapping chunks, writing one JSON record per chunk to `data/processed/chunks.jsonl` plus a `chunks.meta.json` sidecar with the settings and size distribution. Uses LangChain's `RecursiveCharacterTextSplitter`, which tries paragraph → line → word → bare character and only descends to a harsher separator when a piece is still over budget, so breaks land on real boundaries instead of mid-sentence. Default budget is 1,000 characters (~250 tokens) with 150 characters of overlap, both driven from `config.yaml`.

Chunks never span a page break: each page is split independently so every chunk inherits one exact page number, which is what makes a citation verifiable. The cost is that a clause continuing onto the next page gets divided — Phase 2's parent-document retrieval is the proper fix, not a larger chunk size. Chunk IDs are deterministic (`{doc_id}_p{page}_c{index}`), so re-running produces identical IDs and the vector store can be upserted rather than rebuilt. Reports how many chunks exceed bge-small's ~2,000-character input budget so silent truncation at embedding time is impossible.

### `phase1_rag/embed_index.py`
Reads `data/processed/chunks.jsonl`, embeds every chunk on CPU with a sentence-transformers BGE model, and upserts the vectors into an embedded on-disk Qdrant collection under `qdrant_storage/`, writing an `index_{model}.meta.json` sidecar with timings and settings. Collection names encode the embedding model (`claimwise__baai_bge_small_en_v1_5`), so two models can be indexed side by side and compared on the same golden questions instead of overwriting each other. Point IDs are `uuid5` of the chunk ID, making re-indexing an idempotent upsert rather than a duplication.

Every point carries `user_id` in its payload and every search filters on it — the security boundary, wired in from day one while there is still only one user, because a filter retrofitted later gets forgotten in exactly one code path. Filtering happens inside the index traversal rather than after it, so results from another user are never scored at all. Note that the BGE query-instruction prefix is deliberately **not** applied here: those models are trained with the prefix on queries only and documents embedded bare, so the prefix lives in config and is applied by `rag_chain.py`. Each run ends with a filtered smoke query printing real top-3 hits with scores and page numbers, so an index that builds cleanly but returns nothing cannot pass silently.

### `phase1_rag/rag_chain.py`
The end-to-end baseline: embeds the question (with the BGE instruction prefix that documents deliberately don't get), searches Qdrant filtered by `user_id` plus optional insurer/policy-type, assembles one prompt from the top-5 passages, generates through the swappable generator, and verifies the result. Answers arrive as a `RagAnswer` carrying the retrieved chunks, cited pages, timings and token counts — everything the eval harness needs, with no separate instrumentation pass.

Grounding is an explicit contract, not a hope: passages are labelled with their page numbers in the header so citing is a copy rather than a recall task; the system prompt forbids outside knowledge; refusal has fixed wording so refusals are countable without an LLM judge; and arithmetic is banned outright, since Phase 3 adds a deterministic calculator and an LLM doing co-pay maths in its head is a bug. When retrieval returns nothing the module refuses without calling the model at all, saving a request. Every `[p.N]` the model emits is checked against the pages actually retrieved, and a fabricated citation exits non-zero — a free, deterministic signal on every call, and the same one Phase 4.5's GRPO reward function will optimise. The system prompt is a module constant because Phase 4 must generate RAFT training data in exactly this format.

### `phase1_rag/build_eval_set.py`
Generates golden Q&A pairs from sampled chunks and writes `data/eval/golden.jsonl` plus a human-readable `review.md` that places each question beside the exact chunk it came from, so verification never requires cross-referencing two files. Sampling is stratified by policy type and spread across pages so the set covers the corpus rather than clustering on whichever pages happen to be long, and it's seeded for reproducibility.

Its real job is avoiding the circularity trap: an LLM asked to write a question about a passage inherits that passage's vocabulary, producing questions retrieval finds trivially and an inflated hit@5 that measures string matching rather than semantic search. Three defences — a persona-and-paraphrase generation prompt, a deterministic filter rejecting questions whose content words overlap the source chunk beyond a threshold, and hand-seeded negatives drawn from out-of-scope domains (motor, travel, marine, crop) rather than generated, since an LLM asked for an unanswerable question sometimes writes an answerable one. Rejection reasons are counted and reported so leakage is visible rather than assumed. Every item carries `ground_truth_pages`, the field that makes hit@5, MRR and context recall pure-Python comparisons instead of paid API runs. Items start `verified: false`; a human flips them after review.

---

## `phase2_advanced/` — advanced retrieval

### `phase2_advanced/rerank.py`
Cross-encoder reranking, Phase 2's first technique. Wraps `BAAI/bge-reranker-base` to score every (question, chunk) pair and reorder the candidate list. Where a bi-encoder embeds question and chunk *independently* — fast, because chunk vectors are computed once at upload, but the model never sees the two together — a cross-encoder concatenates them and runs full attention across the pair, reading the question while reading the passage. That's roughly 100× more expensive per pair and cannot precompute anything, which is exactly why it runs as a second stage over ~30 candidates rather than all 653.

Chosen as the first technique from measurement rather than intuition: Phase 1 recorded hit@5 0.694 against recall@50 0.953, so for 95% of questions the correct page is already retrieved and merely ranked too low. Reranking cannot invent results, so its ceiling *is* the recall of the candidate set — which makes candidate depth a real tunable, swept empirically since evaluation is free. Reranking reorders the whole candidate list rather than truncating it, so hit@k stays computable at every k and before/after comparison is exact. Runnable standalone to print the before/after ordering for one question with rank-movement arrows, so the effect is visible before committing to a full evaluation.

### `phase2_advanced/hybrid.py`
Hybrid retrieval — Phase 2's second technique. Builds an in-memory BM25 index over `chunks.jsonl`, runs it alongside dense search, and fuses the two rankings with Reciprocal Rank Fusion. Exists because reranking hit a structural ceiling: a cross-encoder reorders the candidate pool but cannot add to it, so the nine remaining misses at depth 20 were invisible to it. Raising pool recall compounds — better candidates in, better reranked output out.

Dense and lexical retrieval fail in opposite directions: embeddings capture meaning but blur exact tokens (P-11, where "co-payment" appeared verbatim in the target chunk and dense search still missed it), while BM25 nails rare exact terms and understands nothing. Fusion is by **rank, not score**, because cosine lives in ~0.4–0.9 while BM25 is unbounded and corpus-dependent — there is no principled shared scale, so RRF discards magnitudes and sums `1/(k+rank)` across retrievers. Runnable standalone to print dense, lexical and fused rankings side by side with an overlap count, so it's immediately visible whether fusion contributes anything or is pure overhead. Note the security caveat documented in the module: dense retrieval filters `user_id` inside Qdrant's traversal, but this BM25 index has no such boundary — a production deployment must build it per user or filter by owned `doc_id` via `allowed_doc_ids`.

**Measured and rejected — `hybrid.enabled` is `false` (D-17).** Kept in the repo because it is a working implementation and the evidence behind the decision, not dead code. Fusion turned out to be a *displacement* trade rather than an addition: the candidate pool is fixed-size, so every lexical candidate admitted evicts a dense one, and across four configurations BM25 recovered 1–3 items while evicting 3–6. Pool recall fell 0.894 → 0.859 at depth 20 and end-to-end hit@5 was an exact wash. The root cause is that the *reranker's precision*, not pool recall, is the binding constraint — fusion buys recall only by widening the pool, which is precisely what degrades the cross-encoder. One finding survived: the life policy gains +9.5 points from lexical matching, which is a routing question carried to Phase 3, not a global fusion setting.

### `phase2_advanced/parent_docs.py`
Parent-document retrieval — Phase 2's third technique. Splits every page into 2,000-character **parents**, then each parent into 400-character **children**, writing `parents.jsonl` and `children.jsonl`. Only children are embedded; when a child is retrieved, the larger parent block it came from is swapped in before reranking and generation. Chunk size is one knob serving two masters that want opposite things — retrieval wants small chunks so each vector points at one topic, generation wants large ones because "subject to the sub-limits in Table B" is worthless without Table B. Phase 1 split the difference at 1,000 characters and served neither.

The nesting order is load-bearing: children are cut *within* a parent's boundaries, so every child has exactly one parent containing it whole. Splitting both tiers straight from the page would leave children straddling parents, forcing the expansion step to pick one and silently drop context. Parent overlap is zero while child overlap is 50 — overlap stops a clause being cut mid-sentence, which matters for children, but parents enter the prompt whole and overlapping them would feed the generator the same sentences twice.

`ParentStore.expand()` deduplicates: several children often share one parent, which is the technique working rather than a fault — agreement between neighbouring children is evidence the parent is relevant — but the parent is emitted once, at the position its best-ranked child earned. A child whose parent is missing passes through unchanged rather than being dropped, because losing a correct retrieval to a stale parent file would be an invisible recall regression. Children index into their own collection (`claimwise_pd`) so the Phase 1 collection survives and the baseline stays reproducible.

Chosen as technique 3 because D-17 established the reranker's precision, not pool recall, as the binding constraint: this attacks that from the opposite side, making each candidate sharper going in and more complete coming out without touching pool size. It is also a falsifiable test of P-17 — if the life policy's 0.619 really is caused by 11,639 characters per page packing too many clauses into each chunk, 400-character children should sharpen that document specifically. If life does not move, the hypothesis is wrong and P-9's column scrambling is the likelier culprit.

### `phase2_advanced/chunk_policy.py`
Per-document chunking policy — Phase 2's fifth technique. Measures each document's characters-per-page at build time and gives dense documents the 400/2000 parent-child treatment while everything else keeps Phase 1's flat 1,000-character chunks, writing one mixed corpus indexed into one collection. Exists because three independent measurements said the same thing: hybrid gave the life policy +9.5 while costing health 4.7 (D-17), parent-documents gave life +14.3 while costing home 14.3 (D-18), and P-17 predicted both from one observation — life packs 11,639 characters into a page against 2,602–5,935 for the rest. Optimal chunk size is a property of the document, not of the corpus.

Selects on measured density rather than document name deliberately. Writing `life → parent_child` into config would produce identical numbers while memorising the eval set; density is measurable at ingest before any question is asked and generalises to a new dense policy from any insurer. The threshold lives in config because it is the falsifiable part of the claim.

A build-time policy rather than a query-time router: everything lands in one collection, so retrieval needs no routing logic, no second search, and no reconciling scores across separately-built indexes — the incomparable-scales problem that forced rank-based fusion in D-17. A router can be wrong at query time; a build-time policy has no query-time decision to get wrong. Mixing granularities works because expansion is opt-in per chunk — dense documents produce children carrying `parent_id`, sparse ones produce flat chunks with none, and `ParentStore.expand()` passes those through untouched. Exits non-zero if every document selects the same strategy, since the policy only earns its complexity when documents actually differ. Also applies the `min_chunk_chars` filter that `parent_docs.py` omits, which is what produced 6-character children in that module's first build.

### `phase2_advanced/full_context.py`
The Phase 2 blog experiment — answers the golden questions with the entire corpus in one prompt, no retrieval, no chunking, no reranking. Tests the claim that long context windows make RAG obsolete: at 491,986 characters (~123K tokens) the whole corpus genuinely fits a 262K window, so the question is not *can* you but *should* you. Only the retrieval stage differs from the RAG path — same system prompt, same citation contract — so the comparison isolates it.

Reports four axes because accuracy alone hides the trade: citation correctness against ground truth, tokens per query (~123,000 against RAG's ~1,475, roughly 80× on every question forever), latency, and cost. Deliberately surfaces the approach's one real advantage rather than burying it — stuffing cannot suffer a retrieval miss, so the nine questions Phase 2 never retrieves correctly are all *visible* to it, and if it answers them that belongs in the write-up. Note it cannot reuse `verify_citations`: with every page in context no citation is ever "fabricated" relative to a retrieved set, which is itself a finding — the free citation-validity signal the RAG path gets for nothing is unavailable to a stuffing architecture.

Prints a cost estimate before spending anything and refuses to run more than ten questions without `--yes`, the same discipline CLAUDE.md imposes on Modal training scripts, because the failure mode is a mistyped `--limit` spending real money before a single line is printed.

---

## `evals/` — evaluation harnesses

### `evals/retrieval_metrics.py`
The free evaluation tier: runs every golden question through retrieval only — no generation, **zero LLM calls** — and reports hit@k, MRR, page recall, and latency percentiles, writing both a summary JSON and a per-item JSONL to `evals/results/`. Because it costs nothing and finishes in seconds, it is the harness Phase 2 runs after every single technique; `--baseline <previous.json>` prints the deltas that Phase 2's attribution table is built from.

Reports hit at two strictnesses. **Page-level** counts any chunk from the correct page, matching how citations work; **exact-chunk** demands the specific source chunk. The gap between them distinguishes "found the right region, wrong passage" (a chunk-size problem) from "didn't find it at all" (an embedding problem). MRR is reported alongside hit@k because a reranker's whole job is moving correct pages *upward*, an improvement hit@5 alone would show as zero. Negatives have no page to find, so it reports their mean top-1 similarity instead — a high score there means refusal rests purely on the model's judgment rather than being backed by weak retrieval. Defaults to verified items only; `--allow-unverified` produces provisional numbers and stamps them as such in the output file so they can't be mistaken for a baseline.

### `evals/run_ragas.py`
The paid evaluation tier: answers every golden question through the full RAG chain, then scores the answers. Reports four metrics for **free** — citation validity (every `[p.N]` checked against pages actually retrieved), citation coverage, false refusal rate on positives, and refusal accuracy on negatives — plus latency, tokens per query and estimated spend. On top of that it runs RAGAS `faithfulness` and `answer_relevancy` through a judge model.

Honours `rerank.enabled` so it measures the pipeline that is actually configured, with `--no-rerank` and `--rerank-depth` to override; the resolved pipeline is printed as the first line of output and stored in the results file. It did neither until 2026-08-14, silently evaluating dense top-5 while config said otherwise — which is how the Phase 1 baseline came to be recorded with no indication of which pipeline produced it (P-18). Aggregation reads `EvaluationResult.scores` rather than coercing the object with `dict()`, and reports a NaN count beside every judged metric, because RAGAS returns NaN when a judge reply won't parse and a mean over the survivors is a different claim from a mean over the set.

Deliberately runs only those two RAGAS metrics. `context_precision` costs one LLM call *per retrieved chunk*, which is what makes a full RAGAS suite ~1,000 calls per 100 questions — but context recall and precision are already computed exactly and for free in `retrieval_metrics.py` from ground-truth pages, so paying a judge to estimate a known number is waste. That cuts the run to roughly 4 calls per question. `--skip-ragas` drops to 1 call per question for the free metrics alone, and a RAGAS failure is caught and reported without discarding those metrics. Only non-refusal positives are scored, since faithfulness of a refusal is meaningless.

---

## `metrics/`

### `metrics/METRICS.md`
The single source of truth for every number in the project. Pre-structured with tables for the Phase 1 corpus, chunking, and RAGAS baseline; the Phase 2 per-technique delta table; and the Modal credit ledger tracking the $30 hard cap (with $5 permanently reserved for Phase 5). Every row records date, git commit hash, config used, and the number. Nothing is written here that wasn't computed from a real run's pasted output.

---

## `data/`

### `data/raw/`
Source policy PDFs. Never committed. Files are expected to be named `insurer__policytype__label.pdf` (double underscores) so that `ingest.py` can derive the metadata that Phase 2 filters on and Phase 3 compares across.

### `data/processed/`
Derived artifacts from ingestion and chunking (`pages.jsonl`, `pages.meta.json`, and later chunk files). Never committed — fully reproducible from `data/raw/` plus `config.yaml`.

### `data/eval/`
The golden question-and-answer evaluation set. **Versioned**, unlike the rest of `data/`, because every RAGAS number is only interpretable against the exact question set that produced it.

---

## `design/` — frontend reference (consumed in Phase 6)

### `design/README.md`
The design handoff document. Specifies the "Industry" design system — a technical blueprint/wireframe aesthetic — including the non-negotiable visual rules (square corners everywhere, hairline-bordered transparent panels instead of filled cards, crosshair registration marks on every framed object, exactly one solid element per screen in steel `#5980a6`), the full colour token set for light and dark themes, the Barlow / Barlow Condensed type scale, and pixel-level layout specs for all six screens (Landing, Chat light, Chat dark, mascot states, Empty/Error, Mobile). Also documents interaction behaviour and the frontend state model.

### `design/ClaimWise UI.dc.html`
The visual prototype: a single canvas holding all six artboards side by side, labelled 01–06. A design reference to be recreated in Next.js, not production code to copy — its `<x-dc>` and `<helmet>` wrapper tags are ignored, but its markup, inline styles, and measurements are the source of truth for structure and exact values.

### `design/reference/industry-styles.css`
The complete Industry design-system stylesheet: colour ramps generated in OKLCH, spacing and elevation scales, and every component class (`.blueprint`, `.btn`, `.card`, `.tag`, `.input`, `.nav`, `.table`, `.dialog`, `.seg`, `.radio`, `.duotone`). Structured in two layers — base component rules first, then an override block at the bottom that flattens everything to square corners and transparent fills. Port this first; that override layer carries the system's intent and must survive the port.

### `design/reference/tokens.css`
The design tokens alone — light-theme `:root` variables plus the `[data-theme="dark"]` overrides, base typography, focus-ring and selection styles, and the `.blueprint` crosshair frame CSS. Ready to paste into `app/globals.css`.

### `design/reference/Wise.tsx`
The Wise robot mascot as a drop-in React component. Pure inline SVG with no image assets, exposing `state` (`idle` | `thinking` | `done` | `error`), `size`, `stroke`, and a `compact` head-only variant for small avatars and the nav icon. Reused everywhere in the UI at different sizes. Its state machine is what will consume the streaming agent-status events from the Phase 5 backend.

---

## Not yet written

Listed for orientation only — these files do not exist yet.

| File | Planned role |
|---|---|
| `metrics/failure_analysis_p1.md` | Phase 1 exit criterion: the 10 worst failures, analysed. Still outstanding. |
| `phase2_advanced/parent_docs.py` | Technique 3: parent-document / small-to-big retrieval — retrieve on small chunks, hand the reranker and the generator the larger surrounding block |
| `phase2_advanced/query_rewrite.py` | Technique 4: query rewriting / multi-query |
| `docs/blog/phase2.md` | Phase 2 write-up, including the hybrid negative result |
