# ClaimWise — Explanation

The **what, how and why** of every component: the technical detail behind each
file and model, the measured results, and the end-to-end picture of how a PDF
becomes a cited answer.

`workflow.md` answers *"what does this file do?"* in one paragraph.
This document answers *"why is it built that way, and what did it measure?"*

> **Metrics honesty.** Every number below was computed from a real run.
> Nothing here is estimated, extrapolated, or aspirational. Values that have
> not been measured yet are marked **pending**, never guessed.

---

## 1. The whole system in one picture

```mermaid
flowchart LR
    P1["<b>Phase 1</b><br/>Naive RAG<br/>baseline numbers"]
    P2["<b>Phase 2</b><br/>Hybrid search<br/>+ reranking"]
    P3["<b>Phase 3</b><br/>LangGraph agents<br/>+ deterministic math"]
    P4["<b>Phase 4</b><br/>RAFT QLoRA<br/>on Modal"]
    P45["<b>Phase 4.5</b><br/>DPO then GRPO<br/><i>credit-gated</i>"]
    P5["<b>Phase 5</b><br/>ZeroGPU Space<br/>+ SSE streaming"]
    P6["<b>Phase 6</b><br/>Next.js frontend<br/>+ Wise mascot"]

    P1 -->|"context recall<br/>is the gate"| P2
    P2 -->|"best pipeline<br/>becomes a tool"| P3
    P3 -->|"failure logs become<br/>training data"| P4
    P4 -->|"only if ≥$15<br/>credits remain"| P45
    P45 --> P5
    P4 -.->|"if 4.5 is skipped"| P5
    P5 -->|"must work via curl<br/>before any UI"| P6
```

The ordering is not arbitrary. **Retrieval quality caps everything**: if the
right chunk is never retrieved, no amount of fine-tuning can recover the answer,
because the model simply never sees the fact. That is why Phases 1–2 obsess over
context recall before a single GPU dollar is spent.

### The division of labour

This is the single most important idea in the project, and the one most
commonly got wrong:

| | Supplies | How it reaches the model | Changes when |
|---|---|---|---|
| **RAG** | *Knowledge* — the clauses in **your** policy | Injected into the prompt at query time | You upload a document |
| **Fine-tuning** | *Skill* — cite pages, stay grounded, refuse to guess | Baked into the weights | We retrain |

Fine-tuning never memorises policy content. If it did, every new policy would
require retraining, and the model would confidently answer from the wrong
insurer's terms.

---

## 2. Phase 1 internals — the build-time path

This runs **once per document**, on CPU, and never involves an LLM. The document
is not being *read* here; it is being made *searchable*.

```mermaid
flowchart TB
    RAW[("data/raw/<br/>insurer__type__label.pdf")]

    subgraph ING["ingest.py — parse & clean"]
        direction TB
        I1["PyMuPDF get_text per page"]
        I2["repair f-ligatures<br/>beneĤt → benefit"]
        I3["NFKC normalise"]
        I4["de-hyphenate<br/>hospi-⏎talisation"]
        I5["strip repeated<br/>headers/footers"]
        I6["drop pages &lt; 40 chars"]
        I1 --> I2 --> I3 --> I4 --> I5 --> I6
    end

    subgraph CHK["chunk.py — split"]
        direction TB
        C1["recursive split<br/>¶ → line → word → char"]
        C2["1000 chars, 150 overlap"]
        C3["never cross a page break"]
        C1 --> C2 --> C3
    end

    subgraph EMB["embed_index.py — vectorise"]
        direction TB
        E1["bge-small-en-v1.5<br/>384-dim, CPU"]
        E2["L2-normalise"]
        E3["uuid5 point IDs<br/>→ idempotent upsert"]
        E1 --> E2 --> E3
    end

    RAW --> ING
    ING --> PG[("pages.jsonl<br/>1 record per page<br/>+ page number")]
    PG --> CHK
    CHK --> CK[("chunks.jsonl<br/>652 chunks")]
    CK --> EMB
    EMB --> QD[("Qdrant collection<br/>claimwise__baai_bge_small_en_v1_5")]

    style RAW fill:#eef6ff,stroke:#5980a6
    style QD fill:#eef6ff,stroke:#5980a6
```

**The page number is the thread that runs through all of it.** It is attached at
parse time, carried through chunking (chunks never span a page break precisely
so this stays exact), stored in the Qdrant payload, and eventually rendered as
a citation chip in the UI. In a regulated domain an answer without a verifiable
source is worthless — "you get ₹2L back" is unusable unless a human can turn to
page 34 and check.

---

## 3. Query-time path

Runs on **every question**, target 2–5 seconds. Dashed nodes are not built yet.

```mermaid
flowchart TB
    Q["User question<br/>+ optional bill photo"]
    Q --> EQ["embed query<br/><i>with</i> BGE instruction prefix"]
    EQ --> SRCH["Qdrant search<br/><b>filtered by user_id</b>"]
    SRCH --> TOP["top-k chunks<br/>+ page metadata"]
    TOP --> RR["rerank to top-5<br/><i>Phase 2</i>"]
    RR --> TOOLS["deterministic tools<br/>co-pay / sub-limit math<br/><i>Phase 3</i>"]
    TOOLS --> PROMPT["assemble ONE prompt<br/>question + chunks + tool output + image"]
    PROMPT --> GEN["generate()<br/><i>swappable slot</i>"]
    GEN --> ANS["streamed answer<br/>with page citations"]

    style SRCH fill:#eef6ff,stroke:#5980a6
    style GEN fill:#eef6ff,stroke:#5980a6
    style RR stroke-dasharray: 4 4
    style TOOLS stroke-dasharray: 4 4
```

Two design commitments visible here:

**`user_id` filtering happens inside the search, not after it.** Retrieving the
global top-5 and then discarding other users' chunks has two failure modes: you
often end up with fewer than 5 results, and the day one code path forgets the
post-filter, a customer sees someone else's policy. Qdrant applies the filter
during index traversal, so a chunk you don't own is never scored at all. It is
wired in from day one, while there is still exactly one user, because a
boundary retrofitted later gets missed in exactly one place.

**`generate(prompt) → answer` is a swappable slot.** Phases 1–3 fill it with a
free OpenRouter model; Phase 5 swaps in the fine-tuned Qwen endpoint. Nothing
upstream changes, which is what makes the Phase 4 benchmark meaningful — the
same retrieval pipeline feeds every generator being compared, so differences are
attributable to the generator alone.

---

## 4. File-by-file — what, how, why

### `common/config.py`

**What.** Loads `config.yaml`; reads nested values by dotted key.

**How.** `yaml.safe_load` into a dict, plus `cfg_get(cfg, "ingest.extraction_mode")`
walking dot-separated segments.

**Why.** Every script's behaviour must be fully described by one versioned file,
so a metric recorded three weeks ago can be reproduced. It raises rather than
defaulting when the file is missing: a silently-defaulted config produces
numbers nobody can reproduce, which is worse than a crash.

---

### `phase1_rag/ingest.py`

**What.** PDFs → one cleaned JSON record per page.

**How.** PyMuPDF extracts text per page in one of two modes. Cleaning runs in a
fixed order — ligature repair, NFKC normalisation, de-hyphenation, boilerplate
removal, whitespace collapse — because each step can create the pattern the next
one looks for. Boilerplate detection counts lines appearing in the first or last
3 lines of ≥60% of pages, with digits masked to `#` so "Page 12 of 88" and
"Page 13 of 88" collapse to one recurring line. Document IDs are SHA-256 content
hashes truncated to 12 characters.

**Why each decision:**

- **Page-scoped records, not one big string.** Provenance is the product. A
  string loses the page number permanently and no downstream step can recover it.
- **Content-hash document IDs.** Renaming a file must not create a duplicate
  document in the vector store; re-ingesting must be idempotent.
- **Boilerplate stripping.** Policy PDFs stamp the insurer name, UIN and page
  number on every page. Left in, that text dominates the embedding of every
  short chunk, making unrelated pages look similar to one another — it actively
  destroys retrieval precision.
- **`extraction_mode` is a switch, not a hardcoded call.** Table-aware parsing
  is a *Phase 2 experiment*. Baking it in now would forfeit the ability to
  report "table-aware parsing bought us N points of context recall".
- **Filename-derived metadata.** `insurer__type__label.pdf` gives us the
  metadata Phase 2 filters on and Phase 3 compares across, with no sidecar
  manifest to keep in sync. Files that don't match still ingest, with a warning
  — one badly named file must not block a corpus.

---

### `phase1_rag/chunk.py`

**What.** Pages → overlapping, page-scoped chunks.

**How.** LangChain's `RecursiveCharacterTextSplitter` tries separators in order
— paragraph, line, word, bare character — descending only when a piece is still
over the 1,000-character budget. 150 characters of overlap. Chunk IDs are
`{doc_id}_p{page}_c{index}`.

**Why.** An embedding is a *single vector*, and a vector can only carry so much
meaning. Our pages average several thousand characters; embed a whole one and
the "waiting period for pre-existing diseases" signal is averaged together with
room rent, co-pay and AYUSH text until the page matches none of those queries
strongly. Chunking exists to keep roughly one idea per vector.

- **Why 1,000 characters.** ≈250 tokens. bge-small truncates at 512 tokens
  (~2,000 chars), so this leaves headroom — nothing is silently cut at embedding
  time. The run reports how many chunks exceed that budget; it must be zero.
- **Why 150 overlap (15%).** A clause straddling a boundary survives intact in
  one of the two neighbouring chunks. Without overlap, the sentence that
  actually answers the question can be split in half and match nothing.
- **Why chunks never cross a page break.** So every chunk inherits one exact
  page number and citations stay verifiable. The cost — a clause continuing onto
  the next page gets divided — is real, and the correct fix is Phase 2's
  parent-document retrieval, not a bigger chunk size.
- **Why recursive rather than fixed slicing.** None of Phase 2's planned
  techniques is "recursive vs fixed splitting", so using the standard splitter
  costs us no future experiment. A baseline should be what any competent team
  would start from, not a deliberately weakened strawman.

---

### `phase1_rag/embed_index.py`

**What.** Chunks → vectors → a filterable Qdrant collection.

**How.** sentence-transformers encodes chunks in batches of 32 on CPU with L2
normalisation; qdrant-client upserts them into an embedded on-disk collection
whose name encodes the embedding model. Point IDs are `uuid5(chunk_id)`.
The run ends with a filtered search against the index it just built.

**Why:**

- **Collection name encodes the model.** `claimwise__baai_bge_small_en_v1_5`.
  Two embedding models can then be indexed side by side and compared on the
  same golden questions instead of overwriting one another.
- **`uuid5` point IDs.** Qdrant requires integer or UUID IDs, and our chunk IDs
  are strings. A deterministic UUID makes re-indexing an upsert rather than a
  duplication.
- **The BGE query prefix is deliberately absent here.** BGE v1.5 models are
  trained with an instruction prefix on the *query* side only, documents bare.
  Applying it to both — or neither — measurably costs recall. It lives in
  config and is applied by `rag_chain.py`.
- **Self-verifying runs.** An index that builds cleanly but returns nothing
  useful is the most expensive kind of silent failure, because every downstream
  metric inherits it. Printing real top-3 hits with scores and page numbers
  makes that impossible to miss — and in fact this is exactly how the ligature
  defect in §7 was caught.

---

## 5. Models and libraries

### BAAI/bge-small-en-v1.5 — embedding model *(in use)*

| Property | Value |
|---|---|
| Architecture | BERT-style bi-encoder |
| Parameters | ~33M |
| Output dimensions | **384** |
| Max input | 512 tokens (~2,000 chars) |
| Similarity | Cosine, on L2-normalised vectors |
| Runs on | CPU |
| Measured throughput | **17.6 chunks/s** (batch 32, 652 chunks) |

**How it works.** A bi-encoder embeds queries and documents *independently* into
the same vector space; similarity is then a dot product. That independence is
what makes search fast — every document vector is computed once at upload time,
and a query only needs one forward pass plus a nearest-neighbour lookup. The
trade-off is that the model never sees the query and document together, so it
cannot reason about their interaction. That is precisely what a cross-encoder
reranker fixes in Phase 2, at ~100× the per-pair cost, which is why it is
applied to ~50 candidates rather than 652.

**Why this one.** It is CLAUDE.md's stated default and keeps Phase 5's on-Space
CPU ingestion fast. bge-base is ~1.5 MTEB retrieval points better for roughly 3×
the CPU time — a smaller gain than reranking should buy, so spending the budget
here first would be optimising the wrong stage. At 652 chunks a full re-index
costs about a minute, making this cheap to revisit; the A/B against bge-base is
scheduled for when the golden eval set exists and the comparison can be made on
real questions.

**The instruction prefix.** For retrieval, BGE v1.5 expects queries to be
prefixed with `"Represent this sentence for searching relevant passages: "` and
documents to be embedded bare. This asymmetry is easy to get wrong in a way that
produces no error — just quietly worse recall.

### Qdrant — vector database *(in use, embedded mode)*

Running in-process against `qdrant_storage/`, no Docker. The client API is
identical to server mode, so Phase 5's move to a hosted instance is a config
change rather than a rewrite. Single-process only, which is the one real
limitation: two terminals cannot open the store simultaneously.

**Why a vector DB at all, at only 652 chunks?** Brute-force cosine over 652
vectors would genuinely be fine. What Qdrant buys is persistence, sub-linear
search as the corpus grows, and — the decisive one — **filters applied during
index traversal**. See §3.

### Planned models *(not yet in use)*

| Model | Role | Phase |
|---|---|---|
| `bge-reranker` (cross-encoder) | Rerank ~50 candidates → top 5 | 2 |
| OpenRouter free model | The `generate()` slot during development | 1–3 |
| Qwen3.5-4B | Fine-tuned generator; natively multimodal for bill/claim-form photos | 4 |
| Qwen3-4B *(fallback)* | Text-only, if LoRA tooling doesn't support the hybrid Gated DeltaNet + MoE architecture | 4 |

Framework mapping, which must not drift: **Unsloth/TRL = training only, on
Modal**. **transformers + `@spaces.GPU` + `TextIteratorStreamer` = serving, on
ZeroGPU**. **Never vLLM** — it allocates GPU at startup, which is fundamentally
incompatible with ZeroGPU's serverless attach/release model.

---

## 6. Results and benchmarking

### 6.1 Corpus

Run A was the original corpus. The Star Health file in it turned out to be a
**product brochure**, not a policy wording — 17,078 characters across 12 pages,
against 174,999 from the SBI health policy, and the document itself pointed
readers elsewhere for detail. It was replaced.

| | Run A *(2026-08-10)* | Run B *(2026-08-10, corpus fixed)* |
|---|---:|---:|
| Documents | 4 | 4 |
| Pages kept | 67 | **102** |
| Total characters | 385,892 | not captured |
| Mean chars/page | 5,760 | not captured |

Per-document page counts after the fix — Star Health went from 12 pages to 47,
and its density from 1,423 chars/page to roughly 2,900:

| Document | Type | Pages |
|---|---|---:|
| `starhealth__health__comprehensive` | health | 47 |
| `sbigeneral__health__alpha` | health | 30 |
| `sbigeneral__home__house-insurance` | home | 17 |
| `iciciprulife__life__prusmart` | life | 8 |

### 6.2 Chunking — 1000 chars / 150 overlap

| Metric | Value |
|---|---:|
| Pages in | 102 |
| Chunks out | **652** |
| Chars per chunk — min / median / mean / max | 52 / 945 / 819 / 1000 |
| Chunks over the 2,000-char embedding budget | **0** |

`max = 1000` confirms the splitter honours its budget; `0` over budget confirms
nothing will be silently truncated at embedding time. Mean below median is the
expected left tail of page-end fragments.

| Document | Chunks | Pages | Mean chars |
|---|---:|---:|---:|
| `sbigeneral__health__alpha` | 229 | 30 | 814 |
| `starhealth__health__comprehensive` | 161 | 47 | 854 |
| `sbigeneral__home__house-insurance` | 134 | 17 | 828 |
| `iciciprulife__life__prusmart` | 128 | 8 | 776 |

### 6.3 Index build — bge-small-en-v1.5

| Metric | Value |
|---|---:|
| Vector dimensions | 384 |
| Chunks indexed | 652 |
| Points in store | 652 |
| Embed time | 37.02 s |
| **Throughput** | **17.6 chunks/s** |
| Upsert time | 6.47 s |

`points == chunks` confirms the deterministic IDs are not colliding.

### 6.4 First retrieval evidence

Query: *"What is the waiting period for pre-existing diseases?"*

| Rank | Score | Page | Insurer | Retrieved |
|---:|---:|---:|---|---|
| 1 | 0.8102 | 30 | starhealth | Optional Cover — Buy Back of Pre-Existing Disease Waiting Period |
| 2 | 0.7795 | 31 | starhealth | Portability, waiting period reduced by prior coverage; 36 months |
| 3 | 0.7517 | 19 | sbigeneral | Changing the 24-month PED waiting period, Section 12.1 |

All three are genuinely on-topic, from both health insurers, with plausible page
numbers, at healthy cosine scores. This is the first real evidence that semantic
retrieval works on this corpus.

> **All numbers in §6.1–6.4 predate the ligature fix in §7** and will be
> superseded by the next full run.

### 6.5 Pending

| Metric | Blocked on |
|---|---|
| Faithfulness, answer relevancy, context recall, context precision | golden eval set + `run_ragas.py` |
| hit@5 | golden eval set |
| p50 / p95 latency, tokens per query | `rag_chain.py` |
| bge-small vs bge-base retrieval quality | golden eval set |

---

## 7. Defects found and fixed

Recording these matters as much as recording metrics. Both were caught by
*inspecting real output*, not by a test suite.

### 7.1 A brochure masquerading as a policy wording

**Symptom.** Star Health yielded 17,078 characters from a 3.86 MB file — one
tenth the text of a comparable policy from a file one fifth the size.

**Cause.** The downloaded PDF was a glossy product brochure. Most of those
megabytes were images.

**Why it mattered.** Questions would have been evaluated against a document that
does not contain their answers. Context recall would have looked like a
retrieval problem when it was a corpus problem, and Phase 2 would have been
spent optimising against a hole.

**Fix.** Replaced with the real policy wording: 12 pages → 47.

### 7.2 Mis-decoded f-ligatures across an entire publisher

**Symptom.** Retrieved text read `beneĤt`, `speciĤed`, `ĥoater`, `OĦce`.

**Cause.** Both SBI PDFs embed a font subset whose f-ligature glyphs decode into
the Latin Extended-A block instead of real letters:

| Corrupt | Real | Seen as |
|---|---|---|
| `ģ` U+0123 | `ff` | staģ, oģer, suģering |
| `Ĥ` U+0124 | `fi` | beneĤt, speciĤed, qualiĤed, beneĤciary |
| `ĥ` U+0125 | `fl` | ĥoater, inĥicted, Reĥux |
| `Ħ` U+0126 | `ffi` | OĦce |

**Scope.** 47 of 102 pages — **100% of both SBI documents** (30/30 and 17/17),
**0%** of Star Health and ICICI.

**Why NFKC didn't catch it.** `Ĥ` is a legitimate Unicode character. Nothing
about the text is malformed; it is simply *wrong*. No normaliser can know that.

**Why it mattered.** The corrupted words are core insurance vocabulary —
*benefit, benefits, beneficiary, specified, defined, qualified, floater,
certified, office*. Each becomes an out-of-vocabulary token, so a chunk about
floater benefits stops matching a query about floater benefits. Worse, the
damage was **entirely one-sided**: every SBI retrieval score was depressed
relative to Star Health. Any per-insurer comparison would have been measuring
font encoding rather than retrieval quality, and the conclusion would have been
confidently wrong.

**Fix.** A character-repair map in `ingest.py`, applied before de-hyphenation.
Ingestion now also warns when any Latin Extended-A character *survives* repair,
so a publisher using an unknown mapping surfaces immediately instead of quietly
poisoning the index.

**Lesson.** The smoke query in `embed_index.py` paid for itself on its first
run. A pipeline that reports only counts and timings would have shown four
green stages and a silently damaged index.

---

## 8. Open decisions

| Decision | Status |
|---|---|
| Extraction mode `text` vs `blocks`/markdown | `text` now; `blocks` is a Phase 2 experiment. Column scrambling is already documented — headings arrive *after* the paragraphs they label |
| bge-small vs bge-base | small indexed; A/B deferred until the golden eval set exists |
| Eval set composition | Corpus is 2 health + 1 home + 1 life. Health-weighted split (~50/25/25) recommended; home and life will be noisy at that sample size |
| Chunk size 1000/150 | Baseline. Revisit only with eval evidence, not intuition |
