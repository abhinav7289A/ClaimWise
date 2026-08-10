# ClaimWise — Workflow

A running index of every file in this repository and what it does.
Updated whenever a file is added or its responsibility changes.

---

## Root

### `CLAUDE.md`
The project constitution. Defines the non-negotiable workflow (Claude writes code, never executes it; every file ships with an Execution Card; work stops until real run output is pasted back), the teaching contract (Concept Brief before each module, trade-off tables before design decisions), the compute and cost constraints (no paid cloud, $30 Modal hard cap, ZeroGPU for serving only), the six-phase build plan with per-phase metrics and exit criteria, and the metrics-honesty rules. Overrides all default assistant behaviour.

### `workflow.md`
This file. A file-by-file map of the repository so the structure stays legible as it grows across six phases.

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
| `phase1_rag/embed_index.py` | Embed chunks on CPU and index them into Qdrant with `{user_id, doc_id, page, section}` |
| `phase1_rag/rag_chain.py` | Retrieve → prompt → generate, with the generator behind a swappable interface |
| `phase1_rag/build_eval_set.py` | Generate ~100 golden Q&A pairs for manual verification |
| `evals/run_ragas.py` | Baseline RAGAS evaluation |
| `.env.example` | Template for API keys (added when the first provider call appears) |
