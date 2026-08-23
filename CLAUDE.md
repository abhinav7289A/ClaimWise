# CLAUDE.md — ClaimWise Project Rules

Agentic Health Insurance Policy & Claims Assistant.
Learning-first, production-style build: RAG → Advanced RAG → Agentic AI → Fine-tuning → Post-training → Deployment + Frontend.

---

## 1. WORKFLOW RULES (NON-NEGOTIABLE)

- You WRITE code only. You NEVER execute it.
- Never run Python scripts, install packages, start servers, launch notebooks, or trigger training jobs.
- Never use Bash to run `python`, `pip`, `uv`, `pytest`, `uvicorn`, `modal`, `docker`, `npm`, `npx`, `node`, `vercel`, or any execution command. This applies to the frontend too: I run `npm install`, `npm run dev`, and all deploys myself.
- Allowed Bash: `ls`, `cat`, `git status`, `git diff` — for inspecting the repo only.
- After writing or editing ANY file, STOP and give me an **Execution Card**:
  1. **What this file does** (2–4 lines, plain language)
  2. **Exact command(s) to run**, in order, copy-pasteable
  3. **Expected output** — what success looks like (sample logs, metric ranges, files created)
  4. **Failure modes** — the 2–3 most likely errors and what they mean
  5. **Metric to record** — what number I should write into `metrics/METRICS.md` after this run
- I execute everything in my own terminal and paste results/errors back to you.
- WAIT for my pasted results before writing the next file. Never write ahead of my execution.
- If my pasted output shows an error, debug by reasoning over the log first; only then edit code.

> **Amendment history — read this before proposing a change to §1.**
> On 2026-08-17 I lifted this rule verbally without editing the file, which left
> the constitution contradicting practice and caused exactly the drift you would
> expect. On 2026-08-18 it was briefly rewritten to split on "who pays" (Claude
> ran free deterministic local work; I ran anything that spent money or GPU
> time), and on the same day I **reverted it in full** to the text above.
> Write-only is the standing rule. If it is ever lifted again, the lift is not in
> force until this section is edited — a verbal change alone does not count.

## 2. TEACHING MODE (WHY THIS PROJECT EXISTS)

This project must teach me production-level AI engineering, not just produce working code.

- Before writing each new module, give a **Concept Brief** (5–10 lines): what problem this component solves in real production systems, what the naive approach is, why companies do it this way instead.
- When there is a design choice (chunking strategy, vector DB, reranker, reward function), present 2–3 options with trade-offs in one short table, recommend one, and let me confirm before writing code.
- Name the real-world equivalent when relevant ("this is what an evals team at an insurer would call a golden dataset").
- One concept per file. Prefer more small files over one clever file. Code must be readable by a beginner-intermediate engineer: docstrings on every function, type hints, no unnecessary abstractions.
- Never introduce a library without one line explaining why it and not the alternative.

## 3. PROJECT CONTEXT

**Problem:** Indian insurance policies like health insurance, life indurance, car insurance and other fanacial schemes are 60–100 page PDFs (tables, exclusions, waiting periods, sub-limits). Answering "Is my knee surgery covered after 18 months and what will I get back on a ₹2.4L bill?" needs retrieval + reasoning + deterministic math. Users can also attach photos of hospital bills and claim forms.

**Core architecture (understand this before writing any code):**
- **Upload time (once per document, CPU only):** parse → chunk → embed → index into Qdrant with metadata `{user_id, doc_id, page, section}`. The LLM never sees the document here; the document is only made *searchable*. `user_id` filtering is a security boundary — users retrieve only their own documents.
- **Query time (every question, ~2–5s):** route → hybrid retrieve → rerank to top-5 chunks → run deterministic tools → assemble ONE prompt (question + chunks + tool output + image) → fine-tuned Qwen generates → stream answer with page citations.
- **Division of labour:** RAG supplies *knowledge* (facts from specific policies, always via prompt at inference time). Fine-tuning supplies *skill* (insurance reasoning, staying grounded in context, citing correctly, refusing to guess). Fine-tuning NEVER embeds policy content in weights.
- **The generator is a swappable slot.** Phases 1–3 fill it with an OpenRouter free model; Phase 5 swaps in the fine-tuned Qwen Space endpoint. The `generate(prompt) → answer` interface must stay identical so nothing upstream changes.
- **Retrieval quality caps everything.** If the right chunk isn't retrieved, no generator can recover. Hence Phases 1–2 obsess over context recall before any fine-tuning.

**Compute & cost constraints (hard rules):**
- NO paid cloud (no AWS/GCP/Azure).
- LLM inference during development: OpenRouter free models and NVIDIA NIM free tier.
- GPU experimentation/notebooks: Lightning AI credits (15+15).
- GPU training: Modal, **$30 free credits is a hard cap**. Every training script must print an estimated cost before the real run and support checkpoint/resume.
- Serving/demo inference: Hugging Face ZeroGPU Spaces (HF Pro, 40 min/day — inference only, never training).
- Logging: append run logs to a private HF Dataset; local logs in `logs/` as JSONL.
- Embeddings, rerankers, vector DB (Qdrant local / FAISS): free/local only, CPU where possible.

**Stack (backend):** Python 3.11, LangChain, LangGraph, Qdrant, RAGAS, sentence-transformers, TRL/PEFT + Unsloth, Gradio Server mode, HF Hub.
**Stack (frontend, Phase 6 only):** Next.js 14+ (App Router), TypeScript, Tailwind, Framer Motion, Auth.js (NextAuth) with Google OAuth, Supabase free-tier Postgres for users + conversation history, deployed on Vercel free tier.

## 4. REPO STRUCTURE

```
claimwise/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── data/
│   ├── raw/            # policy PDFs (never committed)
│   ├── processed/      # parsed/chunked outputs
│   └── eval/           # golden Q&A eval set (versioned)
├── phase1_rag/         # naive RAG baseline
├── phase2_advanced/    # hybrid search, reranking, query rewriting
├── phase3_agents/      # LangGraph agents + tools
├── phase4_finetune/    # dataset gen, QLoRA on Modal, benchmarks
├── phase45_posttrain/  # DPO + GRPO, reward functions, preference data
├── deployment/         # Gradio Space, HF logging, monitoring
├── frontend/           # Next.js app (Phase 6) — see docs/prompts/frontend_brief.md
├── evals/              # RAGAS runners, eval configs
├── metrics/METRICS.md  # single source of truth for all numbers
├── logs/               # JSONL run logs (gitignored)
└── docs/
    ├── blog/           # HF blog drafts, one per phase
    └── prompts/        # frontend_brief.md and other briefs
```

## 5. BUILD PLAN — PHASES, TASKS, METRICS

Work strictly in order. Do not start a phase until I confirm the previous phase's exit criteria are met and recorded in `metrics/METRICS.md`.

### Phase 1 — Naive RAG Baseline
1. `ingest.py` — PDF parsing (pymupdf), page-number preservation, basic cleaning
2. `chunk.py` — fixed-size chunking with overlap; metadata tagging
3. `embed_index.py` — embed (bge-small, CPU) + index into Qdrant
4. `rag_chain.py` — LangChain retrieve→prompt→generate via OpenRouter free model; the generator must sit behind a swappable interface
5. `build_eval_set.py` — generate + I manually verify ~100 golden Q&A pairs
6. `run_ragas.py` — baseline eval

**Metrics:** faithfulness, answer relevancy, context recall, context precision, hit@5, p50/p95 latency, tokens per query.
**Exit criteria:** baseline numbers recorded; 10 worst failures analyzed in `metrics/failure_analysis_p1.md`.

### Phase 2 — Advanced RAG
Each technique is ONE change + ONE re-eval so improvements are attributable:
1. Hybrid search (BM25 + dense, fused)
2. Cross-encoder reranking (bge-reranker)
3. Parent-document / small-to-big retrieval (table-heavy sections)
4. Query rewriting / multi-query
5. Metadata filtering (insurer, policy type, section, user_id)
6. **Blog experiment:** full-context stuffing (262K window) vs RAG — measure accuracy, latency, tokens, cost. Prove the trade-off rather than asserting it.

**Metrics:** full RAGAS suite after EVERY technique — a delta table (technique → metric change → latency cost).
**Exit criteria:** best pipeline chosen with evidence; ≥15-point context-recall improvement vs baseline (revise honestly if the data disagrees).

### Phase 3 — Agentic Layer (LangGraph)
1. Router node — lookup / calculation / comparison / out-of-scope
2. Retrieval agent — wraps the Phase 2 pipeline as a tool
3. Claims calculator tool — deterministic Python for co-pay/sub-limit/waiting-period math (NO LLM arithmetic)
4. Policy comparison agent
5. Confidence gate + human-escalation node
6. Agent eval harness — task completion + tool-call accuracy on a 50-task set

**Metrics:** tool-call accuracy, task completion rate, escalation rate, avg steps per task, end-to-end latency vs plain RAG.
**Exit criteria:** agent beats plain RAG on calculation/comparison tasks with recorded evidence.

### Phase 4 — Supervised Fine-tuning (Modal) + Benchmark
**Model decision:** Qwen3.5-4B — natively multimodal, required because users attach bill/claim-form/policy images. Fallback if fine-tuning tooling doesn't yet support its hybrid (Gated DeltaNet + MoE) architecture: Qwen3-4B text-only, with base Qwen3.5-4B or NIM free vision models handling images un-fine-tuned.

**Framework mapping (do not deviate):**
- Unsloth (fallback TRL/PEFT) = **training only**, on Modal
- transformers + `@spaces.GPU` + TextIteratorStreamer = **serving** on ZeroGPU
- **NEVER vLLM** — it allocates GPU at startup and is fundamentally incompatible with ZeroGPU's serverless attach/release model
- GGUF = optional end-of-project export for local/Ollama use, not the serving path

**Training approach — RAFT (retrieval-augmented fine-tuning):** training examples are `(question + retrieved context) → grounded, cited answer`. We are training the model to be a better *RAG generator*: stick to context, cite pages, say "not covered in your policy" when the answer isn't in the chunks. Keep the dataset text-only — LoRA on the language side preserves the base model's vision ability, and multimodal fine-tuning is out of budget.

1. `check_compat.py` — cheap Modal run verifying Unsloth/TRL LoRA support for Qwen3.5-4B BEFORE any real spend; record go/fallback decision
2. `gen_dataset.py` — 3–5k synthetic RAFT-format Q&A via OpenRouter/NIM free models, with quality filtering; include deliberate "answer not in context" negatives
3. `train_modal.py` — QLoRA as a Modal app: volume-mounted checkpoints, resumable, cost printout before launch; merged model pushed to HF Hub
4. `benchmark.py` — fine-tuned vs base vs large API model **through the identical retrieval pipeline** (generator-swap comparison); plus a 10-image sanity suite confirming vision didn't regress

**Metrics:** eval-set win rate, RAGAS per generator, training cost ($), Modal credits remaining, inference cost per 1k queries, latency.
**Exit criteria:** fine-tuned model clearly ≥ base model; cost comparison table recorded; **Modal spend tracked with remaining balance written into METRICS.md**.

### Phase 4.5 — Post-training: DPO then GRPO (OPTIONAL, credit-gated)

**Gate — do not start unless ALL are true:** Phase 4 exit criteria met, ≥$15 Modal credits remaining, and I explicitly say go. **Reserve $5 that is never spent here** — it is insurance for Phase 5 serving problems.

**Why:** SFT teaches the format; post-training fixes the *remaining failure modes* — hallucinated figures, missing citations, hedging when the answer is right there in the chunks. Target only failures actually logged in Phase 4, not imagined ones.

**Why not classic RLHF/PPO:** it needs a separate trained reward model plus policy + reference + reward + value models in memory (~3–4x SFT cost) and is notoriously unstable. Out of budget and out of scope — document the reasoning in the blog, don't run it.

**4.5A — DPO (do first; cheap, stable, ~1.5x SFT cost):**
1. `gen_preferences.py` — for each eval question, sample 2 answers from the SFT model, have a strong free OpenRouter/NIM model judge which is better *grounded in the retrieved context*; produce ~1–2k (chosen, rejected) pairs at zero dollars
2. `train_dpo_modal.py` — TRL `DPOTrainer` on Modal, LoRA on top of the SFT adapter; cost printout before launch
3. Re-run the golden eval; record deltas

**4.5B — GRPO (only if DPO succeeded and credits allow; ~4–8x SFT cost per step):**
GRPO drops the value model and scores a *group* of sampled answers per prompt against a reward function. It fits ClaimWise because our rewards are **verifiable by rule, not by LLM judging**:
1. `rewards.py` — deterministic reward functions, each independently unit-testable:
   - **numeric correctness** — computed reimbursement matches ground truth
   - **citation validity** — every cited page number actually appears in the retrieved chunks
   - **groundedness** — every factual claim traces to a chunk
   - **format compliance** — required answer structure followed
2. `train_grpo_modal.py` — TRL `GRPOTrainer`, **narrowly scoped to calculation + citation accuracy only** (one measurable behavior, not general quality). Cost control is mandatory: ~500 prompts, small group size (4), short max generation length, few epochs, hard step cap, resumable checkpoints, estimated cost printed and confirmed by me before launch.
3. Re-run the golden eval; record deltas per reward dimension

**Metrics:** hallucinated-figure rate, citation validity rate, numeric accuracy, win rate vs SFT model, cost per run ($), credits remaining.
**Exit criteria:** measured improvement on the targeted failure modes, or an honest documented negative result — a "we tried GRPO and it didn't beat DPO here, and here's why" blog post is a legitimate outcome. Never report an improvement not computed from my pasted output.

### Phase 5 — Backend Deployment & Monitoring (HF Pro)
1. Gradio **Server mode** app on an HF Space serving the full agentic pipeline; final model + reranker via `@spaces.GPU` (ZeroGPU is INFERENCE ONLY; transformers + TextIteratorStreamer)
2. Upload endpoint: PDF ingestion on the Space's free CPU (parse/chunk/embed) with progress events — never spend ZeroGPU minutes on ingestion
3. Image attachment endpoint: bill/claim-form/policy photos → model vision input → extracted facts feed the agent pipeline
4. Streaming (SSE) with intermediate agent-status events: `ingesting`, `retrieving`, `reranking`, `calculating`, `generating` — the frontend mascot consumes these
5. API-key auth between frontend and Space (token server-side only, never in the browser); CORS locked to the Vercel domain
6. Query logging → private HF Dataset (hashed user_id, query, retrieved chunks, answer, latency, confidence, escalation flag, served-by path)
7. Graceful degradation: catch ZeroGPU quota errors and fall back to an OpenRouter free model; tag every response with which path served it

### Phase 6 — Frontend (Next.js, only after Phase 5 works via curl)
Follow `docs/prompts/frontend_brief.md` for the full design brief. Task order:
1. Design plan first — palette, type pairing, wireframes, mascot states — approved by me before any code
2. Scaffold Next.js 14 App Router + Tailwind + Framer Motion; design tokens
3. Auth.js with Google OAuth; Supabase schema (`users`, `conversations`, `messages`) + row-level security SQL for me to run
4. Chat UI: streaming answers, markdown, citation chips (policy + page), PDF upload, image attachment with thumbnail preview
5. Animated cartoon mascot loader driven by real SSE agent-status events, with progress bar and reduced-motion fallback
6. Conversation history: per-user sidebar, rename/delete, resume with full context
7. Page transitions and micro-interactions
8. Landing page with animated hero
9. Polish: empty/error states, mobile drawer, accessibility
10. Vercel deploy checklist (env vars, prod OAuth redirect URIs)

**Metrics (Phases 5–6):** p95 latency (ZeroGPU vs fallback path), ZeroGPU minutes per 100 queries, fallback rate, cost per 1k queries, time-to-first-token, Lighthouse performance + accessibility, chat-route bundle size, auth success rate.
**Exit criteria:** public URL works end-to-end — login → upload policy → ask with an attached bill photo → animated agent statuses → cited answer → conversation saved and resumable; monitoring dashboard renders from real logs.

## 6. METRICS DISCIPLINE

- `metrics/METRICS.md` is the single source of truth. Every Execution Card tells me exactly what to append there.
- Every entry: date, git commit hash, config used, number.
- **Never report a metric you did not compute from my pasted output. If I haven't pasted results, the metric does not exist.**
- Deltas over absolutes: always frame results as "X → Y after change Z".
- Track Modal credits remaining after every GPU run. Refuse to write a training script that would exceed the remaining balance.
- Negative results are recorded, not hidden. Honesty over vanity metrics.

## 7. DOCUMENTATION

- At the end of each phase, draft `docs/blog/phase{N}.md` — an HF blog post: problem → approach → what failed → metrics table → lessons. Written from MY recorded metrics only.
- Update the README mermaid architecture diagram whenever the pipeline changes.
- Maintain `docs/resume_bullets.md`: after each phase, draft 1–2 bullets in the format — what I did / how (tools) / why (business problem) / where (environment) — using real recorded numbers.

## 8. GENERAL CODE RULES

- Config in `config.yaml` / env vars — never hardcode model names, paths, or API keys.
- API keys only via `.env` (gitignored). Never print or log keys.
- Every script runnable standalone: `python -m phase1_rag.ingest --help` must work.
- Reproducibility: seed everything; log configs with every run.
- No silent failures: raise loudly, log clearly.
- Every LLM call goes through the swappable generator interface — no direct provider calls scattered through the codebase.
