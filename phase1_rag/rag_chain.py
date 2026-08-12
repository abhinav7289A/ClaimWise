"""The naive RAG baseline: retrieve, assemble one prompt, generate, verify.

This is where retrieval meets generation, and where hallucination is either
prevented or invited. The naive version stuffs chunks into a prompt and says
"answer the question" — at which point the model happily fills gaps from
pretraining. In insurance that is the worst possible failure mode: a
hallucinated waiting period is specific, plausible, and wrong, and the user has
no way to tell. Refusing is strictly better than guessing.

So grounding is an explicit contract rather than a hope:

* Passages are labelled with their page numbers, making citation mechanical
  rather than something the model has to remember.
* The system prompt states that anything not in the passages does not exist.
* Refusal is a *named, expected output* with fixed wording, not a failure.
* Arithmetic is forbidden. Phase 3 adds a deterministic calculator; an LLM
  doing co-pay maths in its head is a bug, not a feature.

**Citations are verified, not trusted.** Every `[p.N]` the model emits is checked
against the pages actually retrieved. That costs nothing, runs on every call,
and gives us a real citation-validity rate — the same signal Phase 4.5's GRPO
reward function will optimise against.

Usage:
    python -m phase1_rag.rag_chain --help
    python -m phase1_rag.rag_chain --question "What is the waiting period for PED?"
    python -m phase1_rag.rag_chain -q "Is knee surgery covered?" --insurer starhealth
    python -m phase1_rag.rag_chain -q "..." --show-chunks --provider openrouter
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config
from common.generator import Generator, build_generator
from phase1_rag.embed_index import collection_name_for

LOGGER = logging.getLogger("claimwise.rag_chain")

# Accepts [p.19], [p. 19] and [page 19] — small models drift between forms and
# penalising formatting noise would understate real citation accuracy.
CITATION_PATTERN = re.compile(r"\[(?:p\.?|page)\s*(\d+)\]", re.IGNORECASE)

# The grounding contract. Phase 4 generates RAFT training data in exactly this
# format, so the prompt the fine-tuned model sees at inference matches the one
# it was trained on. Change it here and the training data must be regenerated.
SYSTEM_PROMPT = """You are ClaimWise, an assistant that answers questions about \
the user's own insurance policy documents.

Rules you must follow exactly:
1. Answer ONLY from the numbered passages provided. They are the user's actual
   policy. Never use general insurance knowledge.
2. Cite a page for every factual claim, in the form [p.N], using the page number
   shown on the passage you took the claim from.
3. A passage stating a general rule DOES answer a question about a specific case
   falling under that rule. If a passage says something applies to "61 years and
   above", it answers a question about someone aged 65 — apply it and cite it.
4. Refuse ONLY when the passages do not address the question at all. When
   refusing, reply with exactly this sentence and nothing else:
   "{refusal_text}"
5. Quote amounts, percentages, waiting periods and time limits exactly as
   written. Do not compute rupee figures, totals or converted values — a
   separate calculator does that. Applying a stated threshold or condition to
   the user's situation is not computation, and is required.
6. Be concise and factual. No preamble, no disclaimers."""

PROMPT_TEMPLATE = """{passages}

Question: {question}

Answer using only the passages above, citing pages as [p.N]."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by the vector search.

    Attributes:
        chunk_id: Stable chunk identifier from `chunk.py`.
        doc_id: Content hash of the source PDF.
        filename: Source PDF filename.
        insurer: e.g. "starhealth".
        policy_type: e.g. "health".
        doc_label: Free-text document label.
        page: 1-indexed PDF page — what gets cited.
        text: The chunk text placed into the prompt.
        score: Cosine similarity to the query.
    """

    chunk_id: str
    doc_id: str
    filename: str
    insurer: str
    policy_type: str
    doc_label: str
    page: int
    text: str
    score: float


@dataclass
class RagAnswer:
    """A complete answer plus everything needed to evaluate it.

    Attributes:
        question: The question asked.
        answer: The generated answer text.
        retrieved: Chunks placed into the prompt, best first.
        cited_pages: Pages the model cited, in order of appearance.
        invalid_citations: Cited pages that were NOT among the retrieved pages —
            fabricated citations, the metric Phase 4.5 targets directly.
        refused: Whether the model emitted the exact refusal sentence.
        retrieval_seconds: Time spent embedding the query and searching.
        generation_seconds: Time spent in the generator.
        prompt_tokens: Input tokens billed.
        completion_tokens: Output tokens billed.
        provider: Provider that served the generation.
        model: Generator model id.
    """

    question: str
    answer: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    cited_pages: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    refused: bool = False
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    provider: str = ""
    model: str = ""

    @property
    def total_seconds(self) -> float:
        """End-to-end latency, the number reported as p50/p95."""
        return round(self.retrieval_seconds + self.generation_seconds, 3)

    @property
    def total_tokens(self) -> int:
        """Tokens per query, for the cost table."""
        return self.prompt_tokens + self.completion_tokens


def build_search_filter(
    user_id: str, insurer: str | None = None, policy_type: str | None = None
) -> models.Filter:
    """Build the Qdrant filter applied during index traversal.

    `user_id` is always present. It is a security boundary, not an optimisation:
    filtering after search would return fewer results than asked for and would
    leak the moment one code path forgot it.

    Args:
        user_id: Owner whose documents may be searched.
        insurer: Optionally restrict to one insurer.
        policy_type: Optionally restrict to one policy type.

    Returns:
        A filter requiring all supplied conditions.
    """
    conditions = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
    ]
    if insurer:
        conditions.append(
            models.FieldCondition(key="insurer", match=models.MatchValue(value=insurer))
        )
    if policy_type:
        conditions.append(
            models.FieldCondition(
                key="policy_type", match=models.MatchValue(value=policy_type)
            )
        )
    return models.Filter(must=conditions)


def retrieve(
    client: QdrantClient,
    collection_name: str,
    embedder: SentenceTransformer,
    question: str,
    top_k: int,
    query_prefix: str,
    normalize: bool,
    search_filter: models.Filter,
) -> list[RetrievedChunk]:
    """Embed the question and fetch the most similar chunks.

    Args:
        client: An open Qdrant client.
        collection_name: Collection to search.
        embedder: The same model the documents were embedded with. Using a
            different one silently produces meaningless similarities.
        question: The user's question.
        top_k: How many chunks to return.
        query_prefix: BGE instruction prefix — applied to the query only,
            because that is how these models were trained.
        normalize: Must match the setting used at index time.
        search_filter: Filter applied during traversal.

    Returns:
        Chunks ordered best-first.
    """
    vector = embedder.encode(
        query_prefix + question, normalize_embeddings=normalize, convert_to_numpy=True
    ).tolist()

    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=top_k,
        with_payload=True,
        query_filter=search_filter,
    )

    chunks: list[RetrievedChunk] = []
    for point in response.points:
        payload = point.payload or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=payload.get("chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                filename=payload.get("filename", ""),
                insurer=payload.get("insurer", ""),
                policy_type=payload.get("policy_type", ""),
                doc_label=payload.get("doc_label", ""),
                page=int(payload.get("page", 0)),
                text=payload.get("text", ""),
                score=round(point.score, 4),
            )
        )
    return chunks


def format_passages(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, page-labelled passages.

    The page number sits in the header of every passage so citing is a copy
    rather than a recall task — the single cheapest thing you can do to improve
    citation accuracy.

    Args:
        chunks: Retrieved chunks, best first.

    Returns:
        The passage block for the prompt.
    """
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        header = (
            f"[Passage {index} | {chunk.insurer} {chunk.doc_label} "
            f"| {chunk.policy_type} | page {chunk.page}]"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the single prompt sent to the generator.

    Args:
        question: The user's question.
        chunks: Retrieved chunks to ground the answer in.

    Returns:
        The complete user message.
    """
    return PROMPT_TEMPLATE.format(passages=format_passages(chunks), question=question)


def verify_citations(answer_text: str, chunks: list[RetrievedChunk]) -> tuple[list[int], list[int]]:
    """Check every cited page against the pages actually retrieved.

    Deterministic and free, so it runs on every call. A page the model cites
    that was never in its context is a fabricated citation — the failure mode
    Phase 4.5's citation-validity reward exists to eliminate.

    Args:
        answer_text: The generated answer.
        chunks: Chunks that were placed in the prompt.

    Returns:
        A tuple of (cited pages in order of appearance, invalid cited pages).
    """
    cited = [int(match) for match in CITATION_PATTERN.findall(answer_text)]
    available = {chunk.page for chunk in chunks}
    invalid = [page for page in cited if page not in available]
    return cited, invalid


def answer_question(
    question: str,
    client: QdrantClient,
    collection_name: str,
    embedder: SentenceTransformer,
    generator: Generator,
    settings: dict[str, Any],
    insurer: str | None = None,
    policy_type: str | None = None,
) -> RagAnswer:
    """Run the full retrieve → prompt → generate → verify cycle.

    Args:
        question: The user's question.
        client: An open Qdrant client.
        collection_name: Collection to search.
        embedder: Query embedding model, matching the index.
        generator: The swappable generator.
        settings: Resolved rag/embed settings.
        insurer: Optional metadata filter.
        policy_type: Optional metadata filter.

    Returns:
        The answer with its retrieved context, citation check and timings.
    """
    retrieval_started = time.perf_counter()
    chunks = retrieve(
        client,
        collection_name=collection_name,
        embedder=embedder,
        question=question,
        top_k=settings["top_k"],
        query_prefix=settings["query_prefix"],
        normalize=settings["normalize"],
        search_filter=build_search_filter(settings["user_id"], insurer, policy_type),
    )
    retrieval_seconds = time.perf_counter() - retrieval_started

    if not chunks:
        # No context means there is nothing to be grounded in. Refusing here
        # without calling the model saves a request and is the correct answer.
        LOGGER.warning("No chunks retrieved for %r — refusing without generating.", question)
        return RagAnswer(
            question=question,
            answer=settings["refusal_text"],
            refused=True,
            retrieval_seconds=round(retrieval_seconds, 3),
        )

    prompt = build_prompt(question, chunks)
    system = SYSTEM_PROMPT.format(refusal_text=settings["refusal_text"])

    generation_started = time.perf_counter()
    result = generator.generate(prompt, system=system)
    generation_seconds = time.perf_counter() - generation_started

    cited, invalid = verify_citations(result.text, chunks)
    return RagAnswer(
        question=question,
        answer=result.text,
        retrieved=chunks,
        cited_pages=cited,
        invalid_citations=invalid,
        refused=settings["refusal_text"].lower() in result.text.lower(),
        retrieval_seconds=round(retrieval_seconds, 3),
        generation_seconds=round(generation_seconds, 3),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        provider=result.provider,
        model=result.model,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase1_rag.rag_chain",
        description="Ask a question against your indexed policy documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-q", "--question", required=True, help="The question to ask.")
    parser.add_argument("--top-k", type=int, default=None, help="Override rag.top_k.")
    parser.add_argument("--provider", default=None, help="Override generator.provider.")
    parser.add_argument("--model", default=None, help="Override the generator model.")
    parser.add_argument("--embed-model", default=None, help="Override embed.model_name.")
    parser.add_argument("--user-id", default=None, help="Override index.default_user_id.")
    parser.add_argument("--insurer", default=None, help="Restrict retrieval to one insurer.")
    parser.add_argument(
        "--policy-type", default=None, help="Restrict retrieval to one policy type."
    )
    parser.add_argument(
        "--show-chunks", action="store_true", help="Print the retrieved passages."
    )
    parser.add_argument("--json", action="store_true", help="Emit the full result as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def resolve_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge config file values with CLI overrides.

    Args:
        config: Parsed `config.yaml`.
        args: Parsed command-line arguments.

    Returns:
        The fully resolved settings used by this run.
    """
    return {
        "top_k": args.top_k if args.top_k is not None else cfg_get(config, "rag.top_k", 5),
        "refusal_text": cfg_get(
            config, "rag.refusal_text", "That isn't covered in the policy documents you've uploaded."
        ),
        "embed_model": args.embed_model or cfg_get(config, "embed.model_name"),
        "query_prefix": cfg_get(config, "embed.query_prefix", ""),
        "normalize": cfg_get(config, "embed.normalize", True),
        "device": cfg_get(config, "embed.device", "cpu"),
        "qdrant_path": cfg_get(config, "index.path", "qdrant_storage"),
        "collection_prefix": cfg_get(config, "index.collection_prefix", "claimwise"),
        "user_id": args.user_id or cfg_get(config, "index.default_user_id", "local-dev"),
    }


def main(argv: list[str] | None = None) -> int:
    """Answer one question from the command line.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if a citation was fabricated.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    settings = resolve_settings(config, args)

    embedder = SentenceTransformer(settings["embed_model"], device=settings["device"])
    generator = build_generator(config, provider=args.provider, model=args.model)
    collection_name = collection_name_for(
        settings["collection_prefix"], settings["embed_model"]
    )

    client = QdrantClient(path=settings["qdrant_path"])
    try:
        if not client.collection_exists(collection_name):
            LOGGER.error(
                "Collection %s does not exist. Run `python -m phase1_rag.embed_index` first.",
                collection_name,
            )
            return 1
        result = answer_question(
            args.question,
            client=client,
            collection_name=collection_name,
            embedder=embedder,
            generator=generator,
            settings=settings,
            insurer=args.insurer,
            policy_type=args.policy_type,
        )
    finally:
        client.close()

    if args.json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
        return 1 if result.invalid_citations else 0

    print(f"\nQ: {result.question}\n")
    print("=== ANSWER ===")
    print(result.answer)

    if args.show_chunks:
        print("\n=== RETRIEVED PASSAGES ===")
        for index, chunk in enumerate(result.retrieved, start=1):
            snippet = " ".join(chunk.text.split())[:200]
            print(
                f"{index}. score={chunk.score} p.{chunk.page} "
                f"[{chunk.insurer} {chunk.doc_label}] {snippet}..."
            )

    pages = ", ".join(f"p.{page}" for page in result.cited_pages) or "none"
    print("\n=== CHECKS ===")
    print(f"retrieved pages    : {[chunk.page for chunk in result.retrieved]}")
    print(f"cited pages        : {pages}")
    print(f"invalid citations  : {result.invalid_citations or 'none'}")
    print(f"refused            : {result.refused}")
    print(f"latency            : {result.retrieval_seconds}s retrieve + "
          f"{result.generation_seconds}s generate = {result.total_seconds}s")
    print(f"tokens             : {result.prompt_tokens} in / "
          f"{result.completion_tokens} out = {result.total_tokens}")
    print(f"served by          : {result.provider}/{result.model}")

    if result.invalid_citations:
        LOGGER.error(
            "Model cited page(s) %s that were not retrieved — fabricated citation.",
            result.invalid_citations,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
