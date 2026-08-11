"""Embed chunks on CPU and index them into Qdrant with filterable metadata.

Keyword search fails on insurance because users don't speak policy. Someone
asks "is my knee surgery covered?" and the document says "arthroscopic
procedures of the knee joint shall be payable subject to..." — not one content
word in common. An embedding maps both into a space where proximity means
related *meaning*, which is the only reason retrieval works here at all.

**Why a vector database and not a list of vectors.** At 652 chunks a brute-force
cosine scan would be fine. What Qdrant adds is persistence (upload-time work
happens once), sub-linear search as the corpus grows, and — the one that
actually matters — **filters applied during index traversal rather than after
it**. Retrieving the global top-5 and then discarding other users' chunks
returns fewer than 5 results and turns one forgotten filter into a data leak.
Every point here carries `user_id`, and every search filters on it, from day
one, while there is still only one user.

**Collections are per-model.** The name encodes the embedding model, so
bge-small and bge-base can be indexed side by side and compared on the same
golden questions rather than overwriting each other.

**The query prefix is not applied here.** BGE v1.5 models are trained with an
instruction prefix on queries only; documents are embedded bare. That asymmetry
lives in config and is applied by rag_chain.py at query time.

Usage:
    python -m phase1_rag.embed_index --help
    python -m phase1_rag.embed_index
    python -m phase1_rag.embed_index --model BAAI/bge-base-en-v1.5
    python -m phase1_rag.embed_index --recreate --smoke-query "knee surgery waiting period"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from common.config import DEFAULT_CONFIG_PATH, cfg_get, load_config

LOGGER = logging.getLogger("claimwise.embed_index")

# A fixed namespace makes uuid5(chunk_id) deterministic across machines and
# runs, so re-indexing upserts the same points instead of duplicating them.
POINT_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

DISTANCE_BY_NAME = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}

# Fields we filter or group by. Qdrant needs an explicit payload index on each
# to filter efficiently; `user_id` is the one that must never be slow, because
# it is on the path of every single query.
INDEXED_PAYLOAD_FIELDS = ("user_id", "doc_id", "insurer", "policy_type")

DEFAULT_SMOKE_QUERY = "What is the waiting period for pre-existing diseases?"


@dataclass(frozen=True)
class IndexStats:
    """Timings and counts for one index build.

    Attributes:
        chunks_indexed: Number of points upserted.
        vector_size: Embedding dimensionality, read from the loaded model.
        embed_seconds: Wall-clock time spent embedding.
        upsert_seconds: Wall-clock time spent writing to Qdrant.
        chunks_per_second: Embedding throughput, the number that matters when
            comparing models and when estimating Phase 5 upload times.
    """

    chunks_indexed: int
    vector_size: int
    embed_seconds: float
    upsert_seconds: float
    chunks_per_second: float


def slugify_model_name(model_name: str) -> str:
    """Turn a HuggingFace model id into a safe collection-name fragment.

    "BAAI/bge-small-en-v1.5" becomes "baai_bge_small_en_v1_5".

    Args:
        model_name: The model id.

    Returns:
        A lowercase string containing only letters, digits and underscores.
    """
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def collection_name_for(prefix: str, model_name: str) -> str:
    """Build the collection name for a given embedding model.

    Encoding the model in the name is what makes an A/B possible: two models
    produce two collections that can be queried with the same questions.

    Args:
        prefix: Collection prefix from config, e.g. "claimwise".
        model_name: The embedding model id.

    Returns:
        The collection name, e.g. "claimwise__baai_bge_small_en_v1_5".
    """
    return f"{prefix}__{slugify_model_name(model_name)}"


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Read all chunk records from a JSONL file.

    Loaded into memory rather than streamed because embedding batches need
    random access and the corpus is small by design.

    Args:
        chunks_path: Path to `chunks.jsonl` produced by `chunk.py`.

    Returns:
        Decoded chunk records in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or a line is malformed.
    """
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path.resolve()}. "
            "Run `python -m phase1_rag.chunk` first."
        )

    records: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSON at {chunks_path}:{line_number} — {error}"
                ) from error

    if not records:
        raise ValueError(f"{chunks_path} contains no chunks.")
    return records


def load_embedder(model_name: str, device: str) -> SentenceTransformer:
    """Load a sentence-transformers embedding model.

    The first call downloads the model from HuggingFace (~130MB for bge-small,
    ~440MB for bge-base) and caches it under ~/.cache/huggingface.

    Args:
        model_name: HuggingFace model id.
        device: "cpu" or "cuda".

    Returns:
        The loaded model.
    """
    LOGGER.info("Loading embedding model %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    normalize: bool,
) -> list[list[float]]:
    """Embed a list of texts into vectors.

    Args:
        model: The loaded embedding model.
        texts: Texts to embed, in order.
        batch_size: How many texts to encode per forward pass.
        normalize: Scale each vector to unit length. Required for cosine
            distance to behave as expected.

    Returns:
        One vector per input text, as plain Python lists ready for Qdrant.
    """
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return [vector.tolist() for vector in vectors]


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: str,
    recreate: bool,
) -> None:
    """Create the collection and its payload indexes if needed.

    Args:
        client: An open Qdrant client.
        collection_name: Target collection.
        vector_size: Embedding dimensionality.
        distance: One of "cosine", "dot", "euclid".
        recreate: Drop an existing collection first, discarding its contents.

    Raises:
        ValueError: If `distance` is unknown, or if the collection already
            exists with a different vector size — silently mixing
            dimensionalities would corrupt every search result.
    """
    if distance not in DISTANCE_BY_NAME:
        raise ValueError(
            f"Unknown distance {distance!r}; expected one of {sorted(DISTANCE_BY_NAME)}."
        )

    exists = client.collection_exists(collection_name)
    if exists and recreate:
        LOGGER.warning("Dropping existing collection %s (--recreate)", collection_name)
        client.delete_collection(collection_name)
        exists = False

    if exists:
        existing_size = client.get_collection(collection_name).config.params.vectors.size
        if existing_size != vector_size:
            raise ValueError(
                f"Collection {collection_name} exists with vector size {existing_size}, "
                f"but the model produces {vector_size}. Re-run with --recreate."
            )
        LOGGER.info("Reusing existing collection %s", collection_name)
        return

    LOGGER.info("Creating collection %s (size=%d, distance=%s)", collection_name, vector_size, distance)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size, distance=DISTANCE_BY_NAME[distance]
        ),
    )

    for field in INDEXED_PAYLOAD_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except NotImplementedError:
            # Embedded local mode filters correctly without explicit indexes;
            # only the server build needs them for speed.
            LOGGER.debug("Payload index on %r not supported in local mode", field)


def to_point(chunk: dict[str, Any], vector: list[float], user_id: str) -> models.PointStruct:
    """Convert a chunk plus its vector into a Qdrant point.

    Args:
        chunk: A decoded chunk record.
        vector: Its embedding.
        user_id: Owner of the document. Written on every point and filtered on
            by every query — the security boundary.

    Returns:
        A point with a deterministic id derived from `chunk_id`.
    """
    return models.PointStruct(
        id=str(uuid.uuid5(POINT_ID_NAMESPACE, chunk["chunk_id"])),
        vector=vector,
        payload={
            "user_id": user_id,
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "filename": chunk["filename"],
            "insurer": chunk["insurer"],
            "policy_type": chunk["policy_type"],
            "doc_label": chunk["doc_label"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
        },
    )


def index_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict[str, Any]],
    model: SentenceTransformer,
    settings: dict[str, Any],
) -> IndexStats:
    """Embed every chunk and upsert it into the collection.

    Args:
        client: An open Qdrant client.
        collection_name: Target collection.
        chunks: Chunk records to index.
        model: The loaded embedding model.
        settings: Resolved embed/index settings.

    Returns:
        Timings and counts for the build.
    """
    texts = [chunk["text"] for chunk in chunks]

    embed_start = time.perf_counter()
    vectors = embed_texts(
        model, texts, batch_size=settings["batch_size"], normalize=settings["normalize"]
    )
    embed_seconds = time.perf_counter() - embed_start

    points = [
        to_point(chunk, vector, settings["default_user_id"])
        for chunk, vector in zip(chunks, vectors)
    ]

    upsert_start = time.perf_counter()
    batch_size = settings["upsert_batch_size"]
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[start : start + batch_size])
    upsert_seconds = time.perf_counter() - upsert_start

    return IndexStats(
        chunks_indexed=len(points),
        vector_size=len(vectors[0]),
        embed_seconds=round(embed_seconds, 2),
        upsert_seconds=round(upsert_seconds, 2),
        chunks_per_second=round(len(points) / embed_seconds, 1) if embed_seconds else 0.0,
    )


def smoke_query(
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer,
    query: str,
    query_prefix: str,
    user_id: str,
    normalize: bool,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Run one filtered search so the run verifies its own index.

    An index that builds without error but returns nothing useful is the most
    expensive kind of silent failure, because every metric downstream inherits
    it. This prints real results at build time instead.

    Args:
        client: An open Qdrant client.
        collection_name: Collection to search.
        model: The embedding model, used to embed the query.
        query: The question text.
        query_prefix: BGE instruction prefix, applied to the query only.
        user_id: Filter value — results from other users are never scored.
        normalize: Must match how the documents were embedded.
        top_k: How many results to return.

    Returns:
        A list of {score, page, insurer, filename, snippet} dicts.
    """
    vector = model.encode(
        query_prefix + query, normalize_embeddings=normalize, convert_to_numpy=True
    ).tolist()

    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=top_k,
        with_payload=True,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        ),
    )

    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        snippet = " ".join(payload.get("text", "").split())[:180]
        results.append(
            {
                "score": round(point.score, 4),
                "page": payload.get("page"),
                "insurer": payload.get("insurer"),
                "filename": payload.get("filename"),
                "snippet": snippet,
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m phase1_rag.embed_index",
        description="Embed chunks and index them into Qdrant with filterable metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml."
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="Input chunks JSONL. Default: from config."
    )
    parser.add_argument(
        "--model", default=None, help="Override embed.model_name. Builds its own collection."
    )
    parser.add_argument("--device", default=None, help="Override embed.device (cpu / cuda).")
    parser.add_argument(
        "--user-id", default=None, help="Override index.default_user_id written to every point."
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Drop the collection before indexing."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Index only the first N chunks (smoke test)."
    )
    parser.add_argument(
        "--smoke-query",
        default=DEFAULT_SMOKE_QUERY,
        help="Question used to verify the index after building.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def resolve_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge config file values with CLI overrides.

    Args:
        config: Parsed `config.yaml`.
        args: Parsed command-line arguments.

    Returns:
        The fully resolved embed/index settings, logged with the run.
    """
    return {
        "model_name": args.model or cfg_get(config, "embed.model_name", "BAAI/bge-small-en-v1.5"),
        "device": args.device or cfg_get(config, "embed.device", "cpu"),
        "batch_size": cfg_get(config, "embed.batch_size", 32),
        "normalize": cfg_get(config, "embed.normalize", True),
        "query_prefix": cfg_get(config, "embed.query_prefix", ""),
        "qdrant_path": cfg_get(config, "index.path", "qdrant_storage"),
        "collection_prefix": cfg_get(config, "index.collection_prefix", "claimwise"),
        "distance": cfg_get(config, "index.distance", "cosine"),
        "default_user_id": args.user_id or cfg_get(config, "index.default_user_id", "local-dev"),
        "upsert_batch_size": cfg_get(config, "index.upsert_batch_size", 128),
    }


def main(argv: list[str] | None = None) -> int:
    """Run embedding and indexing end to end.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code — 0 on success, 1 if the smoke query returned nothing.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    settings = resolve_settings(config, args)

    processed_dir = Path(cfg_get(config, "paths.processed_dir", "data/processed"))
    input_path = args.input or processed_dir / cfg_get(
        config, "index.input_filename", "chunks.jsonl"
    )

    chunks = load_chunks(input_path)
    if args.limit is not None:
        chunks = chunks[: args.limit]

    model = load_embedder(settings["model_name"], settings["device"])
    collection_name = collection_name_for(settings["collection_prefix"], settings["model_name"])

    client = QdrantClient(path=settings["qdrant_path"])
    try:
        vector_size = model.get_sentence_embedding_dimension()
        ensure_collection(
            client,
            collection_name=collection_name,
            vector_size=vector_size,
            distance=settings["distance"],
            recreate=args.recreate,
        )

        LOGGER.info("Indexing %d chunk(s) into %s", len(chunks), collection_name)
        stats = index_chunks(client, collection_name, chunks, model, settings)

        results = smoke_query(
            client,
            collection_name=collection_name,
            model=model,
            query=args.smoke_query,
            query_prefix=settings["query_prefix"],
            user_id=settings["default_user_id"],
            normalize=settings["normalize"],
        )
        point_count = client.count(collection_name, exact=True).count
    finally:
        client.close()

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_path": input_path.as_posix(),
        "collection": collection_name,
        "settings": settings,
        "points_in_collection": point_count,
        "chunks_indexed": stats.chunks_indexed,
        "vector_size": stats.vector_size,
        "embed_seconds": stats.embed_seconds,
        "upsert_seconds": stats.upsert_seconds,
        "chunks_per_second": stats.chunks_per_second,
        "smoke_query": args.smoke_query,
        "smoke_results": results,
    }
    meta_path = processed_dir / f"index_{slugify_model_name(settings['model_name'])}.meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== INDEX SUMMARY ===")
    print(f"model              : {settings['model_name']}")
    print(f"collection         : {collection_name}")
    print(f"vector size        : {stats.vector_size}")
    print(f"chunks indexed     : {stats.chunks_indexed}")
    print(f"points in store    : {point_count}")
    print(f"embed time         : {stats.embed_seconds}s ({stats.chunks_per_second} chunks/s)")
    print(f"upsert time        : {stats.upsert_seconds}s")
    print(f"run metadata       : {meta_path}")

    print(f"\n--- smoke query: {args.smoke_query!r} ---")
    if not results:
        LOGGER.error(
            "Smoke query returned no results. The index built but is not searchable — "
            "do not proceed to rag_chain.py."
        )
        return 1
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. score={result['score']} p.{result['page']} "
            f"[{result['insurer']}] {result['snippet']}..."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
