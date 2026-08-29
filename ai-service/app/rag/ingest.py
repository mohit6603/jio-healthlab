"""Knowledge-base ingestion CLI.

    python -m app.rag.ingest                 # ingest ai-service/knowledge/
    python -m app.rag.ingest --path docs/    # ingest another directory
    python -m app.rag.ingest --dry-run       # chunk only; no model, no writes
    python -m app.rag.ingest --recreate      # drop the collection first
    python -m app.rag.ingest --stats         # report what is indexed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import get_settings
from ..core.errors import AIError
from ..core.logging import configure_logging, get_logger
from .chunker import chunk_document, estimate_tokens
from .embeddings import get_embedder
from .ingestion import IngestionPipeline
from .loaders import discover, load_file
from .vector_store import get_vector_store

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag.ingest",
        description="Ingest knowledge documents into the Qdrant vector store.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Directory to ingest (defaults to KNOWLEDGE_DIR).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Target collection (defaults to QDRANT_COLLECTION).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the collection before ingesting. Required after changing "
        "the embedding model, since vectors of different models are not "
        "comparable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk only. Does not load the embedding model or write "
        "to Qdrant.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Report what is currently indexed and exit.",
    )
    return parser


def _dry_run(root: Path) -> int:
    """Report the chunk plan without embedding or storing anything."""
    settings = get_settings()
    paths = discover(root)
    if not paths:
        print(f"No ingestible documents found under {root}")
        return 1

    total_chunks = 0
    oversized = 0
    print(f"Dry run over {len(paths)} document(s) in {root}\n")
    print(f"{'source':32} {'chunks':>6} {'min':>5} {'max':>5} {'avg':>5}")
    print("-" * 58)

    for path in paths:
        document = load_file(path, root=root)
        chunks = chunk_document(
            document_id=document.document_id,
            title=document.title,
            source=document.source,
            category=document.category,
            text=document.text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        sizes = [estimate_tokens(chunk.text) for chunk in chunks] or [0]
        oversized += sum(1 for size in sizes if size > settings.embedding_max_tokens)
        total_chunks += len(chunks)
        print(
            f"{document.source:32} {len(chunks):6d} "
            f"{min(sizes):5d} {max(sizes):5d} {sum(sizes) // len(sizes):5d}"
        )

    print("-" * 58)
    print(f"{'TOTAL':32} {total_chunks:6d} chunks")
    if oversized:
        print(
            f"\nWARNING: {oversized} chunk(s) exceed EMBEDDING_MAX_TOKENS="
            f"{settings.embedding_max_tokens}; their tails will be truncated "
            "before embedding. Lower CHUNK_SIZE."
        )
    return 0


def _stats(collection: str | None) -> int:
    store = get_vector_store()
    stats = store.stats(collection)

    if not stats.exists:
        print(f"Collection '{stats.collection}' does not exist yet.")
        return 1

    print(f"Collection : {stats.collection}")
    print(f"Vectors    : {stats.vector_count}")
    print(f"Documents  : {stats.document_count}")
    print(f"Dimension  : {stats.dimension}\n")

    for document in store.list_documents(collection):
        print(
            f"  {document.chunk_count:3d} chunks  "
            f"{document.category:12} {document.source:30} {document.title}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=False)

    root = args.path or settings.knowledge_path

    try:
        if args.stats:
            return _stats(args.collection)
        if args.dry_run:
            return _dry_run(root)

        store = get_vector_store()
        collection = args.collection or settings.qdrant_collection

        if args.recreate:
            print(f"Dropping collection '{collection}' ...")
            store.drop_collection(collection)

        embedder = get_embedder()
        print(f"Loading embedding model '{embedder.model_name}' ...")
        embedder.warm_up()

        pipeline = IngestionPipeline(embedder, store, settings)
        print(f"Ingesting {root} -> '{collection}'\n")
        report = pipeline.ingest_directory(root, collection=collection)

        for result in report.documents:
            replaced = (
                f" (replaced {result.chunks_replaced})" if result.chunks_replaced else ""
            )
            print(
                f"  ok    {result.source:32} {result.chunks_written:3d} chunks"
                f"{replaced}"
            )
        for source, message in report.failures:
            print(f"  FAIL  {source:32} {message}")

        print(
            f"\n{report.document_count} document(s), {report.chunk_count} chunk(s) "
            f"in {report.duration_ms:.0f} ms"
        )
        if report.failures:
            print(f"{len(report.failures)} document(s) failed.")
            return 1
        return 0

    except AIError as exc:
        print(f"\nERROR [{exc.code}] {exc.message}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
