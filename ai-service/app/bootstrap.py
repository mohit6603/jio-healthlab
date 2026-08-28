"""First-run bootstrap: make a fresh stack usable.

    python -m app.bootstrap

Ingests the bundled knowledge base and trains the delay model, but only when
each is actually missing. Safe to run on every ``docker compose up``: a second
run does nothing and costs a couple of seconds.

Failures are reported and do not abort the other step. A missing knowledge
index and a missing ML model are independent problems, and neither should stop
the service starting -- retrieval and prediction degrade separately.
"""

from __future__ import annotations

import sys

from .config import get_settings
from .core.errors import AIError
from .core.logging import configure_logging, get_logger
from .ml.registry import ModelRegistry
from .rag.embeddings import get_embedder
from .rag.ingestion import IngestionPipeline
from .rag.vector_store import get_vector_store

logger = get_logger(__name__)


def knowledge_is_indexed() -> bool:
    """Whether the knowledge collection already holds vectors."""
    stats = get_vector_store().stats()
    return stats.exists and stats.vector_count > 0


def model_is_trained() -> bool:
    settings = get_settings()
    return ModelRegistry(settings.model_path).current_version() is not None


def ensure_knowledge(force: bool = False) -> bool:
    """Ingest the bundled knowledge base if the index is empty."""
    settings = get_settings()

    if not force and knowledge_is_indexed():
        stats = get_vector_store().stats()
        print(
            f"  knowledge: already indexed "
            f"({stats.vector_count} vectors, {stats.document_count} documents)"
        )
        return True

    print(f"  knowledge: ingesting {settings.knowledge_path} ...")
    pipeline = IngestionPipeline(get_embedder(), get_vector_store(), settings)
    report = pipeline.ingest_directory(settings.knowledge_path)

    for source, message in report.failures:
        print(f"    FAILED {source}: {message}")

    print(
        f"  knowledge: {report.document_count} document(s), "
        f"{report.chunk_count} chunk(s) in {report.duration_ms:.0f} ms"
    )
    return not report.failures


def ensure_model(force: bool = False) -> bool:
    """Train the delay model if no artifact exists."""
    settings = get_settings()

    if not force and model_is_trained():
        version = ModelRegistry(settings.model_path).current_version()
        print(f"  ml model: already trained ({version})")
        return True

    print("  ml model: training on synthetic data ...")
    # Imported here so the bootstrap does not pull scikit-learn in when only
    # the knowledge step is wanted.
    from .ml.train import train

    metadata = train(quiet=True)
    print(
        f"  ml model: {metadata.version} ({metadata.algorithm}, "
        f"roc_auc {metadata.metrics.get('roc_auc')})"
    )
    return True


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m app.bootstrap",
        description="Ingest the knowledge base and train the delay model if missing.",
    )
    parser.add_argument("--force", action="store_true", help="Redo both steps.")
    parser.add_argument("--skip-knowledge", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(level=get_settings().log_level, json_output=False)
    print("Bootstrapping JIO HealthLab AI ...")

    failures = 0

    if not args.skip_knowledge:
        try:
            if not ensure_knowledge(force=args.force):
                failures += 1
        except AIError as exc:
            print(f"  knowledge: FAILED [{exc.code}] {exc.message}", file=sys.stderr)
            failures += 1

    if not args.skip_model:
        try:
            ensure_model(force=args.force)
        except (AIError, ValueError, OSError) as exc:
            print(f"  ml model: FAILED {exc}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"\nBootstrap finished with {failures} problem(s).", file=sys.stderr)
        # Non-zero so an orchestrator can surface it, but the service itself
        # is unaffected -- it degrades rather than failing to start.
        return 1

    print("\nBootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
