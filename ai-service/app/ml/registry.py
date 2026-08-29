"""Model artifact storage and metadata.

A prediction is only meaningful if you can say which model produced it and
what that model was trained on. Every artifact is saved with a sidecar
metadata document recording the version, the training data, the feature
schema and the measured metrics, and prediction responses carry the version
back to the caller.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from ..core.logging import get_logger

logger = get_logger(__name__)

ARTIFACT_SUFFIX = ".joblib"
METADATA_SUFFIX = ".json"
#: Symlink-free "current" pointer, so serving does not have to guess.
CURRENT_POINTER = "current.json"


@dataclass(slots=True)
class ModelMetadata:
    """Everything needed to interpret and reproduce a trained model."""

    model_name: str
    version: str
    algorithm: str
    trained_at: str
    dataset: dict[str, Any] = field(default_factory=dict)
    feature_schema: dict[str, list[str]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    #: Selection is done on a validation split; reported metrics come from a
    #: test split the selection never saw.
    selection_metric: str = "roc_auc"
    artifact_path: str = ""
    #: Loud, because these numbers describe a simulation.
    synthetic_data: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_version(now: datetime | None = None) -> str:
    """Timestamp-based version: sortable, unique, human-readable."""
    moment = now or datetime.now(UTC)
    return moment.strftime("v%Y%m%d%H%M%S")


class ModelRegistry:
    """Filesystem-backed store of trained models."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory

    def _artifact_path(self, version: str) -> Path:
        return self._directory / f"{version}{ARTIFACT_SUFFIX}"

    def _metadata_path(self, version: str) -> Path:
        return self._directory / f"{version}{METADATA_SUFFIX}"

    @property
    def pointer_path(self) -> Path:
        return self._directory / CURRENT_POINTER

    # ------------------------------------------------------------- write ---
    def save(self, model: Any, metadata: ModelMetadata) -> Path:
        """Persist a model and its metadata, and mark it current."""
        self._directory.mkdir(parents=True, exist_ok=True)

        artifact = self._artifact_path(metadata.version)
        joblib.dump(model, artifact)
        metadata.artifact_path = str(artifact)

        self._metadata_path(metadata.version).write_text(
            json.dumps(metadata.to_dict(), indent=2), encoding="utf-8"
        )
        self.pointer_path.write_text(
            json.dumps({"version": metadata.version}, indent=2), encoding="utf-8"
        )

        logger.info(
            "model_saved",
            extra={
                "version": metadata.version,
                "algorithm": metadata.algorithm,
                "roc_auc": metadata.metrics.get("roc_auc"),
            },
        )
        return artifact

    # -------------------------------------------------------------- read ---
    def current_version(self) -> str | None:
        """Version marked current, or the newest on disk as a fallback."""
        if self.pointer_path.exists():
            try:
                pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
                version = pointer.get("version")
                if version and self._artifact_path(version).exists():
                    return str(version)
            except (OSError, ValueError):
                logger.warning("model_pointer_unreadable")

        versions = self.list_versions()
        return versions[-1] if versions else None

    def list_versions(self) -> list[str]:
        """All stored versions, oldest first."""
        if not self._directory.exists():
            return []
        return sorted(
            path.stem
            for path in self._directory.glob(f"*{ARTIFACT_SUFFIX}")
            if path.is_file()
        )

    def load(self, version: str | None = None) -> tuple[Any, ModelMetadata]:
        """Load a model and its metadata. Raises if nothing is stored."""
        resolved = version or self.current_version()
        if resolved is None:
            raise FileNotFoundError(
                f"No trained model in {self._directory}. "
                "Run `python -m app.ml.train`."
            )

        artifact = self._artifact_path(resolved)
        if not artifact.exists():
            raise FileNotFoundError(f"Model artifact '{artifact}' does not exist.")

        model = joblib.load(artifact)
        metadata = self.load_metadata(resolved)
        return model, metadata

    def load_metadata(self, version: str) -> ModelMetadata:
        path = self._metadata_path(version)
        if not path.exists():
            raise FileNotFoundError(f"Model metadata '{path}' does not exist.")
        return ModelMetadata(**json.loads(path.read_text(encoding="utf-8")))
