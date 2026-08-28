"""AI service configuration.

Every knob is environment-driven so the same image runs unchanged on a laptop,
in Docker Compose and on ECS. See ``.env.example`` at the repository root.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository-relative root of this service (``ai-service/``).
SERVICE_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment-driven settings for the JIO HealthLab AI service."""

    # ---------------------------------------------------------------- app ---
    app_name: str = "JIO HealthLab AI"
    environment: str = "local"
    ai_service_port: int = 8001
    log_level: str = "INFO"
    log_json: bool = True

    # --------------------------------------------------------- embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    #: Dimensionality of ``embedding_model``. all-MiniLM-L6-v2 emits 384-d
    #: vectors; override when swapping the model.
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    #: Cosine similarity assumes unit-length vectors.
    embedding_normalize: bool = True
    embedding_device: str = "cpu"
    #: Hard input limit of ``embedding_model``. all-MiniLM-L6-v2 truncates at
    #: 256 word pieces (verified against the loaded model).
    embedding_max_tokens: int = 256

    # ---------------------------------------------------------------- llm ---
    #: ``local`` (Hugging Face Transformers) or ``none`` to disable generation
    #: entirely while keeping retrieval available.
    llm_provider: str = "local"
    llm_model: str = "Qwen/Qwen2.5-1.5B"
    llm_max_new_tokens: int = 320
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_device: str = "cpu"
    #: Load the generation model at startup instead of on first use. Off by
    #: default so the container becomes healthy without waiting on a download.
    llm_eager_load: bool = False

    # ------------------------------------------------------------- qdrant ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "healthlab_knowledge"
    qdrant_report_collection: str = "healthlab_reports"
    qdrant_timeout_seconds: float = 15.0

    # ---------------------------------------------------------- retrieval ---
    top_k: int = 5
    #: Chunks scoring below this are discarded before prompt construction.
    score_threshold: float = 0.25
    #: Approximate tokens per chunk. Kept just under ``embedding_max_tokens``:
    #: anything longer is truncated by the embedding model before the vector is
    #: computed, so the tail of a larger chunk would never affect retrieval.
    chunk_size: int = 220
    chunk_overlap: int = 40

    # ---------------------------------------------------------- knowledge ---
    knowledge_dir: str = str(SERVICE_ROOT / "knowledge")
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MiB

    # ----------------------------------------------------------------- ml ---
    model_dir: str = str(SERVICE_ROOT / "app" / "ml" / "artifacts")
    delay_model_name: str = "report-delay-classifier"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # ``model_`` is a protected namespace in pydantic v2; we use
        # ``model_dir`` deliberately, so relax the guard for this class.
        protected_namespaces=(),
    )

    @property
    def knowledge_path(self) -> Path:
        return Path(self.knowledge_dir)

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir)

    @property
    def generation_enabled(self) -> bool:
        return self.llm_provider.lower() not in {"none", "off", "disabled", ""}

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()
