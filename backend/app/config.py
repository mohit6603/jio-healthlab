"""Application configuration.

All settings are sourced from environment variables (or a local ``.env`` file).
No secret may ever be hard-coded here -- see ``.env.example`` for the full list.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for the JIO HealthLab backend service."""

    # ---------------------------------------------------------------- app ---
    app_name: str = "JIO Healthlab"
    environment: str = "local"
    log_level: str = "INFO"
    log_json: bool = True

    # ----------------------------------------------------------- database ---
    database_url: str = "mysql+pymysql://healthlab:healthlab@localhost:3306/jio_healthlab"

    # --------------------------------------------------------------- http ---
    backend_cors_origins: str = (
        "http://localhost,http://localhost:5173,http://127.0.0.1:5173"
    )

    # --------------------------------------------------------- ai service ---
    ai_service_url: str = "http://ai-service:8001"
    ai_service_timeout_seconds: float = 60.0
    ai_service_connect_timeout_seconds: float = 5.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def effective_database_url(self) -> str:
        """Normalise the DB URL.

        Render (and Heroku) hand out ``postgres://`` URLs which SQLAlchemy 2.x
        no longer accepts -- rewrite them to ``postgresql://``.
        """
        url = self.database_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    @property
    def cors_origins(self) -> list[str]:
        """CORS allow-list parsed from the comma-separated env value."""
        return [
            origin.strip()
            for origin in self.backend_cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (safe to call from anywhere, incl. DI)."""
    return Settings()
