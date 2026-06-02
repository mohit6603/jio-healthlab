from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "JIO Healthlab"
    environment: str = "local"
    database_url: str = "mysql+pymysql://healthlab:healthlab@localhost:3306/jio_healthlab"
    backend_cors_origins: str = (
        "http://localhost,http://localhost:5173,http://127.0.0.1:5173"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def effective_database_url(self) -> str:
        """Handle Render's postgres:// scheme (needs postgresql://)."""
        url = self.database_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.backend_cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
