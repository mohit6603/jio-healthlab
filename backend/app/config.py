"""Application configuration.

All settings are sourced from environment variables (or a local ``.env`` file).
No secret may ever be hard-coded here -- see ``.env.example`` for the full list.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Sentinels that must never reach production.
INSECURE_JWT_SECRET = "dev-only-insecure-secret-change-me"
DEFAULT_SEED_PASSWORD = "ChangeMe!Admin123"


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

    # --------------------------------------------------------------- auth ---
    #: MUST be overridden in any deployment. The application refuses to start
    #: in production while this is still the development placeholder.
    jwt_secret_key: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "jio-healthlab"
    #: Short, because an access token cannot be revoked once issued.
    access_token_expire_minutes: int = 15
    #: Long, because refresh tokens are revocable and rotated on every use.
    refresh_token_expire_days: int = 14
    #: Refresh tokens are delivered as an httpOnly cookie, so JavaScript --
    #: including anything an XSS injects -- cannot read them. Disable only for
    #: a non-browser client that cannot hold cookies.
    refresh_cookie_enabled: bool = True
    refresh_cookie_name: str = "healthlab_refresh"
    #: Set automatically in production; off locally because http://localhost
    #: is not a secure context for cookie purposes.
    refresh_cookie_secure: bool | None = None
    #: Strict blocks the cookie on any cross-site navigation, which is what a
    #: CSRF attack needs. There is no legitimate cross-site flow here.
    refresh_cookie_samesite: str = "strict"
    #: Scoped so the cookie is only ever sent to the auth endpoints.
    refresh_cookie_path: str = "/api/auth"

    #: Consecutive failures before an account is temporarily locked.
    login_max_attempts: int = 8
    login_lockout_minutes: int = 15

    #: Seeded on first start so a fresh install is usable. Demo only.
    seed_admin_email: str = "admin@jiohealthlab.example.com"
    seed_admin_password: str = "ChangeMe!Admin123"

    # --------------------------------------------------------- ai service ---
    ai_service_url: str = "http://ai-service:8001"
    ai_service_timeout_seconds: float = 60.0
    ai_service_connect_timeout_seconds: float = 5.0
    #: Health probes must fail fast: the UI polls them, and a hung probe
    #: would make the whole page feel broken when only AI is down.
    ai_health_timeout_seconds: float = 3.0

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

    @property
    def cookie_secure(self) -> bool:
        """Whether to mark the refresh cookie Secure.

        Forced on in production. Locally it must be off, or the browser
        discards the cookie over plain http and sign-in appears to work but
        never survives a reload.
        """
        if self.refresh_cookie_secure is not None:
            return self.refresh_cookie_secure
        return self.is_production

    @property
    def jwt_secret_is_default(self) -> bool:
        return self.jwt_secret_key == INSECURE_JWT_SECRET

    def assert_production_ready(self) -> None:
        """Refuse to run in production with development credentials.

        A placeholder signing key in production means anyone can mint tokens
        for any role. Failing at startup is the only safe behaviour.
        """
        if not self.is_production:
            return
        problems = []
        if self.jwt_secret_is_default:
            problems.append("JWT_SECRET_KEY is still the development placeholder")
        if self.seed_admin_password == DEFAULT_SEED_PASSWORD:
            problems.append("SEED_ADMIN_PASSWORD is still the default")
        if problems:
            raise RuntimeError(
                "Refusing to start in production: " + "; ".join(problems)
            )


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (safe to call from anywhere, incl. DI)."""
    return Settings()
