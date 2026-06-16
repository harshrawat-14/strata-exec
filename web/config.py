"""
Application configuration loaded from environment variables / .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Suppress pydantic "protected namespace" warnings for model_* fields
        protected_namespaces=(),
    )

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./strataexec.db"

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── S3 Storage ────────────────────────────────────────────────────────
    s3_endpoint_url: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_bucket_name: str = "strataexec"
    use_s3: bool = False

    # ── Rust binaries ────────────────────────────────────────────────────
    rust_binary_path: str = "./target/release"

    # ── Data paths ───────────────────────────────────────────────────────
    data_path: str = "./TradeData"
    model_path: str = "./rl/models"
    upload_path: str = "./uploads"
    results_path: str = "./results"


    # ── Simulation limits ────────────────────────────────────────────────
    max_paths: int = 2000
    default_n_paths: int = 500
    max_episodes: int = 200
    default_n_episodes: int = 50

    # ── CORS ─────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── App ──────────────────────────────────────────────────────────────
    app_title: str = "StrataExec API"
    app_version: str = "1.0.0"
    debug: bool = False
    testing: bool = False
    log_level: str = "INFO"
    jwt_secret: str = "supersecretkeyforlocalrun123!"
    jwt_expire_minutes: int = 1440

    @property
    def research_sim_path(self) -> Path:
        return Path(self.rust_binary_path) / "research-sim"

    @property
    def rl_env_path(self) -> Path:
        return Path(self.rust_binary_path) / "rl-env"

    @property
    def upload_dir(self) -> Path:
        p = Path(self.upload_path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def lob_upload_dir(self) -> Path:
        p = self.upload_dir / "lob"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def agg_upload_dir(self) -> Path:
        p = self.upload_dir / "agg_trades"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def model_upload_dir(self) -> Path:
        p = self.upload_dir / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
