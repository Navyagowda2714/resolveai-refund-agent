"""Application settings (env vars prefixed with ``RESOLVEAI_``)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = PACKAGE_DIR / "policy" / "refund_policy.yaml"
DEFAULT_POLICY_DOC_PATH = PACKAGE_DIR / "policy" / "refund_policy.md"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RESOLVEAI_", env_file=".env", extra="ignore")

    db_path: str = "data/resolveai.db"
    policy_path: str = str(DEFAULT_POLICY_PATH)
    policy_doc_path: str = str(DEFAULT_POLICY_DOC_PATH)
    max_tool_calls: int = 8
    enforce_guardrails: bool = True
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
