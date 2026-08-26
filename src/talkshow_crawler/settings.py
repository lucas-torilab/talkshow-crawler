"""App settings, loaded from the environment or a .env file via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    pyannote_api_key: Optional[str] = None

    # Proxy for youtube_transcript_api, to work around YouTube IP-blocking the
    # caption-fetch endpoint after high request volume (see README: "IP blocks").
    # Either pair works; Webshare takes priority if both are set.
    webshare_proxy_username: Optional[str] = None
    webshare_proxy_password: Optional[str] = None
    youtube_proxy_url: Optional[str] = None  # e.g. "http://user:pass@host:port", used for both http/https


@lru_cache
def get_settings() -> Settings:
    return Settings()
