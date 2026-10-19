"""Конфігурація застосунку, константи та утиліти часу (Europe/Kyiv)."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

KYIV_TZ = ZoneInfo("Europe/Kyiv")

ARCHIVE_AUTHOR = "Лісовий архів"
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})
# Ліміт Telegram Bot API на завантаження файлів ботом — 20 МБ
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024

DEFAULT_GDRIVE_URL = "https://drive.google.com/drive/folders/1b_hEjnkQfP8ZVP-OEbnYWqoGSFwXMAP3"
_URL_RE = re.compile(r"https?://[^\s\]\[()<>\"']+")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_bot_token: str
    groq_api_key: str
    gdrive_folder_url: str = DEFAULT_GDRIVE_URL

    # Llama 3.3 70B / 3.1 8B Groq вимкнув 16.08.2026 — актуальні рекомендовані заміни:
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "intfloat/multilingual-e5-small"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    # Косинусна відстань (0 — ідентично, 2 — протилежно). 2.0 = фільтр вимкнено
    retrieval_max_distance: float = 2.0
    max_question_length: int = 1000

    data_dir: Path = Path("./data")
    log_level: str = "INFO"
    gdrive_force_sync: bool = False

    @field_validator("gdrive_folder_url", mode="before")
    @classmethod
    def _clean_url(cls, value: object) -> object:
        """Терпимо до формату `[https://...]` / markdown-посилань у .env."""
        if not isinstance(value, str) or not value.strip():
            return DEFAULT_GDRIVE_URL
        match = _URL_RE.search(value)
        return match.group(0) if match else value.strip()

    # ---- Шляхи всередині data_dir ----
    @property
    def db_path(self) -> Path:
        return self.data_dir / "bot.sqlite3"

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def gdrive_cache_dir(self) -> Path:
        return self.data_dir / "gdrive_cache"

    @property
    def fallback_dir(self) -> Path:
        """Локальна папка-запасний варіант, якщо Google Drive недоступний."""
        return self.data_dir / "initial_docs"

    @property
    def bootstrap_marker(self) -> Path:
        return self.data_dir / ".gdrive_bootstrapped"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.files_dir,
            self.chroma_dir,
            self.logs_dir,
            self.fallback_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


# ---- Час ----
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_kyiv(iso_value: str) -> str:
    """ISO-рядок (UTC) -> `DD.MM.YYYY HH:MM` за київським часом."""
    dt = datetime.fromisoformat(iso_value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KYIV_TZ).strftime("%d.%m.%Y %H:%M")
