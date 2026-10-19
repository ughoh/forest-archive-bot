"""SQLite: користувачі та метадані документів (асинхронно, через aiosqlite)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from app.config import utc_now_iso

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id    INTEGER PRIMARY KEY,
    real_name  TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    file_id     TEXT PRIMARY KEY,          -- внутрішній ідентифікатор версії (він же ключ у векторній базі)
    file_name   TEXT NOT NULL,             -- назва файлу для відображення
    name_key    TEXT NOT NULL UNIQUE,      -- casefold(file_name): ключ для заміни версій
    author_name TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('gdrive', 'user')),
    created_at  TEXT NOT NULL,             -- ISO, UTC
    stored_path TEXT NOT NULL,
    chunks_count INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass(frozen=True)
class DocumentRecord:
    file_id: str
    file_name: str
    author_name: str
    source_type: str
    created_at: str
    stored_path: str
    chunks_count: int = 0

    @property
    def name_key(self) -> str:
        return name_key(self.file_name)


def name_key(file_name: str) -> str:
    # casefold, а не SQLite NOCASE: останній працює лише для ASCII
    return file_name.strip().casefold()


def _row_to_record(row: aiosqlite.Row) -> DocumentRecord:
    return DocumentRecord(
        file_id=row["file_id"],
        file_name=row["file_name"],
        author_name=row["author_name"],
        source_type=row["source_type"],
        created_at=row["created_at"],
        stored_path=row["stored_path"],
        chunks_count=row["chunks_count"],
    )


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("База даних не ініціалізована: викличте connect()")
        return self._conn

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        logger.info("SQLite підключено: %s", self._path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info("SQLite закрито")

    # ---- users ----
    async def get_user_name(self, user_id: int) -> str | None:
        async with self.conn.execute(
            "SELECT real_name FROM users WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
        return row["real_name"] if row else None

    async def set_user_name(self, user_id: int, real_name: str) -> None:
        await self.conn.execute(
            """
            INSERT INTO users (user_id, real_name, created_at) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET real_name = excluded.real_name
            """,
            (user_id, real_name, utc_now_iso()),
        )
        await self.conn.commit()

    # ---- documents ----
    async def get_document_by_name(self, file_name: str) -> DocumentRecord | None:
        async with self.conn.execute(
            "SELECT * FROM documents WHERE name_key = ?", (name_key(file_name),)
        ) as cur:
            row = await cur.fetchone()
        return _row_to_record(row) if row else None

    async def count_documents(self) -> int:
        async with self.conn.execute("SELECT COUNT(*) AS c FROM documents") as cur:
            row = await cur.fetchone()
        return int(row["c"])

    async def upsert_document(self, record: DocumentRecord) -> DocumentRecord | None:
        """Зберігає документ; якщо документ із такою назвою вже був — замінює його.

        Повертає попередній запис (якщо був), щоб викликач видалив старий файл.
        """
        previous = await self.get_document_by_name(record.file_name)
        try:
            await self.conn.execute(
                "DELETE FROM documents WHERE name_key = ?", (record.name_key,)
            )
            await self.conn.execute(
                """
                INSERT INTO documents
                    (file_id, file_name, name_key, author_name, source_type,
                     created_at, stored_path, chunks_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.file_id,
                    record.file_name,
                    record.name_key,
                    record.author_name,
                    record.source_type,
                    record.created_at,
                    record.stored_path,
                    record.chunks_count,
                ),
            )
            await self.conn.commit()
        except Exception:
            await self.conn.rollback()
            raise
        return previous
