"""Первинне наповнення архіву: Google Drive (публічна папка) з локальним fallback."""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
from pathlib import Path

from app.config import ARCHIVE_AUTHOR, SUPPORTED_EXTENSIONS, Settings, utc_now_iso
from app.database import Database
from app.rag_engine import DocumentProcessingError, RAGEngine

logger = logging.getLogger(__name__)

_FOLDER_ID_RE = re.compile(r"/folders/([A-Za-z0-9_-]+)")
_DOWNLOAD_TIMEOUT_SEC = 900


class GDriveError(Exception):
    pass


def extract_folder_id(url: str) -> str | None:
    match = _FOLDER_ID_RE.search(url)
    return match.group(1) if match else None


def _list_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))


async def _download_from_gdrive(settings: Settings) -> list[Path]:
    folder_id = extract_folder_id(settings.gdrive_folder_url)
    if not folder_id:
        raise GDriveError(f"Не вдалося визначити ID папки з URL: {settings.gdrive_folder_url}")

    target = settings.gdrive_cache_dir
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    def _run() -> object:
        import gdown

        return gdown.download_folder(
            url=f"https://drive.google.com/drive/folders/{folder_id}",
            output=str(target),
            quiet=True,
            use_cookies=False,
            remaining_ok=True,  # не падати, якщо у папці понад 50 файлів
        )

    try:
        await asyncio.wait_for(asyncio.to_thread(_run), timeout=_DOWNLOAD_TIMEOUT_SEC)
    except asyncio.TimeoutError as exc:
        raise GDriveError("Таймаут завантаження з Google Drive") from exc
    except Exception as exc:
        raise GDriveError(f"Помилка завантаження з Google Drive: {exc!r}") from exc

    files = _list_files(target)
    if not files:
        raise GDriveError("Папка Google Drive порожня або недоступна (чи вона публічна?)")
    return files


async def bootstrap_initial_documents(settings: Settings, db: Database, rag: RAGEngine) -> None:
    """Завантажує початкові документи один раз.

    * Успішне повне наповнення з Google Drive фіксується маркер-файлом у ./data.
    * Якщо Google Drive недоступний — береться локальна папка `data/initial_docs`
      (маркер тоді не ставиться, і наступний старт знову спробує Google Drive).
    * Документи, що вже є в базі (зокрема замінені користувачами), не перезаписуються.
    """
    marker = settings.bootstrap_marker
    if marker.exists() and not settings.gdrive_force_sync:
        logger.info("Первинне наповнення вже виконано раніше — пропускаємо")
        return

    used_fallback = False
    try:
        logger.info("Завантаження початкових документів з Google Drive...")
        files = await _download_from_gdrive(settings)
        logger.info("З Google Drive отримано файлів: %d", len(files))
    except GDriveError as exc:
        logger.warning("Google Drive недоступний (%s). Пробуємо локальну папку %s",
                       exc, settings.fallback_dir)
        files = _list_files(settings.fallback_dir)
        used_fallback = True

    supported = [f for f in files if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    skipped_unsupported = len(files) - len(supported)
    if not supported:
        logger.warning("Немає придатних початкових документів (PDF/DOCX/TXT). Архів порожній.")
        return

    added = skipped = failed = 0
    for path in supported:
        name = path.name
        try:
            if await db.get_document_by_name(name):
                skipped += 1
                logger.info("'%s' уже є в архіві — пропускаємо", name)
                continue
            result = await rag.ingest(
                db,
                source_path=path,
                file_name=name,
                author_name=ARCHIVE_AUTHOR,
                source_type="gdrive",
                files_dir=settings.files_dir,
                move=not used_fallback,  # локальний fallback не чіпаємо
            )
            added += 1
            logger.info("Додано '%s' (%d фрагментів)", name, result.chunks)
        except DocumentProcessingError as exc:
            failed += 1
            logger.warning("Пропущено '%s': %s", name, exc)
        except Exception:
            failed += 1
            logger.exception("Помилка під час обробки '%s'", name)

    logger.info(
        "Первинне наповнення завершено: додано=%d, вже були=%d, помилок=%d, непідтримуваних=%d",
        added, skipped, failed, skipped_unsupported,
    )

    if not used_fallback and failed == 0:
        marker.write_text(utc_now_iso(), encoding="utf-8")
    shutil.rmtree(settings.gdrive_cache_dir, ignore_errors=True)
