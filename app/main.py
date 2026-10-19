"""Точка входу: ініціалізація компонентів і запуск long polling."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from logging.handlers import RotatingFileHandler

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from app.bot.handlers import router
from app.config import Settings, get_settings
from app.database import Database
from app.gdrive import bootstrap_initial_documents
from app.rag_engine import RAGEngine

logger = logging.getLogger("app.main")


def setup_logging(settings: Settings) -> None:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file = RotatingFileHandler(
        settings.logs_dir / "bot.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(settings.log_level.upper())
    root.addHandler(stream)
    root.addHandler(file)

    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def _bootstrap(settings: Settings, db: Database, rag: RAGEngine) -> None:
    rag.bootstrapping = True
    try:
        await bootstrap_initial_documents(settings, db, rag)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Первинне наповнення архіву завершилось помилкою")
    finally:
        rag.bootstrapping = False


async def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    setup_logging(settings)
    logger.info("Запуск бота «Мишка-шкряботушка»...")

    db = Database(settings.db_path)
    await db.connect()

    rag = RAGEngine(settings)
    await rag.initialize()
    await rag.check_models()

    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher(storage=MemoryStorage(), db=db, rag=rag, settings=settings)
    dp.include_router(router)

    # Наповнення архіву йде у фоні, щоб бот відповідав одразу
    bootstrap_task = asyncio.create_task(_bootstrap(settings, db, rag), name="bootstrap")

    try:
        me = await bot.get_me()
        logger.info("Бот запущено: @%s", me.username)
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        logger.info("Зупинка бота...")
        bootstrap_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await bootstrap_task
        await bot.session.close()
        await db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
