"""Обробники Telegram: команди, FSM реєстрації імені, документи, питання (RAG)."""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest, TelegramNetworkError
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state
from aiogram.types import ErrorEvent, Message
from aiogram.utils.chat_action import ChatActionSender

from app.bot.states import RegistrationStates
from app.config import MAX_FILE_SIZE_BYTES, SUPPORTED_EXTENSIONS, Settings
from app.database import Database
from app.rag_engine import DocumentProcessingError, LLMUnavailableError, RAGEngine

logger = logging.getLogger(__name__)
router = Router(name="main")

# ----------------------------------------------------------------------------
# Тексти (строго українською)
# ----------------------------------------------------------------------------
WELCOME_TEXT = (
    "🐭 Вітаю в «Лісовому архіві»! Я — Мишка-шкряботушка, хранителька порад нашої лісової громади.\n\n"
    "Як це працює:\n"
    "📄 Надішліть мені файл (PDF, Word .docx або TXT) — я збережу його як пораду для громади.\n"
    "❓ Напишіть звичайне повідомлення — я сприйму його як питання й відповім на основі порад з архіву.\n\n"
    "Додатково:\n"
    "• Коли ви вперше надішлете документ, я попрошу вказати ваше справжнє ім'я.\n"
    "• Якщо надішлете новішу версію файлу з тією самою назвою — вона замінить стару.\n"
    "• Наприкінці кожної відповіді я вказую, з якого документа взято пораду."
)
RULES_REMINDER = (
    "🐭 Я вмію працювати лише з текстом і файлами PDF, DOCX та TXT.\n\n"
    "📄 Надішліть файл у одному з цих форматів — це буде порада для громади.\n"
    "❓ Або напишіть питання звичайним текстом — я пошукаю відповідь в архіві.\n\n"
    "Фото, голосові, стікери, архіви та інші типи файлів я, на жаль, не розумію."
)
ASK_NAME_TEXT = (
    "🐭 Я бачу, що ви вперше ділитеся порадою! Перш ніж зберегти файл, підкажіть, "
    "будь ласка, ваше справжнє ім'я — воно буде вказано як автор поради.\n\n"
    "Просто напишіть його у відповідь. (Скасувати — /cancel)"
)
INVALID_NAME_TEXT = (
    "Це не схоже на ім'я 🙈 Напишіть, будь ласка, ім'я літерами (наприклад: «Олена» або "
    "«Іван Петренко»), від 2 до 64 символів."
)
WAITING_NAME_REMINDER = "Спершу напишіть, будь ласка, ваше справжнє ім'я текстовим повідомленням. (Скасувати — /cancel)"
NO_ANSWER_TEXT = "🐭 У «Лісовому архіві» немає такої поради."
ARCHIVE_FILLING_NOTE = "\n\n⏳ Зараз архів ще наповнюється початковими документами — спробуйте повторити питання за кілька хвилин."
LLM_DOWN_TEXT = "🐭 Зараз я не можу звернутися до своєї мудрості (сервіс відповідей недоступний). Спробуйте, будь ласка, трохи згодом."
GENERIC_ERROR_TEXT = "🐭 Ой, щось пішло не так. Спробуйте ще раз трохи згодом."
NETWORK_ERROR_TEXT = "🐭 Не вдалося отримати файл через проблеми з мережею. Спробуйте надіслати його ще раз."
FILE_TOO_BIG_TEXT = "🐭 Файл завеликий. Я можу прийняти файли до 20 МБ."
CANCELLED_TEXT = "Добре, скасовано. Можете надіслати файл або поставити питання."
UNKNOWN_COMMAND_TEXT = "🐭 Такої команди я не знаю. Скористайтеся /start, щоб переглянути правила."

_NAME_RE = re.compile(r"^[^\W\d_]+(?:[ '’ʼ-][^\W\d_]+)*$")
TELEGRAM_MESSAGE_LIMIT = 4096


@dataclass
class PendingDocument:
    """Документ, чекає на ім'я автора (зберігається у FSM)."""

    telegram_file_id: str
    file_name: str
    extension: str


# ----------------------------------------------------------------------------
# Допоміжні функції
# ----------------------------------------------------------------------------
def sanitize_filename(name: str | None, fallback: str) -> str:
    if not name:
        return fallback
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    if len(base) > 150:
        stem, ext = Path(base).stem, Path(base).suffix
        base = stem[: 150 - len(ext)] + ext
    return base or fallback


def normalize_name(raw: str) -> str | None:
    name = re.sub(r"\s+", " ", raw).strip()
    if not (2 <= len(name) <= 64) or not _NAME_RE.match(name):
        return None
    return name


async def send_long(message: Message, text: str) -> None:
    """Надсилає довгий текст частинами в межах ліміту Telegram."""
    while text:
        if len(text) <= TELEGRAM_MESSAGE_LIMIT:
            await message.answer(text)
            return
        cut = text.rfind("\n", 0, TELEGRAM_MESSAGE_LIMIT)
        if cut < TELEGRAM_MESSAGE_LIMIT // 2:
            cut = text.rfind(" ", 0, TELEGRAM_MESSAGE_LIMIT)
        if cut <= 0:
            cut = TELEGRAM_MESSAGE_LIMIT
        await message.answer(text[:cut].rstrip())
        text = text[cut:].lstrip()


async def _safe_edit(status: Message, text: str) -> None:
    try:
        await status.edit_text(text)
    except TelegramAPIError:
        await status.answer(text)


# ----------------------------------------------------------------------------
# Команди
# ----------------------------------------------------------------------------
@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(WELCOME_TEXT)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(CANCELLED_TEXT)


# ----------------------------------------------------------------------------
# FSM: очікування імені
# ----------------------------------------------------------------------------
@router.message(RegistrationStates.waiting_for_name, F.text, ~F.text.startswith("/"))
async def on_name_received(
    message: Message, state: FSMContext, bot: Bot, db: Database, rag: RAGEngine, settings: Settings
) -> None:
    if message.from_user is None:
        return
    name = normalize_name(message.text or "")
    if name is None:
        await message.answer(INVALID_NAME_TEXT)
        return

    data = await state.get_data()
    raw_pending = data.get("pending")
    await state.clear()

    try:
        await db.set_user_name(message.from_user.id, name)
    except Exception:
        logger.exception("Не вдалося зберегти ім'я користувача %s", message.from_user.id)
        await message.answer(GENERIC_ERROR_TEXT)
        return

    logger.info("Користувач %s зареєстрував ім'я '%s'", message.from_user.id, name)
    if not raw_pending:
        await message.answer(f"Дуже приємно, {name}! Надішліть файл ще раз, будь ласка — я його збережу.")
        return

    await message.answer(f"Дуже приємно, {name}! 🌿 Зберігаю ваш документ...")
    await process_document(message, bot, db, rag, settings, PendingDocument(**raw_pending), name)


@router.message(RegistrationStates.waiting_for_name)
async def on_waiting_name_other(message: Message) -> None:
    await message.answer(WAITING_NAME_REMINDER)


# ----------------------------------------------------------------------------
# Документи
# ----------------------------------------------------------------------------
@router.message(F.document, StateFilter(default_state))
async def on_document(
    message: Message, state: FSMContext, bot: Bot, db: Database, rag: RAGEngine, settings: Settings
) -> None:
    doc = message.document
    if doc is None or message.from_user is None:
        return

    file_name = sanitize_filename(doc.file_name, f"document_{doc.file_unique_id}")
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        logger.info("Відхилено непідтримуваний файл '%s' (mime=%s)", file_name, doc.mime_type)
        await message.answer(RULES_REMINDER)
        return
    if doc.file_size and doc.file_size > MAX_FILE_SIZE_BYTES:
        await message.answer(FILE_TOO_BIG_TEXT)
        return

    pending = PendingDocument(telegram_file_id=doc.file_id, file_name=file_name, extension=extension)

    author = await db.get_user_name(message.from_user.id)
    if author is None:
        # Призупиняємо збереження: питаємо справжнє ім'я
        await state.set_state(RegistrationStates.waiting_for_name)
        await state.update_data(pending=asdict(pending))
        await message.answer(ASK_NAME_TEXT)
        return

    await process_document(message, bot, db, rag, settings, pending, author)


async def process_document(
    message: Message,
    bot: Bot,
    db: Database,
    rag: RAGEngine,
    settings: Settings,
    pending: PendingDocument,
    author: str,
) -> None:
    """Завантаження файлу з Telegram -> індексація -> метадані у SQLite."""
    status = await message.answer("⏳ Опрацьовую документ, це може зайняти до хвилини...")
    settings.files_dir.mkdir(parents=True, exist_ok=True)
    incoming = settings.files_dir / f".incoming_{uuid.uuid4().hex}{pending.extension}"

    try:
        await bot.download(pending.telegram_file_id, destination=incoming)
        result = await rag.ingest(
            db,
            source_path=incoming,
            file_name=pending.file_name,
            author_name=author,
            source_type="user",
            files_dir=settings.files_dir,
            move=True,
        )
    except DocumentProcessingError as exc:
        logger.warning("Файл '%s' не опрацьовано: %s", pending.file_name, exc)
        await _safe_edit(status, f"🐭 {exc.user_message}")
    except TelegramNetworkError:
        logger.exception("Мережева помилка під час завантаження файлу з Telegram")
        await _safe_edit(status, NETWORK_ERROR_TEXT)
    except TelegramBadRequest as exc:
        logger.warning("Telegram відхилив завантаження файлу: %s", exc)
        text = FILE_TOO_BIG_TEXT if "too big" in str(exc).lower() else GENERIC_ERROR_TEXT
        await _safe_edit(status, text)
    except Exception:
        logger.exception("Неочікувана помилка під час обробки документа '%s'", pending.file_name)
        await _safe_edit(status, GENERIC_ERROR_TEXT)
    else:
        if result.replaced:
            text = (
                f"🔄 Дякую, {author}! Документ «{pending.file_name}» оновлено: "
                f"стару версію замінено новою (фрагментів: {result.chunks})."
            )
        else:
            text = (
                f"✅ Дякую, {author}! Пораду з документа «{pending.file_name}» "
                f"додано до Лісового архіву (фрагментів: {result.chunks})."
            )
        logger.info("Документ '%s' від '%s' збережено", pending.file_name, author)
        await _safe_edit(status, text)
    finally:
        incoming.unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# Питання (RAG)
# ----------------------------------------------------------------------------
@router.message(F.text, ~F.text.startswith("/"))
async def on_question(message: Message, bot: Bot, rag: RAGEngine, settings: Settings) -> None:
    question = (message.text or "").strip()
    if not question:
        return
    if len(question) > settings.max_question_length:
        await message.answer(
            f"🐭 Питання задовге. Скоротіть його, будь ласка, до {settings.max_question_length} символів."
        )
        return

    try:
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            result = await rag.answer(question)
    except LLMUnavailableError:
        logger.error("Усі моделі Groq недоступні")
        await message.answer(LLM_DOWN_TEXT)
        return
    except Exception:
        logger.exception("Помилка під час відповіді на питання")
        await message.answer(GENERIC_ERROR_TEXT)
        return

    if not result.found:
        text = NO_ANSWER_TEXT + (ARCHIVE_FILLING_NOTE if rag.bootstrapping else "")
        await message.answer(text)
        return

    citations = "\n".join(src.citation() for src in result.sources)
    await send_long(message, f"{result.answer}\n\n{citations}")


# ----------------------------------------------------------------------------
# Невідомі команди та непідтримуваний контент
# ----------------------------------------------------------------------------
@router.message(F.text.startswith("/"))
async def on_unknown_command(message: Message) -> None:
    await message.answer(UNKNOWN_COMMAND_TEXT)


@router.message()
async def on_unsupported(message: Message) -> None:
    await message.answer(RULES_REMINDER)


# ----------------------------------------------------------------------------
# Глобальний обробник помилок
# ----------------------------------------------------------------------------
@router.error()
async def on_error(event: ErrorEvent) -> bool:
    logger.error("Необроблена помилка в апдейті", exc_info=event.exception)
    message = event.update.message
    if message is not None:
        try:
            await message.answer(GENERIC_ERROR_TEXT)
        except TelegramAPIError:
            pass
    return True
