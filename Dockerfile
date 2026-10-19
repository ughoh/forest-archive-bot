# syntax=docker/dockerfile:1

# ---------- Етап 1: збірка залежностей через Poetry ----------
FROM python:3.11-slim AS builder

ARG POETRY_VERSION=1.8.5
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    HF_HOME=/opt/hf_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Poetry живе в окремому venv, щоб не змішуватися із залежностями застосунку
RUN python -m venv /opt/poetry-venv \
    && /opt/poetry-venv/bin/pip install "poetry==${POETRY_VERSION}" \
    && ln -s /opt/poetry-venv/bin/poetry /usr/local/bin/poetry

# Poetry не створює власних віртуальних середовищ — ставить у активне /opt/venv,
# яке потім копіюється у фінальний образ
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN poetry config virtualenvs.create false

WORKDIR /app
# poetry.lock* — glob: збірка працює і без lock-файлу (тоді залежності резолвляться при збірці)
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --without dev --no-ansi

# Завантажуємо модель ембедингів на етапі збірки — старт контейнера буде швидким
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', device='cpu')"

# ---------- Етап 2: робочий образ ----------
FROM python:3.11-slim

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf_cache \
    TOKENIZERS_PARALLELISM=false \
    TZ=Europe/Kyiv \
    DATA_DIR=/app/data

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf_cache /opt/hf_cache
COPY app ./app

VOLUME ["/app/data"]

CMD ["python", "-m", "app.main"]
