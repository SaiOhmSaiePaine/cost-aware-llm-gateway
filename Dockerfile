# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache

WORKDIR /build

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/fastembed \
    /opt/venv/bin/python -c "from fastembed import TextEmbedding; list(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/root/.cache/fastembed').embed(['warmup']))" \
    && cp -a /root/.cache/fastembed/. /opt/fastembed_cache/

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=2

WORKDIR /app

RUN useradd --create-home --shell /bin/bash gateway
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=gateway:gateway /opt/fastembed_cache /opt/fastembed_cache
COPY --chown=gateway:gateway src ./src

USER gateway
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
