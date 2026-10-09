# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn && \
    pip install --no-cache-dir .

COPY . .
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN useradd -m -u 1000 signalrag && \
    mkdir -p /app/storage /app/data && \
    chown -R signalrag:signalrag /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --chown=signalrag:signalrag . .

USER signalrag

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["signalrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
