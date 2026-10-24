FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    RESOLVEAI_DB_PATH=/app/data/resolveai.db

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[anthropic,openai]"

RUN mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "resolveai seed && resolveai serve --host 0.0.0.0 --port 8000"]
