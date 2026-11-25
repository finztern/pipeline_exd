FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    "aiohttp>=3.9.0" \
    "orjson>=3.9.0" \
    "langdetect>=1.0.9" \
    "fast-langdetect>=1.2.0"

COPY lang_detect.py lang_ft.py collector.py .

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

CMD ["python", "collector.py"]
