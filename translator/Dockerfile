FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch вставлен явно: иначе stanza (зависимость argostranslate) утянет CUDA-сборку на гигабайты
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.5.1 \
    "argostranslate>=1.9.6" \
    sentencepiece \
    "aiohttp>=3.9.0"

COPY translator.py .

HEALTHCHECK --interval=10s --timeout=5s --retries=90 --start-period=120s \
    CMD curl -sf --noproxy '*' http://localhost:8003/health || exit 1

CMD ["python", "translator.py"]
