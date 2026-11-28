FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    "aiohttp>=3.9.0" \
    "orjson>=3.9.0" \
    yake \
    yake-rust \
    "tiktoken>=0.5.0" \
    "langdetect>=1.0.9" \
    "aioprometheus>=23.0.0"

# exorde_data без рекурсивных сабмодулей — см. комментарий в bpipe/Dockerfile
RUN git clone --branch full --single-branch --no-recurse-submodules --depth 1 \
    https://github.com/exorde-labs/exorde_data.git /tmp/exorde_data \
    && pip install --no-cache-dir /tmp/exorde_data \
    && rm -rf /tmp/exorde_data

COPY . .

CMD ["python", "upipe.py"]
