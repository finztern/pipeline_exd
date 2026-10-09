FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# transformers>=4.43 убрал M2M100Encoder.embed_scale — конвертер ctranslate2 4.4.0 на нём падает (NLLB)
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.5.1 \
    "numpy<2" \
    "transformers==4.41.2" \
    "tokenizers<0.20" \
    sentencepiece sacremoses \
    "ctranslate2==4.4.0" \
    "aiohttp>=3.9.0"

RUN python3 -c "import transformers; from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder as E; import inspect; assert transformers.__version__=='4.41.2', transformers.__version__; assert 'self.embed_scale' in inspect.getsource(E.__init__); print('ok', transformers.__version__)"

COPY translator_gpu.py .

HEALTHCHECK --interval=15s --timeout=5s --retries=120 --start-period=120s \
    CMD curl -sf http://localhost:8004/health || exit 1

CMD ["python", "translator_gpu.py"]
