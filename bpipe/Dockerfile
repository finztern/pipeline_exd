FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    curl ca-certificates gcc g++ git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# ВАЖНО: ВСЕ python-зависимости ставятся ОДНОЙ командой pip install, чтобы
# pip резолвил их как единый граф. Раньше torch ставился отдельным RUN'ом
# (с --index-url cu121), а весь остальной стек — вторым, независимым
# RUN'ом. Каждый RUN — отдельный вызов pip с чистым резолвером: второй
# RUN не видел constraint из первого и либо (а) молча апгрейдил torch до
# последней версии с обычного PyPI (2.14.0+cu130 — собран под CUDA новее
# драйвера хоста → CUDA недоступна), либо (б) при попытке зафиксировать
# torch==2.5.1+cu121 ЕЩЁ РАЗ упирался в конфликт версий nvidia-cudnn-cu12
# (torch 2.5.1+cu121 жёстко требует ==9.1.0.70, а мы параллельно пинили
# ==9.3.0.75 — ResolutionImpossible).
#
# Решение: --index-url на индекс PyTorch (там нужная cu121-сборка torch
# и её родная зависимость nvidia-cudnn-cu12==9.1.0.70) + --extra-index-url
# на обычный PyPI (для того, чего на индексе PyTorch нет — transformers,
# optimum, cupy и т.д.). Один резолвер, один согласованный граф. Версию
# nvidia-cudnn-cu12 здесь НЕ пинуем вручную — 9.1.0.70 (то, что тянет сам
# torch) — это тоже cuDNN 9.x, чего достаточно для onnxruntime-gpu==1.20.2.
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    torch==2.5.1 \
    cupy-cuda12x \
    "aiohttp>=3.9.0" \
    "orjson>=3.9.0" \
    "huggingface_hub==0.26.5" \
    "transformers==4.46.3" \
    "optimum==1.23.3" \
    "onnx>=1.16.0" \
    "onnxruntime-gpu==1.20.2" \
    "sentence-transformers==3.3.1" \
    "argostranslate>=1.9.0" \
    "vaderSentiment>=3.3.2" \
    "finvader>=1.0.0" \
    httpx[socks] socksio \
    "aioprometheus>=23.0.0" \
    opentelemetry-api

# Проверка на этапе сборки, что реально встал torch нужной эпохи —
# падает сразу при сборке образа, а не тихо в рантайме через месяц.
RUN python3 -c "import torch; assert torch.__version__.startswith('2.5.1'), f'torch версии {torch.__version__}, ожидался 2.5.1'; assert torch.version.cuda == '12.1', f'torch собран под CUDA {torch.version.cuda}, ожидался 12.1'; print(f'✅ torch {torch.__version__} (cuda {torch.version.cuda})')"

# ВАЖНО: optimum==1.23.3 (даже БЕЗ extras [onnxruntime-gpu]) тянет обычный
# CPU-пакет `onnxruntime` как скрытую зависимость. Пакеты `onnxruntime` и
# `onnxruntime-gpu` ставятся ФАЙЛАМИ В ОДНУ И ТУ ЖЕ папку site-packages —
# кто установился физически последним, тот и остаётся на диске, независимо
# от того, что оба формально числятся "установленными" в `pip list`.
# Поэтому переустанавливаем onnxruntime-gpu ПОСЛЕДНИМ шагом, принудительно
# (--force-reinstall), чтобы его файлы гарантированно победили CPU-версию.
# Без этого CUDAExecutionProvider тихо пропадает из available_providers.
#
# ПРИМЕЧАНИЕ: onnxscript/onnx_ir здесь больше не ставим. Они были нужны
# только для конвертации PyTorch → ONNX "на лету" при старте контейнера
# (для моделей без готового .onnx на HuggingFace) — этот путь оказался
# крайне хрупким к версиям и постоянно падал на разных этапах экспорта.
# Такие модели теперь грузятся напрямую через transformers.pipeline() на
# PyTorch (см. lab_initialization.py) — без ONNX-экспорта вообще, быстрее
# и без этого класса проблем. ONNX/ORT здесь остаётся только для
# sentence-transformer, у которого готовый .onnx уже есть на HuggingFace.
RUN pip3 install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu==1.20.2

# Наш базовый образ nvidia/cuda:...-runtime-... содержит только CUDA
# runtime (cudart/cublas), но НЕ cuDNN. onnxruntime-gpu>=1.19 требует
# cuDNN 9.x (libcudnn.so.9), иначе CUDAExecutionProvider тихо пропадает
# из списка доступных провайдеров, и всё уезжает на CPU. cuDNN здесь
# ставится pip-пакетом nvidia-cudnn-cu12 (транзитивная зависимость torch,
# см. выше), а не системно — его .so лежат в site-packages, вне
# стандартных путей линковщика. Основной способ найти их —
# onnxruntime.preload_dlls() в коде (lab_initialization.py);
# LD_LIBRARY_PATH ниже — резервный вариант на случай, если preload_dlls
# недоступен в конкретной версии onnxruntime. Путь проверяем на этапе
# сборки (падает сразу, если что-то пошло не так, а не тихо в рантайме).
RUN python3 -c "import nvidia.cudnn, os; p = os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'); assert os.path.isdir(p), f'{p} не существует'; print(f'✅ cuDNN lib найден: {p}')"
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

# exorde_data ставим БЕЗ рекурсивных сабмодулей: репозиторий тянет за собой
# десятки модулей-скраперов под случайными uuid-именами (reddit, bluesky,
# и т.д.), которые нам не нужны (используются только базовые dataclasses:
# Item, Translation, Classification...) — а `pip install git+...` по
# умолчанию делает `git submodule update --init --recursive`, из-за чего
# сборка может зависать намертво, если хоть один сабмодуль недоступен.
RUN git clone --branch full --single-branch --no-recurse-submodules --depth 1 \
    https://github.com/exorde-labs/exorde_data.git /tmp/exorde_data \
    && pip3 install --no-cache-dir /tmp/exorde_data \
    && rm -rf /tmp/exorde_data

COPY . .

CMD ["python", "bpipe.py"]
