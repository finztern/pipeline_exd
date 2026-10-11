# Exorde Pipeline — сборка под Debian 13 + RTX 3050 8GB

Схема:
```
(твои источники данных) → collector:9000 → upipe:5981 → bpipe:7995 → transactioneer:8002 → Exorde
                                                                            │
                                                                            ▼
                                                                       vpn (Xray/REALITY)
```

Что внутри и почему именно так — коротко:
- **collector** — принимает тексты, фильтрует язык/дубликаты/мусор, режет экстремально большие пейлоады.
- **upipe** — переводит, извлекает keywords, **отбрасывает item, если реальное число токенов превышает лимит** (чтобы GPU не падал), классифицирует.
- **bpipe** — тяжёлый ML: эмбеддинги, sentiment, emotion, zero-shot classification и т.д. на GPU (ONNX Runtime).
- **transactioneer** — грузит готовые батчи в Exorde Upload API, **через VPN**.
- **vpn** — Xray-core (VLESS + REALITY + Vision, с поддержкой пост-квантовой верификации), поднимает SOCKS5 (1080) и HTTP (1081) прокси внутри docker-сети.

Вторая GPU-реплика (`bpipe_2`) отключена — на 8GB VRAM две копии всех моделей одновременно рискуют упасть в CUDA OOM. Она есть в `docker-compose.yml`, закомментирована, с инструкцией как включить, если сменишь видеокарту.

---

## 1. Установка на чистый Debian 13

```bash
sudo apt update && sudo apt install -y ca-certificates curl gnupg git

# ─── Docker + Compose plugin ───────────────────────────────────
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER
newgrp docker

# ─── NVIDIA-драйвер (если ещё не стоит) ────────────────────────
sudo apt install -y nvidia-driver firmware-misc-nonfree
sudo reboot
```

После перезагрузки:
```bash
nvidia-smi   # должен показать RTX 3050
```

```bash
# ─── NVIDIA Container Toolkit (чтобы Docker видел GPU) ─────────
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Проверка:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
Если вывелась таблица с картой — всё ок.

---

## 2. Распаковка и настройка

```bash
mkdir -p ~/exorde-pipeline && cd ~/exorde-pipeline
unzip exorde-pipeline.zip
```

Открой `.env` и проверь:
- `MAIN_ADDRESS` — твой Exorde-адрес. В файле уже стоит значение из твоего исходника — перепроверь, что это правильный адрес.
- `VLESS_URL` — уже вставлена твоя ссылка, ничего менять не нужно.

Остальное в `.env` уже настроено под RTX 3050 8GB.

---

## 3. Запуск

Первый запуск — долгий (~10-20 минут): собираются образы и качаются ML-модели с HuggingFace.

```bash
docker compose up -d --build
docker compose logs -f       # Ctrl+C чтобы выйти из просмотра — контейнеры продолжат работать
```

Дождись, пока все станут healthy:
```bash
docker compose ps
```
Должно быть 5 сервисов (`vpn`, `transactioneer`, `bpipe`, `upipe`, `collector`) со статусом `Up ... (healthy)`.

---

## 4. Проверка — все случаи

### 4.1. VPN поднялся и реально выходит в интернет через REALITY

```bash
docker compose logs vpn | tail -20
# должна быть строка "✅ Конфиг Xray собран: ... pq=да"

curl -x socks5h://127.0.0.1:1080 https://api.ipify.org
# должен вернуть IP сервера 2.27.50.152, а не IP твоей машины
```

Если curl зависает / connection refused / healthcheck vpn не проходит:
```bash
docker compose logs vpn
```
Попробуй отключить пост-квантовую верификацию (известный баг части версий Xray-core с mldsa65) — в `.env`:
```
VPN_DISABLE_PQV=true
```
и пересобери:
```bash
docker compose up -d --build vpn
```

### 4.2. Все сервисы отвечают на /health

```bash
curl -sf http://localhost:9000/health   # collector
docker compose exec upipe curl -sf http://localhost:5981/health
docker compose exec bpipe curl -sf http://localhost:7995/health
docker compose exec transactioneer curl -sf http://localhost:8002/health
```

### 4.3. transactioneer реально ходит через VPN

```bash
docker compose logs transactioneer | grep -i proxy
```
Если раньше Upload API был недоступен без VPN, а теперь батчи уходят без ошибок таймаута/connection refused — прокси работает.

### 4.4. Нормальный текст проходит весь пайплайн

```bash
curl -X POST http://localhost:9000/store_item \
  -H "Content-Type: application/json" \
  -d '{
    "content": "This is a completely normal test tweet about the weather being sunny today in California.",
    "external_id": "test-normal-001",
    "created_at": "'"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"'",
    "domain": "twitter.com",
    "url": "https://twitter.com/test/status/1"
  }'
```
Ожидаемый ответ: `{"message": "OK"}`. Через несколько секунд в `docker compose logs -f bpipe` появится строка про обработанный батч, а в `docker compose logs -f transactioneer` — про отправленный.

### 4.5. Слишком короткий текст — отсеивается на collector, не доходя до GPU

```bash
curl -X POST http://localhost:9000/store_item \
  -H "Content-Type: application/json" \
  -d '{"content": "short", "external_id": "test-short-001", "created_at": "'"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"'", "domain": "twitter.com", "url": "https://twitter.com/test/status/2"}'
```
Ожидаемый ответ: `{"message": "skipped_short"}`. В `docker compose logs bpipe` ничего нового появиться не должно.

### 4.6. Экстремально длинный текст — не должен ронять батч (главная проверка твоего требования)

Токен-лимит (`MAX_MODEL_TOKENS`, по умолчанию 400) больше не обрезает текст — item, превышающий лимит, **отбрасывается целиком** на этапе upipe, до GPU:

```bash
LONG_TEXT=$(python3 -c "print(('This is a very long repeated sentence about markets and finance and technology. ' * 400))")
curl -X POST http://localhost:9000/store_item \
  -H "Content-Type: application/json" \
  -d "{\"content\": \"$LONG_TEXT\", \"external_id\": \"test-long-001\", \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\", \"domain\": \"twitter.com\", \"url\": \"https://twitter.com/test/status/3\"}"
```
```bash
docker compose logs -f upipe | grep -i "Токен-лимит превышен"
```
Должна появиться строка вида `⚠️ [N] Ошибка обработки: Токен-лимит превышен (XXXX > 400), item отброшен`. Это ожидаемое поведение — не баг.

```bash
docker compose logs -f bpipe | grep -i error
```
Не должно быть `CUDA OOM`, `RuntimeError`, `IndexError` — длинный текст до bpipe вообще не доходит, батч не может из-за него упасть.

### 4.7. Пустой / мусорный текст после перевода

```bash
curl -X POST http://localhost:9000/store_item \
  -H "Content-Type: application/json" \
  -d '{"content": "!!!!!!!!!! ################ 1234567890", "external_id": "test-junk-001", "created_at": "'"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"'", "domain": "twitter.com", "url": "https://twitter.com/test/status/4"}'
```
В `docker compose logs upipe` должна быть ошибка `No content to work with` — элемент корректно отбрасывается, а не роняет сервис.

### 4.8. GPU-память под нагрузкой (проверка, что 8GB хватает)

```bash
watch -n 1 nvidia-smi
```
Прогони 4.4-4.6 несколько раз подряд, смотри на `Memory-Usage` — не должно вплотную упираться в 8192MiB. Если упирается — снижай `FIXED_BATCH_SIZE` в `.env` (например до 4) и `docker compose up -d bpipe`.

---

### 4.9. Приоритетные очереди bpipe

Порядок набора батча: `translated` → `default` → `buffer`. Item без перевода и без buffer-признака идёт в `default`.
- `translated`: `translation.language` не `en`/пусто (или `"translated": true`). Не отбрасывается до жёсткого лимита `BPIPE_QUEUE_TRANSLATED_HARD_MAX`; `BPIPE_QUEUE_TRANSLATED_MAX` мягкий (warning). upipe до `UPIPE_TRANSLATED_RETRIES` раз повторяет отправку при 503.
- `buffer`: `priority`/`queue` == `"buffer"` на верхнем уровне JSON или `domain` из `BPIPE_BUFFER_DOMAINS` (с точкой — точное/суффиксное совпадение, без точки — подстрока). Если item одновременно translated и buffer — он translated.
- Переполнение `default`/`buffer`: reject 503 (drop newest, старые не теряются). Buffer-item старше `BPIPE_BUFFER_TTL_SECONDS` удаляется при выборке (0 = выкл).
- `MAX_QUEUE_SIZE` deprecated: используется только как fallback для `BPIPE_QUEUE_DEFAULT_MAX`, если тот не задан; в compose больше не пробрасывается.
- `/queue`: `queue/max/fill` = translated+default (контракт collector), buffer — только в `queues.buffer`.

```bash
curl -s localhost:7995/queue | python3 -m json.tool   # внутри сети: docker compose exec bpipe curl -s localhost:7995/queue
```

---

## 5. Повседневные команды

```bash
docker compose logs -f              # все логи разом
docker compose logs -f bpipe        # логи конкретного сервиса
docker compose restart bpipe        # перезапуск одного сервиса
docker compose down                 # остановить всё
docker compose up -d --build        # пересобрать после правок кода
docker compose down -v              # остановить + стереть кеш моделей (models_cache)
```

## 6. Если что-то не так

- **bpipe долго не становится healthy** — нормально при первом запуске: скачивание и конвертация моделей в ONNX может занять 10-15 минут. Смотри `docker compose logs -f bpipe`.
- **CUDA out of memory** — снизь `FIXED_BATCH_SIZE` в `.env` (сейчас 6, можно до 3-4).
- **transactioneer не может отправить батч** — проверь VPN (пункт 4.1), проверь `MAIN_ADDRESS`.
- **HuggingFace не скачивается (заблокирован)** — поставь `USE_PROXY_BPIPE=true` и `USE_PROXY_UPIPE=true` в `.env`, `docker compose up -d --build bpipe upipe`.
