# Перевод non-en → en (ArgoTranslate, CPU)

## Почему так
- Argo (CTranslate2) на CPU — GPU RTX 3050 целиком под bpipe, без конкуренции за VRAM.
- Отдельный сервис `translator`: падение/перегруз не трогает upipe/bpipe, upipe ходит по HTTP.

## Поведение
| TRANSLATE | en | non-en |
|---|---|---|
| `false` (дефолт) | pass-through | collector → `FOREIGN_FORWARD_URL`, ответ `filtered_lang` (как раньше) |
| `true` | pass-through, Argo не вызывается | пакет установлен + очередь не полна → Argo → `translation`≠raw; иначе → `FOREIGN_FORWARD_URL` |

- Причины ухода на foreign при `true` (счётчик `translate.foreign_reasons`): `translator_down`, `queue_full` (fill ≥ `TRANSLATE_BUSY_FILL`), `no_pack`, `low_confidence`, `too_long`.
- Если translator вернул busy/ошибку/таймаут уже внутри upipe — item тоже уходит на foreign (`upipe_fallback_to_foreign`).
- Argo вернул текст = raw → item дропается (`identical_dropped`), raw в translated не подставляется.
- Один item = один вызов Argo. `TRANSLATE_BATCH_SIZE` = макс. micro-batch за один забор воркера (группировка по языку, модель остаётся «горячей»); реального tensor-батчинга в API Argo нет.

## Конфиг (.env)
| Переменная | Дефолт | Смысл |
|---|---|---|
| `TRANSLATE` | `false` | вкл/выкл перевод |
| `TRANSLATE_QUEUE_SIZE` | `200` | очередь translator; полна → 503 → foreign |
| `TRANSLATE_BATCH_SIZE` | `8` | макс. micro-batch на воркера |
| `TRANSLATE_WORKERS` | `3` | параллельные переводы (потоки CPU) |
| `TRANSLATE_TIMEOUT_SECONDS` | `30` | таймаут одного перевода |
| `TRANSLATE_MAX_CHARS` | `2000` | длиннее → foreign (после перевода всё равно > `MAX_MODEL_TOKENS`) |
| `TRANSLATE_BUSY_FILL` | `0.9` | порог заполнения очереди, после которого collector шлёт на foreign |
| `ARGO_LANG_PACKS` | `es,pt,fr,de,it,ru,ja` | какие `X→en` пакеты ставить (коды Argos) |
| `TRANSLATOR_CPUS` / `TRANSLATOR_OMP_THREADS` | `4` / `2` | лимит CPU контейнера / потоки CT2 на перевод |
| `USE_PROXY_TRANSLATOR` | `true` | скачивание пакетов через VPN |

Пример i5-10400 (6c/12t, 16GB): `TRANSLATE_WORKERS=3`, `TRANSLATOR_OMP_THREADS=2`, `TRANSLATOR_CPUS=4`, `UPIPE_WORKERS=6`, `TRANSLATE_QUEUE_SIZE=200`, `TRANSLATE_BATCH_SIZE=8`. Каждый пакет ≈ 100–300MB RAM — 7 языков ≈ 1–2GB.

## Языковые пакеты
- Ставятся при старте translator из `ARGO_LANG_PACKS`, идемпотентно; кеш в `./argos_data` (переживает пересборку).
- Добавить язык: дописать код в `ARGO_LANG_PACKS` → `docker compose up -d translator`.
- Коды Argos: `ar az ca cs da de el eo es fa fi fr ga he hi hu id it ja ko nl pl pt ru sk sv th tr uk zh zt …`; langdetect→Argos: `zh-cn→zh`, `zh-tw→zt`, `no→nb`.

## Запуск
```bash
cp .env.example .env            # или допиши блок TRANSLATE_* в существующий .env
mkdir -p argos_data
./run.sh up -d --build          # run.sh нужен для BPIPE_REPLICAS; иначе: docker compose up -d --build
docker compose logs -f translator
```
- Первый старт с `TRANSLATE=true`: скачивание пакетов, до нескольких минут (healthcheck ждёт).
- Включить/выключить: правка `TRANSLATE` в `.env` → `./run.sh up -d` (пересоздаст translator/upipe/collector).

## Метрики: `GET http://localhost:9000/stats` (JSON, CORS открыт — под будущий HTML)
```bash
curl -s localhost:9000/stats | python3 -m json.tool
curl -s localhost:9000/stats | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['translate']['rate_per_sec'],d['languages']['seen_count'],d['bpipe']['queue'],d['exorde']['rate'])"
```
- `translate.rate_per_sec` — переводов/сек за скользящие 60с; `translate.queue/queue_max/fill`; `translate.translator_up`.
- `languages.seen_count` — уникальные non-en языки, замеченные collector с момента старта collector (сбрасывается при рестарте); `seen` — счётчик по языкам.
- `bpipe.queue/max/fill/inflight/dropped` — суммарно по репликам.
- `exorde.totals.{accepted,rejected,duplicates}` — суммы из ответов Upload API с момента старта transactioneer (`accepted` = `filtered_items`).
- `exorde.rate.{10m,30m,60m}.{accepted,rejected,duplicates}_per_sec` — среднее/сек по ответам API; если uptime меньше окна, делится на uptime (`window_s`).
- Прямые эндпоинты: `translator:8003/stats`, `transactioneer:8002/stats`; `/queue` collector сохранил формат + ключ `languages`.

## Проверка
```bash
# TRANSLATE=false: non-en не проходит
curl -s -X POST localhost:9000/store_item -H 'Content-Type: application/json' \
 -d '{"content":"Das Wetter ist heute sehr schön und die Sonne scheint über Berlin.","external_id":"t-de-1","created_at":"'$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'","domain":"twitter.com","url":"https://twitter.com/t/1"}'
# → {"message":"filtered_lang"}

# TRANSLATE=true: перевод напрямую
docker compose exec translator curl -s --noproxy '*' -X POST localhost:8003/translate \
 -H 'Content-Type: application/json' -d '{"lang":"de","text":"Das Wetter ist heute sehr schön."}'
docker compose exec translator curl -s --noproxy '*' -X POST localhost:8003/translate \
 -H 'Content-Type: application/json' -d '{"lang":"fr","text":"Il fait très beau aujourd hui à Paris."}'
# → {"translation":"<английский текст ≠ raw>","lang":"de"}

# через весь пайплайн (ответ OK, в логах upipe transl=N растёт)
# те же store_item для de/fr → {"message":"OK"}
docker compose logs upipe | grep transl=
```
- `test_pipeline.sh::test_non_english` ожидает `filtered_lang` — валиден только при `TRANSLATE=false`.
