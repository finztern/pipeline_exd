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

## GPU-переводчик на втором ПК (опционально)
- Папка `translator_gpu/` запускается на втором ПК: `cd translator_gpu && cp .env.example .env && docker compose up -d --build`.
- В `.env` второго ПК: `UPIPE_URL=http://<IP основного>:5981/`. В `.env` основного: `GPU_TRANSLATOR_URL=http://<IP второго>:8004/`.
- Item уходит на GPU при `queue_full` (список причин: `GPU_ROUTE_REASONS`); de/fr/es/it/pt → Opus-MT, остальное → NLLB-600M; результат возвращается в upipe с полем `pretranslated`.
- Второй ПК недоступен → `GPU_RETRY_SECONDS` (2с) item не отправляются туда, уходят на `FOREIGN_FORWARD_URL` (если задан) или отбрасываются.

## Обратное давление в translator_gpu (GPU_BUCKET_CAP)

Воркер переводчика держит тексты в двух местах: `asyncio.Queue` (`GPU_QUEUE_SIZE`) и внутренние `buckets` (по моделям).
Раньше `drain()` перекладывал **всё** из очереди в `buckets` без лимита: очередь оставалась пустой, 503 не отдавался,
а хвост рос в RAM без ограничений. Теперь в `buckets` держится не больше `GPU_BUCKET_CAP` текстов (по умолчанию 400, минимум `2 × GPU_BATCH_SIZE`).
Остальное остаётся в очереди; когда она заполнена, переводчик отвечает 503 (`queue_full`), и collector/upipe отправляют item дальше
(`FOREIGN_FORWARD_URL` или отбрасывание) на `GPU_RETRY_SECONDS` (по умолчанию 2 с).

Максимальный хвост = `GPU_QUEUE_SIZE + GPU_BUCKET_CAP`. В `/stats` и `/health` есть поле `backlog` (очередь + buckets),
в `/stats` ещё `bucket_items` и `bucket_cap`. Если `backlog` постоянно упирается в потолок, а `rejected_busy` растёт, GPU не успевает за входом.

## Детектор языка collector: LANG_BACKEND

`LANG_BACKEND=langdetect` (по умолчанию, поведение без изменений) | `shadow` | `hybrid` (рекомендуется) | `fasttext`.

- `langdetect` — как раньше (`LANGDETECT_TRIALS` проб).
- `shadow` — решения по-прежнему принимает langdetect, но fastText (lite, `lid.176.ftz` внутри пакета `fast-langdetect`, офлайн)
  считается параллельно. Расхождения и скорость видны в `GET :9000/stats` → `lang_detect`
  (`agree_pct`, `ld_en_ft_other`, `ld_other_ft_en`, `top_disagreements`, `ft_prob_hist`, `avg_us`); 1 из 100 расхождений пишется в лог (`🧪 lang-shadow`).
- `hybrid` — fastText решает, если его уверенность >= `LANG_FT_TRUST` (0.5), иначе решает langdetect (как раньше). На реальных примерах fastText при >=0.5
  был прав, ниже 0.5 путает короткие тексты (индонезийский принимает за английский), а на тексте без букв (эмодзи) возвращает `en`: поэтому чистый `fasttext` не рекомендуется.
  Каждый `LANG_AUDIT_EVERY`-й вызов дополнительно считается langdetect, расхождения копятся в `/stats` → `lang_detect` (по аудит-выборке).
  Уверенность fastText приводится к шкале langdetect (`_ft_prob_to_ld_scale`), поэтому пороги `LANG_CONFIDENCE_THRESHOLD` / `LANG_LOW_CONF_MIN` работают как раньше.
- `fasttext` — решает fastText. У него своя калибровка уверенности, поэтому свои пороги: `LANG_FT_CONFIDENCE` (0.80) и
  `LANG_FT_LOW_CONF_MIN` (0.40); подбирай по `ft_prob_hist` из shadow. Если fastText недоступен, collector откатывается на langdetect.
  Китайский: fastText даёт только `zh`, вариант `zh-cn`/`zh-tw` определяется по набору иероглифов (`collector/lang_ft.py`).

`python collector/bench_lang.py --n 3000 [--texts файл.txt]` сравнивает скорость и совпадение языка (в т.ч. fastText).

## Лимит CPU-потоков bpipe: BPIPE_CPU_THREADS

torch, tokenizers (rayon) и ONNX Runtime по умолчанию заводят пулы на все логические ядра, а ORT ещё и крутится вхолостую.
`BPIPE_CPU_THREADS` (по умолчанию 2 на реплику) выставляет `OMP/MKL/OPENBLAS/RAYON_NUM_THREADS`, `torch.set_num_threads`
и `SessionOptions` ORT (`intra_op_num_threads`, `allow_spinning=0`). Инференс идёт на GPU. Откат: `BPIPE_CPU_THREADS=8`.

### Очистка текста (LANG_CLEAN) и тексты без языка

Перед определением языка из текста убираются `‹‹media›› ... ‹‹/media››`, теги вроде `<image>`, URL, упоминания и `#`
(сам текст в item не меняется). Без этого, например, `https://i.redd.it/...` определялся как итальянский (домен `.it`).
Включена по умолчанию для `hybrid`/`fasttext` (`LANG_CLEAN=auto`), для `langdetect`/`shadow` выключена, чтобы их поведение не менялось.

Если после очистки букв почти нет (< `LANG_NEUTRAL_MIN_LETTERS`), а в исходнике были ссылки/маркеры, это «текст без языка»
(`/stats` → `lang_neutral`). `LANG_NEUTRAL_ACTION=drop` (по умолчанию): такие элементы не отправляются на GPU. Раньше они доходили
до переводчика, где результат совпадал с исходником и отбрасывался как `identical`, то есть итог тот же, но без траты GPU.
`LANG_NEUTRAL_ACTION=pass`: пропускать их дальше как английские.

Заметка: переводчик отбрасывает результат, совпавший с исходником (`identical`, кроме маршрута `mul`). Английский текст,
ошибочно принятый за чужой, поэтому в части случаев терялся; hybrid такие тексты отправляет дальше как английские.
