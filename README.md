# Storage Violation Detection API

FastAPI-сервис для обнаружения нарушений хранения объектов на изображениях камер.

Сервис:
* принимает кадры с камер
* сравнивает их с ideal image
* выделяет появившиеся объекты
* возвращает bounding boxes нарушений

Модель основана на change detection segmentation.

## Быстрый запуск (Docker)
1. Склонировать проект
```
git clone https://github.com/s70c3/storage_violation.git
cd storage_violation
```
2. Скачать [веса](https://disk.yandex.ru/d/0VSHkaiwtgwjZQ) и поместить в папку weights. 
3. Запустить сервис
```
docker compose up --build
```
3. Swagger UI:
```
http://localhost:8000/docs
```

## Workflow использования API

Типичный порядок работы:

1. загрузить ideal image через post запрос `/ideal` c параметром "camera_id" (любой идентификатор для сохранения состояния)
2. выставить параметры детекции через запрос `/params` (threshold_hits (количество детекций подряд для признание объекта оставленным, min_side - минимальный размер объекта в пиксель по короткой стороне ))
2. отправлять кадры по ручке `/process_frame`
3. получать детекции в координатах

## Тестовый скрипт

Файл `process_video.py` содержит тестовый скрипт, который позволяет проверить на видео. 

``` python3 process_video.py --ideal-image data/ofis_big.png --video data/ofis_big.mp4```

## Локальный запуск (без API)

Скрипт `run_local.py` запускает тот же пайплайн напрямую (без FastAPI).

- Прогон одного изображения:

```bash
python3 run_local.py \
  --ideal-image data/ideal.png \
  --frame data/frame.jpg \
  --out-image out.jpg \
  --out-json out.json
```

- Прогон видео:

```bash
python3 run_local.py \
  --ideal-image data/ideal.png \
  --video data/video1.avi \
  --out-video out.mp4 \
  --out-jsonl out.jsonl
```

### RTSP + показ в окне OpenCV

```bash
python3 run_local.py \
  --rtsp "rtsp://user:pass@host:554/stream" \
  --ideal-mode median \
  --show \
  --out-video out.mp4
```

- **Выход**: нажмите `q` или `Esc` в окне.

### Фон (идеальное изображение) без загрузки ideal

Можно не передавать `--ideal-image`, а брать фон автоматически:

- **`--ideal-mode first_frame`**: фон = первый кадр (быстро, но хрупко к шуму/людям в кадре).
- **`--ideal-mode mean`**: фон = среднее по последним N кадрам (`--ideal-frames`).
- **`--ideal-mode median`**: фон = медиана по последним N кадрам (устойчивее к выбросам).

Через API те же параметры: `POST /params` с `ideal_mode` и `ideal_frames`.

Пример:

```bash
python3 run_local.py \
  --ideal-mode mean \
  --ideal-frames 25 \
  --bg-median-update-every 5 \
  --video data/video1.avi \
  --out-video out.mp4
```

## Демо через API (без клиента)

В `./data` (в контейнере монтируется в `/app/data`; путь задан константой `DATA_DIR` в `source/demo_pipeline.py`) должны лежать файлы для выбранного пресета:

| `preset` | ideal | video |
|----------|--------|--------|
| **1** (по умолчанию) | `ideal.png` | `video1.avi` |
| **2** | `ofis_big.png` | `ofis_big.mp4` |
| **3** (офис small) | `ofis_small.png` | `ofis_small.mp4` |
| **4** | `ideal2.png` | `video2.avi` |

Поднимите сервис (`docker compose up --build`), затем:

```bash
# первые 10 секунд, пресет 1
curl -fL "http://localhost:28001/demo/process_data_video?duration_sec=10&preset=1" -o demo_output.mp4

# офис small (preset 3), весь файл
curl -fL "http://localhost:28001/demo/process_data_video?duration_sec=-1&preset=3" -o demo_small.mp4

# свои имена файлов в каталоге демо (без путей, только basename)
curl -fG "http://localhost:28001/demo/process_data_video_by_names" \
  --data-urlencode "ideal_name=ofis_small.png" \
  --data-urlencode "video_name=ofis_small.mp4" \
  --data-urlencode "duration_sec=10" \
  -o demo_custom.mp4
```

- **`duration_sec`**: положительное число — секунд с начала; **`-1`** — весь ролик до EOF (без лимита по длине).
- **`preset`**: `1` … `4` — см. таблицу.
- **`/demo/process_data_video_by_names`**: параметры `ideal_name` и `video_name` — любые файлы из каталога демо (буквы, цифры, `.` `_` `-`).
