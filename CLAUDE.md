# Road Defect Engine — ориентир для агента

Читай этот файл вместо разведки по репозиторию. Подробности — в указанных доках,
открывай их точечно под задачу, а не целиком.

## Что это
Офлайн-CV-движок учёта дорожных дефектов по фото (яма/трещины/заплатка, ГОСТ Р
50597-2017). Пайплайн: **YOLO-детекция → класс-зависимая сегментация → дескрипторы
формы → масштаб по эталону (если подтверждён) → относительная глубина → вердикт
ГОСТ → JSON + размеченное фото**. CPU, ~0.2–5 с/фото. Это «подсистема 1 из 5»
платформы; подсистемы 2–5 (SQLite-БД, веб-UI с загрузкой фото/картой/уведомлениями,
GPS) — MVP в `platform/` (цикл 16), интерфейс на stdlib http.server + Jinja2.

## Главный принцип (не нарушать)
**Честность измерений: система никогда не выдаёт числа, которых физически не может
знать.** Нет источника масштаба → нет сантиметров; СЕРТИФИЦИРУЕМЫЕ размеры — только
от подтверждённого эталона, ОЦЕНОЧНЫЕ (высота камеры, `--camera-height`) — только с
меткой `metric.scale_source`, confidence=low и полосой (не для актирования; ТЗ §0/A+).
ИЗМЕРЕННАЯ глубина в см по одному фото невозможна (физика монокуляра) → `depth_cm`
всегда null; выдаются бакет «мелкая/средняя/глубокая» и — при источнике масштаба в
кадре — явно помеченная ОЦЕНКА `metric.depth_cm_estimate` (rel_norm × поперечная
ширина; полоса ошибки, confidence всегда low, certifiable=false — цикл 16).
Ложный масштаб хуже отсутствия масштаба. Рискованные эталоны (разметка/бордюр)
выключены по умолчанию, выдаются только low-confidence.

## Как запускать (Windows, PowerShell основной; venv уже собран)
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m road_defect.cli --input <папка> --output <папка> [--depth --ensemble --pair --camera-height 1.70 [--pitch-prior 11.6]]
.\.venv\Scripts\python.exe -m road_defect.platform.webapp   # веб-UI: http://127.0.0.1:8000/
#   [--data platform_data --import-outputs <папка> --no-model]
.\.venv\Scripts\python.exe -m pytest -q          # 379 тестов, держать зелёными
.\.venv\Scripts\python.exe scripts\smoke_test.py # прогреть модели (кэш весов)
```
Пороги/константы — только через `config.py` (`InferenceConfig`,
`TwoViewDepthConfig`), не хардкодить.
Ключевые дефолты: `imgsz=1024`, `det_conf=0.25`, `det_iou=0.45`, ансамбль ВЫКЛ.

## Карта модулей (`src/road_defect/`)
- `config.py` — все пороги, `InferenceConfig`, `DETECTOR_CANDIDATES`, `DEFECT_CLASSES`, параметры ГОСТ/эталонов. Начинай отсюда.
- `detect.py` — YOLO-детекция, второй pothole-проход (ансамбль), `merge_detections`.
- `segment.py` — маски: ямы — MobileSAM; трещины — классическая экстракция тонких структур в полном разрешении (нейросеть на 1024px не видит трещину 2px).
- `depth.py` — Depth Anything V2 Small, относительные бакеты глубины +
  `estimate_depth_cm` (честная см-ОЦЕНКА при источнике масштаба — эталон или
  высота камеры; ниже шума — верхняя граница).
- `photogrammetry.py` — two-view см-глубина по паре калиброванных бортовых
  видов (перед/за), прототип v0: НЕ в pipeline, валидирован на синтетике
  (`scripts/bench_two_view_depth.py` 21/21) — см. `docs/VEHICLE_CAPTURE.md` §3.1.
- `scale.py` — эталон (люк ГОСТ 3634 / разметка / бордюр) и перевод в см с кросс-проверкой.
- `plane_scale.py` — fallback-масштаб от ВЫСОТЫ КАМЕРЫ (кадры без эталона,
  ВЫКЛ по умолчанию, CLI `--camera-height`): горизонт по карте глубины
  («якорь неба») или приор тангажа (`--pitch-prior`) → лучевая проекция маски
  на полотно → Д/Ш/площадь + mm/px для оценки глубины. Всегда ОЦЕНКА:
  `metric.scale_source="camera_height"`, confidence=low, certifiable=false,
  с полосой; Д/Ш = размер МАСКИ (с тёмным ореолом), не рулеточного
  «разрушения» — валидация и границы в `docs/HEIGHT_SCALE_RESULTS.md`.
- `shape.py` — дескрипторы формы (эксцентриситет, solidity, …).
- `severity.py` — вердикт ГОСТ Р 50597.
- `fusion.py` — режим `--pair`: `analyze_pair`, выбор доминирующей ямы, обратно-дисперсионное слияние. ВАЖНО: НЕ делает re-ID по внешнему виду (см. память).
- `pipeline.py` — оркестрация пообразного прохода. `cli.py` — точка входа.
- `report.py` — запись JSON + размеченного фото. `imgio.py` — I/O картинок.
- `platform/` — подсистемы 2–5 (цикл 16): `db.py` (SQLite; JSON целиком = source
  of truth + плоская проекция дефектов; статус учёта new/in_progress/fixed,
  живёт в reports, переживает повторный импорт, миграция старых БД в init_db),
  `webapp.py` (stdlib http.server + Jinja2: загрузка фото → анализ → отчёт;
  дашборд с фильтром по статусу/карта Leaflet/уведомления; смена статуса и
  удаление отчёта; экспорт реестра `/export.csv` в utf-8-sig; traversal-защита
  /media), `notify.py` (правила ГОСТ/глубокая яма), `geo.py` (EXIF GPS + журнал;
  journal.csv читать в utf-8-sig — там BOM). Без новых pip-зависимостей.
- ⚠️ **Кириллица в путях** повсюду — читать/писать изображения ТОЛЬКО через `imgio.py` (imdecode/imencode), `cv2.imread` на кириллице падает.

## Данные
- Исходники: `Проэкт/Проэкт/Ямки/Яма N/{Front,Back,Эталоны}` (48 ям).
- Ингест пар: `datasets/local_pairs/NNN/{front,back}.jpg` (`scripts/ingest_local_pairs.py`);
  байтовые пары-дубли не копируются (общий кадр соседних ям): 48 ям → 28 уникальных пар.
- Оверлеи пар: `outputs_pairs/`. Полнокадровые оверлеи ЧАСТО несут ложные боксы.

## Скрипты (`scripts/`)
Диагностика: `detect_sweep.py` (разбор пропусков conf-свипом), `scale_debug.py`,
`validate_outputs.py` (инварианты). Данные/обучение: `prepare_rdd2022.py`,
`prepare_local_finetune.py` (исключает held-out eval-сцены), `finetune_rdd2022.py`,
`val_rdd2022.py`, `calibrate_depth_bucket.py`, `build_measure_form.py`,
`pair_quality_table.py`. Замер качества детекции (P0): `make_eval_set.py`,
`build_eval_annotator.py`, `eval_detection.py`, `import_annotations.py`
(конвертер внешней разметки: X-AnyLabeling/YOLO/COCO → annotations.json) —
протокол в `docs/EVAL.md`.
Демо: `make_demo_deck.py` (слайд-дек), `make_demo_report.py` (таблица),
`make_report_pptx.py` (PPTX-презентация из detection_report.html: фото из его
base64, схемы — рендер его же SVG; без новых pip-зависимостей),
`demo_two_view.py` (контролируемое см-демо на синтетике).
Стенд two-view глубины: `synth_road3d.py` (3D-рендер) + `bench_two_view_depth.py`
(21 кейс, гейтит photogrammetry.py).

## Процесс работы (принятый в проекте)
Работа циклами: коммиты `Цикл N: …` (по-русски), в конце цикла — **адверсариальное
пре-коммит ревью** независимыми агентами, которое ГЕЙТИТ коммит (жди финального
отчёта, не коммить раньше). Не коммить/пушить без явной просьбы пользователя.

## Не затирать (разные артефакты!)
- `docs/detection_report.html` — демо детекции для РУКОВОДСТВА (одностраничный
  прокручиваемый; сам HTML автономный = source of truth). Генераторы — снимок в
  `scripts/report_gen/` (build_final.py манифест + build_report.py движок), заточены
  под scratchpad-сессию, см. `scripts/report_gen/README.md`.
- `docs/report.html`, `docs/presentation.html`, `outputs_demo/demo_report.html` —
  отдельные материалы пользователя, НЕ перезаписывать.

## Куда смотреть глубже (по задаче)
- Архитектура/решения — `docs/PROJECT.md`. Статус/история циклов — `docs/STATUS.md`.
- Датасет/разметка — `docs/DATASET.md`. Сценарий показа — `docs/DEMO.md`.
- Работа с парами — `docs/PAIR_WORKFLOW.md`.
- Бортовая съёмка с машины (пути к глубине, риг-эталон, two-view глубина §3.1,
  архитектура сервиса, план пилота) — `docs/VEHICLE_CAPTURE.md`.
- Замер качества детекции (eval-набор, разметка, метрики) — `docs/EVAL.md`;
  пошаговая инструкция разметчику-новичку — `docs/ANNOTATION_GUIDE.md`;
  разметка в X-AnyLabeling (выбран) — `docs/XANYLABELING_GUIDE.md`.
- **Backlog улучшений детекции и разбор ложных срабатываний — `docs/IMPROVEMENTS.md`.**
- Аудит багов 2026-07-07 — `docs/HANDOFF_FABLE.md` (ЗАКРЫТ циклами 12–14,
  оставлен как история; текущий план — верх `docs/STATUS.md` + `IMPROVEMENTS.md`).
- Кросс-сессионные факты и находки — в памяти (`MEMORY.md` + файлы memory/):
  реальность сопоставления пар и честные оговорки по детекциям.
