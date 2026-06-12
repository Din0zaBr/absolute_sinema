# Road Defect Engine — CV-движок анализа дорожных дефектов

Подсистема 1 платформы учёта дорожных дефектов. По фотографии (или серии) находит
дефекты дорожного покрытия, выделяет их контур/форму, при наличии эталона измеряет
размеры в сантиметрах и классифицирует серьёзность по **ГОСТ Р 50597-2017**.

> Подробная документация проекта (решения и их обоснования): [`docs/PROJECT.md`](docs/PROJECT.md)
> Сценарий презентации и демо: [`docs/DEMO.md`](docs/DEMO.md)
> Полный дизайн: [`docs/superpowers/specs/2026-06-10-road-defect-cv-engine-design.md`](docs/superpowers/specs/2026-06-10-road-defect-cv-engine-design.md)
> Журнал работ: [`docs/STATUS.md`](docs/STATUS.md)

## Что движок умеет честно (и чего не умеет)

| Возможность | Любое фото | + эталон в кадре | + серия кадров |
|-------------|:---------:|:----------------:|:--------------:|
| Детекция + класс дефекта | ✅ | ✅ | ✅ |
| Контур / форма (eccentricity, solidity) | ✅ | ✅ | ✅ |
| Диаметр / площадь в **см** | ❌ | ✅ ±10–20% | ✅ |
| **Глубина** в см | ❌ | ❌ (только бакет) | ✅ ГОСТ-грейд |

🔴 **Глубину по одному фото получить нельзя** — это физика монокулярного зрения, а не
ограничение реализации. Достоверная глубина — только из серии кадров (фотограмметрия)
или датчика. Движок никогда не выдаёт «сантиметры глубины», которых не может знать.

## Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# опционально (относительная глубина, флаг --depth):
pip install transformers
```

Сегментация формы MobileSAM работает из коробки — через встроенную реализацию
`ultralytics.models.sam`, веса (`mobile_sam.pt`, ~39 МБ, Apache-2.0) докачиваются
автоматически в `models/`. Официальный пакет `mobile_sam` ставить не обязательно;
если он установлен (`pip install git+https://github.com/ChaoningZhang/MobileSAM.git`),
движок использует его в первую очередь.

## Запуск

```powershell
# анализ папки с фото → outputs/*.json + *_annotated.jpg
$env:PYTHONPATH="src"; python -m road_defect.cli --input . --output outputs
#   флаги: --depth (относит. глубина), --ensemble (второй pothole-проход),
#          --conf, --road-category, --blob-ref (см. docs/PROJECT.md §7)

# смок-тест на демо-фото с диагностикой
$env:PYTHONPATH="src"; python scripts/smoke_test.py

# автономный HTML-отчёт для презентации
python scripts/make_demo_report.py --outputs outputs --originals .
```

## Тесты

```powershell
python -m pytest -q                  # 52 юнит-теста (без сети и моделей)
python scripts/smoke_test.py         # интеграционный смок (качает веса)
python scripts/validate_outputs.py   # инварианты JSON-отчётов после смока
```

## Структура

```
src/road_defect/
  config.py        # ГОСТ-константы, классы, реестр моделей
  pipeline.py      # оркестратор фото → отчёт
  detect.py        # детектор (YOLO, готовые веса)
  segment.py       # сегментация формы (трещины: black-hat в полном разрешении;
                   #   площадные: MobileSAM пакетом или через ultralytics; фолбэк GrabCut)
  shape.py         # дескрипторы формы из маски
  scale.py         # масштаб по люку (ГОСТ 3634, обод 646 мм): Hough+эллипс;
                   #   опц. путь по тёмному эллипсу (--blob-ref, см. STATUS) + гомография
  depth.py         # относительная глубина (опц.)
  severity.py      # серьёзность по ГОСТ Р 50597-2017
  report.py        # JSON-контракт + аннотированное изображение
  imgio.py         # чтение/запись изображений (юникод-пути Windows)
  cli.py           # командная строка
scripts/
  smoke_test.py        # смок на демо-фото      | validate_outputs.py  # инварианты JSON
  detect_sweep.py      # разбор пропусков       | scale_debug.py       # диагностика эталона
  make_demo_report.py  # HTML-отчёт для презентации
  prepare_rdd2022.py   # RDD2022 VOC→YOLO       | val_rdd2022.py       # mAP-стенд
  finetune_rdd2022.py  # дообучение от чекпоинта rezzzq
```

## Лицензии моделей (для продакшена)

`ultralytics` — **AGPL-3.0**: в распространяемом продукте инференс гнать через ONNX
(`onnxruntime`), без импорта `ultralytics`. MobileSAM и Depth Anything V2 **Small** —
Apache-2.0. Depth Anything Base/Large — CC-BY-NC (не использовать). Датасет RDD2022 —
CC BY-SA 4.0. Подробнее — §4 дизайна и `docs/PROJECT.md` §4.10.
