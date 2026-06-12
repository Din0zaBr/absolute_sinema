# Road Defect Engine — CV-движок анализа дорожных дефектов

Подсистема 1 платформы учёта дорожных дефектов. По фотографии (или серии) находит
дефекты дорожного покрытия, выделяет их контур/форму, при наличии эталона измеряет
размеры в сантиметрах и классифицирует серьёзность по **ГОСТ Р 50597-2017**.

> Полный дизайн: [`docs/superpowers/specs/2026-06-10-road-defect-cv-engine-design.md`](docs/superpowers/specs/2026-06-10-road-defect-cv-engine-design.md)

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

# смок-тест на демо-фото с диагностикой
$env:PYTHONPATH="src"; python scripts/smoke_test.py
```

## Тесты

```powershell
pip install pytest
python -m pytest -q            # быстрые юнит-тесты (без моделей)
python -m pytest -m models     # интеграционные (качают веса)
```

## Структура

```
src/road_defect/
  config.py        # ГОСТ-константы, классы, реестр моделей
  pipeline.py      # оркестратор фото → отчёт
  detect.py        # детектор (YOLO, готовые веса)
  segment.py       # сегментация формы (MobileSAM: пакет или ultralytics / GrabCut-фолбэк)
  shape.py         # дескрипторы формы из маски
  scale.py         # масштаб по люку (ГОСТ 3634, обод крышки 646 мм) + гомография
  depth.py         # относительная глубина (опц.)
  severity.py      # серьёзность по ГОСТ Р 50597-2017
  report.py        # JSON-контракт + аннотированное изображение
  imgio.py         # чтение/запись изображений (юникод-пути Windows)
  cli.py           # командная строка
```

## Лицензии моделей (для продакшена)

`ultralytics` — **AGPL-3.0**: в распространяемом продукте инференс гнать через ONNX
(`onnxruntime`), без импорта `ultralytics`. MobileSAM и Depth Anything V2 **Small** —
Apache-2.0. Depth Anything Base/Large — CC-BY-NC (не использовать). Подробнее — §4 дизайна.
```
