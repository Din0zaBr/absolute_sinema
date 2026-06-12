"""Валидация детектора на размеченном наборе (§11: mAP на RDD2022 и т.п.).

Запуск:  python scripts/val_rdd2022.py [data.yaml] [веса]
         (по умолчанию: datasets/RDD2022_Czech_yolo/data.yaml и активные веса rezzzq)

⚠ Честная оговорка по Czech: rezzzq обучалась на ПОЛНОМ RDD2022, включая
Czech train, а val-срез prepare_rdd2022.py взят из этого же train. Поэтому
метрики на нём оптимистичны (train-contaminated) и годятся как:
  • проверка корректности конвертации/маппинга классов (ошибка обрушила бы mAP);
  • smoke валидационного стенда.
Для честной оценки обобщения нужен набор, которого модель не видела
(локальная разметка РФ) — тот же скрипт, другой data.yaml.

Базовый замер 2026-06-12 (Czech val 275 кадров, imgsz 640, CPU):
  all  P=0.957 R=0.947 mAP50=0.966 mAP50-95=0.716
  D00=0.988  D10=0.959  D20=0.995  D40=0.924 (mAP50)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from road_defect import config
    from ultralytics import YOLO

    data = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "datasets" / "RDD2022_Czech_yolo" / "data.yaml")
    weights = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        config.MODELS_DIR / "yolo12s_RDD2022_best.pt")
    if not data.exists():
        print(f"Нет data.yaml: {data}\n"
              "Сначала: python scripts/prepare_rdd2022.py datasets/RDD2022_Czech.zip")
        return 1
    if not weights.exists():
        print(f"Нет весов: {weights} (запусти smoke_test.py — скачает)")
        return 1

    print(f"Веса: {weights}\nДанные: {data}")
    model = YOLO(str(weights))
    model.val(data=str(data), imgsz=640, device="cpu", workers=0,
              plots=False, verbose=True,
              project=str(ROOT / "runs_train"), name="val", exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
