"""Дообучение детектора дефектов от чекпоинта rezzzq (дизайн §3: запасной путь).

Честная оценка ресурсов: полноценное дообучение на RDD2022 Czech (~3.5k фото)
— часы на GPU или сутки на CPU. Этот скрипт даёт готовую инфраструктуру:
  • --make-toy N  — синтетический мини-датасет (тёмные «ямы» на сером фоне),
    чтобы за минуты проверить, что тренировочный цикл работает end-to-end;
  • --data path/to/data.yaml — реальное дообучение на твоём датасете.

Где взять данные:
  RDD2022 (CC-BY-SA-4.0, коммерчески чистые подмножества Czech/Norway):
  https://github.com/sekilab/RoadDamageDetector — конвертировать в YOLO-формат
  (xml VOC -> txt) или взять готовые YOLO-зеркала на Roboflow Universe
  («RDD2022»). Классы ДОЛЖНЫ совпадать с моделью rezzzq:
  0=D00, 1=D10, 2=D20, 3=D40, 4=Repair (порядок проверь по data.yaml зеркала!).

После обучения лучшие веса копируются в models/finetuned_best.pt —
config.DETECTOR_CANDIDATES подхватит их автоматически при следующем запуске
(кандидат local-finetuned стоит первым в реестре).

Примеры:
  python scripts/finetune_rdd2022.py --make-toy 24 --epochs 2 --imgsz 320
  python scripts/finetune_rdd2022.py --data D:/rdd2022/data.yaml --epochs 30 --imgsz 640
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOY_DIR = ROOT / "datasets" / "toy_potholes"
CLASS_NAMES = ["D00", "D10", "D20", "D40", "Repair"]  # порядок как у rezzzq


def make_toy_dataset(n: int) -> Path:
    """Синтетика для смок-проверки цикла обучения: «ямы» (D40) на асфальте."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(42)
    for split, count in (("train", n), ("val", max(4, n // 4))):
        img_dir = TOY_DIR / "images" / split
        lbl_dir = TOY_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            size = 320
            img = rng.integers(115, 150, (size, size, 3), dtype=np.uint8)
            labels = []
            for _ in range(int(rng.integers(1, 4))):
                cx, cy = rng.integers(60, size - 60, 2)
                ax, ay = rng.integers(18, 45, 2)
                shade = int(rng.integers(30, 70))
                cv2.ellipse(img, (int(cx), int(cy)), (int(ax), int(ay)),
                            float(rng.integers(0, 180)), 0, 360,
                            (shade, shade, shade), -1)
                # YOLO-формат: class cx cy w h (нормированные)
                labels.append(f"3 {cx/size:.4f} {cy/size:.4f} "
                              f"{2*ax/size:.4f} {2*ay/size:.4f}")
            cv2.imwrite(str(img_dir / f"toy_{i:03}.jpg"), img)
            (lbl_dir / f"toy_{i:03}.txt").write_text("\n".join(labels),
                                                     encoding="utf-8")
    yaml_path = TOY_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {TOY_DIR.as_posix()}\n"
        "train: images/train\nval: images/val\n"
        f"names: {dict(enumerate(CLASS_NAMES))}\n",
        encoding="utf-8")
    print(f"Игрушечный датасет: {TOY_DIR} ({n} train)")
    return yaml_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Дообучение детектора дефектов.")
    ap.add_argument("--data", default="", help="data.yaml датасета (YOLO-формат)")
    ap.add_argument("--make-toy", type=int, default=0, metavar="N",
                    help="сгенерировать синтетический мини-датасет на N картинок "
                         "и обучиться на нём (смок-проверка цикла)")
    ap.add_argument("--base", default="", help="базовые веса (по умолчанию rezzzq из models/)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--freeze", type=int, default=10,
                    help="заморозить первые N слоёв (стабильнее на малых данных)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-install", action="store_true",
                    help="не копировать веса в models/finetuned_best.pt")
    args = ap.parse_args()

    from road_defect import config
    from ultralytics import YOLO

    if args.make_toy:
        data_yaml = make_toy_dataset(args.make_toy)
    elif args.data:
        data_yaml = Path(args.data)
        if not data_yaml.exists():
            print(f"Нет такого data.yaml: {data_yaml}")
            return 1
    else:
        print("Укажи --data path/to/data.yaml или --make-toy N (см. --help).")
        return 1

    base = args.base or str(config.MODELS_DIR / "yolo12s_RDD2022_best.pt")
    if not Path(base).exists():
        print(f"Базовые веса не найдены: {base}\n"
              "Запусти сначала smoke_test.py (он скачает веса rezzzq).")
        return 1

    print(f"База: {base}\nДанные: {data_yaml}\n"
          f"epochs={args.epochs} imgsz={args.imgsz} freeze={args.freeze} device={args.device}")
    model = YOLO(base)
    results = model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
        device=args.device, freeze=args.freeze, batch=4, workers=0,
        project=str(ROOT / "runs_train"), name="finetune", exist_ok=True,
        verbose=True, plots=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nЛучшие веса: {best}")
    if best.exists() and not args.no_install:
        dst = config.MODELS_DIR / "finetuned_best.pt"
        shutil.copy2(best, dst)
        print(f"Установлены: {dst}\n"
              "Реестр config.DETECTOR_CANDIDATES подхватит их автоматически "
              "(кандидат local-finetuned). Чтобы вернуться к rezzzq — удали файл.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
