"""Диагностика детектора: все сырые детекции на фото при минимальном пороге.

Запуск:  python scripts/detect_sweep.py <фото> [мин_conf=0.01]

Помогает разбираться с пропусками: если дефект не виден даже при conf=0.01 —
порогом не лечится, нужен другой детектор/дообучение; если виден при 0.10–0.25 —
вопрос калибровки порога.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: detect_sweep.py <image> [min_conf]")
        return 1
    image_path = Path(sys.argv[1])
    min_conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

    from road_defect import config, imgio
    from road_defect.detect import Detector

    img = imgio.read_image(image_path)
    if img is None:
        print(f"Не удалось прочитать {image_path}")
        return 1

    cfg = config.InferenceConfig(det_conf=min_conf)
    det = Detector(cfg).load()
    print(f"Детектор: {det.active.name}, conf >= {min_conf}, кадр {img.shape[1]}x{img.shape[0]}")
    found = sorted(det.detect(img), key=lambda d: -d.confidence)
    if not found:
        print("Детекций нет вовсе — порогом не лечится.")
        return 0
    for d in found:
        x, y, w, h = (int(v) for v in d.bbox_xywh)
        print(f"  {d.cls_name:20} conf={d.confidence:.3f} bbox=({x},{y},{w},{h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
