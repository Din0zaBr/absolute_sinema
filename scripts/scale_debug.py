"""Диагностика масштабной привязки: на какой стадии отваливается эталон-люк.

Запуск:  python scripts/scale_debug.py <фото> [<фото> ...]

Печатает: кандидаты Hough с вердиктами подтверждения, тёмные эллипс-блобы
(путь для косых видов) и итог scale_from_manhole.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def debug_one(image_path: Path) -> None:
    from road_defect import imgio
    from road_defect import scale as scale_mod

    print(f"\n=== {image_path.name[:40]} ===")
    img = imgio.read_image(image_path)
    if img is None:
        print("  не прочиталось")
        return
    H, W = img.shape[:2]
    print(f"  кадр {W}x{H}")

    candidates = scale_mod.detect_manhole_circles(img)
    print(f"  [1] Hough: кандидатов {len(candidates)} (по силе отклика):")
    for i, (ccx, ccy, cr) in enumerate(candidates):
        e = scale_mod.fit_manhole_ellipse(img, (ccx, ccy, cr))
        if e is None:
            verdict = "эллипс не подобран"
        else:
            ok = scale_mod._ellipse_confirms_circle(e, (ccx, ccy, cr))
            (ecx, ecy), (MA, ma), _ = e
            verdict = (f"эллипс MA={MA:.0f} ma={ma:.0f} "
                       f"d={np.hypot(ecx - ccx, ecy - ccy):.0f} -> "
                       + ("ПОДТВЕРЖДЁН" if ok else "не подтверждён"))
        print(f"      #{i}: центр=({ccx},{ccy}) r={cr}  {verdict}")

    blobs = scale_mod.detect_manhole_ellipses(img)
    print(f"  [2] тёмные эллипс-блобы: {len(blobs)}")
    for (ecx, ecy), (MA, ma), ang in blobs[:5]:
        print(f"      центр=({ecx:.0f},{ecy:.0f}) MA={MA:.0f} ma={ma:.0f} "
              f"aspect={min(MA, ma) / max(MA, ma):.2f} angle={ang:.0f}")

    ref = scale_mod.scale_from_manhole(img)
    print(f"  [3] итог: available={ref.available} method={ref.method} "
          f"mm_per_px={ref.mm_per_px} conf={ref.confidence}")
    print(f"      note='{ref.note}'")


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: scale_debug.py <image> [...]")
        return 1
    for p in paths:
        debug_one(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
