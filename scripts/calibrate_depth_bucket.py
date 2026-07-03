"""Калибровка порогов бакета глубины (v0) на демо-фото (2026-07-03).

Запуск:  python scripts/calibrate_depth_bucket.py [папка_с_json] [--nulls N]

Что делает (стратегия из синтеза панели дизайна):
  1. Берёт готовые отчёты outputs/*.json (маски дефектов из mask_rle — конвейер
     не перезапускается), считает карту глубины на кадр (Depth Anything V2
     Small, CPU) и rel_norm «кольцевой плоскостной линейки» для каждого дефекта.
  2. НУЛЕВОЕ распределение без расхода разметки: каждая маска сдвигается на
     случайные БЕЗДЕФЕКТНЫЕ участки того же кадра → rel_norm «не-дефектов».
     Порог lo естественно взять выше 95-го перцентиля нулей.
  3. Печатает обе выборки и предлагаемые lo/hi (hi — середина наибольшего
     зазора реальных значений; при n=14 порог заведомо предварительный).

Честность: скрипт ничего не пишет в конфиг — пороги фиксируются руками в
src/road_defect/depth.py с пометкой v0 и датой.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_defect import imgio  # noqa: E402
from road_defect.depth import (  # noqa: E402
    RelativeDepth, mask_to_depth_grid, ring_plane_bucket)


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for run in rle["counts"]:
        if val:
            flat[pos:pos + run] = 1
        pos += run
        val ^= 1
    return flat.reshape((h, w), order="F").astype(bool)


def shifted_null_masks(mask: np.ndarray, forbidden: np.ndarray,
                       n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Копии маски, сдвинутые на случайные места кадра вне дефектов."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bh, bw = y1 - y0 + 1, x1 - x0 + 1
    patch = mask[y0:y1 + 1, x0:x1 + 1]
    H, W = mask.shape
    out: list[np.ndarray] = []
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        ty = int(rng.integers(0, H - bh + 1))
        tx = int(rng.integers(0, W - bw + 1))
        if forbidden[ty:ty + bh, tx:tx + bw][patch].any():
            continue                      # налез на настоящий дефект
        shifted = np.zeros_like(mask)
        shifted[ty:ty + bh, tx:tx + bw] = patch
        out.append(shifted)
    return out


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_dir = Path(args[0]) if args else ROOT / "outputs"
    n_nulls = 15
    for a in sys.argv[1:]:
        if a.startswith("--nulls"):
            n_nulls = int(a.split("=", 1)[1])

    reports = []
    seen_images = set()
    for p in sorted(out_dir.glob("*.json")):
        rep = json.loads(p.read_text(encoding="utf-8"))
        if rep["image"] in seen_images:   # pair-отчёт дублирует вид A
            continue
        seen_images.add(rep["image"])
        if rep["defects"]:
            reports.append(rep)
    if not reports:
        print(f"Нет отчётов с дефектами в {out_dir}")
        return 1

    depth_engine = RelativeDepth()
    rng = np.random.default_rng(20260703)
    real_rows, null_vals = [], []

    for rep in reports:
        img_path = ROOT / rep["image"]
        img = imgio.read_image(img_path)
        if img is None:
            print(f"[!] нет исходника {rep['image'][:40]} — пропуск")
            continue
        depth_engine._load()
        if not depth_engine.available:
            print("Depth Anything недоступна — калибровка невозможна")
            return 1
        # Рабочая сетка как в relative_bucket: длинная сторона ≤ 1024, маски
        # уменьшаются к карте (см. depth.py, 2026-07-03).
        import cv2
        dmap = depth_engine._frame_depth(img).astype(np.float64)
        hd, wd = dmap.shape
        s = 1024.0 / max(hd, wd)
        if s < 1.0:
            dmap = cv2.resize(dmap, (int(round(wd * s)), int(round(hd * s))),
                              interpolation=cv2.INTER_AREA)
        masks = [mask_to_depth_grid(rle_decode(d["mask_rle"]), dmap.shape)
                 for d in rep["defects"]]
        forbidden = np.zeros(dmap.shape, bool)
        for m in masks:
            forbidden |= m
        forbidden = cv2.dilate(forbidden.astype(np.uint8),
                               np.ones((21, 21), np.uint8), 1).astype(bool)

        for d, m in zip(rep["defects"], masks):
            if not m.any():
                real_rows.append((rep["image"][:18], d["class"], None, None,
                                  None, "mask_below_depth_resolution"))
                continue
            r = ring_plane_bucket(dmap, m)
            real_rows.append((rep["image"][:18], d["class"],
                              r.get("depth_bucket"), r.get("_rel_norm"),
                              r.get("_drop"), r.get("method")))
            for nm in shifted_null_masks(m, forbidden, n_nulls, rng):
                nr = ring_plane_bucket(dmap, nm)
                if nr.get("_rel_norm") is not None:
                    null_vals.append(nr["_rel_norm"])

    print("=" * 78)
    print(f"{'кадр':18} {'класс':20} {'бакет':8} {'rel_norm':>8} {'drop':>8}  method")
    print("-" * 78)
    for row in real_rows:
        img8, cls, b, rn, dr, meth = row
        print(f"{img8:18} {cls:20} {str(b):8} "
              f"{('%.3f' % rn) if rn is not None else '  -':>8} "
              f"{('%.4f' % dr) if dr is not None else '  -':>8}  {meth}")
    print("-" * 78)
    reals = sorted(r[3] for r in real_rows if r[3] is not None)
    nulls = sorted(null_vals)
    print(f"реальных rel_norm: {len(reals)} -> {['%.3f' % v for v in reals]}")
    if nulls:
        p95 = float(np.percentile(nulls, 95))
        print(f"нулевых rel_norm: {len(nulls)}, медиана "
              f"{np.median(nulls):.4f}, p95 {p95:.4f}, max {max(nulls):.4f}")
        print(f"предложение lo: >= {p95:.3f} (95-й перцентиль нулей)")
    pos = [v for v in reals if v > 0]
    if len(pos) >= 3:
        gaps = [(pos[i + 1] - pos[i], (pos[i + 1] + pos[i]) / 2)
                for i in range(len(pos) - 1)]
        gap, mid = max(gaps)
        print(f"наибольший зазор реальных: {gap:.3f} вокруг {mid:.3f} "
              f"(кандидат hi; n мало — решать с головой)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
