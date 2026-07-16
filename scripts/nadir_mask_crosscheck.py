"""Кросс-чек семантики маски и высотного масштаба по НАДИРНЫМ кадрам «Эталоны».

Вопрос (docs/HEIGHT_SCALE_RESULTS.md, шаг 1): ширины из высотного масштаба
систематически больше рулеточного span (+37% медиана) — «маска шире рулетки»
или «масштаб врёт»? Надирный кадр ruler_1 (лента ПОПЕРЁК ямы) содержит
собственную линейку: см-риски ленты дают px/см БЕЗ всякой геометрии камеры.
Меряем на надирном кадре ту же сущность, что и высотный масштаб на обзорном, —
МАСКУ детектора+SAM — и сравниваем сантиметры двух независимых источников:
  * совпало (±15–20%) → масштаб чист, перебор против span = семантика маски;
  * не совпало → систематика высотного масштаба, искать в геометрии.

Алгоритм px/см — прототип шагов 1–2 ридера «рейка+рулетка»
(docs/RULER_READER_PLAN.md): HSV-жёлтая вытянутая компонента → выпрямление →
период тёмных рисок по автокорреляции профиля. Гейты честного отказа:
`no_tape` / `tape_not_elongated` / `ticks_aperiodic` / `too_few_ticks`.
Вариация периода по третям ленты — индикатор перспективного дрейфа масштаба
(кадр «почти надир», не идеальный надир).

Запуск (venv; при недоступном HF Hub — $env:HF_HUB_OFFLINE="1" и
$env:TRANSFORMERS_OFFLINE="1", веса в кэше):
  .\\.venv\\Scripts\\python.exe scripts\\nadir_mask_crosscheck.py [--pits 001 028] [--out outputs_height_scale\\nadir]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_defect import imgio  # noqa: E402
from road_defect.pipeline import DefectPipeline  # noqa: E402

# ямы с чистой (не общей) сценой и оценкой высотного масштаба на обзорном виде
DEFAULT_PITS = ("001", "002", "005", "006", "018", "028")
GT_DIR = Path("datasets/gt_photos")

HSV_LO, HSV_HI = (12, 80, 60), (45, 255, 255)   # жёлтая лента, терпимо к тени
MIN_TAPE_AREA_PX = 15000
MIN_ELONGATION = 4.0
# период см-рисок в px ограничен шириной ленты (10–30 мм): ppcm = W/(1.0..3.0)
BAND_LO_W, BAND_HI_W = 3.0, 1.0
MIN_TICKS = 8
MAX_PERIOD_VARIATION = 0.15


def gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    r = int(max(3, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="reflect"), k, mode="same")[r:-r]


def detect_tape(img: np.ndarray):
    """Крупнейшая вытянутая жёлтая компонента → minAreaRect. (rect|None, reason)."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, HSV_LO, HSV_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    best, best_area = None, 0
    had_big = False           # была ли хоть одна КОМПОНЕНТА достаточной площади
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_TAPE_AREA_PX:
            continue
        had_big = True
        if area <= best_area:
            continue
        ys, xs = np.nonzero(lab == i)
        rect = cv2.minAreaRect(
            np.column_stack([xs, ys]).astype(np.float32))
        lo, hi = sorted(rect[1])
        if lo <= 0 or hi / lo < MIN_ELONGATION:
            continue
        best, best_area = rect, area
    if best is None:
        # причина — по крупнейшей КОМПОНЕНТЕ, не по сумме жёлтых px (ревью
        # 2026-07-16: рассыпанная на фрагменты лента давала мислейбл no_tape)
        return None, ("tape_not_elongated" if had_big else "no_tape")
    return best, None


def rectify_strip(img: np.ndarray, rect):
    """Повернуть кадр так, чтобы лента стала вертикальной; вернуть полосу ленты.

    Возвращает (strip BGR, ppcm-ось = вертикаль), ширина полосы = короткая
    сторона rect."""
    import cv2

    (cx, cy), (w, h), ang = rect
    phi = ang if w >= h else ang + 90.0      # угол ДЛИННОЙ оси от x
    L, W = (max(w, h), min(w, h))
    M = cv2.getRotationMatrix2D((cx, cy), phi - 90.0, 1.0)
    hh, ww = img.shape[:2]
    rot = cv2.warpAffine(img, M, (ww, hh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    x0 = int(round(cx - W / 2)); x1 = int(round(cx + W / 2))
    y0 = int(round(cy - L / 2 + 0.05 * L)); y1 = int(round(cy + L / 2 - 0.05 * L))
    x0, y0 = max(x0, 0), max(y0, 0)
    strip = rot[y0:min(y1, hh), x0:min(x1, ww)]
    if strip.size == 0 or strip.shape[0] < strip.shape[1]:
        return None
    return strip


def tick_scale(strip: np.ndarray):
    """px/см по автокорреляции профиля тёмности вдоль выпрямленной ленты.

    Возвращает (dict|None, reason): ppcm, вариация периода по третям, число
    рисок."""
    import cv2

    W = strip.shape[1]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float64)
    prof = 255.0 - gray[:, int(0.2 * W):max(int(0.8 * W), int(0.2 * W) + 1)].mean(axis=1)
    band_lo, band_hi = W / BAND_LO_W, W / BAND_HI_W
    if len(prof) < 4 * band_hi:
        return None, "too_few_ticks"
    dp = prof - gaussian_smooth_1d(prof, band_hi)

    def period_of(x, lo, hi):
        x = x - x.mean()
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        if ac[0] <= 0:
            return None
        ac /= ac[0]
        lo_i, hi_i = int(max(2, lo)), int(min(len(ac) - 2, hi))
        if hi_i <= lo_i + 2:
            return None
        i = lo_i + int(np.argmax(ac[lo_i:hi_i]))
        y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]      # параболическое уточнение
        denom = (y0 - 2 * y1 + y2)
        return i + (0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0)

    p0 = period_of(dp, band_lo, band_hi)
    if p0 is None:
        return None, "ticks_aperiodic"
    n_ticks = len(dp) / p0
    if n_ticks < MIN_TICKS:
        return None, "too_few_ticks"
    thirds = np.array_split(dp, 3)
    ps = [period_of(t, 0.8 * p0, 1.25 * p0) for t in thirds if len(t) > 3 * p0]
    ps = [p for p in ps if p is not None]
    if len(ps) < 2:
        # трети короче 3 периодов — лента слишком коротка для проверки
        # стабильности, это дефицит рисок, а не апериодичность (ревью 2026-07-16)
        return None, "too_few_ticks"
    variation = (max(ps) - min(ps)) / float(np.median(ps))
    if variation > MAX_PERIOD_VARIATION:
        return None, "ticks_aperiodic"
    return {"ppcm": float(p0), "period_variation": round(float(variation), 3),
            "n_ticks": round(float(n_ticks), 1),
            "thirds_ppcm": [round(float(p), 2) for p in ps]}, None


def pothole_mask_extents(pipe: DefectPipeline, path: Path, ppcm: float):
    """Маски детектора+SAM на надирном кадре → размеры в см (все pothole)."""
    import cv2

    report, _ov, masks = pipe.analyze_image(path)
    out = []
    for d, m in zip(report["defects"], masks):
        if d.get("class") != "pothole":
            continue
        mm = np.asarray(m) > 0
        if not mm.any():
            continue
        ys, xs = np.nonzero(mm)
        rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        hi, lo = sorted(rect[1], reverse=True)
        out.append({"conf": round(float(d.get("confidence", 0.0)), 3),
                    "feret_max_cm": round(hi / ppcm, 1),
                    "feret_min_cm": round(lo / ppcm, 1),
                    "area_cm2": round(float(mm.sum()) / ppcm ** 2, 0),
                    "_mask": mm})
    out.sort(key=lambda r: r["area_cm2"], reverse=True)
    return out


def draw_overlay(img, rect, ppcm, det, out_path):
    """Лента (зелёный бокс), 10-см сетка по осям ленты, контуры масок (красный)."""
    import cv2

    ov = img.copy()
    (cx, cy), (w, h), ang = rect
    phi = np.radians(ang if w >= h else ang + 90.0)
    u = np.array([np.cos(phi), np.sin(phi)])          # вдоль ленты
    v = np.array([-u[1], u[0]])
    c = np.array([cx, cy])
    step = 10.0 * ppcm
    ext = 6000.0
    for k in range(-30, 31):
        for axis, other in ((u, v), (v, u)):
            p0 = c + k * step * other - ext * axis
            p1 = c + k * step * other + ext * axis
            col = (0, 200, 0) if k % 5 else (0, 255, 255)
            cv2.line(ov, tuple(np.int32(p0)), tuple(np.int32(p1)), col,
                     2 if k % 5 else 3)
    box = cv2.boxPoints(rect)
    cv2.polylines(ov, [np.int32(box).reshape(-1, 1, 2)], True, (255, 0, 255), 6)
    for r in det:
        cnts, _ = cv2.findContours(r["_mask"].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 0, 255), 8)
    txt = f"grid 10cm  ppcm={ppcm:.1f}"
    cv2.putText(ov, txt, (24, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 10,
                cv2.LINE_AA)
    cv2.putText(ov, txt, (24, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                (255, 255, 255), 4, cv2.LINE_AA)
    s = 1500.0 / max(ov.shape[:2])
    ov = cv2.resize(ov, (int(round(ov.shape[1] * s)), int(round(ov.shape[0] * s))))
    imgio.write_image(out_path, ov)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pits", nargs="*", default=list(DEFAULT_PITS))
    ap.add_argument("--out", default="outputs_height_scale/nadir")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = DefectPipeline(use_depth=False)
    rows = []
    for pit in args.pits:
        path = GT_DIR / f"{pit}_ruler_1.jpg"
        row = {"pit": pit, "file": str(path)}
        if not path.exists():
            row["outcome"] = "no_file"
            rows.append(row)
            continue
        img = imgio.read_image(path)
        rect, err = detect_tape(img)
        if rect is None:
            row["outcome"] = err
            rows.append(row)
            print(f"[{pit}] {err}", flush=True)
            continue
        strip = rectify_strip(img, rect)
        if strip is None:
            row["outcome"] = "tape_not_elongated"
            rows.append(row)
            print(f"[{pit}] rectify failed", flush=True)
            continue
        scale, err = tick_scale(strip)
        if scale is None:
            row["outcome"] = err
            rows.append(row)
            print(f"[{pit}] {err}", flush=True)
            continue
        row.update(scale)
        row["tape_width_px"] = round(float(min(rect[1])), 1)
        row["tape_width_cm"] = round(float(min(rect[1])) / scale["ppcm"], 2)
        det = pothole_mask_extents(pipe, path, scale["ppcm"])
        row["masks"] = [{k: v for k, v in r.items() if k != "_mask"} for r in det]
        row["outcome"] = "ok" if det else "ok_no_detection"
        draw_overlay(img, rect, scale["ppcm"], det, out_dir / f"{pit}_nadir.jpg")
        rows.append(row)
        print(f"[{pit}] ppcm={scale['ppcm']:.1f} (вариация {scale['period_variation']}, "
              f"рисок {scale['n_ticks']}, лента {row['tape_width_cm']} см) "
              f"масок: {len(det)} {[ (r['feret_max_cm'], r['feret_min_cm']) for r in det ]}",
              flush=True)

    (out_dir / "nadir_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # сравнение с высотным масштабом (обзорные виды)
    summ_path = Path("outputs_height_scale/summary.json")
    if summ_path.exists():
        per_pit = json.loads(summ_path.read_text(encoding="utf-8"))["per_pit"]
        print("\n| Яма | span GT | надир: маска (Д×Ш, см) | обзор: маска (Д×Ш, см) |")
        print("|---|---|---|---|")
        for row in rows:
            if not row.get("masks"):
                continue
            m0 = row["masks"][0]
            d = per_pit.get(row["pit"], {})
            obl = [f"{v['feret_max']:.0f}×{v['feret_min']:.0f}"
                   for v in d.get("views", {}).values() if v.get("outcome") == "ok"]
            print(f"| {row['pit']} | {d.get('span_cm')} "
                  f"| {m0['feret_max_cm']}×{m0['feret_min_cm']} | {', '.join(obl) or '—'} |")


if __name__ == "__main__":
    main()
