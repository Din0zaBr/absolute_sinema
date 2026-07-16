"""Эксперимент «масштаб от высоты камеры»: H = 1.70 м (ответы пользователя
2026-07-16: 170 см — ВЫСОТА ОБЪЕКТИВА; один человек, в полный рост, все 48 ям;
калибровочного кадра нет → f_px из EXIF с допуском ±10%).

Идея: известная высота камеры + f_px из EXIF + наклон, оцененный ПО КАДРУ,
задают метрику плоскости полотна → см для длины/ширины ямы без эталона в кадре,
и mm/px для честной ОЦЕНКИ глубины через существующий estimate_depth_cm
(rel_norm x поперечная ширина). Наклон: карта Depth Anything аффинно-инвариантна
(d = k/Z + s), но у НЕБА 1/Z = 0 → медиана неба даёт s, а робастная плоскость
дороги — градиент; их пересечение — линия горизонта → n = K^T * l (нормаль
полотна в камере) → лучевое проецирование контура маски на плоскость.

ЧЕСТНОСТЬ (главный принцип проекта):
  * Это ЭКСПЕРИМЕНТ вне пайплайна (как two_view_real_pairs.py): в продукт —
    только после валидации против span_cm (надирная поперечная рулетка,
    datasets/gt_depth_readings.json — слой span параллаксом НЕ затронут).
  * ТЗ v2 §2.1 запрещает «EXIF + H» как источник СЕРТИФИЦИРУЕМЫХ чисел —
    поэтому все числа здесь: certifiable=false, confidence=low, с полосой.
  * depth_cm остаётся null; глубина — только оценка/верхняя граница из
    estimate_depth_cm. Полевой факт: ямы набора 1–3 см (DEPTH_RESULTS.md).
  * Гейты честного отказа: нет ямы / неоднозначные кандидаты / нет неба /
    развал плоскости / горизонт в зоне дороги / крен / яма у горизонта.
  * Fallback --prior-pitch-deg: для кадров БЕЗ неба (дворы) геометрия строится
    из ПРИОРА тангажа (медиана sky-anchored видов этого же набора), roll=0.
    Это приор, а не измерение по кадру: метка horizon_method='prior_pitch',
    полоса масштаба расширена на чувствительность к ±band (по умолчанию ±3°).
  * Общие сцены (один кадр у нескольких ям) исключаются из головной метрики:
    сопоставление «детекция ↔ конкретная яма GT» там неоднозначно (память
    проекта: matched != re-ID). Сцены берутся из journal.csv (колонка scene);
    md5-дубли кадров — только fallback: после дедупа пар в ingest (2026-07-16)
    дублей в local_pairs нет.

Запуск (venv; при недоступном HF Hub ставьте $env:HF_HUB_OFFLINE="1" и
$env:TRANSFORMERS_OFFLINE="1" — веса в кэше, иначе минуты ретраев):
  .\\.venv\\Scripts\\python.exe scripts\\height_scale_experiment.py --selftest
  .\\.venv\\Scripts\\python.exe scripts\\height_scale_experiment.py [--pairs 006 012] [--out outputs_height_scale]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_defect import imgio  # noqa: E402
from road_defect.depth import RelativeDepth, estimate_depth_cm, mask_to_depth_grid  # noqa: E402
from road_defect.pipeline import DefectPipeline, _select_dominant_pothole  # noqa: E402

# --- калибровка из EXIF и ответов пользователя (2026-07-16) -------------------
F_PX = 3025.0        # realme 13+ 5G, 4.84 мм / 1.6 мкм биннинг (как two_view_real_pairs)
F_BAND_PCT = 10.0    # кадр 4096x1844 — кроп «во весь экран» от 4:3; без
                     # калибровочного кадра пересчёт f_px честно ±10%
H_M = 1.70           # ВЫСОТА ОБЪЕКТИВА над полотном (ответ пользователя)
H_BAND_PCT = 3.0     # один человек в полный рост: покачивание рук/микрорельеф

# --- оценка горизонта по карте глубины: пороги и гейты ------------------------
GRID_LONG_SIDE = 1024        # рабочая сетка карты (аспект — строго от кадра)
SKY_TOP_FRAC = 0.45          # небо ищем только в верхней доле кадра
SKY_BAND_FRAC = 0.05         # «небо» = значения в нижних 5% диапазона карты
SKY_MIN_PX = 1500            # минимальная поддержка неба, px сетки
ROAD_ROW_LO_FRAC = 0.50      # опора плоскости — строки [0.50..0.97] высоты
ROAD_ROW_HI_FRAC = 0.97
PLANE_MAX_SUBSAMPLE = 40000
PLANE_MIN_SUPPORT = 3000
PLANE_REWEIGHT_ITERS = 3
HORIZON_SYS_GRID_PX = 15.0   # систематика якоря неба (~1° при f_grid ~ 760)
PITCH_MIN_DEG, PITCH_MAX_DEG = 4.0, 55.0
ROLL_MAX_DEG = 8.0
PIT_MIN_BELOW_HORIZON_DEG = 2.0
MAX_GROUND_DIST_M = 12.0
CONTOUR_MAX_PTS = 3000

GT_PATH = Path("datasets/gt_depth_readings.json")


# --- геометрия ----------------------------------------------------------------

def my_depth_grid(rd: RelativeDepth, img: np.ndarray):
    """Карта глубины кадра в сетке с аспектом СТРОГО от исходного кадра.

    Модель могла вернуть сетку другого аспекта/размера — приводим сами
    (значения гладкие, интерполяция не искажает; та же логика, что в depth.py).
    Возвращает (карта float64, k_to_orig: px исходного кадра на px сетки).
    """
    import cv2

    raw = rd._frame_depth(img)
    h0, w0 = img.shape[:2]
    target_long = min(GRID_LONG_SIDE, max(raw.shape))
    s = target_long / max(h0, w0)
    gw, gh = max(2, int(round(w0 * s))), max(2, int(round(h0 * s)))
    interp = cv2.INTER_AREA if raw.shape[1] >= gw else cv2.INTER_LINEAR
    d = cv2.resize(raw.astype(np.float32), (gw, gh), interpolation=interp)
    return d.astype(np.float64), float(w0) / gw


def fit_road_plane(d: np.ndarray, excl: np.ndarray):
    """Робастная плоскость d ~ gx*x + gy*y + c по нижней полосе кадра.

    Возвращает (dict | None, reason | None): coef, sigma (MAD-σ остатков),
    n_inliers, cov (3x3 ковариация коэффициентов).
    """
    rr, cc = d.shape
    lo, hi = int(ROAD_ROW_LO_FRAC * rr), int(ROAD_ROW_HI_FRAC * rr)
    sup = np.zeros_like(d, dtype=bool)
    sup[lo:hi] = True
    sup &= ~excl
    ys, xs = np.nonzero(sup)
    if len(xs) < PLANE_MIN_SUPPORT:
        return None, "plane_support_too_small"
    if len(xs) > PLANE_MAX_SUBSAMPLE:
        idx = np.random.default_rng(42).choice(len(xs), PLANE_MAX_SUBSAMPLE,
                                               replace=False)
        xs, ys = xs[idx], ys[idx]
    z = d[ys, xs].astype(np.float64)
    A = np.column_stack([xs.astype(np.float64), ys.astype(np.float64),
                         np.ones(len(xs))])
    keep = np.ones(len(xs), dtype=bool)
    coef = None
    for _ in range(PLANE_REWEIGHT_ITERS):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = z - A @ coef
        med = float(np.median(res[keep]))
        mad = float(np.median(np.abs(res[keep] - med))) + 1e-12
        keep = np.abs(res - med) <= 3.0 * 1.4826 * mad
        if int(keep.sum()) < PLANE_MIN_SUPPORT // 2:
            return None, "plane_fit_degenerate"
    res_in = z[keep] - A[keep] @ coef
    sigma = 1.4826 * float(np.median(np.abs(res_in - np.median(res_in)))) + 1e-12
    try:
        cov = (sigma ** 2) * np.linalg.inv(A[keep].T @ A[keep])
    except np.linalg.LinAlgError:
        return None, "plane_fit_degenerate"
    return {"coef": coef, "sigma": sigma, "n_inliers": int(keep.sum()),
            "cov": cov}, None


def find_sky(d: np.ndarray):
    """Кандидаты неба: значения в нижних SKY_BAND_FRAC диапазона карты, только
    в верхней доле кадра. Возвращает (медиана | None, число px, маска)."""
    rr = d.shape[0]
    top = d[: int(SKY_TOP_FRAC * rr)]
    d_lo, d_hi = np.percentile(d, [1.0, 99.0])
    thr = d_lo + SKY_BAND_FRAC * (d_hi - d_lo)
    m = top <= thr
    n = int(m.sum())
    if n < SKY_MIN_PX:
        return None, n, m
    return float(np.median(top[m])), n, m


def normal_from_horizon(l_orig: np.ndarray, w0: int, h0: int, f_px: float):
    """Нормаль полотна в камере из линии горизонта: n ∝ K^T l, знак — «вверх»
    (n·луч < 0 для пикселя дороги внизу кадра)."""
    cx, cy = w0 / 2.0, h0 / 2.0
    n = np.array([f_px * l_orig[0], f_px * l_orig[1],
                  cx * l_orig[0] + cy * l_orig[1] + l_orig[2]], dtype=np.float64)
    nn = float(np.linalg.norm(n))
    if nn <= 0:
        return None
    n /= nn
    p_road = np.array([0.0, (0.9 * h0 - cy) / f_px, 1.0])
    if float(n @ p_road) > 0:
        n = -n
    return n


def project_to_ground(pts_uv: np.ndarray, n: np.ndarray, w0: int, h0: int,
                      f_px: float, h_m: float):
    """Лучевое проецирование пикселей на плоскость n·P = -H.

    Возвращает (P (N,3) в метрах (координаты камеры), sin_beta (N) — синус
    угла луча ниже горизонта, valid (N))."""
    cx, cy = w0 / 2.0, h0 / 2.0
    p = np.column_stack([(pts_uv[:, 0] - cx) / f_px,
                         (pts_uv[:, 1] - cy) / f_px,
                         np.ones(len(pts_uv))])
    denom = p @ n
    norms = np.linalg.norm(p, axis=1)
    sin_beta = -denom / norms
    valid = denom < -1e-9
    t = np.where(valid, -h_m / np.where(valid, denom, -1.0), np.nan)
    P = p * t[:, None]
    return P, sin_beta, valid


def ground_frame(n: np.ndarray):
    """Ортонормальный базис в плоскости полотна (для 2D-метрики футпринта)."""
    e1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def footprint_metrics(contour_uv: np.ndarray, n: np.ndarray, w0: int, h0: int,
                      f_px: float, h_m: float, t_hat_img: np.ndarray):
    """Метрика футпринта ямы на полотне: фереты (minAreaRect), площадь,
    поперечная ширина вдоль t_hat_img (перпендикуляр к градиенту плоскости).

    Возвращает (dict | None, reason | None)."""
    import cv2

    P, sin_beta, valid = project_to_ground(contour_uv, n, w0, h0, f_px, h_m)
    if not bool(valid.all()):
        return None, "pit_touches_horizon"
    beta_min = float(np.degrees(np.arcsin(np.clip(sin_beta.min(), -1, 1))))
    if beta_min < PIT_MIN_BELOW_HORIZON_DEG:
        return None, "pit_too_close_to_horizon"
    e1, e2 = ground_frame(n)
    xy = np.column_stack([P @ e1, P @ e2])
    foot = -h_m * n                      # проекция камеры на полотно
    dists = np.linalg.norm(P - foot, axis=1)
    if float(dists.max()) > MAX_GROUND_DIST_M:
        return None, "pit_too_far"
    rect = cv2.minAreaRect(xy.astype(np.float32))
    sides = sorted([rect[1][0], rect[1][1]], reverse=True)
    # площадь Гаусса по упорядоченному контуру (проекция сохраняет порядок)
    x, y = xy[:, 0], xy[:, 1]
    area_m2 = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    # поперечное направление в плоскости: проекция шага на t_hat_img
    c_uv = contour_uv.mean(axis=0, keepdims=True)
    step = np.vstack([c_uv, c_uv + t_hat_img[None, :]])
    Ps, _, v2 = project_to_ground(step, n, w0, h0, f_px, h_m)
    if not bool(v2.all()):
        return None, "pit_touches_horizon"
    t3 = Ps[1] - Ps[0]
    mm_per_px = float(np.linalg.norm(t3)) * 1000.0
    t3 /= np.linalg.norm(t3)
    proj_t = P @ t3
    g3 = np.cross(n, t3)                 # продольное направление в плоскости
    proj_g = P @ g3
    return {"feret_max_cm": round(sides[0] * 100.0, 1),
            "feret_min_cm": round(sides[1] * 100.0, 1),
            "area_cm2": round(area_m2 * 1e4, 0),
            "width_transverse_cm": round(float(proj_t.max() - proj_t.min()) * 100.0, 1),
            "length_longitudinal_cm": round(float(proj_g.max() - proj_g.min()) * 100.0, 1),
            "dist_ground_m": round(float(np.median(dists)), 2),
            "beta_min_deg": round(beta_min, 2),
            "mm_per_px_transverse": round(mm_per_px, 3)}, None


def estimate_horizon(d: np.ndarray, excl_grid: np.ndarray, k_to_orig: float,
                     w0: int, h0: int):
    """Плоскость дороги + якорь неба → линия горизонта и её неопределённость.

    Возвращает (dict | None, reason | None): l_orig (a,b,c) в исходных px,
    theta_deg, roll_deg, sigma_offset_grid_px, диагностика."""
    plane, err = fit_road_plane(d, excl_grid)
    if plane is None:
        return None, err
    gx, gy, c0 = [float(v) for v in plane["coef"]]
    gnorm = float(np.hypot(gx, gy))
    if gnorm <= 0 or gy <= 0:
        return None, "plane_gradient_degenerate"   # диспаритет должен расти вниз

    d_sky, n_sky, sky_mask = find_sky(d)
    if d_sky is None:
        return None, "no_sky_anchor"
    # гейт согласованности: у ИСТИННОГО неба экстраполяция плоскости в его
    # пиксели много ниже d_sky; «даль дороги», притворившаяся небом, лежит НА
    # плоскости (разница ~0) → отказ.
    ys, xs = np.nonzero(sky_mask)
    plane_at_sky = gx * xs + gy * ys + c0
    margin = max(3.0 * plane["sigma"], 2.0 * HORIZON_SYS_GRID_PX * gnorm)
    if float(np.median(plane_at_sky - d_sky)) > -margin:
        return None, "no_sky_anchor"

    rr, cc = d.shape
    cxg, cyg = cc / 2.0, rr / 2.0
    val_pp = gx * cxg + gy * cyg + c0

    def offset_of(coefs, dsky):
        gx_, gy_, c_ = coefs
        gn_ = float(np.hypot(gx_, gy_))
        return (dsky - (gx_ * cxg + gy_ * cyg + c_)) / gn_

    # неопределённость положения горизонта: систематика якоря + ковариация фита
    base = np.array([gx, gy, c0])
    J = np.zeros(3)
    for i in range(3):
        hstep = 1e-6 * (abs(base[i]) + 1e-9)
        up, dn = base.copy(), base.copy()
        up[i] += hstep
        dn[i] -= hstep
        J[i] = (offset_of(up, d_sky) - offset_of(dn, d_sky)) / (2 * hstep)
    var_fit = float(J @ plane["cov"] @ J)
    sigma_offset = float(np.sqrt(max(var_fit, 0.0) + HORIZON_SYS_GRID_PX ** 2))

    def line_and_normal(dsky):
        l = np.array([gx, gy, k_to_orig * (c0 - dsky)], dtype=np.float64)
        return l, normal_from_horizon(l, w0, h0, F_PX)

    l0, n_c = line_and_normal(d_sky)
    if n_c is None:
        return None, "plane_gradient_degenerate"
    theta = float(np.degrees(np.arcsin(np.clip(-n_c[2], -1, 1))))
    roll = float(np.degrees(np.arcsin(np.clip(n_c[0], -1, 1))))
    if not (PITCH_MIN_DEG <= theta <= PITCH_MAX_DEG):
        return None, "pitch_out_of_range"
    if abs(roll) > ROLL_MAX_DEG:
        return None, "roll_too_large"
    # горизонт обязан лежать выше зоны опоры плоскости (центральный столбец)
    y_h_center = -(l0[0] * (w0 / 2.0) + l0[2]) / l0[1]
    if y_h_center > (ROAD_ROW_LO_FRAC - 0.05) * h0:
        return None, "horizon_inside_road_support"

    # нормали при сдвиге якоря на ±sigma (для полосы масштаба)
    n_lo = line_and_normal(d_sky - sigma_offset * gnorm)[1]
    n_hi = line_and_normal(d_sky + sigma_offset * gnorm)[1]
    ghat = np.array([gx, gy]) / gnorm
    return {"l_orig": l0, "n": n_c, "n_lo": n_lo, "n_hi": n_hi,
            "theta_deg": round(theta, 2), "roll_deg": round(roll, 2),
            "sigma_offset_grid_px": round(sigma_offset, 1),
            "d_sky": d_sky, "n_sky_px": n_sky,
            "plane_sigma": round(plane["sigma"], 5),
            "plane_inliers": plane["n_inliers"],
            "ghat_img": ghat, "y_h_center_orig": round(float(y_h_center), 1)}, None


def prior_horizon(theta_deg: float, band_deg: float, w0: int, h0: int):
    """Геометрия из ПРИОРА тангажа (для кадров без неба): синтетический
    горизонт, roll=0. НЕ измерение по кадру — вызывающий код обязан пометить
    результат horizon_method='prior_pitch'."""
    def _n(th):
        v_h = h0 / 2.0 - F_PX * np.tan(np.radians(th))
        return normal_from_horizon(np.array([0.0, 1.0, -v_h]), w0, h0, F_PX)

    n0 = _n(theta_deg)
    if n0 is None:
        return None
    v_h0 = h0 / 2.0 - F_PX * np.tan(np.radians(theta_deg))
    return {"l_orig": np.array([0.0, 1.0, -v_h0]), "n": n0,
            "n_lo": _n(theta_deg - band_deg), "n_hi": _n(theta_deg + band_deg),
            "theta_deg": round(float(theta_deg), 2), "roll_deg": 0.0,
            "sigma_offset_grid_px": None, "d_sky": None, "n_sky_px": 0,
            "plane_sigma": None, "plane_inliers": 0,
            "ghat_img": np.array([0.0, 1.0]),
            "y_h_center_orig": round(float(v_h0), 1)}


# --- прогон одного вида ---------------------------------------------------------

def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    import cv2

    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(cnt) > CONTOUR_MAX_PTS:
        cnt = cnt[:: int(np.ceil(len(cnt) / CONTOUR_MAX_PTS))]
    return cnt


def analyze_view(pipe: DefectPipeline, rd: RelativeDepth, pit: str, view: str,
                 path: Path, overlay_path: Path | None = None,
                 prior_pitch_deg: float | None = None,
                 prior_band_deg: float = 3.0) -> dict:
    row: dict = {"pit": pit, "view": view, "file": str(path)}
    t0 = time.time()
    report, _overlay, masks = pipe.analyze_image(path)
    defect, status = _select_dominant_pothole(report["defects"])
    row["det_status"] = status
    if defect is None:
        row["outcome"] = status
        return row
    row["det_conf"] = round(float(defect.get("confidence", 0.0)), 3)
    mask = np.asarray(masks[report["defects"].index(defect)]) > 0
    if not mask.any():
        row["outcome"] = "empty_mask"
        return row

    img = imgio.read_image(path)
    h0, w0 = img.shape[:2]
    rd._load()
    if not rd.available:
        row["outcome"] = "depth_model_unavailable"
        return row
    d, k_to_orig = my_depth_grid(rd, img)

    # исключить из опоры плоскости ВСЕ маски дефектов (раздутые)
    import cv2

    union = np.zeros((h0, w0), dtype=bool)
    for mm in masks:
        union |= np.asarray(mm) > 0
    excl = mask_to_depth_grid(union, d.shape)
    excl = cv2.dilate(excl.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                      iterations=1).astype(bool)

    hz, err = estimate_horizon(d, excl, k_to_orig, w0, h0)
    row["horizon_method"] = "sky_anchor"
    if hz is None and err == "no_sky_anchor" and prior_pitch_deg is not None:
        hz = prior_horizon(prior_pitch_deg, prior_band_deg, w0, h0)
        row["horizon_method"] = "prior_pitch"
    if hz is None:
        row["outcome"] = err
        return row
    row.update({k: hz[k] for k in ("theta_deg", "roll_deg",
                                   "sigma_offset_grid_px", "n_sky_px",
                                   "plane_sigma", "plane_inliers",
                                   "y_h_center_orig")})

    cnt = largest_contour(mask)
    if cnt is None:
        row["outcome"] = "empty_mask"
        return row
    t_hat = np.array([-hz["ghat_img"][1], hz["ghat_img"][0]])
    foot, err = footprint_metrics(cnt, hz["n"], w0, h0, F_PX, H_M, t_hat)
    if foot is None:
        row["outcome"] = err
        return row
    row["footprint"] = foot

    # полоса масштаба: f + высота + фактическая неопределённость горизонта
    mm0 = foot["mm_per_px_transverse"]
    pitch_band = 0.0
    for n_alt in (hz["n_lo"], hz["n_hi"]):
        if n_alt is None:
            continue
        alt, err_alt = footprint_metrics(cnt, n_alt, w0, h0, F_PX, H_M, t_hat)
        if alt is not None:
            pitch_band = max(pitch_band,
                             abs(alt["mm_per_px_transverse"] / mm0 - 1.0) * 100.0)
    scale_band_pct = F_BAND_PCT + H_BAND_PCT + pitch_band
    row["scale_band_pct"] = round(scale_band_pct, 1)
    row["pitch_band_pct"] = round(pitch_band, 1)

    # бакет + честная см-оценка глубины (существующий механизм)
    bucket = rd.relative_bucket(img, mask)
    row["depth_bucket"] = bucket.get("depth_bucket")
    row["bucket_method"] = bucket.get("method")
    row["rel_norm"] = bucket.get("_rel_norm")
    est = estimate_depth_cm(bucket, mm0, scale_error_band_pct=scale_band_pct)
    row["depth_estimate"] = {k: est[k] for k in
                             ("available", "point_cm", "low_cm", "high_cm",
                              "upper_bound_cm", "method", "confidence",
                              "vs_gost_5cm", "reason")}
    wpx = bucket.get("_width_px")
    if wpx:
        row["width_flat_cm"] = round(
            float(wpx) * float(bucket.get("_px_scale_to_orig") or 1.0)
            * mm0 / 10.0, 1)

    row["elapsed_s"] = round(time.time() - t0, 1)
    row["outcome"] = "ok"

    if overlay_path is not None:
        try:
            ov = img.copy()
            l = hz["l_orig"]
            pts = []
            for u in (0.0, float(w0 - 1)):
                v = -(l[0] * u + l[2]) / l[1]
                pts.append((int(round(u)), int(round(v))))
            cv2.line(ov, pts[0], pts[1], (0, 255, 255), 6)
            cv2.drawContours(ov, [cnt.astype(np.int32).reshape(-1, 1, 2)], -1,
                             (0, 0, 255), 6)
            de = row["depth_estimate"]
            dep = (f"depth<= {de['upper_bound_cm']}cm" if de.get("upper_bound_cm")
                   else f"depth~{de['point_cm']}cm [{de['low_cm']}..{de['high_cm']}]"
                   if de.get("point_cm") is not None else "depth: n/a")
            pref = "PRIOR " if row["horizon_method"] == "prior_pitch" else ""
            txt = (f"{pref}{pit}/{view} th={hz['theta_deg']} roll={hz['roll_deg']} "
                   f"D={foot['dist_ground_m']}m feret={foot['feret_max_cm']}x"
                   f"{foot['feret_min_cm']}cm +-{row['scale_band_pct']}% {dep}")
            cv2.putText(ov, txt, (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(ov, txt, (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (255, 255, 255), 3, cv2.LINE_AA)
            s = 1400.0 / ov.shape[1]
            ov = cv2.resize(ov, (1400, int(round(ov.shape[0] * s))))
            imgio.write_image(overlay_path, ov)
        except Exception as e:                        # оверлей не гейтит расчёт
            row["overlay_error"] = repr(e)
    return row


# --- selftest: синтетическая круговая проверка геометрии ------------------------

def _selftest() -> None:
    theta, roll = np.radians(10.0), np.radians(2.0)
    w0, h0 = 4096, 1844
    cx, cy = w0 / 2.0, h0 / 2.0
    n_true = np.array([np.sin(roll) * np.cos(theta),
                       -np.cos(roll) * np.cos(theta),
                       -np.sin(theta)])

    gw = 1024
    k = w0 / gw
    gh = int(round(h0 / k))
    xs, ys = np.meshgrid(np.arange(gw), np.arange(gh))
    u, v = (xs + 0.5) * k, (ys + 0.5) * k
    p = np.stack([(u - cx) / F_PX, (v - cy) / F_PX, np.ones_like(u)], axis=-1)
    q = np.maximum(-(p @ n_true) / H_M, 0.0)        # 1/Z, небо -> 0
    d = 7.0 * q + 0.3

    hz, err = estimate_horizon(d, np.zeros_like(d, dtype=bool), k, w0, h0)
    assert hz is not None, f"selftest: горизонт не оценен: {err}"
    assert abs(hz["theta_deg"] - 10.0) < 0.4, f"theta: {hz['theta_deg']}"
    assert abs(hz["roll_deg"] - 2.0) < 0.5, f"roll: {hz['roll_deg']}"

    # mm/px в контрольном пикселе против аналитики
    test_uv = np.array([[2048.0, 1300.0]])
    P, _, ok = project_to_ground(test_uv, hz["n"], w0, h0, F_PX, H_M)
    assert ok.all()
    p1 = np.array([(2048.0 - cx) / F_PX, (1300.0 - cy) / F_PX, 1.0])
    z_true = -H_M / float(p1 @ n_true)              # вдоль луча
    z_axis = z_true * 1.0                            # p_z = 1 -> Z оптической оси
    mm_true = 1000.0 * z_axis / F_PX
    t_hat = np.array([-hz["ghat_img"][1], hz["ghat_img"][0]])
    step = np.vstack([test_uv[0], test_uv[0] + t_hat])
    Ps, _, _ = project_to_ground(step, hz["n"], w0, h0, F_PX, H_M)
    mm_est = float(np.linalg.norm(Ps[1] - Ps[0])) * 1000.0
    assert abs(mm_est / mm_true - 1.0) < 0.01, (mm_est, mm_true)

    # прямоугольник 40x30 см на полотне -> проекция в кадр -> восстановление
    e1, e2 = ground_frame(n_true)
    e2f = e2 if e2[2] > 0 else -e2               # «вперёд» от камеры
    G = -H_M * n_true + 4.0 * e2f
    corners = []
    for a, b in ((-0.2, -0.15), (0.2, -0.15), (0.2, 0.15), (-0.2, 0.15)):
        Pc = G + a * e1 + b * e2f
        corners.append([F_PX * Pc[0] / Pc[2] + cx, F_PX * Pc[1] / Pc[2] + cy])
    quad = np.array(corners)
    foot, err = footprint_metrics(quad, hz["n"], w0, h0, F_PX, H_M, t_hat)
    assert foot is not None, err
    assert abs(foot["feret_max_cm"] - 40.0) < 1.0, foot
    assert abs(foot["feret_min_cm"] - 30.0) < 1.0, foot

    # кадр без неба (весь кадр — дорога при theta=25) обязан дать отказ
    theta2 = np.radians(25.0)
    n2 = np.array([0.0, -np.cos(theta2), -np.sin(theta2)])
    q2 = np.maximum(-(p @ n2) / H_M, 1e-6)
    d2 = 7.0 * q2 + 0.3
    hz2, err2 = estimate_horizon(d2, np.zeros_like(d2, dtype=bool), k, w0, h0)
    assert hz2 is None and err2 in ("no_sky_anchor", "pitch_out_of_range"), err2
    print("selftest: OK (theta/roll/mm_per_px/footprint/no-sky gate)")


# --- сводка -----------------------------------------------------------------

def summarize(rows: list[dict], gt: dict, shared: dict) -> dict:
    ok = [r for r in rows if r.get("outcome") == "ok"]
    ref = Counter(r["outcome"] for r in rows if r.get("outcome") != "ok")
    per_pit: dict[str, dict] = {}
    for r in rows:
        per_pit.setdefault(r["pit"], {"pit": r["pit"],
                                      "span_cm": gt.get(r["pit"]),
                                      "shared_scene": shared.get(r["pit"], False),
                                      "views": {}})
        e = {"outcome": r["outcome"]}
        if r.get("outcome") == "ok":
            f = r["footprint"]
            e.update(hm=r.get("horizon_method"),
                     theta=r["theta_deg"], D_m=f["dist_ground_m"],
                     feret_max=f["feret_max_cm"], feret_min=f["feret_min_cm"],
                     wt=f["width_transverse_cm"], area_cm2=f["area_cm2"],
                     band_pct=r["scale_band_pct"], bucket=r.get("depth_bucket"),
                     est=r.get("depth_estimate"))
        per_pit[r["pit"]]["views"][r["view"]] = e

    def _errs(key):
        out = []
        for r in ok:
            span = gt.get(r["pit"])
            if span is None or shared.get(r["pit"], False):
                continue
            out.append(r["footprint"][key] - span)
        return np.array(out, dtype=float)

    stats: dict = {"n_views": len(rows), "n_ok": len(ok),
                   "by_horizon_method": dict(Counter(
                       r.get("horizon_method", "?") for r in ok).most_common()),
                   "refusals": dict(ref.most_common())}
    # span (хорда рулетки) сопоставим только с малой осью/поперечной шириной;
    # длинная ось маски — заведомо иная семантика, её сравнение читалось как
    # «провал масштаба» (ревью 2026-07-16) — исключено.
    for key in ("feret_min_cm", "width_transverse_cm"):
        e = _errs(key)
        if len(e):
            stats[f"span_vs_{key}"] = {
                "n": int(len(e)), "mae_cm": round(float(np.mean(np.abs(e))), 1),
                "median_err_cm": round(float(np.median(e)), 1),
                "median_abs_err_cm": round(float(np.median(np.abs(e))), 1)}
    # Front<->Back: только ЧИСТЫЕ сцены (общие сцены = front/back могут вести
    # разные физические ямы, а dedup-копии считались независимыми — ревью
    # 2026-07-16); поперечная ширина — устойчивая ось, продольный ферет
    # зависит от ракурсной семантики маски.
    fb: dict[str, list] = {"feret_max": [], "wt": []}
    for pit, d0 in per_pit.items():
        if d0["shared_scene"]:
            continue
        vf, vb = d0["views"].get("front", {}), d0["views"].get("back", {})
        if vf.get("outcome") == "ok" and vb.get("outcome") == "ok":
            for dim in fb:
                a, b = vf[dim], vb[dim]
                fb[dim].append(abs(a - b) / max((a + b) / 2.0, 1e-6) * 100.0)
    for dim, vals in fb.items():
        if vals:
            stats[f"front_back_{dim}_rel_diff_pct_clean"] = {
                "n_pits": len(vals), "median": round(float(np.median(vals)), 1),
                "p90": round(float(np.percentile(vals, 90)), 1)}
    thetas = [r["theta_deg"] for r in ok]
    if thetas:
        stats["theta_deg"] = {"median": round(float(np.median(thetas)), 1),
                              "min": round(min(thetas), 1),
                              "max": round(max(thetas), 1)}
    ests = [r["depth_estimate"] for r in ok if r.get("depth_estimate")]
    pts = [e["point_cm"] for e in ests if e.get("point_cm") is not None]
    ubs = [e["upper_bound_cm"] for e in ests if e.get("upper_bound_cm") is not None]
    stats["depth_estimates"] = {
        "n_point": len(pts), "n_upper_bound": len(ubs),
        "n_unavailable": len(ests) - len(pts) - len(ubs),
        "point_median_cm": round(float(np.median(pts)), 1) if pts else None,
        "point_max_cm": round(max(pts), 1) if pts else None,
        "upper_bound_median_cm": round(float(np.median(ubs)), 1) if ubs else None,
        # Согласованность с полевым фактом «1–3 см» проверяема только для
        # ТОЧЕЧНЫХ оценок; верхние границы им не фальсифицируемы и в метрику
        # не входят (ревью 2026-07-16: прежняя share-метрика была тавтологией).
        "points_within_field_3cm": (round(
            float(np.mean([p <= 3.0 for p in pts])), 2) if pts else None),
        "note": "верхние границы полевым фактом не фальсифицируемы"}
    return {"stats": stats, "per_pit": per_pit}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--root", default="datasets/local_pairs")
    ap.add_argument("--out", default="outputs_height_scale")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prior-pitch-deg", type=float, default=None,
                    help="Fallback-приор тангажа для кадров без неба "
                         "(медиана sky-anchored видов; метка prior_pitch)")
    ap.add_argument("--prior-pitch-band-deg", type=float, default=3.0)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    root = Path(args.root)
    out_dir = Path(args.out)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")

    jobs = [(d.name, d / "front.jpg", d / "back.jpg")
            for d in sorted(root.iterdir()) if d.is_dir()]
    if args.pairs:
        want = set(args.pairs)
        jobs = [j for j in jobs if j[0] in want]

    # GT span + общие сцены. Первичный источник сцен — journal.csv (колонка
    # scene: группа >1 ямы = общий кадр): после дедупа пар в ingest
    # (2026-07-16) md5-дублей в local_pairs больше НЕТ, и детект по ним молча
    # включил бы мультиямные сцены в GT-статистику (ревью цикла 18). md5
    # оставлен fallback'ом для наборов без журнала.
    gt: dict[str, float] = {}
    if GT_PATH.exists():
        data = json.loads(GT_PATH.read_text(encoding="utf-8"))
        for p in data.get("pits", []):
            if p.get("span_cm") is not None:
                gt[str(p["pit"])] = float(p["span_cm"])
    scene_of: dict[str, str] = {}
    journal = root.resolve().parent / "journal.csv"
    if journal.exists():
        with open(journal, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("id"):
                    scene_of[r["id"]] = r.get("scene") or r["id"]
    scene_sizes = Counter(scene_of.values())
    hash2pits: dict[str, set] = {}
    for name, pa, pb in jobs:
        for f in (pa, pb):
            if f.exists():
                h = hashlib.md5(f.read_bytes()).hexdigest()
                hash2pits.setdefault(h, set()).add(name)
    shared = {name: (scene_sizes[scene_of[name]] > 1 if name in scene_of else
                     any(name in s and len(s) > 1 for s in hash2pits.values()))
              for name, _, _ in jobs}

    pipe = DefectPipeline(use_depth=False)
    rd = RelativeDepth()
    rows: list[dict] = []
    cache: dict[tuple, dict] = {}
    for name, pa, pb in jobs:
        for view, path in (("front", pa), ("back", pb)):
            if not path.exists():
                continue
            h = hashlib.md5(path.read_bytes()).hexdigest()
            if h in cache:
                row = dict(cache[h], pit=name, view=view, file=str(path),
                           dedup_of=cache[h]["pit"])
            else:
                ov = None if args.no_overlays else out_dir / "overlays" / f"{name}_{view}.jpg"
                row = analyze_view(pipe, rd, name, view, path, overlay_path=ov,
                                   prior_pitch_deg=args.prior_pitch_deg,
                                   prior_band_deg=args.prior_pitch_band_deg)
                cache[h] = row
            rows.append(row)
            f = row.get("footprint")
            hm = row.get("horizon_method")
            print(f"[{name}/{view}] {row['outcome']}"
                  + (f"{' PRIOR' if hm == 'prior_pitch' else ''}"
                     f" th={row.get('theta_deg')} D={f['dist_ground_m']}m "
                     f"feret={f['feret_max_cm']}x{f['feret_min_cm']}см "
                     f"(span_gt={gt.get(name)}) est={row['depth_estimate']}"
                     if f else ""), flush=True)
            with rows_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(
                    {k: v for k, v in row.items()
                     if k not in ()}, ensure_ascii=False, default=str) + "\n")

    summary = summarize(rows, gt, shared)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\n=== СВОДКА ===")
    print(json.dumps(summary["stats"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
