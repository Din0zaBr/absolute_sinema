"""Масштаб плоскости полотна от ИЗВЕСТНОЙ ВЫСОТЫ КАМЕРЫ (без эталона в кадре).

Физика: высота объектива H над полотном + фокус f (EXIF/конфиг) + линия
горизонта задают метрику плоскости дороги: контур маски дефекта проецируется
лучами на плоскость, откуда длина/ширина/площадь в см и поперечный mm/px для
см-ОЦЕНКИ глубины (depth.estimate_depth_cm). Горизонт оценивается по карте
Depth Anything: карта аффинно-инвариантна (d = k/Z + s), но у НЕБА 1/Z = 0 —
медиана неба даёт s, робастная плоскость дороги — градиент; их пересечение —
горизонт → нормаль полотна n ∝ Kᵀl. Для кадров без неба (дворы) — приор
тангажа (cfg.pitch_prior_deg), помечаемый отдельно.

ЧЕСТНОСТЬ (главный принцип проекта, ТЗ v2 §2.1):
  * источник НЕ сертифицируем: потребитель обязан помечать размеры
    confidence='low', certifiable=false, с полосой ошибки (pipeline так и
    делает: metric.scale_source='camera_height');
  * поперечный масштаб mm/px ≈ 1000·H/(v_дефекта − v_горизонта) — от f почти
    не зависит; согласуется с объектом известного размера в пределах ~±10–15%
    (n=1, приёмочная валидация НЕ проводилась — docs/HEIGHT_SCALE_RESULTS.md);
    Д/Ш — размер МАСКИ (с тёмным ореолом), а не рулеточного «разрушения»;
  * приор тангажа (pitch_prior_deg) НЕ проверяется по кадру — по карте глубины
    подтверждаются только крен и направление уклона; валиден лишь для той же
    постановки съёмки (стоя, с заявленной высоты);
  * кадры читаются без EXIF-поворота (imgio): портретный кадр лежит «на
    боку» — штатный исход на нём — честный отказ (no_sky/roll), не поддержан;
  * гейты честного отказа (available=False + reason, чисел нет): no_focal /
    no_sky_anchor / plane_* / pitch_out_of_range / roll_too_large /
    horizon_inside_road_support / pit_too_close_to_horizon / pit_too_far /
    empty_mask. Отказ — штатный результат, а не сбой.

Происхождение: эксперимент scripts/height_scale_experiment.py (2026-07-16),
синтетическая круговая проверка геометрии — tests/test_plane_scale.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

# Ширина плёночного кадра 35 мм — знаменатель пересчёта EXIF-эквивалента.
_FILM_WIDTH_MM = 36.0
_EXIF_FL35_TAG = 41989          # FocalLengthIn35mmFilm
_EXIF_IFD_TAG = 0x8769


@dataclass
class FrameGroundGeometry:
    """Геометрия «камера над полотном» одного кадра (результат resolve_*)."""
    horizon_method: str          # 'sky_anchor' | 'prior_pitch'
    f_px: float
    f_source: str                # 'config' | 'exif_fl35'
    height_m: float
    theta_deg: float             # тангаж вниз от горизонтали
    roll_deg: float
    l_orig: np.ndarray           # линия горизонта (a, b, c), исходные px
    n: np.ndarray                # нормаль полотна в камере («вверх»)
    n_lo: np.ndarray | None      # нормали при сдвиге горизонта на ±sigma —
    n_hi: np.ndarray | None      # для полосы масштаба
    ghat_img: np.ndarray         # направление градиента плоскости в кадре
    image_wh: tuple[int, int]
    diag: dict                   # диагностика (небо/плоскость/сигмы)

    def to_dict(self) -> dict:
        return {"horizon_method": self.horizon_method,
                "f_px": round(self.f_px, 1), "f_source": self.f_source,
                "height_m": self.height_m,
                "theta_deg": round(self.theta_deg, 2),
                "roll_deg": round(self.roll_deg, 2), **self.diag}


# --- фокус из EXIF ------------------------------------------------------------

def f_px_from_exif(image_path: str | Path,
                   image_wh: tuple[int, int]) -> tuple[float | None, str]:
    """(f_px | None, источник/причина): f = fl35 × длинная_сторона / 36.

    EXIF читается PIL (cv2 EXIF не отдаёт); пересчёт неоднозначен на
    кроп-режимах смартфонов — потребитель обязан носить cfg.f_band_pct.
    """
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            ex = im.getexif()
            fl35 = None
            try:
                fl35 = ex.get_ifd(_EXIF_IFD_TAG).get(_EXIF_FL35_TAG)
            except Exception:  # noqa: BLE001 — битый IFD = нет данных
                pass
            if fl35 is None:
                fl35 = ex.get(_EXIF_FL35_TAG)
    except Exception:  # noqa: BLE001 — нечитаемый EXIF = честный отказ
        return None, "no_focal:exif_unreadable"
    try:
        fl35 = float(fl35) if fl35 is not None else None
    except Exception:  # noqa: BLE001
        fl35 = None
    if not fl35 or fl35 <= 0:
        return None, "no_focal:no_fl35_in_exif"
    return fl35 * max(image_wh) / _FILM_WIDTH_MM, "exif_fl35"


# --- внутренняя геометрия (порт эксперимента, валидирован синтетикой) ---------

def depth_to_grid(raw: np.ndarray, w0: int, h0: int,
                  long_side: int) -> tuple[np.ndarray, float]:
    """Карта глубины в сетке с аспектом СТРОГО от кадра; (map, px кадра/px сетки)."""
    import cv2

    target_long = min(long_side, max(raw.shape))
    s = target_long / max(h0, w0)
    gw, gh = max(2, int(round(w0 * s))), max(2, int(round(h0 * s)))
    interp = cv2.INTER_AREA if raw.shape[1] >= gw else cv2.INTER_LINEAR
    d = cv2.resize(raw.astype(np.float32), (gw, gh), interpolation=interp)
    return d.astype(np.float64), float(w0) / gw


def _fit_road_plane(d: np.ndarray, excl: np.ndarray,
                    cfg: config.CameraHeightScaleConfig):
    rr = d.shape[0]
    lo, hi = int(cfg.road_row_lo_frac * rr), int(cfg.road_row_hi_frac * rr)
    sup = np.zeros_like(d, dtype=bool)
    sup[lo:hi] = True
    sup &= ~excl
    ys, xs = np.nonzero(sup)
    if len(xs) < cfg.plane_min_support:
        return None, "plane_support_too_small"
    if len(xs) > cfg.plane_max_subsample:
        idx = np.random.default_rng(42).choice(len(xs), cfg.plane_max_subsample,
                                               replace=False)
        xs, ys = xs[idx], ys[idx]
    z = d[ys, xs].astype(np.float64)
    A = np.column_stack([xs.astype(np.float64), ys.astype(np.float64),
                         np.ones(len(xs))])
    keep = np.ones(len(xs), dtype=bool)
    coef = None
    for _ in range(cfg.plane_reweight_iters):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = z - A @ coef
        med = float(np.median(res[keep]))
        mad = float(np.median(np.abs(res[keep] - med))) + 1e-12
        keep = np.abs(res - med) <= 3.0 * 1.4826 * mad
        if int(keep.sum()) < cfg.plane_min_support // 2:
            return None, "plane_fit_degenerate"
    res_in = z[keep] - A[keep] @ coef
    sigma = 1.4826 * float(np.median(np.abs(res_in - np.median(res_in)))) + 1e-12
    try:
        cov = (sigma ** 2) * np.linalg.inv(A[keep].T @ A[keep])
    except np.linalg.LinAlgError:
        return None, "plane_fit_degenerate"
    return {"coef": coef, "sigma": sigma, "n_inliers": int(keep.sum()),
            "cov": cov}, None


def _find_sky(d: np.ndarray, cfg: config.CameraHeightScaleConfig):
    rr = d.shape[0]
    top = d[: int(cfg.sky_top_frac * rr)]
    d_lo, d_hi = np.percentile(d, [1.0, 99.0])
    thr = d_lo + cfg.sky_band_frac * (d_hi - d_lo)
    m = top <= thr
    n = int(m.sum())
    if n < cfg.sky_min_px:
        return None, n, m
    return float(np.median(top[m])), n, m


def normal_from_horizon(l_orig: np.ndarray, w0: int, h0: int,
                        f_px: float) -> np.ndarray | None:
    """Нормаль полотна из линии горизонта: n ∝ Kᵀl; знак — «вверх»
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

    Возвращает (P (N,3) в метрах, sin угла луча ниже горизонта, valid)."""
    cx, cy = w0 / 2.0, h0 / 2.0
    p = np.column_stack([(pts_uv[:, 0] - cx) / f_px,
                         (pts_uv[:, 1] - cy) / f_px,
                         np.ones(len(pts_uv))])
    denom = p @ n
    sin_beta = -denom / np.linalg.norm(p, axis=1)
    valid = denom < -1e-9
    t = np.where(valid, -h_m / np.where(valid, denom, -1.0), np.nan)
    return p * t[:, None], sin_beta, valid


def _estimate_horizon_sky(plane: dict, d: np.ndarray, k_to_orig: float,
                          w0: int, h0: int, f_px: float,
                          cfg: config.CameraHeightScaleConfig):
    """Якорь неба при ГОТОВОЙ плоскости → (частичная геометрия | None, reason).

    Плоскость фитится в resolve_frame_geometry (она нужна и prior-ветке)."""
    gx, gy, c0 = [float(v) for v in plane["coef"]]
    gnorm = float(np.hypot(gx, gy))

    d_sky, n_sky, sky_mask = _find_sky(d, cfg)
    if d_sky is None:
        return None, "no_sky_anchor"
    # Гейт согласованности: истинное небо лежит МНОГО ниже экстраполяции
    # плоскости в его пиксели; «даль дороги», притворившаяся небом, — на
    # плоскости (разница ~0) → отказ.
    ys, xs = np.nonzero(sky_mask)
    plane_at_sky = gx * xs + gy * ys + c0
    margin = max(3.0 * plane["sigma"], 2.0 * cfg.horizon_sys_grid_px * gnorm)
    if float(np.median(plane_at_sky - d_sky)) > -margin:
        return None, "no_sky_anchor"

    rr, cc = d.shape
    cxg, cyg = cc / 2.0, rr / 2.0
    val_pp = gx * cxg + gy * cyg + c0

    def offset_of(coefs, dsky):
        gx_, gy_, c_ = coefs
        return (dsky - (gx_ * cxg + gy_ * cyg + c_)) / float(np.hypot(gx_, gy_))

    base = np.array([gx, gy, c0])
    J = np.zeros(3)
    for i in range(3):
        hstep = 1e-6 * (abs(base[i]) + 1e-9)
        up, dn = base.copy(), base.copy()
        up[i] += hstep
        dn[i] -= hstep
        J[i] = (offset_of(up, d_sky) - offset_of(dn, d_sky)) / (2 * hstep)
    var_fit = float(J @ plane["cov"] @ J)
    sigma_offset = float(np.sqrt(max(var_fit, 0.0)
                                 + cfg.horizon_sys_grid_px ** 2))

    def line_and_normal(dsky):
        l = np.array([gx, gy, k_to_orig * (c0 - dsky)], dtype=np.float64)
        return l, normal_from_horizon(l, w0, h0, f_px)

    l0, n_c = line_and_normal(d_sky)
    if n_c is None:
        return None, "plane_gradient_degenerate"
    theta = float(np.degrees(np.arcsin(np.clip(-n_c[2], -1, 1))))
    roll = float(np.degrees(np.arcsin(np.clip(n_c[0], -1, 1))))
    if not (cfg.pitch_min_deg <= theta <= cfg.pitch_max_deg):
        return None, "pitch_out_of_range"
    if abs(roll) > cfg.roll_max_deg:
        return None, "roll_too_large"
    y_h_center = -(l0[0] * (w0 / 2.0) + l0[2]) / l0[1]
    if y_h_center > (cfg.road_row_lo_frac - 0.05) * h0:
        return None, "horizon_inside_road_support"

    n_lo = line_and_normal(d_sky - sigma_offset * gnorm)[1]
    n_hi = line_and_normal(d_sky + sigma_offset * gnorm)[1]
    return {"l_orig": l0, "n": n_c, "n_lo": n_lo, "n_hi": n_hi,
            "theta_deg": theta, "roll_deg": roll,
            "ghat_img": np.array([gx, gy]) / gnorm,
            "diag": {"n_sky_px": n_sky,
                     "sigma_offset_grid_px": round(sigma_offset, 1),
                     "plane_sigma": round(plane["sigma"], 5),
                     "plane_inliers": plane["n_inliers"],
                     "y_h_center_orig": round(float(y_h_center), 1)}}, None


def _prior_geometry(f_px: float, w0: int, h0: int,
                    cfg: config.CameraHeightScaleConfig, ghat: np.ndarray):
    """Геометрия из ПРИОРА тангажа (кадры без неба): θ — из конфига (НЕ
    измерен по кадру!), а крен/направление уклона — ИЗМЕРЕННЫЕ (ghat —
    градиент плоскости дороги в карте глубины; направление ∇d инвариантно к
    аффинной неоднозначности карты). Потребитель помечает prior_pitch."""
    theta, band = cfg.pitch_prior_deg, cfg.pitch_prior_band_deg
    cx, cy = w0 / 2.0, h0 / 2.0

    def _line(th):
        # горизонт: точки со смещением −f·tanθ от главной точки вдоль ĝ
        c = f_px * np.tan(np.radians(th)) - ghat[0] * cx - ghat[1] * cy
        return np.array([ghat[0], ghat[1], c], dtype=np.float64)

    def _n(th):
        return normal_from_horizon(_line(th), w0, h0, f_px)

    n0 = _n(theta)
    if n0 is None:
        return None, "plane_gradient_degenerate"
    l0 = _line(theta)
    y_h = (-(l0[0] * cx + l0[2]) / l0[1]) if abs(l0[1]) > 1e-9 else None
    return {"l_orig": l0, "n": n0,
            "n_lo": _n(theta - band), "n_hi": _n(theta + band),
            "theta_deg": float(theta),
            "roll_deg": float(np.degrees(np.arcsin(np.clip(n0[0], -1, 1)))),
            "ghat_img": np.asarray(ghat, dtype=np.float64),
            "diag": {"n_sky_px": 0, "sigma_offset_grid_px": None,
                     "plane_sigma": None, "plane_inliers": 0,
                     "roll_from_gradient": True,
                     "y_h_center_orig": (round(float(y_h), 1)
                                         if y_h is not None else None)}}, None


# --- публичный API -------------------------------------------------------------

def resolve_frame_geometry(img: np.ndarray, raw_depth: np.ndarray,
                           image_path: str | Path | None,
                           exclude_boxes_xywh: list,
                           cfg: config.CameraHeightScaleConfig,
                           ) -> tuple[FrameGroundGeometry | None, str | None]:
    """Геометрия кадра: якорь неба, при отказе «нет неба» — приор тангажа.

    exclude_boxes_xywh — bbox'ы детекций в исходных px: сами дефекты не должны
    искажать опору плоскости.
    """
    import cv2

    if not cfg.enabled:
        return None, "disabled"
    h0, w0 = img.shape[:2]
    f_px, f_source = ((float(cfg.f_px), "config") if cfg.f_px is not None
                      else (None, ""))
    if f_px is None:
        if image_path is None:
            return None, "no_focal:no_path"
        f_px, f_source = f_px_from_exif(image_path, (w0, h0))
        if f_px is None:
            return None, f_source          # уже вида no_focal:*

    d, k = depth_to_grid(raw_depth, w0, h0, cfg.grid_long_side)
    excl = np.zeros(d.shape, dtype=bool)
    for bx in exclude_boxes_xywh or []:
        x, y, w, h = [float(v) for v in bx]
        x0, y0 = int(max(x / k, 0)), int(max(y / k, 0))
        x1 = int(min((x + w) / k + 1, d.shape[1]))
        y1 = int(min((y + h) / k + 1, d.shape[0]))
        if x1 > x0 and y1 > y0:
            excl[y0:y1, x0:x1] = True
    if excl.any():
        excl = cv2.dilate(excl.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                          iterations=1).astype(bool)

    plane, err = _fit_road_plane(d, excl, cfg)
    if plane is None:
        return None, err
    gx, gy, _c0 = [float(v) for v in plane["coef"]]
    gnorm = float(np.hypot(gx, gy))
    if gnorm <= 0 or gy <= 0:
        return None, "plane_gradient_degenerate"   # диспаритет растёт вниз

    geom, err = _estimate_horizon_sky(plane, d, k, w0, h0, f_px, cfg)
    method = "sky_anchor"
    if geom is None and err == "no_sky_anchor" and cfg.pitch_prior_deg is not None:
        # Приор: θ из конфига, но крен/уклон — измеренные по градиенту
        # плоскости; кадр с креном за гейтом честно отвергается и в приоре
        # (ревью 2026-07-16: раньше приор не смотрел на кадр вовсе).
        roll_grad = float(np.degrees(np.arctan2(gx, gy)))
        if abs(roll_grad) > cfg.roll_max_deg:
            return None, "roll_too_large"
        geom, err = _prior_geometry(f_px, w0, h0, cfg,
                                    np.array([gx, gy]) / gnorm)
        method = "prior_pitch"
    if geom is None:
        return None, err
    return FrameGroundGeometry(
        horizon_method=method, f_px=f_px, f_source=f_source,
        height_m=float(cfg.height_m), theta_deg=geom["theta_deg"],
        roll_deg=geom["roll_deg"], l_orig=geom["l_orig"], n=geom["n"],
        n_lo=geom["n_lo"], n_hi=geom["n_hi"], ghat_img=geom["ghat_img"],
        image_wh=(w0, h0), diag=geom["diag"]), None


def _footprint(contours: list, n: np.ndarray, geom: FrameGroundGeometry,
               cfg: config.CameraHeightScaleConfig):
    """Метрика набора контуров (все компоненты маски) на полотне для заданной
    нормали; (dict | None, reason). Фереты/гейты — по всем точкам, площадь —
    сумма Гаусса по каждому контуру (маска может быть многокомпонентной —
    ревью 2026-07-16: «крупнейший контур» занижал длину трещин)."""
    import cv2

    w0, h0 = geom.image_wh
    contour_uv = np.vstack(contours)
    P, sin_beta, valid = project_to_ground(contour_uv, n, w0, h0,
                                           geom.f_px, geom.height_m)
    if not bool(valid.all()):
        return None, "pit_touches_horizon"
    beta_min = float(np.degrees(np.arcsin(np.clip(sin_beta.min(), -1, 1))))
    if beta_min < cfg.pit_min_below_horizon_deg:
        return None, "pit_too_close_to_horizon"
    foot = -geom.height_m * n
    dists = np.linalg.norm(P - foot, axis=1)
    if float(dists.max()) > cfg.max_ground_dist_m:
        return None, "pit_too_far"
    e1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    xy = np.column_stack([P @ e1, P @ e2])
    rect = cv2.minAreaRect(xy.astype(np.float32))
    hi, lo = sorted(rect[1], reverse=True)
    area_m2 = 0.0
    off = 0
    for c in contours:
        k_ = len(c)
        x, y = xy[off:off + k_, 0], xy[off:off + k_, 1]
        off += k_
        if k_ >= 3:
            area_m2 += 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                                       - np.dot(y, np.roll(x, -1))))
    # поперечное направление (не сжимается ракурсом) — перпендикуляр к
    # градиенту плоскости в кадре
    t_hat = np.array([-geom.ghat_img[1], geom.ghat_img[0]])
    c_uv = contour_uv.mean(axis=0, keepdims=True)
    step = np.vstack([c_uv, c_uv + t_hat[None, :]])
    Ps, _, v2 = project_to_ground(step, n, w0, h0, geom.f_px, geom.height_m)
    if not bool(v2.all()):
        return None, "pit_touches_horizon"
    t3 = Ps[1] - Ps[0]
    mm_per_px = float(np.linalg.norm(t3)) * 1000.0
    t3 /= np.linalg.norm(t3)
    proj_t = P @ t3
    proj_g = P @ np.cross(n, t3)
    return {"feret_max_cm": round(hi * 100.0, 1),
            "feret_min_cm": round(lo * 100.0, 1),
            "area_cm2": round(area_m2 * 1e4, 1),
            "area_m2": round(area_m2, 4),
            "equivalent_diameter_cm": round(
                200.0 * float(np.sqrt(area_m2 / np.pi)), 1),
            "width_transverse_cm": round(
                float(proj_t.max() - proj_t.min()) * 100.0, 1),
            "length_longitudinal_cm": round(
                float(proj_g.max() - proj_g.min()) * 100.0, 1),
            "dist_ground_m": round(float(np.median(dists)), 2),
            "beta_min_deg": round(beta_min, 2),
            "mm_per_px_transverse": round(mm_per_px, 3)}, None


def defect_ground_metrics(mask: np.ndarray, geom: FrameGroundGeometry,
                          cfg: config.CameraHeightScaleConfig,
                          ) -> tuple[dict | None, str | None]:
    """Метрика дефекта на полотне + полная полоса масштаба.

    Возвращает (dict | None, reason): фереты/площадь/эквивалентный диаметр в
    см, mm_per_px_transverse (для estimate_depth_cm), scale_band_pct =
    f_band + height_band + чувствительность к неопределённости горизонта.
    """
    import cv2

    m = np.asarray(mask) > 0
    if not m.any():
        return None, "empty_mask"
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    contours = [c.reshape(-1, 2).astype(np.float64) for c in cnts if len(c) >= 3]
    if not contours:
        return None, "empty_mask"
    total = sum(len(c) for c in contours)
    if total > cfg.contour_max_pts:
        step = int(np.ceil(total / cfg.contour_max_pts))
        contours = [c[::step] for c in contours if len(c[::step]) >= 3]
        if not contours:
            return None, "empty_mask"

    foot, err = _footprint(contours, geom.n, geom, cfg)
    if foot is None:
        return None, err
    pitch_band = 0.0
    for n_alt in (geom.n_lo, geom.n_hi):
        if n_alt is None:
            continue
        alt, _err = _footprint(contours, n_alt, geom, cfg)
        if alt is not None:
            pitch_band = max(pitch_band,
                             abs(alt["mm_per_px_transverse"]
                                 / foot["mm_per_px_transverse"] - 1.0) * 100.0)
    foot["pitch_band_pct"] = round(pitch_band, 1)
    foot["scale_band_pct"] = round(
        cfg.f_band_pct + cfg.height_band_pct + pitch_band, 1)
    return foot, None
