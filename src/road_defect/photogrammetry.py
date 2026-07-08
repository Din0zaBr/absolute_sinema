"""Two-view метрическая глубина ямы — Сценарий C в минимальной бортовой форме.

СТАТУС: прототип v0 (цикл 15). НЕ подключён к pipeline/severity; валидирован
только синтетическим стендом (scripts/bench_two_view_depth.py, результаты —
docs/VEHICLE_CAPTURE.md §3.1). `depth_certifiable` в v0 всегда False:
сертифицируемость появится только после валидации на реальных бортовых кадрах
против ручного замера рейкой (docs/VEHICLE_CAPTURE.md §8, п. 3–5).

ФИЗИКА (plane + parallax; вывод и числа — docs/VEHICLE_CAPTURE.md §3).
Камера откалибрована относительно дороги (intrinsics + высота h + тангаж) —
это «риг-эталон» бортовой схемы. Точка дна на глубине dz ниже плоскости,
обратно спроецированная на плоскость, смещена К камере:

    q = C + (P − C) · h/(h + dz),

где P — истинная позиция точки (XY на земле), C — позиция камеры (XY).
Для двух видов одной точки (перед/за) при равных высотах камер:

    q_b − q_f = dz · (C_b − C_f) / (h + dz)

— параллакс направлен вдоль базы и НЕ зависит от положения точки; отсюда
dz = h·Δ/(B − Δ). База B = |C_b − C_f| заранее неизвестна (GPS ±метры) и
восстанавливается регистрацией видов по текстуре асфальта ВОКРУГ ямы
(кольцо лежит на плоскости — его регистрация заодно гасит ошибку тангажа,
см. reg_model='affine'). Разные высоты камер решаются численно
(solve_depth_from_parallax: параллакс зависит от позиции точки вдоль базы).

Честные отказы (в духе depth.py): нет регистрации / рассинхрон масштаба
(неверная высота камеры ИЛИ тангаж ≳1°) / мало покрытия дна (взаимное
затенение, вода, бестекстурье) → method с причиной, глубины НЕТ. Плоская
заплатка — валидное измерение ≈0, а не отказ.

Ограничения v0 (зафиксированы ревью 2026-07-08):
  * скан параллакса одномерный вдоль базы; при РАЗНЫХ высотах камер у ямы
    сбоку от линии базы возникает поперечная компонента
    δ⊥ = dz·v·(h_b−h_f)/((h_f+dz)(h_b+dz)) (~3–4 px BEV при Δh=25 см,
    v=1.2 м) — она не моделируется: это потеря покрытия/отказ, не ложные см;
  * решатель валиден для точек МЕЖДУ камерами (x∈[0,B], встречная схема);
    расширение на серии одной стороны при разных высотах требует проверки
    монотонности (см. solve_depth_from_parallax).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class RigView:
    """Калибровка одного вида: pinhole intrinsics + поза относительно дороги.

    Система координат вида («своя» земля): начало — под камерой, ось X — вдоль
    проекции оптической оси на дорогу, Y — влево, Z — вверх. Тангаж
    pitch_deg > 0 — оптическая ось наклонена ВНИЗ от горизонта. Крен считается
    нулевым (жёсткий монтаж); ошибка тангажа от подвески моделируется стендом.
    """
    f_px: float
    cx: float
    cy: float
    cam_height_m: float
    pitch_deg: float


def ground_to_image_h(view: RigView) -> np.ndarray:
    """Гомография «земля (X, Y, 1) → пиксель (u, v, 1)» для плоскости Z=0.

    Вывод: p_cam = R·(P − C), C = (0, 0, h); для точек плоскости
    H = K·[r1 | r2 | R·(−C)]. Проверяемые инварианты (tests):
    точка на оси (X, 0) → u = cx; X = h/tan(pitch) → v = cy.
    """
    th = math.radians(view.pitch_deg)
    h = view.cam_height_m
    K = np.array([[view.f_px, 0.0, view.cx],
                  [0.0, view.f_px, view.cy],
                  [0.0, 0.0, 1.0]])
    rt = np.array([
        [0.0, -1.0, 0.0],
        [-math.sin(th), 0.0, math.cos(th) * h],
        [math.cos(th), 0.0, math.sin(th) * h],
    ])
    return K @ rt


def image_to_ground(view: RigView, uv: np.ndarray) -> np.ndarray:
    """Пиксели (N, 2) → точки плоскости дороги (N, 2) в системе вида.

    Валидна только НИЖЕ горизонта; точки у/выше горизонта дают огромные |X| —
    вызывающий код обязан работать в окрестности ямы (окно BEV), где это
    исключено геометрией постановки.
    """
    hi = np.linalg.inv(ground_to_image_h(view))
    pts = np.column_stack([uv, np.ones(len(uv))]) @ hi.T
    return pts[:, :2] / pts[:, 2:3]


def _sample_ground_grid(image: np.ndarray, view: RigView,
                        grid_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Сэмплировать изображение в точках земли grid_xy (H, W, 2) системы вида.

    Возвращает (float32-канва, маска валидности «источник внутри кадра»).
    Одна перевыборка исходника — без промежуточных ресемплов.
    """
    import cv2

    hmat = ground_to_image_h(view)
    g = grid_xy.reshape(-1, 2)
    p = np.column_stack([g, np.ones(len(g))]) @ hmat.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[:, :2] / p[:, 2:3]
    behind = p[:, 2] <= 1e-9          # за камерой/у горизонта — невалидно
    uv[behind] = -1e6
    map_x = uv[:, 0].reshape(grid_xy.shape[:2]).astype(np.float32)
    map_y = uv[:, 1].reshape(grid_xy.shape[:2]).astype(np.float32)
    hh, ww = image.shape[:2]
    valid = ((map_x >= 0) & (map_x <= ww - 1) &
             (map_y >= 0) & (map_y <= hh - 1))
    src = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out = cv2.remap(src.astype(np.float32), map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return out, valid


def _mask_ground_center_and_extent(view: RigView, mask: np.ndarray):
    """Центр (медиана) и полуразмеры видимой ямы на земле, в системе вида."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None
    if len(xs) > 4000:
        idx = np.random.default_rng(0).choice(len(xs), 4000, replace=False)
        xs, ys = xs[idx], ys[idx]
    g = image_to_ground(view, np.column_stack([xs, ys]).astype(np.float64))
    center = np.median(g, axis=0)
    half = np.maximum((np.percentile(g, 99, axis=0)
                       - np.percentile(g, 1, axis=0)) / 2.0, 0.05)
    return center, half


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def register_views_on_ring(bev_f: np.ndarray, bev_b: np.ndarray,
                           excl_f: np.ndarray, excl_b: np.ndarray,
                           cfg: config.TwoViewDepthConfig):
    """Регистрация back-BEV → front-BEV по фичам ПЛОСКОСТИ (яма исключена).

    Возвращает (A 2x3, n_inliers, rms_px) либо (None, n, причина-строка).
    """
    import cv2

    # Ниже — внутренности сопоставления, не калибровочные ручки: CLAHE
    # выравнивает контраст перед AKAZE; ratio-тест Лоу 0.8 и параметры
    # RANSAC (maxIters/confidence) — стандартные значения из литературы.
    det = cv2.AKAZE_create()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    m_f = (~excl_f).astype(np.uint8) * 255
    m_b = (~excl_b).astype(np.uint8) * 255
    kf, df = det.detectAndCompute(clahe.apply(_to_u8(bev_f)), m_f)
    kb, db = det.detectAndCompute(clahe.apply(_to_u8(bev_b)), m_b)
    if df is None or db is None or len(kf) < cfg.reg_min_inliers or len(kb) < cfg.reg_min_inliers:
        return None, 0, "registration_failed:few_features"
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(db, df, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2)
            if m.distance < 0.8 * n.distance]
    if len(good) < cfg.reg_min_inliers:
        return None, len(good), "registration_failed:few_matches"
    src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kf[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    if cfg.reg_model == "similarity":
        A, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.reg_ransac_px,
            maxIters=5000, confidence=0.999)
    else:
        A, inl = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.reg_ransac_px,
            maxIters=5000, confidence=0.999)
    n_inl = int(inl.sum()) if inl is not None else 0
    if A is None or n_inl < cfg.reg_min_inliers:
        return None, n_inl, "registration_failed:ransac"
    sel = inl.ravel().astype(bool)
    pred = src.reshape(-1, 2)[sel] @ A[:, :2].T + A[:, 2]
    rms = float(np.sqrt(np.mean(np.sum((pred - dst.reshape(-1, 2)[sel]) ** 2,
                                       axis=1))))
    # Гейты искажения. Отражение (det<=0) для двух видов одной плоскости
    # физически невозможно (обе системы видов правые) — заведомо ложная
    # регистрация. Равномерный масштаб (= sqrt det) отличен от 1 — метрики
    # видов рассинхронизированы (высота калибровки ИЛИ тангаж ≳1°) → отказ:
    # ложная база даст ложные сантиметры. Анизотропия (разброс сингулярных
    # чисел) — легальный след ошибки тангажа, который аффинная модель и
    # должна поглощать, — допускается до предела reg_aniso_tol.
    d2 = float(np.linalg.det(A[:, :2]))
    if d2 <= 0.0:
        return None, n_inl, "registration_failed:reflection"
    sv = np.linalg.svd(A[:, :2], compute_uv=False)
    g = float(np.sqrt(d2))
    if abs(g - 1.0) > cfg.reg_scale_tol:
        return None, n_inl, "registration_scale_mismatch"
    if np.max(np.abs(sv - 1.0)) > cfg.reg_aniso_tol:
        return None, n_inl, "registration_distorted"
    return A, n_inl, rms


def solve_depth_from_parallax(delta_m: np.ndarray, x_along_m: np.ndarray,
                              baseline_m: float, h_f: float, h_b: float,
                              dz_max: float) -> np.ndarray:
    """Глубина dz из параллакса Δ вдоль базы (векторно, бисекция).

    Модель (вывод в докстринге модуля): для точки на расстоянии x от C_f
    вдоль базы δ(dz) = dz·[x/(h_f+dz) − (x−B)/(h_b+dz)]. При h_f == h_b
    сводится к δ = dz·B/(h+dz) (позиция сокращается) — решается замкнуто.
    δ монотонна по dz → бисекция безопасна.
    """
    delta = np.asarray(delta_m, dtype=np.float64)
    if abs(h_f - h_b) < 1e-9:
        # при равных высотах позиция сокращается для ЛЮБОЙ точки — формула
        # точна и вне [0, B] (серии одной стороны безопасны)
        return h_f * delta / np.maximum(baseline_m - delta, 1e-6)
    x = np.asarray(x_along_m, dtype=np.float64)
    # Монотонность δ(dz) гарантирована только для точек МЕЖДУ камерами:
    # f'(dz) >= 0 требует x>=0 И x<=B (ревью 2026-07-08). Вне диапазона —
    # ошибка вызывающего кода (встречная геометрия его исключает).
    if np.any((x < -0.5) | (x > baseline_m + 0.5)):
        raise ValueError(
            "solve_depth_from_parallax: x вне [0, B] — при разных высотах "
            "камер бисекция валидна только между камерами (встречная схема)")
    x = np.clip(x, 0.0, baseline_m)
    lo = np.zeros_like(delta)
    hi = np.full_like(delta, dz_max * 1.5)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        pred = mid * (x / (h_f + mid) - (x - baseline_m) / (h_b + mid))
        too_small = pred < delta
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    return 0.5 * (lo + hi)


def _refused(method: str, **extra) -> dict:
    out = {"depth_cm_median": None, "depth_cm_p90": None,
           "depth_certifiable": False, "coverage": 0.0, "n_matches": 0,
           "baseline_m": None, "method": method}
    out.update(extra)
    return out


def estimate_two_view_depth(img_front: np.ndarray, mask_front: np.ndarray,
                            view_front: RigView,
                            img_back: np.ndarray, mask_back: np.ndarray,
                            view_back: RigView,
                            cfg: config.TwoViewDepthConfig = config.DEFAULT_TWO_VIEW,
                            ) -> dict:
    """Оценка глубины ямы по двум калиброванным встречным видам.

    mask_* — видимая область ямы В СВОЁМ виде (от сегментации пайплайна);
    любой ненулевой тип приводится к bool (>0). Возвращает dict:
      method — 'two_view_plane_parallax' либо причина честного отказа; при
        отказе depth_cm_* = None. Причины: empty_mask,
        mask_below_bev_resolution, mask_beyond_working_range,
        registration_failed:few_features/few_matches/ransac/reflection,
        registration_scale_mismatch (метрики видов рассинхронизированы:
        высота калибровки ИЛИ тангаж ≳1°), registration_distorted,
        registration_failed:degenerate_baseline, scan_window_degenerate,
        no_textured_floor, apparent_floors_disjoint,
        insufficient_floor_coverage, depth_exceeds_scan_ceiling,
        floor_core_degenerate, floor_core_unmatched.
      depth_cm_median / depth_cm_p90 — статистика ТОЛЬКО по ядру ямы в
        ИСТИННЫХ координатах (глубина ямы = глубина её центра; склоны и
        периферия в статистику не входят): p90 ≈ глубина глубокой части ядра.
      depth_certifiable — False жёстко (прототип v0 до полевой валидации).
      coverage — доля текстурированного apparent-дна переднего вида с
        согласованным матчем; coverage_core — матчи, чья истинная позиция в
        ядре, делённые на площадь ядра.
      saturated_frac — доля матчей у потолка скана (дно глубже dz_max).
      n_matches, n_core_matches, baseline_m, mask_iou, n_reg_inliers,
      reg_rms_px, s_max_px — диагностика.
    """
    import cv2

    # нормализация масок: uint8 0/255 и прочие ненулевые — в bool (иначе
    # *255 переполняло uint8 и валило оценщик на валидном входе — ревью)
    mask_front = np.asarray(mask_front) > 0
    mask_back = np.asarray(mask_back) > 0
    if not mask_front.any() or not mask_back.any():
        return _refused("empty_mask")

    s_m = cfg.bev_mm_px / 1000.0

    # --- 1. BEV каждого вида вокруг его ВИДИМОЙ ямы --------------------------
    def _bev(image, mask, view):
        center, half = _mask_ground_center_and_extent(view, mask)
        if center is None:
            return "empty"
        # кап рабочей зоны: маска у горизонта взрывает обратную проекцию —
        # без капа сетка BEV аллоцировала бы десятки ГБ (ревью 2026-07-08)
        if (float(np.hypot(center[0], center[1])) > cfg.bev_max_range_m
                or float(np.max(half)) > cfg.bev_max_extent_m):
            return "out_of_range"
        hx = float(half[0] + cfg.bev_margin_m)
        hy = float(half[1] + cfg.bev_margin_m)
        nx = max(int(round(2 * hx / s_m)), 32)
        ny = max(int(round(2 * hy / s_m)), 32)
        xs = center[0] - hx + (np.arange(nx) + 0.5) * s_m
        ys = center[1] - hy + (np.arange(ny) + 0.5) * s_m
        grid = np.stack(np.meshgrid(xs, ys), axis=-1)           # (ny, nx, 2)
        bev, valid = _sample_ground_grid(image, view, grid)
        mk, _ = _sample_ground_grid(mask.astype(np.uint8) * 255, view, grid)
        if not (mk > 127).any():
            # тонкая маска исчезла при переводе в BEV-сетку — сигнала нет
            return "below_resolution"
        origin = np.array([xs[0], ys[0]])
        return {"bev": bev, "valid": valid, "mask": mk > 127, "origin": origin,
                "center": center}

    _BEV_REFUSALS = {"empty": "empty_mask",
                     "out_of_range": "mask_beyond_working_range",
                     "below_resolution": "mask_below_bev_resolution"}
    bf = _bev(img_front, mask_front, view_front)
    bb = _bev(img_back, mask_back, view_back)
    for rec in (bf, bb):
        if isinstance(rec, str):
            return _refused(_BEV_REFUSALS[rec])

    # --- 2. Регистрация по кольцу (яма + запас исключены) --------------------
    k_excl = np.ones((cfg.patch_px, cfg.patch_px), np.uint8)
    excl_f = cv2.dilate(bf["mask"].astype(np.uint8), k_excl) > 0
    excl_b = cv2.dilate(bb["mask"].astype(np.uint8), k_excl) > 0
    excl_f |= ~bf["valid"]
    excl_b |= ~bb["valid"]
    A, n_inl, reg_info = register_views_on_ring(
        bf["bev"], bb["bev"], excl_f, excl_b, cfg)
    if A is None:
        return _refused(reg_info, n_reg_inliers=n_inl)

    # BEV-пиксель ↔ земля: G = origin + (px + 0.5)·s? — сетка строилась по
    # центрам ячеек, origin уже в центре пикселя (0,0). Земля-земля:
    # G_f = M·G_b + t, из сопряжения аффинного A (px_b → px_f) сеточными картами.
    def _px_to_ground(rec):
        # G = origin + diag(s)·px  (px = (col, row))
        m = np.array([[s_m, 0.0], [0.0, s_m]])
        return m, rec["origin"]

    m_f, o_f = _px_to_ground(bf)
    m_b, o_b = _px_to_ground(bb)
    lin = m_f @ A[:, :2] @ np.linalg.inv(m_b)
    off = m_f @ A[:, 2] + o_f - lin @ o_b
    # Позиции камер в СВОЕЙ земле — (0, 0); задняя в системе передней:
    c_f = np.zeros(2)
    c_b = lin @ np.zeros(2) + off
    baseline = float(np.linalg.norm(c_b - c_f))
    if baseline < cfg.min_baseline_m:
        return _refused("registration_failed:degenerate_baseline",
                        n_reg_inliers=n_inl)
    bhat = (c_b - c_f) / baseline
    nhat = np.array([-bhat[1], bhat[0]])

    # --- 3. Скан-кадр: сетка передней системы, оси (b̂, n̂) -------------------
    # Δ_max по потолку глубины; каждая точка дна в задней проекции лежит
    # ДАЛЬШЕ вдоль b̂, чем в передней (вывод в докстринге) → скан s >= 0.
    h_min = min(view_front.cam_height_m, view_back.cam_height_m)
    delta_max = cfg.dz_max_m * baseline / (h_min + cfg.dz_max_m)
    s_max = int(math.ceil(delta_max / s_m))

    gc = bf["center"]
    ys0, xs0 = np.nonzero(bf["mask"])
    # полуразмеры окна: видимая маска в координатах (b̂, n̂); grid_f строился
    # как origin + (col, row)·s_m
    pts = np.column_stack([xs0, ys0]) * s_m + o_f - gc
    proj_u = pts @ bhat
    proj_v = pts @ nhat
    # +0.15 м — геометрический запас окна (внутренность, не порог решения)
    half_u = float(np.max(np.abs(proj_u))) + 0.15
    half_v = float(np.max(np.abs(proj_v))) + 0.15
    pad = cfg.patch_px * s_m
    nu = int(round((2 * half_u + delta_max + 2 * pad) / s_m))
    nv = int(round((2 * half_v + 2 * pad) / s_m))
    if nu < cfg.patch_px * 2 or nv < cfg.patch_px * 2:
        return _refused("scan_window_degenerate", n_reg_inliers=n_inl)
    u0 = -half_u - pad
    v0 = -half_v - pad
    uu = u0 + np.arange(nu) * s_m
    vv = v0 + np.arange(nv) * s_m
    guv, gvv = np.meshgrid(uu, vv)
    grid_f = gc[None, None, :] + guv[..., None] * bhat + gvv[..., None] * nhat
    # та же сетка в системе задней камеры:
    lin_inv = np.linalg.inv(lin)
    grid_b = (grid_f - off) @ lin_inv.T

    scan_f, val_f = _sample_ground_grid(img_front, view_front, grid_f)
    scan_b, val_b = _sample_ground_grid(img_back, view_back, grid_b)
    mk_f, _ = _sample_ground_grid(mask_front.astype(np.uint8) * 255,
                                  view_front, grid_f)
    mk_b, _ = _sample_ground_grid(mask_back.astype(np.uint8) * 255,
                                  view_back, grid_b)
    floor = (mk_f > 127) & val_f
    floor_b = (mk_b > 127) & val_b

    # Перекрытие видимых масок в общей системе. Физика (уточнение ревью
    # 2026-07-08): контур apparent-маски — проекция КРОМКИ (она на плоскости),
    # т.е. истинное отверстие ямы при любой глубине, поэтому у честной пары
    # IoU высок всегда; к камерам смещается СОДЕРЖИМОЕ, не контур. Низкий
    # IoU — патология (сегментация, край кадра, грубая регистрация) → отказ.
    inter = int((floor & floor_b).sum())
    union = int((floor | floor_b).sum())
    mask_iou = inter / max(union, 1)
    if mask_iou < cfg.mask_overlap_iou_min:
        return _refused("apparent_floors_disjoint",
                        n_reg_inliers=n_inl, baseline_m=round(baseline, 3),
                        mask_iou=round(mask_iou, 3))
    # партнёр матча обязан лежать в видимой маске ЗАДНЕГО вида (с допуском
    # на несовершенство сегментации): матчи «дно ↔ не-дно» ложны по построению
    floor_b_dil = cv2.dilate(floor_b.astype(np.uint8),
                             np.ones((cfg.patch_px, cfg.patch_px), np.uint8)) > 0

    # --- 4. NCC-скан вдоль b̂ (два прохода: argmax → соседние для параболы) ---
    p = cfg.patch_px

    def _box(x):
        return cv2.boxFilter(x, ddepth=-1, ksize=(p, p),
                             borderType=cv2.BORDER_REFLECT)

    def _bandpass(x):
        # Убрать НЧ (радиальное затемнение чаши, ламбертово затенение, тени):
        # плавный градиент яркости коррелирует сам с собой на МНОГИХ сдвигах и
        # давал широкие ложные пики NCC с огромным параллаксом там, где мелкая
        # текстура слаба (стенд 2026-07-08: p90 18 см при истине 8). Матчим
        # только мм-текстуру покрытия.
        return x - cv2.GaussianBlur(x, (0, 0), float(p))

    f32 = _bandpass(scan_f.astype(np.float32))
    b32 = _bandpass(scan_b.astype(np.float32))
    mu_f = _box(f32)
    var_f = np.maximum(_box(f32 * f32) - mu_f * mu_f, 0.0)
    sd_f = np.sqrt(var_f)
    vfrac_f = _box(val_f.astype(np.float32))
    # Локализуемость вдоль скана: RMS градиента по u в патче. Без него
    # размазанные ракурсом полосы дальнего дна матчились друг с другом
    # с произвольным параллаксом (стенд: p90 15–18 см при истине 8, D=6).
    gu_f = cv2.Sobel(f32, cv2.CV_32F, 1, 0, ksize=3)
    e_f = np.sqrt(np.maximum(_box(gu_f * gu_f), 0.0))
    gu_b = cv2.Sobel(b32, cv2.CV_32F, 1, 0, ksize=3)
    e_b = np.sqrt(np.maximum(_box(gu_b * gu_b), 0.0))
    texture_ok = (e_f >= cfg.texture_grad_min) & (vfrac_f > 0.999)

    s_max = min(s_max, nu - p - 1)
    if s_max < 2:
        return _refused("scan_window_degenerate", n_reg_inliers=n_inl)

    def _ncc_at(s):
        """Карта NCC front(u) vs back(u+s), валидная на ширине nu−s."""
        w = nu - s
        fb = _box(f32[:, :w] * b32[:, s:])
        mu_b = _box(b32[:, s:])
        var_b = np.maximum(_box(b32[:, s:] * b32[:, s:]) - mu_b * mu_b, 0.0)
        vfrac_b = _box(val_b[:, s:].astype(np.float32))
        denom = sd_f[:, :w] * np.sqrt(var_b)
        ncc = (fb - mu_f[:, :w] * mu_b) / np.maximum(denom, 1e-6)
        ncc[(denom < 1e-5) | (vfrac_b < 0.999)
            | (e_b[:, s:] < cfg.texture_grad_min)
            | ~floor_b_dil[:, s:]] = -2.0
        return ncc

    best_v = np.full((nv, nu), -2.0, np.float32)
    best_s = np.full((nv, nu), -1, np.int32)
    for s in range(0, s_max + 1):
        ncc = _ncc_at(s)
        upd = ncc > best_v[:, :ncc.shape[1]]
        best_v[:, :ncc.shape[1]][upd] = ncc[upd]
        best_s[:, :ncc.shape[1]][upd] = s
    left = np.full((nv, nu), np.nan, np.float32)
    right = np.full((nv, nu), np.nan, np.float32)
    for s in range(0, s_max + 1):
        ncc = np.full((nv, nu), np.nan, np.float32)
        ncc[:, :nu - s] = _ncc_at(s)
        sel_l = best_s == s + 1
        left[sel_l] = ncc[sel_l]
        sel_r = best_s == s - 1
        right[sel_r] = ncc[sel_r]

    cand = floor & texture_ok & (best_s >= 0) & (best_v >= cfg.ncc_min)
    n_scanned = int((floor & texture_ok).sum())
    if n_scanned == 0:
        return _refused("no_textured_floor", n_reg_inliers=n_inl,
                        baseline_m=round(baseline, 3))

    # субпиксель: парабола по (left, best, right); на краях — целочисленный пик
    s_sub = best_s.astype(np.float32)
    ok3 = cand & np.isfinite(left) & np.isfinite(right)
    denom = (left - 2.0 * best_v + right)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = 0.5 * (left - right) / denom
    # сосед со значением-сентинелем −2.0 (гейты NCC) — не точка параболы
    good_par = (ok3 & (left > -1.5) & (right > -1.5)
                & np.isfinite(frac) & (np.abs(frac) <= 1.0) & (denom < 0))
    s_sub[good_par] += frac[good_par]

    saturated = cand & (best_s >= s_max - 1)
    measurable = cand & ~saturated

    # пространственная согласованность: поле Δ гладкое, одиночные пики — вон
    s_field = np.where(measurable, s_sub, np.nan)
    med = cv2.medianBlur(np.nan_to_num(s_field, nan=-1e3).astype(np.float32), 5)
    consistent = measurable & (np.abs(s_sub - med) <= cfg.consistency_px) & (med > -1e2)

    # --- 4.5 Ядро дна в ИСТИННЫХ координатах ---------------------------------
    # Глубина ямы — глубина её ЦЕНТРА, а не склонов: без гейта ядра мелкие
    # склоны занижали p90 (10 см → 1.7: нога ГОСТ ложно снята), а ложные
    # матчи «изофот» чаши завышали (8 → 12–18 см). Стенд 2026-07-08.
    # Две правки ревью 2026-07-08:
    #   1) ядро — по distance transform маски (контур маски = проекция кромки
    #      = истинное отверстие), НЕ по эрозии кругового диаметра: та
    #      вырождалась на вытянутых колейных ямах (BLOCKER: тихий фолбэк
    #      «ядро пусто → вся маска» давал медиану 0.9 см при истине 6);
    #   2) принадлежность матча ядру — по его ИСТИННОЙ позиции
    #      P = q·(h+dz)/h (раз-смещение на измеренный параллакс): следы
    #      взаимно видимой полосы дна глубокой ямы смещены к камере на
    #      dz·D/(h+dz) и в apparent-координатах выпадали из ядра (ложный
    #      отказ dz=8 @ D=4 при взаимно видимых 44.5% дна); заодно ложные
    #      матчи с большим Δ «раз-смещаются» за пределы ядра и гибнут.
    n_match = int(consistent.sum())
    n_sat = int(saturated.sum())
    coverage = float((consistent | saturated).sum()) / float(n_scanned)
    sat_frac = n_sat / max(n_match + n_sat, 1)

    dt = cv2.distanceTransform(floor.astype(np.uint8), cv2.DIST_L2, 5)
    r_in = float(dt.max())
    diag = {"n_reg_inliers": n_inl, "reg_rms_px": round(float(reg_info), 2),
            "baseline_m": round(baseline, 3), "coverage": round(coverage, 3),
            "n_matches": n_match, "saturated_frac": round(sat_frac, 3),
            "mask_iou": round(mask_iou, 3), "s_max_px": int(s_max)}
    if r_in < 3.0:
        # маска тоньше ~3 px BEV — ядро не выделить, измерение вырождено
        return _refused("floor_core_degenerate", **diag)
    core = dt >= cfg.core_inradius_frac * r_in
    core_area = int(core.sum())

    if n_match < cfg.min_matches or coverage < cfg.min_coverage:
        return _refused("insufficient_floor_coverage", **diag)
    if sat_frac > cfg.saturated_max:
        # дно глубже потолка скана dz_max: статистика по несатурированному
        # остатку была бы цензурированной (занижение) — честный отказ
        return _refused("depth_exceeds_scan_ceiling", **diag)

    # --- 5. Параллакс → глубина; отбор по ядру в истинных координатах --------
    vi, ui = np.nonzero(consistent)
    delta = s_sub[consistent].astype(np.float64) * s_m
    g_u = (gc @ bhat) + u0 + ui.astype(np.float64) * s_m   # координата вдоль b̂
    g_v = (gc @ nhat) + v0 + vi.astype(np.float64) * s_m   # координата вдоль n̂
    x_along = g_u - (c_f @ bhat)                            # C_f = (0, 0)
    dz_all = solve_depth_from_parallax(
        delta, x_along, baseline,
        view_front.cam_height_m, view_back.cam_height_m, cfg.dz_max_m)
    dz_all = np.clip(dz_all, 0.0, cfg.dz_max_m)

    # раз-смещение к истинной позиции: q = P·h/(h+dz) → P = q·(h+dz)/h
    # (радиально от точки под передней камерой = начала координат)
    scale_true = (view_front.cam_height_m + dz_all) / view_front.cam_height_m
    pu = np.round((g_u * scale_true - (gc @ bhat) - u0) / s_m).astype(np.int64)
    pv = np.round((g_v * scale_true - (gc @ nhat) - v0) / s_m).astype(np.int64)
    inb = (pu >= 0) & (pu < nu) & (pv >= 0) & (pv < nv)
    in_core = np.zeros(len(ui), bool)
    in_core[inb] = core[pv[inb], pu[inb]]
    n_core = int(in_core.sum())
    cov_core = n_core / max(core_area, 1)
    diag.update({"coverage_core": round(cov_core, 3),
                 "n_core_matches": n_core})
    if n_core < cfg.min_core_matches or cov_core < cfg.min_core_coverage:
        # центр дна не сматчен с обеих сторон (взаимное затенение крутой
        # ямы, вода, бестекстурье) — глубина ФИЗИЧЕСКИ не измерена этой парой
        return _refused("floor_core_unmatched", **diag)

    dz = dz_all[in_core]
    out = {"depth_cm_median": round(float(np.median(dz)) * 100.0, 1),
           "depth_cm_p90": round(float(np.percentile(dz, 90)) * 100.0, 1),
           "depth_certifiable": False,
           "method": "two_view_plane_parallax"}
    out.update(diag)
    return out
