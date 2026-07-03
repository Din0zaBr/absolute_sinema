"""Относительная глубина дефекта (опционально, Сценарий A/B).

ВАЖНО (честность): этот модуль НИКОГДА не выдаёт глубину в сантиметрах. Только
относительный бакет «shallow|medium|deep» как индикатор приоритета. Достоверная
см-глубина — только Сценарий C (photogrammetry.py) или датчик глубины.

Использует Depth Anything V2 Small через transformers, если доступен; иначе no-op.

Бакет считается «кольцевой плоскостной линейкой» (панель дизайна + судья,
2026-07-03; замена нормировки на полнокадровый диапазон, которая зависела от
КОМПОЗИЦИИ кадра — небо/горизонт делали все дефекты «shallow», ревью 2026-07-02):

  1. К кольцу вокруг маски (с защитной полосой от протечки маски) робастно
     подгоняется ПЛОСКОСТЬ дороги: карта Depth Anything аффинно-инвариантна
     (d = k·f(Z) + s, k>0 и s неизвестны), и все величины ниже — разности
     значений одной карты, поэтому s сокращается, а k гасится делением.
  2. Числитель: медианная просадка ядра маски ПОД экстраполированной
     плоскостью (плоскость − карта; больше depth = ближе, яма = меньше).
  3. Знаменатель-«линейка»: перепад плоскости на ПОПЕРЕЧНОЙ (перпендикулярной
     направлению градиента) ширине маски. Геометрия pinhole-камеры на высоте h
     над дорогой: просадка дна dz видна как удлинение луча → drop ≈ dz/(h·D);
     градиент плоскости ≈ 1/(f·h) на пиксель; поперечная ширина W_фактическая
     проецируется в ≈ f·W/D px БЕЗ ракурсного сжатия → линейка ≈ W/(h·D) и
     rel_norm ≈ dz/W: дистанция D, зум f, высота h и аффинные k,s карты
     сокращаются. (Ревью 2026-07-03, подтверждено pinhole-репро: кругово-
     эквивалентный диаметр ракурсно СЖАТОГО футпринта давал rel_norm ∝ √D —
     бакет зависел от дистанции съёмки.)
  4. Надир/слабая перспектива: линейки нет — ЧЕСТНЫЙ ОТКАЗ
     `degenerate_local_ruler`, а не подмена. Ворота разрешимости
     ruler ≥ 3σ/BUCKET_HI: иначе любой drop, прошедший ворота шума, давал бы
     «deep» тождеством, а не измерением (ревью 2026-07-03: текстурная линейка
     3σ была математически вырождена — medium был недостижим).
  5. rel_norm ≈ «глубина/поперечная ширина» — КРУТИЗНА просадки, индикатор
     приоритета, НЕ глубина: широкая пологая просадка легально мельче
     компактной ямы той же глубины.
"""
from __future__ import annotations

import numpy as np

from . import config

# --- Параметры «кольцевой плоскостной линейки» ------------------------------
# Пороги v0 ФИЗИЧЕСКИ мотивированы (rel_norm ~ глубина/поперечная ширина:
# пологая просадка <0.08, опасная крутизна >0.35). Прогон калибровки
# (scripts/calibrate_depth_bucket.py, 2026-07-03) показал честный факт: на всех
# 14 демо-дефектах карта DA-V2-Small вообще НЕ кодирует ямы рельефом (провала
# в месте крупнейшей ямы нет, drop ±0.03 при 3σ-воротах), все — below_local_noise;
# нулевое распределение (210 сдвигов масок) — тоже нули. Позитивная калибровка
# порогов возможна только на фото с ярко выраженными глубокими ямами —
# рекалибровать при появлении набора РФ. ЕДИНИЦЫ ШКАЛЫ НЕ совместимы со
# старыми порогами 0.02/0.06 (те были долями полнокадрового диапазона).
BUCKET_LO = 0.08                 # ниже — shallow
BUCKET_HI = 0.35                 # выше — deep
_MIN_MASK_AREA_PX = 100          # мельче — сигнал разрушен инференсом ~518 px
_MIN_RING_PX = 200               # носитель плоскости
_RING_SUBSAMPLE = 20000          # сабсэмпл кольца для CPU
_NOISE_GATE = 3.0                # просадка < 3σ шероховатости — не сигнал
_REL_NORM_CAP = 4.0              # защита от взрыва на «зеркальной» карте


class RelativeDepth:
    def __init__(self):
        self._pipe = None
        self._tried = False
        self.available = False
        self._frame_key = None   # кадр, для которого закэширована карта глубины
        self._depth_map = None   # полнокадровый инференс — 1 раз на кадр, не на дефект

    def _load(self):
        if self._tried:
            return
        self._tried = True
        try:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                "depth-estimation",
                model=config.DEPTH_MODEL_HF_ID,
                device=-1,  # CPU
            )
            self.available = True
        except Exception:
            self._pipe = None
            self.available = False

    def _frame_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        """Карта глубины кадра с кэшем: N дефектов = 1 инференс, не N.

        (До фикса 2026-06-12 полный инференс модели гонялся на КАЖДЫЙ дефект.)
        """
        import hashlib

        key = (image_bgr.shape, hashlib.md5(image_bgr.tobytes()).hexdigest())
        if key != self._frame_key:
            import cv2
            from PIL import Image

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            out = self._pipe(Image.fromarray(rgb))
            pred = out.get("predicted_depth") if isinstance(out, dict) else None
            if pred is not None:
                # Сырой выход модели: float в РОДНОЙ сетке инференса (~518 px).
                # Именно в ней и работаем: PIL-вариант out["depth"] — это
                # апсемпл до 12 МП + квантование в 0..255, на котором кольцо
                # и плоскость мерили артефакты интерполяции (репро 2026-07-03).
                self._depth_map = pred.squeeze().cpu().numpy().astype(np.float32)
            else:
                self._depth_map = np.asarray(out["depth"], dtype=np.float32)
            self._frame_key = key
        return self._depth_map

    def relative_bucket(self, image_bgr: np.ndarray, mask: np.ndarray) -> dict:
        """Относительная глубина дефекта vs окружающее покрытие.

        Возвращает {'depth_cm': None, 'depth_bucket': str|None, 'certifiable': False}.
        """
        self._load()
        result = {"depth_cm": None, "depth_bucket": None,
                  "depth_certifiable": False, "method": None}
        if not self.available or not np.asarray(mask).any():
            result["method"] = "unavailable"
            return result

        import cv2

        depth = self._frame_depth(image_bgr)
        # Рабочая сетка: длинная сторона ≤ 1024. transformers может вернуть
        # predicted_depth уже интерполированным до 12 МП (проверено 2026-07-03) —
        # на такой сетке геометрия кольца (клипы r_ring) теряет смысл, а
        # морфология дорожает на порядок; даунскейл значений не искажает
        # (карта аффинно-инвариантна и гладкая).
        hd, wd = depth.shape
        s = 1024.0 / max(hd, wd)
        if s < 1.0:
            depth = cv2.resize(depth, (int(round(wd * s)), int(round(hd * s))),
                               interpolation=cv2.INTER_AREA)
        m = mask_to_depth_grid(np.asarray(mask) > 0, depth.shape)
        if not m.any():
            # Маска исчезла при переводе в сетку карты (тонкая трещина):
            # глубинного сигнала на этом разрешении физически нет.
            result["method"] = "mask_below_depth_resolution"
            return result

        result.update(ring_plane_bucket(depth, m))
        return result


def mask_to_depth_grid(m: np.ndarray, depth_shape: tuple) -> np.ndarray:
    """Перевести маску в сетку карты глубины (обычно ВНИЗ к ~518 px инференса).

    Измерение ведётся в родной сетке модели, а не в 12-МП апсемпле: там
    кольцо/плоскость мерили бы артефакты интерполяции (репро 2026-07-03)."""
    if m.shape == tuple(depth_shape):
        return m
    import cv2

    return cv2.resize(m.astype(np.uint8),
                      (depth_shape[1], depth_shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def ring_plane_bucket(depth: np.ndarray, m: np.ndarray) -> dict:
    """Бакет «кольцевой плоскостной линейкой» (см. док модуля). Чистая функция
    (карта глубины + маска) — тестируется на синтетике без модели.

    Возвращает {'depth_bucket': str|None, 'method': str, ...диагностика}.
    """
    import cv2

    H, W = depth.shape
    area = int(m.sum())
    if area < _MIN_MASK_AREA_PX:
        # Инференс Small ~518 px + resize сглаживает шире такого дефекта —
        # сигнал глубины физически разрушен, честный отказ вместо шума.
        return {"depth_bucket": None, "method": "mask_below_depth_resolution"}
    diam = 2.0 * float(np.sqrt(area / np.pi))
    guard = max(3, int(round(0.15 * diam)))          # полоса от протечки маски
    r_ring = int(np.clip(round(0.75 * diam), 12, 60))

    def _ellipse(r: int):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    m8 = m.astype(np.uint8)
    inner = cv2.dilate(m8, _ellipse(guard), iterations=1).astype(bool)
    ring = cv2.dilate(m8, _ellipse(guard + r_ring), iterations=1).astype(bool) & ~inner
    ys, xs = np.nonzero(ring)
    if len(xs) < _MIN_RING_PX or xs.std() < 2.0 or ys.std() < 2.0:
        # Кольца нет или оно выродилось в полоску у края кадра —
        # плоскости не на что опереться.
        return {"depth_bucket": None, "method": "no_ring_support"}
    if len(xs) > _RING_SUBSAMPLE:
        idx = np.random.default_rng(42).choice(len(xs), _RING_SUBSAMPLE,
                                               replace=False)
        xs, ys = xs[idx], ys[idx]

    # 1) Робастная плоскость дороги по кольцу. Координаты нормированы на W/H
    #    для обусловленности lstsq; остатки в float64.
    z = depth[ys, xs].astype(np.float64)
    basis = np.column_stack([xs / W, ys / H, np.ones(len(xs))])
    coef, *_ = np.linalg.lstsq(basis, z, rcond=None)
    res = z - basis @ coef
    med = float(np.median(res))
    mad = float(np.median(np.abs(res - med)))
    keep = np.abs(res - med) <= 3.0 * 1.4826 * mad + 1e-12
    if int(keep.sum()) >= 50:
        # один reweight-проход: выкинуть чужие объекты/протёкшую яму из кольца
        coef, *_ = np.linalg.lstsq(basis[keep], z[keep], rcond=None)
        res_keep = z[keep] - basis[keep] @ coef
    else:
        res_keep = res
    sigma = 1.4826 * float(np.median(np.abs(res_keep - np.median(res_keep)))) + 1e-9

    # 2) Числитель: просадка ядра маски ПОД плоскостью (плоскость − карта;
    #    больше depth = ближе, яма = меньше). Выпуклости (заплатка) дают <0.
    def _plane_minus_depth(sel: np.ndarray) -> np.ndarray:
        yy, xx = np.nonzero(sel)
        plane = coef[0] * (xx / W) + coef[1] * (yy / H) + coef[2]
        return plane - depth[yy, xx].astype(np.float64)

    core = cv2.erode(m8, _ellipse(guard), iterations=1).astype(bool)
    if core.any():
        drop = float(np.median(_plane_minus_depth(core)))
    else:
        # тонкая маска (трещина): эрозия опустошила ядро — верхний квартиль
        drop = float(np.percentile(_plane_minus_depth(m), 75))

    if drop < _NOISE_GATE * sigma:
        # Просадка неотличима от шероховатости покрытия — честный shallow.
        return {"depth_bucket": "shallow",
                "method": "depth_anything_v2_small_ring_plane:below_local_noise",
                "_rel_norm": 0.0, "_sigma": round(sigma, 5),
                "_drop": round(drop, 5)}

    # 3) Линейка: перепад плоскости на ПОПЕРЕЧНОЙ ширине маски (проекция
    #    пикселей маски на перпендикуляр к градиенту; поперечная ширина не
    #    сжимается ракурсом — см. док модуля, ревью 2026-07-03). При
    #    нормированных координатах градиент на пиксель = (a/W, b/H).
    gx, gy = coef[0] / W, coef[1] / H
    gn = float(np.hypot(gx, gy))
    ym, xm = np.nonzero(m)
    if gn > 0.0:
        t = xm * (-gy / gn) + ym * (gx / gn)
        width_px = float(np.percentile(t, 97.5) - np.percentile(t, 2.5))
    else:
        width_px = 0.0
    ruler = gn * width_px

    # Ворота разрешимости: если deep-порог линейки ниже пола шума (drop ≥ 3σ
    # уже гарантированы воротами выше), «deep» был бы тождеством, а не
    # измерением — честный отказ (надир, слабая перспектива, вырожденная
    # маска). Ревью 2026-07-03: текстурная линейка max(…, 3σ) всегда давала
    # deep и была удалена.
    if ruler < _NOISE_GATE * sigma / BUCKET_HI:
        return {"depth_bucket": None, "method": "degenerate_local_ruler",
                "_sigma": round(sigma, 5), "_drop": round(drop, 5)}

    rel_norm = float(np.clip(drop / ruler, 0.0, _REL_NORM_CAP))
    bucket = ("shallow" if rel_norm < BUCKET_LO
              else "deep" if rel_norm > BUCKET_HI else "medium")
    return {"depth_bucket": bucket, "method": "depth_anything_v2_small_ring_plane",
            "_rel_norm": round(rel_norm, 4), "_sigma": round(sigma, 5),
            "_drop": round(drop, 5)}
