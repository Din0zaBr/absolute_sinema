"""Относительная глубина дефекта (опционально, Сценарий A/B).

ВАЖНО (честность): этот модуль НИКОГДА не выдаёт ИЗМЕРЕННУЮ глубину в
сантиметрах (metric.depth_cm всегда null, depth_certifiable=False). Достоверная
см-глубина — только Сценарий C (photogrammetry.py) или датчик глубины.
Помимо бакета «shallow|medium|deep» модуль умеет давать явно помеченную
ОЦЕНКУ глубины в см (estimate_depth_cm, запрос заказчика 2026-07-14):
rel_norm ≈ dz/W (см. п.5 ниже) × поперечная ширина W в см ≈ dz; источник
mm/px — эталон (scale.py) или высота камеры (plane_scale.py, 2026-07-16),
провенанс — metric.scale_source. Оценка всегда confidence='low' с широкой
полосой; без ИСТОЧНИКА масштаба сантиметры НЕ выдаются (ложный масштаб хуже
отсутствия масштаба).

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
     ruler ≥ noise_gate·σ/hi (значения — InferenceConfig, v0: 3σ/0.35): иначе
     любой drop, прошедший ворота шума, давал бы «deep» тождеством, а не
     измерением (ревью 2026-07-03: текстурная линейка 3σ была математически
     вырождена — medium был недостижим).
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
#
# Канон порогов — config.InferenceConfig (depth_bucket_lo/hi,
# depth_noise_gate_sigma): правило «пороги только через config» (аудит
# 2026-07-07, №8). Модульные имена сохранены для существующих импортов
# (tests/test_depth_bucket.py, scripts/calibrate_depth_bucket.py).
BUCKET_LO = config.DEFAULT_INFERENCE.depth_bucket_lo   # ниже — shallow
BUCKET_HI = config.DEFAULT_INFERENCE.depth_bucket_hi   # выше — deep
_NOISE_GATE = config.DEFAULT_INFERENCE.depth_noise_gate_sigma
# Ниже — внутренности алгоритма/бюджет CPU, не калибровочные ручки:
_MIN_MASK_AREA_PX = 100          # мельче — сигнал разрушен инференсом ~518 px
_MIN_RING_PX = 200               # носитель плоскости
_RING_SUBSAMPLE = 20000          # сабсэмпл кольца для CPU
_REL_NORM_CAP = 4.0              # защита от взрыва на «зеркальной» карте


class RelativeDepth:
    def __init__(self, cfg: config.InferenceConfig = config.DEFAULT_INFERENCE):
        self.cfg = cfg
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

    def frame_depth(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Публичный доступ к сырой карте глубины кадра (кэш — см. _frame_depth).

        None, если модель недоступна. Нужен plane_scale.py (масштаб от высоты
        камеры): горизонт оценивается по той же карте, что и бакеты, — один
        инференс на кадр.
        """
        self._load()
        if not self.available:
            return None
        return self._frame_depth(image_bgr)

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
        # Перевод пиксельных величин сетки карты в пиксели ИСХОДНОГО кадра
        # (mm_per_px эталона задан в исходных px). Аспект сохраняется обоими
        # ресайзами — достаточно отношения ширин.
        result["_px_scale_to_orig"] = float(image_bgr.shape[1]) / depth.shape[1]
        m = mask_to_depth_grid(np.asarray(mask) > 0, depth.shape)
        if not m.any():
            # Маска исчезла при переводе в сетку карты (тонкая трещина):
            # глубинного сигнала на этом разрешении физически нет.
            result["method"] = "mask_below_depth_resolution"
            return result

        result.update(ring_plane_bucket(
            depth, m, lo=self.cfg.depth_bucket_lo, hi=self.cfg.depth_bucket_hi,
            noise_gate=self.cfg.depth_noise_gate_sigma))
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


def ring_plane_bucket(depth: np.ndarray, m: np.ndarray,
                      lo: float = BUCKET_LO, hi: float = BUCKET_HI,
                      noise_gate: float = _NOISE_GATE) -> dict:
    """Бакет «кольцевой плоскостной линейкой» (см. док модуля). Чистая функция
    (карта глубины + маска) — тестируется на синтетике без модели.

    lo/hi/noise_gate по умолчанию — значения config.DEFAULT_INFERENCE;
    RelativeDepth передаёт пороги СВОЕГО InferenceConfig (аудит №8).
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

    # 3) Линейка: перепад плоскости на ПОПЕРЕЧНОЙ ширине маски (проекция
    #    пикселей маски на перпендикуляр к градиенту; поперечная ширина не
    #    сжимается ракурсом — см. док модуля, ревью 2026-07-03). При
    #    нормированных координатах градиент на пиксель = (a/W, b/H).
    #    Считается ДО ворот шума: ветке below_local_noise линейка нужна для
    #    честной ВЕРХНЕЙ границы глубины (estimate_depth_cm), сами ворота и
    #    порядок отказов не меняются (запрос см-оценки 2026-07-14).
    gx, gy = coef[0] / W, coef[1] / H
    gn = float(np.hypot(gx, gy))
    ym, xm = np.nonzero(m)
    if gn > 0.0:
        t = xm * (-gy / gn) + ym * (gx / gn)
        width_px = float(np.percentile(t, 97.5) - np.percentile(t, 2.5))
    else:
        width_px = 0.0
    ruler = gn * width_px
    ruler_resolvable = ruler >= noise_gate * sigma / hi

    if drop < noise_gate * sigma:
        # Просадка неотличима от шероховатости покрытия — честный shallow.
        # _rel_upper — минимально РАЗРЕШИМЫЙ rel_norm (просадка ровно на
        # воротах): реальный rel_norm ямы ниже него → верхняя граница глубины.
        # При вырожденной линейке границы нет (None), fabricate нечего.
        rel_upper = (float(noise_gate * sigma / ruler)
                     if ruler_resolvable else None)
        return {"depth_bucket": "shallow",
                "method": "depth_anything_v2_small_ring_plane:below_local_noise",
                "_rel_norm": 0.0, "_sigma": round(sigma, 5),
                "_drop": round(drop, 5), "_ruler": round(ruler, 6),
                "_width_px": round(width_px, 1),
                "_rel_upper": (round(rel_upper, 4)
                               if rel_upper is not None else None)}

    # Ворота разрешимости: если deep-порог линейки ниже пола шума (drop ≥ 3σ
    # уже гарантированы воротами выше), «deep» был бы тождеством, а не
    # измерением — честный отказ (надир, слабая перспектива, вырожденная
    # маска). Ревью 2026-07-03: текстурная линейка max(…, 3σ) всегда давала
    # deep и была удалена.
    if not ruler_resolvable:
        return {"depth_bucket": None, "method": "degenerate_local_ruler",
                "_sigma": round(sigma, 5), "_drop": round(drop, 5),
                "_ruler": round(ruler, 6), "_width_px": round(width_px, 1)}

    rel_norm = float(np.clip(drop / ruler, 0.0, _REL_NORM_CAP))
    bucket = ("shallow" if rel_norm < lo
              else "deep" if rel_norm > hi else "medium")
    return {"depth_bucket": bucket, "method": "depth_anything_v2_small_ring_plane",
            "_rel_norm": round(rel_norm, 4), "_sigma": round(sigma, 5),
            "_drop": round(drop, 5), "_ruler": round(ruler, 6),
            "_width_px": round(width_px, 1)}


# --- Оценка глубины в сантиметрах (запрос заказчика 2026-07-14) ---------------
_EST_NOTE = ("Оценка, НЕ измерение: rel_norm (относительная карта глубины) x "
             "поперечная ширина ямы в см (по доступному источнику масштаба — "
             "см. metric.scale_source: эталон или высота камеры). Погрешность "
             "велика; для акта нужен ручной замер или two-view съёмка "
             "(Сценарий C).")


def estimate_depth_cm(bucket_result: dict, mm_per_px: float | None,
                      scale_error_band_pct: float | None = None,
                      cfg: config.InferenceConfig = config.DEFAULT_INFERENCE) -> dict:
    """Честная ОЦЕНКА глубины дефекта в см из результата ring_plane_bucket.

    Физика: rel_norm ≈ dz/W (глубина / ПОПЕРЕЧНАЯ ширина; дистанция, зум,
    высота камеры и аффинные k,s карты сокращаются — см. док модуля), а
    поперечная ширина не сжимается ракурсом, значит W_см ≈ width_px · mm/px.
    Отсюда dz ≈ rel_norm · W_см. Для просадки ниже шума карты выдаётся только
    ВЕРХНЯЯ граница (реальный rel_norm ниже минимально разрешимого).

    Гейты честности (available=False + reason, никаких чисел):
      - нет эталона масштаба (mm_per_px=None) — см выдумать нельзя;
      - глубинного сигнала нет (отказы ring_plane_bucket) — оценивать нечего;
      - линейка вырождена (надир/слабая перспектива) — даже граница не имеет
        смысла.

    Оценка ВСЕГДА confidence='low' и certifiable=False: mm/px от эталона
    измерен у эталона, а не у ямы (перспективный перенос — та же оговорка,
    что у length_cm/area); mm/px от высоты камеры — поперечный у самой ямы,
    но несёт полосу источника; сам rel_norm шумный. Полоса ошибки —
    мультипликативная: [point/(1+b), point*(1+b)],
    b = (базовая + полоса источника масштаба)/100.

    Чистая функция: только словарь + числа, тестируется без модели.
    """
    est = {"available": False, "point_cm": None, "low_cm": None,
           "high_cm": None, "upper_bound_cm": None, "method": None,
           "confidence": None, "certifiable": False, "vs_gost_5cm": None,
           "reason": None, "note": _EST_NOTE}
    if not cfg.depth_cm_estimate_enabled:
        est["reason"] = "disabled"
        return est

    method = bucket_result.get("method") or ""
    width_px = bucket_result.get("_width_px")
    if width_px is None:
        # Отказы до линейки: unavailable / mask_below_depth_resolution /
        # no_ring_support — глубинного сигнала нет, честно передаём причину.
        est["reason"] = method or "no_depth_signal"
        return est
    if mm_per_px is None or not np.isfinite(mm_per_px) or mm_per_px <= 0:
        est["reason"] = "no_scale_reference"
        return est

    # ring_plane_bucket мог быть вызван и напрямую (синтетика/калибровка) —
    # тогда карта уже в пикселях исходного кадра, множитель 1.
    k = bucket_result.get("_px_scale_to_orig") or 1.0
    width_cm = float(width_px) * float(k) * float(mm_per_px) / 10.0
    if not np.isfinite(width_cm) or width_cm <= 0:
        est["reason"] = "degenerate_width"
        return est

    band = (cfg.depth_cm_est_base_err_pct
            + (scale_error_band_pct if scale_error_band_pct is not None
               else 30.0)) / 100.0
    gost = config.GOST50597_MAX_DEPTH_CM

    if method.endswith("below_local_noise"):
        rel_upper = bucket_result.get("_rel_upper")
        if rel_upper is None:
            est["reason"] = "below_noise_unresolvable_ruler"
            return est
        # Верхняя граница расширяется полосой ВВЕРХ (консервативно): ниже неё
        # глубина «по оценке», точки нет — просадка неотличима от шероховатости.
        ub = float(rel_upper) * width_cm * (1.0 + band)
        est.update(available=True, upper_bound_cm=round(ub, 1),
                   method="below_noise_upper_bound", confidence="low",
                   vs_gost_5cm="below" if ub < gost else "uncertain")
        return est

    rel = bucket_result.get("_rel_norm")
    if rel is None or bucket_result.get("depth_bucket") is None:
        # degenerate_local_ruler и прочие отказы с посчитанной шириной.
        est["reason"] = method or "no_depth_signal"
        return est

    point = float(rel) * width_cm
    low, high = point / (1.0 + band), point * (1.0 + band)
    est.update(available=True, point_cm=round(point, 1),
               low_cm=round(low, 1), high_cm=round(high, 1),
               method="rel_norm_x_transverse_width", confidence="low",
               vs_gost_5cm=("above" if low >= gost
                            else "below" if high < gost else "uncertain"))
    return est
