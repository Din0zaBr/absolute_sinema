"""Слияние двух видов одной ямы (F1): фото «яма впереди» + «яма позади».

Честная физика. Два встречно-косых вида ОДНОЙ ямы дают НЕ независимые ошибки
разного знака — оба укорачивают план по закону cos(tilt) (cos чётна), т.е.
смещение СИНФАЗНОЕ и простым усреднением НЕ компенсируется. Поэтому:

  • ПЛОЩАДЬ корректируется за наклон в каждом виде (area / cos(tilt) — точный,
    не зависящий от азимута закон для плоской области), и лишь затем сливается;
  • длина/ширина/экв.диаметр сливаются по СЫРЫМ (не скорректированным) значениям
    и честно помечаются «не скорректированы за наклон» — describe_mask не даёт
    протяжённость вдоль азимута наклона, поэтому корректную поправку длины
    выдумывать нельзя (это было бы ложное число);
  • слияние — обратно-дисперсионное (снижает дисперсию двух независимых
    замеров), но полоса ошибки НИКОГДА не уже лучшего одиночного вида: синфазное
    смещение наклона слиянием не убирается, выгода идёт в уверенность, не в банд;
  • полосы считаются в «площадных» процентах (area ~ mm_per_px², т.е. её
    относительная ошибка ≈ 2× линейной полосы масштаба); в metric.error_band_pct
    уходит линейный эквивалент (÷2) — та же семантика, что у одиночного кадра, —
    а площадная полоса отдаётся явно в cross_view.area_error_band_pct;
  • СОГЛАСИЕ видов — честный сигнал уверенности (+1 ступень, но не до high, если
    хоть один вид low), РАСХОЖДЕНИЕ — тревога (понижение до low, шире полоса);
  • глубина в см НЕ выдаётся НИКОГДА (два косых кадра ≠ Сценарий C SfM с
    известной базой) — depth_cm=None, depth_certifiable=False.

Чистый модуль: math + numpy, без cv2/сетей; тестируется на словарях.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import config


_CONF_RANK = {"low": 0, "medium": 1, "high": 2}
_RANK_CONF = {0: "low", 1: "medium", 2: "high"}


def _worse(a: str | None, b: str | None) -> str:
    """Худшая (нижняя) из двух уверенностей."""
    ra = _CONF_RANK.get(a or "low", 0)
    rb = _CONF_RANK.get(b or "low", 0)
    return _RANK_CONF[min(ra, rb)]


def _promote(conf: str | None, notches: int = 1) -> str:
    """Повысить уверенность на N ступеней, с потолком 'high'."""
    r = _CONF_RANK.get(conf or "low", 0)
    return _RANK_CONF[min(2, r + notches)]


def _abs_diff(x, y) -> float:
    """|x-y| для безразмерных дескрипторов формы; 0, если данных нет."""
    if x is None or y is None:
        return 0.0
    return abs(float(x) - float(y))


def _rel_diff(x, y) -> float:
    """Относительное расхождение размеров; 0 для отсутствующего размера."""
    if x is None or y is None:
        return 0.0
    return abs(float(x) - float(y)) / max(abs(float(x)), abs(float(y)), 1e-9)


def inverse_variance_fuse(values, errs_pct):
    """Обратно-дисперсионное слияние замеров с относительными ошибками (%).

    w_i = 1/err_i**2;  fused = Σ w_i v_i / Σ w_i;  fused_err% = 1/sqrt(Σ 1/err_i**2).
    Снижает дисперсию (два по 18% → ~12.7%). Это лечит ШУМ, но НЕ смещение —
    применять только к согласованным замерам (см. док модуля).
    """
    vals = [float(v) for v in values]
    errs = [max(abs(float(e)), 1e-6) for e in errs_pct]
    weights = [1.0 / (e * e) for e in errs]
    wsum = sum(weights)
    fused = sum(w * v for w, v in zip(weights, vals)) / wsum
    fused_err = 1.0 / math.sqrt(sum(1.0 / (e * e) for e in errs))
    return fused, fused_err


def tilt_corrected_area(area, tilt_deg, cos_floor):
    """Площадь плоской области с поправкой за наклон: area / cos(tilt).

    tilt_deg=None → без изменений (наклон неизвестен). cos зажат снизу cos_floor
    (≈ наклон 60°): за пределами и закон cos, и сама оценка наклона ненадёжны
    (ср. отсечку аспекта 0.45 / ~63° в scale.detect_manhole_ellipses).
    """
    if area is None:
        return None
    if tilt_deg is None:
        return float(area)
    cos_t = max(math.cos(math.radians(float(tilt_deg))), float(cos_floor))
    return float(area) / cos_t


@dataclass
class ViewMeasurement:
    """Размеры ямы в ОДНОМ виде (из defect-блока single-image отчёта)."""
    image_name: str
    available: bool
    defect_class: str = "pothole"
    length_cm: float | None = None
    width_cm: float | None = None
    equivalent_diameter_cm: float | None = None
    area_cm2: float | None = None
    area_m2: float | None = None
    tilt_deg: float | None = None
    error_band_pct: float | None = None
    confidence: str | None = None
    eccentricity: float | None = None
    solidity: float | None = None

    @classmethod
    def from_defect(cls, defect: dict, image_name: str) -> "ViewMeasurement":
        """Собрать из defect-блока отчёта (metric + shape). tilt берётся из
        аддитивного metric['view_tilt_deg'], который пишет pipeline."""
        m = defect.get("metric", {}) or {}
        sh = defect.get("shape", {}) or {}
        return cls(
            image_name=image_name,
            available=bool(m.get("available")),
            defect_class=defect.get("class", "pothole"),
            length_cm=m.get("length_cm"),
            width_cm=m.get("width_cm"),
            equivalent_diameter_cm=m.get("equivalent_diameter_cm"),
            area_cm2=m.get("area_cm2"),
            area_m2=m.get("area_m2"),
            tilt_deg=m.get("view_tilt_deg"),
            error_band_pct=m.get("error_band_pct"),
            confidence=m.get("confidence"),
            eccentricity=sh.get("eccentricity"),
            solidity=sh.get("solidity"),
        )


@dataclass
class FusedMeasurement:
    """Слитая метрика двух видов. to_dict() зеркалит ключи single-image metric."""
    available: bool
    equivalent_diameter_cm: float | None = None
    length_cm: float | None = None
    width_cm: float | None = None
    area_cm2: float | None = None
    area_m2: float | None = None
    confidence: str | None = None
    error_band_pct: float | None = None
    cross_view: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        """Стабильная схема metric §8 + аддитивный вложенный cross_view.

        depth_* жёстко null/false: два косых кадра НЕ дают сертифицируемую
        глубину (это не Сценарий C). Ключи depth_* присутствуют, чтобы схема
        для потребителя совпадала с одиночным кадром.
        """
        return {
            "available": self.available,
            "equivalent_diameter_cm": self.equivalent_diameter_cm,
            "length_cm": self.length_cm,
            "width_cm": self.width_cm,
            "area_cm2": self.area_cm2,
            "area_m2": self.area_m2,
            "depth_cm": None,
            "depth_bucket": None,
            "depth_method": None,
            "depth_certifiable": False,
            "confidence": self.confidence,
            "error_band_pct": self.error_band_pct,
            "view_tilt_deg": None,  # per-view наклоны — в cross_view ниже
            "cross_view": self.cross_view,
        }


def fuse_pair(a: ViewMeasurement, b: ViewMeasurement,
              cfg: config.InferenceConfig = config.DEFAULT_INFERENCE) -> FusedMeasurement:
    """Слить два вида одной ямы в одну метрику (см. док модуля)."""
    if not (a.available and b.available):
        return FusedMeasurement(
            available=False,
            note="Слияние невозможно: хотя бы в одном виде нет масштаба — см не выдаются.",
            cross_view={"n_views": 2, "matched": True, "fused": False,
                        "reason": "no_scale_in_one_view"},
        )

    if a.area_cm2 is None or b.area_cm2 is None:
        return FusedMeasurement(
            available=False,
            note="Слияние невозможно: в одном из видов нет площади дефекта.",
            cross_view={"n_views": 2, "matched": True, "fused": False,
                        "reason": "missing_area"},
        )

    # Слияние валидно лишь для ОДНОЙ ямы. Если форма видов сильно расходится —
    # это, вероятно, РАЗНЫЕ ямы (или грубо разная сегментация); не сливаем, чтобы
    # не выдать высокоуверенное число по двум несвязанным дефектам (ревью).
    shape_div = max(_abs_diff(a.eccentricity, b.eccentricity),
                    _abs_diff(a.solidity, b.solidity))
    if shape_div > cfg.fusion_shape_tol:
        return FusedMeasurement(
            available=False,
            note=("Слияние не выполнено: форма видов расходится "
                  f"({shape_div:.2f} > {cfg.fusion_shape_tol}) — вероятно РАЗНЫЕ ямы."),
            cross_view={"n_views": 2, "matched": True, "fused": False,
                        "reason": "shape_mismatch", "shape_divergence": round(shape_div, 3)},
        )

    # Поправка площади за наклон применяется ТОЛЬКО когда наклон ИЗВЕСТЕН в ОБОИХ
    # видах (эталон-люк). Если хоть один вид без наклона (разметка/борт), площадь
    # НЕ корректируется НИ в одном виде (симметрично — иначе сравнение фабрикует
    # ложное расхождение), а полоса честно расширяется (ревью 2026-06-19).
    floor = cfg.fusion_cos_floor
    area_tilt_corrected = (a.tilt_deg is not None) and (b.tilt_deg is not None)
    if area_tilt_corrected:
        aa = tilt_corrected_area(a.area_cm2, a.tilt_deg, floor)
        ab = tilt_corrected_area(b.area_cm2, b.tilt_deg, floor)
    else:
        aa, ab = float(a.area_cm2), float(b.area_cm2)

    ea = a.error_band_pct if a.error_band_pct is not None else 30.0
    eb = b.error_band_pct if b.error_band_pct is not None else 30.0
    worse_band = max(ea, eb)

    # Согласие/расхождение — по ДЕ-РАКУРСНОЙ площади. Сырые length/width НЕ берём
    # в порог: оба вида укорочены ракурсом синфазно, и их «расхождение» — артефакт,
    # а не реальный конфликт (ревью 2026-06-20). «Разные ямы» ловит refuse-гейт по
    # форме (eccentricity/solidity) выше — он же отсекает чрезмерную разницу наклона.
    area_dis = abs(aa - ab) / max(aa, ab, 1e-9)
    len_dis = _rel_diff(a.length_cm, b.length_cm)   # только для прозрачности в JSON
    wid_dis = _rel_diff(a.width_cm, b.width_cm)

    # Сигмы для слияния ПЛОЩАДЕЙ — площадные: area ~ mm_per_px² → отн. ошибка ≈ 2e.
    # Значение слияния от равномасштабного множителя весов не меняется, а полоса
    # честно удваивается против линейной (док модуля).
    area_cm2, area_band = inverse_variance_fuse([aa, ab], [2.0 * ea, 2.0 * eb])
    area_m2 = round(area_cm2 / 10000.0, 4)

    # length/width — НИЖНЯЯ ОЦЕНКА: ракурс только УКОРАЧИВАЕТ план, поэтому берём
    # МАКС сырых значений видов (а не среднее, которое сильнее укороченный вид тянул
    # бы вниз и мог занизить ногу ГОСТ «длина ≥ 15 см»). eqd — из слитой площади.
    def _lower_bound(va, vb):
        vals = [v for v in (va, vb) if v is not None]
        return max(vals) if vals else None

    length_cm = _lower_bound(a.length_cm, b.length_cm)
    width_cm = _lower_bound(a.width_cm, b.width_cm)
    eqd_cm = math.sqrt(4.0 * area_cm2 / math.pi) if area_cm2 > 0 else None

    base_conf = _worse(a.confidence, b.confidence)
    either_low = "low" in (a.confidence, b.confidence)
    # Повышение уверенности оправдано лишь НЕЗАВИСИМЫМ согласием: хотя бы один вид
    # с ИЗВЕСТНЫМ наклоном (люк). Два вида по ОДНОЙ разметке (tilt=None в обоих)
    # делят один и тот же систематический сдвиг класса ширины — их согласие
    # неинформативно и НЕ должно повышать уверенность (ревью 2026-06-20).
    allow_promote = (a.tilt_deg is not None) or (b.tilt_deg is not None)

    note = ("length/width — нижняя оценка (макс сырых, не скорректированы за наклон); "
            + ("площадь скорректирована за наклон."
               if area_tilt_corrected else
               "наклон неизвестен — площадь НЕ скорректирована, полоса расширена."))

    # Полоса НИКОГДА не уже ХУДШЕГО одиночного вида: синфазное смещение наклона
    # слиянием не убирается — выгода идёт в уверенность, не в полосу (док модуля).
    # Всё ниже — в ПЛОЩАДНЫХ процентах (те же единицы, что area_band и area_dis).
    band_floor_area = 2.0 * worse_band
    if not area_tilt_corrected:
        band_floor_area = max(band_floor_area, cfg.fusion_uncorrected_area_band_pct)

    if area_dis <= cfg.fusion_agree_tol:
        agreement = "agree"
        conf = _promote(base_conf, 1) if allow_promote else base_conf
        if either_low:                      # никогда не до high, если вид low
            conf = _worse(conf, "medium")
        band_area = max(area_band, band_floor_area)
    elif area_dis >= cfg.fusion_disagree_max:
        agreement = "disagree"
        conf = "low"                        # тревога: измерение ненадёжно
        band_area = max(band_floor_area, area_dis * 100.0)
        note += f" РАСХОЖДЕНИЕ площади видов {area_dis * 100:.0f}% — измерение ненадёжно."
    else:
        agreement = "partial"
        conf = base_conf
        band_area = max(area_band, band_floor_area, area_dis * 100.0)

    # В контракт metric идёт линейный эквивалент (½ площадной) — семантика поля
    # совпадает с одиночным кадром; band_area ≥ 2·worse_band ⇒ band ≥ worse_band.
    band = band_area / 2.0
    note += (f" Полоса ошибки: ±{band:.0f}% линейные размеры, "
             f"±{band_area:.0f}% площадь.")

    cross_view = {
        "n_views": 2,
        "matched": True,
        "fused": True,
        "disagreement_pct": round(area_dis * 100.0, 1),   # порог считается по площади
        "area_disagreement_pct": round(area_dis * 100.0, 1),
        "area_error_band_pct": round(band_area, 1),   # площадная полоса (≈2× линейной)
        "length_disagreement_pct": round(len_dis * 100.0, 1),
        "width_disagreement_pct": round(wid_dis * 100.0, 1),
        "shape_divergence": round(shape_div, 3),
        "per_view_area_cm2": [round(aa, 1), round(ab, 1)],
        "per_view_tilt_deg": [a.tilt_deg, b.tilt_deg],
        "per_view_images": [a.image_name, b.image_name],
        "agreement": agreement,
        # Флаг отражает ФАКТИЧЕСКОЕ повышение (conf строго выше base_conf), а не
        # намерение: при потолке 'high' или капе either_low промоушен — no-op, и
        # заявлять его нечестно (аудит 2026-07-07).
        "confidence_promoted": bool(
            agreement == "agree"
            and _CONF_RANK[conf] > _CONF_RANK.get(base_conf, 0)),
        "fused_method": "inverse_variance_area + max_lower_bound_length",
        "area_tilt_corrected": area_tilt_corrected,
        "length_tilt_corrected": False,
        "length_is_lower_bound": True,
    }
    return FusedMeasurement(
        available=True,
        equivalent_diameter_cm=round(eqd_cm, 1) if eqd_cm is not None else None,
        length_cm=round(length_cm, 1) if length_cm is not None else None,
        width_cm=round(width_cm, 1) if width_cm is not None else None,
        area_cm2=round(area_cm2, 1),
        area_m2=area_m2,
        confidence=conf,
        error_band_pct=round(band, 1),
        cross_view=cross_view,
        note=note,
    )
