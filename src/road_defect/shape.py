"""Дескрипторы формы дефекта из бинарной маски (пиксельное пространство).

Чистый модуль: numpy + skimage, без моделей. Покрыт юнит-тестами на синтетических
масках (круг → eq_diameter≈2R, eccentricity≈0; эллипс → известная eccentricity).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class ShapeDescriptors:
    """Метрики формы в пикселях. Безразмерные (eccentricity, solidity) —
    масштабонезависимы и выдаются как есть даже без эталона."""
    area_px2: float
    perimeter_px: float
    equivalent_diameter_px: float   # sqrt(4*area/pi) — отчётный «диаметр»
    major_axis_px: float
    minor_axis_px: float
    eccentricity: float             # 0 = круг, →1 вытянутость
    solidity: float                 # area / convex_hull_area  («выпуклость»)
    orientation_deg: float          # угол главной оси, градусы
    bbox_px: tuple                  # (min_row, min_col, max_row, max_col)
    centroid_px: tuple              # (row, col)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox_px"] = list(self.bbox_px)
        d["centroid_px"] = list(self.centroid_px)
        return d


def describe_mask(mask: np.ndarray) -> ShapeDescriptors | None:
    """Вернуть дескрипторы крупнейшей связной компоненты маски.

    mask — 2D-массив (bool/0..1/0..255). None, если маска пустая.
    """
    from skimage.measure import label, regionprops

    binary = np.asarray(mask) > 0
    if not binary.any():
        return None

    labeled = label(binary)
    props = regionprops(labeled)
    if not props:
        return None
    # крупнейшая компонента — сам дефект (мелкий шум игнорируем)
    region = max(props, key=lambda r: r.area)

    area = float(region.area)
    eq_diam = math.sqrt(4.0 * area / math.pi)
    return ShapeDescriptors(
        area_px2=area,
        perimeter_px=float(region.perimeter),
        equivalent_diameter_px=eq_diam,
        major_axis_px=float(region.axis_major_length),
        minor_axis_px=float(region.axis_minor_length),
        eccentricity=float(region.eccentricity),
        solidity=float(region.solidity),
        orientation_deg=float(math.degrees(region.orientation)),
        bbox_px=tuple(int(v) for v in region.bbox),
        centroid_px=(float(region.centroid[0]), float(region.centroid[1])),
    )


def apply_scale(shape: ShapeDescriptors, mm_per_px: float) -> dict:
    """Перевести пиксельные размеры в метрические (см) при известном mm/px.

    Длины × mm/px, площади × (mm/px)². Возвращает компактный dict для отчёта.
    """
    mm = mm_per_px
    return {
        "equivalent_diameter_cm": round(shape.equivalent_diameter_px * mm / 10.0, 1),
        "length_cm": round(shape.major_axis_px * mm / 10.0, 1),
        "width_cm": round(shape.minor_axis_px * mm / 10.0, 1),
        "area_cm2": round(shape.area_px2 * (mm ** 2) / 100.0, 1),
        "area_m2": round(shape.area_px2 * (mm ** 2) / 1_000_000.0, 4),
    }
