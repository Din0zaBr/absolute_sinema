"""Дескрипторы формы дефекта из бинарной маски (пиксельное пространство).

Чистый модуль: numpy + skimage + cv2-геометрия, без моделей. Покрыт юнит-тестами
на синтетических масках (круг → eq_diameter≈2R, eccentricity≈0; полоса →
length_px ≈ истинной длине).

Конвенции отчёта (важно для ГОСТ-метрики):
  • length_px/width_px — ФИЗИЧЕСКАЯ протяжённость дефекта в плане: стороны
    минимального повёрнутого прямоугольника (cv2.minAreaRect) по ВСЕМ пикселям
    маски. Ось эллипса инерции regionprops для этого не годится: для
    равномерной полосы axis_major_length = 2L/√3 ≈ 1.155·L — систематическое
    завышение на 15.5%, которое ложно переключало бы ногу ГОСТ «длина ≥ 15 см»
    (находка адверсариального ревью 2026-06-12, подтверждена численно).
  • area_px2 — суммарная площадь ВСЕХ связных компонент маски: совпадает с
    mask_rle отчёта; фрагментированная маска трещины не теряет площадь.
  • major/minor_axis_px, eccentricity, solidity, orientation — дескрипторы
    ФОРМЫ крупнейшей компоненты (эллипс инерции, как и раньше).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class ShapeDescriptors:
    """Метрики формы в пикселях. Безразмерные (eccentricity, solidity) —
    масштабонезависимы и выдаются как есть даже без эталона."""
    area_px2: float                 # суммарно по всем компонентам (== mask_rle)
    perimeter_px: float             # периметр крупнейшей компоненты
    equivalent_diameter_px: float   # sqrt(4*area/pi) — отчётный «диаметр»
    length_px: float                # длинная сторона minAreaRect всей маски
    width_px: float                 # короткая сторона minAreaRect всей маски
    major_axis_px: float            # ось эллипса инерции (дескриптор формы)
    minor_axis_px: float
    eccentricity: float             # 0 = круг, →1 вытянутость
    solidity: float                 # area / convex_hull_area  («выпуклость»)
    orientation_deg: float          # угол главной оси, градусы
    bbox_rowcol_px: tuple           # (min_row, min_col, max_row, max_col)
    centroid_rowcol_px: tuple       # (row, col) — оси явно в имени, чтобы не
                                    # путать с bbox_px дефекта (x, y, w, h)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox_rowcol_px"] = list(self.bbox_rowcol_px)
        d["centroid_rowcol_px"] = list(self.centroid_rowcol_px)
        return d


def describe_mask(mask: np.ndarray) -> ShapeDescriptors | None:
    """Вернуть дескрипторы маски дефекта.

    mask — 2D-массив (bool/0..1/0..255). None, если маска пустая.
    Площадь и протяжённость — по всей маске; дескрипторы формы (эллипс
    инерции, solidity) — по крупнейшей связной компоненте.
    """
    import cv2
    from skimage.measure import label, regionprops

    binary = np.asarray(mask) > 0
    if not binary.any():
        return None

    labeled = label(binary)
    props = regionprops(labeled)
    if not props:
        return None
    # крупнейшая компонента задаёт форму (мелкий шум не влияет на эллипс)
    region = max(props, key=lambda r: r.area)

    area = float(binary.sum())
    eq_diam = math.sqrt(4.0 * area / math.pi)

    # Физическая протяжённость: минимальный повёрнутый прямоугольник всей маски.
    pts = cv2.findNonZero(binary.astype(np.uint8))
    (_, _), (d1, d2), _ = cv2.minAreaRect(pts)
    # minAreaRect мерит между центрами крайних пикселей — +1 на их ширину
    length_px, width_px = max(d1, d2) + 1.0, min(d1, d2) + 1.0

    return ShapeDescriptors(
        area_px2=area,
        perimeter_px=float(region.perimeter),
        equivalent_diameter_px=eq_diam,
        length_px=float(length_px),
        width_px=float(width_px),
        major_axis_px=float(region.axis_major_length),
        minor_axis_px=float(region.axis_minor_length),
        eccentricity=float(region.eccentricity),
        solidity=float(region.solidity),
        orientation_deg=float(math.degrees(region.orientation)),
        bbox_rowcol_px=tuple(int(v) for v in region.bbox),
        centroid_rowcol_px=(float(region.centroid[0]), float(region.centroid[1])),
    )


def apply_scale(shape: ShapeDescriptors, mm_per_px: float) -> dict:
    """Перевести пиксельные размеры в метрические (см) при известном mm/px.

    Длины × mm/px, площади × (mm/px)². length/width — физическая протяжённость
    (minAreaRect), а не ось эллипса инерции (см. док модуля).
    """
    mm = mm_per_px
    return {
        "equivalent_diameter_cm": round(shape.equivalent_diameter_px * mm / 10.0, 1),
        "length_cm": round(shape.length_px * mm / 10.0, 1),
        "width_cm": round(shape.width_px * mm / 10.0, 1),
        "area_cm2": round(shape.area_px2 * (mm ** 2) / 100.0, 1),
        "area_m2": round(shape.area_px2 * (mm ** 2) / 1_000_000.0, 4),
    }
