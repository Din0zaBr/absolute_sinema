"""Юнит-тесты дескрипторов формы на синтетических масках."""
import math

import numpy as np

from road_defect.shape import describe_mask, apply_scale


def _disk(radius, size=400):
    yy, xx = np.ogrid[:size, :size]
    c = size // 2
    return ((yy - c) ** 2 + (xx - c) ** 2) <= radius ** 2


def test_circle_equivalent_diameter():
    r = 80
    mask = _disk(r)
    sd = describe_mask(mask)
    # эквивалентный диаметр круга ≈ 2R
    assert math.isclose(sd.equivalent_diameter_px, 2 * r, rel_tol=0.03)
    # круг почти не эксцентричен
    assert sd.eccentricity < 0.15
    # выпуклость близка к 1
    assert sd.solidity > 0.95


def test_ellipse_eccentricity():
    size = 400
    yy, xx = np.ogrid[:size, :size]
    c = size // 2
    a, b = 120, 60  # большая/малая полуось
    mask = (((xx - c) / a) ** 2 + ((yy - c) / b) ** 2) <= 1.0
    sd = describe_mask(mask)
    expected_ecc = math.sqrt(1 - (b / a) ** 2)
    assert math.isclose(sd.eccentricity, expected_ecc, rel_tol=0.1)
    # большая ось длиннее малой
    assert sd.major_axis_px > sd.minor_axis_px


def test_empty_mask_returns_none():
    assert describe_mask(np.zeros((50, 50), bool)) is None


def test_apply_scale_units():
    mask = _disk(50)
    sd = describe_mask(mask)
    # 2 мм/px → диаметр 100px ≈ 200мм = 20см
    scaled = apply_scale(sd, mm_per_px=2.0)
    assert math.isclose(scaled["equivalent_diameter_cm"], 20.0, rel_tol=0.05)
    assert scaled["area_m2"] > 0
