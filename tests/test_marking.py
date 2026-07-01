"""Юнит-тесты эталона по разметке (ГОСТ Р 51256) — классика, без сетей."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from road_defect import config
from road_defect.scale import scale_from_marking


def _road(h=1200, w=1600, level=100):
    """Тёмный «асфальт» равномерной яркости (краска должна быть ярче на запас)."""
    return np.full((h, w, 3), level, np.uint8)


def test_marking_width_is_short_side():
    # Продольная (вертикальная) белая полоса 40 px шириной x 600 px длиной.
    img = _road()
    cv2.rectangle(img, (800, 300), (840, 900), (255, 255, 255), -1)  # 40 wide
    ref = scale_from_marking(img, road_category="IV")
    assert ref.available
    # меряется ШИРИНА (40), не длина (600)
    assert abs(ref.measured_px - 40) < 12
    assert ref.subtype == "line_1_1"
    # IV: линия 1.1 = 80 мм → mm_per_px ≈ 80/40 = 2.0
    assert abs(ref.mm_per_px - config.GOST51256_LINE_1_1_BY_CATEGORY["IV"] / ref.measured_px) < 0.2
    assert ref.confidence in ("medium", "low") and ref.confidence != "high"


def test_marking_category_pins_width():
    img = _road()
    cv2.rectangle(img, (800, 300), (840, 900), (255, 255, 255), -1)
    ref_iv = scale_from_marking(img, road_category="IV")
    ref_ia = scale_from_marking(img, road_category="IA")
    assert ref_iv.known_mm == config.GOST51256_LINE_1_1_BY_CATEGORY["IV"]   # 80
    assert ref_ia.known_mm == config.GOST51256_LINE_1_1_BY_CATEGORY["IA"]   # 150


def test_marking_ambiguous_refuses():
    # Продольная полоса без категории дороги — ширина неоднозначна → отказ.
    img = _road()
    cv2.rectangle(img, (800, 300), (840, 900), (255, 255, 255), -1)
    ref = scale_from_marking(img, road_category=None)
    assert ref.available is False
    assert "неоднозн" in ref.note.lower()


def test_marking_stop_line_disambiguated():
    # Широкая ПОПЕРЕЧНАЯ (горизонтальная) белая полоса → СТОП-линия (400 мм).
    img = _road()
    cv2.rectangle(img, (300, 600), (1100, 660), (255, 255, 255), -1)  # 60 tall, horizontal
    ref = scale_from_marking(img, road_category="IV")
    assert ref.available
    assert ref.subtype == "stop_line"
    assert ref.known_mm == config.GOST51256_STOP_LINE_MM   # 400
    assert abs(ref.measured_px - 60) < 14


def test_marking_tapered_stripe_rejected():
    # Белая полоса с сильным таперингом (трапеция): ширина непостоянна →
    # единый масштаб ненадёжен → честный отказ даже при известной категории.
    img = _road()
    poly = np.array([[790, 300], [850, 300], [820, 900], [810, 900]], np.int32)
    cv2.fillPoly(img, [poly], (255, 255, 255))  # 60 px → 10 px, taper 6:1
    ref = scale_from_marking(img, road_category="IV")
    assert ref.available is False


def test_marking_none_on_plain_road():
    assert scale_from_marking(_road(), road_category="IV").available is False


def test_marking_confidence_always_low():
    # Класс ширины разметки неоднозначен → уверенность ЖЁСТКО low (не medium/high).
    img = _road()
    cv2.rectangle(img, (800, 300), (840, 900), (255, 255, 255), -1)
    ref = scale_from_marking(img, road_category="IV")
    assert ref.available and ref.confidence == "low"
    assert ref.error_band_pct == config.DEFAULT_INFERENCE.mark_error_band_pct


def test_marking_edge_stripe_rejected_no_dark_flank():
    # Штрих у самого края кадра: тёмный асфальт по ОБЕ стороны подтвердить нельзя
    # (одна сторона вне кадра) → не разметка.
    img = _road()
    cv2.rectangle(img, (0, 300), (40, 900), (255, 255, 255), -1)
    assert scale_from_marking(img, road_category="IV").available is False


def test_marking_bright_streak_without_dark_flanks_rejected():
    # Яркая полоса, у которой сбоку тоже ярко (не краска на тёмном асфальте):
    # положительный признак краски не выполняется → отказ.
    img = _road(level=90)
    cv2.rectangle(img, (700, 200), (1000, 1000), (245, 245, 245), -1)  # широкий светлый блок
    cv2.rectangle(img, (820, 200), (860, 1000), (255, 255, 255), -1)   # штрих внутри блока
    assert scale_from_marking(img, road_category="IV").available is False


def test_marking_stopline_needs_category():
    # Поперечная широкая полоса без категории дороги → отказ (нет якоря класса).
    img = _road()
    cv2.rectangle(img, (300, 600), (1100, 660), (255, 255, 255), -1)
    assert scale_from_marking(img, road_category=None).available is False
