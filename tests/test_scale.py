"""Юнит-тесты масштабной привязки (без нейросетей)."""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from road_defect import config
from road_defect.scale import (
    ReferenceMeasurement,
    detect_manhole_circle,
    homography_from_4_points,
    measure_distance_mm,
    scale_from_manhole,
)


def _synthetic_manhole(size=(3000, 4000), center=(2000, 1500), r=600):
    """Кадр «асфальта» с тёмным диском-люком; размеры — как у 12-МП фото."""
    h, w = size
    img = np.full((h, w, 3), 128, np.uint8)
    cv2.circle(img, center, r, (40, 40, 40), -1)
    return img


def test_detect_manhole_on_large_photo():
    # Регрессия: фикс. радиусы 25-400 px теряли люк на полноразмерных фото.
    img = _synthetic_manhole()
    c = detect_manhole_circle(img)
    assert c is not None
    cx, cy, r = c
    assert abs(cx - 2000) < 60 and abs(cy - 1500) < 60
    assert abs(r - 600) < 60


def test_no_manhole_on_plain_image():
    img = np.full((480, 640, 3), 128, np.uint8)
    assert detect_manhole_circle(img) is None


def test_scale_from_manhole_synthetic():
    img = _synthetic_manhole()
    ref = scale_from_manhole(img)
    assert ref.available
    # Якорь по умолчанию — обод крышки закрытого люка 646 мм.
    assert ref.known_mm == config.GOST3634_COVER_OUTER_MM
    assert math.isclose(ref.mm_per_px, 646.0 / 1200.0, rel_tol=0.1)
    assert ref.circle_px is not None and ref.ellipse_px is not None


def test_reference_to_dict_matches_contract():
    # Контракт §8: scale.reference вложенный, homography_applied — bool.
    d = ReferenceMeasurement(available=False).to_dict()
    assert {"available", "mm_per_px", "reference",
            "homography_applied", "error_band_pct"} <= set(d)
    assert isinstance(d["homography_applied"], bool)
    assert {"type", "known_mm", "measured_px"} <= set(d["reference"])
    # геометрия overlay в JSON не утекает
    assert "circle_px" not in d and "ellipse_px" not in d


def test_homography_recovers_known_distance():
    # Прямоугольный эталон 1000x500 мм, спроецированный с перспективой.
    world = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float32)
    image = np.array([[100, 120], [520, 90], [560, 410], [80, 450]], dtype=np.float32)
    H = homography_from_4_points(image, world)

    # Расстояние между углами 0 и 1 в мире = 1000 мм; проверяем через H.
    d = measure_distance_mm(H, image[0], image[1])
    assert math.isclose(d, 1000.0, rel_tol=1e-3)

    d2 = measure_distance_mm(H, image[1], image[2])
    assert math.isclose(d2, 500.0, rel_tol=1e-3)


def test_homography_diagonal():
    world = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float32)
    image = world.copy()  # тривиальная (identity-подобная) проекция
    H = homography_from_4_points(image, world)
    diag = measure_distance_mm(H, image[0], image[2])
    assert math.isclose(diag, math.hypot(1000, 500), rel_tol=1e-3)
