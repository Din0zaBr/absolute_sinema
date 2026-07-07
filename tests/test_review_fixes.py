"""Регрессы под аудит 2026-07-07 (docs/HANDOFF_FABLE.md): каждая находка —
падающий тест до фикса, зелёный после.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from road_defect import imgio, severity
from road_defect.detect import Detection, merge_detections
from road_defect.report import _finite_or_none
from road_defect.shape import describe_mask


def _det(cls, x, y, w, h, conf=0.8):
    return Detection(cls_name=cls, raw_label=cls, confidence=conf,
                     bbox_xywh=(float(x), float(y), float(w), float(h)))


# --- P1 #2: NaN/Inf не должен читаться как «в норме» ------------------------
def test_severity_nan_is_indeterminate_not_compliant():
    v = severity.classify(length_cm=float("nan"), area_m2=float("nan"),
                          depth_cm=float("nan"))
    assert v.non_conforming == "indeterminate_without_depth"
    assert v.length_exceeds is None and v.area_exceeds is None
    assert v.depth_exceeds is None


def test_severity_inf_length_is_indeterminate():
    v = severity.classify(length_cm=float("inf"), area_m2=0.1, depth_cm=None)
    assert v.length_exceeds is None            # inf → неизвестно, не True
    assert v.non_conforming == "indeterminate_without_depth"


def test_severity_finite_noncompliant_still_yes():
    v = severity.classify(length_cm=20.0, area_m2=0.1, depth_cm=6.0)
    assert v.non_conforming == "yes"


# --- P1 #2: JSON-контракт без NaN/Inf --------------------------------------
def test_finite_or_none_sanitizes_nonfinite():
    got = _finite_or_none({"a": float("nan"), "b": [1.0, float("inf")], "c": 2,
                           "d": np.float32("nan")})
    assert got == {"a": None, "b": [1.0, None], "c": 2, "d": None}


# --- P2 #4: слияние ансамбля дедуплицирует один объект ----------------------
def test_merge_dedups_moderate_iou_same_class():
    primary = [_det("pothole", 100, 100, 100, 100)]
    secondary = [_det("pothole", 140, 100, 100, 100, conf=0.9)]  # IoU ~0.43
    assert len(merge_detections(primary, secondary)) == 1


def test_merge_dedups_contained_box_low_iou():
    # seg-бокс целиком внутри обычного (IoU ~0.16) — ловится containment по площади
    primary = [_det("pothole", 100, 100, 200, 200)]
    secondary = [_det("pothole", 150, 150, 80, 80, conf=0.9)]
    assert len(merge_detections(primary, secondary)) == 1


def test_merge_keeps_genuinely_separate():
    primary = [_det("pothole", 0, 0, 50, 50)]
    secondary = [_det("pothole", 400, 400, 50, 50, conf=0.9)]
    assert len(merge_detections(primary, secondary)) == 2


def test_merge_keeps_adjacent_distinct_despite_center_in_box():
    # Два РАЗНЫХ соседних дефекта: центр меньшего попал в большой бокс, но по
    # площади он вложен лишь частично (IoU низкий, containment < 0.8) → НЕ склеивать.
    primary = [_det("pothole", 0, 0, 200, 200)]
    secondary = [_det("pothole", 150, 150, 100, 100, conf=0.9)]  # center (200,200)
    assert len(merge_detections(primary, secondary)) == 2


# --- P3 #15: describe_mask(None) → None, не крэш ----------------------------
def test_describe_mask_none_returns_none():
    assert describe_mask(None) is None


# --- P3 #16: запись без суффикса даёт валидный .jpg ------------------------
def test_write_image_without_suffix_becomes_jpg(tmp_path):
    p = tmp_path / "noext"
    assert imgio.write_image(p, np.zeros((4, 4, 3), np.uint8))
    assert (tmp_path / "noext.jpg").exists()
    assert imgio.read_image(tmp_path / "noext.jpg") is not None


# --- P1 #1: путь люка отвергает светлую/гладкую «яму», принимает тёмный диск -
def test_manhole_appearance_gate_rejects_bright_disk_accepts_dark():
    from road_defect import scale

    dark = np.full((220, 220, 3), 185, np.uint8)      # светлый асфальт
    cv2.circle(dark, (110, 110), 45, (28, 28, 28), -1)  # тёмный литой люк
    ell = ((110.0, 110.0), (90.0, 90.0), 0.0)
    assert scale._ellipse_looks_like_manhole(dark, ell) is True

    bright = np.full((220, 220, 3), 110, np.uint8)     # тёмный асфальт
    cv2.circle(bright, (110, 110), 45, (165, 165, 165), -1)  # СВЕТЛАЯ заплатка
    assert scale._ellipse_looks_like_manhole(bright, ell) is False
