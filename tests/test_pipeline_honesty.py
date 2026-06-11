"""Honest-mode тесты (дизайн §11): в Сценариях A/B движок никогда не выдаёт
сантиметры глубины — depth_cm = null, depth_certifiable = false, а вердикт ГОСТ
без глубины остаётся indeterminate_without_depth.

Детектор и масштаб замоканы: тесты гоняют реальный код pipeline без сетей.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from road_defect import imgio
from road_defect import scale as scale_mod
from road_defect.detect import Detection
from road_defect.pipeline import DefectPipeline
from road_defect.scale import ReferenceMeasurement


class FakeDetector:
    """Одна «яма» 120x100 px с готовой маской в координатах кадра."""

    is_fallback = False

    def load(self):
        return self

    def detect(self, img):
        H, W = img.shape[:2]
        mask = np.zeros((H, W), bool)
        mask[100:200, 100:220] = True
        return [Detection(cls_name="pothole", raw_label="D40", confidence=0.9,
                          bbox_xywh=(100.0, 100.0, 120.0, 100.0), mask=mask)]


def _run(tmp_path, monkeypatch, with_ref: bool) -> dict:
    img = np.full((400, 400, 3), 120, np.uint8)
    path = tmp_path / "frame.jpg"
    assert imgio.write_image(path, img)

    pipe = DefectPipeline(use_depth=False)
    pipe.detector = FakeDetector()
    ref = (ReferenceMeasurement(
               available=True, type="manhole_gost3634_cover", known_mm=646,
               measured_px=323.0, mm_per_px=2.0, confidence="high",
               error_band_pct=12.0)
           if with_ref else ReferenceMeasurement(available=False))
    monkeypatch.setattr(scale_mod, "scale_from_manhole", lambda *a, **k: ref)

    report, _, _ = pipe.analyze_image(path)
    return report


def test_scenario_a_never_reports_cm(tmp_path, monkeypatch):
    rep = _run(tmp_path, monkeypatch, with_ref=False)
    assert rep["mode"] == "single"
    assert rep["scale"]["available"] is False
    (d,) = rep["defects"]
    m = d["metric"]
    assert m["available"] is False
    assert m["depth_cm"] is None
    assert m["depth_certifiable"] is False
    assert m["equivalent_diameter_cm"] is None  # без эталона см не выдаются
    assert d["severity"]["non_conforming"] == "indeterminate_without_depth"


def test_scenario_b_sizes_yes_depth_no(tmp_path, monkeypatch):
    rep = _run(tmp_path, monkeypatch, with_ref=True)
    assert rep["mode"] == "single_with_reference"
    (d,) = rep["defects"]
    m = d["metric"]
    assert m["available"] is True
    assert m["equivalent_diameter_cm"] is not None  # размеры в см есть
    # ...но глубина в см не выдаётся НИКОГДА (только Сценарий C/датчик):
    assert m["depth_cm"] is None
    assert m["depth_certifiable"] is False
    assert d["severity"]["depth_exceeds"] is None
    assert d["severity"]["non_conforming"] == "indeterminate_without_depth"
    assert d["severity"]["repair_deadline_days"] is None


def test_scale_block_follows_contract(tmp_path, monkeypatch):
    rep = _run(tmp_path, monkeypatch, with_ref=True)
    scale = rep["scale"]
    assert scale["reference"]["known_mm"] == 646
    assert scale["reference"]["type"] == "manhole_gost3634_cover"
    assert isinstance(scale["homography_applied"], bool)
