"""Валидатор инвариантов отчётов: битый вход — диагностика, не трейсбек
(ревью 2026-07-02: отсутствующий bbox_px ронял валидатор KeyError)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validate_outputs", ROOT / "scripts" / "validate_outputs.py")
vo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vo)

W, H = 100, 80


def _report(**defect_overrides) -> dict:
    defect = {
        "id": 1, "class": "pothole", "confidence": 0.9,
        "bbox_px": [10.0, 10.0, 30.0, 20.0],
        "mask_rle": {"size": [H, W], "counts": [W * H]},
        "metric": {"available": False, "depth_cm": None,
                   "depth_certifiable": False, "equivalent_diameter_cm": None},
        "severity": {"non_conforming": "indeterminate_without_depth"},
    }
    defect.update(defect_overrides)
    return {
        "image": "x.jpg", "mode": "single", "image_size_px": [W, H],
        "scale": {"available": False, "homography_applied": False,
                  "reference": {"type": None, "known_mm": None,
                                "measured_px": None}},
        "defects": [defect], "location": {"available": False},
        "warnings": [], "engine_version": "test",
    }


def test_valid_report_passes():
    assert vo.check_report(Path("x.json"), _report()) == []


def test_missing_bbox_is_diagnosed_not_keyerror():
    rep = _report()
    del rep["defects"][0]["bbox_px"]
    errors = vo.check_report(Path("x.json"), rep)   # не должен кинуть KeyError
    assert any("bbox_px" in e for e in errors)


def test_malformed_bbox_is_diagnosed():
    errors = vo.check_report(Path("x.json"), _report(bbox_px=[1, 2, 3]))
    assert any("bbox_px" in e for e in errors)
    errors = vo.check_report(Path("x.json"), _report(bbox_px="10,10,30,20"))
    assert any("bbox_px" in e for e in errors)


def test_bool_and_nan_bbox_are_diagnosed():
    # Ревью 2026-07-03: bool — подкласс int (json true), NaN — float, любое
    # сравнение с ним ложно — оба класса битости проходили молча.
    errors = vo.check_report(Path("x.json"), _report(bbox_px=[True] * 4))
    assert any("bbox_px" in e for e in errors)
    errors = vo.check_report(Path("x.json"),
                             _report(bbox_px=[float("nan")] * 4))
    assert any("bbox_px" in e for e in errors)


def test_out_of_frame_bbox_still_flagged():
    errors = vo.check_report(Path("x.json"),
                             _report(bbox_px=[90.0, 70.0, 30.0, 20.0]))
    assert any("вне кадра" in e for e in errors)


def test_depth_cm_violation_flagged():
    rep = _report()
    rep["defects"][0]["metric"]["depth_cm"] = 5.0
    errors = vo.check_report(Path("x.json"), rep)
    assert any("honest-mode" in e for e in errors)
