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


def test_cm_sizes_without_available_are_diagnosed():
    # Ревью 2026-07-15: проверялся только equivalent_diameter_cm — length/width/
    # area с сантиметрами при available=false проходили валидатор молча.
    metric = {"available": False, "depth_cm": None, "depth_certifiable": False,
              "equivalent_diameter_cm": None, "length_cm": 55.0,
              "width_cm": 30.0, "area_m2": 0.12}
    errors = vo.check_report(Path("x.json"), _report(metric=metric))
    joined = " ".join(errors)
    for key in ("length_cm", "width_cm", "area_m2"):
        assert key in joined, f"{key} не диагностирован: {errors}"


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


def _camera_metric(**over) -> dict:
    """Валидный metric источника «высота камеры» (plane_scale, 2026-07-16)."""
    m = {"available": True, "depth_cm": None, "depth_certifiable": False,
         "equivalent_diameter_cm": 40.0, "length_cm": 80.0, "width_cm": 50.0,
         "area_cm2": 3000.0, "area_m2": 0.3, "confidence": "low",
         "error_band_pct": 35.0, "scale_source": "camera_height",
         "camera_scale": {"horizon_method": "sky_anchor", "theta_deg": 11.0,
                          "mm_per_px_transverse": 2.3, "f_source": "exif_fl35",
                          "certifiable": False}}
    m.update(over)
    return m


def test_camera_height_metric_valid_passes():
    assert vo.check_report(Path("x.json"), _report(metric=_camera_metric())) == []


def test_camera_height_confidence_must_be_low():
    errors = vo.check_report(
        Path("x.json"), _report(metric=_camera_metric(confidence="medium")))
    assert any("'low'" in e for e in errors)


def test_camera_height_requires_error_band():
    errors = vo.check_report(
        Path("x.json"), _report(metric=_camera_metric(error_band_pct=None)))
    assert any("полос" in e for e in errors)


def test_camera_height_certifiable_must_be_false():
    m = _camera_metric()
    m["camera_scale"] = {**m["camera_scale"], "certifiable": True}
    errors = vo.check_report(Path("x.json"), _report(metric=m))
    assert any("certifiable" in e for e in errors)


def test_camera_scale_block_requires_source_label():
    errors = vo.check_report(
        Path("x.json"), _report(metric=_camera_metric(scale_source=None)))
    assert any("scale_source" in e for e in errors)


def test_available_with_source_key_requires_known_source():
    # available=true при присутствующем, но пустом scale_source — «откуда см?»
    m = _camera_metric(scale_source=None)
    m["camera_scale"] = None
    errors = vo.check_report(Path("x.json"), _report(metric=m))
    assert any("без источника масштаба" in e for e in errors)


def test_mixed_fused_metric_passes_and_is_gated():
    # fused-пара «эталон+камера»: провенанс mixed, те же инварианты
    m = _camera_metric(scale_source="mixed")
    assert vo.check_report(Path("x.json"), _report(metric=m)) == []
    errors = vo.check_report(
        Path("x.json"),
        _report(metric=_camera_metric(scale_source="mixed",
                                      confidence="medium")))
    assert any("'low'" in e for e in errors)


def test_camera_scale_requires_block_and_known_method():
    m = _camera_metric(camera_scale=None)
    errors = vo.check_report(Path("x.json"), _report(metric=m))
    assert any("без блока camera_scale" in e for e in errors)
    m = _camera_metric()
    m["camera_scale"] = {**m["camera_scale"], "horizon_method": "guess"}
    errors = vo.check_report(Path("x.json"), _report(metric=m))
    assert any("horizon_method" in e for e in errors)
    m = _camera_metric()
    del m["camera_scale"]["mm_per_px_transverse"]
    errors = vo.check_report(Path("x.json"), _report(metric=m))
    assert any("mm_per_px_transverse" in e for e in errors)


def test_camera_band_bool_is_diagnosed():
    errors = vo.check_report(
        Path("x.json"), _report(metric=_camera_metric(error_band_pct=True)))
    assert any("полос" in e for e in errors)


def test_estimate_available_requires_low_confidence():
    m = {"available": False, "depth_cm": None, "depth_certifiable": False,
         "equivalent_diameter_cm": None,
         "depth_cm_estimate": {"available": True, "certifiable": False,
                               "confidence": None, "point_cm": 1.0,
                               "low_cm": 0.5, "high_cm": 2.0}}
    errors = vo.check_report(Path("x.json"), _report(metric=m))
    assert any("требует" in e and "low" in e for e in errors)
