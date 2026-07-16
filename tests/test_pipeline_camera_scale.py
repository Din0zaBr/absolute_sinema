"""Ветка camera_height в pipeline БЕЗ моделей (ревью 2026-07-16: дыра
покрытия): FakeDetector с маской-квадом на полотне + FakeDepth с
аналитической картой + замоканный эталон. Плюс регрессия BLOCKER'а слияния:
pair-режим обязан сохранять провенанс camera_height и не отмывать low.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from road_defect import config, imgio
from road_defect import scale as scale_mod
from road_defect.detect import Detection
from road_defect.fusion import ViewMeasurement, fuse_pair
from road_defect.pipeline import DefectPipeline
from road_defect.scale import ReferenceMeasurement

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validate_outputs", ROOT / "scripts" / "validate_outputs.py")
vo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vo)

W0, H0, F_PX, H_M = 4096, 1844, 3025.0, 1.70


def _normal(theta_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    th, r = np.radians(theta_deg), np.radians(roll_deg)
    return np.array([np.sin(r) * np.cos(th), -np.cos(r) * np.cos(th),
                     -np.sin(th)])


def _synthetic_depth(n: np.ndarray, gw: int = 1024) -> np.ndarray:
    k = W0 / gw
    gh = int(round(H0 / k))
    xs, ys = np.meshgrid(np.arange(gw), np.arange(gh))
    u, v = (xs + 0.5) * k, (ys + 0.5) * k
    p = np.stack([(u - W0 / 2.0) / F_PX, (v - H0 / 2.0) / F_PX,
                  np.ones_like(u)], axis=-1)
    q = np.maximum(-(p @ n) / H_M, 0.0)
    return (7.0 * q + 0.3).astype(np.float32)


def _quad_mask(n: np.ndarray, dist_m: float, half_w: float = 0.20,
               half_l: float = 0.15) -> np.ndarray:
    e1 = np.cross(n, np.array([0.0, 0.0, 1.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2f = e2 if e2[2] > 0 else -e2
    G = -H_M * n + dist_m * e2f
    pts = []
    for a, b in ((-half_w, -half_l), (half_w, -half_l),
                 (half_w, half_l), (-half_w, half_l)):
        P = G + a * e1 + b * e2f
        pts.append([F_PX * P[0] / P[2] + W0 / 2.0,
                    F_PX * P[1] / P[2] + H0 / 2.0])
    mask = np.zeros((H0, W0), dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(np.round(pts)).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


class MaskDetector:
    is_fallback = False

    def __init__(self, mask):
        self._m = mask

    def load(self):
        return self

    def detect(self, img):
        ys, xs = np.nonzero(self._m)
        bbox = (float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
        return [Detection(cls_name="pothole", raw_label="D40", confidence=0.9,
                          bbox_xywh=bbox, mask=self._m)]


class FakeDepth:
    """frame_depth -> аналитическая карта; relative_bucket -> канонический
    below-noise результат ring_plane_bucket (сигнал ниже шума)."""
    available = True

    def __init__(self, dmap):
        self._d = dmap

    def frame_depth(self, img):
        return self._d

    def relative_bucket(self, img, mask):
        return {"depth_cm": None, "depth_bucket": "shallow",
                "depth_certifiable": False,
                "method": "depth_anything_v2_small_ring_plane:below_local_noise",
                "_rel_norm": 0.0, "_rel_upper": 0.05,
                "_width_px": 120.0, "_px_scale_to_orig": 4.0}


def _camera_pipe(mask, dmap, monkeypatch, ref=None, prior=None):
    cfg = config.InferenceConfig(camera_scale=config.CameraHeightScaleConfig(
        height_m=H_M, f_px=F_PX, pitch_prior_deg=prior))
    pipe = DefectPipeline(cfg=cfg, use_depth=True)
    pipe.detector = MaskDetector(mask)
    pipe.depth = FakeDepth(dmap)
    monkeypatch.setattr(
        scale_mod, "scale_from_manhole",
        lambda *a, **k: ref or ReferenceMeasurement(available=False))
    return pipe


def _frame(tmp_path, name="frame.jpg"):
    img = np.full((H0, W0, 3), 120, np.uint8)
    path = tmp_path / name
    assert imgio.write_image(path, img)
    return path


def test_camera_metric_happy_path_sky_anchor(tmp_path, monkeypatch):
    n_true = _normal(10.0, 2.0)
    pipe = _camera_pipe(_quad_mask(n_true, 4.0), _synthetic_depth(n_true),
                        monkeypatch)
    report, _, _ = pipe.analyze_image(_frame(tmp_path))
    (d,) = report["defects"]
    m = d["metric"]
    assert m["available"] is True
    assert m["scale_source"] == "camera_height"
    assert m["confidence"] == "low"
    assert m["error_band_pct"] > 0
    assert m["depth_cm"] is None and m["depth_certifiable"] is False
    cam = m["camera_scale"]
    assert cam["horizon_method"] == "sky_anchor"
    assert cam["f_source"] == "config"
    assert cam["certifiable"] is False
    assert cam["mm_per_px_transverse"] > 0
    # оценка глубины посчитана от camera-mm/px (ниже шума -> верхняя граница)
    est = m["depth_cm_estimate"]
    assert est["available"] is True and est["upper_bound_cm"] > 0
    assert est["confidence"] == "low" and est["certifiable"] is False
    # вердикт ГОСТ атрибутирован как оценка
    assert "оценка от высоты камеры" in d["severity"]["note"]
    assert any("Масштаб от высоты камеры" in w for w in report["warnings"])
    # связка с контрактом §8: валидатор не находит нарушений
    assert vo.check_report(Path("x.json"), report) == []


def test_camera_defect_refusal_keeps_metric_empty(tmp_path, monkeypatch):
    n_true = _normal(10.0)
    pipe = _camera_pipe(_quad_mask(n_true, 60.0, 1.5, 1.5),
                        _synthetic_depth(n_true), monkeypatch)
    report, _, _ = pipe.analyze_image(_frame(tmp_path))
    (d,) = report["defects"]
    m = d["metric"]
    assert m["available"] is False
    for key in ("length_cm", "width_cm", "area_cm2", "area_m2",
                "equivalent_diameter_cm"):
        assert m[key] is None
    assert m["scale_source"] is None and m["camera_scale"] is None
    assert m["depth_cm_estimate"]["reason"] == "no_scale_reference"
    assert any("pit_too_close_to_horizon" in w for w in report["warnings"])
    assert vo.check_report(Path("x.json"), report) == []


def test_reference_has_priority_over_camera(tmp_path, monkeypatch):
    n_true = _normal(10.0)
    ref = ReferenceMeasurement(
        available=True, type="manhole_gost3634_cover", known_mm=646,
        measured_px=323.0, mm_per_px=2.0, confidence="high",
        error_band_pct=12.0)
    pipe = _camera_pipe(_quad_mask(n_true, 4.0), _synthetic_depth(n_true),
                        monkeypatch, ref=ref)
    report, _, _ = pipe.analyze_image(_frame(tmp_path))
    (d,) = report["defects"]
    m = d["metric"]
    assert m["scale_source"] == "reference"
    assert m["camera_scale"] is None
    assert not any("Масштаб от высоты камеры" in w for w in report["warnings"])
    assert "оценка от высоты камеры" not in (d["severity"]["note"] or "")


def test_pair_two_camera_views_keep_provenance_and_low(tmp_path, monkeypatch):
    # Регрессия BLOCKER'а ревью 2026-07-16: fused-metric терял scale_source/
    # camera_scale, и валидатор был слеп к pair-отчётам.
    n_true = _normal(10.0, 2.0)
    pipe = _camera_pipe(_quad_mask(n_true, 4.0), _synthetic_depth(n_true),
                        monkeypatch)
    pa, pb = _frame(tmp_path, "front.jpg"), _frame(tmp_path, "back.jpg")
    report, _, _, _ = pipe.analyze_pair(pa, pb)
    assert report["mode"] == "two_view_fused"
    m = report["defects"][0]["metric"]
    assert m["scale_source"] == "camera_height"
    assert m["confidence"] == "low"
    assert m["camera_scale"] is not None
    assert m["camera_scale"]["certifiable"] is False
    assert m["cross_view"]["confidence_promoted"] is False
    assert "оценка от высоты камеры" in report["defects"][0]["severity"]["note"]
    assert vo.check_report(Path("x.json"), report) == []


def test_fuse_mixed_reference_and_camera_stays_low():
    # Смешанная пара «люк (medium, tilt) + камера (low)» раньше отмывалась
    # до medium промоушеном по tilt эталонного вида.
    common = dict(available=True, length_cm=50.0, width_cm=30.0,
                  equivalent_diameter_cm=40.0, area_cm2=1200.0, area_m2=0.12,
                  eccentricity=0.5, solidity=0.9)
    vm_ref = ViewMeasurement(image_name="a.jpg", tilt_deg=30.0,
                             error_band_pct=12.0, confidence="medium",
                             scale_source="reference", **common)
    vm_cam = ViewMeasurement(
        image_name="b.jpg", tilt_deg=None, error_band_pct=25.0,
        confidence="low", scale_source="camera_height",
        camera_scale={"horizon_method": "sky_anchor",
                      "mm_per_px_transverse": 2.3, "f_source": "config",
                      "certifiable": False}, **common)
    fused = fuse_pair(vm_ref, vm_cam)
    assert fused.available is True
    assert fused.confidence == "low"
    assert fused.scale_source == "mixed"
    assert fused.camera_scale is not None
    assert fused.camera_scale["from_view"] == "b.jpg"
    assert fused.cross_view["confidence_promoted"] is False
    assert fused.cross_view["per_view_scale_sources"] == [
        "reference", "camera_height"]
    d = fused.to_dict()
    assert d["scale_source"] == "mixed" and d["camera_scale"] is not None
