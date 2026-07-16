"""Тесты честной см-ОЦЕНКИ глубины (depth.estimate_depth_cm, 2026-07-14).

Физика: rel_norm ≈ dz/W_поперечная (см. depth.py), значит
dz ≈ rel_norm × W_см. Подстановкой rel_norm = drop/(gn·width_px) и
W_см = width_px·mm_per_px/10 ширина сокращается: оценка равна
drop/gn · mm_per_px/10 — перцентильная обрезка ширины на точку не влияет,
что и проверяет сквозной pinhole-тест (известная глубина восстанавливается).

Гейты честности: нет эталона → нет сантиметров; ниже шума → только верхняя
граница; вырожденная линейка → отказ. Оценка никогда не сертифицируется.
"""
import numpy as np
import pytest

pytest.importorskip("cv2")

from road_defect import config
from road_defect.depth import estimate_depth_cm, ring_plane_bucket


def _pinhole_scene(D0, dz, R=0.35, h=1.5, f=400.0, H=400, W=400,
                   noise=1e-4, seed=3):
    """Точная pinhole-геометрия (как в test_depth_bucket): дорога —
    disparity-плоскость y/(f·h), яма радиуса R на дистанции D0 — ракурсно
    сжатый эллипс с просадкой dz/(h·D0)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    depth = yy / (f * h)
    y0 = f * h / D0
    wx, wy = f * R / D0, f * R * h / D0 ** 2
    mask = ((xx - W / 2) / wx) ** 2 + ((yy - y0) / wy) ** 2 <= 1.0
    depth[mask] -= dz / (h * D0)
    depth += rng.normal(0.0, noise, size=depth.shape)
    return depth, mask


def _road_scene(H=400, W=400, pit_depth=0.0, noise=0.003, seed=7,
                cx=200, cy=260, r=40, grad=1.5):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    depth = 2.0 + grad * (yy / H)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    depth -= pit_depth * np.exp(-dist2 / (2 * (r * 0.6) ** 2))
    depth += rng.normal(0.0, noise, size=depth.shape)
    mask = dist2 <= r * r
    return depth, mask


def _measured(rel=0.3, width_px=100.0, k=1.0):
    """Синтетический результат ring_plane_bucket с ИЗМЕРЕННЫМ rel_norm."""
    return {"depth_bucket": "medium",
            "method": "depth_anything_v2_small_ring_plane",
            "_rel_norm": rel, "_width_px": width_px,
            "_px_scale_to_orig": k}


# --- сквозная физика ---------------------------------------------------------

def test_pinhole_known_depth_recovered_in_cm():
    # Яма глубиной 10 см (dz=0.10 м) на D0=3 м, f=400 px: истинный масштаб на
    # дистанции ямы mm/px = D0/f·1000 = 7.5. Оценка обязана вернуть ~10 см.
    out = ring_plane_bucket(*_pinhole_scene(D0=3.0, dz=0.10))
    assert out["depth_bucket"] is not None
    est = estimate_depth_cm(out, mm_per_px=7.5)
    assert est["available"] is True
    assert est["method"] == "rel_norm_x_transverse_width"
    assert est["point_cm"] == pytest.approx(10.0, rel=0.15)
    assert est["low_cm"] <= est["point_cm"] <= est["high_cm"]


def test_same_pit_same_cm_at_any_distance():
    # Инвариант дистанции: та же яма с 6 м при ПРАВИЛЬНОМ mm/px этой дистанции
    # даёт те же сантиметры (mm/px растёт ∝ D, rel_norm неизменен).
    near = estimate_depth_cm(ring_plane_bucket(*_pinhole_scene(D0=3.0, dz=0.10)),
                             mm_per_px=7.5)
    far = estimate_depth_cm(ring_plane_bucket(*_pinhole_scene(D0=6.0, dz=0.10)),
                            mm_per_px=15.0)
    assert near["point_cm"] == pytest.approx(far["point_cm"], rel=0.15)


def test_estimate_scales_linearly_with_mm_per_px():
    out = ring_plane_bucket(*_pinhole_scene(D0=3.0, dz=0.10))
    e1 = estimate_depth_cm(out, mm_per_px=7.5)
    e2 = estimate_depth_cm(out, mm_per_px=15.0)
    assert e2["point_cm"] == pytest.approx(2.0 * e1["point_cm"], rel=1e-6)


def test_grid_scale_factor_applied():
    # Карта глубины в сетке ~518 px, эталон — в пикселях исходного кадра:
    # _px_scale_to_orig обязан войти в ширину (и в точку) линейно.
    e1 = estimate_depth_cm(_measured(k=1.0), mm_per_px=2.0)
    e2 = estimate_depth_cm(_measured(k=3.0), mm_per_px=2.0)
    assert e2["point_cm"] == pytest.approx(3.0 * e1["point_cm"], rel=1e-6)


# --- гейты честности ---------------------------------------------------------

def test_no_scale_reference_no_cm():
    out = ring_plane_bucket(*_pinhole_scene(D0=3.0, dz=0.10))
    for bad in (None, 0.0, -1.0, float("nan")):
        est = estimate_depth_cm(out, mm_per_px=bad)
        assert est["available"] is False
        assert est["reason"] == "no_scale_reference"
        assert est["point_cm"] is None and est["upper_bound_cm"] is None


def test_below_noise_yields_upper_bound_only():
    # Плоская дорога без ямы: просадка ниже ворот шума → точки НЕТ, есть
    # честная верхняя граница «глубина не больше X см».
    out = ring_plane_bucket(*_road_scene(pit_depth=0.0))
    assert out["method"].endswith("below_local_noise")
    est = estimate_depth_cm(out, mm_per_px=5.0)
    assert est["available"] is True
    assert est["method"] == "below_noise_upper_bound"
    assert est["point_cm"] is None
    assert est["upper_bound_cm"] > 0
    # шум 0.003 на градиенте 1.5 — граница заведомо мельче ГОСТ-порога 5 см
    assert est["upper_bound_cm"] < config.GOST50597_MAX_DEPTH_CM
    assert est["vs_gost_5cm"] == "below"


def test_degenerate_ruler_refuses_without_numbers():
    out = ring_plane_bucket(*_road_scene(pit_depth=0.5, grad=0.0))
    assert out["method"] == "degenerate_local_ruler"
    est = estimate_depth_cm(out, mm_per_px=5.0)
    assert est["available"] is False
    assert est["reason"] == "degenerate_local_ruler"
    assert est["point_cm"] is None and est["upper_bound_cm"] is None


def test_upstream_refusals_pass_reason_through():
    for method in ("unavailable", "mask_below_depth_resolution",
                   "no_ring_support"):
        est = estimate_depth_cm({"method": method}, mm_per_px=5.0)
        assert est["available"] is False
        assert est["reason"] == method
    est = estimate_depth_cm({}, mm_per_px=5.0)
    assert est["available"] is False
    assert est["reason"] == "no_depth_signal"


def test_disabled_via_config():
    cfg = config.InferenceConfig(depth_cm_estimate_enabled=False)
    est = estimate_depth_cm(_measured(), mm_per_px=5.0, cfg=cfg)
    assert est["available"] is False
    assert est["reason"] == "disabled"


def test_estimate_never_certifiable_and_never_above_low():
    for est in (estimate_depth_cm(_measured(), mm_per_px=5.0),
                estimate_depth_cm(_measured(), mm_per_px=None),
                estimate_depth_cm({"method": "no_ring_support"}, 5.0)):
        assert est["certifiable"] is False
        assert est["confidence"] in (None, "low")
        assert "оценка" in est["note"].lower() or "НЕ измерение" in est["note"]


def test_config_rejects_negative_base_err():
    with pytest.raises(ValueError):
        config.InferenceConfig(depth_cm_est_base_err_pct=-1.0)


# --- полоса и сравнение с ГОСТ ----------------------------------------------

def test_band_widens_with_scale_error():
    tight = estimate_depth_cm(_measured(), mm_per_px=2.0,
                              scale_error_band_pct=12.0)
    wide = estimate_depth_cm(_measured(), mm_per_px=2.0,
                             scale_error_band_pct=50.0)
    assert tight["point_cm"] == wide["point_cm"]
    assert wide["high_cm"] > tight["high_cm"]
    assert wide["low_cm"] < tight["low_cm"]


def test_vs_gost_5cm_enum():
    # point=10, полоса (50+12)% → low≈6.2 > 5 → выше порога даже по низу полосы
    above = estimate_depth_cm(_measured(rel=0.5, width_px=100, k=2.0),
                              mm_per_px=1.0, scale_error_band_pct=12.0)
    assert above["point_cm"] == pytest.approx(10.0)
    assert above["vs_gost_5cm"] == "above"
    # point=1 → high≈1.6 < 5 → ниже порога даже по верху полосы
    below = estimate_depth_cm(_measured(rel=0.05, width_px=100, k=1.0),
                              mm_per_px=2.0, scale_error_band_pct=12.0)
    assert below["vs_gost_5cm"] == "below"
    # point=6 → полоса накрывает 5 см → честное «не определено»
    straddle = estimate_depth_cm(_measured(rel=0.3, width_px=100, k=1.0),
                                 mm_per_px=2.0, scale_error_band_pct=12.0)
    assert straddle["vs_gost_5cm"] == "uncertain"
    assert straddle["low_cm"] < config.GOST50597_MAX_DEPTH_CM < straddle["high_cm"]
