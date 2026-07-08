"""Тесты two-view глубины (photogrammetry.py, цикл 15).

Основная валидация живёт в синтетическом стенде scripts/bench_two_view_depth.py
(19 кейсов, включая честные отказы) — здесь быстрые инварианты геометрии,
решателя и маленький e2e, чтобы регрессии ловились обычным pytest.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from road_defect import config
from road_defect.photogrammetry import (RigView, estimate_two_view_depth,
                                        ground_to_image_h, image_to_ground,
                                        solve_depth_from_parallax)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "synth_road3d", ROOT / "scripts" / "synth_road3d.py")
synth = importlib.util.module_from_spec(_spec)
# регистрация ДО exec: dataclass-декоратору нужен sys.modules[__module__]
sys.modules["synth_road3d"] = synth
_spec.loader.exec_module(synth)

VIEW = RigView(f_px=1000.0, cx=640.0, cy=360.0, cam_height_m=1.3,
               pitch_deg=math.degrees(math.atan2(1.3, 2.5)))
# быстрый конфиг для e2e: грубее BEV, ниже потолок скана
FAST_CFG = config.TwoViewDepthConfig(bev_mm_px=6.0, dz_max_m=0.12)


def test_ground_homography_invariants():
    h = ground_to_image_h(VIEW)
    # точка на оси в плане (Y=0) → u = cx
    p = h @ np.array([3.0, 0.0, 1.0])
    assert p[0] / p[2] == pytest.approx(VIEW.cx)
    # дистанция, куда смотрит оптическая ось (X = h/tg θ) → центр кадра
    x_axis = VIEW.cam_height_m / math.tan(math.radians(VIEW.pitch_deg))
    p = h @ np.array([x_axis, 0.0, 1.0])
    assert p[0] / p[2] == pytest.approx(VIEW.cx)
    assert p[1] / p[2] == pytest.approx(VIEW.cy)
    # ближе к камере → ниже в кадре; левее (Y>0) → левее в кадре (u меньше)
    p_near = h @ np.array([x_axis * 0.6, 0.0, 1.0])
    assert p_near[1] / p_near[2] > VIEW.cy
    p_left = h @ np.array([x_axis, 0.4, 1.0])
    assert p_left[0] / p_left[2] < VIEW.cx


def test_image_ground_roundtrip():
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(1.5, 6.0, 50),
                           rng.uniform(-1.5, 1.5, 50)])
    h = ground_to_image_h(VIEW)
    uvw = np.column_stack([pts, np.ones(len(pts))]) @ h.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    np.testing.assert_allclose(image_to_ground(VIEW, uv), pts, atol=1e-9)


def test_parallax_identity_against_full_projection():
    # Ядро физики модуля: q_b − q_f = dz·(C_b − C_f)/(h+dz). Пиксели точек дна
    # берём из ПОЛНОЙ 3D-проекции синтетической камеры, обратно — плоской
    # гомографией модуля. Сверка в мировой системе.
    h_cam, dz, d1, d2 = 1.30, 0.07, 3.0, 4.5
    pitch_f = math.degrees(math.atan2(h_cam, d1))
    pitch_b = math.degrees(math.atan2(h_cam, d2))
    cam_f = synth.SynthCamera(pos=(-d1, 0.0, h_cam), yaw_deg=0.0,
                              pitch_deg=pitch_f, f_px=1400.0)
    cam_b = synth.SynthCamera(pos=(d2, 0.0, h_cam), yaw_deg=180.0,
                              pitch_deg=pitch_b, f_px=1400.0)
    view_f = RigView(1400.0, cam_f.cx, cam_f.cy, h_cam, pitch_f)
    view_b = RigView(1400.0, cam_b.cx, cam_b.cy, h_cam, pitch_b)
    pts = np.array([[0.05, -0.10, -dz], [0.0, 0.0, -dz], [-0.08, 0.12, -dz]])

    q_f = image_to_ground(view_f, cam_f.project(pts))
    q_b = image_to_ground(view_b, cam_b.project(pts))
    # свои системы видов → мир: перед (yaw 0): x=X−d1, y=Y;
    # зад (yaw 180): x=d2−X, y=−Y
    q_fw = np.column_stack([q_f[:, 0] - d1, q_f[:, 1]])
    q_bw = np.column_stack([d2 - q_b[:, 0], -q_b[:, 1]])
    expected = np.tile([dz * (d1 + d2) / (h_cam + dz), 0.0], (len(pts), 1))
    np.testing.assert_allclose(q_bw - q_fw, expected, atol=1e-9)


def test_solve_depth_equal_heights_closed_form():
    dz, b, h = 0.06, 8.0, 1.3
    delta = dz * b / (h + dz)
    out = solve_depth_from_parallax(np.array([delta]), np.array([3.0]),
                                    b, h, h, 0.2)
    assert out[0] == pytest.approx(dz, abs=1e-9)


def test_solve_depth_unequal_heights_bisection():
    h_f, h_b, b, x, dz = 1.30, 1.05, 8.0, 3.2, 0.07
    delta = dz * (x / (h_f + dz) - (x - b) / (h_b + dz))
    out = solve_depth_from_parallax(np.array([delta]), np.array([x]),
                                    b, h_f, h_b, 0.2)
    assert out[0] == pytest.approx(dz, abs=1e-4)


@pytest.fixture(scope="module")
def rendered_pit():
    return synth.build_two_view_case(dz=0.05, d1=2.5, d2=2.5, seed=3,
                                     f_px=1000.0, width=1280, height=720)


def test_e2e_recovers_known_depth(rendered_pit):
    c = rendered_pit
    res = estimate_two_view_depth(
        c["front"]["image"], c["front"]["mask"], c["view_front"],
        c["back"]["image"], c["back"]["mask"], c["view_back"], FAST_CFG)
    assert res["method"] == "two_view_plane_parallax"
    assert res["depth_certifiable"] is False          # прототип: жёстко
    assert res["depth_cm_p90"] == pytest.approx(5.0, abs=0.8)
    assert res["baseline_m"] == pytest.approx(5.0, rel=0.02)


def test_e2e_wrong_height_calibration_refuses(rendered_pit):
    # Рассинхрон высоты (метрики видов) обязан давать отказ, а не ложные см.
    c = rendered_pit
    bad_back = RigView(f_px=1000.0, cx=640.0, cy=360.0, cam_height_m=1.05,
                       pitch_deg=c["view_back"].pitch_deg)
    res = estimate_two_view_depth(
        c["front"]["image"], c["front"]["mask"], c["view_front"],
        c["back"]["image"], c["back"]["mask"], bad_back, FAST_CFG)
    assert res["method"] == "registration_scale_mismatch"
    assert res["depth_cm_p90"] is None


def test_e2e_empty_mask_refuses(rendered_pit):
    c = rendered_pit
    empty = np.zeros_like(c["front"]["mask"])
    res = estimate_two_view_depth(
        c["front"]["image"], empty, c["view_front"],
        c["back"]["image"], empty, c["view_back"], FAST_CFG)
    assert res["method"] == "empty_mask"
    assert res["depth_cm_p90"] is None


def test_e2e_flat_patch_yields_zero_not_fabricated(rendered_pit):
    # Заплатка (глубины НЕТ): либо честный отказ, либо ≈0 — не «глубина по цвету».
    c = synth.build_two_view_case(dz=0.0, d1=2.5, d2=2.5, seed=4, kind="patch",
                                  f_px=1000.0, width=1280, height=720)
    res = estimate_two_view_depth(
        c["front"]["image"], c["front"]["mask"], c["view_front"],
        c["back"]["image"], c["back"]["mask"], c["view_back"], FAST_CFG)
    if res["method"] == "two_view_plane_parallax":
        assert res["depth_cm_p90"] <= 1.0
    else:
        assert res["depth_cm_p90"] is None


def test_uint8_255_mask_is_normalized(rendered_pit):
    # Ревью 2026-07-08: маска uint8 0/255 переполнялась при *255 и роняла
    # оценщик крэшем вместо результата. Теперь вход нормализуется.
    c = rendered_pit
    m_f = (c["front"]["mask"].astype(np.uint8)) * 255
    m_b = (c["back"]["mask"].astype(np.uint8)) * 255
    res = estimate_two_view_depth(
        c["front"]["image"], m_f, c["view_front"],
        c["back"]["image"], m_b, c["view_back"], FAST_CFG)
    assert res["method"] == "two_view_plane_parallax"
    assert res["depth_cm_p90"] == pytest.approx(5.0, abs=0.8)


def test_mask_near_horizon_refuses(rendered_pit):
    # Ревью 2026-07-08: маска у горизонта взрывала обратную проекцию
    # (BEV-сетка на десятки ГБ). Теперь кап рабочей зоны — честный отказ.
    # Нужен пологий риг (как реальный регистратор): при пологом тангаже
    # горизонт в кадре и пара пикселей под ним — сотни метров дистанции.
    c = rendered_pit
    flat = RigView(f_px=1000.0, cx=640.0, cy=360.0, cam_height_m=1.3,
                   pitch_deg=8.0)
    horizon_row = int(flat.cy - flat.f_px * math.tan(math.radians(8.0)))
    mask = np.zeros_like(c["front"]["mask"])
    mask[horizon_row + 2: horizon_row + 6, 600:680] = True
    res = estimate_two_view_depth(
        c["front"]["image"], mask, flat,
        c["back"]["image"], c["back"]["mask"], c["view_back"], FAST_CFG)
    assert res["method"] == "mask_beyond_working_range"
    assert res["depth_cm_p90"] is None


def test_textureless_frames_refuse():
    # Гладкие кадры: фичам не за что зацепиться — отказ регистрации.
    img = np.full((720, 1280), 128, np.uint8)
    mask = np.zeros((720, 1280), bool)
    mask[520:600, 560:720] = True
    res = estimate_two_view_depth(img, mask, VIEW, img, mask, VIEW,
                                  FAST_CFG)
    assert res["method"].startswith("registration_failed")
    assert res["depth_cm_p90"] is None


def test_two_view_config_validation():
    with pytest.raises(ValueError):
        config.TwoViewDepthConfig(patch_px=20)          # чётный
    with pytest.raises(ValueError):
        config.TwoViewDepthConfig(reg_model="homography")
    with pytest.raises(ValueError):
        config.TwoViewDepthConfig(bev_mm_px=0.0)
    with pytest.raises(ValueError):
        config.TwoViewDepthConfig(dz_max_m=-0.1)
    with pytest.raises(ValueError):
        config.TwoViewDepthConfig(core_inradius_frac=1.5)
    with pytest.raises(ValueError):
        config.TwoViewDepthConfig(saturated_max=-0.2)
