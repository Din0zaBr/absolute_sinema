"""Синтетическая круговая проверка plane_scale (масштаб от высоты камеры).

Чистая геометрия без моделей: карта диспаритета строится аналитически
(d = k/Z + s) из ИЗВЕСТНОЙ нормали полотна — оценщик обязан восстановить
тангаж/крен/mm_per_px/футпринт; кадр без неба обязан давать честный отказ,
а при заданном приоре тангажа — оценку с меткой prior_pitch. Порт selftest
эксперимента scripts/height_scale_experiment.py (2026-07-16).
"""
import numpy as np
import pytest

from road_defect import config
from road_defect.plane_scale import (defect_ground_metrics, f_px_from_exif,
                                     project_to_ground, resolve_frame_geometry)

W0, H0 = 4096, 1844
F_PX = 3025.0
H_M = 1.70


def _cfg(**over) -> config.CameraHeightScaleConfig:
    return config.CameraHeightScaleConfig(height_m=H_M, f_px=F_PX, **over)


def _normal(theta_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    th, r = np.radians(theta_deg), np.radians(roll_deg)
    return np.array([np.sin(r) * np.cos(th), -np.cos(r) * np.cos(th),
                     -np.sin(th)])


def _synthetic_depth(n: np.ndarray, q_floor: float = 0.0,
                     gw: int = 1024) -> np.ndarray:
    """Аналитическая карта d = 7·max(1/Z, q_floor) + 0.3 (небо -> 1/Z = 0)."""
    k = W0 / gw
    gh = int(round(H0 / k))
    xs, ys = np.meshgrid(np.arange(gw), np.arange(gh))
    u, v = (xs + 0.5) * k, (ys + 0.5) * k
    p = np.stack([(u - W0 / 2.0) / F_PX, (v - H0 / 2.0) / F_PX,
                  np.ones_like(u)], axis=-1)
    q = np.maximum(-(p @ n) / H_M, q_floor)
    return (7.0 * q + 0.3).astype(np.float32)


def _img() -> np.ndarray:
    return np.zeros((H0, W0, 3), dtype=np.uint8)


def _quad_mask(n: np.ndarray, dist_m: float, half_w: float = 0.20,
               half_l: float = 0.15) -> np.ndarray:
    """Маска прямоугольника 2·half_w × 2·half_l (м) на полотне в dist_m
    от камеры (проекция углов -> fillPoly)."""
    import cv2

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


def test_sky_anchor_roundtrip_theta_roll_and_scale():
    n_true = _normal(10.0, 2.0)
    geom, err = resolve_frame_geometry(_img(), _synthetic_depth(n_true), None,
                                       [], _cfg())
    assert geom is not None, err
    assert geom.horizon_method == "sky_anchor"
    assert geom.f_source == "config"
    assert abs(geom.theta_deg - 10.0) < 0.4
    assert abs(geom.roll_deg - 2.0) < 0.5

    # mm/px поперёк в контрольном пикселе против аналитики (Z вдоль оси)
    test_uv = np.array([[2048.0, 1300.0]])
    p1 = np.array([(2048.0 - W0 / 2) / F_PX, (1300.0 - H0 / 2) / F_PX, 1.0])
    mm_true = 1000.0 * (-H_M / float(p1 @ n_true)) / F_PX
    t_hat = np.array([-geom.ghat_img[1], geom.ghat_img[0]])
    Ps, _, ok = project_to_ground(np.vstack([test_uv[0], test_uv[0] + t_hat]),
                                  geom.n, W0, H0, geom.f_px, H_M)
    assert ok.all()
    mm_est = float(np.linalg.norm(Ps[1] - Ps[0])) * 1000.0
    assert abs(mm_est / mm_true - 1.0) < 0.01


def test_footprint_recovers_known_rectangle():
    n_true = _normal(10.0, 2.0)
    cfg = _cfg()
    geom, err = resolve_frame_geometry(_img(), _synthetic_depth(n_true), None,
                                       [], cfg)
    assert geom is not None, err
    foot, err = defect_ground_metrics(_quad_mask(n_true, 4.0), geom, cfg)
    assert foot is not None, err
    assert abs(foot["feret_max_cm"] - 40.0) < 1.5, foot
    assert abs(foot["feret_min_cm"] - 30.0) < 1.5, foot
    # прибиты и производные (ревью 2026-07-16): area/eqd идут в ГОСТ-вердикт
    assert abs(foot["area_cm2"] - 1200.0) < 30.0, foot
    assert abs(foot["equivalent_diameter_cm"] - 39.1) < 1.5, foot
    assert abs(foot["width_transverse_cm"] - 40.0) < 1.5, foot
    assert abs(foot["length_longitudinal_cm"] - 30.0) < 1.5, foot
    assert foot["mm_per_px_transverse"] > 0
    # полоса = f + высота + НЕтривиальная чувствительность горизонта
    # (раньше assert был тривиально истинен при потере n_lo/n_hi)
    assert foot["pitch_band_pct"] > 0, foot
    assert foot["scale_band_pct"] == pytest.approx(
        cfg.f_band_pct + cfg.height_band_pct + foot["pitch_band_pct"], abs=0.11)
    assert 3.0 <= foot["dist_ground_m"] <= 5.0


def test_no_sky_refuses_without_prior_and_uses_prior_when_set():
    # истина с креном 3°: приор берёт θ из конфига, а крен/уклон ОБЯЗАН
    # измерить по градиенту плоскости (ревью 2026-07-16)
    n_true = _normal(25.0, 3.0)                # горизонт выше кадра — неба нет
    d = _synthetic_depth(n_true, q_floor=1e-6)
    geom, err = resolve_frame_geometry(_img(), d, None, [], _cfg())
    assert geom is None and err == "no_sky_anchor"

    cfg = _cfg(pitch_prior_deg=25.0)
    geom, err = resolve_frame_geometry(_img(), d, None, [], cfg)
    assert geom is not None, err
    assert geom.horizon_method == "prior_pitch"
    assert geom.theta_deg == pytest.approx(25.0)
    assert abs(abs(geom.roll_deg) - 3.0) < 0.5          # крен — измеренный
    assert geom.diag.get("roll_from_gradient") is True
    foot, err = defect_ground_metrics(_quad_mask(n_true, 4.0), geom, cfg)
    assert foot is not None, err
    # приор совпал с истиной -> размеры восстановлены
    assert abs(foot["feret_max_cm"] - 40.0) < 2.0
    assert abs(foot["feret_min_cm"] - 30.0) < 2.0

    # кадр с креном за гейтом обязан отказывать и в prior-режиме
    d_roll = _synthetic_depth(_normal(25.0, 10.0), q_floor=1e-6)
    geom, err = resolve_frame_geometry(_img(), d_roll, None, [], cfg)
    assert geom is None and err == "roll_too_large"


def test_frame_gates_roll_and_pitch():
    geom, err = resolve_frame_geometry(
        _img(), _synthetic_depth(_normal(10.0, 10.0)), None, [], _cfg())
    assert geom is None and err == "roll_too_large"
    geom, err = resolve_frame_geometry(
        _img(), _synthetic_depth(_normal(3.0)), None, [], _cfg())
    assert geom is None and err == "pitch_out_of_range"


def test_defect_gates_far_and_near_horizon():
    n_true = _normal(10.0)
    cfg = _cfg()
    geom, err = resolve_frame_geometry(_img(), _synthetic_depth(n_true), None,
                                       [], cfg)
    assert geom is not None, err
    foot, err = defect_ground_metrics(_quad_mask(n_true, 13.0, 0.5, 0.5),
                                      geom, cfg)
    assert foot is None and err == "pit_too_far"
    foot, err = defect_ground_metrics(_quad_mask(n_true, 60.0, 1.5, 1.5),
                                      geom, cfg)
    assert foot is None and err == "pit_too_close_to_horizon"
    foot, err = defect_ground_metrics(np.zeros((H0, W0), dtype=bool), geom, cfg)
    assert foot is None and err == "empty_mask"


def test_disabled_and_no_focal_refusals():
    n_true = _normal(10.0)
    d = _synthetic_depth(n_true)
    geom, err = resolve_frame_geometry(
        _img(), d, None, [], config.CameraHeightScaleConfig())   # height=None
    assert geom is None and err == "disabled"
    geom, err = resolve_frame_geometry(
        _img(), d, None, [], config.CameraHeightScaleConfig(height_m=H_M))
    assert geom is None and err == "no_focal:no_path"


def test_f_px_from_exif(tmp_path):
    from PIL import Image

    p = tmp_path / "t.jpg"
    ex = Image.Exif()
    ex[41989] = 26                                # FocalLengthIn35mmFilm
    Image.new("RGB", (40, 30)).save(p, exif=ex)
    f, src = f_px_from_exif(p, (4096, 1844))
    assert src == "exif_fl35"
    assert f == pytest.approx(26 * 4096 / 36.0)

    p2 = tmp_path / "noexif.jpg"
    Image.new("RGB", (40, 30)).save(p2)
    f, src = f_px_from_exif(p2, (4096, 1844))
    assert f is None and src == "no_focal:no_fl35_in_exif"


def test_config_validation():
    with pytest.raises(ValueError):
        config.CameraHeightScaleConfig(height_m=0.05)
    with pytest.raises(ValueError):
        config.CameraHeightScaleConfig(height_m=1.7, pitch_prior_deg=80.0)
    with pytest.raises(ValueError):
        config.CameraHeightScaleConfig(height_m=1.7, f_px=-3.0)
