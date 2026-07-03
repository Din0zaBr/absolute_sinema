"""Тесты «кольцевой плоскостной линейки» (depth.ring_plane_bucket, 2026-07-03).

Главная регрессия — исходный баг ревью 2026-07-02: бакет НЕ должен зависеть от
композиции кадра (старая нормировка на полнокадровый np.ptp делала все дефекты
«shallow», как только в кадр попадало небо/дальний план).
"""
import numpy as np
import pytest

pytest.importorskip("cv2")

from road_defect.depth import (BUCKET_HI, BUCKET_LO, mask_to_depth_grid,
                               ring_plane_bucket)


def _road_scene(H=400, W=400, pit_depth=0.0, noise=0.003, seed=7,
                cx=200, cy=260, r=40, grad=1.5):
    """Наклонная «дорога» (перспективный градиент по y: ниже = ближе = больше)
    + гауссова вмятина глубиной pit_depth + шероховатость покрытия noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    depth = 2.0 + grad * (yy / H)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    depth -= pit_depth * np.exp(-dist2 / (2 * (r * 0.6) ** 2))  # яма = дальше
    depth += rng.normal(0.0, noise, size=depth.shape)
    mask = dist2 <= r * r
    return depth, mask


def test_affine_invariance_of_rel_norm():
    # Карта аффинно-инвариантна (d -> k*d + s, k>0): rel_norm обязан совпадать.
    depth, mask = _road_scene(pit_depth=0.2)
    base = ring_plane_bucket(depth, mask)
    scaled = ring_plane_bucket(0.3 * depth + 100.0, mask)
    stretched = ring_plane_bucket(3.0 * depth - 50.0, mask)
    assert base["depth_bucket"] is not None
    assert scaled["depth_bucket"] == base["depth_bucket"] == stretched["depth_bucket"]
    assert base["_rel_norm"] == pytest.approx(scaled["_rel_norm"], rel=1e-4)
    assert base["_rel_norm"] == pytest.approx(stretched["_rel_norm"], rel=1e-4)


def test_sky_in_frame_does_not_change_bucket():
    # РЕГРЕССИЯ исходного бага: приклеиваем сверху «небо» (далёкий план с
    # огромным перепадом карты) — старая формула рушила rel_norm в ~разы,
    # новая должна дать тот же бакет и почти тот же rel_norm.
    depth, mask = _road_scene(pit_depth=0.2)
    H, W = depth.shape
    sky = np.full((H, W), -30.0)
    sky[: H // 2] = -60.0                      # ещё и градиент в небе
    padded = np.vstack([sky, depth])           # composition меняется радикально
    padded_mask = np.vstack([np.zeros_like(mask), mask])

    base = ring_plane_bucket(depth, mask)
    pad = ring_plane_bucket(padded, padded_mask)
    assert base["depth_bucket"] is not None
    assert pad["depth_bucket"] == base["depth_bucket"]
    assert pad["_rel_norm"] == pytest.approx(base["_rel_norm"], rel=0.02)


def test_close_crop_does_not_change_bucket():
    # Обратная сторона того же бага: тесный кроп вокруг дефекта сжимал ptp и
    # толкал бакет в deep. Теперь кроп (с сохранением кольца) бакет не меняет.
    depth, mask = _road_scene(pit_depth=0.2, cx=200, cy=200, r=40)
    crop = np.s_[80:320, 80:320]               # кольцо (~+92 px) целиком внутри
    base = ring_plane_bucket(depth, mask)
    cropped = ring_plane_bucket(depth[crop], mask[crop])
    assert cropped["depth_bucket"] == base["depth_bucket"]
    assert cropped["_rel_norm"] == pytest.approx(base["_rel_norm"], rel=0.05)


def test_buckets_are_ordered_by_pit_depth():
    flat = ring_plane_bucket(*_road_scene(pit_depth=0.0))
    med = ring_plane_bucket(*_road_scene(pit_depth=0.09))
    deep = ring_plane_bucket(*_road_scene(pit_depth=0.45))
    assert flat["depth_bucket"] == "shallow"
    assert flat["_rel_norm"] == 0.0            # ниже ворот 3σ — не сигнал
    assert med["depth_bucket"] == "medium"
    assert BUCKET_LO < med["_rel_norm"] < BUCKET_HI
    assert deep["depth_bucket"] == "deep"
    assert deep["_rel_norm"] > med["_rel_norm"]


def test_bump_patch_is_shallow():
    # Выпуклость (заплатка/наплыв) — просадка отрицательна → честный shallow.
    depth, mask = _road_scene(pit_depth=-0.3)
    out = ring_plane_bucket(depth, mask)
    assert out["depth_bucket"] == "shallow"
    assert out["method"].endswith("below_local_noise")


def test_tiny_mask_is_refused():
    depth, _ = _road_scene()
    mask = np.zeros(depth.shape, bool)
    mask[200:207, 200:210] = True              # 70 px < 100
    out = ring_plane_bucket(depth, mask)
    assert out["depth_bucket"] is None
    assert out["method"] == "mask_below_depth_resolution"


def test_mask_covering_frame_refused_without_exception():
    depth, _ = _road_scene(H=200, W=200)
    out = ring_plane_bucket(depth, np.ones(depth.shape, bool))
    assert out["depth_bucket"] is None
    assert out["method"] == "no_ring_support"


def test_nadir_frame_refuses_honestly():
    # Ревью 2026-07-03: текстурная линейка max(…, 3σ) была вырождена — после
    # ворот drop ≥ 3σ ЛЮБАЯ просадка давала deep тождеством. Надир (градиента
    # плоскости нет) — теперь честный отказ: линейки нет.
    depth, mask = _road_scene(pit_depth=0.5, grad=0.0, noise=0.003)
    out = ring_plane_bucket(depth, mask)
    assert out["depth_bucket"] is None
    assert out["method"] == "degenerate_local_ruler"


def test_weak_gradient_refuses_instead_of_false_deep():
    # Тот же вырожденный режим при слабой перспективе: deep-порог линейки ниже
    # пола шума — сантиметровая вмятина получала бы max-приоритет. Отказ.
    depth, mask = _road_scene(pit_depth=0.05, grad=0.02, noise=0.003)
    out = ring_plane_bucket(depth, mask)
    assert out["depth_bucket"] is None
    assert out["method"] == "degenerate_local_ruler"


def _pinhole_scene(D0, dz, R=0.35, h=1.5, f=400.0, H=400, W=400,
                   noise=1e-4, seed=3):
    """Точная pinhole-геометрия (репро скептиков ревью 2026-07-03): дорога —
    disparity-плоскость y/(f·h), яма радиуса R на дистанции D0 — ракурсно
    СЖАТЫЙ эллипс с просадкой dz/(h·D0) (удлинение луча)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    depth = yy / (f * h)                        # y=0 — горизонт
    y0 = f * h / D0
    wx, wy = f * R / D0, f * R * h / D0 ** 2    # поперёк не сжата, вдоль — да
    mask = ((xx - W / 2) / wx) ** 2 + ((yy - y0) / wy) ** 2 <= 1.0
    depth[mask] -= dz / (h * D0)
    depth += rng.normal(0.0, noise, size=depth.shape)
    return depth, mask


def test_same_pit_same_bucket_at_any_distance():
    # ГЛАВНАЯ регрессия ревью 2026-07-03: кругово-эквивалентный диаметр
    # ракурсно сжатого футпринта давал rel_norm ∝ √D — бакет одной и той же
    # ямы менялся от дистанции съёмки. Поперечная линейка инвариантна:
    # rel_norm ≈ dz/(2R) на любой дистанции.
    near = ring_plane_bucket(*_pinhole_scene(D0=3.0, dz=0.10))
    far = ring_plane_bucket(*_pinhole_scene(D0=6.0, dz=0.10))
    assert near["depth_bucket"] == far["depth_bucket"] == "medium"
    assert near["_rel_norm"] == pytest.approx(far["_rel_norm"], rel=0.10)
    assert near["_rel_norm"] == pytest.approx(0.10 / (2 * 0.35), rel=0.15)


def test_mask_to_depth_grid_downscales_blob_and_drops_subpixel_crack():
    # Измерение идёт в родной сетке карты (~518 px): крупная яма выживает при
    # уменьшении маски, субпиксельная трещина честно исчезает (нет сигнала).
    full = np.zeros((2000, 4000), bool)
    full[900:1300, 1800:2300] = True                 # яма 400×500 px
    small = mask_to_depth_grid(full, (259, 518))
    assert small.shape == (259, 518)
    assert small.sum() > 100                          # выжила и измерима

    crack = np.zeros((2000, 4000), bool)
    crack[1000:1002, 500:3500] = True                 # 2 px — субпиксель в 518
    assert not mask_to_depth_grid(crack, (259, 518)).any()


def test_thin_crack_mask_does_not_crash():
    # Тонкая маска (трещина): эрозия опустошает ядро — ветка перцентиля.
    depth, _ = _road_scene()
    mask = np.zeros(depth.shape, bool)
    mask[200:203, 60:340] = True               # 3 px × 280 px
    out = ring_plane_bucket(depth, mask)
    assert out["depth_bucket"] in ("shallow", "medium", "deep")
