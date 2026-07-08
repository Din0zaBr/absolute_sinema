"""Синтетическая 3D-сцена «дорога + яма»: точный pinhole-рендер лучевым маршем.

Для стенда two-view глубины (scripts/bench_two_view_depth.py) и юнит-тестов
(tests/test_photogrammetry.py). Мир: плоскость дороги Z=0, яма — гладкая
эллиптическая чаша ИЗВЕСТНОЙ глубины с центром в начале координат.

Что честно смоделировано:
  * обе камеры видят ОДНУ мировую текстуру (альбедо привязано к координатам
    земли) — база для реалистичного сопоставления между видами;
  * затенение ламбертово от фиксированного «солнца» — НЕ зависит от точки
    съёмки, поэтому согласовано между видами (как матовый асфальт);
  * окклюзия ближней кромкой — первое пересечение луча с поверхностью
    (марш + бисекция), никаких аппроксимаций «маленькой глубины»;
  * независимый сенсорный шум и рассинхрон gain/bias между видами.
Что НЕ смоделировано (границы стенда, см. docs/VEHICLE_CAPTURE.md §3):
вода/зеркальные блики, смаз движения, реальная сегментация масок.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@dataclass(frozen=True)
class SynthCamera:
    """Pinhole-камера в мире. yaw 0 = смотрит вдоль +X, 180 = вдоль −X."""
    pos: tuple
    yaw_deg: float
    pitch_deg: float          # вниз от горизонта
    f_px: float = 1400.0
    width: int = 1920
    height: int = 1080

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    def rotation(self) -> np.ndarray:
        """Строки — оси камеры (x_cam→вправо кадра, y_cam→вниз, z_cam→вперёд)
        в мировых координатах; p_cam = R·(p_world − C)."""
        psi = math.radians(self.yaw_deg)
        th = math.radians(self.pitch_deg)
        fwd = np.array([math.cos(psi) * math.cos(th),
                        math.sin(psi) * math.cos(th),
                        -math.sin(th)])
        up = np.array([0.0, 0.0, 1.0])
        xc = np.cross(fwd, up)
        xc /= np.linalg.norm(xc)
        yc = np.cross(fwd, xc)          # правая тройка: y = z × x
        return np.stack([xc, yc, fwd])

    def project(self, pts_world: np.ndarray) -> np.ndarray:
        """(N, 3) мир → (N, 2) пиксели. Для тестов тождества параллакса."""
        r = self.rotation()
        pc = (np.asarray(pts_world, float) - np.asarray(self.pos, float)) @ r.T
        return np.column_stack([
            self.f_px * pc[:, 0] / pc[:, 2] + self.cx,
            self.f_px * pc[:, 1] / pc[:, 2] + self.cy,
        ])


@dataclass(frozen=True)
class PitSpec:
    """Эллиптическая чаша: плоское дно (flat_core доля радиуса) + cos-скат.

    kind='patch' — плоская тёмная заплатка той же формы (глубина 0):
    null-кейс честности (система не должна фабриковать глубину по цвету).
    """
    semi_x: float = 0.30
    semi_y: float = 0.22
    depth_m: float = 0.05
    flat_core: float = 0.45
    kind: str = "pothole"

    def _t(self, x, y):
        return np.sqrt((np.asarray(x, float) / self.semi_x) ** 2
                       + (np.asarray(y, float) / self.semi_y) ** 2)

    def depth_at(self, x, y) -> np.ndarray:
        t = self._t(x, y)
        prof = np.zeros_like(t)
        if self.kind == "patch" or self.depth_m <= 0.0:
            return prof
        core = t <= self.flat_core
        ramp = (t > self.flat_core) & (t < 1.0)
        prof[core] = 1.0
        prof[ramp] = 0.5 * (1.0 + np.cos(
            math.pi * (t[ramp] - self.flat_core) / (1.0 - self.flat_core)))
        return self.depth_m * prof

    def interior(self, x, y) -> np.ndarray:
        """Вес «внутри дефекта» 0..1 (общий для ямы и заплатки)."""
        return np.clip((1.0 - self._t(x, y)) / 0.06, 0.0, 1.0)

    def inside(self, x, y, margin: float = 1.0) -> np.ndarray:
        return self._t(x, y) <= margin


class RoadScene:
    """Мировая текстура асфальта + аналитическая яма. Альбедо/глубина/нормали —
    функции координат земли, одинаковые для всех камер."""

    def __init__(self, pit: PitSpec, seed: int = 0,
                 extent_m: float = 1.7, tex_mm: float = 1.5):
        import cv2

        self.pit = pit
        self.extent = float(extent_m)
        self.tex_m = tex_mm / 1000.0
        self.n = int(round(2 * extent_m / self.tex_m))
        rng = np.random.default_rng(seed)
        base = rng.standard_normal((self.n, self.n)).astype(np.float32)

        def layer(sigma, w):
            g = cv2.GaussianBlur(base, (0, 0), sigma)
            return w * g / (float(g.std()) + 1e-9)

        tex = layer(1.2, 0.55) + layer(5.0, 0.30) + layer(18.0, 0.15)
        tex = 0.52 + 0.075 * tex / float(tex.std())
        spots = (rng.random((self.n, self.n)) < 0.0015).astype(np.float32)
        spots = cv2.GaussianBlur(spots, (0, 0), 1.2)
        tex -= 2.5 * spots
        self.albedo_map = np.clip(tex, 0.20, 0.85)
        rough = cv2.GaussianBlur(
            rng.standard_normal((self.n, self.n)).astype(np.float32), (0, 0), 2.0)
        self.rough_map = 0.06 * rough / float(rough.std())

    def _lookup(self, grid: np.ndarray, x, y) -> np.ndarray:
        """Билинейная выборка мировой карты; за краем — тайлинг (далёкий фон)."""
        fx = (np.asarray(x, float) + self.extent) / self.tex_m
        fy = (np.asarray(y, float) + self.extent) / self.tex_m
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        ax = (fx - x0).astype(np.float32)
        ay = (fy - y0).astype(np.float32)
        x0m, x1m = x0 % self.n, (x0 + 1) % self.n
        y0m, y1m = y0 % self.n, (y0 + 1) % self.n
        g = grid
        return ((g[y0m, x0m] * (1 - ax) + g[y0m, x1m] * ax) * (1 - ay)
                + (g[y1m, x0m] * (1 - ax) + g[y1m, x1m] * ax) * ay)

    def albedo_at(self, x, y) -> np.ndarray:
        a = self._lookup(self.albedo_map, x, y)
        w = self.pit.interior(x, y)
        d = self.pit.depth_at(x, y)
        a = a * (1.0 - 0.30 * w) * (1.0 - 2.0 * d) \
            + self._lookup(self.rough_map, x, y) * w
        return np.clip(a, 0.05, 1.0)

    def shade_at(self, x, y) -> np.ndarray:
        eps = 0.003
        dddx = (self.pit.depth_at(x + eps, y)
                - self.pit.depth_at(x - eps, y)) / (2 * eps)
        dddy = (self.pit.depth_at(x, y + eps)
                - self.pit.depth_at(x, y - eps)) / (2 * eps)
        # поверхность z = −depth → нормаль ∝ (∂depth/∂x, ∂depth/∂y, 1)
        norm = np.sqrt(dddx ** 2 + dddy ** 2 + 1.0)
        light = np.array([0.35, 0.25, 0.90])
        light = light / np.linalg.norm(light)
        lam = np.maximum(
            (dddx * light[0] + dddy * light[1] + light[2]) / norm, 0.0)
        return 0.55 + 0.45 * lam

    def render(self, cam: SynthCamera, gain: float = 1.0, bias: float = 0.0,
               noise: float = 2.0, seed: int = 1) -> dict:
        """Рендер вида. Возвращает {'image' uint8, 'mask' видимая область
        дефекта (вкл. стенки), 'hit_depth' глубина поверхности в точке
        пересечения (диагностика)}."""
        rng = np.random.default_rng(seed)
        uu, vv = np.meshgrid(np.arange(cam.width, dtype=np.float64),
                             np.arange(cam.height, dtype=np.float64))
        r = cam.rotation()
        d_cam = np.stack([(uu - cam.cx) / cam.f_px,
                          (vv - cam.cy) / cam.f_px,
                          np.ones_like(uu)], axis=-1)
        d_w = d_cam @ r                       # каждый луч: R.T @ d_cam
        pos = np.asarray(cam.pos, float)
        dzr = d_w[..., 2]
        ground = dzr < -1e-6
        safe_dz = np.where(ground, dzr, -1.0)
        t_pl = np.where(ground, -pos[2] / safe_dz, np.nan)
        gx = pos[0] + t_pl * d_w[..., 0]
        gy = pos[1] + t_pl * d_w[..., 1]

        img = np.full((cam.height, cam.width), 175.0, np.float32)
        hit_depth = np.zeros((cam.height, cam.width), np.float32)
        mask = np.zeros((cam.height, cam.width), bool)

        plain = ground & ~self.pit.inside(gx, gy, margin=1.03)
        img[plain] = 255.0 * self.albedo_at(gx[plain], gy[plain]) \
            * self.shade_at(gx[plain], gy[plain])

        sub = ground & self.pit.inside(gx, gy, margin=1.03)
        if sub.any():
            d = d_w[sub]                                  # (M, 3)
            t0 = t_pl[sub] * (1.0 - 1e-4)
            t1 = (-pos[2] - self.pit.depth_m - 1e-3) / d[:, 2]

            def phi(t):
                x = pos[0] + t * d[:, 0]
                y = pos[1] + t * d[:, 1]
                z = pos[2] + t * d[:, 2]
                return z + self.pit.depth_at(x, y)

            # первый заход луча под поверхность: марш + бисекция
            steps = 192
            t_lo = t0.copy()
            t_hi = t1.copy()
            found = np.zeros(len(d), bool)
            t_prev = t0.copy()
            for k in range(1, steps + 1):
                tk = t0 + (t1 - t0) * (k / steps)
                below = (phi(tk) < 0.0) & ~found
                t_lo[below] = t_prev[below]
                t_hi[below] = tk[below]
                found |= below
                t_prev = np.where(found, t_prev, tk)
            for _ in range(14):
                tm = 0.5 * (t_lo + t_hi)
                below = phi(tm) < 0.0
                t_hi = np.where(below, tm, t_hi)
                t_lo = np.where(below, t_lo, tm)
            t_hit = 0.5 * (t_lo + t_hi)
            hx = pos[0] + t_hit * d[:, 0]
            hy = pos[1] + t_hit * d[:, 1]
            img[sub] = 255.0 * self.albedo_at(hx, hy) * self.shade_at(hx, hy)
            hd = self.pit.depth_at(hx, hy)
            hit_depth[sub] = hd
            # видимая область дефекта: попадание внутрь опоры (яма — вкл.
            # стенки; заплатка — по весу интерьера)
            mask[sub] = self.pit.interior(hx, hy) > 0.02

        img = img * gain + bias + rng.normal(0.0, noise, img.shape)
        return {"image": np.clip(np.round(img), 0, 255).astype(np.uint8),
                "mask": mask, "hit_depth": hit_depth}


def build_two_view_case(dz: float, d1: float, d2: float,
                        h_f: float = 1.30, h_b: float = 1.30,
                        pitch_err_f: float = 0.0, pitch_err_b: float = 0.0,
                        h_nom_f: float | None = None,
                        h_nom_b: float | None = None,
                        kind: str = "pothole", seed: int = 0,
                        f_px: float = 1400.0, width: int = 1920,
                        height: int = 1080,
                        pit_semi: tuple = (0.30, 0.22),
                        flat_core: float = 0.45) -> dict:
    """Сцена + два вида «перед/за» + номинальные RigView для оценщика.

    Рендер использует ИСТИННЫЙ тангаж (номинал + ошибка подвески) и истинные
    высоты; оценщику отдаются НОМИНАЛЬНЫЕ (h_nom_* по умолчанию = истинным).
    Задняя камера смотрит назад с лёгким рысканием и боковым смещением —
    регистрация обязана снять это сама (в реальности рыскание неизвестно).
    """
    from road_defect.photogrammetry import RigView

    pit = PitSpec(semi_x=pit_semi[0], semi_y=pit_semi[1],
                  depth_m=dz, kind=kind, flat_core=flat_core)
    scene = RoadScene(pit, seed=seed)
    pitch_f = math.degrees(math.atan2(h_f, d1))
    pitch_b = math.degrees(math.atan2(h_b, d2))
    cam_f = SynthCamera(pos=(-d1, 0.05, h_f), yaw_deg=0.0,
                        pitch_deg=pitch_f + pitch_err_f,
                        f_px=f_px, width=width, height=height)
    cam_b = SynthCamera(pos=(d2, -0.08, h_b), yaw_deg=183.0,
                        pitch_deg=pitch_b + pitch_err_b,
                        f_px=f_px, width=width, height=height)
    r_f = scene.render(cam_f, gain=1.00, bias=0.0, noise=2.0, seed=seed * 2 + 1)
    r_b = scene.render(cam_b, gain=1.04, bias=3.0, noise=2.0, seed=seed * 2 + 2)
    view_f = RigView(f_px=f_px, cx=width / 2.0, cy=height / 2.0,
                     cam_height_m=h_nom_f if h_nom_f is not None else h_f,
                     pitch_deg=pitch_f)
    view_b = RigView(f_px=f_px, cx=width / 2.0, cy=height / 2.0,
                     cam_height_m=h_nom_b if h_nom_b is not None else h_b,
                     pitch_deg=pitch_b)
    return {"scene": scene, "front": r_f, "back": r_b,
            "view_front": view_f, "view_back": view_b,
            "true_baseline_m": d1 + d2, "cam_front": cam_f, "cam_back": cam_b}
