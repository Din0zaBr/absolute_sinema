"""Масштабная привязка по эталону + ректификация плоскости дороги.

Честная физика (см. дизайн §5/§10):
  • По ОДНОМУ кругу известного диаметра без фокусного расстояния камеры полная
    метрическая гомография плоскости НЕ определяется (8 DOF гомографии против
    5 DOF эллипса). Поэтому:
      - первичный путь — изотропный локальный масштаб по люку (хорош при near-nadir
        и копланарности эталона с дефектом, ±10–20%);
      - из эллипса люка оцениваем НАКЛОН и предупреждаем, если вид косой;
      - полную гомографию строим, только когда есть 4 известные копланарные точки
        (например, прямоугольный эталон) — `homography_from_4_points`.
  • Когда появится фокусное (оригиналы с EXIF) — добавим pose-from-circle.

Что именно меряется на люке: на ЗАКРЫТОМ люке (типовой случай на дороге) виден
наружный обод крышки ≈ 646 мм (ГОСТ 3634), а не лаз 600 мм — лаз наблюдаем
только у открытого колодца. Поэтому якорь по умолчанию — обод крышки 646 мм;
для открытого проёма вызывающий код передаёт known_mm=600 явно. Выбор якоря
отражается в reference.type/known_mm JSON-отчёта (дизайн §6/§13.3).

Защита от ложного эталона: кандидат Hough подтверждается эллипсом контура
(совпадение центра и диаметра). Неподтверждённый круг (колесо, круглая яма,
заплатка) масштабом НЕ становится — честнее не выдать см, чем выдать неверные.

Чистый модуль (numpy + opencv), без нейросетей.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class ReferenceMeasurement:
    available: bool
    type: str | None = None            # 'manhole_gost3634_cover' | 'marking' | None
    known_mm: float | None = None
    measured_px: float | None = None
    mm_per_px: float | None = None
    tilt_deg: float | None = None      # оценка наклона плоскости (0 = сверху)
    method: str | None = None
    confidence: str | None = None      # 'high' | 'medium' | 'low'
    error_band_pct: float | None = None
    homography: list | None = None     # 3x3 px->mm, если построена
    note: str = ""
    # Геометрия для overlay (в JSON-отчёт не сериализуется):
    circle_px: tuple | None = None     # (cx, cy, r) из Hough
    ellipse_px: tuple | None = None    # ((cx,cy),(MA,ma),angle) уточнённый контур

    def to_dict(self) -> dict:
        """Блок `scale` JSON-контракта §8 (+ расширения tilt_deg/method/confidence/note)."""
        return {
            "available": self.available,
            "mm_per_px": self.mm_per_px,
            "reference": {
                "type": self.type,
                "known_mm": self.known_mm,
                "measured_px": self.measured_px,
            },
            "homography_applied": self.homography is not None,
            "error_band_pct": self.error_band_pct,
            "tilt_deg": self.tilt_deg,
            "method": self.method,
            "confidence": self.confidence,
            "note": self.note,
        }


def detect_manhole_circle(image_bgr: np.ndarray,
                          cfg: config.InferenceConfig = config.DEFAULT_INFERENCE):
    """Найти люк как окружность через Hough. Возвращает (cx, cy, r) в пикселях
    ОРИГИНАЛЬНОГО кадра или None.

    Hough гоняется на копии с длинной стороной <= cfg.imgsz, а радиусы заданы
    долей короткой стороны кадра — детекция не зависит от разрешения фото
    (12-МП смартфонный кадр и превью 480 px обрабатываются одинаково).
    """
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(1.0, cfg.imgsz / max(h, w))
    if s < 1.0:
        gray = cv2.resize(gray, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                          interpolation=cv2.INTER_AREA)
    gh, gw = gray.shape
    gray = cv2.medianBlur(gray, 5)
    short = min(gh, gw)
    min_r = max(10, int(short * cfg.manhole_min_radius_frac))
    max_r = max(min_r + 1, int(short * cfg.manhole_max_radius_frac))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(1, gw // 4),
        param1=120, param2=60, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    # крупнейший круг — наиболее вероятный люк (эталон обычно крупный объект);
    # подтверждение, что это действительно люк, делает scale_from_manhole.
    cx, cy, r = max(circles, key=lambda c: c[2])
    return int(round(cx / s)), int(round(cy / s)), int(round(r / s))


def fit_manhole_ellipse(image_bgr: np.ndarray, circle):
    """Уточнить контур люка эллипсом в окрестности найденного круга.

    Возвращает ((cx,cy),(MA,ma),angle) cv2-эллипса или None.
    Эллипс даёт оценку наклона: ma/MA ≈ cos(tilt).
    """
    import cv2

    cx, cy, r = circle
    h, w = image_bgr.shape[:2]
    pad = int(r * 1.4)
    x0, y0 = max(cx - pad, 0), max(cy - pad, 0)
    x1, y1 = min(cx + pad, w), min(cy + pad, h)
    roi = image_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        if len(c) >= 5 and cv2.contourArea(c) > 0.2 * np.pi * r * r:
            (ecx, ecy), (MA, ma), ang = cv2.fitEllipse(c)
            best = ((ecx + x0, ecy + y0), (MA, ma), ang)
            break
    return best


def _ellipse_confirms_circle(ellipse, circle) -> bool:
    """Эллипс подтверждает круг-кандидат: центр и диаметр согласованы."""
    (ecx, ecy), (MA, ma), _ = ellipse
    cx, cy, r = circle
    if max(MA, ma) <= 0:
        return False
    center_ok = np.hypot(ecx - cx, ecy - cy) < 0.35 * r
    diameter_ok = 1.4 * r <= max(MA, ma) <= 2.9 * r
    return bool(center_ok and diameter_ok)


def scale_from_manhole(image_bgr: np.ndarray,
                       known_mm: float = config.GOST3634_COVER_OUTER_MM,
                       cfg: config.InferenceConfig = config.DEFAULT_INFERENCE
                       ) -> ReferenceMeasurement:
    """Изотропный локальный масштаб по люку (первичный путь, Сценарий B).

    По умолчанию меряем наружный обод крышки ЗАКРЫТОГО люка (646 мм по ГОСТ
    3634) — именно его видно на дороге. Для открытого проёма передать
    known_mm=config.GOST3634_CLEAR_OPENING_MM (600).
    """
    circle = detect_manhole_circle(image_bgr, cfg)
    if circle is None:
        return ReferenceMeasurement(available=False,
                                    note="Люк в кадре не найден (Hough).")

    ellipse = fit_manhole_ellipse(image_bgr, circle)
    if ellipse is None or not _ellipse_confirms_circle(ellipse, circle):
        # Круг есть, но контур не подтвердил, что это люк, — масштаб не выдаём:
        # ложный эталон тихо исказил бы все см и вердикт ГОСТ.
        return ReferenceMeasurement(
            available=False, circle_px=circle,
            note=("Найден круглый объект, но контур не подтвердил люк "
                  "(возможно колесо/яма/заплатка) — масштаб не выдан."),
        )

    (_, _), (MA, ma), _ = ellipse
    # Большая ось эллипса — истинный диаметр круга в плоскости дороги.
    diameter_px = max(MA, ma)
    mm_per_px = known_mm / diameter_px

    ratio = float(np.clip(min(MA, ma) / max(MA, ma), 0.05, 1.0))
    tilt_deg = float(np.degrees(np.arccos(ratio)))
    if tilt_deg < 15:
        confidence, err = "high", 12.0
    elif tilt_deg > 40:
        confidence, err = "low", 30.0
    else:
        confidence, err = "medium", 18.0

    return ReferenceMeasurement(
        available=True, type="manhole_gost3634_cover", known_mm=known_mm,
        measured_px=round(diameter_px, 1), mm_per_px=round(mm_per_px, 4),
        tilt_deg=round(tilt_deg, 1),
        method="hough_circle+ellipse_fit", confidence=confidence,
        error_band_pct=err, circle_px=circle, ellipse_px=ellipse,
        note=(f"Изотропный масштаб по ободу крышки люка {known_mm:.0f} мм (ГОСТ 3634). "
              + ("Вид близок к надиру." if confidence == "high"
                 else "Косой вид — точность снижена; рекомендуется серийная съёмка."
                 if confidence == "low" else "Умеренный наклон.")),
    )


def homography_from_4_points(image_pts: np.ndarray, world_mm_pts: np.ndarray):
    """Полная гомография px->мм по 4 известным копланарным точкам.

    Используется, когда в кадре есть прямоугольный эталон известных размеров или
    четыре точки с известными координатами на плоскости дороги. Возвращает 3x3 H.
    """
    import cv2

    image_pts = np.asarray(image_pts, dtype=np.float32)
    world_mm_pts = np.asarray(world_mm_pts, dtype=np.float32)
    H, _ = cv2.findHomography(image_pts, world_mm_pts, method=0)
    return H


def measure_distance_mm(H: np.ndarray, p1_px, p2_px) -> float:
    """Расстояние между двумя пикселями в мм через гомографию плоскости."""
    import cv2

    pts = np.array([[p1_px, p2_px]], dtype=np.float32)
    world = cv2.perspectiveTransform(pts, H)[0]
    return float(np.linalg.norm(world[0] - world[1]))
