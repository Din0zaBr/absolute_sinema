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


def detect_manhole_circles(image_bgr: np.ndarray,
                           cfg: config.InferenceConfig = config.DEFAULT_INFERENCE,
                           max_candidates: int = 8) -> list:
    """Кандидаты-окружности (люк) через Hough, в пикселях ОРИГИНАЛЬНОГО кадра.

    Возвращает список (cx, cy, r) в порядке силы отклика аккумулятора Hough
    (сильнейший первым) — НЕ по размеру: эвристика «крупнейший круг = люк»
    на реальных 12-МП фото выбирала гигантский фантомный круг из текстуры
    асфальта (r≈400+), и настоящий люк даже не проверялся подтверждением
    (диагностика 2026-06-12, scripts/scale_debug.py).

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
        return []
    out = []
    for cx, cy, r in np.round(circles[0]).astype(int)[:max_candidates]:
        out.append((int(round(cx / s)), int(round(cy / s)), int(round(r / s))))
    return out


def detect_manhole_circle(image_bgr: np.ndarray,
                          cfg: config.InferenceConfig = config.DEFAULT_INFERENCE):
    """Сильнейший кандидат-окружность или None (обёртка для совместимости)."""
    candidates = detect_manhole_circles(image_bgr, cfg)
    return candidates[0] if candidates else None


def detect_manhole_ellipses(image_bgr: np.ndarray,
                            cfg: config.InferenceConfig = config.DEFAULT_INFERENCE,
                            max_candidates: int = 8) -> list:
    """Кандидаты-люки как тёмные эллиптические блобы (работает при косом виде).

    HoughCircles ищет только почти-круги: на реальных смартфонных фото люк
    сплюснут перспективой в эллипс 2:1+ и в кандидаты Hough не попадает вовсе
    (диагностика 2026-06-12). Здесь ищем напрямую: тёмные области (крышка
    чугунного люка темнее асфальта) → контуры → fitEllipse → фильтры:
      • размер: 8%..90% короткой стороны кадра — мелкий блоб это пятно, а не
        эталон (первая версия фильтров пропустила пятно 83 px как «люк» —
        ложный масштаб, поймано на реальном фото 2026-06-12);
      • эллипс ЦЕЛИКОМ в кадре — обрезанный люк не измерить;
      • сплюснутость не более 5:1 (дальше масштаб бессмыслен);
      • контур заполняет эллипс (0.75..1.25 площади) — отсекает трещины/тени;
      • внутри заметно темнее кольца снаружи;
      • опора на кромку: ≥50% периметра эллипса лежит на краях Canny — у литой
        крышки кромка резкая, у масляного пятна/тени размытая.

    Возвращает [((cx,cy),(MA,ma),angle), ...] в пикселях оригинала,
    отсортированные по качеству подгонки (лучший первым).
    """
    import cv2

    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray_full.shape
    s = min(1.0, cfg.imgsz / max(h, w))
    gray = gray_full
    if s < 1.0:
        gray = cv2.resize(gray_full,
                          (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                          interpolation=cv2.INTER_AREA)
    gray = cv2.medianBlur(gray, 5)
    gh, gw = gray.shape
    short = min(gh, gw)

    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))

    min_d = max(20.0, 0.08 * short)
    max_d = 2 * short * cfg.manhole_max_radius_frac
    scored = []
    for c in contours:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if area < 200:
            continue
        (ecx, ecy), (MA, ma), ang = cv2.fitEllipse(c)
        if max(MA, ma) <= 0 or not (min_d <= max(MA, ma) <= max_d):
            continue
        # аспект < 0.45 (наклон > ~63°): изотропный масштаб там бессмыслен,
        # а сильно сплюснутые «эллипсы» — это в основном тени (FP 2026-06-12)
        if min(MA, ma) / max(MA, ma) < 0.45:
            continue
        # эллипс целиком в кадре (точные экстенты повёрнутого эллипса)
        a2, b2 = (MA / 2.0) ** 2, (ma / 2.0) ** 2
        rad = np.radians(ang)
        ext_x = float(np.sqrt(a2 * np.cos(rad) ** 2 + b2 * np.sin(rad) ** 2))
        ext_y = float(np.sqrt(a2 * np.sin(rad) ** 2 + b2 * np.cos(rad) ** 2))
        if (ecx - ext_x < 2 or ecx + ext_x > gw - 2
                or ecy - ext_y < 2 or ecy + ext_y > gh - 2):
            continue
        fill = area / (np.pi * MA * ma / 4.0 + 1e-6)
        if not (0.75 <= fill <= 1.25):
            continue

        # контраст: внутри эллипса темнее «кольца» вокруг
        m_in = np.zeros(gray.shape, np.uint8)
        cv2.ellipse(m_in, ((ecx, ecy), (MA, ma), ang), 255, -1)
        m_ring = cv2.dilate(m_in, np.ones((15, 15), np.uint8)) & ~m_in
        if not m_ring.any():
            continue
        if float(gray[m_in > 0].mean()) > float(gray[m_ring > 0].mean()) - 8.0:
            continue

        # опора на кромку: периметр эллипса должен лежать на краях Canny
        poly = cv2.ellipse2Poly((int(round(ecx)), int(round(ecy))),
                                (int(round(MA / 2)), int(round(ma / 2))),
                                int(round(ang)), 0, 360, 5)
        on_edge = sum(1 for px, py in poly
                      if 0 <= px < gw and 0 <= py < gh and edges[py, px])
        if len(poly) == 0 or on_edge / len(poly) < 0.5:
            continue

        scored.append((abs(1.0 - fill),
                       ((ecx / s, ecy / s), (MA / s, ma / s), ang)))
    scored.sort(key=lambda t: t[0])
    return [e for _, e in scored[:max_candidates]]


def fit_manhole_ellipse(image_bgr: np.ndarray, circle):
    """Уточнить контур люка эллипсом в окрестности найденного круга.

    Из всех достаточно крупных контуров берётся эллипс с центром, ближайшим к
    кругу-кандидату (раньше брался первый попавшийся — порядок findContours
    произволен, и эллипс мог сесть на тень/трещину рядом с люком).

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
    best, best_dist = None, float("inf")
    for c in contours:
        if len(c) >= 5 and cv2.contourArea(c) > 0.2 * np.pi * r * r:
            (ecx, ecy), (MA, ma), ang = cv2.fitEllipse(c)
            dist = float(np.hypot(ecx + x0 - cx, ecy + y0 - cy))
            if dist < best_dist:
                best, best_dist = ((ecx + x0, ecy + y0), (MA, ma), ang), dist
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


def _center_in_any_box(cx: float, cy: float, boxes) -> bool:
    """Точка внутри какого-либо bbox (x, y, w, h)?"""
    for bx, by, bw, bh in boxes or []:
        if bx <= cx <= bx + bw and by <= cy <= by + bh:
            return True
    return False


def scale_from_manhole(image_bgr: np.ndarray,
                       known_mm: float = config.GOST3634_COVER_OUTER_MM,
                       cfg: config.InferenceConfig = config.DEFAULT_INFERENCE,
                       exclude_boxes=None,
                       allow_blob_reference: bool | None = None) -> ReferenceMeasurement:
    """Изотропный локальный масштаб по люку (первичный путь, Сценарий B).

    По умолчанию меряем наружный обод крышки ЗАКРЫТОГО люка (646 мм по ГОСТ
    3634) — именно его видно на дороге. Для открытого проёма передать
    known_mm=config.GOST3634_CLEAR_OPENING_MM (600).

    Два пути поиска:
      1) Hough-круг + подтверждение контуром (вид близкий к надиру);
      2) тёмный эллиптический блоб (косой вид: люк сплюснут в эллипс 2:1+,
         HoughCircles его не видит в принципе). ВЫКЛЮЧЕН по умолчанию: на
         реальном фото тень под бордюром прошла все фильтры и стала ложным
         эталоном (2026-06-12). Включается allow_blob_reference=True (или
         cfg.allow_blob_reference) — после «золотой» валидации масштаба §11.

    exclude_boxes — bbox'ы найденных дефектов: кандидат с центром внутри
    детекции (круглая яма, тёмная заплатка) эталоном не становится.
    """
    if allow_blob_reference is None:
        allow_blob_reference = cfg.allow_blob_reference
    candidates = detect_manhole_circles(image_bgr, cfg)

    # Путь 1: перебор Hough-кандидатов в порядке силы отклика — первый, чей
    # контур подтверждён эллипсом. Раньше проверялся ровно один «крупнейший»
    # круг — на реальных фото это был фантом из текстуры асфальта, и настоящий
    # люк не рассматривался вовсе.
    circle = ellipse = method = None
    for cand in candidates:
        if _center_in_any_box(cand[0], cand[1], exclude_boxes):
            continue
        e = fit_manhole_ellipse(image_bgr, cand)
        if e is not None and _ellipse_confirms_circle(e, cand):
            circle, ellipse, method = cand, e, "hough_circle+ellipse_fit"
            break

    # Путь 2 (opt-in): тёмные эллиптические блобы (работает при сильном наклоне).
    if ellipse is None and allow_blob_reference:
        for e in detect_manhole_ellipses(image_bgr, cfg):
            if _center_in_any_box(e[0][0], e[0][1], exclude_boxes):
                continue
            ellipse, method = e, "dark_blob_ellipse"
            break

    if ellipse is None:
        # Ничего не подтверждено — масштаб не выдаём: ложный эталон тихо
        # исказил бы все см и вердикт ГОСТ.
        blob_part = " и поиском тёмных эллипсов" if allow_blob_reference else ""
        if candidates:
            note = (f"Кандидатов-кругов: {len(candidates)}, но люк не "
                    f"подтверждён контуром{blob_part} "
                    "(возможно колесо/яма/заплатка) — масштаб не выдан.")
        else:
            note = f"Люк в кадре не найден (Hough{blob_part})."
        return ReferenceMeasurement(
            available=False, circle_px=candidates[0] if candidates else None,
            note=note)

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
    if method == "dark_blob_ellipse" and confidence == "high":
        # блоб-путь валидирован косвеннее Hough+контура
        confidence, err = "medium", max(err, 18.0)

    return ReferenceMeasurement(
        available=True, type="manhole_gost3634_cover", known_mm=known_mm,
        measured_px=round(diameter_px, 1), mm_per_px=round(mm_per_px, 4),
        tilt_deg=round(tilt_deg, 1),
        method=method, confidence=confidence,
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
