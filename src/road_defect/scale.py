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

from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class ReferenceMeasurement:
    available: bool
    # 'manhole_gost3634_cover' | 'marking' | 'curb_gost6665' | None
    type: str | None = None
    known_mm: float | None = None
    measured_px: float | None = None
    mm_per_px: float | None = None
    tilt_deg: float | None = None      # оценка наклона плоскости (0 = сверху)
    method: str | None = None
    confidence: str | None = None      # 'high' | 'medium' | 'low'
    error_band_pct: float | None = None
    homography: list | None = None     # 3x3 px->mm, если построена
    note: str = ""
    # Мульти-эталон (F2): подтип якоря и след кросс-валидации.
    subtype: str | None = None         # 'manhole_cover' | 'stop_line' | 'line_1_1' | ...
    cross_checked: bool = False        # масштаб подтверждён вторым РАЗНОТИПНЫМ эталоном
    agreeing_types: list = field(default_factory=list)  # типы согласившихся эталонов
    candidates_n: int = 1              # сколько эталонов-кандидатов было доступно
    # Геометрия для overlay (в JSON-отчёт не сериализуется):
    circle_px: tuple | None = None     # (cx, cy, r) из Hough
    ellipse_px: tuple | None = None    # ((cx,cy),(MA,ma),angle) уточнённый контур
    # Полилинии эталонов разметки/борта для overlay (не сериализуются):
    polylines_px: list | None = None   # [np.ndarray Nx2, ...]

    def to_dict(self) -> dict:
        """Блок `scale` JSON-контракта §8 (+ расширения tilt/method/confidence/note,
        + аддитивные subtype/cross_checked/agreeing_types/candidates_n для мульти-эталона)."""
        return {
            "available": self.available,
            "mm_per_px": self.mm_per_px,
            "reference": {
                "type": self.type,
                "subtype": self.subtype,
                "known_mm": self.known_mm,
                "measured_px": self.measured_px,
            },
            "homography_applied": self.homography is not None,
            "error_band_pct": self.error_band_pct,
            "tilt_deg": self.tilt_deg,
            "method": self.method,
            "confidence": self.confidence,
            "cross_checked": self.cross_checked,
            "agreeing_types": list(self.agreeing_types),
            "candidates_n": self.candidates_n,
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
    # Canny+контуры на копии <=768 px: ROI фантомного кандидата может быть
    # почти полнокадровым, и полный прогон для каждого из 8 кандидатов давал
    # секунды на 12-МП кадр (находка ревью 2026-06-12). Точность достаточна:
    # эллипс уточняется субпиксельно самим fitEllipse, координаты масштабируются.
    s = min(1.0, 768.0 / max(roi.shape[:2]))
    small = roi if s >= 1.0 else cv2.resize(
        roi, (max(1, int(roi.shape[1] * s)), max(1, int(roi.shape[0] * s))),
        interpolation=cv2.INTER_AREA)
    rs = r * s
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_dist = None, float("inf")
    for c in contours:
        if len(c) >= 5 and cv2.contourArea(c) > 0.2 * np.pi * rs * rs:
            (ecx, ecy), (MA, ma), ang = cv2.fitEllipse(c)
            ecx, ecy, MA, ma = ecx / s, ecy / s, MA / s, ma / s
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


def _ellipse_looks_like_manhole(image_bgr: np.ndarray, ellipse) -> bool:
    """Независимая проверка ВНЕШНЕГО ВИДА диска-люка (не только геометрия).

    Путь 1 (Hough+контур) принимал эталон на чисто геометрическом согласии двух
    краевых детекторов — но круглая ЯМА/заплатка, пропущенная детектором (и
    потому не попавшая в exclude_boxes), тоже даёт согласованный круг+эллипс и
    становилась бы ложным масштабом 646 мм (аудит 2026-07-07, нарушение принципа
    «ложный масштаб хуже отсутствия»). Те же признаки уже есть в Пути 2
    (detect_manhole_ellipses) — здесь применяем их и к Пути 1:
      • сердцевина эллипса заметно ТЕМНЕЕ кольца вокруг (литой люк темнее асфальта);
      • ≥50% периметра эллипса лежит на кромке Canny (у крышки резкая кромка,
        у ямы/пятна/тени — размытая).
    """
    import cv2

    (ecx, ecy), (MA, ma), ang = ellipse
    if max(MA, ma) <= 0:
        return False
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    m_in = np.zeros((h, w), np.uint8)
    cv2.ellipse(m_in, ((float(ecx), float(ecy)), (float(MA), float(ma)),
                       float(ang)), 255, -1)
    if not m_in.any():
        return False
    k = max(3, (int(round(0.06 * max(MA, ma))) | 1))   # кольцо ~6% диаметра, нечёт
    ring = cv2.dilate(m_in, np.ones((k, k), np.uint8)) & ~m_in
    if not ring.any():
        return False
    if float(gray[m_in > 0].mean()) > float(gray[ring > 0].mean()) - 8.0:
        return False
    edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150),
                       np.ones((3, 3), np.uint8))
    poly = cv2.ellipse2Poly((int(round(ecx)), int(round(ecy))),
                            (int(round(MA / 2)), int(round(ma / 2))),
                            int(round(ang)), 0, 360, 5)
    if len(poly) == 0:
        return False
    on_edge = sum(1 for px, py in poly
                  if 0 <= px < w and 0 <= py < h and edges[py, px])
    return on_edge / len(poly) >= 0.5


def _center_in_any_box(cx: float, cy: float, boxes) -> bool:
    """Точка внутри какого-либо bbox (x, y, w, h)?"""
    for bx, by, bw, bh in boxes or []:
        if bx <= cx <= bx + bw and by <= cy <= by + bh:
            return True
    return False


def _ellipse_tilt_deg(ellipse) -> float:
    """Наклон плоскости по аспекту эллипса: ma/MA ≈ cos(tilt). 0° = вид сверху."""
    (_, _), (MA, ma), _ = ellipse
    ratio = float(np.clip(min(MA, ma) / max(MA, ma), 0.05, 1.0))
    return float(np.degrees(np.arccos(ratio)))


def _reference_on_ground_plane(ellipse, frame_h: int, tilt_deg: float,
                               cfg: config.InferenceConfig) -> tuple[bool, str]:
    """Диск-эталон (люк) обязан лежать на дорожной плоскости — в НИЖНЕЙ части
    кадра. Круглый эллипс ВЫСОКО в кадре физически не может быть люком на земле:
    далёкий люк у горизонта виден под скользящим углом и был бы сильно сплюснут
    (высокий наклон), а near-круглый отклик наверху — это вертикальная
    поверхность (знак/баннер/стена).

    Прецедент (цикл 8): круг на баннере забора прошёл Hough+контур и дал ложный
    масштаб — центр на 12% высоты кадра при наклоне 44.9°. Гейтим пересечение
    «верх кадра И слишком кругло», а не одну позицию: сплюснутый люк у горизонта
    (верх кадра, но высокий наклон) — валиден и проходит.

    Возвращает (на_плоскости, причина_отказа)."""
    (_, cy), _, _ = ellipse
    cy_frac = (cy / frame_h) if frame_h else 1.0
    if (cy_frac < cfg.manhole_horizon_frac
            and tilt_deg < cfg.manhole_min_grazing_tilt_deg):
        return False, (
            f"Кандидат-люк отклонён: центр в верхних {cy_frac * 100:.0f}% кадра "
            f"и слишком круглый (наклон {tilt_deg:.0f}° < "
            f"{cfg.manhole_min_grazing_tilt_deg:.0f}°) — это вертикальная "
            "поверхность (знак/баннер/стена), а не диск-люк на дороге; "
            "масштаб не выдан.")
    return True, ""


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
    # Гейт дорожной плоскости применяется ВНУТРИ перебора: отклонённый кандидат
    # (вертикальная поверхность) пропускается, и проверяется СЛЕДУЮЩИЙ — иначе
    # ложный баннер-круг с сильнейшим откликом Hough закрыл бы путь настоящему
    # люку-послабее в том же кадре (ревью 2026-07-05).
    circle = ellipse = method = None
    gate_reason = ""   # причина отказа гейта, если ВСЕ кандидаты вертикальные
    for cand in candidates:
        if _center_in_any_box(cand[0], cand[1], exclude_boxes):
            continue
        e = fit_manhole_ellipse(image_bgr, cand)
        if (e is not None and _ellipse_confirms_circle(e, cand)
                and _ellipse_looks_like_manhole(image_bgr, e)):
            on_plane, reason = _reference_on_ground_plane(
                e, image_bgr.shape[0], _ellipse_tilt_deg(e), cfg)
            if not on_plane:
                gate_reason = reason
                continue
            circle, ellipse, method = cand, e, "hough_circle+ellipse_fit"
            break

    # Путь 2 (opt-in): тёмные эллиптические блобы (работает при сильном наклоне).
    if ellipse is None and allow_blob_reference:
        for e in detect_manhole_ellipses(image_bgr, cfg):
            if _center_in_any_box(e[0][0], e[0][1], exclude_boxes):
                continue
            on_plane, reason = _reference_on_ground_plane(
                e, image_bgr.shape[0], _ellipse_tilt_deg(e), cfg)
            if not on_plane:
                gate_reason = reason
                continue
            ellipse, method = e, "dark_blob_ellipse"
            break

    if ellipse is None:
        # Ничего не подтверждено — масштаб не выдаём: ложный эталон тихо
        # исказил бы все см и вердикт ГОСТ. Причина гейта плоскости (если она
        # отсеяла кандидатов) информативнее общего «не подтверждён».
        blob_part = " и поиском тёмных эллипсов" if allow_blob_reference else ""
        if gate_reason:
            note = gate_reason
        elif candidates:
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
    tilt_deg = _ellipse_tilt_deg(ellipse)

    # type/subtype отражают ФАКТИЧЕСКИЙ якорь (закрытый обод 646 vs открытый лаз
    # 600), а не захардкоженную строку — иначе JSON врал бы про использованный
    # размер (аудит 2026-07-07). Выбор по близости переданного known_mm.
    is_clear_opening = (abs(known_mm - config.GOST3634_CLEAR_OPENING_MM)
                        < abs(known_mm - config.GOST3634_COVER_OUTER_MM))
    ref_type = ("manhole_gost3634_clear_opening" if is_clear_opening
                else "manhole_gost3634_cover")
    anchor_txt = "лаза (открытый проём)" if is_clear_opening else "обода крышки"

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
        available=True, type=ref_type, known_mm=known_mm,
        measured_px=round(diameter_px, 1), mm_per_px=round(mm_per_px, 4),
        tilt_deg=round(tilt_deg, 1),
        method=method, confidence=confidence,
        error_band_pct=err, circle_px=circle, ellipse_px=ellipse,
        note=(f"Изотропный масштаб по {anchor_txt} люка {known_mm:.0f} мм (ГОСТ 3634). "
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


# ===========================================================================
#  Мульти-эталон (F2): разметка + борт + люк → один масштаб с кросс-проверкой
# ===========================================================================
#
#  Тот же принцип, что у люка: ЛОЖНЫЙ эталон хуже отсутствия. Каждый детектор
#  независимо проходит свои фильтры ДО попадания в кандидаты; кросс-валидация
#  лишь повышает/понижает уверенность УЖЕ подтверждённого базового эталона и
#  никогда не воскрешает отбракованного кандидата. При конфликте разнотипных
#  эталонов масштаб не усредняется молча — он честно помечается ненадёжным.

_CONF_RANK = {"low": 0, "medium": 1, "high": 2}
_RANK_CONF = {0: "low", 1: "medium", 2: "high"}


def _promote_conf(conf: str | None, notches: int = 1) -> str:
    r = _CONF_RANK.get(conf or "low", 0)
    return _RANK_CONF[min(2, r + notches)]


# --- Разметка как эталон (ГОСТ Р 51256) ------------------------------------

def _detect_paint_stripes(image_bgr: np.ndarray,
                          cfg: config.InferenceConfig = config.DEFAULT_INFERENCE) -> list:
    """Кандидаты-полосы дорожной разметки (классика, без сетей).

    Белая краска: высокая яркость L и низкая насыщенность S (HLS). Полоса —
    сильно вытянутая, почти заполняющая свой повёрнутый прямоугольник компонента
    (это отсекает стрелки/символы/заплатки). Возвращает записи в пикселях
    ОРИГИНАЛА с профилем перпендикулярной ширины (для проверки тапера/равномерности)
    и углом оси к горизонтали (для классификации стоп-линия/линия 1.1).
    """
    import cv2

    h, w = image_bgr.shape[:2]
    s = min(1.0, cfg.imgsz / max(h, w))
    img = image_bgr
    if s < 1.0:
        img = cv2.resize(image_bgr,
                         (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                         interpolation=cv2.INTER_AREA)
    gh, gw = img.shape[:2]
    short = min(gh, gw)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L = hls[:, :, 1].astype(np.float32)
    S = hls[:, :, 2].astype(np.float32)
    # Краска ЯРЧЕ дороги на запас: порог-перцентиль на почти-равномерном асфальте
    # сел бы на сам уровень дороги и пометил бы половину покрытия как «краску».
    road_level = float(np.percentile(L, cfg.mark_road_level_pctl))
    thr = road_level + cfg.mark_l_margin
    paint = (((L >= thr) & (S <= cfg.mark_s_max)).astype(np.uint8)) * 255
    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    paint = cv2.morphologyEx(paint, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(paint, connectivity=8)
    stripes = []
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 50:
            continue
        comp = labels == lbl
        ys, xs = np.nonzero(comp)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        (rcx, rcy), (rw, rh), _ = cv2.minAreaRect(pts)
        long_side, short_side = max(rw, rh), min(rw, rh)
        if short_side < 1.0:
            continue
        if long_side / short_side < cfg.mark_stripe_min_elongation:
            continue
        if long_side < cfg.mark_min_len_frac * short:
            continue
        if area / (long_side * short_side + 1e-6) < cfg.mark_fill_min:
            continue

        # Принципиальная ось через SVD — без возни с конвенцией угла minAreaRect.
        mean = pts.mean(axis=0)
        centered = pts - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis, perp = vt[0], vt[1]
        t = centered @ axis
        u = centered @ perp

        # Положительный признак КРАСКИ (а не блика/полосы засветки/края кадра):
        # асфальт по ОБЕ стороны штриха должен быть заметно ТЕМНЕЕ его тела.
        # Без этого любой яркий вытянутый объект становился ложным эталоном
        # (ревью 2026-06-19). Замеряем L на сдвиге за кромку с обеих сторон.
        half_w = float((u.max() - u.min()) / 2.0)
        interior_L = float(L[ys, xs].mean())
        flank_gap = half_w + max(3.0, 0.6 * half_w)
        samp_t = np.linspace(float(t.min()) * 0.8, float(t.max()) * 0.8, 9)

        def _flank_L(sign: float):
            vals = []
            for tt in samp_t:
                px = mean + axis * float(tt) + perp * (sign * flank_gap)
                xi, yi = int(round(float(px[0]))), int(round(float(px[1])))
                if 0 <= xi < gw and 0 <= yi < gh:
                    vals.append(float(L[yi, xi]))
            return float(np.median(vals)) if vals else None

        left_L, right_L = _flank_L(1.0), _flank_L(-1.0)
        if (left_L is None or right_L is None
                or left_L > interior_L - cfg.mark_min_flank_contrast
                or right_L > interior_L - cfg.mark_min_flank_contrast):
            continue  # нет тёмного асфальта по обе стороны — это не разметка

        K = max(cfg.mark_min_samples + 2, 7)
        edges = np.linspace(float(t.min()), float(t.max()), K + 1)
        widths = []
        for i in range(K):
            sel = (t >= edges[i]) & (t <= edges[i + 1])
            if int(sel.sum()) >= 3:
                widths.append(float(u[sel].max() - u[sel].min()))

        a = abs(float(np.degrees(np.arctan2(axis[1], axis[0]))))
        angle_to_horiz = min(a, 180.0 - a)
        p0 = mean + axis * float(t.min())
        p1 = mean + axis * float(t.max())
        stripes.append({
            "center": (float(rcx / s), float(rcy / s)),
            "length_px": float(long_side / s),
            "angle_to_horiz_deg": float(angle_to_horiz),
            "frame_width_px": float(gw / s),
            "widths_px": [wd / s for wd in widths],
            "polyline": np.array([[p0[0] / s, p0[1] / s],
                                  [p1[0] / s, p1[1] / s]], np.float32),
        })
    stripes.sort(key=lambda d: d["length_px"], reverse=True)
    return stripes


def _classify_marking(stripe: dict, road_category: str | None,
                      cfg: config.InferenceConfig = config.DEFAULT_INFERENCE):
    """(known_mm, subtype) для полосы или (None, …) если неоднозначно.

    ВАЖНО (ревью 2026-06-19): ориентация в кадре НЕ доказывает класс линии —
    ракурсно «горизонтальной» становится и продольная линия, а ширина класса
    (1.1 80–150 / краевая 100–200 / СТОП 400 мм) по одному штриху неоднозначна.
    Поэтому здесь — только грубая гипотеза класса, а итоговый масштаб всё равно
    выдаётся как low-confidence с широкой полосой (scale_from_marking) и должен
    подтверждаться люком. Дополнительные отказы здесь убирают самые грубые ошибки:
      • СТОП-линия пересекает полосу → её протяжённость должна быть заметной долей
        ширины кадра, иначе это короткий ракурсный продольный штрих → отказ;
      • категория дороги обязательна (иначе ширина 1.1 неизвестна) → отказ.
    """
    if not (road_category and road_category in config.GOST51256_LINE_1_1_BY_CATEGORY):
        return None, "no_road_category"
    ang = stripe["angle_to_horiz_deg"]
    if ang <= 25.0:
        span_frac = stripe["length_px"] / max(stripe.get("frame_width_px", 1.0), 1.0)
        if span_frac < cfg.mark_stopline_min_span_frac:
            return None, "transverse_too_short"  # скорее ракурсная продольная
        return float(config.GOST51256_STOP_LINE_MM), "stop_line"
    if ang >= 50.0:
        return float(config.GOST51256_LINE_1_1_BY_CATEGORY[road_category]), "line_1_1"
    return None, "diagonal_ambiguous"


def scale_from_marking(image_bgr: np.ndarray,
                       cfg: config.InferenceConfig = config.DEFAULT_INFERENCE,
                       exclude_boxes=None,
                       road_category: str | None = None) -> ReferenceMeasurement:
    """Изотропный масштаб по ширине дорожной разметки (ГОСТ Р 51256).

    Меряется ШИРИНА полосы (короткая сторона), не длина. Уверенность ВСЕГДА low
    с широкой полосой (см. config.use_marking_reference): класс ширины разметки
    по одному штриху неоднозначен. Требует положительного признака краски (тёмный
    асфальт по обе стороны штриха) и честно отказывает (available=False) при
    неоднозначном классе, непостоянной ширине (трещина/тень) или сильном таперинге.
    Повышается до medium только при кросс-подтверждении люком (resolve_scale).
    """
    stripes = _detect_paint_stripes(image_bgr, cfg)
    if not stripes:
        return ReferenceMeasurement(available=False, note="Разметка в кадре не найдена.")

    last_note = "Разметка найдена, но не прошла проверки масштаба."
    for st in stripes:
        if _center_in_any_box(st["center"][0], st["center"][1], exclude_boxes):
            continue
        widths = st["widths_px"]
        if len(widths) < cfg.mark_min_samples:
            last_note = "Разметка: мало замеров ширины — масштаб не выдан."
            continue
        warr = np.asarray(widths, dtype=float)
        mean_w = float(warr.mean())
        if mean_w <= 0:
            continue
        if float(warr.std() / mean_w) > cfg.mark_width_cv_max:
            last_note = ("Разметка: ширина непостоянна вдоль полосы "
                         "(трещина/тень?) — масштаб не выдан.")
            continue
        if float(warr.max() / max(warr.min(), 1e-6)) > cfg.mark_width_taper_max:
            last_note = ("Разметка: сильный перспективный таперинг полосы — "
                         "единый масштаб ненадёжен, не выдан.")
            continue
        known_mm, subtype = _classify_marking(st, road_category, cfg)
        if known_mm is None:
            last_note = ("Класс/ширина разметки неоднозначны (нет категории дороги, "
                         "диагональная полоса или короткий поперечный штрих) — "
                         "масштаб по разметке не выдан.")
            continue
        measured_px = float(np.median(warr))
        if measured_px <= 0:
            continue
        # Уверенность ЖЁСТКО 'low' с широкой полосой: класс ширины разметки по
        # одному штриху неоднозначен (1.1 80–150 / краевая 100–200 / СТОП 400 мм),
        # ошибка класса до ~2.5–5× не покрывается узкой полосой. Повышение — только
        # через кросс-подтверждение люком в resolve_scale (_cross_validate).
        return ReferenceMeasurement(
            available=True, type="marking", subtype=subtype,
            known_mm=float(known_mm), measured_px=round(measured_px, 1),
            mm_per_px=round(known_mm / measured_px, 4), tilt_deg=None,
            method="paint_stripe_width", confidence="low",
            error_band_pct=cfg.mark_error_band_pct,
            polylines_px=[st["polyline"]],
            note=(f"Масштаб по ширине разметки (гипотеза {subtype}, {known_mm:.0f} мм, "
                  "ГОСТ Р 51256). ЭКСПЕРИМЕНТАЛЬНО: класс ширины по одному штриху "
                  "неоднозначен — low-confidence, требует подтверждения люком."),
        )
    return ReferenceMeasurement(available=False, note=last_note)


# --- Борт (бордюр) как эталон (ГОСТ 6665) — выключен по умолчанию -----------

def scale_from_curb(image_bgr: np.ndarray,
                    cfg: config.InferenceConfig = config.DEFAULT_INFERENCE,
                    exclude_boxes=None) -> ReferenceMeasurement:
    """Экспериментальный масштаб по бортовому камню. ВЫКЛЮЧЕН по умолчанию.

    Честность прежде всего. Тень под бортом — задокументированный ложный эталон
    (PROJECT.md §4.4, п.5), поэтому ОДНА тёмная линия эталоном НЕ становится:
    требуются ДВЕ почти-параллельные длинные линии (верхняя грань + основание
    борта) с осмысленным и устойчивым зазором между ними; зазор трактуется как
    высота видимой грани борта 150 мм (ГОСТ 6665). Верх борта приподнят над
    покрытием (~150 мм), поэтому масштаб систематически смещён — уверенность
    жёстко capped «low», и resolve_scale использует борт ТОЛЬКО как
    подтверждающий эталон, никогда как единственный источник см.

    Ограничение v1: контроль швов (длина камня 1000 мм) не реализован — борт
    остаётся cross-check-эталоном, поэтому это допустимо (см. PROJECT.md).
    """
    import cv2

    if not cfg.allow_curb_reference:
        return ReferenceMeasurement(
            available=False,
            note="Эталон по борту выключен (включить флагом --curb-ref).")

    h, w = image_bgr.shape[:2]
    s = min(1.0, cfg.imgsz / max(h, w))
    img = image_bgr
    if s < 1.0:
        img = cv2.resize(image_bgr,
                         (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                         interpolation=cv2.INTER_AREA)
    gh, gw = img.shape[:2]
    short = min(gh, gw)
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80,
                            minLineLength=int(0.25 * short), maxLineGap=20)
    if lines is None or len(lines) < 2:
        return ReferenceMeasurement(
            available=False,
            note="Борт: не найдено двух параллельных длинных линий "
                 "(одна линия/тень эталоном не становится).")

    # Сегменты: угол (0..180) и средняя точка. Знаковый перпендикулярный сдвиг
    # считаем ПОСЛЕ кластеризации в ЕДИНОЙ нормали кластера — если считать его
    # в собственном угле каждого сегмента, на склейке 0/180° знак cos
    # переворачивается и зазор фабрикуется (×14 ошибка, ревью 2026-06-19).
    segs = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        ang = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))) % 180.0)
        length = float(np.hypot(x2 - x1, y2 - y1))
        segs.append((ang, (x1 + x2) / 2.0, (y1 + y2) / 2.0, length))

    # Доминирующее направление: самый длинный сегмент задаёт угол кластера.
    segs.sort(key=lambda t: t[3], reverse=True)
    base_ang = segs[0][0]
    cluster = [sg for sg in segs
               if min(abs(sg[0] - base_ang), 180.0 - abs(sg[0] - base_ang)) <= 10.0]
    if len(cluster) < 2:
        return ReferenceMeasurement(
            available=False,
            note="Борт: вторая параллельная линия не найдена — эталон не выдан.")

    # Единая нормаль кластера → знак сдвига консистентен на всех сегментах.
    # Средние точки сегментов сохраняются: позиция КАНДИДАТА нужна для
    # exclude_boxes (ревью 2026-07-02: проверялся центр КАДРА, не кандидата).
    base_rad = np.radians(base_ang)
    nx, ny = -np.sin(base_rad), np.cos(base_rad)
    offsets = sorted((mx * nx + my * ny, mx, my) for _, mx, my, _ in cluster)
    # Слить близкие сдвиги (две Canny-кромки ОДНОЙ линии) в группы-кромки, чтобы
    # «зазор» не считался как полный размах по нескольким линиям (ревью 2026-06-20).
    merge_tol = 0.01 * short
    groups = [[offsets[0]]]
    for off in offsets[1:]:
        if off[0] - groups[-1][-1][0] <= merge_tol:
            groups[-1].append(off)
        else:
            groups.append([off])
    centers = [sum(o[0] for o in g) / len(g) for g in groups]
    if len(centers) < 2:
        # Кромки практически совпадают — это одна линия (две её кромки), не борт.
        return ReferenceMeasurement(
            available=False,
            note="Борт: параллельные кромки слишком близки (одна линия) — "
                 "эталон не выдан.")
    if len(centers) > 2:
        # Верх борта + основание + лоток/шов/тень: какой зазор есть грань 150 мм —
        # неоднозначно. Честнее отказать, чем взять произвольную пару.
        return ReferenceMeasurement(
            available=False,
            note="Борт: >2 параллельных линий — неоднозначно (лоток/шов/тень?) — "
                 "эталон не выдан.")
    gap_px = (centers[1] - centers[0]) / s   # в координатах оригинала
    min_gap = 0.03 * (short / s)
    max_gap = 0.6 * (short / s)
    if gap_px < min_gap:
        return ReferenceMeasurement(
            available=False,
            note="Борт: параллельные кромки слишком близки (одна линия) — "
                 "эталон не выдан.")
    if gap_px > max_gap:
        return ReferenceMeasurement(
            available=False,
            note="Борт: неправдоподобно большой зазор между кромками — отклонён.")

    # Позиция КАНДИДАТА — средняя точка сегментов обеих кромок (в координатах
    # оригинала), а не центр кадра: круглая яма у края кадра не должна
    # пропускать борт, а борт в центре — отклоняться из-за ямы в углу.
    pts = [(mx, my) for g in groups for _, mx, my in g]
    cand_x = sum(p[0] for p in pts) / len(pts) / s
    cand_y = sum(p[1] for p in pts) / len(pts) / s
    if _center_in_any_box(cand_x, cand_y, exclude_boxes):
        return ReferenceMeasurement(
            available=False, note="Борт: кандидат внутри bbox дефекта — отклонён.")

    known_mm = float(config.GOST6665_CURB_FACE_HEIGHT_MM)
    return ReferenceMeasurement(
        available=True, type="curb_gost6665", subtype="curb_face",
        known_mm=known_mm, measured_px=round(gap_px, 1),
        mm_per_px=round(known_mm / gap_px, 4), tilt_deg=None,
        method="parallel_curb_edges", confidence="low", error_band_pct=30.0,
        note=("Масштаб по высоте грани борта 150 мм (ГОСТ 6665) — "
              "ВЕРХ БОРТА ПРИПОДНЯТ над покрытием, масштаб смещён; "
              "используется только как подтверждающий эталон."),
    )


# --- Выбор и кросс-валидация эталонов --------------------------------------

def _rank_candidates(cands: list) -> list:
    """Доступные базовые эталоны по убыванию пригодности.

    Борт исключён из базовых (только cross-check). Сортировка: уверенность ↓,
    полоса ошибки ↑, приоритет типа ↓ (тай-брейк).
    """
    avail = [c for c in cands
             if c.available and c.type != "curb_gost6665" and c.mm_per_px]

    def key(c):
        return (_CONF_RANK.get(c.confidence or "low", 0),
                -(c.error_band_pct if c.error_band_pct is not None else 999.0),
                config.REFERENCE_TYPE_PRIOR.get(c.type, 0))

    return sorted(avail, key=key, reverse=True)


def _cross_validate(base: ReferenceMeasurement, others: list,
                    cfg: config.InferenceConfig) -> ReferenceMeasurement:
    """Подтвердить/опровергнуть базовый масштаб разнотипными эталонами.

    Согласие (mm/px в пределах допуска) → +ступень уверенности; значение и
    полоса ошибки остаются базовыми (сужение без слияния значений нечестно).
    Конфликт (расхождение больше порога) → понижение до low, расширение полосы и
    предупреждение в note (маркер «конфликт эталонов» — конвейер выносит в
    warnings). Борт может только ПОДТВЕРЖДАТЬ (свой допуск), но не опровергать.
    Работает на копии — кандидаты не мутируются.
    """
    import dataclasses

    result = dataclasses.replace(base, agreeing_types=list(base.agreeing_types))
    if result.mm_per_px is None:
        return result

    def _agree(a: float, b: float) -> float:
        # Симметрично: вердикт не зависит от того, кто выбран базой.
        return abs(a - b) / max((a + b) / 2.0, 1e-9)

    candidates = [o for o in others
                  if o.available and o.mm_per_px is not None and o.type != result.type]

    # 1) Конфликт ДОМИНИРУЕТ и «липкий»: если хоть один РАЗНОТИПНЫЙ не-бортовой
    #    эталон расходится сильнее порога — масштаб ненадёжен, и ни последующее
    #    согласие, ни борт не «воскрешают» уверенность (ревью 2026-06-19).
    conflicts = [o for o in candidates if o.type != "curb_gost6665"
                 and _agree(result.mm_per_px, o.mm_per_px) >= cfg.xcheck_conflict]
    if conflicts:
        worst = max(conflicts, key=lambda o: _agree(result.mm_per_px, o.mm_per_px))
        ag = _agree(result.mm_per_px, worst.mm_per_px)
        result.confidence = "low"
        result.cross_checked = False
        result.error_band_pct = round(max(
            result.error_band_pct or 0.0, worst.error_band_pct or 0.0, ag * 100.0), 1)
        result.note = (
            result.note + " " +
            f"конфликт эталонов: {result.type} mm/px {result.mm_per_px} vs "
            f"{worst.type} {worst.mm_per_px} (расхождение {ag * 100:.0f}%) — "
            "масштаб ненадёжен").strip()
        return result

    # 2) Нет конфликтов → подтверждение разнотипными эталонами. ПОВЫШАТЬ уверенность
    #    вправе лишь НЕЗАВИСИМЫЙ эталон НЕ СЛАБЕЕ базового: слабый (low)
    #    класс-неоднозначный эталон-разметка не должен делать люк «high»
    #    (ревью 2026-06-20). Борт — НИКОГДА (систематически смещён): только
    #    подтверждает присутствие (cross_checked). Повышение — не более одной ступени.
    #    Полоса ошибки НЕ сужается: mm_per_px остаётся значением БАЗОВОГО эталона
    #    (усреднение сломало бы связь known_mm/measured_px ↔ mm_per_px в §8),
    #    а заявлять точность слитой оценки, не выдавая её значение, нечестно
    #    (ревью 2026-07-02). Выгода согласия идёт в уверенность, не в полосу —
    #    та же логика, что в fusion.py для двух видов.
    base_rank = _CONF_RANK.get(result.confidence or "low", 0)
    agreeing_strong = []
    for o in candidates:
        is_curb = o.type == "curb_gost6665"
        tol = cfg.curb_crosscheck_tol if is_curb else cfg.xcheck_tol
        if _agree(result.mm_per_px, o.mm_per_px) <= tol:
            if o.type not in result.agreeing_types:
                result.agreeing_types.append(o.type)
            result.cross_checked = True
            if (not is_curb) and _CONF_RANK.get(o.confidence or "low", 0) >= base_rank:
                agreeing_strong.append(o)
    if agreeing_strong:
        result.confidence = _promote_conf(result.confidence)  # одна ступень
    return result


def resolve_scale(image_bgr: np.ndarray,
                  cfg: config.InferenceConfig = config.DEFAULT_INFERENCE,
                  exclude_boxes=None,
                  road_category: str | None = None) -> ReferenceMeasurement:
    """Единая точка масштаба (F2): собрать все доступные эталоны, выбрать базовый
    и кросс-валидировать. Заменяет прямой вызов scale_from_manhole в конвейере.

    Намеренно вызывает scale_from_manhole по имени модуля — это сохраняет
    monkeypatch honest-mode тестов (они патчат scale_from_manhole на модуле).
    Возвращает ОДИН ReferenceMeasurement: конвейер потребляет один mm_per_px,
    как и раньше. Если базового эталона нет (борт в одиночку не считается) —
    available=False с самой информативной заметкой об отказе.
    """
    cands: list = []
    notes: list = []

    base_manhole = scale_from_manhole(image_bgr, cfg=cfg, exclude_boxes=exclude_boxes)
    if base_manhole.available:
        cands.append(base_manhole)
    elif base_manhole.note:
        notes.append(base_manhole.note)

    if cfg.use_marking_reference:
        m = scale_from_marking(image_bgr, cfg=cfg, exclude_boxes=exclude_boxes,
                               road_category=road_category)
        if m.available:
            cands.append(m)
        elif m.note:
            notes.append(m.note)

    curb_candidates: list = []
    if cfg.allow_curb_reference:
        c = scale_from_curb(image_bgr, cfg=cfg, exclude_boxes=exclude_boxes)
        if c.available:
            curb_candidates.append(c)
        elif c.note:
            notes.append(c.note)

    n_available = len(cands) + len(curb_candidates)
    base_eligible = _rank_candidates(cands)
    if not base_eligible:
        return ReferenceMeasurement(
            available=False, candidates_n=n_available,
            note=" ".join(n for n in notes if n) or "Эталон масштаба не найден.")

    base = base_eligible[0]
    others = [c for c in cands if c is not base] + curb_candidates
    result = _cross_validate(base, others, cfg)
    result.candidates_n = n_available
    return result
