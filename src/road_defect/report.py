"""Сборка JSON-контракта (см. дизайн §8) и аннотированного изображения.

JSON стабилен и потребляется подсистемами 2–5 (backend/БД/карта/отчёты).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _finite_or_none(obj):
    """Рекурсивно заменить не-конечные float (NaN/Inf) на None.

    json.dumps(allow_nan=True) по умолчанию пишет NaN/Infinity — это невалидный
    JSON, который роняет строгие парсеры подсистем 2–5; None («неизвестно») —
    честнее и валиднее выдуманного числа (аудит 2026-07-07)."""
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_or_none(v) for v in obj]
    return obj


# Цвета классов (BGR) для overlay.
_CLASS_COLORS = {
    "pothole": (0, 0, 255),
    "patch": (0, 165, 255),
    "alligator_crack": (0, 255, 255),
    "longitudinal_crack": (255, 255, 0),
    "transverse_crack": (255, 0, 255),
}
_DEFAULT_COLOR = (0, 255, 0)


def mask_to_rle(mask: np.ndarray) -> dict:
    """COCO-стиль RLE (column-major). counts начинается с числа нулей.

    ВНИМАНИЕ по осям (контракт §8): mask_rle.size = [H, W] (row, col — как
    arr.shape), тогда как report.image_size_px = [W, H] (width, height). Разные
    порядки в одном JSON — исторический контракт; потребители подсистем 2–5
    должны учитывать оба (валидатор scripts/validate_outputs.py это проверяет)."""
    arr = np.asarray(mask).astype(np.uint8)
    m = arr.ravel(order="F")
    if m.size == 0:
        return {"size": list(arr.shape), "counts": []}
    changes = np.where(np.diff(m.astype(np.int16)) != 0)[0] + 1
    bounds = np.concatenate(([0], changes, [m.size]))
    run_lengths = np.diff(bounds)
    first_value = int(m[0])
    counts = []
    if first_value == 1:
        counts.append(0)  # COCO RLE всегда стартует с серии нулей
    counts.extend(int(r) for r in run_lengths)
    return {"size": list(arr.shape), "counts": counts}


def build_report(image_name: str, image_size_px, mode: str,
                 reference: dict, defects: list, warnings: list,
                 location: dict | None = None,
                 fusion: dict | None = None) -> dict:
    report = {
        "image": image_name,
        "mode": mode,
        "image_size_px": list(image_size_px),
        "scale": reference,
        "defects": defects,
        "location": location or {"available": False, "source": "manual_later"},
        "warnings": warnings,
        "engine_version": __import__("road_defect").__version__,
    }
    # Блок слияния двух видов (только pair-режим; в одиночных отчётах отсутствует).
    if fusion is not None:
        report["fusion"] = fusion
    return report


def save_report(report: dict, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.json"
    path.write_text(
        json.dumps(_finite_or_none(report), ensure_ascii=False, indent=2,
                   allow_nan=False),
        encoding="utf-8")
    return path


def unique_stem(stem: str, used: set) -> str:
    """Уникальное имя результата в рамках прогона.

    Смартфонные серии вида IMG_20260611_123456/123457 не должны затирать
    отчёты друг друга. Добавляет стем в `used` и при коллизии — суффикс _N.
    """
    candidate = stem
    n = 2
    while candidate in used:
        candidate = f"{stem}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def draw_overlay(image_bgr: np.ndarray, defects: list, masks: list,
                 reference_circle=None, reference_ellipse=None,
                 reference_label: str = "manhole ref",
                 reference_polylines=None) -> np.ndarray:
    """Нарисовать маски, боксы и подписи. masks параллелен defects.

    reference_polylines — линии эталона-разметки/борта (Nx2 px) для overlay."""
    import cv2

    vis = image_bgr.copy()
    overlay = image_bgr.copy()
    for d, mask in zip(defects, masks):
        color = _CLASS_COLORS.get(d["class"], _DEFAULT_COLOR)
        if mask is not None and np.asarray(mask).any():
            overlay[np.asarray(mask) > 0] = color
        x, y, w, h = [int(v) for v in d["bbox_px"]]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
    vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

    for d in defects:
        color = _CLASS_COLORS.get(d["class"], _DEFAULT_COLOR)
        x, y, w, h = [int(v) for v in d["bbox_px"]]
        metric = d.get("metric", {})
        if metric.get("available") and metric.get("equivalent_diameter_cm") is not None:
            size_txt = f"~{metric['equivalent_diameter_cm']}cm"
        else:
            size_txt = f"{int(d['shape']['equivalent_diameter_px'])}px"
        label = f"{d['class']} {d['confidence']:.2f} {size_txt}"
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
        _label(vis, label, x, y, color)

    # Эталон: рисуем уточнённый эллипс (по нему посчитан mm_per_px), а круг
    # Hough — только как фолбэк, чтобы картинка совпадала с измерением.
    if reference_ellipse is not None:
        (ecx, ecy), (MA, ma), ang = reference_ellipse
        cv2.ellipse(vis, ((ecx, ecy), (MA, ma), ang), (255, 0, 0), 3)
        _label(vis, reference_label,
               int(ecx - MA / 2), int(ecy - ma / 2), (255, 0, 0))
    elif reference_circle is not None:
        cx, cy, r = reference_circle
        cv2.circle(vis, (cx, cy), r, (255, 0, 0), 3)
        _label(vis, reference_label, cx - r, cy - r, (255, 0, 0))
    elif reference_polylines:
        for poly in reference_polylines:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, (255, 0, 0), 3)
        first = np.asarray(reference_polylines[0], dtype=np.int32).reshape(-1, 2)
        _label(vis, reference_label, int(first[0][0]), int(first[0][1]), (255, 0, 0))
    return vis


def _label(img, text, x, y, color):
    import cv2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
