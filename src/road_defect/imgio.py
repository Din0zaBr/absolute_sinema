"""Чтение/запись изображений с юникод-путями (Windows-safe).

cv2.imread/imwrite на Windows открывают файл через байтовый fopen (ANSI/cp1251):
кириллический путь тихо даёт None при чтении и False при записи. Обход —
np.fromfile + cv2.imdecode на чтении, cv2.imencode + tofile на записи.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_image(path: str | Path) -> np.ndarray | None:
    """Прочитать изображение (BGR) по любому пути. None, если не читается."""
    import cv2

    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def write_image(path: str | Path, image: np.ndarray) -> bool:
    """Записать изображение по любому пути. False при любой ошибке."""
    import cv2

    path = Path(path)
    ok, buf = cv2.imencode(path.suffix if path.suffix else ".jpg", image)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return True
