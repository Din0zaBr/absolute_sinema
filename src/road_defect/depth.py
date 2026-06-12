"""Относительная глубина дефекта (опционально, Сценарий A/B).

ВАЖНО (честность): этот модуль НИКОГДА не выдаёт глубину в сантиметрах. Только
относительный бакет «shallow|medium|deep» как индикатор приоритета. Достоверная
см-глубина — только Сценарий C (photogrammetry.py) или датчик глубины.

Использует Depth Anything V2 Small через transformers, если доступен; иначе no-op.
"""
from __future__ import annotations

import numpy as np

from . import config


class RelativeDepth:
    def __init__(self):
        self._pipe = None
        self._tried = False
        self.available = False
        self._frame_key = None   # кадр, для которого закэширована карта глубины
        self._depth_map = None   # полнокадровый инференс — 1 раз на кадр, не на дефект

    def _load(self):
        if self._tried:
            return
        self._tried = True
        try:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=-1,  # CPU
            )
            self.available = True
        except Exception:
            self._pipe = None
            self.available = False

    def _frame_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        """Карта глубины кадра с кэшем: N дефектов = 1 инференс, не N.

        (До фикса 2026-06-12 полный инференс модели гонялся на КАЖДЫЙ дефект.)
        """
        import hashlib

        key = (image_bgr.shape, hashlib.md5(image_bgr.tobytes()).hexdigest())
        if key != self._frame_key:
            import cv2
            from PIL import Image

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            depth = self._pipe(Image.fromarray(rgb))["depth"]
            self._depth_map = np.asarray(depth, dtype=np.float32)
            self._frame_key = key
        return self._depth_map

    def relative_bucket(self, image_bgr: np.ndarray, mask: np.ndarray) -> dict:
        """Относительная глубина дефекта vs окружающее покрытие.

        Возвращает {'depth_cm': None, 'depth_bucket': str|None, 'certifiable': False}.
        """
        self._load()
        result = {"depth_cm": None, "depth_bucket": None,
                  "depth_certifiable": False, "method": None}
        if not self.available or not np.asarray(mask).any():
            result["method"] = "unavailable"
            return result

        import cv2

        depth = self._frame_depth(image_bgr)
        if depth.shape != mask.shape:
            depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]))

        m = np.asarray(mask) > 0
        # «кольцо» вокруг дефекта = опорная поверхность
        ring = cv2.dilate(m.astype(np.uint8), np.ones((25, 25), np.uint8), 1).astype(bool)
        ring &= ~m
        if not ring.any():
            result["method"] = "no_ring"
            return result

        # Depth Anything: больше значение = ближе. Углубление → дальше → меньше.
        rel = float(np.median(depth[ring]) - np.median(depth[m]))
        rel_norm = rel / (float(np.ptp(depth)) + 1e-6)  # доля диапазона

        lo, hi = 0.02, 0.06  # эмпирические границы относительной просадки
        bucket = "shallow" if rel_norm < lo else "deep" if rel_norm > hi else "medium"
        result.update(depth_bucket=bucket, method="depth_anything_v2_small_relative",
                      _rel_norm=round(rel_norm, 4))
        return result
