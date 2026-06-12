"""Сегментация формы дефекта по bbox-промпту (MobileSAM) + классические фолбэки.

Если у детектора уже есть маска (seg-модель) — используем её. Иначе MobileSAM по
боксу: сначала официальный пакет mobile_sam, при его отсутствии — та же модель
через встроенную реализацию ultralytics.models.sam (новых пакетов не требует).
Если недоступно и это — GrabCut/порог внутри bbox как грубый фолбэк, чтобы
конвейер всегда давал контур.
"""
from __future__ import annotations

import numpy as np

from . import config


class Segmenter:
    def __init__(self):
        self._predictor = None
        self._tried = False
        self.backend = "none"
        self._frame_key = None  # отпечаток кадра, чей эмбеддинг закэширован

    def _try_load(self):
        if self._tried:
            return
        self._tried = True
        if self._try_load_mobile_sam_pkg():
            return
        if self._try_load_ultralytics_sam():
            return
        self.backend = "grabcut_fallback"

    def _try_load_mobile_sam_pkg(self) -> bool:
        """Официальный пакет mobile_sam (если установлен) + веса с HF."""
        try:
            from huggingface_hub import hf_hub_download
            from mobile_sam import sam_model_registry, SamPredictor

            ckpt = hf_hub_download(repo_id=config.SEGMENTER.repo_id,
                                   filename=config.SEGMENTER.filename,
                                   local_dir=str(config.MODELS_DIR))
            sam = sam_model_registry["vit_t"](checkpoint=ckpt)
            sam.to("cpu").eval()
            self._predictor = SamPredictor(sam)
            self.backend = "mobile_sam"
            return True
        except Exception:
            self._predictor = None
            return False

    def _try_load_ultralytics_sam(self) -> bool:
        """MobileSAM через ultralytics.models.sam (веса докачиваются с CDN)."""
        try:
            from ultralytics.models.sam import Predictor as SAMPredictor

            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            ckpt = config.MODELS_DIR / config.SEGMENTER_ULTRALYTICS.filename
            predictor = SAMPredictor(overrides=dict(
                task="segment", mode="predict", imgsz=1024,
                model=str(ckpt), conf=0.25, verbose=False,
                save=False,  # иначе ultralytics пишет свои overlay в runs/
            ))
            predictor.setup_model(verbose=False)  # скачает/загрузит веса сразу
            self._predictor = predictor
            self.backend = "mobile_sam_ultralytics"
            return True
        except Exception:
            self._predictor = None
            return False

    def _set_frame(self, image_bgr: np.ndarray) -> None:
        """Прогнать image encoder один раз на кадр (тяжёлая операция на CPU).

        Кадр опознаётся по форме + разреженной выборке пикселей: повторные
        вызовы для того же кадра (N дефектов) переиспользуют эмбеддинг.
        """
        key = (image_bgr.shape, image_bgr[::97, ::97].tobytes())
        if key == self._frame_key:
            return
        if self.backend == "mobile_sam":
            import cv2
            self._predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        else:  # ultralytics SAM ждёт BGR (как cv2.imread)
            self._predictor.set_image(image_bgr)
        self._frame_key = key

    def segment(self, image_bgr: np.ndarray, bbox_xywh) -> np.ndarray:
        """Вернуть бинарную маску дефекта в пределах кадра."""
        self._try_load()
        x, y, w, h = [int(round(v)) for v in bbox_xywh]

        if self.backend == "mobile_sam":
            self._set_frame(image_bgr)
            box = np.array([x, y, x + w, y + h])
            masks, scores, _ = self._predictor.predict(box=box, multimask_output=True)
            best = masks[int(np.argmax(scores))]
            return best.astype(bool)

        if self.backend == "mobile_sam_ultralytics":
            self._set_frame(image_bgr)
            res = self._predictor(bboxes=[[x, y, x + w, y + h]],
                                  multimask_output=True)[0]
            if res.masks is not None and len(res.masks.data):
                scores = res.boxes.conf.cpu().numpy()
                best = res.masks.data[int(np.argmax(scores))].cpu().numpy()
                mask = np.asarray(best) > 0.5
                # Маска обязана совпадать с кадром (как в detect.py) — иначе
                # кривые пиксели попадут в метрику ГОСТ; честнее GrabCut.
                if mask.shape == image_bgr.shape[:2]:
                    return mask
            return self._grabcut(image_bgr, (x, y, w, h))

        return self._grabcut(image_bgr, (x, y, w, h))

    @staticmethod
    def _grabcut(image_bgr: np.ndarray, rect) -> np.ndarray:
        """Грубый фолбэк: GrabCut в пределах bbox."""
        import cv2

        x, y, w, h = rect
        H, W = image_bgr.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        mask = np.zeros((H, W), np.uint8)
        if w < 5 or h < 5:
            return mask.astype(bool)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image_bgr, mask, (x, y, w, h), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0)
            return out.astype(bool)
        except Exception:
            box = np.zeros((H, W), bool)
            box[y:y + h, x:x + w] = True
            return box
