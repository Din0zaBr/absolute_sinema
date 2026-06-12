"""Сегментация формы дефекта: класс-зависимая стратегия + фолбэки.

Если у детектора уже есть маска (seg-модель) — используем её. Иначе:
  • линейные трещины (config.LINEAR_DEFECT_CLASSES) — классическая экстракция
    тонких тёмных структур (black-hat) в ПОЛНОМ разрешении внутри bbox. SAM здесь
    не годится: он инференсится на 1024px, и трещина 1–3 px на 12-МП кадре после
    даунскейла субпиксельна — заливает bbox вместо линии (верификация 2026-06-12);
  • площадные дефекты (яма/заплатка/сетка) — MobileSAM по боксу: сначала
    официальный пакет mobile_sam, при его отсутствии — та же модель через
    встроенную реализацию ultralytics.models.sam (новых пакетов не требует);
  • последний фолбэк всегда — GrabCut внутри bbox, чтобы конвейер дал контур.

Какой путь дал маску — в self.last_method ('crack_blackhat' | 'mobile_sam' |
'mobile_sam_ultralytics' | 'grabcut'); pipeline пишет его в JSON (mask_method).
"""
from __future__ import annotations

import numpy as np

from . import config


class Segmenter:
    def __init__(self):
        self._predictor = None
        self._tried = False
        self.backend = "none"
        self.last_method = None  # чем получена маска последнего segment()
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

    def segment(self, image_bgr: np.ndarray, bbox_xywh,
                cls_name: str | None = None,
                min_area: int = config.DEFAULT_INFERENCE.mask_min_area_linear_px
                ) -> np.ndarray:
        """Вернуть бинарную маску дефекта в пределах кадра.

        min_area — порог отбраковки конвейера: если крэк-экстрактор дал меньше,
        честнее уйти в SAM/GrabCut (маска грубее, но уверенная детекция не
        пропадёт из отчёта целиком).
        """
        x, y, w, h = [int(round(v)) for v in bbox_xywh]

        if cls_name in config.LINEAR_DEFECT_CLASSES:
            mask, method = self._crack_mask(image_bgr, (x, y, w, h))
            if mask is not None and int(mask.sum()) >= min_area:
                self.last_method = method
                return mask
            # неправдоподобно (тень/однородный асфальт/крохи) — обычная цепочка

        self._try_load()
        if self.backend == "mobile_sam":
            self._set_frame(image_bgr)
            box = np.array([x, y, x + w, y + h])
            masks, scores, _ = self._predictor.predict(box=box, multimask_output=True)
            best = masks[int(np.argmax(scores))]
            self.last_method = "mobile_sam"
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
                    self.last_method = "mobile_sam_ultralytics"
                    return mask
            self.last_method = "grabcut"
            return self._grabcut(image_bgr, (x, y, w, h))

        self.last_method = "grabcut"
        return self._grabcut(image_bgr, (x, y, w, h))

    @staticmethod
    def _crack_mask(image_bgr: np.ndarray, rect) -> tuple:
        """Тонкие структуры (трещины) в полном разрешении внутри bbox.

        Сначала black-hat (тёмная трещина на светлом асфальте — типовой
        случай), затем top-hat (трещина, забитая светлой пылью, — светлее
        окружения; такие встречаются на реальных фото). Морфология с ядром
        шире трещины даёт отклик ровно на тонких линиях; широкие тени и
        однородный асфальт отклика почти не дают.

        Возвращает (mask, 'crack_blackhat'|'crack_tophat') или (None, None),
        если оба результата неправдоподобны — тогда вызывающий код идёт по
        обычной цепочке SAM/GrabCut.
        """
        import cv2

        x, y, w, h = rect
        H, W = image_bgr.shape[:2]
        pad_x, pad_y = max(2, w // 10), max(2, h // 10)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(W, x + w + pad_x), min(H, y + h + pad_y)
        if x1 - x0 < 5 or y1 - y0 < 5:
            return None, None
        gray = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

        for op, method in ((cv2.MORPH_BLACKHAT, "crack_blackhat"),
                           (cv2.MORPH_TOPHAT, "crack_tophat")):
            keep = Segmenter._thin_structures(gray, op)
            if keep is not None:
                mask = np.zeros((H, W), bool)
                mask[y0:y1, x0:x1] = keep
                return mask, method
        return None, None

    @staticmethod
    def _thin_structures(gray: np.ndarray, morph_op: int):
        """Маска тонких тёмных (BLACKHAT) или светлых (TOPHAT) структур в кропе.

        None — если результат неправдоподобен: нет отклика, пусто после
        фильтров или заливка >45% области.
        """
        import cv2

        ch, cw = gray.shape
        # Ядро шире типичной трещины (кап 51 px); прямоугольное — сепарабельно
        # и быстро даже на больших кропах.
        k = int(min(51, max(15, min(cw, ch) // 8)))
        k += 1 - k % 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        response = cv2.morphologyEx(gray, morph_op, kernel)
        if int(response.max()) < 10:
            return None  # нет выраженной тонкой структуры
        _, binm = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Фильтр компонент: мусор по площади (< ~0.05% области) и по форме —
        # трещина вытянута, а пятнистые тени деревьев и текстурные кляксы
        # округлы (ловили blackhat на реальных фото, верификация 2026-06-12).
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        min_area = max(30, cw * ch // 2000)
        keep = np.zeros(binm.shape, bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            ys, xs = np.nonzero(labels == i)
            cov = np.cov(np.vstack([xs, ys]).astype(np.float64))
            evals = np.linalg.eigvalsh(cov)  # по возрастанию
            elongation = float(np.sqrt(evals[1] / (evals[0] + 1e-6)))
            if elongation >= 2.0:
                keep[labels == i] = True
        if not keep.any() or keep.mean() > 0.45:
            return None
        return keep

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
