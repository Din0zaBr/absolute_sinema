"""Детектор дорожных дефектов (готовые веса YOLO).

Грузит первый загрузившийся кандидат из реестра (config.DETECTOR_CANDIDATES).
Если доступен только COCO-фолбэк — честно проставляет `is_fallback=True`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config


@dataclass(eq=False)
class Detection:
    # eq=False: сравнение по идентичности. Автогенерённый __eq__ сравнивал бы поле
    # mask (np.ndarray) и падал бы «ambiguous truth value» при ==/in/hashing разных
    # экземпляров с масками. Нигде не сравниваем по значению (аудит 2026-07-07).
    cls_name: str                 # нормализованный класс (наш словарь) или сырое имя
    raw_label: str                # как назвала модель
    confidence: float
    bbox_xywh: tuple              # (x, y, w, h) в пикселях
    mask: np.ndarray | None = None  # маска из seg-модели, если есть


def _acquire_weights(cand: config.ModelCandidate) -> str | None:
    """Скачать/найти веса кандидата. Возвращает локальный путь или None."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if cand.source == "ultralytics":
        # ultralytics сам докачает по имени с CDN
        return cand.filename
    if cand.source == "local":
        p = config.MODELS_DIR / cand.filename
        return str(p) if p.exists() else None
    if cand.source == "hf_hub":
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=cand.repo_id, filename=cand.filename,
                                   local_dir=str(config.MODELS_DIR))
        except Exception as e:  # noqa: BLE001
            # Не молчим: без лога нельзя отличить отсутствие пакета/сети от опечатки
            # в repo_id/filename или гейтинга токена — иначе деградацию до COCO
            # не диагностировать (аудит 2026-07-07).
            logging.getLogger(__name__).warning(
                "Не удалось получить веса %s (%s/%s): %s",
                cand.name, cand.repo_id, cand.filename, e)
            return None
    return None


def _iou_xywh(a: tuple, b: tuple) -> float:
    """IoU двух bbox в формате (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _containment(a: tuple, b: tuple) -> float:
    """Доля МЕНЬШЕГО из двух боксов, попавшая в пересечение (0..1).

    Высокая доля = один бокс почти целиком внутри другого (seg-бокс внутри
    обычного — один объект), а НЕ два соседних дефекта, у которых лишь центр
    случайно попал в чужой бокс (тогда доля мала). Оба бокса — (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller > 0 else 0.0


# Ярлыки «ямы» в нижнем регистре: канон + коды таксономии (D40). Дрейф
# регистра/пробелы в весах ('d40', 'D40 ', 'Pothole') не должны опустошать
# проход (ревью 2026-07-07: get() по DEFECT_CLASSES был регистрозависим).
_POTHOLE_RAW_LABELS = {"pothole"} | {
    k.lower() for k, v in config.DEFECT_CLASSES.items() if v == "pothole"}


def _is_pothole_label(raw: str) -> bool:
    """Ярлык модели -> «это класс ямы?»: таксономия, без регистра и пробелов.

    Второй проход ансамбля фильтруется этим предикатом: жёсткое сравнение
    строки молча опустошало бы проход при дрейфе ярлыка весов — ensemble_active
    оставался True при вечных 0 детекций (аудит 2026-07-07, ловушка №19).
    """
    return str(raw).strip().lower() in _POTHOLE_RAW_LABELS


def _second_pass_potholes(dets: list) -> list:
    """Оставить из второго прохода только ямы, канонизировав cls_name:
    нестандартный регистр ярлыка не должен просачиваться в AREAL_/JSON."""
    out = []
    for d in dets:
        if _is_pothole_label(d.raw_label):
            d.cls_name = "pothole"
            out.append(d)
    return out


def merge_detections(primary: list, secondary: list,
                     iou_thr: float = 0.35) -> list:
    """Слить детекции второго прохода в основной список.

    Вторичная детекция добавляется, только если для неё нет первичной ТОГО ЖЕ
    класса, которая (а) пересекается по IoU ≥ порога ЛИБО (б) почти целиком
    содержит её (containment ≥ 0.8). Порог IoU 0.35 — ниже NMS (0.45): боксы двух
    разных семейств моделей на одной яме дают умеренный IoU. Containment по ПЛОЩАДИ
    (а не «центр в боксе») ловит «seg-бокс внутри обычного», но НЕ склеивает два
    соседних дефекта, у которых лишь центр попал в чужой бокс (аудит 2026-07-07,
    уточнено по верификации). Основной детектор — авторитет там, где оба видят дефект.
    """
    merged = list(primary)
    for det in secondary:
        if any(d.cls_name == det.cls_name
               and (_iou_xywh(d.bbox_xywh, det.bbox_xywh) >= iou_thr
                    or _containment(d.bbox_xywh, det.bbox_xywh) >= 0.8)
               for d in primary):
            continue
        merged.append(det)
    return merged


class Detector:
    """Обёртка над YOLO с авто-выбором рабочих весов.

    cfg.ensemble_pothole=True добавляет второй проход одноклассовым
    pothole-seg детектором (keremberke): rezzzq слеп к нетипичным ямам
    (засыпанная яма — 0 детекций при conf=0.01, разбор 2026-06-12), а
    keremberke видел её с conf=0.81. Слияние — merge_detections.
    """

    def __init__(self, cfg: config.InferenceConfig = config.DEFAULT_INFERENCE,
                 candidates: list[config.ModelCandidate] | None = None):
        self.cfg = cfg
        self._model = None
        self._pothole_model = None      # второй проход (ансамбль)
        self._tried_second_pass = False
        self.ensemble_active = False
        self.ensemble_failed = False    # запрошен, но не поднялся (для warnings)
        self.active: config.ModelCandidate | None = None
        self.is_fallback = False
        self._candidates = candidates or config.DETECTOR_CANDIDATES

    def load(self) -> "Detector":
        from ultralytics import YOLO

        last_err = None
        for cand in self._candidates:
            weights = _acquire_weights(cand)
            if weights is None:
                continue
            try:
                self._model = YOLO(weights)
                self.active = cand
                self.is_fallback = cand.name.endswith("coco")
                return self
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError(f"Не удалось загрузить ни один детектор. Последняя ошибка: {last_err}")

    def _load_pothole_second_pass(self) -> None:
        """Лениво поднять второй pothole-детектор для ансамбля.

        Если запрошенный ансамбль поднять не удалось — это НЕ молчаливый
        случай: self.ensemble_failed читается конвейером и попадает в warnings
        отчёта (пользователь явно просил два прохода).
        """
        if self._tried_second_pass or not self.cfg.ensemble_pothole:
            return
        self._tried_second_pass = True
        if self.active and self.active.name == "keremberke-yolov8m-pothole-seg":
            return  # основной уже keremberke — второй проход избыточен, не отказ
        from ultralytics import YOLO

        cand = next((c for c in self._candidates
                     if c.name == "keremberke-yolov8m-pothole-seg"), None)
        weights = _acquire_weights(cand) if cand else None
        if weights is None:
            self.ensemble_failed = True
            return
        try:
            model = YOLO(weights)
        except Exception:
            self._pothole_model = None
            self.ensemble_failed = True
            return
        # Смок-ассерт ярлыка (ловушка №19): веса без класса ямы дали бы
        # ensemble_active=True при гарантированных 0 детекций — молчаливо
        # пустой второй проход. Честный отказ попадает в warnings отчёта.
        names = getattr(model, "names", None) or {}
        labels = [str(v) for v in
                  (names.values() if isinstance(names, dict) else names)]
        if labels and not any(_is_pothole_label(lab) for lab in labels):
            logging.getLogger(__name__).warning(
                "Второй pothole-проход: среди классов весов %s нет ямы %s — "
                "ансамбль не активирован.", cand.name, labels)
            self._pothole_model = None
            self.ensemble_failed = True
            return
        self._pothole_model = model
        self.ensemble_active = True

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        if self._model is None:
            self.load()
        out = self._predict(self._model, image_bgr)
        if self.cfg.ensemble_pothole:
            self._load_pothole_second_pass()
            if self._pothole_model is not None:
                extra = _second_pass_potholes(
                    self._predict(self._pothole_model, image_bgr))
                out = merge_detections(out, extra,
                                       iou_thr=self.cfg.ensemble_merge_iou)
        return out

    def _predict(self, model, image_bgr: np.ndarray) -> list[Detection]:
        results = model.predict(
            image_bgr, conf=self.cfg.det_conf, iou=self.cfg.det_iou,
            imgsz=self.cfg.imgsz, retina_masks=True, verbose=False,
        )
        out: list[Detection] = []
        for res in results:
            names = res.names
            boxes = res.boxes
            masks = res.masks
            if boxes is None:
                continue
            for i in range(len(boxes)):
                raw = names[int(boxes.cls[i])]
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                mask = None
                if masks is not None and masks.data is not None:
                    m = masks.data[i].cpu().numpy()
                    mask = (m > 0.5)
                    # Маска обязана быть в координатах оригинала (retina_masks).
                    # Если форма не совпала — отбрасываем: ниже её досчитает
                    # Segmenter по bbox, а кривые letterbox-пиксели в метрику
                    # ГОСТ попасть не должны.
                    if mask.shape != image_bgr.shape[:2]:
                        mask = None
                out.append(Detection(
                    cls_name=config.DEFECT_CLASSES.get(raw, raw),
                    raw_label=raw,
                    confidence=float(boxes.conf[i]),
                    bbox_xywh=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    mask=mask,
                ))
        return out
