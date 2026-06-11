"""Оркестратор CV-движка: фото -> DefectReport (JSON + overlay).

Склеивает detect → segment → shape → scale → depth → severity → report.
Модели создаются лениво и переиспользуются между кадрами.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config, imgio, scale as scale_mod, shape as shape_mod, severity as severity_mod
from . import report as report_mod
from .detect import Detector
from .segment import Segmenter
from .depth import RelativeDepth


class DefectPipeline:
    def __init__(self, cfg: config.InferenceConfig = config.DEFAULT_INFERENCE,
                 use_depth: bool = False, road_category: str = "IV"):
        self.cfg = cfg
        self.use_depth = use_depth
        self.road_category = road_category
        self.detector = Detector(cfg)
        self.segmenter = Segmenter()
        self.depth = RelativeDepth() if use_depth else None

    # --- одиночное фото ----------------------------------------------------
    def analyze_image(self, image_path: str | Path) -> tuple[dict, np.ndarray, list]:
        image_path = Path(image_path)
        img = imgio.read_image(image_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
        H, W = img.shape[:2]
        warnings: list[str] = []

        # 1) масштаб по эталону (люк); геометрия для overlay — из того же замера
        ref = scale_mod.scale_from_manhole(img, cfg=self.cfg)
        if not ref.available:
            warnings.append(ref.note or
                            "Эталон (люк) в кадре не найден — метрические размеры недоступны.")
        mm_per_px = ref.mm_per_px if ref.available else None
        mode = "single_with_reference" if ref.available else "single"

        # 2) детекция
        detections = self.detector.detect(img)
        if self.detector.is_fallback:
            warnings.append("Используется COCO-фолбэк детектора (нет весов под дефекты) — "
                            "классы не дорожные; это лишь проверка конвейера.")
        if not detections:
            warnings.append("Дефекты не обнаружены.")

        defects, masks = [], []
        for i, det in enumerate(detections, start=1):
            # 3) маска формы
            mask = det.mask if det.mask is not None else \
                self.segmenter.segment(img, det.bbox_xywh)
            if mask is None or np.asarray(mask).sum() < self.cfg.mask_min_area_px:
                continue
            sd = shape_mod.describe_mask(mask)
            if sd is None:
                continue

            # 4) метрика (если есть масштаб); полный набор ключей контракта §8
            metric = {"available": False,
                      "equivalent_diameter_cm": None, "length_cm": None,
                      "width_cm": None, "area_cm2": None, "area_m2": None,
                      "depth_cm": None, "depth_bucket": None,
                      "depth_certifiable": False,
                      "confidence": None, "error_band_pct": None}
            length_cm = area_m2 = None
            if mm_per_px is not None:
                scaled = shape_mod.apply_scale(sd, mm_per_px)
                length_cm = scaled["length_cm"]
                area_m2 = scaled["area_m2"]
                metric.update(available=True, confidence=ref.confidence,
                              error_band_pct=ref.error_band_pct, **scaled)

            # 5) относительная глубина (опц.)
            if self.depth is not None:
                d = self.depth.relative_bucket(img, mask)
                metric["depth_bucket"] = d.get("depth_bucket")
                metric["depth_method"] = d.get("method")

            # 6) серьёзность по ГОСТ
            verdict = severity_mod.classify(
                length_cm=length_cm, area_m2=area_m2, depth_cm=None,
                road_category=self.road_category,
            )

            defects.append({
                "id": i,
                "class": det.cls_name,
                "raw_label": det.raw_label,
                "confidence": round(det.confidence, 3),
                "bbox_px": [round(v, 1) for v in det.bbox_xywh],
                "mask_rle": report_mod.mask_to_rle(mask),
                "shape": sd.to_dict(),
                "metric": metric,
                "severity": verdict.to_dict(),
            })
            masks.append(mask)

        report = report_mod.build_report(
            image_name=image_path.name, image_size_px=(W, H), mode=mode,
            reference=ref.to_dict(), defects=defects, warnings=warnings,
        )
        ref_label = (f"manhole ref (GOST {ref.known_mm:.0f}mm)"
                     if ref.available and ref.known_mm else "manhole ref")
        overlay = report_mod.draw_overlay(
            img, defects, masks,
            reference_circle=ref.circle_px if ref.available else None,
            reference_ellipse=ref.ellipse_px if ref.available else None,
            reference_label=ref_label,
        )
        return report, overlay, masks
