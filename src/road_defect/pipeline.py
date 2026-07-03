"""Оркестратор CV-движка: фото -> DefectReport (JSON + overlay).

Склеивает detect → segment → shape → scale → depth → severity → report.
Модели создаются лениво и переиспользуются между кадрами.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config, imgio, scale as scale_mod, shape as shape_mod, severity as severity_mod
from . import report as report_mod
from . import fusion as fusion_mod
from .detect import Detector
from .segment import Segmenter
from .depth import RelativeDepth


def _reference_label(ref) -> str:
    """Короткая подпись эталона для overlay (тип + размер + след кросс-проверки)."""
    if not ref.available:
        return "ref"
    short = {"manhole_gost3634_cover": "manhole", "marking": "marking",
             "curb_gost6665": "curb"}.get(ref.type or "", ref.type or "ref")
    km = f" {ref.known_mm:.0f}mm" if ref.known_mm else ""
    return f"{short}{km}{' +xcheck' if ref.cross_checked else ''}"


def _select_dominant_pothole(defects: list):
    """Одна доминирующая яма в кадре (наибольшая уверенность). Возвращает
    (defect|None, status). Неоднозначность (≥2 ямы в пределах 0.10 уверенности)
    или отсутствие → None: молчаливое сопоставление разных ям недопустимо."""
    pots = sorted((d for d in defects if d.get("class") == "pothole"),
                  key=lambda d: d.get("confidence", 0.0), reverse=True)
    if not pots:
        return None, "no_pothole"
    if len(pots) >= 2 and (pots[0]["confidence"] - pots[1]["confidence"]) < 0.10:
        return None, "ambiguous_multiple_potholes"
    return pots[0], "ok"


def _view_summary(name: str, rep: dict, defect, status: str) -> dict:
    m = (defect.get("metric", {}) if defect else {})
    return {
        "image": name,
        "pothole_found": defect is not None,
        "select_status": status,
        "scale_available": rep["scale"].get("available", False),
        "area_cm2": m.get("area_cm2"),
        "view_tilt_deg": m.get("view_tilt_deg"),
        "confidence": m.get("confidence"),
    }


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
        report, overlay, masks, _, _ = self._analyze(image_path)
        return report, overlay, masks

    def _analyze(self, image_path: str | Path):
        """Как analyze_image, но дополнительно возвращает исходный кадр и эталон
        (img, ref) — они нужны analyze_pair, чтобы перерисовать overlay после
        слияния (иначе на картинке остались бы до-слиянные сантиметры)."""
        image_path = Path(image_path)
        img = imgio.read_image(image_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
        H, W = img.shape[:2]
        warnings: list[str] = []

        # 1) детекция — первой: её bbox'ы исключаются из кандидатов в эталоны
        #    (круглая яма/тёмная заплатка не должна стать «люком» масштаба)
        detections = self.detector.detect(img)

        # 2) масштаб по эталону: мульти-эталон (люк/разметка/борт) с кросс-проверкой.
        #    Геометрия для overlay берётся из того же замера.
        ref = scale_mod.resolve_scale(
            img, cfg=self.cfg,
            exclude_boxes=[d.bbox_xywh for d in detections],
            road_category=self.road_category)
        if not ref.available:
            warnings.append(ref.note or
                            "Эталон масштаба в кадре не найден — метрические размеры недоступны.")
        elif "конфликт эталонов" in (ref.note or ""):
            warnings.append(ref.note)
        elif ref.cross_checked and ref.agreeing_types:
            warnings.append("Масштаб подтверждён разнотипным эталоном "
                            f"({', '.join(ref.agreeing_types)}) — уверенность повышена.")
        mm_per_px = ref.mm_per_px if ref.available else None
        mode = "single_with_reference" if ref.available else "single"

        if self.detector.is_fallback:
            warnings.append("Используется COCO-фолбэк детектора (нет весов под дефекты) — "
                            "классы не дорожные; это лишь проверка конвейера.")
        if getattr(self.detector, "ensemble_failed", False):
            warnings.append("Запрошен ансамбль (--ensemble), но второй pothole-детектор "
                            "не загрузился — отработал только основной проход.")
        if not detections:
            warnings.append("Дефекты не обнаружены.")

        defects, masks = [], []
        for i, det in enumerate(detections, start=1):
            # 3) маска формы
            # Линейные трещины легально тонкие — общий порог терял их целиком.
            min_area = (self.cfg.mask_min_area_linear_px
                        if det.cls_name in config.LINEAR_DEFECT_CLASSES
                        else self.cfg.mask_min_area_px)
            if det.mask is not None:
                mask, mask_method = det.mask, "detector_mask"
            else:
                mask = self.segmenter.segment(img, det.bbox_xywh,
                                              cls_name=det.cls_name,
                                              min_area=min_area)
                mask_method = self.segmenter.last_method
            if mask is None or np.asarray(mask).sum() < min_area:
                # Не молчим: уверенная детекция без валидной маски — это
                # информация для оператора, а не повод исчезнуть из отчёта.
                warnings.append(
                    f"Детекция {det.cls_name} (conf {det.confidence:.2f}) "
                    f"отброшена: маска меньше {min_area} px.")
                continue
            sd = shape_mod.describe_mask(mask)
            if sd is None:
                continue

            # 4) метрика (если есть масштаб); полный набор ключей контракта §8
            metric = {"available": False,
                      "equivalent_diameter_cm": None, "length_cm": None,
                      "width_cm": None, "area_cm2": None, "area_m2": None,
                      "depth_cm": None, "depth_bucket": None,
                      "depth_method": None,  # ключи стабильны и без --depth
                      "depth_certifiable": False,
                      "confidence": None, "error_band_pct": None,
                      # наклон вида (из эталона) — нужен слиянию двух видов (F1);
                      # стабильный ключ как depth_*: всегда присутствует
                      "view_tilt_deg": ref.tilt_deg if ref.available else None}
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
                "mask_method": mask_method,
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
        overlay = self._draw(img, defects, masks, ref)
        return report, overlay, masks, img, ref

    @staticmethod
    def _draw(img, defects, masks, ref):
        return report_mod.draw_overlay(
            img, defects, masks,
            reference_circle=ref.circle_px if ref.available else None,
            reference_ellipse=ref.ellipse_px if ref.available else None,
            reference_polylines=ref.polylines_px if ref.available else None,
            reference_label=_reference_label(ref),
        )

    # --- слияние двух видов одной ямы (F1) ---------------------------------
    def analyze_pair(self, image_path_a: str | Path,
                     image_path_b: str | Path):
        """Слить два вида ОДНОЙ ямы («яма впереди» + «яма позади») для более
        точной оценки размеров. Прогоняет analyze_image на каждом кадре,
        сопоставляет доминирующую яму и сливает уже посчитанные см (см. fusion.py).

        Возвращает (fused_report, overlay_a, overlay_b, masks_a). Носитель
        геометрии (image_size_px, mask_rle, scale) — вид A. Глубина в см не
        выдаётся никогда (два косых кадра ≠ Сценарий C).
        """
        rep_a, ov_a, masks_a, img_a, ref_a = self._analyze(image_path_a)
        rep_b, ov_b, _, _, _ = self._analyze(image_path_b)
        name_a, name_b = Path(image_path_a).name, Path(image_path_b).name

        da, sa = _select_dominant_pothole(rep_a["defects"])
        db, sb = _select_dominant_pothole(rep_b["defects"])

        warnings = [f"[вид A] {w}" for w in rep_a["warnings"]]
        warnings += [f"[вид B] {w}" for w in rep_b["warnings"]]

        defects = [dict(d) for d in rep_a["defects"]]  # геометрию несёт вид A
        matched = da is not None and db is not None
        fused = None
        if matched:
            vm_a = fusion_mod.ViewMeasurement.from_defect(da, name_a)
            vm_b = fusion_mod.ViewMeasurement.from_defect(db, name_b)
            fused = fusion_mod.fuse_pair(vm_a, vm_b, cfg=self.cfg)

        if fused is not None and fused.available:
            idx = rep_a["defects"].index(da)
            verdict = severity_mod.classify(
                length_cm=fused.length_cm, area_m2=fused.area_m2,
                depth_cm=None, road_category=self.road_category)
            defects[idx] = {**defects[idx], "metric": fused.to_dict(),
                            "severity": verdict.to_dict()}
            # Перерисовать overlay вида A с fused-метрикой: числа на картинке
            # обязаны совпадать с JSON под тем же стемом (до-слиянный размер
            # уже впечатан в пиксели старого overlay).
            ov_a = self._draw(img_a, defects, masks_a, ref_a)
            mode = "two_view_fused"
            warnings.append(
                f"Слияние двух видов ямы: согласие='{fused.cross_view.get('agreement')}', "
                f"расхождение {fused.cross_view.get('disagreement_pct')}%. {fused.note}")
        else:
            mode = "two_view_unmatched"
            if not matched:
                reason = "в одном из видов нет уверенной одиночной ямы для сопоставления"
            elif fused is not None and fused.note:
                reason = fused.note   # нет масштаба / разные ямы (форма) и т.п.
            else:
                reason = "слияние невозможно"
            warnings.append(f"Слияние двух видов НЕ выполнено: {reason}. "
                            "Показаны одиночные результаты вида A.")

        fusion_block = {
            "n_views": 2,
            "matched": matched,
            "fused": bool(fused is not None and fused.available),
            "agreement": (fused.cross_view.get("agreement")
                          if fused is not None and fused.available else None),
            "disagreement_pct": (fused.cross_view.get("disagreement_pct")
                                 if fused is not None and fused.available else None),
            "per_view": [_view_summary(name_a, rep_a, da, sa),
                         _view_summary(name_b, rep_b, db, sb)],
            "note": ("Глубина в см НЕ выдаётся: два косых кадра — это не Сценарий C "
                     "(SfM с известной базой). Только площадь скорректирована за наклон."),
        }

        report = report_mod.build_report(
            image_name=name_a, image_size_px=tuple(rep_a["image_size_px"]),
            mode=mode, reference=rep_a["scale"], defects=defects,
            warnings=warnings, fusion=fusion_block,
        )
        return report, ov_a, ov_b, masks_a
