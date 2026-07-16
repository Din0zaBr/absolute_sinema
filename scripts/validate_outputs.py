"""Проверка инвариантов JSON-отчётов (контракт дизайна §8 + honest-mode).

Запуск:  python scripts/validate_outputs.py [папка_с_json]   (по умолчанию outputs/)

Для каждого *.json проверяет:
- структуру блока scale (вложенный reference, homography_applied);
- честность глубины: depth_cm == null и depth_certifiable == false всегда
  (см-глубина допустима только в Сценарии C, которого ещё нет); см-ОЦЕНКА
  живёт только в metric.depth_cm_estimate: certifiable=false, confidence
  не выше 'low', числа только при available=true и с полосой вокруг точки;
- mask_rle.size == [H, W] кадра и сумма серий RLE == H*W;
- bbox в границах кадра, confidence в [0, 1];
- согласованность metric.available с наличием размеров в см.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_report(path: Path, rep: dict) -> list[str]:
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(f"{path.name}: {msg}")

    for key in ("image", "mode", "image_size_px", "scale", "defects",
                "location", "warnings", "engine_version"):
        if key not in rep:
            err(f"нет ключа верхнего уровня '{key}'")
    if errors:
        return errors

    W, H = rep["image_size_px"]
    scale = rep["scale"]
    if not isinstance(scale.get("available"), bool):
        err("scale.available не bool")
    if not isinstance(scale.get("homography_applied"), bool):
        err("scale.homography_applied не bool")
    if "reference" not in scale:
        err("scale без вложенного блока reference (контракт §8)")
    elif scale["available"]:
        ref = scale["reference"]
        if not ref.get("type") or not ref.get("known_mm"):
            err("scale.available, но reference.type/known_mm пусты")

    for d in rep["defects"]:
        tag = f"defect id={d.get('id')}"
        m = d.get("metric", {})
        if m.get("depth_cm") is not None:
            err(f"{tag}: depth_cm != null — нарушение honest-mode")
        if m.get("depth_certifiable") is not False:
            err(f"{tag}: depth_certifiable != false — нарушение honest-mode")
        est = m.get("depth_cm_estimate")
        if est is not None and not isinstance(est, dict):
            err(f"{tag}: depth_cm_estimate не dict/null: {est!r}")
        elif isinstance(est, dict):
            def _num(v):
                return (isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v))
            if est.get("certifiable") is not False:
                err(f"{tag}: depth_cm_estimate.certifiable != false — "
                    "оценка выдаёт себя за измерение (honest-mode)")
            if est.get("available"):
                # available без явной пометки 'low' — дыра (ревью 2026-07-16)
                if est.get("confidence") != "low":
                    err(f"{tag}: depth_cm_estimate.available требует "
                        f"confidence='low', получено {est.get('confidence')!r}")
            elif est.get("confidence") not in (None, "low"):
                err(f"{tag}: depth_cm_estimate.confidence "
                    f"{est.get('confidence')!r} — оценка не бывает увереннее 'low'")
            nums = ("point_cm", "low_cm", "high_cm", "upper_bound_cm")
            if est.get("available"):
                if not (_num(est.get("point_cm")) or _num(est.get("upper_bound_cm"))):
                    err(f"{tag}: оценка available, но нет ни point_cm, ни "
                        "upper_bound_cm")
                if _num(est.get("point_cm")) and not (
                        _num(est.get("low_cm")) and _num(est.get("high_cm"))
                        and est["low_cm"] <= est["point_cm"] <= est["high_cm"]):
                    err(f"{tag}: полоса оценки не накрывает точку: "
                        f"{est.get('low_cm')}..{est.get('high_cm')} vs "
                        f"{est.get('point_cm')}")
            elif any(est.get(k) is not None for k in nums):
                err(f"{tag}: числа оценки при available=false — "
                    "откуда сантиметры?")
        if m.get("available") and m.get("equivalent_diameter_cm") is None:
            err(f"{tag}: metric.available, но equivalent_diameter_cm = null")
        if not m.get("available"):
            # Ревью 2026-07-15: проверка только eqd пропускала length/width/area
            # с сантиметрами при available=false — теперь все см-поля.
            for key in ("equivalent_diameter_cm", "length_cm", "width_cm",
                        "area_cm2", "area_m2"):
                if m.get(key) is not None:
                    err(f"{tag}: {key} без metric.available — откуда масштаб?")

        # Масштаб от высоты камеры (plane_scale, 2026-07-16) — оценочный
        # источник: обязан носить low-confidence, полосу и certifiable=false.
        # 'mixed' — pair-слияние эталонного и camera-вида (fusion.py); ключ
        # scale_source присутствует во всех НОВЫХ metric (одиночных и fused) —
        # его отсутствие в старых отчётах легально, наличие обязывает.
        src = m.get("scale_source")
        cam = m.get("camera_scale")
        if "scale_source" in m and m.get("available") and src not in (
                "reference", "camera_height", "mixed"):
            err(f"{tag}: metric.available без источника масштаба "
                f"(scale_source={src!r}) — откуда сантиметры?")
        if cam is not None and src not in ("camera_height", "mixed"):
            err(f"{tag}: camera_scale без scale_source camera_height/mixed")
        if src in ("camera_height", "mixed"):
            if m.get("confidence") != "low":
                err(f"{tag}: масштаб от высоты камеры не бывает увереннее "
                    f"'low', получено {m.get('confidence')!r} (honest-mode)")
            band = m.get("error_band_pct")
            if (not isinstance(band, (int, float)) or isinstance(band, bool)
                    or not math.isfinite(band) or band <= 0):
                err(f"{tag}: масштаб от высоты камеры без полосы ошибки: "
                    f"{band!r}")
            if not isinstance(cam, dict):
                err(f"{tag}: scale_source={src!r} без блока camera_scale")
            else:
                if cam.get("certifiable") is not False:
                    err(f"{tag}: camera_scale.certifiable != false — оценка "
                        "выдаёт себя за измерение (honest-mode)")
                if cam.get("horizon_method") not in ("sky_anchor",
                                                     "prior_pitch"):
                    err(f"{tag}: camera_scale.horizon_method неизвестен: "
                        f"{cam.get('horizon_method')!r}")
                for kreq in ("mm_per_px_transverse", "f_source"):
                    if not cam.get(kreq):
                        err(f"{tag}: camera_scale без {kreq}")

        conf = d.get("confidence")
        if (not isinstance(conf, (int, float)) or isinstance(conf, bool)
                or not math.isfinite(conf) or not (0.0 <= conf <= 1.0)):
            # null/строка/NaN дают TypeError в прямом сравнении и маскировались бы
            # общим «не разобрался» — ловим явно (аудит 2026-07-07).
            err(f"{tag}: confidence отсутствует или вне [0, 1]: {conf!r}")
        # Битый bbox_px — диагностика, а не KeyError валидатора (ревью
        # 2026-07-02). bool — подкласс int, NaN — float и проходит любые
        # сравнения ложью: оба класса битости ловим явно (ревью 2026-07-03).
        bbox = d.get("bbox_px")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or not all(isinstance(v, (int, float))
                           and not isinstance(v, bool)
                           and math.isfinite(v) for v in bbox)):
            err(f"{tag}: bbox_px отсутствует или не 4 конечных числа: {bbox!r}")
        else:
            x, y, w, h = bbox
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1 or w <= 0 or h <= 0:
                err(f"{tag}: bbox {bbox} вне кадра {W}x{H}")

        rle = d.get("mask_rle", {})
        if rle.get("size") != [H, W]:
            err(f"{tag}: mask_rle.size {rle.get('size')} != [{H}, {W}]")
        counts = rle.get("counts", [])
        if counts and sum(counts) != H * W:
            err(f"{tag}: сумма серий RLE {sum(counts)} != {H * W}")

        if "non_conforming" not in d.get("severity", {}):
            err(f"{tag}: severity без non_conforming")
    return errors


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs"
    reports = sorted(out_dir.glob("*.json"))
    if not reports:
        print(f"Нет *.json в {out_dir}")
        return 1
    all_errors: list[str] = []
    n_defects = 0
    for p in reports:
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
            all_errors.extend(check_report(p, rep))
            n_defects += len(rep["defects"])
        except Exception as e:  # noqa: BLE001
            all_errors.append(f"{p.name}: не разобрался ({e})")
    print(f"Проверено отчётов: {len(reports)}, дефектов: {n_defects}")
    if all_errors:
        print("НАРУШЕНИЯ:")
        for e in all_errors:
            print(f"  - {e}")
        return 1
    print("Все инварианты соблюдены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
