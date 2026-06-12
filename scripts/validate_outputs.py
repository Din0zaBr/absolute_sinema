"""Проверка инвариантов JSON-отчётов (контракт дизайна §8 + honest-mode).

Запуск:  python scripts/validate_outputs.py [папка_с_json]   (по умолчанию outputs/)

Для каждого *.json проверяет:
- структуру блока scale (вложенный reference, homography_applied);
- честность глубины: depth_cm == null и depth_certifiable == false всегда
  (см-глубина допустима только в Сценарии C, которого ещё нет);
- mask_rle.size == [H, W] кадра и сумма серий RLE == H*W;
- bbox в границах кадра, confidence в [0, 1];
- согласованность metric.available с наличием размеров в см.
"""
from __future__ import annotations

import json
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
        if m.get("available") and m.get("equivalent_diameter_cm") is None:
            err(f"{tag}: metric.available, но equivalent_diameter_cm = null")
        if not m.get("available") and m.get("equivalent_diameter_cm") is not None:
            err(f"{tag}: размеры в см без metric.available — откуда масштаб?")

        if not (0.0 <= d.get("confidence", -1) <= 1.0):
            err(f"{tag}: confidence вне [0, 1]")
        x, y, w, h = d.get("bbox_px", (0, 0, 0, 0))
        if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1 or w <= 0 or h <= 0:
            err(f"{tag}: bbox {d['bbox_px']} вне кадра {W}x{H}")

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
