"""Смок-тест CV-движка на демо-фото с подробной диагностикой.

Запуск:  PYTHONPATH=src python scripts/smoke_test.py
Печатает версии окружения, какой детектор/сегментатор загрузился, и прогоняет
конвейер по всем фото в корне проекта, сохраняя outputs/*.json и *_annotated.jpg.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main() -> int:
    # На Windows перенаправленный stdout кодируется cp1251 — символы вне неё
    # (✓, └) роняют print. Заменяем некодируемое, а не падаем (как в cli.py).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 — нестандартный поток (тесты, embed)
            pass
    print("=" * 70)
    print("СМОК-ТЕСТ: CV-движок дорожных дефектов")
    print("=" * 70)

    # 1) окружение
    try:
        import torch, cv2, numpy, skimage
        import ultralytics
        print(f"torch        {torch.__version__}")
        print(f"opencv       {cv2.__version__}")
        print(f"ultralytics  {ultralytics.__version__}")
        print(f"numpy        {numpy.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] нет зависимостей: {e}")
        return 1

    from road_defect import config, imgio
    from road_defect.pipeline import DefectPipeline
    from road_defect import report as report_mod

    images = sorted(p for p in ROOT.iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"\nНайдено изображений: {len(images)}")
    if not images:
        print("[!] Положите фото в корень проекта.")
        return 1

    # 2) загрузка моделей (с диагностикой)
    pipe = DefectPipeline(use_depth=False)
    try:
        pipe.detector.load()
        a = pipe.detector.active
        tag = " (ФОЛБЭК COCO — не детектит ямы!)" if pipe.detector.is_fallback else ""
        print(f"Детектор: {a.name}{tag}")
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] детектор не загрузился: {e}")
        traceback.print_exc()
        return 1

    out = ROOT / "outputs"
    print("\n" + "-" * 70)
    total = 0
    used_stems: set = set()
    for img_path in images:
        try:
            rep, overlay, _ = pipe.analyze_image(img_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [ОШИБКА] {img_path.name[:20]}: {e}")
            traceback.print_exc()
            continue
        stem = report_mod.unique_stem(img_path.stem[:24], used_stems)
        report_mod.save_report(rep, out, stem)
        if not imgio.write_image(out / f"{stem}_annotated.jpg", overlay):
            print(f"  [ОШИБКА] не удалось записать {stem}_annotated.jpg")
        n = len(rep["defects"])
        total += n
        ref = "ЛЮК✓" if rep["scale"]["available"] else "без эталона"
        classes = ", ".join(sorted({d["class"] for d in rep["defects"]})) or "—"
        print(f"\n  {img_path.name[:18]:18} | дефектов: {n:2} | {ref:11} | {rep['mode']:20} | {classes}")
        for d in rep["defects"][:4]:
            m = d["metric"]
            size = (f"~{m['equivalent_diameter_cm']}см"
                    if m.get("available") and m.get("equivalent_diameter_cm") is not None
                    else f"{int(d['shape']['equivalent_diameter_px'])}px")
            print(f"       └ {d['class']:18} conf={d['confidence']:.2f} d={size} "
                  f"ecc={d['shape']['eccentricity']:.2f}")

    print("-" * 70)
    # backend известен только после ленивой загрузки (первый segment()) —
    # печатаем один раз после прогона, а не 'none' до него.
    print(f"Сегментатор backend: {pipe.segmenter.backend}")
    print(f"ИТОГО дефектов: {total}. Результаты: {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
