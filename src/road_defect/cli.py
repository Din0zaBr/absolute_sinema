"""CLI: папка с фото -> outputs/*.json + *_annotated.jpg.

Пример:
    python -m road_defect.cli --input . --output outputs --depth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, imgio
from . import report as report_mod
from .pipeline import DefectPipeline

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv=None) -> int:
    # На Windows перенаправленный stdout кодируется cp1251 — символы вне неё
    # (стрелки, галочки) роняют print. Заменяем некодируемое, а не падаем.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 — нестандартный поток (тесты, embed)
            pass
    ap = argparse.ArgumentParser(description="Анализ дорожных дефектов по фото.")
    ap.add_argument("--input", required=True, help="файл или папка с изображениями")
    ap.add_argument("--output", default=str(config.OUTPUTS_DIR), help="папка результатов")
    ap.add_argument("--depth", action="store_true", help="включить относительную глубину")
    ap.add_argument("--road-category", default="IV", help="категория дороги для сроков ГОСТ")
    ap.add_argument("--conf", type=float, default=config.DEFAULT_INFERENCE.det_conf)
    ap.add_argument("--blob-ref", action="store_true",
                    help="разрешить эталон по тёмному эллипсу при косом виде "
                         "(экспериментально: риск ложного масштаба, см. STATUS)")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        print(f"Путь не найден: {inp}", file=sys.stderr)
        return 1
    if inp.is_file():
        if inp.suffix.lower() not in IMG_EXT:
            print(f"Не изображение ({', '.join(sorted(IMG_EXT))}): {inp}", file=sys.stderr)
            return 1
        images = [inp]
    else:
        images = sorted(p for p in inp.iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        print(f"Нет изображений в {inp}", file=sys.stderr)
        return 1

    cfg = config.InferenceConfig(det_conf=args.conf,
                                 allow_blob_reference=args.blob_ref)
    pipe = DefectPipeline(cfg=cfg, use_depth=args.depth, road_category=args.road_category)

    print(f"Обработка {len(images)} изображений -> {out}")
    ok_count = 0
    used_stems: set = set()
    for img_path in images:
        try:
            report, overlay, _ = pipe.analyze_image(img_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [ОШИБКА] {img_path.name}: {e}", file=sys.stderr)
            continue
        stem = report_mod.unique_stem(img_path.stem, used_stems)
        report_mod.save_report(report, out, stem)
        if not imgio.write_image(out / f"{stem}_annotated.jpg", overlay):
            print(f"  [ОШИБКА] не удалось записать {stem}_annotated.jpg", file=sys.stderr)
        n = len(report["defects"])
        ref = "эталон✓" if report["scale"]["available"] else "без эталона"
        print(f"  {img_path.name[:24]:24}  дефектов: {n:2}  {ref}  режим: {report['mode']}")
        ok_count += 1
    print("Готово." if ok_count else "Ни одно изображение не обработано.")
    return 0 if ok_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
