"""CLI: папка с фото -> outputs/*.json + *_annotated.jpg.

Пример:
    python -m road_defect.cli --input . --output outputs --depth
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from . import config, imgio
from . import report as report_mod
from .pipeline import DefectPipeline

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Порядок = приоритет: канонические front/back главнее коротких алиасов —
# лишний кадр 1.jpg рядом с front.jpg не должен молча подменить пару.
PAIR_FRONT_STEMS = ("front", "before", "pered", "vpered", "впереди", "view_a", "a", "1")
PAIR_BACK_STEMS = ("back", "after", "zad", "szadi", "pozadi", "сзади", "view_b", "b", "2")


def _validate_image_path(path: Path) -> str | None:
    if not path.exists():
        return f"Путь не найден: {path}"
    if path.suffix.lower() not in IMG_EXT:
        return f"Не изображение ({', '.join(sorted(IMG_EXT))}): {path}"
    return None


def _named_image(folder: Path, stems: tuple[str, ...]) -> Path | None:
    """Файл по ПРИОРИТЕТУ имён из stems (front/back главнее алиасов);
    при одинаковом стеме с разными расширениями — первый по алфавиту."""
    by_stem: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            by_stem.setdefault(p.stem.lower(), p)
    for stem in stems:
        if stem in by_stem:
            return by_stem[stem]
    return None


def _discover_pairs(root: Path) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    """Найти пары вида <root>/<pair_id>/front.* + back.*.

    Имена `front/back` — основной контракт. Дополнительные короткие алиасы
    нужны только чтобы не ломаться на реальной полевой съёмке.
    """
    pairs: list[tuple[str, Path, Path]] = []
    errors: list[str] = []
    if not root.exists():
        return pairs, [f"Путь не найден: {root}"]
    if not root.is_dir():
        return pairs, [f"--pairs-dir должен быть папкой: {root}"]
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        front = _named_image(folder, PAIR_FRONT_STEMS)
        back = _named_image(folder, PAIR_BACK_STEMS)
        if front is None or back is None:
            errors.append(
                f"{folder.name}: нужна пара front.* и back.* "
                f"({', '.join(sorted(IMG_EXT))})")
            continue
        pairs.append((folder.name, front, back))
    if not pairs and not errors:
        errors.append(f"В {root} нет подпапок с парами front/back.")
    return pairs, errors


def _save_pair_result(pipe, a: Path, b: Path, out: Path,
                      pair_prefix: str | None = None,
                      used_stems: set | None = None) -> tuple[dict, str] | None:
    try:
        report, ov_a, ov_b, _ = pipe.analyze_pair(a, b)
    except Exception as e:  # noqa: BLE001
        print(f"  [ОШИБКА] пара {a.name}/{b.name}: {e}", file=sys.stderr)
        return None
    out.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    report["pair_inputs"] = {"front": str(a), "back": str(b)}
    if pair_prefix:
        report["pair_id"] = pair_prefix
    used = used_stems if used_stems is not None else set()
    # ВСЕ стемы пары префиксуются id папки (002__front, 002__front__back_pair):
    # раньше пофайловые overlay получали счётчик дедупа (front_2), который при
    # любом пропуске пары расходился с папкой 002 — кадр было не привязать к
    # паре (аудит 2026-07-07, №10). Конвенцию описывают docs/PAIR_WORKFLOW.md
    # и scripts/report_gen/README.md (манифест снимка ссылается на старые имена).
    view_prefix = f"{pair_prefix}__" if pair_prefix else ""
    stem_a = report_mod.unique_stem(f"{view_prefix}{a.stem}", used)
    stem_b = report_mod.unique_stem(f"{view_prefix}{b.stem}", used)
    pair_stem = report_mod.unique_stem(
        f"{view_prefix}{a.stem}__{b.stem}_pair", used)
    report_mod.save_report(report, out, pair_stem)
    # overlay вида A (носитель геометрии отчёта) дублируем под стем отчёта —
    # так его находит make_demo_report ({stem}.json ↔ {stem}_annotated.jpg).
    if not imgio.write_image(out / f"{pair_stem}_annotated.jpg", ov_a):
        print(f"  [ОШИБКА] не удалось записать {pair_stem}_annotated.jpg", file=sys.stderr)
    if not imgio.write_image(out / f"{stem_a}_annotated.jpg", ov_a):
        print(f"  [ОШИБКА] не удалось записать {stem_a}_annotated.jpg", file=sys.stderr)
    if not imgio.write_image(out / f"{stem_b}_annotated.jpg", ov_b):
        print(f"  [ОШИБКА] не удалось записать {stem_b}_annotated.jpg", file=sys.stderr)
    return report, pair_stem


def _run_pair(pipe, pair, out: Path) -> int:
    """Слияние двух видов одной ямы (--pair FRONT BACK): один fused JSON + два overlay."""
    a, b = Path(pair[0]), Path(pair[1])
    for path in (a, b):
        err = _validate_image_path(path)
        if err:
            print(err, file=sys.stderr)
            return 1
    if a.resolve() == b.resolve():
        print("--pair: FRONT и BACK — это один и тот же файл. Нужны ДВА разных "
              "вида одной ямы: «согласие» фото с самим собой фиктивно и ложно "
              "повышало бы уверенность слияния (аудит 2026-07-07).", file=sys.stderr)
        return 1
    saved = _save_pair_result(pipe, a, b, out)
    if saved is None:
        return 1
    report, _ = saved
    fz = report["fusion"]
    print(f"  пара {a.name[:18]}/{b.name[:18]}  режим: {report['mode']}  "
          f"слияние: {'да' if fz['fused'] else 'нет'}  дефектов: {len(report['defects'])}")
    print("Готово.")
    return 0


def _run_pairs_dir(pipe, pairs_dir: str, out: Path) -> int:
    pairs, errors = _discover_pairs(Path(pairs_dir))
    for err in errors:
        print(f"  [ПРОПУСК] {err}", file=sys.stderr)
    if not pairs:
        print("Ни одной пары front/back не найдено.", file=sys.stderr)
        return 1
    if out.exists():
        # Миграционный гейт (цикл 13): pair-отчёты старой конвенции
        # (002__front_2__back_2_pair.json) новый прогон НЕ перезапишет —
        # смешанная папка двоит пары в сводках и глобах (pair_quality_table,
        # make_demo_report, манифест report_gen берёт старый JSON первым).
        expected = {f"{pid}__{front.stem}__{back.stem}_pair.json"
                    for pid, front, back in pairs}
        stale = sorted(p.name for p in out.glob("*_pair.json")
                       if p.name not in expected)
        if stale:
            print(f"  [ВНИМАНИЕ] в {out} лежат {len(stale)} pair-отчётов, "
                  f"которые этот прогон не перезапишет (например {stale[0]}): "
                  "смешанная папка двоит пары в сводках/глобах. Рекомендуется "
                  "чистая --output (конвенция имён сменилась в цикле 13).",
                  file=sys.stderr)

    print(f"Обработка пар: {len(pairs)} -> {out}")
    ok_count = 0
    used_stems: set = set()
    summary_rows: list[dict[str, str | int]] = []
    for pair_id, front, back in pairs:
        saved = _save_pair_result(pipe, front, back, out,
                                  pair_prefix=pair_id,
                                  used_stems=used_stems)
        if saved is None:
            continue
        report, stem = saved
        fz = report["fusion"]
        status = "fused" if fz["fused"] else "unmatched"
        print(f"  {pair_id:20}  {status:9}  режим: {report['mode']:<18}  "
              f"дефектов: {len(report['defects']):2}  файл: {stem}.json")
        summary_rows.append({
            "pair_id": pair_id,
            "front": str(front),
            "back": str(back),
            "mode": report["mode"],
            "fusion_fused": str(bool(fz.get("fused", False))).lower(),
            "fusion_matched": str(bool(fz.get("matched", False))).lower(),
            "defects": len(report["defects"]),
            "report": f"{stem}.json",
            "overlay": f"{stem}_annotated.jpg",
        })
        ok_count += 1
    if summary_rows:
        summary_path = out / "pairs_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "pair_id", "front", "back", "mode", "fusion_fused",
                "fusion_matched", "defects", "report", "overlay",
            ])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Сводка: {summary_path}")
    print("Готово." if ok_count else "Ни одна пара не обработана.")
    return 0 if ok_count else 1


def main(argv=None) -> int:
    # На Windows перенаправленный stdout кодируется cp1251 — символы вне неё
    # (стрелки, галочки) роняют print. Заменяем некодируемое, а не падаем.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 — нестандартный поток (тесты, embed)
            pass
    ap = argparse.ArgumentParser(description="Анализ дорожных дефектов по фото.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="файл или папка с изображениями")
    src.add_argument("--pair", nargs=2, metavar=("FRONT", "BACK"),
                     help="два фото ОДНОЙ ямы (впереди и позади) — слияние видов "
                          "для более точного размера (F1)")
    src.add_argument("--pairs-dir",
                     help="папка с подпапками пар: 001/front.jpg + 001/back.jpg, "
                          "002/front.jpg + 002/back.jpg ...")
    ap.add_argument("--output", default=str(config.OUTPUTS_DIR), help="папка результатов")
    ap.add_argument("--depth", action="store_true", help="включить относительную глубину")
    ap.add_argument("--road-category", default="IV",
                    choices=sorted(config.GOST50597_REPAIR_DEADLINE_DAYS),
                    help="категория дороги для сроков ГОСТ (табл. 5.3)")
    ap.add_argument("--conf", type=float, default=config.DEFAULT_INFERENCE.det_conf)
    ap.add_argument("--blob-ref", action="store_true",
                    help="разрешить эталон по тёмному эллипсу при косом виде "
                         "(экспериментально: риск ложного масштаба, см. STATUS)")
    ap.add_argument("--ensemble", action="store_true",
                    help="второй pothole-проход (keremberke) поверх основного "
                         "детектора: ловит нетипичные ямы ценой x2 времени")
    ap.add_argument("--curb-ref", action="store_true",
                    help="эталон по борту/бордюру (экспериментально, OFF по умолчанию: "
                         "риск ложного масштаба — тень под бордюром; только как "
                         "подтверждающий эталон, см. PROJECT.md §4.4)")
    ap.add_argument("--marking-ref", action="store_true",
                    help="эталон по дорожной разметке (экспериментально, OFF по умолчанию: "
                         "класс ширины разметки по одному штриху неоднозначен → "
                         "low-confidence, требует подтверждения люком; см. PROJECT.md §4.4)")
    args = ap.parse_args(argv)

    out = Path(args.output)
    cfg = config.InferenceConfig(det_conf=args.conf,
                                 allow_blob_reference=args.blob_ref,
                                 ensemble_pothole=args.ensemble,
                                 allow_curb_reference=args.curb_ref,
                                 use_marking_reference=args.marking_ref)
    pipe = DefectPipeline(cfg=cfg, use_depth=args.depth, road_category=args.road_category)

    if args.pair:
        return _run_pair(pipe, args.pair, out)
    if args.pairs_dir:
        return _run_pairs_dir(pipe, args.pairs_dir, out)

    inp = Path(args.input)
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
