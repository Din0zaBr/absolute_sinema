"""Раскладка собранных фото в каноническую структуру DATASET.md §3.

Вход  — папка выезда со структурой «Ямки/Яма N/{Front,Back,Эталоны}»
        (см. «Структура папок.txt» сборщика).
Выход — datasets/local_pairs/NNN/front.jpg + back.jpg   (пары для --pairs-dir)
        datasets/gt_photos/NNN_ruler_K.jpg              (кадры с рулеткой, НЕ для обучения)
        datasets/journal.csv                            (черновик журнала §4: id, сцена,
                                                         дата, GPS из EXIF; замеры пустые)

Сцена (колонка scene) — группа ям, снятых ОДНИМ общим кадром front (сборщик
дублирует такой кадр по папкам соседних ям; группируем по md5). id сцены =
наименьший id ямы группы. Нужна для честного recall по сценам и для
train/val-сплита без утечки одинаковых кадров между сплитами.

Копирует, не перемещает. Идемпотентен: существующий файл того же размера
не перезаписывается (--force — перезаписать). Существующий journal.csv не
трогается — черновик тогда пишется в journal_draft.csv, слияние за человеком.

Запуск:
  .\\.venv\\Scripts\\python.exe scripts\\ingest_local_pairs.py "Проэкт\\Проэкт\\Ямки"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

# Канонический набор — road_defect.cli.IMG_EXT (дублируем литералом: ingest
# намеренно не тянет тяжёлый пакет ради константы). Держать в согласии.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
JOURNAL_FIELDS = [
    "id", "scene", "date", "street_or_gps", "type", "length_cm", "width_cm",
    "depth_cm", "reference_in_frame", "weather", "notes",
]


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def _exif(path: Path) -> dict:
    try:
        with Image.open(path) as im:
            raw = im._getexif() or {}
    except Exception:
        return {}
    return {TAGS.get(k, k): v for k, v in raw.items()}


def _gps_decimal(gps_info: dict | None) -> str:
    """GPSInfo EXIF -> 'lat lon' десятичными градусами (формат журнала §4)."""
    if not gps_info:
        return ""
    try:
        def to_deg(triplet):
            d, m, s = (float(x) for x in triplet)
            return d + m / 60.0 + s / 3600.0

        lat = to_deg(gps_info[2]) * (-1 if gps_info[1] == "S" else 1)
        lon = to_deg(gps_info[4]) * (-1 if gps_info[3] == "W" else 1)
        return f"{lat:.5f} {lon:.5f}"
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return ""


def _verify_readable(path: Path) -> str | None:
    """None, если изображение читается; иначе текст проблемы."""
    try:
        with Image.open(path) as im:
            im.verify()
        return None
    except Exception as exc:  # noqa: BLE001 - любой сбой чтения = негодный кадр
        return f"{type(exc).__name__}: {exc}"


def _copy(src: Path, dst: Path, force: bool) -> str:
    if dst.exists() and not force and dst.stat().st_size == src.stat().st_size:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copy"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # cp1251-консоль Windows
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="папка выезда с подпапками «Яма N»")
    ap.add_argument("--datasets", default="datasets", help="корень datasets/")
    ap.add_argument("--type", default="pothole", dest="defect_type",
                    help="грубый тип дефекта для журнала (§4)")
    ap.add_argument("--force", action="store_true", help="перезаписывать существующие файлы")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"Источник не папка: {source}")
        return 2

    pit_dirs: list[tuple[int, Path]] = []
    for d in source.iterdir():
        if d.is_dir():
            m = re.search(r"(\d+)", d.name)
            if m:
                pit_dirs.append((int(m.group(1)), d))
    if not pit_dirs:
        print(f"В {source} нет подпапок вида «Яма N»")
        return 2
    pit_dirs.sort()

    ids = [n for n, _ in pit_dirs]
    dups = {n for n in ids if ids.count(n) > 1}
    if dups:
        print(f"ОШИБКА: повторяющиеся номера ям: {sorted(dups)}")
        return 2

    ds_root = Path(args.datasets)
    pairs_root = ds_root / "local_pairs"
    gt_root = ds_root / "gt_photos"

    problems: list[str] = []   # жёсткие сбои -> exit 1 (пара пропущена/битая)
    notes: list[str] = []      # мягкие замечания, пара всё равно обработана
    rows: list[dict] = []
    copied = skipped = 0
    # Сборщик снимает группу соседних ям ОДНИМ общим кадром (front/back дублируются
    # по папкам ям, рулетки индивидуальны). Группируем по md5 front — это «сцены»
    # (см. док модуля); группы уходят в колонку scene журнала.
    front_md5s: dict[str, list[str]] = defaultdict(list)

    for num, pit_dir in pit_dirs:
        pid = f"{num:03d}"
        fronts = _images(pit_dir / "Front")
        backs = _images(pit_dir / "Back")
        rulers = _images(pit_dir / "Эталоны")

        if len(fronts) != 1 or len(backs) != 1:
            problems.append(
                f"{pit_dir.name}: front={len(fronts)}, back={len(backs)} — нужна ровно одна пара, пропущена")
            continue
        if not rulers:
            notes.append(f"{pit_dir.name}: нет кадров с рулеткой (Эталоны) — пара всё равно скопирована")

        # Битый front/back делает пару непригодной — жёсткий пропуск. Битая
        # РУЛЕТКА (некритичный gt-кадр) не должна ронять валидную пару: роняем
        # только эту рулетку, пара копируется (ревью 2026-07-05).
        pair_plan = [(fronts[0], pairs_root / pid / "front.jpg"),
                     (backs[0], pairs_root / pid / "back.jpg")]
        pair_bad = False
        for src, _ in pair_plan:
            issue = _verify_readable(src)
            if issue:
                problems.append(f"{src}: не читается ({issue}) — пара пропущена")
                pair_bad = True
        if pair_bad:
            continue

        good_rulers: list[tuple[Path, Path]] = []
        for k, r in enumerate(rulers, 1):
            issue = _verify_readable(r)
            if issue:
                notes.append(f"{r}: рулетка не читается ({issue}) — пропущена, пара скопирована")
            else:
                good_rulers.append((r, gt_root / f"{pid}_ruler_{k}.jpg"))

        for src, dst in pair_plan + good_rulers:
            if _copy(src, dst, args.force) == "copy":
                copied += 1
            else:
                skipped += 1

        exif = _exif(fronts[0])
        dt = str(exif.get("DateTimeOriginal", ""))
        date = dt.split(" ")[0].replace(":", "-") if dt else ""
        front_md5s[hashlib.md5(fronts[0].read_bytes()).hexdigest()].append(pid)
        rows.append({
            "id": pid,
            "scene": "",  # заполняется ниже, когда известны все группы
            "date": date,
            "street_or_gps": _gps_decimal(exif.get("GPSInfo")),
            "type": args.defect_type,
            "length_cm": "", "width_cm": "", "depth_cm": "",
            "reference_in_frame": "", "weather": "",
            "notes": f"gt_photos: {len(good_rulers)} кадр(а) с рулеткой",
        })

    pid2scene = {pid: pids[0] for pids in front_md5s.values() for pid in pids}
    for row in rows:
        row["scene"] = pid2scene.get(row["id"], row["id"])

    # корень datasets/ иначе создаётся лениво первым _copy; при прогоне, где ВСЕ
    # пары битые (ни одного копирования), его нет — журнал упал бы FileNotFoundError
    ds_root.mkdir(parents=True, exist_ok=True)
    journal = ds_root / "journal.csv"
    if journal.exists():
        journal = ds_root / "journal_draft.csv"
        print(f"journal.csv уже существует — черновик пишется в {journal}")
    with open(journal, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"Ям обработано: {len(rows)} из {len(pit_dirs)}; файлов скопировано: {copied}, "
          f"пропущено (уже на месте): {skipped}")
    print(f"Сцен (уникальных кадров front): {len(front_md5s)}")
    for pids in sorted(front_md5s.values()):
        if len(pids) > 1:
            print(f"  сцена {pids[0]}: общий кадр у ям {', '.join(pids)}")
    print(f"Пары:    {pairs_root}")
    print(f"Рулетки: {gt_root}")
    print(f"Журнал:  {journal} (замеры пустые — заполнить по кадрам gt_photos)")
    if notes:
        print("\nЗАМЕЧАНИЯ (не влияют на код возврата):")
        for m in notes:
            print(f"  - {m}")
    if problems:
        print("\nПРОБЛЕМЫ:")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
