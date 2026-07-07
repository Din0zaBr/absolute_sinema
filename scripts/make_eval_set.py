"""Скелет held-out eval-набора детекции (P0 из docs/IMPROVEMENTS.md, цикл 14).

Копирует ОТОБРАННЫЕ кадры (сцены из карты FP + пробы recall + чистые пары +
корневые демо-фото) в `datasets/eval_v1/frames/` под стабильными id
(`029_front.jpg`, `demo_6dyN2B31.jpg`), пишет `manifest.csv` (id, источник,
md5) и `holdout_scenes.txt` — список сцен, которые `prepare_local_finetune.py`
ИСКЛЮЧАЕТ из набора дообучения (held-out дисциплина: eval никогда не в
обучении, docs/EVAL.md).

Скрипт НЕ размечает: ground truth появляется только из ручной разметки
(`build_eval_annotator.py` -> annotations.json). Честность: выдуманных боксов
в eval-наборе быть не может.

Запуск:
  .\\.venv\\Scripts\\python.exe scripts\\make_eval_set.py
  # затем: прогнать движок по кадрам и построить разметчик (docs/EVAL.md)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Отбор v1 — сцены, НАЗВАННЫЕ в карте FP/пропусков docs/IMPROVEMENTS.md
# (категория -> сцены datasets/local_pairs). Ровно те конфузоры, по которым
# нужен замер «доли FP по категориям», + пробы recall и чистые пары.
EVAL_SCENES: dict[str, list[str]] = {
    "car": ["016", "029", "032"],              # FP на припаркованных машинах
    "off_road": ["003", "009", "038", "044", "047"],  # стены/клумбы/плакаты
    "patch": ["002", "015", "024"],            # заплатки, названные pothole
    "shadow_curb": ["005", "013"],             # тень-маска, бордюры/лотки
    "recall_far": ["001", "007", "021", "026"],  # яма видна только вблизи
    "clean": ["006", "012", "018"],            # чистые пары (санити precision)
}


def _readable(path: Path) -> bool:
    """Битый JPEG не должен попасть в eval (паттерн prepare_local_finetune)."""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 — без Pillow проверку не форсируем
        return True
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001
        return False


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _copy(src: Path, dst: Path, force: bool, src_md5: str) -> None:
    # Скип по md5, не по размеру: изменённый исходник того же размера оставлял
    # бы старую копию при СВЕЖЕМ md5 в manifest — integrity набора лгала бы
    # (ревью 2026-07-07).
    if dst.exists() and not force and _md5(dst) == src_md5:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def collect_frames(pairs_dir: Path, demo_dir: Path | None,
                   scenes: dict[str, list[str]]) -> tuple[list[dict], list[str]]:
    """Список кадров-кандидатов [{frame_id, scene, view, source, group}] +
    предупреждения. Дедуп по md5 СОДЕРЖИМОГО: одна и та же фотография не должна
    попасть в метрики дважды под разными id (двойной счёт = ложная точность)."""
    rows: list[dict] = []
    warns: list[str] = []
    seen_md5: dict[str, str] = {}
    for group, ids in scenes.items():
        for sid in ids:
            folder = pairs_dir / sid
            if not folder.is_dir():
                warns.append(f"сцена {sid} ({group}): нет папки {folder} — пропуск")
                continue
            for view in ("front", "back"):
                src = folder / f"{view}.jpg"
                if not src.is_file():
                    warns.append(f"{sid}_{view}: нет кадра — пропуск")
                    continue
                if not _readable(src):
                    warns.append(f"{sid}_{view}: кадр не читается — пропуск")
                    continue
                digest = _md5(src)
                if digest in seen_md5:
                    warns.append(f"{sid}_{view}: тот же кадр, что {seen_md5[digest]} "
                                 "(общая сцена) — пропуск, двойного счёта не будет")
                    continue
                fid = f"{sid}_{view}"
                seen_md5[digest] = fid
                rows.append({"frame_id": fid, "scene": sid, "view": view,
                             "source": str(src), "group": group, "md5": digest})
    if demo_dir is not None:
        used_fids = {r["frame_id"] for r in rows}
        for src in sorted(demo_dir.glob("*.jpg")):
            if not _readable(src):
                warns.append(f"{src.name}: демо-кадр не читается — пропуск")
                continue
            digest = _md5(src)
            if digest in seen_md5:
                warns.append(f"{src.name}: дубль {seen_md5[digest]} — пропуск")
                continue
            # id уникален: общий 8-символьный префикс стемов молча схлопывал
            # два разных фото в один frame_id (ревью 2026-07-07)
            fid, k = f"demo_{src.stem[:8]}", 8
            while fid in used_fids and k < len(src.stem):
                k += 4
                fid = f"demo_{src.stem[:k]}"
            n = 2
            while fid in used_fids:
                fid = f"demo_{src.stem[:8]}_{n}"
                n += 1
            used_fids.add(fid)
            seen_md5[digest] = fid
            rows.append({"frame_id": fid, "scene": fid, "view": "single",
                         "source": str(src), "group": "demo", "md5": digest})
    return rows, warns


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default="datasets/local_pairs")
    ap.add_argument("--out", default="datasets/eval_v1")
    ap.add_argument("--demo-dir", default=str(ROOT),
                    help="папка одиночных демо-фото (*.jpg); '' — не брать")
    ap.add_argument("--force", action="store_true",
                    help="перезаписывать существующие копии кадров")
    args = ap.parse_args()

    pairs_dir = Path(args.pairs)
    if not pairs_dir.is_dir():
        print(f"Нет папки пар: {pairs_dir}")
        return 2
    demo_dir = Path(args.demo_dir) if args.demo_dir else None

    rows, warns = collect_frames(pairs_dir, demo_dir, EVAL_SCENES)
    if not rows:
        print("Ни одного кадра не отобрано — проверь --pairs/--demo-dir")
        return 2

    out = Path(args.out)
    frames_dir = out / "frames"
    for r in rows:
        _copy(Path(r["source"]), frames_dir / f"{r['frame_id']}.jpg",
              args.force, r["md5"])

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "scene", "view", "group",
                                          "source", "md5"])
        w.writeheader()
        w.writerows(rows)

    # held-out список для prepare_local_finetune (только сцены пар; демо-фото
    # в local_pairs не живут и в дообучение не попадают по построению)
    holdout = sorted({r["scene"] for r in rows if r["group"] != "demo"})
    (out / "holdout_scenes.txt").write_text(
        "# Сцены eval_v1: held-out, prepare_local_finetune их ИСКЛЮЧАЕТ из\n"
        "# набора дообучения (P0, docs/EVAL.md). Не редактировать вручную —\n"
        "# перегенерируется scripts/make_eval_set.py.\n"
        + "\n".join(holdout) + "\n", encoding="utf-8")

    by_group: dict[str, int] = {}
    for r in rows:
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1
    print(f"Кадров в eval_v1: {len(rows)}  "
          f"({', '.join(f'{g}: {n}' for g, n in sorted(by_group.items()))})")
    print(f"Сцен в held-out: {len(holdout)}  -> {out / 'holdout_scenes.txt'}")
    for wmsg in warns:
        print(f"[!] {wmsg}")
    print(f"Набор: {out}")
    print("Дальше (docs/EVAL.md): прогнать движок по кадрам, построить разметчик:")
    print(f"  python -m road_defect.cli --input {frames_dir} --output outputs_eval/reports")
    print("  python scripts/build_eval_annotator.py --proposals outputs_eval/reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
