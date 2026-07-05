"""Раскладка собранных пар в YOLO-скелет для дообучения детектора (DATASET.md §6).

Готовит НАБОР ПОД РУЧНУЮ РАЗМЕТКУ: копирует уникальные кадры сцен (front/back)
в images/{train,val}, создаёт ПУСТЫЕ labels/**/*.txt (заготовки под боксы в
CVAT/LabelImg) и data.yaml с таксономией rezzzq (0=D00…4=Repair). Разметка боксов
и запуск обучения — руками/на GPU (скрипт только раскладывает).

Сплит — по СЦЕНАМ (соседние ямы сборщик снимает ОДНИМ кадром → это одна
аннотируемая картинка с несколькими ямами; вся сцена целиком в один сплит,
против утечки — DATASET.md §6). Дедуп по md5 кадра front, как в ingest: 48 пар
сворачиваются в уникальные сцены, лишние копии одинаковых кадров не плодятся.
Сплит детерминированный (без ГСЧ — воспроизводимость); честная оговорка: это
сплит по СЦЕНАМ, а не по улицам — для локационного нужен разбор GPS (журнал §4).

Запуск:
  .\\.venv\\Scripts\\python.exe scripts\\prepare_local_finetune.py
  # разметить боксы в datasets\\local_rf_yolo\\images\\**\\*.jpg -> labels\\**\\*.txt,
  # затем: python scripts\\finetune_rdd2022.py --data datasets\\local_rf_yolo\\data.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Таксономия ДОЛЖНА совпадать с finetune_rdd2022.CLASS_NAMES и весами rezzzq —
# контракт держим тестом (tests/test_prepare_local_finetune.py).
CLASS_NAMES = ["D00", "D10", "D20", "D40", "Repair"]


def scene_reps(pairs_dir: Path) -> list[tuple[str, Path, Path]]:
    """Уникальные сцены как (scene_id, front, back). scene_id = наименьший id
    группы пар с ОДИНАКОВЫМ кадром front (дедуп по md5, как в ingest)."""
    groups: dict[str, list[str]] = defaultdict(list)
    fronts: dict[str, Path] = {}
    for d in sorted(p for p in pairs_dir.iterdir() if p.is_dir()):
        front = d / "front.jpg"
        if not front.is_file():
            continue
        groups[hashlib.md5(front.read_bytes()).hexdigest()].append(d.name)
        fronts[d.name] = front
    reps = []
    for pids in groups.values():
        rep = sorted(pids)[0]
        reps.append((rep, fronts[rep], pairs_dir / rep / "back.jpg"))
    return sorted(reps)


def assign_splits(scene_ids: list[str], val_frac: float = 0.15) -> dict[str, str]:
    """Детерминированный сплит по сценам (без ГСЧ — воспроизводимо).

    n_val = round(n·val_frac), зажат в [1, n-1]: при n≥2 и 0<frac<1 и val, и
    train гарантированно непусты для ЛЮБОЙ доли (а не только ≤0.5 — прошлая
    схема «каждая k-я» молча упиралась в 50%, ревью 2026-07-05). Val-сцены
    разнесены равномерно по отсортированному списку. При n<2 или val_frac≤0 —
    всё в train (вызывающий предупредит и укажет val на train)."""
    ids = sorted(scene_ids)
    n = len(ids)
    if n < 2 or val_frac <= 0:
        return {s: "train" for s in ids}
    n_val = min(n - 1, max(1, round(n * val_frac)))
    step = n / n_val
    val_idx = {int((j + 0.5) * step) for j in range(n_val)}
    return {s: ("val" if i in val_idx else "train") for i, s in enumerate(ids)}


def data_yaml_text(out_dir: Path, class_names: list[str],
                   val_rel: str = "images/val") -> str:
    """Ultralytics data.yaml. val_rel обычно images/val, но при вырожденном
    наборе (одна сцена) вызывающий передаёт images/train, чтобы обучение вообще
    запускалось (пустой val ронял бы ultralytics — ревью 2026-07-05).
    (Свой эмиттер YAML: скрипт автономен; таксономию держит контракт-тест.)"""
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    path = out_dir.resolve().as_posix()
    return (f"# сгенерирован scripts/prepare_local_finetune.py (DATASET.md §6)\n"
            f"path: {path}\n"
            f"train: images/train\n"
            f"val: {val_rel}\n"
            f"names:\n{names}\n")


def _readable(path: Path) -> bool:
    """Мягкая проверка декодируемости кадра (как ingest гварднёт front/back):
    битый/обрезанный JPEG не должен уехать в набор и уронить даталоадер обучения.
    Без Pillow проверку не форсируем (не блокируем прогон)."""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return True
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001
        return False


def _copy(src: Path, dst: Path, force: bool) -> None:
    # автономный дубль ingest_local_pairs._copy (idempotent skip): скрипты —
    # самостоятельные CLI по паттерну репозитория, намеренно без общего модуля.
    if dst.exists() and not force and dst.stat().st_size == src.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _clean_stale(out: Path, expected_img: set[Path],
                 expected_lbl: set[Path]) -> list[Path]:
    """Убрать из images/labels файлы, которых нет в плане этого прогона (сцена
    сменила сплит / другой --val-frac / пара удалена). Иначе одна сцена осталась
    бы в ОБОИХ сплитах -> утечка train/val (ревью 2026-07-05). Возвращает
    НЕПУСТЫЕ label'ы, что пришлось удалить, — по ним ручная разметка потеряна."""
    dropped: list[Path] = []
    for img in out.glob("images/*/*.jpg"):
        if img not in expected_img:
            img.unlink()
    for lbl in out.glob("labels/*/*.txt"):
        if lbl not in expected_lbl:
            if lbl.read_text(encoding="utf-8").strip():
                dropped.append(lbl)
            lbl.unlink()
    return dropped


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default="datasets/local_pairs")
    ap.add_argument("--out", default="datasets/local_rf_yolo")
    ap.add_argument("--val-frac", type=float, default=0.15, dest="val_frac")
    ap.add_argument("--force", action="store_true", help="перезаписывать существующие файлы")
    args = ap.parse_args()

    pairs_dir = Path(args.pairs)
    if not pairs_dir.is_dir():
        print(f"Нет папки пар: {pairs_dir}")
        return 2
    reps = scene_reps(pairs_dir)
    if not reps:
        print(f"В {pairs_dir} нет пар NNN/front.jpg")
        return 2

    out = Path(args.out)
    splits = assign_splits([s for s, _, _ in reps], args.val_frac)

    # план прогона + множества ОЖИДАЕМЫХ файлов (для зачистки устаревших)
    plan: list[tuple[Path, Path, Path]] = []   # (src, img, lbl)
    expected_img: set[Path] = set()
    expected_lbl: set[Path] = set()
    skipped: list[str] = []
    counts = {"train": 0, "val": 0}
    seen_scene = set()
    for scene, front, back in reps:
        split = splits[scene]
        for view, src in (("front", front), ("back", back)):
            if not src.is_file():
                continue
            if not _readable(src):
                skipped.append(f"{scene}_{view}: кадр не читается — пропуск")
                continue
            img = out / "images" / split / f"{scene}_{view}.jpg"
            lbl = out / "labels" / split / f"{scene}_{view}.txt"
            plan.append((src, img, lbl))
            expected_img.add(img)
            expected_lbl.add(lbl)
            if scene not in seen_scene:
                seen_scene.add(scene)
                counts[split] += 1

    if not plan:
        print("Нет читаемых кадров для раскладки")
        return 2

    dropped = _clean_stale(out, expected_img, expected_lbl)

    n_img = n_stub = 0
    for src, img, lbl in plan:
        _copy(src, img, args.force)
        n_img += 1
        if not lbl.exists():               # пустая заготовка под разметку
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("", encoding="utf-8")
            n_stub += 1

    has_val = counts["val"] > 0
    val_rel = "images/val" if has_val else "images/train"
    (out / "data.yaml").write_text(
        data_yaml_text(out, CLASS_NAMES, val_rel), encoding="utf-8")

    val_scenes = sorted(s for s, sp in splits.items() if sp == "val")
    print(f"Сцен: {len(reps)}  (train {counts['train']} / val {counts['val']})")
    print(f"Кадров скопировано: {n_img}, пустых label-заготовок создано: {n_stub}")
    print(f"val-сцены: {', '.join(val_scenes) or '—'}")
    if not has_val:
        print("[!] val пуст (мало сцен) — data.yaml указывает val на train: "
              "обучение запустится, но метрики val ОПТИМИСТИЧНЫ, набери ещё сцен.")
    for s in skipped:
        print(f"[!] {s}")
    if dropped:
        print(f"[!] удалены устаревшие копии сцен, сменивших сплит; из них с "
              f"РАЗМЕТКОЙ (потеряна): {', '.join(p.name for p in dropped)}")
    print(f"Набор:    {out}   (data.yaml готов)")
    print("Дальше: разметь боксы (CVAT/LabelImg, YOLO txt) по DATASET.md §6, затем")
    print(f"  python scripts/finetune_rdd2022.py --data {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
