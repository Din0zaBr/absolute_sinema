"""Раскладка собранных пар в YOLO-скелет для дообучения детектора (DATASET.md §6).

Готовит НАБОР ПОД РУЧНУЮ РАЗМЕТКУ: копирует уникальные кадры сцен (front/back)
в images/{train,val}, создаёт ПУСТЫЕ labels/**/*.txt (заготовки под боксы в
CVAT/LabelImg) и data.yaml с таксономией rezzzq (0=D00…4=Repair). Разметка боксов
и запуск обучения — руками/на GPU (скрипт только раскладывает).

Сплит — по СЦЕНАМ (соседние ямы сборщик снимает ОДНИМ кадром → это одна
аннотируемая картинка с несколькими ямами; вся сцена целиком в один сплит,
против утечки — DATASET.md §6). Дедуп по md5 кадра front, как в ingest: 48 пар
сворачиваются в уникальные сцены, лишние копии одинаковых кадров не плодятся.
Сцены held-out eval-набора (datasets/eval_v1/holdout_scenes.txt, P0 из
IMPROVEMENTS.md) ИСКЛЮЧАЮТСЯ из набора целиком — eval никогда не в обучении
(docs/EVAL.md; отключается только явным --holdout '').
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


def _scene_groups(pairs_dir: Path) -> list[tuple[str, list[str], Path, Path]]:
    """Группы пар с ОДИНАКОВЫМ кадром front (дедуп по md5, как в ingest):
    (rep = наименьший id группы, все id группы, front, back)."""
    groups: dict[str, list[str]] = defaultdict(list)
    fronts: dict[str, Path] = {}
    for d in sorted(p for p in pairs_dir.iterdir() if p.is_dir()):
        front = d / "front.jpg"
        if not front.is_file():
            continue
        groups[hashlib.md5(front.read_bytes()).hexdigest()].append(d.name)
        fronts[d.name] = front
    out = []
    for pids in groups.values():
        rep = sorted(pids)[0]
        out.append((rep, sorted(pids), fronts[rep], pairs_dir / rep / "back.jpg"))
    return sorted(out)


def scene_reps(pairs_dir: Path) -> list[tuple[str, Path, Path]]:
    """Уникальные сцены как (scene_id, front, back). scene_id = наименьший id
    группы пар с ОДИНАКОВЫМ кадром front (дедуп по md5, как в ingest)."""
    return [(rep, front, back)
            for rep, _members, front, back in _scene_groups(pairs_dir)]


def load_holdout(path: Path) -> set[str]:
    """Сцены held-out eval-набора (scripts/make_eval_set.py): по id на строку,
    '#' — комментарий. Кодировка utf-8(-sig): BOM от Windows-редактора молча
    «съедал» первую сцену (ревью 2026-07-07); не-UTF-8 (UTF-16 из PowerShell
    `>`) — внятная ошибка, а не трейсбек."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"holdout-файл {path} не в UTF-8 (сохрани как UTF-8; PowerShell "
            f"`>`/Out-File по умолчанию пишет UTF-16): {e}") from e
    out: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def load_eval_md5s(manifest: Path) -> set[str]:
    """md5 кадров eval-набора из manifest.csv (контентная сверка: та же
    ФОТОГРАФИЯ не должна попасть в обучение ни под каким id — группы сцен
    клеятся только по front, back-канал закрывается именно этой сверкой)."""
    import csv

    out: set[str] = set()
    with manifest.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("md5"):
                out.add(row["md5"])
    return out


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
    сменила сплит / другой --val-frac / пара удалена / ушла в held-out). Иначе
    одна сцена осталась бы в ОБОИХ сплитах -> утечка train/val (ревью
    2026-07-05). НЕПУСТЫЕ label'ы не уничтожаются, а перемещаются в
    <out>/labels_removed/ (ревью 2026-07-07: рост holdout-списка стирал бы
    часы ручной разметки безвозвратно); возвращает их новые пути."""
    moved: list[Path] = []
    for img in out.glob("images/*/*.jpg"):
        if img not in expected_img:
            img.unlink()
    for lbl in out.glob("labels/*/*.txt"):
        if lbl not in expected_lbl:
            if lbl.read_text(encoding="utf-8").strip():
                keep = out / "labels_removed" / lbl.name
                n = 2
                while keep.exists():
                    keep = out / "labels_removed" / f"{lbl.stem}_{n}.txt"
                    n += 1
                keep.parent.mkdir(parents=True, exist_ok=True)
                lbl.replace(keep)
                moved.append(keep)
            else:
                lbl.unlink()
    return moved


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default="datasets/local_pairs")
    ap.add_argument("--out", default="datasets/local_rf_yolo")
    ap.add_argument("--val-frac", type=float, default=0.15, dest="val_frac")
    ap.add_argument("--holdout", default=None,
                    help="файл сцен held-out eval-набора — они ИСКЛЮЧАЮТСЯ из "
                         "дообучения (P0, docs/EVAL.md). По умолчанию — сосед "
                         "набора пар: <pairs>/../eval_v1/holdout_scenes.txt "
                         "(нет файла — гейт неактивен); '' — отключить явно")
    ap.add_argument("--force", action="store_true", help="перезаписывать существующие файлы")
    args = ap.parse_args()

    pairs_dir = Path(args.pairs)
    if not pairs_dir.is_dir():
        print(f"Нет папки пар: {pairs_dir}")
        return 2
    groups = _scene_groups(pairs_dir)
    if not groups:
        print(f"В {pairs_dir} нет пар NNN/front.jpg")
        return 2

    # Гейт контаминации (P0): сцена уходит из обучения, если ЛЮБОЙ id её
    # md5-группы в held-out (одна и та же фотография не должна попасть в
    # обучение под id соседней пары — иначе eval перестаёт быть honest).
    # Дефолт пути — ОТ НАБОРА ПАР (<pairs>/../eval_v1/...), не от cwd: кастомный
    # --pairs (тесты, чужая раскладка) не должен молча цеплять боевой holdout;
    # resolve() — иначе `--pairs .` давал parent «.» и гейт тихо промахивался.
    # Статус гейта печатается ВСЕГДА (ревью 2026-07-07: молчаливо-неактивная
    # защита неотличима от активной — ложная гарантия хуже отсутствия).
    if args.holdout is None:
        holdout_path = pairs_dir.resolve().parent / "eval_v1" / "holdout_scenes.txt"
        if not holdout_path.is_file():
            print(f"[!] holdout-гейт НЕАКТИВЕН: нет {holdout_path} — eval-сцены "
                  "НЕ исключаются (собери набор: scripts/make_eval_set.py)")
            holdout_path = None
    elif args.holdout:
        holdout_path = Path(args.holdout)
        if not holdout_path.is_file():
            print(f"Явно указанный --holdout не найден: {holdout_path} — это "
                  "ошибка, а не отключение гейта (отключение: --holdout '').")
            return 2
    else:
        holdout_path = None
        print("[!] holdout-гейт ОТКЛЮЧЁН явно (--holdout ''): eval-сцены "
              "НЕ исключаются — метрики на eval_v1 будут train-contaminated")

    holdout: set[str] = set()
    eval_md5s: set[str] = set()
    if holdout_path is not None:
        try:
            holdout = load_holdout(holdout_path)
        except ValueError as e:
            print(e)
            return 2
        manifest = holdout_path.parent / "manifest.csv"
        if manifest.is_file():
            try:
                eval_md5s = load_eval_md5s(manifest)
            except Exception as e:  # noqa: BLE001
                print(f"[!] manifest eval-набора не читается ({e}) — "
                      "контентная md5-сверка пропущена")

    def _content_hits(g) -> list[str]:
        """Кадры группы, байт-в-байт совпадающие с кадрами eval-набора."""
        if not eval_md5s:
            return []
        _rep, _members, front, back = g
        return [p.name for p in (front, back) if p.is_file()
                and hashlib.md5(p.read_bytes()).hexdigest() in eval_md5s]

    if holdout_path is not None:
        excl_id = [g for g in groups if set(g[1]) & holdout]
        rest = [g for g in groups if not (set(g[1]) & holdout)]
        excl_md5 = [g for g in rest if _content_hits(g)]
        groups = [g for g in rest if not _content_hits(g)]
        print(f"[held-out] гейт активен: {len(holdout)} сцен из {holdout_path}; "
              f"исключено по id: {len(excl_id)} "
              f"({', '.join(rep for rep, _m, _f, _b in excl_id) or '—'})")
        if excl_md5:
            print(f"[held-out] исключено по СОДЕРЖИМОМУ (кадр байт-в-байт в "
                  f"eval_v1): {', '.join(rep for rep, _m, _f, _b in excl_md5)}")
    reps = [(rep, front, back) for rep, _members, front, back in groups]
    if not reps:
        print("После исключения held-out сцен не осталось ни одной для "
              "обучения — набор дообучения пуст, собери больше пар.")
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
        print(f"[!] у сцен, ушедших из плана, была РУЧНАЯ РАЗМЕТКА — сохранена "
              f"в {out / 'labels_removed'}: {', '.join(p.name for p in dropped)}")
    print(f"Набор:    {out}   (data.yaml готов)")
    print("Дальше: разметь боксы (CVAT/LabelImg, YOLO txt) по DATASET.md §6, затем")
    print(f"  python scripts/finetune_rdd2022.py --data {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
