"""Импорт внешней разметки eval-набора -> datasets/eval_v1/annotations.json.

Кадры datasets/eval_v1/frames/ размечаются во внешней программе (выбран
X-AnyLabeling, пошагово — docs/XANYLABELING_GUIDE.md), этот скрипт переводит её
файлы в наш annotations.json (схема v1, docs/EVAL.md). Перед записью результат
проходит ТОТ ЖЕ валидатор, что использует замер (eval_detection.load_annotations),
и сверяется с manifest.csv. Импорт «всё или ничего»: любая проблема — файл не
тронут, все проблемы печатаются списком; прежняя разметка уходит в *.bak-*.

Форматы (--format auto распознаёт по содержимому --input):
  xany — родные пер-кадровые JSON X-AnyLabeling/LabelMe: автосохраняются рядом
         с кадрами, экспортировать ничего не нужно (рекомендуемый путь);
  yolo — экспорт YOLO: classes.txt + <кадр>.txt с нормированными «класс cx cy w h»;
  coco — один COCO-файл (--input указывает на .json-файл, не папку).

Честность импорта:
  • метки — строго из протокола EVAL.md: дефекты + конфузоры (car/off_road/
    shadow_curb/other) как обычные классы; коды RDD (D00/D40/Repair/...)
    принимаются как синонимы. Неизвестная метка валит импорт со списком
    кадров, где встретилась;
  • кадр с файлом разметки — даже без единой рамки — получает status="reviewed":
    «смотрел, дефектов нет» (пустые кадры перечисляются в сводке — проверь);
    кадр БЕЗ файла в annotations.json не попадает, замер сам покажет его как
    «manifest БЕЗ разметки»;
  • полигоны (в т.ч. от SAM-ассиста) сводятся к их осевому bbox — это и есть
    GT-рамка; линии/точки рамкой не считаются — ошибка;
  • несовпадение размеров с кадром-оригиналом (размечена ужатая копия) —
    ошибка: координаты на оригинал не переносимы.

Запуск (дефолты = стандартный путь X-AnyLabeling: JSONы в папке кадров):
  .\\.venv\\Scripts\\python.exe scripts\\import_annotations.py
  .\\.venv\\Scripts\\python.exe scripts\\import_annotations.py --input <папка|coco.json> --format yolo
  # затем замер: .\\.venv\\Scripts\\python.exe scripts\\eval_detection.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_defect import config  # noqa: E402

try:  # Pillow опционален: без него пропускается только сверка размеров (xany/coco)
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None


def _load_eval_detection():
    """Списки классов и валидатор берём ИЗ харнесса замера: один источник
    правды — конвертер не может пропустить то, что завалит eval_detection."""
    spec = importlib.util.spec_from_file_location(
        "eval_detection", Path(__file__).resolve().parent / "eval_detection.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ED = _load_eval_detection()
DEFECT_LABELS = ED.DEFECT_LABELS
CONFUSOR_CATEGORIES = ED.CONFUSOR_CATEGORIES

# Синонимы: коды RDD/датасета (D40, Repair, ...) -> канонические имена классов.
LABEL_ALIASES = {k.lower(): v for k, v in config.DEFECT_CLASSES.items()}


def normalize_label(raw) -> str:
    """« Pothole » / shadow-curb / D40 -> pothole / shadow_curb / pothole."""
    s = re.sub(r"[\s\-]+", "_", str(raw).strip().lower())
    return LABEL_ALIASES.get(s, s)


def _num(v: float):
    """Числа рамок: до 0.1 px, целые — целыми (диффы annotations.json чище)."""
    r = round(float(v), 1)
    return int(r) if r == int(r) else r


def image_size_px(path: Path) -> tuple[int, int] | None:
    """(W, H) кадра или None (нет файла/Pillow). Без cv2: пути бывают с
    кириллицей, Pillow читает их корректно (паттерн make_eval_set)."""
    if Image is None or not path.is_file():
        return None
    try:
        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:  # noqa: BLE001
        return None


# --- разбор форматов -----------------------------------------------------------
# Все парсеры возвращают fid -> {"size": [W, H] | None,
#   "shapes": [(raw_label, 4 числа, полигон?, нормированные?)]}
_BBOX_SHAPE_TYPES = ("rectangle", "polygon", "rotation")


def shape_to_bbox(shape: dict) -> tuple[list[float], bool]:
    """[x, y, w, h] по осевому охвату точек + признак «не прямоугольник».

    rectangle у X-AnyLabeling бывает 2-точечным (диагональ, старые версии) и
    4-точечным (углы, новые) — min/max покрывает оба; polygon (в т.ч. от SAM)
    и rotation честно сводятся к своему осевому bbox."""
    stype = shape.get("shape_type") or "rectangle"
    if stype not in _BBOX_SHAPE_TYPES:
        raise ValueError(f"фигура {stype!r} не переводится в рамку — "
                         "обведи прямоугольником (или полигоном)")
    pts = shape.get("points") or []
    if len(pts) < 2 or any(not isinstance(p, (list, tuple)) or len(p) != 2
                           for p in pts):
        raise ValueError(f"у фигуры {len(pts)} точек — рамку не построить")
    try:
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
    except (TypeError, ValueError):
        raise ValueError("точки фигуры — не числа") from None
    x0, y0 = min(xs), min(ys)
    w, h = max(xs) - x0, max(ys) - y0
    if w <= 0 or h <= 0:
        raise ValueError("вырожденная рамка (нулевая ширина/высота)")
    return [x0, y0, w, h], stype != "rectangle"


def parse_xany_dir(input_dir: Path, problems: list[str]) -> dict[str, dict]:
    parsed: dict[str, dict] = {}
    for jp in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8-sig"))
        except Exception as e:  # noqa: BLE001
            problems.append(f"{jp.name}: не читается как JSON ({e})")
            continue
        if not isinstance(data, dict) or "shapes" not in data:
            problems.append(f"{jp.name}: нет блока shapes — не похоже на "
                            "разметку X-AnyLabeling/LabelMe")
            continue
        fid = jp.stem
        ipath = data.get("imagePath")
        if ipath and Path(str(ipath)).stem != fid:
            problems.append(
                f"{jp.name}: внутри imagePath={ipath!r}, а имя файла {fid!r} — "
                "кадр переименован? id кадра обязан совпадать с именем в frames/")
            continue
        w, h = data.get("imageWidth"), data.get("imageHeight")
        size = ([float(w), float(h)]
                if isinstance(w, (int, float)) and isinstance(h, (int, float))
                and w > 0 and h > 0 else None)
        shapes = []
        for i, sh in enumerate(data.get("shapes") or [], 1):
            if not isinstance(sh, dict):
                problems.append(f"{fid}: фигура №{i} — не объект: {sh!r}")
                continue
            try:
                bbox, is_poly = shape_to_bbox(sh)
            except ValueError as e:
                problems.append(f"{fid}: фигура №{i} ({sh.get('label')!r}): {e}")
                continue
            shapes.append((sh.get("label"), bbox, is_poly, False))
        parsed[fid] = {"size": size, "shapes": shapes}
    return parsed


def parse_yolo_dir(input_dir: Path, problems: list[str]) -> dict[str, dict]:
    classes_file = input_dir / "classes.txt"
    if not classes_file.is_file():
        problems.append(f"YOLO: нет {classes_file} — без него номера классов "
                        "не расшифровать (X-AnyLabeling пишет его при экспорте)")
        return {}
    try:
        classes_text = classes_file.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        problems.append(f"{classes_file.name}: не в UTF-8 (похоже, UTF-16 из "
                        "PowerShell) — пересохрани файл в UTF-8")
        return {}
    classes = [ln.strip() for ln in classes_text.splitlines() if ln.strip()]
    parsed: dict[str, dict] = {}
    for tp in sorted(input_dir.glob("*.txt")):
        if tp.name == "classes.txt":
            continue
        fid = tp.stem
        try:
            txt_lines = tp.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeDecodeError:
            problems.append(f"{fid}: {tp.name} не в UTF-8 (похоже, UTF-16 из "
                            "PowerShell) — пересохрани файл в UTF-8")
            continue
        shapes = []
        for no, ln in enumerate(txt_lines, 1):
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 5:
                problems.append(
                    f"{fid}: строка {no} — ожидалось «класс cx cy w h» (5 чисел), "
                    f"найдено {len(parts)}; сегментационный экспорт не подходит, "
                    "нужны рамки")
                continue
            try:
                ci = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                problems.append(f"{fid}: строка {no} не разбирается как числа: {ln!r}")
                continue
            if not 0 <= ci < len(classes):
                problems.append(f"{fid}: строка {no} — класс №{ci} вне "
                                f"classes.txt ({len(classes)} имён)")
                continue
            shapes.append((classes[ci], vals, False, True))
        parsed[fid] = {"size": None, "shapes": shapes}
    return parsed


def parse_coco_file(path: Path, problems: list[str],
                    notes: list[str]) -> dict[str, dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        problems.append(f"{path.name}: не читается как JSON ({e})")
        return {}
    if not isinstance(data, dict) or "images" not in data:
        problems.append(f"{path.name}: нет блока images — это точно COCO-экспорт?")
        return {}
    images = data.get("images")
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    if (not isinstance(images, list) or not isinstance(annotations, list)
            or not isinstance(categories, list)):
        problems.append(f"{path.name}: images/annotations/categories должны "
                        "быть списками — экспорт битый")
        return {}
    cats = {c.get("id"): c.get("name") for c in categories if isinstance(c, dict)}
    img_ids = {im.get("id") for im in images if isinstance(im, dict)}
    anns_by_img: dict = {}
    for i, a in enumerate(annotations, 1):
        if not isinstance(a, dict):
            problems.append(f"{path.name}: annotation №{i} — не объект: {a!r}")
            continue
        if a.get("image_id") not in img_ids:
            # молча выбросить рамку нельзя: это признак битого экспорта
            problems.append(f"{path.name}: annotation id={a.get('id')!r} "
                            f"ссылается на несуществующий image_id="
                            f"{a.get('image_id')!r}")
            continue
        anns_by_img.setdefault(a.get("image_id"), []).append(a)
    parsed: dict[str, dict] = {}
    empty: list[str] = []
    seen_fids: dict[str, str] = {}
    for im in images:
        if not isinstance(im, dict):
            problems.append(f"{path.name}: элемент images — не объект: {im!r}")
            continue
        fid = Path(str(im.get("file_name", ""))).stem
        if not fid:
            problems.append(f"{path.name}: image id={im.get('id')!r} без file_name")
            continue
        if fid in seen_fids:
            # два file_name с одним стемом (папки/расширения) молча схлопнулись
            # бы в один кадр с потерей рамок — это гейтится, не угадывается
            problems.append(f"{path.name}: два изображения дают один id кадра "
                            f"{fid!r} ({seen_fids[fid]!r} и "
                            f"{im.get('file_name')!r}) — почисти экспорт")
            continue
        seen_fids[fid] = str(im.get("file_name"))
        anns = anns_by_img.get(im.get("id"), [])
        if not anns:
            empty.append(fid)  # «чистый» и «не размечал» в COCO неотличимы
            continue
        shapes = []
        for a in anns:
            bbox = a.get("bbox")
            try:
                vals = ([float(v) for v in bbox]
                        if isinstance(bbox, list) and len(bbox) == 4 else None)
            except (TypeError, ValueError):
                vals = None
            if vals is None:
                problems.append(f"{fid}: annotation без корректного bbox: {bbox!r}")
                continue
            shapes.append((cats.get(a.get("category_id")), vals, False, False))
        w, h = im.get("width"), im.get("height")
        size = ([float(w), float(h)]
                if isinstance(w, (int, float)) and isinstance(h, (int, float))
                and w > 0 and h > 0 else None)
        parsed[fid] = {"size": size, "shapes": shapes}
    if empty:
        notes.append(
            "кадры без рамок в COCO ПРОПУЩЕНЫ (формат не отличает «чистый» от "
            f"«не размечал»): {', '.join(sorted(empty))} — чистые кадры фиксируй "
            "в родном формате X-AnyLabeling (пустой JSON кадра)")
    return parsed


def detect_format(input_path: Path, problems: list[str]) -> str | None:
    if input_path.is_file():
        return "coco"
    jsons = list(input_path.glob("*.json"))
    txts = [p for p in input_path.glob("*.txt") if p.name != "classes.txt"]
    if jsons and txts:
        problems.append(f"в {input_path} и *.json, и *.txt — какой формат "
                        "импортировать, не угадать: укажи --format явно")
        return None
    if jsons:
        return "xany"
    if txts:
        return "yolo"
    problems.append(f"в {input_path} не найдено файлов разметки (*.json / *.txt) — "
                    "проверь --input (куда сохраняет программа разметки)")
    return None


# --- сборка annotations.json ----------------------------------------------------
def assemble_frames(parsed: dict[str, dict], frames_dir: Path,
                    manifest_ids: list[str] | None, problems: list[str],
                    stats: dict) -> dict[str, dict]:
    frames: dict[str, dict] = {}
    unknown: dict[str, set[str]] = {}
    for fid in sorted(parsed):
        entry = parsed[fid]
        if manifest_ids is not None and fid not in manifest_ids:
            problems.append(
                f"{fid}: кадра нет в manifest.csv — разметка чужого/старого "
                "набора? Кадры должны быть ровно из datasets/eval_v1/frames/")
            continue
        real = image_size_px(frames_dir / f"{fid}.jpg")
        size = entry["size"] or (list(map(float, real)) if real else None)
        if size is None:
            problems.append(
                f"{fid}: неизвестен размер кадра — нет ни в разметке, ни кадра "
                f"{frames_dir / (fid + '.jpg')} (--frames)")
            continue
        if real is not None and entry["size"] is not None:
            stats["size_checked"] += 1
            if [round(v) for v in entry["size"]] != list(real):
                problems.append(
                    f"{fid}: размер в разметке {entry['size']} != реального кадра "
                    f"{list(real)} — размечена ужатая/другая копия, координаты "
                    "на оригинал не переносимы")
                continue
        defects, confusors = [], []
        for raw_label, vals, is_poly, is_norm in entry["shapes"]:
            label = normalize_label(raw_label)
            if is_norm:  # YOLO: нормированные cx cy w h -> пиксели оригинала
                W, H = size
                cx, cy, w, h = vals
                bbox = [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H]
            else:
                bbox = list(vals)
            # NaN/Infinity штатно проходят и json.loads, и float() из YOLO-строки
            if not all(math.isfinite(float(v)) for v in bbox):
                problems.append(f"{fid}: нечисловая рамка {raw_label!r}: {bbox!r}")
                continue
            bbox = [_num(v) for v in bbox]
            if bbox[2] <= 0 or bbox[3] <= 0:
                problems.append(f"{fid}: вырожденная рамка {raw_label!r}: {bbox}")
                continue
            if (bbox[0] >= size[0] or bbox[1] >= size[1]
                    or bbox[0] + bbox[2] <= 0 or bbox[1] + bbox[3] <= 0):
                problems.append(f"{fid}: рамка {raw_label!r} целиком вне кадра "
                                f"{int(size[0])}x{int(size[1])}: {bbox}")
                continue
            if is_poly:
                stats["polygons"] += 1
            if label in DEFECT_LABELS:
                defects.append({"label": label, "bbox_xywh": bbox})
                stats["defects"][label] = stats["defects"].get(label, 0) + 1
            elif label in CONFUSOR_CATEGORIES:
                confusors.append({"category": label, "bbox_xywh": bbox})
                stats["confusors"][label] = stats["confusors"].get(label, 0) + 1
            else:
                unknown.setdefault(str(raw_label), set()).add(fid)
        if not entry["shapes"]:
            stats["empty"].append(fid)
        frames[fid] = {
            "image_size_px": [int(round(size[0])), int(round(size[1]))],
            "status": "reviewed",
            "defects": defects, "confusors": confusors,
            "note": "", "proposals_resolved": []}
    for raw, fids in sorted(unknown.items()):
        problems.append(
            f"неизвестная метка {raw!r} (кадры: {', '.join(sorted(fids))}). "
            f"Дефекты: {', '.join(DEFECT_LABELS)}; конфузоры: "
            f"{', '.join(CONFUSOR_CATEGORIES)}. Переименуй класс в программе "
            "разметки и сохрани заново")
    return frames


def load_manifest_ids(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig") as f:
        return [row["frame_id"] for row in csv.DictReader(f)
                if row.get("frame_id")]


def write_annotations(frames: dict, out_path: Path) -> Path | None:
    """Записать атомарно; вернуть путь бэкапа прежнего файла (если был).
    Перед подменой файл проходит load_annotations замера: не прошло — не пишем."""
    payload = {"version": ED.ANNOTATIONS_VERSION,
               "frames": {fid: frames[fid] for fid in sorted(frames)}}
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        ED.load_annotations(tmp)
    except ValueError:
        tmp.unlink(missing_ok=True)
        raise
    bak = None
    if out_path.is_file():
        try:
            unchanged = out_path.read_text(encoding="utf-8") == text
        except Exception:  # noqa: BLE001 — нечитаемый старый файл тем более беречь
            unchanged = False
        if not unchanged:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            bak = out_path.with_name(f"annotations.bak-{stamp}.json")
            n = 2
            while bak.exists():  # два импорта в одну секунду — не затирать бэкап
                bak = out_path.with_name(f"annotations.bak-{stamp}-{n}.json")
                n += 1
            shutil.copy2(out_path, bak)
    os.replace(tmp, out_path)
    return bak


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default="datasets/eval_v1/frames",
                    help="папка с разметкой (xany: *.json; yolo: *.txt+classes.txt) "
                         "или COCO-файл; по умолчанию — папка кадров, куда "
                         "X-AnyLabeling автосохраняет свои JSONы")
    ap.add_argument("--format", choices=("auto", "xany", "yolo", "coco"),
                    default="auto")
    ap.add_argument("--frames", default="datasets/eval_v1/frames",
                    help="папка кадров-оригиналов (сверка размеров; размеры для yolo)")
    ap.add_argument("--manifest", default="datasets/eval_v1/manifest.csv")
    ap.add_argument("--out", default="datasets/eval_v1/annotations.json")
    args = ap.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Нет такого пути: {input_path}")
        return 2

    problems: list[str] = []
    notes: list[str] = []
    fmt = args.format if args.format != "auto" else detect_format(input_path,
                                                                  problems)
    if fmt in ("xany", "yolo") and not input_path.is_dir():
        problems.append(f"--format {fmt} ждёт ПАПКУ с файлами разметки, "
                        f"а {input_path} — файл")
    if fmt == "coco" and input_path.is_dir():
        problems.append("--format coco ждёт ФАЙЛ экспорта "
                        "(--input путь к .json-файлу)")

    parsed: dict[str, dict] = {}
    if not problems and fmt:
        if fmt == "xany":
            parsed = parse_xany_dir(input_path, problems)
        elif fmt == "yolo":
            parsed = parse_yolo_dir(input_path, problems)
        else:
            parsed = parse_coco_file(input_path, problems, notes)
        if not parsed and not problems:
            problems.append(f"в {input_path} не нашлось ни одного кадра с разметкой")

    manifest_ids = load_manifest_ids(Path(args.manifest))
    if manifest_ids is None:
        notes.append(f"manifest {args.manifest} не найден — сверка состава "
                     "набора не выполнена")
    if Image is None:
        notes.append("Pillow не найден — размеры кадров с оригиналами не сверены")

    stats = {"defects": {}, "confusors": {}, "polygons": 0,
             "empty": [], "size_checked": 0}
    # assemble идёт и при парс-проблемах: пользователь должен увидеть ВСЕ
    # проблемы одним списком, а не чинить их по одной между перезапусками
    # (запись всё равно гейтится пустотой problems)
    frames = assemble_frames(parsed, Path(args.frames), manifest_ids,
                             problems, stats)

    if problems:
        print(f"Импорт ({fmt or 'формат не определён'}) НЕ выполнен — "
              f"проблем: {len(problems)}; annotations.json не тронут:")
        for p in problems:
            print(f"  [!] {p}")
        for n in notes:
            print(f"  [!] {n}")
        return 2

    out_path = Path(args.out)
    try:
        bak = write_annotations(frames, out_path)
    except ValueError as e:
        print(f"Результат не прошёл валидатор замера — файл НЕ записан: {e}")
        return 2

    print(f"Формат: {fmt}; кадров импортировано: {len(frames)} (все reviewed)")
    if manifest_ids is not None:
        missing = sorted(set(manifest_ids) - set(frames))
        print(f"Покрытие: {len(frames)}/{len(manifest_ids)} кадров manifest")
        if missing:
            print(f"  ещё без разметки: {', '.join(missing)}")
    if stats["defects"]:
        print("Дефекты:   " + ", ".join(
            f"{k}: {v}" for k, v in sorted(stats["defects"].items())))
    else:
        print("Дефекты:   ни одного — проверь, так ли это на самом деле")
    if stats["confusors"]:
        print("Конфузоры: " + ", ".join(
            f"{k}: {v}" for k, v in sorted(stats["confusors"].items())))
    if stats["polygons"]:
        print(f"Полигонов сведено к осевому bbox: {stats['polygons']}")
    if stats["empty"]:
        print(f"Пустые reviewed-кадры («смотрел, дефектов нет»): "
              f"{', '.join(sorted(stats['empty']))} — проверь, что они правда чистые")
    if stats["size_checked"]:
        print(f"Размеры сверены с кадрами frames/: {stats['size_checked']} шт.")
    for n in notes:
        print(f"[!] {n}")
    if bak:
        print(f"Прежняя разметка сохранена: {bak}")
    print(f"Записано: {out_path} (схема v{ED.ANNOTATIONS_VERSION}, "
          "провалидировано load_annotations замера)")
    print("Дальше: .\\.venv\\Scripts\\python.exe scripts\\eval_detection.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
