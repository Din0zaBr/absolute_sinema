"""Подготовка RDD2022 (Pascal VOC XML) к дообучению: конвертация в YOLO-формат.

Запуск:  python scripts/prepare_rdd2022.py datasets/RDD2022_Czech.zip
         python scripts/prepare_rdd2022.py datasets/RDD2022_Czech  (распакованный)

Что делает:
  1) при необходимости распаковывает архив;
  2) находит train/{images, annotations/xmls} внутри (страновая структура RDD2022);
  3) конвертирует VOC-боксы в YOLO txt с маппингом классов ПОД ВЕСА rezzzq:
     0=D00, 1=D10, 2=D20, 3=D40, 4=Repair — прочие метки (D43, D44, D50, D01…)
     пропускаются с подсчётом; картинки без валидных меток становятся фоном;
  4) делит train на train/val (90/10, сид 42 — воспроизводимо);
  5) пишет data.yaml для scripts/finetune_rdd2022.py и ultralytics val.

Лицензия данных: CC BY-SA 4.0 (см. дизайн §9).
"""
from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Порядок классов как в весах rezzzq/yolo12s-road-damage-rdd2022.
CLASS_TO_ID = {"D00": 0, "D10": 1, "D20": 2, "D40": 3, "Repair": 4}
CLASS_NAMES = list(CLASS_TO_ID)
VAL_FRACTION = 0.10


def _find_train_dir(root: Path) -> Path | None:
    """Найти каталог train с images/ и annotations/xmls/ в любой вложенности."""
    for cand in root.rglob("train"):
        if (cand / "images").is_dir() and (cand / "annotations" / "xmls").is_dir():
            return cand
    return None


def _zip_signature(src: Path) -> str:
    """Подпись архива для штампа распаковки: размер + mtime_ns.

    Без чтения 245 МБ содержимого; любое обновление/перекачивание архива
    меняет подпись — ошибка возможна только в безопасную сторону
    (лишняя переспаковка)."""
    st = src.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _ensure_extracted(src: Path) -> Path:
    """Для zip — путь к ПОЛНОСТЬЮ распакованной копии; иначе src как есть.

    Прерванная или устаревшая распаковка не должна молча становиться «готовым»
    источником — иначе датасет строится усечённым/неактуальным при коде
    возврата 0 (репро агентов-скептиков, 2026-07-02 и 2026-07-03). Поэтому:
      • распаковка атомарна: во временную папку *.extracting + rename;
      • ПОСЛЕ успешного rename рядом с архивом пишется штамп <имя>.zip.extracted
        с подписью архива — существующая папка принимается ТОЛЬКО при совпадении
        штампа с текущим архивом. Счёт файлов такое не ловит: подменённый архив
        с тем же/меньшим числом файлов, legacy-копия с усечённым последним
        файлом, недостача, замаскированная посторонним Thumbs.db;
      • сверка числа файлов с архивом остаётся второй линией защиты — на случай
        ручного удаления файлов из распакованной копии при целом штампе.
    """
    if src.suffix.lower() != ".zip":
        return src
    extract_dir = src.with_suffix("")
    stamp = src.with_name(src.name + ".extracted")
    signature = _zip_signature(src)
    with zipfile.ZipFile(src) as z:
        n_members = sum(1 for n in z.namelist() if not n.endswith("/"))
        if extract_dir.exists():
            stale = None
            if (not stamp.exists()
                    or stamp.read_text(encoding="utf-8").strip() != signature):
                stale = "архив изменился или распаковка не подтверждена штампом"
            else:
                n_disk = sum(1 for p in extract_dir.rglob("*") if p.is_file())
                if n_disk < n_members:
                    stale = f"на диске {n_disk} файлов из {n_members} в архиве"
            if stale:
                print(f"[!] {extract_dir.name}: {stale} — распаковываю заново.")
                shutil.rmtree(extract_dir)
        if not extract_dir.exists():
            print(f"Распаковка {src.name} ...")
            tmp_dir = extract_dir.with_name(extract_dir.name + ".extracting")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            z.extractall(tmp_dir)
            tmp_dir.rename(extract_dir)
            stamp.write_text(signature, encoding="utf-8")
    return extract_dir


def _voc_to_yolo_lines(xml_path: Path, skipped: Counter) -> list[str] | None:
    """Строки YOLO-аннотации из VOC XML. None, если XML не разобрался."""
    try:
        xml = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    size = xml.find("size")
    if size is None:
        return None
    w = float(size.findtext("width") or 0)
    h = float(size.findtext("height") or 0)
    if w <= 0 or h <= 0:
        return None

    lines = []
    for obj in xml.iter("object"):
        name = (obj.findtext("name") or "").strip()
        cls_id = CLASS_TO_ID.get(name)
        if cls_id is None:
            skipped[name] += 1
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x0 = max(0.0, float(bb.findtext("xmin") or 0))
        y0 = max(0.0, float(bb.findtext("ymin") or 0))
        x1 = min(w, float(bb.findtext("xmax") or 0))
        y1 = min(h, float(bb.findtext("ymax") or 0))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: prepare_rdd2022.py <RDD2022_*.zip | распакованная папка>")
        return 1
    src = Path(sys.argv[1])
    if not src.exists():
        print(f"Не найдено: {src}")
        return 1

    src = _ensure_extracted(src)

    train_dir = _find_train_dir(src)
    if train_dir is None:
        print(f"Не нашёл train/{{images, annotations/xmls}} внутри {src}")
        return 1
    print(f"Источник: {train_dir}")

    out = ROOT / "datasets" / f"{src.name}_yolo"
    if out.exists():
        # train/val-назначение позиционное (sorted + сид): при изменившемся
        # наборе картинка может сменить split, а старая копия остаться в другом —
        # утечка train→val. Выход всегда строится с нуля.
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = sorted((train_dir / "images").glob("*.jpg"))
    rng = np.random.default_rng(42)
    val_mask = rng.random(len(images)) < VAL_FRACTION

    skipped: Counter = Counter()
    kept = Counter()
    n_bg = 0
    n_bad_xml = 0
    for img, is_val in zip(images, val_mask):
        xml_path = train_dir / "annotations" / "xmls" / f"{img.stem}.xml"
        lines = _voc_to_yolo_lines(xml_path, skipped) if xml_path.exists() else []
        if lines is None:
            n_bad_xml += 1      # битый/усечённый XML — не молчим (см. итог ниже)
            lines = []
        split = "val" if is_val else "train"
        shutil.copy2(img, out / "images" / split / img.name)
        (out / "labels" / split / f"{img.stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8")
        if lines:
            for ln in lines:
                kept[CLASS_NAMES[int(ln.split()[0])]] += 1
        else:
            n_bg += 1

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.as_posix()}\n"
        "train: images/train\nval: images/val\n"
        f"names: {dict(enumerate(CLASS_NAMES))}\n",
        encoding="utf-8")

    n_val = int(val_mask.sum())
    print(f"Готово: {out}")
    print(f"  картинок: {len(images)} (train {len(images) - n_val} / val {n_val}), фоновых: {n_bg}")
    print(f"  боксов по классам: {dict(kept)}")
    if n_bad_xml:
        print(f"  [!] битых XML (ParseError/нет size): {n_bad_xml} — "
              "их картинки учтены ФОНОМ; проверьте целостность датасета")
    if skipped:
        print(f"  пропущено меток вне таксономии rezzzq: {dict(skipped)}")
    print(f"  data.yaml: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
