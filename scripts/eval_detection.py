"""Харнесс P0: precision/recall детекции + доля FP по категориям (IMPROVEMENTS.md).

Сверяет отчёты движка (`*.json` из `python -m road_defect.cli --input
datasets/eval_v1/frames`) с РУЧНОЙ разметкой `datasets/eval_v1/annotations.json`
(создаётся build_eval_annotator.py). Модели не поднимает и ничего не детектит —
чистый замер по готовым файлам, воспроизводим и тестируем без сети.

Честность замера:
  • считаются только кадры со status="reviewed" (черновая разметка — не GT);
  • reviewed-кадр БЕЗ отчёта движка — ошибка (непрогнанный кадр не равен
    «нулю детекций»), обходится явным --allow-missing;
  • misclass (яма-предсказание на GT-заплатке) — это FP с категорией gt_patch,
    а не «почти попал».

Категории FP: gt_<класс> (misclass по разметке), car / off_road / shadow_curb /
other (по confusor-боксам разметки), duplicate (второй бокс на ту же GT-яму),
edge (мелочь у кромки кадра, эвристика v0), unknown.

Запуск:
  .\\.venv\\Scripts\\python.exe scripts\\eval_detection.py
  # подробности и протокол — docs/EVAL.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_defect import config  # noqa: E402

DEFECT_LABELS = sorted(set(config.DEFECT_CLASSES.values()))
CONFUSOR_CATEGORIES = ("car", "off_road", "shadow_curb", "other")
ANNOTATIONS_VERSION = 1


# --- геометрия (bbox = [x, y, w, h] в пикселях оригинала) --------------------
def iou_xywh(a: list, b: list) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def contained_frac(inner: list, outer: list) -> float:
    """Доля площади inner, накрытая outer (маленький бокс «в» большом)."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    x0, y0 = max(ix, ox), max(iy, oy)
    x1, y1 = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = iw * ih
    return inter / area if area > 0 else 0.0


def near_edge_small(bbox: list, image_wh: tuple,
                    band_frac: float, size_frac: float) -> bool:
    """Эвристика v0 «мелочь у кромки» (IMPROVEMENTS: боксы 42–62px в углах)."""
    W, H = image_wh
    x, y, w, h = bbox
    m = min(W, H)
    band = band_frac * m
    touches = (x <= band or y <= band
               or (W - (x + w)) <= band or (H - (y + h)) <= band)
    return touches and max(w, h) <= size_frac * m


# --- разметка ----------------------------------------------------------------
def load_annotations(path: Path) -> dict:
    """Загрузить и провалидировать annotations.json. Ошибки — громко: грязный
    GT хуже отсутствия GT (ложные метрики убедительнее честного «не мерили»)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != ANNOTATIONS_VERSION:
        raise ValueError(f"annotations.json: ожидалась version="
                         f"{ANNOTATIONS_VERSION}, найдено {data.get('version')!r}")
    frames = data.get("frames")
    if not isinstance(frames, dict) or not frames:
        raise ValueError("annotations.json: пустой/некорректный блок frames")

    def _check_bbox(fid: str, kind: str, bbox) -> None:
        ok = (isinstance(bbox, list) and len(bbox) == 4
              and all(isinstance(v, (int, float)) for v in bbox)
              and bbox[2] > 0 and bbox[3] > 0)
        if not ok:
            raise ValueError(f"{fid}: некорректный bbox_xywh у {kind}: {bbox!r}")

    for fid, fr in frames.items():
        size = fr.get("image_size_px")
        if (not isinstance(size, list) or len(size) != 2
                or not all(isinstance(v, (int, float)) and v > 0 for v in size)):
            raise ValueError(f"{fid}: некорректный image_size_px: {size!r}")
        if fr.get("status") not in ("reviewed", "draft"):
            raise ValueError(f"{fid}: status должен быть reviewed|draft, "
                             f"найдено {fr.get('status')!r}")
        for d in fr.get("defects", []):
            if d.get("label") not in DEFECT_LABELS:
                raise ValueError(f"{fid}: неизвестный label дефекта "
                                 f"{d.get('label')!r} (допустимо: {DEFECT_LABELS})")
            _check_bbox(fid, "дефекта", d.get("bbox_xywh"))
        for c in fr.get("confusors", []):
            if c.get("category") not in CONFUSOR_CATEGORIES:
                raise ValueError(f"{fid}: неизвестная категория конфузора "
                                 f"{c.get('category')!r} "
                                 f"(допустимо: {CONFUSOR_CATEGORIES})")
            _check_bbox(fid, "конфузора", c.get("bbox_xywh"))
    return frames


def load_predictions(report_path: Path) -> list[dict]:
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    preds = []
    for d in rep.get("defects", []):
        preds.append({"label": d.get("class"),
                      "confidence": float(d.get("confidence") or 0.0),
                      "bbox": [float(v) for v in d["bbox_px"]]})
    return preds


# --- сопоставление одного кадра ----------------------------------------------
def evaluate_frame(frame: dict, preds: list[dict], iou_thr: float = 0.5,
                   contain_thr: float = 0.5, edge_band_frac: float = 0.02,
                   edge_size_frac: float = 0.06) -> dict:
    """Класс-осознанное жадное сопоставление (по убыванию conf, IoU >= порога,
    только одинаковый label) + категоризация несматченных предсказаний.

    Возвращает {'tp': [(label, pred)], 'fp': [(pred, category)],
    'fn': [gt_defect]} — чистая функция, тестируется на синтетике."""
    W, H = frame["image_size_px"]
    gts = list(frame.get("defects", []))
    confusors = list(frame.get("confusors", []))
    consumed = [False] * len(gts)

    tp: list[tuple[str, dict]] = []
    unmatched: list[dict] = []
    for p in sorted(preds, key=lambda q: -q["confidence"]):
        best_i, best_iou = -1, 0.0
        for i, g in enumerate(gts):
            if consumed[i] or g["label"] != p["label"]:
                continue
            v = iou_xywh(p["bbox"], g["bbox_xywh"])
            if v >= iou_thr and v > best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            consumed[best_i] = True
            tp.append((p["label"], p))
        else:
            unmatched.append(p)

    fp: list[tuple[dict, str]] = []
    for p in unmatched:
        category = None
        # 1) второй бокс на уже закрытую GT того же класса — дубль
        for i, g in enumerate(gts):
            if (consumed[i] and g["label"] == p["label"]
                    and iou_xywh(p["bbox"], g["bbox_xywh"]) >= iou_thr):
                category = "duplicate"
                break
        # 2) misclass: накрыл GT-дефект ДРУГОГО класса (яма на заплатке).
        #    Квалификация по IoU ИЛИ по вложенности бокса в GT: модель типично
        #    боксует ЧАСТЬ длинной заплатки — IoU мал, но бокс целиком внутри
        #    (ревью 2026-07-07: такие FP утекали в unknown). При нескольких
        #    кандидатах — максимальное перекрытие, не порядок разметки.
        if category is None:
            best = (0.0, None)
            for g in gts:
                if g["label"] == p["label"]:
                    continue
                v = iou_xywh(p["bbox"], g["bbox_xywh"])
                frac = contained_frac(p["bbox"], g["bbox_xywh"])
                score = max(v if v >= iou_thr else 0.0,
                            frac if frac >= contain_thr else 0.0)
                if score > best[0]:
                    best = (score, f"gt_{g['label']}")
            category = best[1]
        # 3) конфузор разметки: бокс преимущественно внутри области
        if category is None:
            best = (0.0, None)
            for c in confusors:
                frac = contained_frac(p["bbox"], c["bbox_xywh"])
                if frac >= contain_thr and frac > best[0]:
                    best = (frac, c["category"])
            category = best[1]
        # 4) мелочь у кромки кадра
        if category is None and near_edge_small(
                p["bbox"], (W, H), edge_band_frac, edge_size_frac):
            category = "edge"
        fp.append((p, category or "unknown"))

    fn = [g for i, g in enumerate(gts) if not consumed[i]]
    return {"tp": tp, "fp": fp, "fn": fn}


# --- агрегация и отчёт ---------------------------------------------------------
def aggregate(results: dict[str, dict]) -> dict:
    """results: frame_id -> evaluate_frame(...). Возвращает сводку по классам,
    матрицу FP-категорий и плоские списки для CSV."""
    per_label = {lab: {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
                 for lab in DEFECT_LABELS}
    fp_by_cat: dict[str, dict[str, int]] = {}
    fp_rows, fn_rows = [], []
    for fid, r in results.items():
        for lab, _p in r["tp"]:
            per_label[lab]["tp"] += 1
            per_label[lab]["gt"] += 1
        for g in r["fn"]:
            per_label[g["label"]]["fn"] += 1
            per_label[g["label"]]["gt"] += 1
            fn_rows.append({"frame_id": fid, "label": g["label"],
                            **dict(zip("xywh", g["bbox_xywh"]))})
        for p, cat in r["fp"]:
            lab = p["label"] if p["label"] in per_label else "_foreign"
            per_label.setdefault(lab, {"gt": 0, "tp": 0, "fp": 0, "fn": 0})
            per_label[lab]["fp"] += 1
            fp_by_cat.setdefault(lab, {})
            fp_by_cat[lab][cat] = fp_by_cat[lab].get(cat, 0) + 1
            fp_rows.append({"frame_id": fid, "label": p["label"],
                            "confidence": round(p["confidence"], 3),
                            "category": cat, **dict(zip("xywh", p["bbox"]))})
    total = {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
    for row in per_label.values():
        for k in total:
            total[k] += row[k]
    return {"per_label": per_label, "total": total,
            "fp_by_cat": fp_by_cat, "fp_rows": fp_rows, "fn_rows": fn_rows}


def _ratio(num: int, den: int) -> str:
    return f"{num / den:.3f}" if den else "  —  "


def coverage_report(frames: dict, manifest_path: Path | None) -> None:
    """Сверка разметки с manifest.csv: тихое усыхание набора (удалённые из
    annotations.json кадры) должно быть ВИДНО в каждом замере (ревью 2026-07-07:
    иначе «чистка под метрику» неотличима от честного прогона)."""
    drafts = sorted(fid for fid, fr in frames.items() if fr["status"] == "draft")
    if drafts:
        print(f"draft-кадры (в замер НЕ идут): {', '.join(drafts)}")
    if manifest_path is None or not manifest_path.is_file():
        print("[!] manifest.csv не найден — сверка покрытия набора невозможна "
              "(полон ли замер, неизвестно)")
        return
    with manifest_path.open(encoding="utf-8-sig") as f:
        manifest_ids = [row["frame_id"] for row in csv.DictReader(f)
                        if row.get("frame_id")]
    missing = sorted(set(manifest_ids) - set(frames))
    ghosts = sorted(set(frames) - set(manifest_ids))
    print(f"Покрытие набора: manifest {len(manifest_ids)} / в разметке "
          f"{len(set(manifest_ids) & set(frames))} / reviewed "
          f"{sum(1 for fr in frames.values() if fr['status'] == 'reviewed')}")
    if missing:
        print(f"[!] кадры manifest БЕЗ разметки (выпали из замера): "
              f"{', '.join(missing)}")
    if ghosts:
        print(f"[!] кадры разметки вне manifest (чужой/старый набор?): "
              f"{', '.join(ghosts)}")


def print_report(agg: dict, n_frames: int, skipped_draft: int) -> None:
    print(f"Кадров в замере: {n_frames} (reviewed); пропущено draft: {skipped_draft}")
    print(f"{'класс':22} {'GT':>4} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'precision':>9} {'recall':>7}")
    print("-" * 60)
    rows = {**agg["per_label"], "ИТОГО": agg["total"]}
    for lab, r in rows.items():
        if lab != "ИТОГО" and r["gt"] == 0 and r["fp"] == 0:
            continue                      # класс без данных не шумит в таблице
        print(f"{lab:22} {r['gt']:>4} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{_ratio(r['tp'], r['tp'] + r['fp']):>9} "
              f"{_ratio(r['tp'], r['gt']):>7}")
    if agg["fp_by_cat"]:
        print("\nFP по категориям (класс предсказания -> откуда взялся):")
        for lab, cats in sorted(agg["fp_by_cat"].items()):
            body = ", ".join(f"{c}: {n}" for c, n in
                             sorted(cats.items(), key=lambda kv: -kv[1]))
            print(f"  {lab:20} {body}")


def write_csvs(agg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "eval_summary.csv").open("w", newline="",
                                             encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "gt", "tp", "fp", "fn", "precision", "recall"])
        for lab, r in {**agg["per_label"], "total": agg["total"]}.items():
            w.writerow([lab, r["gt"], r["tp"], r["fp"], r["fn"],
                        _ratio(r["tp"], r["tp"] + r["fp"]).strip(),
                        _ratio(r["tp"], r["gt"]).strip()])
    for name, rows, fields in (
            ("eval_fp.csv", agg["fp_rows"],
             ["frame_id", "label", "confidence", "category", "x", "y", "w", "h"]),
            ("eval_fn.csv", agg["fn_rows"],
             ["frame_id", "label", "x", "y", "w", "h"])):
        with (out_dir / name).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--annotations", default="datasets/eval_v1/annotations.json")
    ap.add_argument("--manifest", default=None,
                    help="manifest.csv набора для сверки покрытия "
                         "(по умолчанию — рядом с annotations.json)")
    ap.add_argument("--reports", default="outputs_eval/reports",
                    help="папка с *.json отчётами движка по кадрам eval-набора")
    ap.add_argument("--out", default="outputs_eval",
                    help="куда писать eval_summary/eval_fp/eval_fn CSV")
    ap.add_argument("--iou", type=float, default=0.5, help="порог IoU матчинга")
    ap.add_argument("--contain", type=float, default=0.5,
                    help="порог доли бокса внутри конфузора")
    ap.add_argument("--edge-band-frac", type=float, default=0.02)
    ap.add_argument("--edge-size-frac", type=float, default=0.06)
    ap.add_argument("--allow-missing", action="store_true",
                    help="reviewed-кадр без отчёта = 0 детекций (по умолчанию — ошибка)")
    args = ap.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.is_file():
        print(f"Нет разметки: {ann_path}\n"
              "Сначала разметь eval-набор: scripts/build_eval_annotator.py "
              "(протокол — docs/EVAL.md)")
        return 2
    try:
        frames = load_annotations(ann_path)
    except ValueError as e:
        print(f"Разметка не прошла валидацию: {e}")
        return 2

    reports_dir = Path(args.reports)
    reviewed = {fid: fr for fid, fr in frames.items()
                if fr["status"] == "reviewed"}
    skipped_draft = len(frames) - len(reviewed)
    if not reviewed:
        print("Нет ни одного кадра со status='reviewed' — мерить нечего "
              "(черновая разметка в метрики не идёт).")
        return 2

    missing = [fid for fid in reviewed
               if not (reports_dir / f"{fid}.json").is_file()]
    if missing and not args.allow_missing:
        print(f"Нет отчётов движка для {len(missing)} reviewed-кадров "
              f"(например {missing[0]}.json) в {reports_dir}.\n"
              "Непрогнанный кадр НЕ равен «нулю детекций». Прогони движок:\n"
              "  python -m road_defect.cli --input datasets/eval_v1/frames "
              "--output outputs_eval/reports\n"
              "либо явно согласись считать их пустыми: --allow-missing")
        return 2

    results = {}
    for fid, fr in sorted(reviewed.items()):
        rp = reports_dir / f"{fid}.json"
        preds = load_predictions(rp) if rp.is_file() else []
        results[fid] = evaluate_frame(
            fr, preds, iou_thr=args.iou, contain_thr=args.contain,
            edge_band_frac=args.edge_band_frac,
            edge_size_frac=args.edge_size_frac)
    if missing:
        print(f"[!] reviewed-кадры без отчёта посчитаны как 0 детекций "
              f"(--allow-missing): {', '.join(missing)}. Их GT ушли в FN "
              "(recall честный), но их несостоявшиеся FP в замер НЕ попали — "
              "precision может быть ЗАВЫШЕН; сравнение «до/после» с разным "
              "составом отчётов невалидно.")

    params = {"iou": args.iou, "contain": args.contain,
              "edge_band_frac": args.edge_band_frac,
              "edge_size_frac": args.edge_size_frac,
              "frames_reviewed": len(reviewed),
              "skipped_draft": skipped_draft,
              "missing_reports": missing}
    print("Параметры замера: " + ", ".join(
        f"{k}={v}" for k, v in params.items() if not isinstance(v, list)))
    manifest_path = (Path(args.manifest) if args.manifest
                     else ann_path.parent / "manifest.csv")
    coverage_report(frames, manifest_path)

    agg = aggregate(results)
    print_report(agg, len(reviewed), skipped_draft)
    out_dir = Path(args.out)
    write_csvs(agg, out_dir)
    # Штамп параметров рядом с CSV: два прогона с разными --iou не должны
    # выглядеть одинаковыми артефактами (ревью 2026-07-07).
    (out_dir / "eval_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nCSV: {out_dir / 'eval_summary.csv'} (+ eval_fp.csv, eval_fn.csv, "
          "eval_params.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
