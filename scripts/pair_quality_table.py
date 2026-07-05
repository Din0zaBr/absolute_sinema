"""Таблица качества парного прогона (PAIR_WORKFLOW.md §«Как понять качество»).

Совмещает JSON-отчёты `--pairs-dir` с журналом сбора (DATASET.md §4) и строит
по-парную таблицу + сводку recall. «Правда» на этом этапе — сам факт дефекта:
каждая пара датасета снята вокруг одной реальной ямы (журнал, поле type).

Семантика (важно, чтобы не занижать recall):
  «яма найдена в виде» = select_status != no_pothole, т.е. детектор выдал
  ХОТЬ ОДНУ pothole-детекцию в этом виде. per_view.pothole_found из JSON —
  ДРУГОЕ: «выбрана одна доминирующая яма для сопоставления видов»; при ≥2
  близких по уверенности ямах он честно false (ambiguous_multiple_potholes),
  хотя ямы найдены.

Сцены: соседние ямы сборщик снимает одним общим кадром, поэтому 48 пар —
это меньше уникальных сцен (колонка scene журнала, пишет ingest_local_pairs).
Пары одной сцены — одни и те же фото, признак «найдено» у них общий; честный
recall считается и по парам, и по сценам.

Метрики (цель прототипа — recall ≥ 80%):
  recall_any    — яма найдена хотя бы в одном виде пары;
  recall_both   — яма найдена в обоих видах;
  scene_recall  — то же по уникальным сценам;
  matched_rate  — доля пар, где доминирующие ямы двух видов сопоставлены (F1);
  fused_rate    — доля пар со слитой метрикой; требует МАСШТАБА в обоих
                  видах (эталон в кадре), без него честно 0.

Лишние детекции здесь только подсчитываются (extra_defects) — реальный это
дефект или ложный, по JSON не определить, нужна визуальная проверка.

Запуск:
  .\\.venv\\Scripts\\python.exe scripts\\pair_quality_table.py --outputs outputs_pairs
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

TABLE_FIELDS = [
    "pair_id", "scene", "gt_type", "mode",
    "front_detected", "back_detected", "detected_any", "detected_both",
    "front_select", "back_select", "matched", "fused",
    "n_defects", "extra_defects", "classes", "max_pothole_conf",
    "scale_available", "depth_buckets",
]


def _load_reports(out_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(out_dir.glob("*_pair.json")):
        try:
            with open(path, encoding="utf-8") as f:
                reports.append(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [ПРОПУСК] {path.name}: {exc}", file=sys.stderr)
    return reports


def _load_journal(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {row["id"]: row for row in csv.DictReader(f) if row.get("id")}


def _journal_row(journal: dict[str, dict], pair_id: str) -> dict:
    """Строка журнала для pair_id. Журнал ingest'а нумерует ям zero-padded
    (`001`), а `--pairs-dir` на «сырой» папке с папками `1`,`2` даёт pair_id
    без нулей — пробуем оба написания, иначе gt_type/scene молча теряются
    (ревью 2026-07-05)."""
    return journal.get(pair_id) or journal.get(pair_id.zfill(3)) or {}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # cp1251-консоль Windows
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs", default="outputs_pairs", help="папка с *_pair.json")
    ap.add_argument("--journal", default="datasets/journal.csv", help="журнал сбора (§4)")
    ap.add_argument("--csv", default=None,
                    help="куда писать таблицу (по умолчанию <outputs>/quality_table.csv)")
    args = ap.parse_args()

    out_dir = Path(args.outputs)
    reports = _load_reports(out_dir)
    if not reports:
        print(f"В {out_dir} нет *_pair.json — сначала прогони --pairs-dir.")
        return 1
    journal = _load_journal(Path(args.journal))
    has_scenes = any(_journal_row(journal, str(r.get("pair_id", ""))).get("scene")
                     for r in reports)

    rows = []
    for r in reports:
        pid = str(r.get("pair_id", ""))
        jrow = _journal_row(journal, pid)
        fusion = r.get("fusion") or {}
        # per_view пайплайн всегда строит по позиции [вид A=front, вид B=back]
        # (pipeline.analyze_pair). Берём ПО ПОЗИЦИИ, а не по стему имени: имена
        # могут быть алиасами (1/2, a/b, before/after) — тогда стемы 'front'/'back'
        # не совпадут и recall занизится до нуля (ревью 2026-07-05).
        pv = fusion.get("per_view") or []
        fs = (pv[0] if len(pv) > 0 else {}).get("select_status", "")
        bs = (pv[1] if len(pv) > 1 else {}).get("select_status", "")
        # найдена = есть хоть одна pothole-детекция (см. док модуля);
        # отсутствие per_view в отчёте считаем «не найдено», не роняем таблицу
        front = bool(fs) and fs != "no_pothole"
        back = bool(bs) and bs != "no_pothole"
        defects = r.get("defects") or []
        potholes = [d for d in defects if d.get("class") == "pothole"]
        confs = [d.get("confidence") for d in potholes if d.get("confidence") is not None]
        # бакет глубины пайплайн кладёт в defect["metric"]["depth_bucket"]
        # (pipeline.py), НЕ в defect["depth"]["bucket"] — иначе колонка всегда
        # пустая даже при --depth (ревью 2026-07-05).
        buckets = sorted({(d.get("metric") or {}).get("depth_bucket") or "-"
                          for d in defects} - {"-"})
        rows.append({
            "pair_id": pid,
            "scene": jrow.get("scene") or pid,
            "gt_type": jrow.get("type", ""),
            "mode": r.get("mode", ""),
            "front_detected": front,
            "back_detected": back,
            "detected_any": front or back,
            "detected_both": front and back,
            "front_select": fs,
            "back_select": bs,
            "matched": bool(fusion.get("matched")),
            "fused": bool(fusion.get("fused")),
            "n_defects": len(defects),
            "extra_defects": max(0, len(defects) - len(potholes)),
            "classes": " ".join(sorted({d.get("class", "?") for d in defects})),
            "max_pothole_conf": max(confs) if confs else "",
            "scale_available": bool((r.get("scale") or {}).get("available")),
            "depth_buckets": " ".join(buckets),
        })

    csv_path = Path(args.csv) if args.csv else out_dir / "quality_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    n_any = sum(r["detected_any"] for r in rows)
    n_both = sum(r["detected_both"] for r in rows)
    n_matched = sum(r["matched"] for r in rows)
    n_fused = sum(r["fused"] for r in rows)
    n_scale = sum(r["scale_available"] for r in rows)
    n_extra = sum(r["extra_defects"] for r in rows)
    n_amb = sum(1 for r in rows
                if "ambiguous" in r["front_select"] or "ambiguous" in r["back_select"])
    missed = [r["pair_id"] for r in rows if not r["detected_any"]]

    print(f"Пар: {n}   (таблица: {csv_path})")
    print(f"recall_any  (яма хотя бы в одном виде): {n_any}/{n} = {n_any / n:.0%}")
    print(f"recall_both (яма в обоих видах):        {n_both}/{n} = {n_both / n:.0%}")

    if has_scenes:
        scenes: dict[str, list[dict]] = {}
        for r in rows:
            scenes.setdefault(r["scene"], []).append(r)
        ns = len(scenes)
        ns_any = sum(any(r["detected_any"] for r in g) for g in scenes.values())
        ns_both = sum(any(r["detected_both"] for r in g) for g in scenes.values())
        missed_scenes = sorted(s for s, g in scenes.items()
                               if not any(r["detected_any"] for r in g))
        print(f"Сцен (уникальных кадров): {ns}")
        print(f"scene_recall_any:  {ns_any}/{ns} = {ns_any / ns:.0%}")
        print(f"scene_recall_both: {ns_both}/{ns} = {ns_both / ns:.0%}")
    else:
        missed_scenes = missed
        print("В журнале нет колонки scene — по-сценные метрики пропущены "
              "(перегенерируй журнал: scripts/ingest_local_pairs.py)")

    print(f"matched_rate (виды сопоставлены):       {n_matched}/{n} = {n_matched / n:.0%}"
          f"   (несопоставление из-за >=2 близких ям: {n_amb} пар)")
    print(f"fused_rate   (метрики слиты, нужен масштаб в обоих видах): "
          f"{n_fused}/{n} = {n_fused / n:.0%}")
    print(f"масштаб доступен: {n_scale}/{n}; не-pothole детекций суммарно: {n_extra}")
    if missed:
        print(f"Яма не найдена вообще: пары {', '.join(missed)}"
              + (f" (сцены {', '.join(missed_scenes)})" if has_scenes else ""))
        print("  -> кандидаты в datasets/local_pairs_hard и в приоритет разметки дообучения")
    return 0


if __name__ == "__main__":
    sys.exit(main())
