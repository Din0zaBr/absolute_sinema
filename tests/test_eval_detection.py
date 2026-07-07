"""Харнесс P0 (scripts/eval_detection.py): сопоставление, категории FP,
валидация разметки, дисциплина reviewed/missing. Чистая математика — без моделей.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "eval_detection", ROOT / "scripts" / "eval_detection.py")
ed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ed)


def _frame(size=(1000, 800), defects=(), confusors=(), status="reviewed"):
    return {"image_size_px": list(size), "status": status,
            "defects": list(defects), "confusors": list(confusors)}


def _gt(label, x, y, w, h):
    return {"label": label, "bbox_xywh": [x, y, w, h]}


def _conf(category, x, y, w, h):
    return {"category": category, "bbox_xywh": [x, y, w, h]}


def _pred(label, x, y, w, h, conf=0.8):
    return {"label": label, "confidence": conf, "bbox": [x, y, w, h]}


# --- сопоставление одного кадра ----------------------------------------------
def test_exact_match_is_tp_and_only_tp():
    fr = _frame(defects=[_gt("pothole", 100, 100, 200, 150)])
    r = ed.evaluate_frame(fr, [_pred("pothole", 105, 102, 195, 150)])
    assert [lab for lab, _ in r["tp"]] == ["pothole"]
    assert r["fp"] == [] and r["fn"] == []


def test_missed_gt_is_fn_and_stray_pred_is_fp_unknown():
    fr = _frame(defects=[_gt("pothole", 100, 100, 200, 150)])
    r = ed.evaluate_frame(fr, [_pred("pothole", 600, 500, 80, 60)])
    assert r["tp"] == []
    assert [g["label"] for g in r["fn"]] == ["pothole"]
    assert [cat for _, cat in r["fp"]] == ["unknown"]


def test_negative_frame_all_preds_are_fp():
    # Кадр без дефектов и конфузоров — легитимный негатив.
    r = ed.evaluate_frame(_frame(), [_pred("pothole", 10, 10, 50, 50)])
    assert r["tp"] == [] and r["fn"] == []
    assert len(r["fp"]) == 1


def test_class_aware_matching_pothole_on_patch_is_misclass_fp():
    # Яма-предсказание на GT-заплатке: FP категории gt_patch + FN по заплатке —
    # ровно категория «заплатка» из карты FP (IMPROVEMENTS.md).
    fr = _frame(defects=[_gt("patch", 100, 100, 200, 150)])
    r = ed.evaluate_frame(fr, [_pred("pothole", 100, 100, 200, 150)])
    assert r["tp"] == []
    assert [cat for _, cat in r["fp"]] == ["gt_patch"]
    assert [g["label"] for g in r["fn"]] == ["patch"]


def test_duplicate_pred_on_same_gt_is_fp_duplicate():
    fr = _frame(defects=[_gt("pothole", 100, 100, 200, 150)])
    preds = [_pred("pothole", 100, 100, 200, 150, conf=0.9),
             _pred("pothole", 110, 105, 190, 145, conf=0.6)]
    r = ed.evaluate_frame(fr, preds)
    assert len(r["tp"]) == 1
    assert [cat for _, cat in r["fp"]] == ["duplicate"]


def test_greedy_matching_prefers_higher_confidence():
    # Две почти совпадающие детекции: TP достаётся более уверенной.
    fr = _frame(defects=[_gt("pothole", 100, 100, 200, 150)])
    preds = [_pred("pothole", 100, 100, 200, 150, conf=0.5),
             _pred("pothole", 101, 101, 199, 149, conf=0.9)]
    r = ed.evaluate_frame(fr, preds)
    (lab, p), = r["tp"]
    assert p["confidence"] == 0.9


def test_confusor_containment_categorizes_fp():
    # Бокс «ямы» на колесе припаркованной машины -> категория car.
    fr = _frame(confusors=[_conf("car", 50, 50, 400, 300)])
    r = ed.evaluate_frame(fr, [_pred("pothole", 100, 100, 80, 60)])
    assert [cat for _, cat in r["fp"]] == ["car"]


def test_pred_half_out_of_confusor_not_categorized_as_it():
    # Меньше половины бокса внутри конфузора -> не его категория (unknown).
    fr = _frame(confusors=[_conf("car", 0, 0, 100, 100)])
    r = ed.evaluate_frame(fr, [_pred("pothole", 80, 0, 100, 100)])
    assert [cat for _, cat in r["fp"]] == ["unknown"]


def test_small_corner_box_is_edge_category():
    fr = _frame(size=(1152, 2560))
    # 50x50 у самого угла (IMPROVEMENTS: «боксы 42–62px в углах»)
    r = ed.evaluate_frame(fr, [_pred("pothole", 2, 2, 50, 50)])
    assert [cat for _, cat in r["fp"]] == ["edge"]


def test_large_box_touching_border_is_not_edge():
    fr = _frame(size=(1152, 2560))
    r = ed.evaluate_frame(fr, [_pred("pothole", 0, 0, 500, 400)])
    assert [cat for _, cat in r["fp"]] == ["unknown"]


def test_iou_threshold_boundary():
    # IoU ровно на пороге принимается (>=), чуть ниже — нет.
    fr = _frame(defects=[_gt("pothole", 0, 0, 100, 100)])
    exact = ed.evaluate_frame(fr, [_pred("pothole", 0, 0, 100, 100)], iou_thr=1.0)
    assert len(exact["tp"]) == 1
    shifted = ed.evaluate_frame(fr, [_pred("pothole", 40, 0, 100, 100)],
                                iou_thr=0.5)
    assert shifted["tp"] == []          # IoU ~0.43 < 0.5


def test_fp_category_priority_duplicate_beats_misclass_and_confusor():
    # Пин приоритета категорий (ревью: перестановки веток выживали): pred,
    # квалифицирующийся на duplicate И gt_patch И конфузор, получает duplicate.
    fr = _frame(defects=[_gt("pothole", 100, 100, 200, 150),
                         _gt("patch", 100, 100, 200, 150)],
                confusors=[_conf("car", 0, 0, 1000, 800)])
    preds = [_pred("pothole", 100, 100, 200, 150, conf=0.9),
             _pred("pothole", 102, 102, 198, 148, conf=0.5)]
    r = ed.evaluate_frame(fr, preds)
    assert [cat for _, cat in r["fp"]] == ["duplicate"]


def test_fp_category_misclass_beats_confusor():
    fr = _frame(defects=[_gt("patch", 100, 100, 200, 150)],
                confusors=[_conf("car", 0, 0, 1000, 800)])
    r = ed.evaluate_frame(fr, [_pred("pothole", 100, 100, 200, 150)])
    assert [cat for _, cat in r["fp"]] == ["gt_patch"]


def test_misclass_by_containment_inside_large_patch():
    # Ревью [MEDIUM]: модель боксует ЧАСТЬ длинной заплатки — IoU мал, но бокс
    # целиком внутри GT-заплатки. Раньше утекало в unknown.
    fr = _frame(defects=[_gt("patch", 50, 50, 600, 400)])
    r = ed.evaluate_frame(fr, [_pred("pothole", 100, 100, 100, 80)])
    assert [cat for _, cat in r["fp"]] == ["gt_patch"]


def test_misclass_attribution_picks_best_overlap_not_list_order():
    fr = _frame(defects=[_gt("alligator_crack", 100, 100, 100, 100),
                         _gt("patch", 120, 100, 160, 100)])
    r = ed.evaluate_frame(fr, [_pred("pothole", 100, 100, 100, 100)])
    assert [cat for _, cat in r["fp"]] == ["gt_alligator_crack"]  # IoU 1.0 > 0.8


def test_edge_rule_uses_min_side_and_any_border():
    fr = _frame(size=(1152, 2560))
    # 100 px > 6% от МЕНЬШЕЙ стороны (69) — не edge, хоть и в углу
    r = ed.evaluate_frame(fr, [_pred("pothole", 2, 2, 100, 100)])
    assert [cat for _, cat in r["fp"]] == ["unknown"]
    # мелочь у ЛЕВОЙ кромки в середине высоты (не угол) — edge
    r2 = ed.evaluate_frame(fr, [_pred("pothole", 0, 1200, 40, 40)])
    assert [cat for _, cat in r2["fp"]] == ["edge"]


# --- агрегация -----------------------------------------------------------------
def test_aggregate_counts_foreign_class_fp():
    # COCO-фолбэк реален: pred с классом вне таксономии не должен молча
    # выпадать из precision (ревью: мутация «выбросить» выживала).
    r = ed.evaluate_frame(_frame(), [_pred("person", 10, 10, 50, 50)])
    agg = ed.aggregate({"f": r})
    assert agg["per_label"]["_foreign"]["fp"] == 1
    assert agg["total"]["fp"] == 1
    assert agg["fp_rows"][0]["label"] == "person"



def test_aggregate_counts_and_fp_matrix():
    fr1 = _frame(defects=[_gt("pothole", 100, 100, 200, 150)])
    fr2 = _frame(defects=[_gt("patch", 100, 100, 200, 150)],
                 confusors=[_conf("car", 500, 500, 300, 200)])
    r1 = ed.evaluate_frame(fr1, [_pred("pothole", 100, 100, 200, 150)])
    r2 = ed.evaluate_frame(fr2, [_pred("pothole", 100, 100, 200, 150),
                                 _pred("pothole", 550, 550, 100, 80)])
    agg = ed.aggregate({"f1": r1, "f2": r2})
    pot = agg["per_label"]["pothole"]
    assert (pot["gt"], pot["tp"], pot["fp"], pot["fn"]) == (1, 1, 2, 0)
    pat = agg["per_label"]["patch"]
    assert (pat["gt"], pat["fn"]) == (1, 1)
    assert agg["fp_by_cat"]["pothole"] == {"gt_patch": 1, "car": 1}
    assert {row["category"] for row in agg["fp_rows"]} == {"gt_patch", "car"}
    assert agg["fn_rows"][0]["label"] == "patch"


# --- валидация разметки ---------------------------------------------------------
def _write_ann(path: Path, frames: dict, version=1):
    path.write_text(json.dumps({"version": version, "frames": frames},
                               ensure_ascii=False), encoding="utf-8")


def test_load_annotations_rejects_bad_version_label_and_bbox(tmp_path):
    p = tmp_path / "a.json"
    _write_ann(p, {"f": _frame()}, version=2)
    with pytest.raises(ValueError, match="version"):
        ed.load_annotations(p)
    _write_ann(p, {"f": _frame(defects=[_gt("hole", 0, 0, 10, 10)])})
    with pytest.raises(ValueError, match="label"):
        ed.load_annotations(p)
    _write_ann(p, {"f": _frame(defects=[_gt("pothole", 0, 0, 0, 10)])})
    with pytest.raises(ValueError, match="bbox"):
        ed.load_annotations(p)
    _write_ann(p, {"f": _frame(confusors=[_conf("wheel", 0, 0, 9, 9)])})
    with pytest.raises(ValueError, match="категория"):
        ed.load_annotations(p)


# --- main: дисциплина reviewed / missing ----------------------------------------
def _run_main(tmp_path, monkeypatch, frames: dict, reports: dict,
              extra_args: list | None = None) -> int:
    ann = tmp_path / "annotations.json"
    _write_ann(ann, frames)
    rep_dir = tmp_path / "reports"
    rep_dir.mkdir(exist_ok=True)
    for fid, defects in reports.items():
        (rep_dir / f"{fid}.json").write_text(
            json.dumps({"defects": defects}), encoding="utf-8")
    argv = ["prog", "--annotations", str(ann), "--reports", str(rep_dir),
            "--out", str(tmp_path / "out")] + (extra_args or [])
    monkeypatch.setattr(sys, "argv", argv)
    return ed.main()


def _rep_defect(label, x, y, w, h, conf=0.8):
    return {"class": label, "confidence": conf, "bbox_px": [x, y, w, h]}


def test_main_happy_path_writes_csvs(tmp_path, monkeypatch, capsys):
    frames = {"f1": _frame(defects=[_gt("pothole", 100, 100, 200, 150)])}
    reports = {"f1": [_rep_defect("pothole", 100, 100, 200, 150)]}
    assert _run_main(tmp_path, monkeypatch, frames, reports) == 0
    out = capsys.readouterr().out
    assert "ИТОГО" in out
    assert "Параметры замера: iou=0.5" in out       # штамп порогов (ревью)
    summary = (tmp_path / "out" / "eval_summary.csv").read_text(encoding="utf-8-sig")
    assert "pothole,1,1,0,0,1.000,1.000" in summary
    assert (tmp_path / "out" / "eval_fp.csv").is_file()
    assert (tmp_path / "out" / "eval_fn.csv").is_file()
    params = json.loads((tmp_path / "out" / "eval_params.json")
                        .read_text(encoding="utf-8"))
    assert params["iou"] == 0.5 and params["frames_reviewed"] == 1


def test_csv_exact_rows_and_bom(tmp_path, monkeypatch):
    # Пин формата CSV (ревью: перестановка precision<->recall и x<->y выживали;
    # eval_fp — координаты hard-negative кропов P4).
    frames = {"f1": _frame(defects=[_gt("pothole", 100, 100, 200, 150)])}
    reports = {"f1": [_rep_defect("pothole", 100, 100, 200, 150, conf=0.9),
                      _rep_defect("pothole", 600, 500, 80, 60, conf=0.4)]}
    assert _run_main(tmp_path, monkeypatch, frames, reports) == 0
    raw = (tmp_path / "out" / "eval_summary.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")            # utf-8-sig для Excel
    assert "pothole,1,1,1,0,0.500,1.000" in raw.decode("utf-8-sig")
    fp_lines = [l for l in (tmp_path / "out" / "eval_fp.csv")
                .read_text(encoding="utf-8-sig").splitlines() if l]
    assert fp_lines[0] == "frame_id,label,confidence,category,x,y,w,h"
    assert fp_lines[1].startswith("f1,pothole,0.4,unknown,600")


def test_main_prints_coverage_against_manifest(tmp_path, monkeypatch, capsys):
    # Ревью [MEDIUM]: тихое усыхание набора (кадр выпал из annotations.json)
    # обязано быть видно в каждом замере.
    frames = {"f1": _frame()}
    (tmp_path / "manifest.csv").write_text(
        "frame_id,scene,view,group,source,md5\nf1,s,v,g,x,aa\nf2,s,v,g,x,bb\n",
        encoding="utf-8-sig")
    assert _run_main(tmp_path, monkeypatch, frames, {"f1": []}) == 0
    out = capsys.readouterr().out
    assert "Покрытие набора: manifest 2" in out
    assert "f2" in out                                # выпавший кадр назван


def test_main_draft_frames_are_skipped(tmp_path, monkeypatch, capsys):
    frames = {"f1": _frame(defects=[_gt("pothole", 0, 0, 10, 10)]),
              "f2": _frame(status="draft",
                           defects=[_gt("pothole", 0, 0, 10, 10)])}
    reports = {"f1": [], "f2": []}
    assert _run_main(tmp_path, monkeypatch, frames, reports) == 0
    out = capsys.readouterr().out
    assert "пропущено draft: 1" in out


def test_main_only_drafts_refuses(tmp_path, monkeypatch, capsys):
    frames = {"f1": _frame(status="draft")}
    assert _run_main(tmp_path, monkeypatch, frames, {"f1": []}) == 2
    assert "reviewed" in capsys.readouterr().out


def test_main_missing_report_is_error_unless_allowed(tmp_path, monkeypatch, capsys):
    frames = {"f1": _frame(defects=[_gt("pothole", 0, 0, 100, 100)])}
    assert _run_main(tmp_path, monkeypatch, frames, {}) == 2
    assert "НЕ равен «нулю детекций»" in capsys.readouterr().out

    assert _run_main(tmp_path, monkeypatch, frames, {},
                     ["--allow-missing"]) == 0
    out = capsys.readouterr().out
    assert "0 детекций" in out          # громкая оговорка
    assert "f1" in out                  # кадры названы поимённо
    assert "ЗАВЫШЕН" in out             # и предупреждение про precision (ревью)
    summary = (tmp_path / "out" / "eval_summary.csv").read_text(encoding="utf-8-sig")
    assert "pothole,1,0,0,1" in summary  # честный FN, не выдуманный TP
