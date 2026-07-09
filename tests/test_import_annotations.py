"""Конвертер внешней разметки (scripts/import_annotations.py): родной формат
X-AnyLabeling (пер-кадровые JSON), YOLO, COCO -> annotations.json схемы v1.
Чистые файловые операции — без моделей и сети."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ia = _load("import_annotations")
ed = _load("eval_detection")


# --- обвязка -------------------------------------------------------------------
def _setup(tmp_path, fids):
    frames = tmp_path / "frames"
    frames.mkdir()
    man = tmp_path / "manifest.csv"
    man.write_text("frame_id,scene,view,group,source,md5\n"
                   + "".join(f"{f},s,front,g,src,0\n" for f in fids),
                   encoding="utf-8")
    return frames, man, tmp_path / "annotations.json"


def _args(frames, man, out, inp=None, extra=()):
    return ["--input", str(inp or frames), "--frames", str(frames),
            "--manifest", str(man), "--out", str(out), *extra]


def _xany(frames_dir, fid, shapes, size=(100, 80), image_path=None):
    (frames_dir / f"{fid}.json").write_text(json.dumps({
        "version": "2.4.0", "flags": {}, "shapes": shapes,
        "imagePath": image_path or f"{fid}.jpg", "imageData": None,
        "imageWidth": size[0], "imageHeight": size[1]}), encoding="utf-8")


def _rect(label, x0, y0, x1, y1):
    return {"label": label, "points": [[x0, y0], [x1, y1]],
            "shape_type": "rectangle", "group_id": None, "flags": {}}


# --- родной формат X-AnyLabeling -------------------------------------------------
def test_xany_rects_polygons_confusors_and_empty_frame(tmp_path, capsys):
    frames, man, out = _setup(tmp_path, ["001_front", "001_back"])
    _xany(frames, "001_front", [
        _rect("pothole", 50, 40, 10, 20),        # 2 точки «обратной» растяжкой
        {"label": "patch", "shape_type": "rectangle",  # 4-точечный (новые версии)
         "points": [[10, 60], [50, 60], [50, 70], [10, 70]]},
        {"label": "alligator_crack", "shape_type": "polygon",  # SAM-полигон
         "points": [[60, 10], [90, 5], [80, 30]]},
        _rect("car", 0, 0, 30, 10),
    ])
    _xany(frames, "001_back", [])                 # просмотрен, дефектов нет
    assert ia.main(_args(frames, man, out)) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert list(data["frames"]) == ["001_back", "001_front"]  # сортировано
    fr = data["frames"]["001_front"]
    assert fr["status"] == "reviewed" and fr["image_size_px"] == [100, 80]
    assert {d["label"]: d["bbox_xywh"] for d in fr["defects"]} == {
        "pothole": [10, 20, 40, 20],
        "patch": [10, 60, 40, 10],
        "alligator_crack": [60, 5, 30, 25]}
    assert fr["confusors"] == [{"category": "car", "bbox_xywh": [0, 0, 30, 10]}]
    empty = data["frames"]["001_back"]
    assert empty["status"] == "reviewed"
    assert empty["defects"] == [] and empty["confusors"] == []
    # результат читает валидатор замера, пустой кадр назван в сводке вслух
    assert set(ed.load_annotations(out)) == {"001_front", "001_back"}
    assert "001_back" in capsys.readouterr().out


def test_unknown_label_fails_and_names_frames(tmp_path, capsys):
    frames, man, out = _setup(tmp_path, ["001_front"])
    _xany(frames, "001_front", [_rect("manhole", 0, 0, 10, 10)])
    assert ia.main(_args(frames, man, out)) == 2
    assert not out.exists()
    text = capsys.readouterr().out
    assert "manhole" in text and "001_front" in text


def test_rdd_codes_and_loose_names_normalize(tmp_path):
    frames, man, out = _setup(tmp_path, ["001_front"])
    _xany(frames, "001_front", [_rect("D40", 0, 0, 10, 10),
                                _rect("Repair", 20, 0, 30, 10),
                                _rect("Off Road", 40, 0, 50, 10)])
    assert ia.main(_args(frames, man, out)) == 0
    fr = json.loads(out.read_text(encoding="utf-8"))["frames"]["001_front"]
    assert [d["label"] for d in fr["defects"]] == ["pothole", "patch"]
    assert [c["category"] for c in fr["confusors"]] == ["off_road"]


def test_normalize_label_variants():
    assert ia.normalize_label(" Pothole ") == "pothole"
    assert ia.normalize_label("shadow-curb") == "shadow_curb"
    assert ia.normalize_label("d20") == "alligator_crack"


def test_non_bbox_shape_rejected(tmp_path, capsys):
    frames, man, out = _setup(tmp_path, ["001_front"])
    _xany(frames, "001_front",
          [{"label": "longitudinal_crack", "shape_type": "line",
            "points": [[0, 0], [50, 50]]}])
    assert ia.main(_args(frames, man, out)) == 2
    assert "line" in capsys.readouterr().out


def test_frame_outside_manifest_fails(tmp_path, capsys):
    frames, man, out = _setup(tmp_path, ["001_front"])
    _xany(frames, "999_front", [_rect("pothole", 0, 0, 10, 10)])
    assert ia.main(_args(frames, man, out)) == 2
    assert "manifest" in capsys.readouterr().out


def test_renamed_image_path_fails(tmp_path, capsys):
    frames, man, out = _setup(tmp_path, ["001_front"])
    _xany(frames, "001_front", [_rect("pothole", 0, 0, 10, 10)],
          image_path="переименован.jpg")
    assert ia.main(_args(frames, man, out)) == 2
    assert "переименован" in capsys.readouterr().out


def test_annotated_downscaled_copy_fails(tmp_path, capsys):
    Image = pytest.importorskip("PIL.Image")
    frames, man, out = _setup(tmp_path, ["001_front"])
    Image.new("RGB", (200, 160)).save(frames / "001_front.jpg")
    _xany(frames, "001_front", [_rect("pothole", 0, 0, 10, 10)], size=(100, 80))
    assert ia.main(_args(frames, man, out)) == 2
    assert "координаты" in capsys.readouterr().out


def test_empty_input_dir_fails(tmp_path, capsys):
    frames, man, out = _setup(tmp_path, ["001_front"])
    assert ia.main(_args(frames, man, out)) == 2
    assert not out.exists()


def test_mixed_json_and_txt_requires_explicit_format(tmp_path, capsys):
    frames, man, out = _setup(tmp_path, ["001_front"])
    _xany(frames, "001_front", [])
    (frames / "001_front.txt").write_text("", encoding="utf-8")
    assert ia.main(_args(frames, man, out)) == 2
    assert "--format" in capsys.readouterr().out


def test_rewrite_backs_up_previous_annotations(tmp_path):
    frames, man, out = _setup(tmp_path, ["001_front"])
    _xany(frames, "001_front", [_rect("pothole", 0, 0, 10, 10)])
    assert ia.main(_args(frames, man, out)) == 0
    _xany(frames, "001_front", [_rect("pothole", 0, 0, 20, 20)])
    assert ia.main(_args(frames, man, out)) == 0
    baks = list(tmp_path.glob("annotations.bak-*.json"))
    assert len(baks) == 1
    old = json.loads(baks[0].read_text(encoding="utf-8"))
    assert old["frames"]["001_front"]["defects"][0]["bbox_xywh"] == [0, 0, 10, 10]


# --- YOLO / COCO -----------------------------------------------------------------
def test_yolo_export_import(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    frames, man, out = _setup(tmp_path, ["001_front", "001_back"])
    Image.new("RGB", (100, 80)).save(frames / "001_front.jpg")
    (frames / "classes.txt").write_text("pothole\ncar\n", encoding="utf-8")
    (frames / "001_front.txt").write_text(
        "0 0.5 0.5 0.2 0.25\n1 0.15 0.0625 0.3 0.125\n", encoding="utf-8")
    assert ia.main(_args(frames, man, out)) == 0
    fr = json.loads(out.read_text(encoding="utf-8"))["frames"]["001_front"]
    assert fr["defects"] == [{"label": "pothole", "bbox_xywh": [40, 30, 20, 20]}]
    assert fr["confusors"] == [{"category": "car", "bbox_xywh": [0, 0, 30, 10]}]


# --- отказы по находкам адверсариального пре-коммит ревью 2026-07-09 -------------
def test_coco_duplicate_file_name_stems_fail(tmp_path, capsys):
    # два file_name с одним стемом молча схлопывались с потерей рамок (BLOCKER)
    frames, man, out = _setup(tmp_path, ["001_front"])
    coco = tmp_path / "export_coco.json"
    coco.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": "001_front.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "sub/001_front.jpg", "width": 100, "height": 80}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 7, "bbox": [5, 6, 30, 40]},
            {"id": 11, "image_id": 2, "category_id": 7, "bbox": [1, 2, 3, 4]}],
        "categories": [{"id": 7, "name": "pothole"}]}), encoding="utf-8")
    assert ia.main(_args(frames, man, out, inp=coco)) == 2
    assert not out.exists()
    assert "001_front" in capsys.readouterr().out


def test_coco_orphan_annotation_fails(tmp_path, capsys):
    # annotation с несуществующим image_id молча выбрасывалась
    frames, man, out = _setup(tmp_path, ["001_front"])
    coco = tmp_path / "export_coco.json"
    coco.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "001_front.jpg",
                    "width": 100, "height": 80}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 7, "bbox": [5, 6, 30, 40]},
            {"id": 11, "image_id": 999, "category_id": 7, "bbox": [1, 2, 3, 4]}],
        "categories": [{"id": 7, "name": "pothole"}]}), encoding="utf-8")
    assert ia.main(_args(frames, man, out, inp=coco)) == 2
    assert not out.exists()
    assert "999" in capsys.readouterr().out


def test_coco_nonnumeric_bbox_is_problem_not_traceback(tmp_path, capsys):
    frames, man, out = _setup(tmp_path, ["001_front"])
    coco = tmp_path / "export_coco.json"
    coco.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "001_front.jpg",
                    "width": 100, "height": 80}],
        "annotations": [{"id": 10, "image_id": 1, "category_id": 7,
                         "bbox": [1, "a", 10, 10]}],
        "categories": [{"id": 7, "name": "pothole"}]}), encoding="utf-8")
    assert ia.main(_args(frames, man, out, inp=coco)) == 2
    assert "bbox" in capsys.readouterr().out


def test_garbage_values_reported_as_problems_not_traceback(tmp_path, capsys):
    # NaN в координатах (json.loads принимает литерал NaN) и точки-объекты
    frames, man, out = _setup(tmp_path, ["001_front", "001_back"])
    _xany(frames, "001_front",
          [{"label": "pothole", "shape_type": "rectangle",
            "points": [[0, 0], [float("nan"), 10]]}])
    _xany(frames, "001_back",
          [{"label": "pothole", "shape_type": "rectangle",
            "points": [{"x": 1}, [10, 10]]}])
    assert ia.main(_args(frames, man, out)) == 2
    assert not out.exists()
    text = capsys.readouterr().out
    assert "001_front" in text and "001_back" in text


def test_all_problems_reported_in_one_pass(tmp_path, capsys):
    # парс-проблема НЕ должна прятать проблемы стадии сборки (метки, manifest)
    frames, man, out = _setup(tmp_path, ["001_front", "001_back"])
    _xany(frames, "001_front",
          [{"label": "longitudinal_crack", "shape_type": "line",
            "points": [[0, 0], [50, 50]]},
           _rect("manhole", 0, 0, 10, 10)])
    _xany(frames, "999_front", [_rect("pothole", 0, 0, 10, 10)])
    assert ia.main(_args(frames, man, out)) == 2
    text = capsys.readouterr().out
    assert "line" in text and "manhole" in text and "999_front" in text


def test_utf16_yolo_files_fail_loudly(tmp_path, capsys):
    # PowerShell 5.1 пишет UTF-16 по умолчанию — частый случай ручной правки
    frames, man, out = _setup(tmp_path, ["001_front"])
    (frames / "classes.txt").write_text("pothole\n", encoding="utf-16")
    (frames / "001_front.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert ia.main(_args(frames, man, out)) == 2
    assert "UTF-8" in capsys.readouterr().out


def test_bbox_fully_outside_frame_fails(tmp_path, capsys):
    frames, man, out = _setup(tmp_path, ["001_front"])
    _xany(frames, "001_front", [_rect("pothole", 500, 500, 600, 600)])  # кадр 100x80
    assert ia.main(_args(frames, man, out)) == 2
    assert "вне кадра" in capsys.readouterr().out


def test_coco_all_empty_prints_skip_note_on_fail_path(tmp_path, capsys):
    frames, man, out = _setup(tmp_path, ["001_front"])
    coco = tmp_path / "export_coco.json"
    coco.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "001_front.jpg",
                    "width": 100, "height": 80}],
        "annotations": [], "categories": []}), encoding="utf-8")
    assert ia.main(_args(frames, man, out, inp=coco)) == 2
    assert "ПРОПУЩЕНЫ" in capsys.readouterr().out


def test_bom_in_xany_json_is_tolerated(tmp_path):
    frames, man, out = _setup(tmp_path, ["001_front"])
    payload = json.dumps({"version": "2.4.0", "flags": {},
                          "shapes": [_rect("pothole", 0, 0, 10, 10)],
                          "imagePath": "001_front.jpg", "imageData": None,
                          "imageWidth": 100, "imageHeight": 80})
    (frames / "001_front.json").write_bytes(b"\xef\xbb\xbf"
                                            + payload.encode("utf-8"))
    assert ia.main(_args(frames, man, out)) == 0


def test_two_rewrites_keep_two_distinct_backups(tmp_path):
    # два импорта в одну секунду не должны затирать друг другу бэкап
    frames, man, out = _setup(tmp_path, ["001_front"])
    for w in (10, 20, 30):
        _xany(frames, "001_front", [_rect("pothole", 0, 0, w, w)])
        assert ia.main(_args(frames, man, out)) == 0
    baks = sorted(tmp_path.glob("annotations.bak-*.json"))
    assert len(baks) == 2
    contents = [json.loads(b.read_text(encoding="utf-8"))["frames"]
                ["001_front"]["defects"][0]["bbox_xywh"] for b in baks]
    assert [0, 0, 10, 10] in contents and [0, 0, 20, 20] in contents


def test_coco_import_skips_boxless_images(tmp_path, capsys):
    frames, man, out = _setup(tmp_path, ["001_front", "001_back"])
    coco = tmp_path / "export_coco.json"
    coco.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": "001_front.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "001_back.jpg", "width": 100, "height": 80}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 7, "bbox": [5, 6, 30, 40]}],
        "categories": [{"id": 7, "name": "pothole"}]}), encoding="utf-8")
    assert ia.main(_args(frames, man, out, inp=coco)) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert list(data["frames"]) == ["001_front"]
    assert data["frames"]["001_front"]["defects"] == [
        {"label": "pothole", "bbox_xywh": [5, 6, 30, 40]}]
    assert "001_back" in capsys.readouterr().out  # пропуск назван вслух
