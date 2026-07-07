"""Разметчик (scripts/build_eval_annotator.py): встраивание кадров и масштаба,
предложения движка в ПИКСЕЛЯХ ОРИГИНАЛА, предзагрузка существующей разметки."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_eval_annotator", ROOT / "scripts" / "build_eval_annotator.py")
bea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bea)

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _eval_dir(tmp_path: Path, size=(64, 32)) -> Path:
    d = tmp_path / "eval_v1"
    (d / "frames").mkdir(parents=True)
    for fid, color in (("001_front", (200, 0, 0)), ("001_back", (0, 200, 0))):
        Image.new("RGB", size, color).save(d / "frames" / f"{fid}.jpg", "JPEG")
    (d / "manifest.csv").write_text(
        "frame_id,scene,view,group,source,md5\n"
        "001_front,001,front,clean,x,aa\n"
        "001_back,001,back,clean,x,bb\n", encoding="utf-8-sig")
    return d


def _page_data(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>',
                  html, re.S)
    assert m, "в странице нет data-блока"
    return json.loads(m.group(1).replace("<\\/", "</"))


def _run(tmp_path, monkeypatch, eval_dir: Path, out: Path,
         proposals: Path | None = None, max_px: int = 16) -> int:
    argv = ["prog", "--eval-dir", str(eval_dir), "--out", str(out),
            "--max-px", str(max_px)]
    if proposals is not None:
        argv += ["--proposals", str(proposals)]
    monkeypatch.setattr(sys, "argv", argv)
    return bea.main()


def test_page_embeds_frames_with_original_size_and_scale(tmp_path, monkeypatch):
    d = _eval_dir(tmp_path, size=(64, 32))
    out = tmp_path / "annotator.html"
    assert _run(tmp_path, monkeypatch, d, out, max_px=16) == 0

    data = _page_data(out.read_text(encoding="utf-8"))
    assert [f["id"] for f in data["frames"]] == ["001_front", "001_back"]
    f = data["frames"][0]
    assert (f["w"], f["h"]) == (64, 32)          # оригинальный размер
    assert f["dw"] == 16                          # ужат до max-px
    assert f["scale"] == pytest.approx(4.0)       # disp -> orig
    assert f["uri"].startswith("data:image/jpeg;base64,")
    assert data["defectLabels"] == bea.DEFECT_LABELS
    assert data["initial"] is None


def test_proposals_come_from_reports_in_original_px(tmp_path, monkeypatch):
    d = _eval_dir(tmp_path)
    rep_dir = tmp_path / "reports"
    rep_dir.mkdir()
    (rep_dir / "001_front.json").write_text(json.dumps({
        "defects": [
            {"class": "pothole", "confidence": 0.71, "bbox_px": [4, 8, 20, 10]},
            {"class": "person", "confidence": 0.9, "bbox_px": [0, 0, 5, 5]},
        ]}), encoding="utf-8")
    out = tmp_path / "annotator.html"
    assert _run(tmp_path, monkeypatch, d, out, proposals=rep_dir) == 0

    data = _page_data(out.read_text(encoding="utf-8"))
    props = data["proposals"]["001_front"]
    assert props == [{"label": "pothole", "conf": 0.71,
                      "bbox": [4.0, 8.0, 20.0, 10.0]}]  # оригинальные px, без COCO-класса
    assert "001_back" not in data["proposals"]


def test_existing_annotations_are_preloaded(tmp_path, monkeypatch):
    d = _eval_dir(tmp_path)
    ann = {"version": 1, "frames": {"001_front": {
        "image_size_px": [64, 32], "status": "reviewed",
        "defects": [{"label": "pothole", "bbox_xywh": [1, 2, 30, 20]}],
        "confusors": [], "note": ""}}}
    (d / "annotations.json").write_text(json.dumps(ann), encoding="utf-8")
    out = tmp_path / "annotator.html"
    assert _run(tmp_path, monkeypatch, d, out) == 0

    data = _page_data(out.read_text(encoding="utf-8"))
    assert data["initial"]["frames"]["001_front"]["status"] == "reviewed"


def test_missing_manifest_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "--eval-dir",
                                      str(tmp_path / "nope"),
                                      "--out", str(tmp_path / "a.html")])
    assert bea.main() == 2
    assert "make_eval_set" in capsys.readouterr().out


def test_script_close_tag_in_note_survives_embedding(tmp_path, monkeypatch):
    # Ревью: '</script>' в note импортированной разметки без экранирования
    # обрезал бы data-блок страницы.
    d = _eval_dir(tmp_path)
    ann = {"version": 1, "frames": {"001_front": {
        "image_size_px": [64, 32], "status": "draft",
        "defects": [], "confusors": [], "note": "</script><b>xss"}}}
    (d / "annotations.json").write_text(json.dumps(ann), encoding="utf-8")
    out = tmp_path / "a.html"
    assert _run(tmp_path, monkeypatch, d, out) == 0
    data = _page_data(out.read_text(encoding="utf-8"))
    assert data["initial"]["frames"]["001_front"]["note"] == "</script><b>xss"


def test_scale_consistent_when_thumbnail_rounds_height(tmp_path, monkeypatch):
    # Ревью: масштаб один на обе оси (по ширине); при округлении thumbnail
    # высота обязана сходиться в пределах пикселя дисплея.
    d = _eval_dir(tmp_path, size=(64, 33))
    out = tmp_path / "a.html"
    assert _run(tmp_path, monkeypatch, d, out, max_px=16) == 0
    f = _page_data(out.read_text(encoding="utf-8"))["frames"][0]
    assert f["scale"] == pytest.approx(4.0)           # 64/16, не 33/8
    assert abs(f["h"] / f["scale"] - f["dh"]) <= 0.5


def test_broken_proposal_report_is_skipped(tmp_path, monkeypatch):
    d = _eval_dir(tmp_path)
    rep_dir = tmp_path / "reports"
    rep_dir.mkdir()
    (rep_dir / "001_back.json").write_text("{broken", encoding="utf-8")
    out = tmp_path / "a.html"
    assert _run(tmp_path, monkeypatch, d, out, proposals=rep_dir) == 0
    assert "001_back" not in _page_data(out.read_text(encoding="utf-8"))["proposals"]


def test_page_scopes_storage_by_set_and_persists_proposal_decisions(tmp_path, monkeypatch):
    # Ревью [HIGH]: решения по предложениям должны переживать перезагрузку
    # (state.proposals_resolved), ключ localStorage — от состава набора.
    d = _eval_dir(tmp_path)
    out = tmp_path / "a.html"
    assert _run(tmp_path, monkeypatch, d, out) == 0
    html = out.read_text(encoding="utf-8")
    data = _page_data(html)
    assert re.fullmatch(r"[0-9a-f]{8}", data["setId"])
    assert "'road_defect_eval_annotations_' + DATA.setId" in html
    assert "proposals_resolved" in html               # персист решений в state


def test_manifest_contract_with_make_eval_set(tmp_path, monkeypatch):
    # Кросс-контракт: manifest, который пишет make_eval_set, читается
    # load_manifest разметчика (колонка frame_id, кодировка utf-8-sig).
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "make_eval_set", ROOT / "scripts" / "make_eval_set.py")
    mes = ilu.module_from_spec(spec)
    spec.loader.exec_module(mes)

    pairs = tmp_path / "pairs"
    (pairs / "001").mkdir(parents=True)
    Image.new("RGB", (16, 16), (10, 0, 0)).save(pairs / "001" / "front.jpg", "JPEG")
    Image.new("RGB", (16, 16), (0, 10, 0)).save(pairs / "001" / "back.jpg", "JPEG")
    out = tmp_path / "eval_v1"
    monkeypatch.setattr(mes, "EVAL_SCENES", {"clean": ["001"]})
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs), "--out", str(out),
                         "--demo-dir", ""])
    assert mes.main() == 0

    rows = bea.load_manifest(out)
    assert [r["frame_id"] for r in rows] == ["001_front", "001_back"]
