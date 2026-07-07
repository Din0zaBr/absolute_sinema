"""Скелет eval-набора (scripts/make_eval_set.py): отбор кадров, дедуп по md5,
manifest + holdout, идемпотентность."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "make_eval_set", ROOT / "scripts" / "make_eval_set.py")
mes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mes)

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _img_pair(root: Path, pid: str, front_color, back_color) -> None:
    d = root / pid
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), front_color).save(d / "front.jpg", "JPEG")
    Image.new("RGB", (16, 16), back_color).save(d / "back.jpg", "JPEG")


def test_collect_frames_dedups_identical_photo_across_scenes(tmp_path):
    # 001 и 002 делят один front (общая сцена) — в eval он попадает ОДИН раз:
    # одна и та же фотография дважды в метриках = двойной счёт.
    pairs = tmp_path / "pairs"
    _img_pair(pairs, "001", (10, 0, 0), (0, 10, 0))
    _img_pair(pairs, "002", (10, 0, 0), (0, 20, 0))   # тот же front
    rows, warns = mes.collect_frames(
        pairs, None, {"clean": ["001", "002"]})
    ids = [r["frame_id"] for r in rows]
    assert "001_front" in ids and "002_front" not in ids
    assert "001_back" in ids and "002_back" in ids     # back'и разные — оба в наборе
    assert any("общая сцена" in w for w in warns)


def test_collect_frames_missing_scene_warns_but_continues(tmp_path):
    pairs = tmp_path / "pairs"
    _img_pair(pairs, "001", (10, 0, 0), (0, 10, 0))
    rows, warns = mes.collect_frames(pairs, None, {"clean": ["001", "999"]})
    assert [r["frame_id"] for r in rows] == ["001_front", "001_back"]
    assert any("999" in w for w in warns)


def test_main_writes_frames_manifest_and_holdout(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs"
    _img_pair(pairs, "001", (10, 0, 0), (0, 10, 0))
    _img_pair(pairs, "003", (30, 0, 0), (0, 30, 0))
    demo = tmp_path / "demo"
    demo.mkdir()
    Image.new("RGB", (16, 16), (99, 99, 99)).save(demo / "citycam.jpg", "JPEG")
    out = tmp_path / "eval_v1"

    monkeypatch.setattr(mes, "EVAL_SCENES", {"clean": ["001", "003"]})
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs), "--out", str(out),
                         "--demo-dir", str(demo)])
    assert mes.main() == 0

    frames = sorted(p.name for p in (out / "frames").glob("*.jpg"))
    assert frames == ["001_back.jpg", "001_front.jpg", "003_back.jpg",
                      "003_front.jpg", "demo_citycam.jpg"]
    manifest = (out / "manifest.csv").read_text(encoding="utf-8-sig")
    assert "001_front" in manifest and "demo_citycam" in manifest
    holdout = (out / "holdout_scenes.txt").read_text(encoding="utf-8")
    lines = [l for l in holdout.splitlines()
             if l.strip() and not l.startswith("#")]
    assert lines == ["001", "003"]        # демо-кадры не из пар — не в holdout

    # Идемпотентность: повторный прогон не падает и не плодит файлов.
    assert mes.main() == 0
    assert sorted(p.name for p in (out / "frames").glob("*.jpg")) == frames


def test_main_missing_pairs_dir_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--pairs",
                                      str(tmp_path / "nope"), "--demo-dir", ""])
    assert mes.main() == 2


def test_demo_frame_ids_stay_unique_on_common_prefix(tmp_path):
    # Ревью: общий 8-символьный префикс стемов схлопывал два разных фото в
    # один frame_id (дубль в manifest, вторая копия затирала первую).
    demo = tmp_path / "demo"
    demo.mkdir()
    Image.new("RGB", (16, 16), (10, 0, 0)).save(demo / "photocam_alpha.jpg", "JPEG")
    Image.new("RGB", (16, 16), (0, 10, 0)).save(demo / "photocam_beta.jpg", "JPEG")
    rows, _ = mes.collect_frames(tmp_path, demo, {})
    ids = [r["frame_id"] for r in rows]
    assert len(ids) == 2 and len(set(ids)) == 2


def test_broken_demo_frame_skipped_with_warning(tmp_path):
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "broken.jpg").write_bytes(b"\xff\xd8truncated")
    rows, warns = mes.collect_frames(tmp_path, demo, {})
    assert rows == []
    assert any("не читается" in w for w in warns)


def test_copy_refreshes_stale_frame_with_same_size(tmp_path):
    # Ревью: скип по равенству размера оставлял старую копию при новом md5 в
    # manifest — integrity набора лгала. Скип теперь по md5.
    src, dst = tmp_path / "s.jpg", tmp_path / "d.jpg"
    src.write_bytes(b"AAAA")
    dst.write_bytes(b"BBBB")                      # тот же размер, иное содержимое
    mes._copy(src, dst, force=False, src_md5=mes._md5(src))
    assert dst.read_bytes() == b"AAAA"
