"""Разрешение исходников в калибровке глубины (ревью 2026-07-03): у парных
отчётов image — всегда 'front.jpg', исходник ищется по pair_inputs/--images."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "calibrate_depth_bucket", ROOT / "scripts" / "calibrate_depth_bucket.py")
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)


def test_pair_inputs_path_wins(tmp_path):
    src = tmp_path / "pairs" / "001" / "front.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"\xff\xd8fake")
    rep = {"image": "front.jpg",
           "pair_inputs": {"front": str(src), "back": "x"}}
    assert cal._resolve_image(rep, tmp_path, None) == src


def test_images_dir_recursive_fallback(tmp_path):
    deep = tmp_path / "photos" / "trip1" / "IMG_1.jpg"
    deep.parent.mkdir(parents=True)
    deep.write_bytes(b"\xff\xd8fake")
    rep = {"image": "IMG_1.jpg"}
    assert cal._resolve_image(rep, tmp_path, tmp_path / "photos") == deep


def test_missing_source_returns_none(tmp_path):
    assert cal._resolve_image({"image": "nope.jpg"}, tmp_path, None) is None
