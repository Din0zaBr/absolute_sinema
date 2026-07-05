"""Разрешение исходников в калибровке глубины (ревью 2026-07-03): у парных
отчётов image — всегда 'front.jpg', исходник ищется по pair_inputs/--images."""
import importlib.util
from pathlib import Path

import pytest

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


def test_pair_reports_resolve_to_their_own_folder(tmp_path):
    # у всех пар image=='front.jpg'; rglob по имени схлопывал их к первому
    # кадру — теперь разрешение держится папки пары NNN (ревью 2026-07-05).
    images = tmp_path / "local_pairs"
    for pid in ("001", "002"):
        (images / pid).mkdir(parents=True)
        (images / pid / "front.jpg").write_bytes(b"\xff\xd8fake")
    out = tmp_path / "outputs"

    r1 = cal._resolve_image({"image": "front.jpg", "pair_id": "001"}, out, images)
    r2 = cal._resolve_image({"image": "front.jpg", "pair_id": "002"}, out, images)

    assert r1 == images / "001" / "front.jpg"
    assert r2 == images / "002" / "front.jpg"   # НЕ схлопнуто к первой паре


def test_pair_id_without_matching_folder_resolves_to_none(tmp_path):
    images = tmp_path / "local_pairs"
    (images / "001").mkdir(parents=True)
    (images / "001" / "front.jpg").write_bytes(b"\xff\xd8fake")
    # запрошен pair_id 002 — честный отказ, а не чужой 001/front.jpg
    assert cal._resolve_image(
        {"image": "front.jpg", "pair_id": "002"}, tmp_path / "out", images) is None


def test_report_without_image_key_is_skipped_not_crash(tmp_path):
    # отчёт без ключа image раньше ронял калибровку KeyError'ом (ревью 2026-07-05)
    assert cal._resolve_image({"pair_id": "001"}, tmp_path, tmp_path) is None


@pytest.mark.parametrize("argv,exp_nulls,exp_images", [
    (["outs", "--nulls", "5", "--images", "imgs"], 5, "imgs"),   # пробельная форма
    (["outs", "--nulls=7", "--images=imgs"], 7, "imgs"),          # =-форма
    (["outs"], 15, None),                                         # дефолты
])
def test_arg_parser_accepts_both_flag_forms(argv, exp_nulls, exp_images):
    # пробельная форма из docstring раньше падала IndexError'ом (ревью 2026-07-05)
    ns = cal._build_parser().parse_args(argv)
    assert ns.out_dir == "outs"
    assert ns.nulls == exp_nulls
    assert ns.images == exp_images
