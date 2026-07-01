import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from road_defect import cli


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


class FakePairPipe:
    def __init__(self):
        self.calls = []

    def analyze_pair(self, a, b):
        a = Path(a)
        b = Path(b)
        self.calls.append((a.name, b.name))
        overlay = np.zeros((8, 8, 3), dtype=np.uint8)
        report = {
            "image": a.name,
            "mode": "two_view_unmatched",
            "image_size_px": [8, 8],
            "scale": {"available": False},
            "defects": [],
            "warnings": [],
            "fusion": {"fused": False, "matched": False},
        }
        return report, overlay, overlay.copy(), []


def test_discover_pairs_finds_named_front_back_and_reports_bad_folders(tmp_path):
    root = tmp_path / "pairs"
    _touch(root / "001" / "front.jpg")
    _touch(root / "001" / "back.jpg")
    _touch(root / "002" / "a.png")
    _touch(root / "002" / "b.png")
    _touch(root / "broken" / "front.jpg")

    pairs, errors = cli._discover_pairs(root)

    assert [(pair_id, front.name, back.name) for pair_id, front, back in pairs] == [
        ("001", "front.jpg", "back.jpg"),
        ("002", "a.png", "b.png"),
    ]
    assert any("broken" in err for err in errors)


def test_run_pairs_dir_writes_reports_overlays_and_summary(tmp_path):
    root = tmp_path / "pairs"
    _touch(root / "001" / "front.jpg")
    _touch(root / "001" / "back.jpg")
    _touch(root / "002" / "front.jpg")
    _touch(root / "002" / "back.jpg")
    out = tmp_path / "out"
    pipe = FakePairPipe()

    assert cli._run_pairs_dir(pipe, str(root), out) == 0

    json_names = sorted(p.name for p in out.glob("*.json"))
    assert json_names == [
        "001__front__back_pair.json",
        "002__front_2__back_2_pair.json",
    ]
    assert (out / "001__front__back_pair_annotated.jpg").exists()
    assert (out / "002__front_2__back_2_pair_annotated.jpg").exists()

    report = json.loads((out / json_names[0]).read_text(encoding="utf-8"))
    assert report["pair_id"] == "001"
    assert report["pair_inputs"]["front"].endswith("front.jpg")
    assert report["pair_inputs"]["back"].endswith("back.jpg")

    with (out / "pairs_summary.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["pair_id"] for row in rows] == ["001", "002"]
    assert [row["report"] for row in rows] == json_names
    assert rows[0]["fusion_fused"] == "false"
    assert rows[0]["defects"] == "0"


def test_run_pairs_dir_returns_error_when_no_pairs(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()

    assert cli._run_pairs_dir(FakePairPipe(), str(root), tmp_path / "out") == 1
