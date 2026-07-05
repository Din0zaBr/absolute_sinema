"""Раскладка фото выезда: сцены (общий кадр front у соседних ям) должны
попадать в колонку scene журнала — на них держится честный по-сценный recall
и train/val-сплит без утечки одинаковых кадров."""
import csv
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ingest_local_pairs", ROOT / "scripts" / "ingest_local_pairs.py")
ing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ing)


def _jpg(path: Path, color=(120, 120, 120)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path, "JPEG")
    return path


def _pit(src: Path, n: int, front_color=(120, 120, 120)) -> Path:
    pit = src / f"Яма {n}"
    _jpg(pit / "Front" / "f.jpg", front_color)
    _jpg(pit / "Back" / "b.jpg", (60, 60, 60))
    _jpg(pit / "Эталоны" / "r.jpg", (200, 200, 200))
    return pit


def _run(monkeypatch, src: Path, ds: Path) -> int:
    monkeypatch.setattr(sys, "argv",
                        ["ingest_local_pairs.py", str(src), "--datasets", str(ds)])
    return ing.main()


def test_scene_groups_by_identical_front_land_in_journal(tmp_path, monkeypatch):
    src = tmp_path / "Ямки"
    p1 = _pit(src, 1)
    p2 = _pit(src, 2, front_color=(10, 10, 10))
    _pit(src, 3, front_color=(240, 240, 240))
    # сборщик дублирует ОБЩИЙ кадр по папкам соседних ям: front ямы 2 = front ямы 1
    shutil.copy2(p1 / "Front" / "f.jpg", p2 / "Front" / "f.jpg")
    ds = tmp_path / "datasets"

    assert _run(monkeypatch, src, ds) == 0

    with (ds / "journal.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    assert rows["001"]["scene"] == "001"
    assert rows["002"]["scene"] == "001"   # общий кадр -> общая сцена
    assert rows["003"]["scene"] == "003"
    assert (ds / "local_pairs" / "002" / "front.jpg").exists()
    assert (ds / "gt_photos" / "003_ruler_1.jpg").exists()


def test_second_run_is_idempotent_and_writes_draft_not_journal(tmp_path, monkeypatch):
    src = tmp_path / "Ямки"
    _pit(src, 1)
    ds = tmp_path / "datasets"
    assert _run(monkeypatch, src, ds) == 0
    journal_before = (ds / "journal.csv").read_bytes()

    assert _run(monkeypatch, src, ds) == 0

    assert (ds / "journal.csv").read_bytes() == journal_before
    draft = ds / "journal_draft.csv"
    assert draft.exists()
    with draft.open(encoding="utf-8-sig", newline="") as f:
        assert [r["scene"] for r in csv.DictReader(f)] == ["001"]
