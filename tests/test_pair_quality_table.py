"""Таблица качества пар: «найдена» = есть pothole-детекция в виде
(select_status != no_pothole), а НЕ pothole_found (выбор доминирующей ямы) —
иначе recall занижается ровно на сценах с несколькими ямами
(регресс первого прогона 48 пар: 94% вместо честных 98%)."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "pair_quality_table", ROOT / "scripts" / "pair_quality_table.py")
pqt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pqt)


def _report(out: Path, pid: str, front_status: str, back_status: str,
            potholes=(), matched=False, fused=False) -> None:
    report = {
        "pair_id": pid,
        "mode": "two_view_fused" if fused else "two_view_unmatched",
        "scale": {"available": False},
        "defects": [{"class": "pothole", "confidence": c} for c in potholes],
        "fusion": {
            "matched": matched, "fused": fused,
            "per_view": [
                {"image": "front.jpg", "select_status": front_status,
                 "pothole_found": front_status == "ok"},
                {"image": "back.jpg", "select_status": back_status,
                 "pothole_found": back_status == "ok"},
            ],
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{pid}__front__back_pair.json").write_text(
        json.dumps(report), encoding="utf-8")


def _journal(path: Path, scenes: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "scene", "type"])
        w.writeheader()
        w.writerows({"id": pid, "scene": scene, "type": "pothole"}
                    for pid, scene in scenes.items())


def _run(monkeypatch, out: Path, journal: Path) -> list[dict]:
    monkeypatch.setattr(sys, "argv", ["pair_quality_table.py",
                                      "--outputs", str(out),
                                      "--journal", str(journal)])
    assert pqt.main() == 0
    with (out / "quality_table.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_ambiguous_multiple_potholes_counts_as_detected(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    _report(out, "001", "ambiguous_multiple_potholes", "ambiguous_multiple_potholes",
            potholes=(0.85, 0.80))
    journal = tmp_path / "journal.csv"
    _journal(journal, {"001": "001"})

    rows = {r["pair_id"]: r for r in _run(monkeypatch, out, journal)}

    assert rows["001"]["front_detected"] == "True"
    assert rows["001"]["back_detected"] == "True"
    assert rows["001"]["detected_both"] == "True"
    assert rows["001"]["matched"] == "False"


def test_no_pothole_view_and_scene_fallback_without_journal(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    _report(out, "001", "ok", "no_pothole", potholes=(0.7,))
    _report(out, "002", "no_pothole", "no_pothole")

    rows = {r["pair_id"]: r for r in _run(monkeypatch, out,
                                          tmp_path / "нет_журнала.csv")}

    assert rows["001"]["front_detected"] == "True"
    assert rows["001"]["back_detected"] == "False"
    assert rows["002"]["detected_any"] == "False"
    # журнала нет -> сцена честно падает в pair_id, таблица не падает
    assert rows["001"]["scene"] == "001"
    assert rows["002"]["scene"] == "002"


def test_scene_metrics_dedupe_pairs_of_one_scene(tmp_path, monkeypatch, capsys):
    out = tmp_path / "outputs"
    # сцена 001: две пары с одинаковым результатом (общий кадр)
    _report(out, "001", "ok", "ok", potholes=(0.8,), matched=True)
    _report(out, "002", "ok", "ok", potholes=(0.8,), matched=True)
    # сцена 003: яма не найдена вовсе
    _report(out, "003", "no_pothole", "no_pothole")
    journal = tmp_path / "journal.csv"
    _journal(journal, {"001": "001", "002": "001", "003": "003"})

    rows = _run(monkeypatch, out, journal)
    printed = capsys.readouterr().out

    assert [r["scene"] for r in rows] == ["001", "001", "003"]
    assert "scene_recall_any:  1/2" in printed
    assert "recall_any  (яма хотя бы в одном виде): 2/3" in printed
    assert "011" not in printed  # пропуски берутся из данных, не захардкожены
    assert "пары 003" in printed
