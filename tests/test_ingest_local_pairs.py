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


def _run(monkeypatch, src: Path, ds: Path, *extra: str) -> int:
    monkeypatch.setattr(sys, "argv",
                        ["ingest_local_pairs.py", str(src), "--datasets", str(ds),
                         *extra])
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
    assert (ds / "gt_photos" / "003_ruler_1.jpg").exists()


def test_duplicate_scene_pair_not_copied_but_rulers_and_journal_kept(tmp_path, monkeypatch):
    # Дедуп пар (2026-07-16): пара копируется ОДИН раз на сцену — под id
    # ямы-представителя; у ямы-дубля остаются рулетки и строка журнала с пометкой.
    src = tmp_path / "Ямки"
    p1 = _pit(src, 1)
    p2 = _pit(src, 2, front_color=(10, 10, 10))
    shutil.copy2(p1 / "Front" / "f.jpg", p2 / "Front" / "f.jpg")
    ds = tmp_path / "datasets"

    assert _run(monkeypatch, src, ds) == 0

    assert (ds / "local_pairs" / "001" / "front.jpg").exists()
    assert not (ds / "local_pairs" / "002").exists()   # дубль сцены не копируется
    assert (ds / "gt_photos" / "002_ruler_1.jpg").exists()  # рулетки индивидуальны
    with (ds / "journal.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    assert rows["002"]["scene"] == "001"
    assert "байтовый дубль пары 001" in rows["002"]["notes"]
    assert "дубль" not in rows["001"]["notes"]


def test_prune_dups_removes_stale_pair_dirs_only_with_flag(tmp_path, monkeypatch):
    # Остатки прогонов до дедупа: без флага папка-дубль лежит нетронутой
    # (молча не удаляем), с --prune-dups — вычищается.
    src = tmp_path / "Ямки"
    p1 = _pit(src, 1)
    p2 = _pit(src, 2)
    shutil.copy2(p1 / "Front" / "f.jpg", p2 / "Front" / "f.jpg")
    ds = tmp_path / "datasets"
    stale = ds / "local_pairs" / "002"   # как будто создана старым ингестом
    stale.mkdir(parents=True)
    shutil.copy2(p2 / "Front" / "f.jpg", stale / "front.jpg")
    shutil.copy2(p2 / "Back" / "b.jpg", stale / "back.jpg")

    assert _run(monkeypatch, src, ds) == 0
    assert stale.exists()                # без флага — только предупреждение

    assert _run(monkeypatch, src, ds, "--prune-dups") == 0
    assert not stale.exists()
    assert (ds / "local_pairs" / "001" / "front.jpg").exists()


def test_shared_front_with_individual_back_is_not_a_dup(tmp_path, monkeypatch):
    # Дубль = совпадение ОБОИХ кадров. Общий front при индивидуальном back —
    # уникальная пара: копируется под своим id (ревью цикла 18: дедуп по
    # одному front молча терял бы уникальный back).
    src = tmp_path / "Ямки"
    p1 = _pit(src, 1)
    p2 = _pit(src, 2, front_color=(10, 10, 10))
    shutil.copy2(p1 / "Front" / "f.jpg", p2 / "Front" / "f.jpg")
    _jpg(p2 / "Back" / "b.jpg", (90, 90, 90))   # back отличается от ямы 1
    ds = tmp_path / "datasets"

    assert _run(monkeypatch, src, ds) == 0

    assert (ds / "local_pairs" / "002" / "back.jpg").exists()  # пара скопирована
    with (ds / "journal.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    assert rows["002"]["scene"] == "001"        # сцена — по-прежнему по front
    assert "back" in rows["002"]["notes"]       # пометка про индивидуальный back


def test_prune_dups_never_touches_non_byte_dup_folder(tmp_path, monkeypatch):
    # Совпадение ИМЕНИ папки с id дубля — не повод сносить чужое содержимое:
    # удаляется только проверенный байтовый дубль (ревью цикла 18).
    src = tmp_path / "Ямки"
    p1 = _pit(src, 1)
    p2 = _pit(src, 2)
    shutil.copy2(p1 / "Front" / "f.jpg", p2 / "Front" / "f.jpg")
    ds = tmp_path / "datasets"
    alien = ds / "local_pairs" / "002"          # под id дубля — ЧУЖАЯ пара
    _jpg(alien / "front.jpg", (1, 2, 3))
    _jpg(alien / "back.jpg", (4, 5, 6))
    sidecar = ds / "local_pairs" / "002" / "manual_notes.txt"

    assert _run(monkeypatch, src, ds, "--prune-dups") == 0
    assert alien.exists()                       # содержимое не совпало — не тронута
    assert (alien / "front.jpg").exists()

    # и байтовый дубль с посторонним файлом тоже не удаляется целиком
    shutil.copy2(p2 / "Front" / "f.jpg", alien / "front.jpg")
    shutil.copy2(p2 / "Back" / "b.jpg", alien / "back.jpg")
    sidecar.write_text("ручные пометки", encoding="utf-8")
    assert _run(monkeypatch, src, ds, "--prune-dups") == 0
    assert sidecar.exists()


def test_missing_ruler_is_soft_warning_not_exit_failure(tmp_path, monkeypatch):
    # Пара без кадра-рулетки копируется целиком — это мягкое замечание, а не
    # сбой: exit code должен остаться 0 (ревью 2026-07-05).
    src = tmp_path / "Ямки"
    pit = src / "Яма 1"
    _jpg(pit / "Front" / "f.jpg")
    _jpg(pit / "Back" / "b.jpg")
    # папки «Эталоны» нет вовсе
    ds = tmp_path / "datasets"

    assert _run(monkeypatch, src, ds) == 0
    assert (ds / "local_pairs" / "001" / "front.jpg").exists()
    with (ds / "journal.csv").open(encoding="utf-8-sig", newline="") as f:
        assert [r["id"] for r in csv.DictReader(f)] == ["001"]


def test_corrupt_ruler_does_not_lose_valid_pair(tmp_path, monkeypatch):
    # Битая рулетка (некритичный gt-кадр) не должна ронять валидную пару:
    # пара копируется, рулетка пропускается, exit 0 (ревью 2026-07-05).
    src = tmp_path / "Ямки"
    pit = src / "Яма 1"
    _jpg(pit / "Front" / "f.jpg")
    _jpg(pit / "Back" / "b.jpg")
    (pit / "Эталоны").mkdir(parents=True)
    (pit / "Эталоны" / "r.jpg").write_bytes(b"not a jpeg")  # битый файл
    ds = tmp_path / "datasets"

    assert _run(monkeypatch, src, ds) == 0
    assert (ds / "local_pairs" / "001" / "front.jpg").exists()
    assert (ds / "local_pairs" / "001" / "back.jpg").exists()
    assert not (ds / "gt_photos" / "001_ruler_1.jpg").exists()  # битая рулетка не скопирована
    with (ds / "journal.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    assert rows["001"]["notes"].startswith("gt_photos: 0")  # честный счёт рулеток


def test_corrupt_front_photo_skips_whole_pit(tmp_path, monkeypatch):
    # Битый front — пара непригодна, жёсткий пропуск и exit 1.
    src = tmp_path / "Ямки"
    pit = src / "Яма 1"
    (pit / "Front").mkdir(parents=True)
    (pit / "Front" / "f.jpg").write_bytes(b"not a jpeg")
    _jpg(pit / "Back" / "b.jpg")
    ds = tmp_path / "datasets"

    assert _run(monkeypatch, src, ds) == 1
    assert not (ds / "local_pairs" / "001").exists()


def test_bad_pair_still_forces_exit_failure(tmp_path, monkeypatch):
    # Жёсткий сбой (нет ровно одной пары) обязан валить код возврата в 1.
    src = tmp_path / "Ямки"
    pit = src / "Яма 1"
    _jpg(pit / "Front" / "f1.jpg")
    _jpg(pit / "Front" / "f2.jpg")  # две «передних» — пара не собирается
    _jpg(pit / "Back" / "b.jpg")

    assert _run(monkeypatch, src, tmp_path / "datasets") == 1


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
