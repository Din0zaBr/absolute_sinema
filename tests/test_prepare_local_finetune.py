"""Раскладка пар в YOLO-скелет для дообучения (DATASET.md §6): дедуп сцен по
md5 кадра front, сплит по сценам без утечки, data.yaml, контракт классов."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "prepare_local_finetune", ROOT / "scripts" / "prepare_local_finetune.py")
pmf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pmf)


def _pair(root: Path, pid: str, front: bytes, back: bytes) -> None:
    d = root / pid
    d.mkdir(parents=True)
    (d / "front.jpg").write_bytes(front)
    (d / "back.jpg").write_bytes(back)


def _img_pair(root: Path, pid: str, front_color, back_color) -> None:
    """Пара с ВАЛИДНЫМИ JPEG (main проверяет декодируемость). Одинаковый
    front_color -> одинаковые байты -> одна сцена (дедуп по md5)."""
    from PIL import Image
    d = root / pid
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), front_color).save(d / "front.jpg", "JPEG")
    Image.new("RGB", (16, 16), back_color).save(d / "back.jpg", "JPEG")


def test_scene_reps_dedup_by_front_md5(tmp_path):
    # 001 и 002 имеют ОДИН кадр front -> одна сцена (rep = наименьший id 001);
    # 003 — отдельная сцена. Соседние ямы в общем кадре не плодят копий.
    _pair(tmp_path, "001", b"FRONT_A", b"back1")
    _pair(tmp_path, "002", b"FRONT_A", b"back2")   # тот же front, что у 001
    _pair(tmp_path, "003", b"FRONT_B", b"back3")
    reps = pmf.scene_reps(tmp_path)
    ids = [s for s, _, _ in reps]
    assert ids == ["001", "003"]                    # 002 свёрнута в сцену 001
    assert reps[0][1] == tmp_path / "001" / "front.jpg"
    assert reps[0][2] == tmp_path / "001" / "back.jpg"


def test_assign_splits_deterministic_and_spread():
    scenes = [f"{i:03d}" for i in range(1, 15)]      # 14 сцен
    a = pmf.assign_splits(scenes, 0.15)
    b = pmf.assign_splits(scenes, 0.15)
    assert a == b                                    # без ГСЧ — воспроизводимо
    # n_val = round(14·0.15)=2, разнесены: индексы 3,10 -> 004, 011
    val = sorted(s for s, sp in a.items() if sp == "val")
    assert val == ["004", "011"]
    assert sum(v == "train" for v in a.values()) == 12


def test_assign_splits_min_one_val_and_train_tiny_set():
    # 2 сцены: val и train оба непусты (n_val зажат в [1, n-1])
    sp = pmf.assign_splits(["001", "003"], 0.15)
    assert sorted(s for s, v in sp.items() if v == "val") == ["003"]
    assert sp["001"] == "train"


def test_assign_splits_high_val_frac_not_capped_at_50pct():
    # прошлая схема max(2, round(1/frac)) резала долю val до 50%; теперь честно
    scenes = [f"{i:03d}" for i in range(1, 11)]      # 10 сцен
    sp = pmf.assign_splits(scenes, 0.7)
    assert sum(v == "val" for v in sp.values()) == 7   # ~70%, не 50%
    assert sum(v == "train" for v in sp.values()) == 3


def test_assign_splits_single_scene_all_train():
    assert pmf.assign_splits(["001"], 0.15) == {"001": "train"}


def test_data_yaml_has_classes_paths(tmp_path):
    txt = pmf.data_yaml_text(tmp_path / "ds", pmf.CLASS_NAMES)
    assert "train: images/train" in txt and "val: images/val" in txt
    assert "3: D40" in txt and "4: Repair" in txt
    assert "path: " in txt and "ds" in txt           # абсолютный путь набора


def test_class_names_match_finetune_contract():
    # порядок классов ДОЛЖЕН совпадать с finetune_rdd2022 (веса rezzzq)
    spec = importlib.util.spec_from_file_location(
        "finetune_rdd2022", ROOT / "scripts" / "finetune_rdd2022.py")
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)
    assert pmf.CLASS_NAMES == ft.CLASS_NAMES


def test_main_lays_out_skeleton(tmp_path, monkeypatch, capsys):
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    _img_pair(pairs, "001", (100, 0, 0), (0, 0, 1))
    _img_pair(pairs, "002", (100, 0, 0), (0, 0, 2))  # тот же front -> сцена 001
    _img_pair(pairs, "003", (0, 100, 0), (0, 0, 3))
    out = tmp_path / "yolo"
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs), "--out", str(out)])
    assert pmf.main() == 0

    imgs = sorted(p.name for p in (out / "images").rglob("*.jpg"))
    # 2 сцены × (front+back) = 4 кадра, 002 свёрнута (нет 002_*)
    assert imgs == ["001_back.jpg", "001_front.jpg", "003_back.jpg", "003_front.jpg"]
    # на каждый кадр — пустая заготовка label
    label_files = sorted((out / "labels").rglob("*.txt"))
    assert [p.name for p in label_files] == [
        "001_back.txt", "001_front.txt", "003_back.txt", "003_front.txt"]
    assert all(p.read_text() == "" for p in label_files)   # пустые заготовки
    assert (out / "data.yaml").is_file()

    # ни одна сцена не в обоих сплитах (стемы train ∩ val пусты)
    train = {p.stem for p in (out / "images" / "train").glob("*.jpg")}
    val = {p.stem for p in (out / "images" / "val").glob("*.jpg")}
    assert train and val and not (train & val)


def test_main_missing_pairs_dir_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--pairs", str(tmp_path / "nope")])
    assert pmf.main() == 2


def _run(pairs, out, monkeypatch, val_frac=None):
    argv = ["prog", "--pairs", str(pairs), "--out", str(out)]
    if val_frac is not None:
        argv += ["--val-frac", str(val_frac)]
    monkeypatch.setattr(sys, "argv", argv)
    return pmf.main()


def test_rerun_changed_split_has_no_leakage(tmp_path, monkeypatch):
    # [1]: сцена, сменившая сплит при повторном прогоне, НЕ должна оказаться в
    # обоих images/train и images/val (утечка). Устаревшие копии зачищаются.
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    for i in range(1, 7):                              # 6 сцен (разные front)
        _img_pair(pairs, f"{i:03d}", (30 * i, 0, 0), (0, 30 * i, 0))
    out = tmp_path / "yolo"

    assert _run(pairs, out, monkeypatch, 0.15) == 0    # ~1 val
    assert _run(pairs, out, monkeypatch, 0.5) == 0     # ~3 val -> сцены переехали

    def scenes_in(split):
        return {p.name.split("_")[0] for p in (out / "images" / split).glob("*.jpg")}
    train, val = scenes_in("train"), scenes_in("val")
    assert train and val and not (train & val)         # ни одна не в обоих
    assert train | val == {f"{i:03d}" for i in range(1, 7)}
    # никаких осиротевших кадров сверх 6 сцен × 2 вида
    assert len(list((out / "images").rglob("*.jpg"))) == 12


def test_single_scene_points_val_at_train(tmp_path, monkeypatch, capsys):
    # [2]: одна сцена -> val пуст; data.yaml должен указывать val на train,
    # иначе finetune упал бы на несуществующем images/val.
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    _img_pair(pairs, "001", (100, 0, 0), (0, 100, 0))
    out = tmp_path / "yolo"
    assert _run(pairs, out, monkeypatch) == 0
    assert not (out / "images" / "val").exists()
    assert "val: images/train" in (out / "data.yaml").read_text(encoding="utf-8")
    assert "val пуст" in capsys.readouterr().out


def test_holdout_scene_is_excluded_from_training_set(tmp_path, monkeypatch, capsys):
    # Гейт контаминации P0: сцена из holdout_scenes.txt не попадает в набор
    # дообучения (eval никогда не в обучении, docs/EVAL.md).
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    for i in (1, 2, 3):
        _img_pair(pairs, f"{i:03d}", (40 * i, 0, 0), (0, 40 * i, 0))
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("# eval_v1\n002\n", encoding="utf-8")
    out = tmp_path / "yolo"
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs), "--out", str(out),
                         "--holdout", str(holdout)])
    assert pmf.main() == 0
    scenes = {p.name.split("_")[0] for p in (out / "images").rglob("*.jpg")}
    assert scenes == {"001", "003"}
    assert "held-out" in capsys.readouterr().out


def test_holdout_excludes_whole_md5_group_by_any_member(tmp_path, monkeypatch):
    # 001 и 002 — одна фотография front (одна сцена, rep=001). Holdout называет
    # только 002 — исключиться обязана ВСЯ группа: тот же кадр под id 001 в
    # обучении сделал бы eval нечестным.
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    _img_pair(pairs, "001", (10, 0, 0), (0, 10, 0))
    _img_pair(pairs, "002", (10, 0, 0), (0, 20, 0))   # тот же front, что 001
    _img_pair(pairs, "003", (30, 0, 0), (0, 30, 0))
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("002\n", encoding="utf-8")
    out = tmp_path / "yolo"
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs), "--out", str(out),
                         "--holdout", str(holdout)])
    assert pmf.main() == 0
    scenes = {p.name.split("_")[0] for p in (out / "images").rglob("*.jpg")}
    assert scenes == {"003"}                          # группа 001+002 исключена


def test_explicit_missing_holdout_is_error_not_silent_off(tmp_path, monkeypatch, capsys):
    # Ревью 2026-07-07 [HIGH]: опечатка в явном --holdout НЕ должна молча
    # выключать гейт контаминации — это ошибка (отключение только --holdout '').
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    _img_pair(pairs, "001", (10, 0, 0), (0, 10, 0))
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs),
                         "--out", str(tmp_path / "yolo"),
                         "--holdout", str(tmp_path / "no_such.txt")])
    assert pmf.main() == 2
    assert "не найден" in capsys.readouterr().out


def test_default_holdout_is_resolved_next_to_pairs(tmp_path, monkeypatch, capsys):
    # Пин дефолтной проводки (ревью: мутации «дефолт выключен» и «дефолт от
    # cwd» выживали): без --holdout берётся <pairs>/../eval_v1/holdout_scenes.txt.
    # id 555/777 вне боевого списка — cwd-мутация не смогла бы их исключить.
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    _img_pair(pairs, "555", (10, 0, 0), (0, 10, 0))
    _img_pair(pairs, "777", (30, 0, 0), (0, 30, 0))
    (tmp_path / "eval_v1").mkdir()
    (tmp_path / "eval_v1" / "holdout_scenes.txt").write_text("555\n",
                                                             encoding="utf-8")
    out = tmp_path / "yolo"
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs), "--out", str(out)])
    assert pmf.main() == 0
    scenes = {p.name.split("_")[0] for p in (out / "images").rglob("*.jpg")}
    assert scenes == {"777"}
    assert "гейт активен" in capsys.readouterr().out


def test_default_holdout_missing_warns_gate_inactive(tmp_path, monkeypatch, capsys):
    # Свежий клон/нет eval-набора: гейт неактивен, но об этом ГРОМКО сказано
    # (молчаливо-неактивная защита — ложная гарантия, ревью 2026-07-07).
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    _img_pair(pairs, "001", (10, 0, 0), (0, 10, 0))
    _img_pair(pairs, "003", (30, 0, 0), (0, 30, 0))
    out = tmp_path / "yolo"
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs), "--out", str(out)])
    assert pmf.main() == 0
    scenes = {p.name.split("_")[0] for p in (out / "images").rglob("*.jpg")}
    assert scenes == {"001", "003"}
    assert "НЕАКТИВЕН" in capsys.readouterr().out


def test_holdout_with_bom_still_excludes_first_scene(tmp_path, monkeypatch):
    # BOM Windows-редактора «съедал» первую строку -> сцена молча оставалась
    # в обучении (частичная контаминация, ревью 2026-07-07).
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    _img_pair(pairs, "001", (10, 0, 0), (0, 10, 0))
    _img_pair(pairs, "003", (30, 0, 0), (0, 30, 0))
    holdout = tmp_path / "holdout.txt"
    holdout.write_bytes("001\n".encode("utf-8-sig"))
    out = tmp_path / "yolo"
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs), "--out", str(out),
                         "--holdout", str(holdout)])
    assert pmf.main() == 0
    scenes = {p.name.split("_")[0] for p in (out / "images").rglob("*.jpg")}
    assert scenes == {"003"}


def test_holdout_in_utf16_is_clear_error(tmp_path, monkeypatch, capsys):
    # PowerShell `>` пишет UTF-16: вместо трейсбека — внятный отказ.
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    _img_pair(pairs, "001", (10, 0, 0), (0, 10, 0))
    holdout = tmp_path / "holdout.txt"
    holdout.write_bytes("001\n".encode("utf-16"))
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs),
                         "--out", str(tmp_path / "yolo"),
                         "--holdout", str(holdout)])
    assert pmf.main() == 2
    assert "UTF-8" in capsys.readouterr().out


def test_holdout_excludes_scene_by_frame_content_md5(tmp_path, monkeypatch, capsys):
    # Контентная сверка (ревью: группы клеятся только по front — back-канал):
    # кадр сцены 777 байт-в-байт лежит в eval-манифесте -> сцена исключается,
    # хотя её id в holdout НЕ назван.
    pytest.importorskip("PIL")
    import hashlib
    pairs = tmp_path / "local_pairs"
    _img_pair(pairs, "555", (10, 0, 0), (0, 10, 0))
    _img_pair(pairs, "777", (30, 0, 0), (0, 30, 0))
    _img_pair(pairs, "888", (50, 0, 0), (0, 50, 0))
    ev = tmp_path / "eval_v1"
    ev.mkdir()
    (ev / "holdout_scenes.txt").write_text("555\n", encoding="utf-8")
    back_md5 = hashlib.md5((pairs / "777" / "back.jpg").read_bytes()).hexdigest()
    (ev / "manifest.csv").write_text(
        "frame_id,scene,view,group,source,md5\n"
        f"x_back,x,back,clean,y,{back_md5}\n", encoding="utf-8-sig")
    out = tmp_path / "yolo"
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs), "--out", str(out)])
    assert pmf.main() == 0
    scenes = {p.name.split("_")[0] for p in (out / "images").rglob("*.jpg")}
    assert scenes == {"888"}                          # 555 по id, 777 по md5
    assert "СОДЕРЖИМОМУ" in capsys.readouterr().out


def test_grown_holdout_moves_hand_labels_instead_of_deleting(tmp_path, monkeypatch):
    # Ревью 2026-07-07: рост holdout-списка на уже размеченной раскладке
    # стирал бы часы ручной работы — непустые label уезжают в labels_removed/.
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    _img_pair(pairs, "001", (10, 0, 0), (0, 10, 0))
    _img_pair(pairs, "003", (30, 0, 0), (0, 30, 0))
    out = tmp_path / "yolo"
    assert _run(pairs, out, monkeypatch) == 0
    lbl = next((out / "labels").rglob("001_front.txt"))
    lbl.write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")   # ручная разметка

    holdout = tmp_path / "holdout.txt"
    holdout.write_text("001\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs), "--out", str(out),
                         "--holdout", str(holdout)])
    assert pmf.main() == 0
    assert not list((out / "labels").rglob("001_front.txt"))
    saved = out / "labels_removed" / "001_front.txt"
    assert saved.read_text(encoding="utf-8").startswith("3 0.5")


def test_all_scenes_held_out_refuses(tmp_path, monkeypatch, capsys):
    pytest.importorskip("PIL")
    pairs = tmp_path / "local_pairs"
    _img_pair(pairs, "001", (10, 0, 0), (0, 10, 0))
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("001\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--pairs", str(pairs),
                         "--out", str(tmp_path / "yolo"),
                         "--holdout", str(holdout)])
    assert pmf.main() == 2
    assert "пуст" in capsys.readouterr().out


def test_unreadable_frame_is_skipped(tmp_path, monkeypatch, capsys):
    # [4]: битый кадр не должен уехать в набор (иначе даталоадер обучения упадёт).
    pytest.importorskip("PIL")
    from PIL import Image

    def valid(pid, view, color):
        d = pairs / pid
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color).save(d / f"{view}.jpg", "JPEG")

    pairs = tmp_path / "local_pairs"
    valid("001", "front", (9, 9, 9))                    # валидный front
    (pairs / "001" / "back.jpg").write_bytes(b"\xff\xd8truncated")  # битый back
    valid("002", "front", (200, 200, 200))              # иной front -> 2-я сцена
    valid("002", "back", (50, 50, 50))
    out = tmp_path / "yolo"

    assert _run(pairs, out, monkeypatch) == 0
    imgs = {p.name for p in (out / "images").rglob("*.jpg")}
    assert "001_front.jpg" in imgs
    assert "001_back.jpg" not in imgs                   # битый back пропущен
    assert "back" in capsys.readouterr().out.lower()
