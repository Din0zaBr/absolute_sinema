"""Регрессия репро агента-скептика (2026-07-02): прерванная распаковка zip
не должна молча давать усечённый источник при повторном запуске.
Распаковка обязана быть атомарной (tmp + rename) и самолечащейся."""
import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "prepare_rdd2022", ROOT / "scripts" / "prepare_rdd2022.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

N = 6


def _make_zip(zip_path: Path) -> list[str]:
    """Мини-архив со структурой RDD2022: annotations сортируются раньше images —
    как в реальном архиве, поэтому прерывание режет именно картинки."""
    with zipfile.ZipFile(zip_path, "w") as z:
        for i in range(N):
            z.writestr(f"ds/train/annotations/xmls/img_{i}.xml", "<annotation/>")
        for i in range(N):
            z.writestr(f"ds/train/images/img_{i}.jpg", b"\xff\xd8fake")
    with zipfile.ZipFile(zip_path) as z:
        return z.namelist()


def _n_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file())


def test_clean_zip_extracted_fully(tmp_path):
    zip_path = tmp_path / "ds.zip"
    _make_zip(zip_path)
    out = prep._ensure_extracted(zip_path)
    assert out == tmp_path / "ds"
    assert _n_files(out) == 2 * N
    assert not (tmp_path / "ds.extracting").exists()   # tmp не остаётся


def test_partial_extraction_healed_on_rerun(tmp_path):
    # Старый баг: частичная папка под финальным именем (наследие прерванного
    # extractall) молча считалась готовым источником → усечённый датасет.
    zip_path = tmp_path / "ds.zip"
    names = _make_zip(zip_path)
    partial = tmp_path / "ds"
    with zipfile.ZipFile(zip_path) as z:
        for name in names[:N + 2]:      # все xml + 2 из N картинок
            z.extract(name, partial)
    assert _n_files(partial) == N + 2

    out = prep._ensure_extracted(zip_path)
    assert _n_files(out) == 2 * N       # недостача обнаружена и вылечена


def test_interrupted_tmp_dir_is_replaced(tmp_path):
    # Новый путь атомарен: прерывание оставляет только ds.extracting, финального
    # имени нет — повторный запуск сносит огрызок и распаковывает заново.
    zip_path = tmp_path / "ds.zip"
    _make_zip(zip_path)
    leftover = tmp_path / "ds.extracting"
    (leftover / "ds" / "train").mkdir(parents=True)
    (leftover / "ds" / "train" / "junk.bin").write_bytes(b"x")

    out = prep._ensure_extracted(zip_path)
    assert _n_files(out) == 2 * N
    assert not leftover.exists()


def test_replaced_zip_triggers_reextract(tmp_path):
    # Ревью 2026-07-03: подменённый архив с ТЕМ ЖЕ числом файлов, но другим
    # содержимым — счёт файлов молчит, устаревание ловит только штамп-подпись.
    zip_path = tmp_path / "ds.zip"
    _make_zip(zip_path)
    prep._ensure_extracted(zip_path)

    zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w") as z:
        for i in range(N):
            z.writestr(f"ds/train/annotations/xmls/img_{i}.xml", "<annotation/>")
        for i in range(N):
            z.writestr(f"ds/train/images/new_{i}.jpg", b"\xff\xd8new!")

    out = prep._ensure_extracted(zip_path)
    assert (out / "ds" / "train" / "images" / "new_0.jpg").exists()
    assert not (out / "ds" / "train" / "images" / "img_0.jpg").exists()


def test_legacy_extraction_without_stamp_is_reextracted(tmp_path):
    # Legacy-копия (до введения штампа): последняя картинка не извлечена, но
    # недостачу маскирует посторонний Thumbs.db — счёт сходится. Штампа нет →
    # копия не доверяется и распаковывается заново.
    zip_path = tmp_path / "ds.zip"
    names = _make_zip(zip_path)
    legacy = tmp_path / "ds"
    with zipfile.ZipFile(zip_path) as z:
        for name in names[:-1]:
            z.extract(name, legacy)
    (legacy / "Thumbs.db").write_bytes(b"junk")
    assert _n_files(legacy) == 2 * N        # маскировка удалась — счёт молчит

    out = prep._ensure_extracted(zip_path)
    assert (out / "ds" / "train" / "images" / f"img_{N - 1}.jpg").exists()
    assert not (out / "Thumbs.db").exists()


def test_manual_deletion_healed_despite_valid_stamp(tmp_path):
    # Вторая линия защиты: штамп цел (архив не менялся), но из копии вручную
    # удалили файл — недостачу ловит сверка числа файлов.
    zip_path = tmp_path / "ds.zip"
    _make_zip(zip_path)
    out = prep._ensure_extracted(zip_path)
    (out / "ds" / "train" / "images" / "img_0.jpg").unlink()

    out = prep._ensure_extracted(zip_path)
    assert _n_files(out) == 2 * N


def test_non_zip_source_passthrough(tmp_path):
    folder = tmp_path / "already_extracted"
    folder.mkdir()
    assert prep._ensure_extracted(folder) == folder
