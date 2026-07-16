"""Тесты геопривязки платформы: разбор «lat lon», журнал ям, EXIF-GPS."""
from pathlib import Path

import pytest

from road_defect.platform import geo


# --- parse_latlon -------------------------------------------------------------

def test_parse_latlon_space_separated():
    assert geo.parse_latlon("47.27743 39.70409") == pytest.approx((47.27743, 39.70409))


def test_parse_latlon_comma_separator_and_junk():
    assert geo.parse_latlon("  47.27743, 39.70409 (у школы) ") == pytest.approx(
        (47.27743, 39.70409))


def test_parse_latlon_negative():
    assert geo.parse_latlon("-33.9 151.2") == pytest.approx((-33.9, 151.2))


@pytest.mark.parametrize("text", [
    None, "", "ул. Ленина", "ул. Ленина 5",       # нет двух чисел
    "1 2 3",                                       # три числа — не координаты
    "91 30", "-91 30",                             # широта вне диапазона
    "47 181", "47 -181",                           # долгота вне диапазона
])
def test_parse_latlon_rejects_garbage(text):
    assert geo.parse_latlon(text) is None


# --- journal_latlon -----------------------------------------------------------

@pytest.fixture()
def journal(tmp_path: Path) -> Path:
    path = tmp_path / "journal.csv"
    path.write_text(
        "id,scene,date,street_or_gps,type\n"
        "006,006,2026-07-04,47.27932 39.70642,pothole\n"
        "012,012,2026-07-04,\"47.28216, 39.70977\",pothole\n"
        "013,013,2026-07-04,адрес без координат,pothole\n",
        encoding="utf-8")
    return path


def test_journal_latlon_exact_id(journal):
    assert geo.journal_latlon(journal, "006") == pytest.approx((47.27932, 39.70642))


def test_journal_latlon_zfill(journal):
    # id пары в отчёте может быть «6», в журнале — «006»
    assert geo.journal_latlon(journal, "6") == pytest.approx((47.27932, 39.70642))


def test_journal_latlon_comma_value(journal):
    assert geo.journal_latlon(journal, "12") == pytest.approx((47.28216, 39.70977))


def test_journal_latlon_row_without_coords(journal):
    assert geo.journal_latlon(journal, "013") is None


def test_journal_latlon_unknown_id(journal):
    assert geo.journal_latlon(journal, "999") is None


def test_journal_latlon_missing_file(tmp_path):
    assert geo.journal_latlon(tmp_path / "нет.csv", "006") is None


def test_journal_latlon_with_bom(tmp_path):
    # реальный datasets/journal.csv начинается с UTF-8 BOM — колонка id
    # обязана находиться и в этом случае
    path = tmp_path / "journal_bom.csv"
    path.write_text("id,street_or_gps\n007,47.1 39.2\n", encoding="utf-8-sig")
    assert geo.journal_latlon(path, "007") == pytest.approx((47.1, 39.2))


# --- exif_gps -------------------------------------------------------------------

def test_exif_gps_reads_dms(tmp_path):
    from PIL import Image

    p = tmp_path / "фото.jpg"  # кириллица в пути — обычное дело в проекте
    exif = Image.Exif()
    # GPS IFD: 1/3 — полушария, 2/4 — градусы/минуты/секунды
    exif[0x8825] = {1: "N", 2: (47.0, 16.0, 38.75), 3: "E", 4: (39.0, 42.0, 14.7)}
    Image.new("RGB", (4, 4), "gray").save(p, exif=exif)

    coords = geo.exif_gps(p)
    assert coords == pytest.approx((47.2774306, 39.7040833), abs=1e-5)


def test_exif_gps_south_west_sign(tmp_path):
    from PIL import Image

    p = tmp_path / "sw.jpg"
    exif = Image.Exif()
    exif[0x8825] = {1: "S", 2: (33.0, 54.0, 0.0), 3: "W", 4: (151.0, 12.0, 0.0)}
    Image.new("RGB", (4, 4)).save(p, exif=exif)

    lat, lon = geo.exif_gps(p)
    assert lat == pytest.approx(-33.9)
    assert lon == pytest.approx(-151.2)


def test_exif_gps_no_gps_block(tmp_path):
    from PIL import Image

    p = tmp_path / "plain.jpg"
    Image.new("RGB", (4, 4)).save(p)
    assert geo.exif_gps(p) is None


def test_exif_gps_missing_or_broken_file(tmp_path):
    assert geo.exif_gps(tmp_path / "нет_такого.jpg") is None
    broken = tmp_path / "мусор.jpg"
    broken.write_bytes(b"not a jpeg at all")
    assert geo.exif_gps(broken) is None
