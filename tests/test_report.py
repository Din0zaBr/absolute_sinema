"""Юнит-тесты RLE и сборки JSON-отчёта."""
import numpy as np

from road_defect.report import build_report, mask_to_rle, unique_stem


def test_unique_stem_series_no_overwrite():
    # Регрессия: обрезка стема до 16 символов затирала отчёты серий.
    used: set = set()
    a = unique_stem("IMG_20260611_123456", used)
    b = unique_stem("IMG_20260611_123457", used)
    c = unique_stem("IMG_20260611_123456", used)  # настоящий дубль имени
    assert a == "IMG_20260611_123456"
    assert b == "IMG_20260611_123457"
    assert c != a and c.startswith("IMG_20260611_123456")
    assert len({a, b, c}) == 3


def _rle_decode(rle):
    """Декодер COCO-RLE (column-major) для проверки roundtrip."""
    counts = rle["counts"]
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in counts:
        flat[pos:pos + c] = val
        pos += c
        val ^= 1
    return flat.reshape((h, w), order="F")


def test_rle_roundtrip_circle():
    yy, xx = np.ogrid[:100, :100]
    mask = ((yy - 50) ** 2 + (xx - 50) ** 2) <= 30 ** 2
    rle = mask_to_rle(mask)
    decoded = _rle_decode(rle)
    assert np.array_equal(decoded.astype(bool), mask)
    assert sum(rle["counts"]) == mask.size


def test_rle_starts_with_foreground():
    # маска, начинающаяся с 1 в (0,0) при column-major порядке
    mask = np.ones((4, 4), dtype=bool)
    rle = mask_to_rle(mask)
    # COCO RLE должен стартовать с нулевого count
    assert rle["counts"][0] == 0
    assert _rle_decode(rle).all()


def test_build_report_schema():
    report = build_report(
        image_name="x.jpg", image_size_px=(640, 480), mode="single",
        reference={"available": False}, defects=[], warnings=["нет эталона"],
    )
    assert report["image"] == "x.jpg"
    assert report["mode"] == "single"
    assert report["location"]["available"] is False
    assert "engine_version" in report
