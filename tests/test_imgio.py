"""Юнит-тесты юникод-безопасного чтения/записи изображений (Windows)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from road_defect.imgio import read_image, write_image


def test_cyrillic_path_roundtrip(tmp_path):
    # Регрессия: cv2.imread/imwrite молча не работают с кириллицей на Windows.
    img = np.zeros((10, 12, 3), np.uint8)
    img[2:5, 3:7] = (10, 200, 30)
    path = tmp_path / "Фото дороги" / "яма_1.png"
    path.parent.mkdir()
    assert write_image(path, img) is True
    back = read_image(path)
    assert back is not None
    assert back.shape == img.shape
    assert np.array_equal(back, img)  # png — без потерь


def test_read_missing_returns_none(tmp_path):
    assert read_image(tmp_path / "нет_такого.jpg") is None


def test_write_to_missing_dir_returns_false(tmp_path):
    img = np.zeros((4, 4, 3), np.uint8)
    assert write_image(tmp_path / "нет_папки" / "x.jpg", img) is False
