"""Генератор формы замеров: структура автономной HTML, экранирование, миниатюры.

Логика выгрузки CSV живёт в JS страницы (порядок колонок / кавычки / utf-8-sig
проверены прогоном под node при сборке); здесь — Python-часть: карточки, ссылки
на оригиналы, честные заглушки и безопасное экранирование пользовательского
текста в атрибутах.
"""
import html as htmlmod
import importlib.util
import json
import re
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_measure_form", ROOT / "scripts" / "build_measure_form.py")
bmf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bmf)


def _pit(pid, scene, **over):
    base = {
        "id": pid, "scene": scene, "date": "2026-07-04", "street_or_gps": "47 39",
        "type": "pothole", "length_cm": "", "width_cm": "", "depth_cm": "",
        "reference_in_frame": "", "weather": "", "notes": "",
        "rulers": [], "front_thumb": None,
    }
    base.update(over)
    return base


def test_build_html_cards_images_and_links():
    pits = [
        _pit("001", "001",
             rulers=[{"uri": "data:image/jpeg;base64,AAA",
                      "href": "../datasets/gt_photos/001_ruler_1.jpg"}],
             front_thumb="data:image/jpeg;base64,BBB"),
        _pit("002", "001"),  # общая сцена, без рулеток и front
    ]
    out = bmf.build_html(pits, "note")
    assert out.count('class="pit"') == 2
    assert "Яма 001" in out and "Яма 002" in out
    assert "data:image/jpeg;base64,AAA" in out
    assert "../datasets/gt_photos/001_ruler_1.jpg" in out
    assert 'title="открыть оригинал"' in out           # ссылка на оригинал
    assert "нет кадров с рулеткой" in out               # заглушка у 002
    assert "сцена 001 (общий кадр)" in out              # бейдж сцены у 002 (scene!=id)
    assert 'id="dl"' in out                             # кнопка выгрузки
    assert '"length_cm"' in out and '"reference_in_frame"' in out  # FIELDS в JS


def test_build_html_escapes_user_text_in_attributes():
    out = bmf.build_html([_pit("001", "001", notes='край "1" <b>')], "")
    assert 'край "1" <b>' not in out                    # не влезло сыро
    assert "&quot;1&quot;" in out and "&lt;b&gt;" in out


def test_build_html_data_fixed_is_valid_json():
    pits = [_pit("001", "001", street_or_gps='55.7, 37.6 "центр"')]
    out = bmf.build_html(pits, "")
    for m in re.findall(r'data-fixed="([^"]*)"', out):
        parsed = json.loads(htmlmod.unescape(m))
        assert parsed["id"] == "001"
        assert parsed["street_or_gps"] == '55.7, 37.6 "центр"'


def test_data_uri_encodes_real_image(tmp_path):
    p = tmp_path / "x.jpg"
    Image.new("RGB", (40, 30), (10, 20, 30)).save(p, "JPEG")
    uri = bmf._data_uri(p, 16, 70)
    assert uri.startswith("data:image/jpeg;base64,")


def test_data_uri_returns_none_on_unreadable(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not an image")
    assert bmf._data_uri(p, 16, 70) is None


def test_select_preserves_value_outside_option_list():
    # значение журнала вне списка (type='crack') НЕ должно молча стать первой
    # опцией — иначе выгрузка затёрла бы поле (ревью 2026-07-05).
    out = bmf._select("type", "crack", bmf.TYPE_OPTIONS)
    assert '<option value="crack" selected>' in out
    # а известное значение выбирается штатно, без дубля
    out2 = bmf._select("type", "pothole", bmf.TYPE_OPTIONS)
    assert out2.count("<option") == len(bmf.TYPE_OPTIONS)
    assert '<option value="pothole" selected>' in out2


def test_rel_href_empty_on_cross_drive(monkeypatch):
    # разные диски Windows: os.path.relpath кидает ValueError — ссылку опускаем,
    # а не роняем всю сборку формы (ревью 2026-07-05).
    def boom(*a, **k):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")
    monkeypatch.setattr(bmf.os.path, "relpath", boom)
    assert bmf._rel_href(Path("D:/x/a.jpg"), Path("C:/y")) == ""


def test_pid_forms_zero_pad_tolerance():
    assert bmf._pid_forms("1") == ["1", "001"]
    assert bmf._pid_forms("001") == ["001"]


def test_fields_match_ingest_journal_fields():
    # FIELDS дублирует ingest_local_pairs.JOURNAL_FIELDS литералом — контракт
    # между скриптами держим тестом (ревью 2026-07-05).
    spec = importlib.util.spec_from_file_location(
        "ingest_local_pairs", ROOT / "scripts" / "ingest_local_pairs.py")
    ing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ing)
    assert bmf.FIELDS == ing.JOURNAL_FIELDS
