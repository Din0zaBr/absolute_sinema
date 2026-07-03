"""Бейджи вердикта ГОСТ в демо-отчёте: контракт severity — СТРОКИ
"yes"|"no"|"indeterminate_without_depth" (ревью 2026-06-12: сравнение
с True/False делало бейджи «соответствует/НЕ соответствует» недостижимыми).
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "make_demo_report", ROOT / "scripts" / "make_demo_report.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def test_yes_renders_non_conforming_badge():
    out = demo.severity_badge({"non_conforming": "yes"})
    assert "НЕ соответствует ГОСТ" in out and "bad" in out


def test_no_renders_conforming_badge():
    out = demo.severity_badge({"non_conforming": "no"})
    assert ">соответствует ГОСТ<" in out and '"badge ok"' in out


def test_indeterminate_renders_warn_badge():
    out = demo.severity_badge({"non_conforming": "indeterminate_without_depth"})
    assert "не подтверждена" in out and "warn" in out


def test_indeterminate_without_scale_names_true_gap():
    # Ревью 2026-07-02: warn-бейдж обязан называть недостающее честно —
    # для этого статуса не хватает МАСШТАБА, а не глубины.
    out = demo.severity_badge({"non_conforming": "indeterminate_without_scale"})
    assert "эталона масштаба" in out and "warn" in out
    assert "глубина" not in out
