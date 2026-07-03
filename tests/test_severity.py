"""Юнит-тесты классификации серьёзности по ГОСТ Р 50597-2017."""
from road_defect.severity import classify


def test_all_exceed_non_conforming():
    v = classify(length_cm=20, area_m2=0.08, depth_cm=7, road_category="IV")
    assert v.non_conforming == "yes"
    assert v.depth_exceeds is True
    assert v.repair_deadline_days == 10  # кат IV = 10 сут
    assert v.hazard_signing_required is True


def test_within_limits():
    v = classify(length_cm=10, area_m2=0.03, depth_cm=3, road_category="IV")
    assert v.non_conforming == "no"
    assert v.hazard_signing_required is False


def test_depth_missing_is_indeterminate():
    # типичный одиночный кадр: длина/площадь есть, глубины нет
    v = classify(length_cm=20, area_m2=0.08, depth_cm=None, road_category="IV")
    assert v.non_conforming == "indeterminate_without_depth"
    assert v.depth_exceeds is None
    assert v.length_exceeds is True
    assert v.area_exceeds is True
    assert v.repair_deadline_days is None  # без глубины срок не назначаем


def test_no_scale_no_metrics():
    v = classify(length_cm=None, area_m2=None, depth_cm=None)
    assert v.non_conforming == "indeterminate_without_depth"


def test_depth_present_scale_missing_names_true_gap():
    # Ревью 2026-07-02: ветка «глубина есть, размеров нет» (через конвейер
    # недостижима — см-глубина не выдаётся) обязана называть недостающее
    # честно: не хватает МАСШТАБА, а не глубины.
    v = classify(length_cm=None, area_m2=None, depth_cm=7.0, road_category="IV")
    assert v.non_conforming == "indeterminate_without_scale"
    assert v.depth_exceeds is True
    assert v.length_exceeds is None and v.area_exceeds is None
    assert v.repair_deadline_days is None


def test_boundary_values():
    # ровно на пороге (≥ означает несоответствие при выполнении всех трёх)
    v = classify(length_cm=15.0, area_m2=0.06, depth_cm=5.0, road_category="II")
    assert v.non_conforming == "yes"
    assert v.repair_deadline_days == 5
