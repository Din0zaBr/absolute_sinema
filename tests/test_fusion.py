"""Юнит-тесты слияния двух видов (F1). Чистая математика — без сетей и cv2."""
import math

from road_defect import config
from road_defect.fusion import (
    FusedMeasurement,
    ViewMeasurement,
    fuse_pair,
    inverse_variance_fuse,
    tilt_corrected_area,
)


def _view(area_cm2, tilt_deg=0.0, err=18.0, conf="medium", name="v"):
    """Удобный конструктор вида с согласованными длинами под площадь круга."""
    eqd = 2.0 * math.sqrt(area_cm2 / math.pi)
    return ViewMeasurement(
        image_name=name, available=True, defect_class="pothole",
        length_cm=eqd, width_cm=eqd, equivalent_diameter_cm=eqd,
        area_cm2=area_cm2, area_m2=area_cm2 / 10000.0,
        tilt_deg=tilt_deg, error_band_pct=err, confidence=conf,
    )


def test_inverse_variance_reduction():
    val, err = inverse_variance_fuse([10, 10], [18, 18])
    assert math.isclose(val, 10.0, rel_tol=1e-9)
    assert math.isclose(err, 18.0 / math.sqrt(2), rel_tol=1e-3)  # ~12.728
    # Неравные веса: значение тянется к более точному замеру, ошибка < лучшей.
    val2, err2 = inverse_variance_fuse([10, 20], [12, 30])
    assert err2 < 12.0
    assert 10.0 <= val2 < 20.0 and val2 < 12.0  # ближе к точному (10)


def test_tilt_corrected_area_known_answer():
    assert math.isclose(tilt_corrected_area(100, 60, 0.5), 200.0, rel_tol=1e-9)
    assert math.isclose(tilt_corrected_area(100, 0, 0.5), 100.0, rel_tol=1e-9)
    # наклон 80° → cos≈0.17 < пол 0.5 → зажимается, делим на 0.5
    assert math.isclose(tilt_corrected_area(100, 80, 0.5), 200.0, rel_tol=1e-9)
    assert tilt_corrected_area(100, None, 0.5) == 100.0
    assert tilt_corrected_area(None, 30, 0.5) is None


def test_fuse_pair_agreement_promotes():
    f = fuse_pair(_view(100.0), _view(110.0))  # расхождение ~9% < 20%
    assert f.available
    assert f.cross_view["agreement"] == "agree"
    assert f.confidence == "high"             # medium+medium, согласие → +1
    # полоса не уже лучшего одиночного вида и не больше него
    assert math.isclose(f.error_band_pct, 18.0, rel_tol=1e-6)
    assert f.area_cm2 is not None and f.length_cm is not None


def test_fuse_pair_disagreement_flags_and_widens():
    f = fuse_pair(_view(100.0), _view(200.0))  # расхождение 50% > 40%
    assert f.available                          # число всё ещё выдаётся...
    assert f.cross_view["agreement"] == "disagree"
    assert f.confidence == "low"                # ...но честно помечено ненадёжным
    assert f.error_band_pct > 18.0              # полоса расширена
    assert "РАСХОЖДЕНИЕ" in f.note
    assert f.area_cm2 is not None


def test_fuse_pair_low_view_caps_confidence():
    # один вид low (err 30) + один medium, согласие → НЕ до high (потолок medium)
    f = fuse_pair(_view(100.0, err=30.0, conf="low"), _view(105.0))
    assert f.cross_view["agreement"] == "agree"
    assert f.confidence != "high"
    assert f.confidence == "medium"


def test_fuse_pair_refuses_without_scale_in_one_view():
    a = _view(100.0)
    b = ViewMeasurement(image_name="b", available=False)
    f = fuse_pair(a, b)
    assert f.available is False
    assert f.cross_view["fused"] is False


def test_fused_depth_always_null():
    f = fuse_pair(_view(100.0), _view(110.0))
    d = f.to_dict()
    assert d["depth_cm"] is None
    assert d["depth_certifiable"] is False
    assert d["depth_bucket"] is None
    # схема metric стабильна: те же ключи, что у одиночного кадра
    assert {"available", "equivalent_diameter_cm", "length_cm", "width_cm",
            "area_cm2", "area_m2", "confidence", "error_band_pct"} <= set(d)
    assert d["cross_view"]["fused"] is True


def test_fuse_pair_area_not_corrected_when_tilt_unknown():
    # Наклон неизвестен в обоих видах (эталон-разметка/борт): площадь НЕ
    # корректируется, флаг честный, полоса расширена до запаса на ракурс.
    f = fuse_pair(_view(100.0, tilt_deg=None), _view(110.0, tilt_deg=None))
    assert f.available
    assert f.cross_view["area_tilt_corrected"] is False
    assert f.error_band_pct >= config.DEFAULT_INFERENCE.fusion_uncorrected_area_band_pct


def test_fuse_pair_asymmetric_tilt_no_false_disagreement():
    # Один вид с известным наклоном (люк), другой без (разметка). Одинаковая яма
    # НЕ должна давать ложное расхождение: коррекция не применяется ни к одному.
    a = _view(76.6, tilt_deg=40.0)   # «люк»: наклон известен
    b = _view(76.6, tilt_deg=None)   # «разметка»: наклон неизвестен
    f = fuse_pair(a, b)
    assert f.cross_view["agreement"] == "agree"
    assert f.cross_view["area_tilt_corrected"] is False
    assert f.cross_view["disagreement_pct"] == 0.0


def test_fuse_pair_same_area_different_shape_refused():
    # Одинаковая площадь при РАЗНОЙ форме (вытянутая щель vs округлая яма) —
    # это РАЗНЫЕ дефекты: ловится refuse-гейтом по форме (eccentricity/solidity).
    a = ViewMeasurement(image_name="a", available=True, defect_class="pothole",
                        length_cm=50.0, width_cm=5.0, equivalent_diameter_cm=17.8,
                        area_cm2=100.0, area_m2=0.0100, tilt_deg=0.0,
                        error_band_pct=18.0, confidence="medium",
                        eccentricity=0.99, solidity=0.95)
    b = ViewMeasurement(image_name="b", available=True, defect_class="pothole",
                        length_cm=10.0, width_cm=12.7, equivalent_diameter_cm=12.7,
                        area_cm2=105.0, area_m2=0.0105, tilt_deg=0.0,
                        error_band_pct=18.0, confidence="medium",
                        eccentricity=0.55, solidity=0.92)
    f = fuse_pair(a, b)
    assert f.available is False
    assert f.cross_view["reason"] == "shape_mismatch"


def test_fuse_pair_length_is_lower_bound_not_pulled_down():
    # length/width — НИЖНЯЯ оценка (макс сырых): сильнее укороченный вид НЕ тянет
    # слитую длину вниз (защита ноги ГОСТ «длина ≥ 15 см» от занижения).
    a = _view(100.0, err=25.0); a.length_cm, a.width_cm = 16.0, 6.0   # near-nadir
    b = _view(100.0, err=12.0); b.length_cm, b.width_cm = 10.0, 4.0   # foreshortened, tight band
    f = fuse_pair(a, b)
    assert f.length_cm == 16.0     # макс, НЕ inverse-variance (которое дало бы ~11-12)
    assert f.cross_view["length_is_lower_bound"] is True


def test_fuse_pair_marking_only_pair_not_promoted():
    # Два вида по ОДНОЙ разметке (tilt=None в обоих, low): согласие НЕ повышает
    # уверенность — они делят один систематический сдвиг класса ширины.
    a = _view(100.0, tilt_deg=None, err=50.0, conf="low")
    b = _view(105.0, tilt_deg=None, err=50.0, conf="low")
    f = fuse_pair(a, b)
    assert f.cross_view["agreement"] == "agree"
    assert f.cross_view["confidence_promoted"] is False
    assert f.confidence == "low"


def test_fuse_pair_shape_mismatch_refuses():
    # Сильно разная форма видов → вероятно РАЗНЫЕ ямы → не сливаем.
    a = _view(100.0)
    a.eccentricity, a.solidity = 0.2, 0.95
    b = _view(105.0)
    b.eccentricity, b.solidity = 0.9, 0.50
    f = fuse_pair(a, b)
    assert f.available is False
    assert f.cross_view["reason"] == "shape_mismatch"


def test_view_measurement_from_defect_reads_tilt():
    defect = {
        "class": "pothole",
        "metric": {"available": True, "length_cm": 30.0, "width_cm": 20.0,
                   "equivalent_diameter_cm": 25.0, "area_cm2": 500.0,
                   "area_m2": 0.05, "error_band_pct": 18.0, "confidence": "medium",
                   "view_tilt_deg": 22.5},
        "shape": {"eccentricity": 0.4, "solidity": 0.9},
    }
    vm = ViewMeasurement.from_defect(defect, "frameA.jpg")
    assert vm.available and vm.tilt_deg == 22.5
    assert vm.area_cm2 == 500.0 and vm.confidence == "medium"
    assert vm.eccentricity == 0.4
