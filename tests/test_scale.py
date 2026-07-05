"""Юнит-тесты масштабной привязки (без нейросетей)."""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from road_defect import config
from road_defect import scale as scale_mod
from road_defect.scale import (
    ReferenceMeasurement,
    detect_manhole_circle,
    detect_manhole_ellipses,
    homography_from_4_points,
    measure_distance_mm,
    resolve_scale,
    scale_from_curb,
    scale_from_manhole,
)


def _synthetic_manhole(size=(3000, 4000), center=(2000, 1500), r=600):
    """Кадр «асфальта» с тёмным диском-люком; размеры — как у 12-МП фото."""
    h, w = size
    img = np.full((h, w, 3), 128, np.uint8)
    cv2.circle(img, center, r, (40, 40, 40), -1)
    return img


def test_detect_manhole_on_large_photo():
    # Регрессия: фикс. радиусы 25-400 px теряли люк на полноразмерных фото.
    img = _synthetic_manhole()
    c = detect_manhole_circle(img)
    assert c is not None
    cx, cy, r = c
    assert abs(cx - 2000) < 60 and abs(cy - 1500) < 60
    assert abs(r - 600) < 60


def test_no_manhole_on_plain_image():
    img = np.full((480, 640, 3), 128, np.uint8)
    assert detect_manhole_circle(img) is None


def test_scale_from_manhole_synthetic():
    img = _synthetic_manhole()
    ref = scale_from_manhole(img)
    assert ref.available
    # Якорь по умолчанию — обод крышки закрытого люка 646 мм.
    assert ref.known_mm == config.GOST3634_COVER_OUTER_MM
    assert math.isclose(ref.mm_per_px, 646.0 / 1200.0, rel_tol=0.1)
    assert ref.circle_px is not None and ref.ellipse_px is not None


def test_ground_plane_gate_rejects_round_high_in_frame():
    # Круглый эталон в верхней части кадра — вертикальная поверхность
    # (баннер/знак), а не люк на дороге (ложный масштаб 047/048, цикл 8).
    cfg = config.DEFAULT_INFERENCE
    ok, reason = scale_mod._reference_on_ground_plane(
        ((2000, 200), (400, 400), 0.0), frame_h=3000, tilt_deg=2.0, cfg=cfg)
    assert ok is False and "вертикальн" in reason


def test_ground_plane_gate_keeps_foreshortened_high_in_frame():
    # Высоко, но СИЛЬНО сплюснут — далёкий люк у горизонта под скользящим
    # углом: валиден, гейтом не режется (иначе теряли бы настоящие эталоны).
    cfg = config.DEFAULT_INFERENCE
    ok, reason = scale_mod._reference_on_ground_plane(
        ((2000, 200), (400, 120), 0.0), frame_h=3000, tilt_deg=72.0, cfg=cfg)
    assert ok is True and reason == ""


def test_ground_plane_gate_keeps_round_low_in_frame():
    # Круглый, но НИЗКО в кадре — люк под ногами (вид близкий к надиру): валиден.
    cfg = config.DEFAULT_INFERENCE
    ok, _ = scale_mod._reference_on_ground_plane(
        ((2000, 2400), (400, 400), 0.0), frame_h=3000, tilt_deg=2.0, cfg=cfg)
    assert ok is True


def test_scale_from_manhole_rejects_round_disc_high_in_frame():
    # Сквозной путь: круглый диск в верхних 23% кадра отклоняется ИМЕННО гейтом
    # плоскости (а не «не найден») — проверяем по тексту причины.
    img = _synthetic_manhole(center=(2000, 700), r=500)
    ref = scale_from_manhole(img)
    assert not ref.available
    assert "вертикальн" in ref.note


def test_scale_from_manhole_accepts_round_disc_low_in_frame():
    # Регрессия обратной стороны гейта: тот же диск НИЗКО в кадре — валидный люк.
    img = _synthetic_manhole(center=(2000, 2300), r=500)
    ref = scale_from_manhole(img)
    assert ref.available and ref.tilt_deg is not None


def test_gate_falls_through_to_valid_lower_candidate():
    # Гейт применяется ВНУТРИ перебора: верхний круглый диск (вертикальная
    # поверхность) отсеивается, а настоящий люк ниже в кадре всё равно находится
    # — иначе сильный ложный кандидат закрыл бы путь валидному (ревью 2026-07-05).
    img = np.full((3000, 4000, 3), 128, np.uint8)
    cv2.circle(img, (2000, 700), 500, (40, 40, 40), -1)    # верхний: будет отсеян
    cv2.circle(img, (2000, 2300), 500, (40, 40, 40), -1)   # нижний: валидный люк
    ref = scale_from_manhole(img)
    assert ref.available
    (_, cy), _, _ = ref.ellipse_px
    assert cy > 1500   # выбран НИЖНИЙ диск, не отсеянный верхний


def _synthetic_tilted_manhole(size=(3000, 4000), center=(2000, 1500),
                              semi_axes=(600, 300)):
    """Косой вид: люк сплюснут перспективой в эллипс (здесь 2:1, наклон 60°)."""
    h, w = size
    img = np.full((h, w, 3), 128, np.uint8)
    cv2.ellipse(img, center, semi_axes, 0, 0, 360, (40, 40, 40), -1)
    return img


def test_tilted_manhole_found_via_ellipse_path_opt_in():
    # HoughCircles не видит эллипс 2:1 — работает только блоб-путь (opt-in).
    img = _synthetic_tilted_manhole()
    ref = scale_from_manhole(img, allow_blob_reference=True)
    assert ref.available
    assert ref.method == "dark_blob_ellipse"
    # большая ось 1200 px = 646 мм
    assert math.isclose(ref.mm_per_px, 646.0 / 1200.0, rel_tol=0.12)
    assert ref.tilt_deg > 40 and ref.confidence == "low"  # честно: вид косой


def test_blob_reference_disabled_by_default():
    # По умолчанию блоб-путь выключен: на реальном фото тень под бордюром
    # прошла все фильтры и стала ложным эталоном (2026-06-12). До «золотой»
    # валидации масштаба (§11) косой люк честно не выдаёт масштаб.
    img = _synthetic_tilted_manhole()
    assert not scale_from_manhole(img).available


def test_very_flat_ellipse_rejected_even_opt_in():
    # Аспект < 0.45 (наклон > ~63°) — отбой: там и масштаб бессмыслен,
    # и основной источник ложных эталонов (тени).
    img = _synthetic_tilted_manhole(semi_axes=(600, 160))
    assert detect_manhole_ellipses(img) == []
    assert not scale_from_manhole(img, allow_blob_reference=True).available


def test_defect_bbox_excluded_from_reference_candidates():
    # Тёмный эллипс, накрытый bbox детекции (круглая яма), эталоном не становится.
    img = _synthetic_tilted_manhole()
    ref = scale_from_manhole(img, exclude_boxes=[(1300, 1200, 1400, 700)],
                             allow_blob_reference=True)
    assert not ref.available


def test_cut_off_ellipse_at_frame_edge_rejected():
    # Обрезанный кадром люк не измерить — эталоном не становится.
    img = _synthetic_tilted_manhole(center=(300, 1500))  # полуось 600 > 300
    assert detect_manhole_ellipses(img) == []


def test_soft_dark_stain_rejected_by_edge_support():
    # Размытое тёмное пятно (масло/тень): нет резкой кромки — не эталон.
    img = _synthetic_tilted_manhole()
    img = cv2.GaussianBlur(img, (151, 151), 0)
    assert detect_manhole_ellipses(img) == []
    assert not scale_from_manhole(img).available


def test_dark_crack_is_not_an_ellipse_candidate():
    # Толстая тёмная ломаная (трещина/тень) не проходит фильтр заполненности.
    img = np.full((3000, 4000, 3), 128, np.uint8)
    pts = np.array([[200, 2800], [1200, 1500], [2200, 1800], [3800, 300]], np.int32)
    cv2.polylines(img, [pts], False, (40, 40, 40), 25)
    assert detect_manhole_ellipses(img) == []
    assert not scale_from_manhole(img).available


def test_reference_to_dict_matches_contract():
    # Контракт §8: scale.reference вложенный, homography_applied — bool.
    d = ReferenceMeasurement(available=False).to_dict()
    assert {"available", "mm_per_px", "reference",
            "homography_applied", "error_band_pct"} <= set(d)
    assert isinstance(d["homography_applied"], bool)
    assert {"type", "known_mm", "measured_px"} <= set(d["reference"])
    # геометрия overlay в JSON не утекает
    assert "circle_px" not in d and "ellipse_px" not in d


def test_homography_recovers_known_distance():
    # Прямоугольный эталон 1000x500 мм, спроецированный с перспективой.
    world = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float32)
    image = np.array([[100, 120], [520, 90], [560, 410], [80, 450]], dtype=np.float32)
    H = homography_from_4_points(image, world)

    # Расстояние между углами 0 и 1 в мире = 1000 мм; проверяем через H.
    d = measure_distance_mm(H, image[0], image[1])
    assert math.isclose(d, 1000.0, rel_tol=1e-3)

    d2 = measure_distance_mm(H, image[1], image[2])
    assert math.isclose(d2, 500.0, rel_tol=1e-3)


def test_homography_diagonal():
    world = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float32)
    image = world.copy()  # тривиальная (identity-подобная) проекция
    H = homography_from_4_points(image, world)
    diag = measure_distance_mm(H, image[0], image[2])
    assert math.isclose(diag, math.hypot(1000, 500), rel_tol=1e-3)


# --- Мульти-эталон: resolve_scale + кросс-валидация + борт (F2) -------------

def test_resolve_scale_manhole_only_unchanged():
    # Регрессия контракта: при единственном эталоне-люке resolve_scale обязан
    # вернуть тот же масштаб, что и scale_from_manhole (разметки/борта в кадре нет).
    img = _synthetic_manhole()
    direct = scale_from_manhole(img)
    res = resolve_scale(img)
    assert res.available and direct.available
    assert math.isclose(res.mm_per_px, direct.mm_per_px, rel_tol=1e-9)
    assert res.type == direct.type
    assert res.confidence == direct.confidence
    assert res.error_band_pct == direct.error_band_pct


def _ref(type_, mmpp, conf="medium", err=18.0, subtype=None):
    return ReferenceMeasurement(
        available=True, type=type_, subtype=subtype, known_mm=100.0,
        measured_px=100.0 / mmpp if mmpp else None, mm_per_px=mmpp,
        confidence=conf, error_band_pct=err)


_MARK_ON = config.InferenceConfig(use_marking_reference=True)
_CURB_MARK_ON = config.InferenceConfig(use_marking_reference=True,
                                       allow_curb_reference=True)


def test_resolve_scale_markings_off_by_default(monkeypatch):
    # Разметка ВЫКЛЮЧЕНА по умолчанию: даже доступный marking-эталон не берётся.
    monkeypatch.setattr(scale_mod, "scale_from_manhole",
                        lambda *a, **k: ReferenceMeasurement(available=False))
    monkeypatch.setattr(scale_mod, "scale_from_marking",
                        lambda *a, **k: _ref("marking", 2.2, "low", subtype="line_1_1"))
    res = resolve_scale(np.full((400, 400, 3), 120, np.uint8), road_category="IV")
    assert res.available is False        # markings off → нет эталона
    assert res.type != "marking"


def test_resolve_scale_crossvalidation_promotes(monkeypatch):
    # Независимое согласие эталонов РАВНОЙ силы (low люк + low разметка, близкий
    # mm/px) → подтверждение и +1 ступень уверенности (low→medium).
    monkeypatch.setattr(scale_mod, "scale_from_manhole",
                        lambda *a, **k: _ref("manhole_gost3634_cover", 2.0, "low"))
    monkeypatch.setattr(scale_mod, "scale_from_marking",
                        lambda *a, **k: _ref("marking", 2.2, "low", subtype="line_1_1"))
    res = resolve_scale(np.full((400, 400, 3), 120, np.uint8),
                        cfg=_MARK_ON, road_category="IV")
    assert res.type == "manhole_gost3634_cover"   # база — люк (приоритет типа)
    assert res.mm_per_px == 2.0                    # база НЕ усредняется
    assert res.cross_checked is True
    assert res.confidence == "medium"              # low + согласие → +1
    assert "marking" in res.agreeing_types


def test_resolve_scale_weak_witness_does_not_promote_to_high(monkeypatch):
    # Слабый (low) класс-неоднозначный эталон-разметка НЕ должен делать
    # medium-люк «high» — он лишь подтверждает присутствие (ревью 2026-06-20).
    monkeypatch.setattr(scale_mod, "scale_from_manhole",
                        lambda *a, **k: _ref("manhole_gost3634_cover", 2.0, "medium", err=18.0))
    monkeypatch.setattr(scale_mod, "scale_from_marking",
                        lambda *a, **k: _ref("marking", 2.1, "low", subtype="line_1_1"))
    res = resolve_scale(np.full((400, 400, 3), 120, np.uint8),
                        cfg=_MARK_ON, road_category="IV")
    assert res.confidence == "medium"     # НЕ повышен до high слабым свидетелем
    assert res.cross_checked is True      # но согласие зафиксировано
    assert res.error_band_pct >= 18.0     # и полоса НЕ сужена под слабого свидетеля
    assert "marking" in res.agreeing_types


def test_resolve_scale_conflict_demotes_and_warns(monkeypatch):
    # Расхождение эталонов ~55% → понижение до low, без усреднения, с пометкой.
    monkeypatch.setattr(scale_mod, "scale_from_manhole",
                        lambda *a, **k: _ref("manhole_gost3634_cover", 2.0, "medium"))
    monkeypatch.setattr(scale_mod, "scale_from_marking",
                        lambda *a, **k: _ref("marking", 3.5, "low", subtype="line_1_1"))
    res = resolve_scale(np.full((400, 400, 3), 120, np.uint8),
                        cfg=_MARK_ON, road_category="IV")
    assert res.mm_per_px == 2.0                    # масштаб люка, НЕ среднее
    assert res.cross_checked is False
    assert res.confidence == "low"
    assert "конфликт эталонов" in res.note


def test_cross_validate_conflict_is_sticky_curb_cannot_resurrect(monkeypatch):
    # Конфликт ДОМИНИРУЕТ: разметка конфликтует с люком → low; борт, даже
    # согласный, НЕ воскрешает уверенность (порядок эталонов не важен).
    monkeypatch.setattr(scale_mod, "scale_from_manhole",
                        lambda *a, **k: _ref("manhole_gost3634_cover", 2.0, "medium"))
    monkeypatch.setattr(scale_mod, "scale_from_marking",
                        lambda *a, **k: _ref("marking", 3.5, "low", subtype="line_1_1"))
    monkeypatch.setattr(scale_mod, "scale_from_curb",
                        lambda *a, **k: _ref("curb_gost6665", 2.05, "low"))
    res = resolve_scale(np.full((400, 400, 3), 120, np.uint8),
                        cfg=_CURB_MARK_ON, road_category="IV")
    assert res.confidence == "low"        # конфликт «липкий», борт не повышает
    assert res.cross_checked is False


def test_cross_validate_curb_confirms_but_never_promotes(monkeypatch):
    # Борт согласен с люком → cross_checked=True, но уверенность/полоса НЕ растут
    # (борт систематически смещён — только подтверждает присутствие).
    monkeypatch.setattr(scale_mod, "scale_from_manhole",
                        lambda *a, **k: _ref("manhole_gost3634_cover", 2.0, "low", err=30.0))
    monkeypatch.setattr(scale_mod, "scale_from_curb",
                        lambda *a, **k: _ref("curb_gost6665", 2.05, "low", err=30.0))
    res = resolve_scale(np.full((400, 400, 3), 120, np.uint8),
                        cfg=config.InferenceConfig(allow_curb_reference=True))
    assert res.confidence == "low"        # борт не повышает уверенность
    assert res.error_band_pct >= 30.0     # и не сужает полосу
    assert res.cross_checked is True
    assert "curb_gost6665" in res.agreeing_types


def test_cross_validate_agreement_keeps_base_band(monkeypatch):
    # Ревью 2026-07-02: согласие эталонов НЕ сужает полосу — mm_per_px остаётся
    # базовым (без слияния значений), значит и заявленная точность базовая.
    # Выгода согласия идёт в уверенность (+1 ступень), как в fusion.py.
    monkeypatch.setattr(scale_mod, "scale_from_manhole",
                        lambda *a, **k: _ref("manhole_gost3634_cover", 2.0, "low", err=18.0))
    monkeypatch.setattr(scale_mod, "scale_from_marking",
                        lambda *a, **k: _ref("marking", 2.1, "low", err=25.0,
                                             subtype="line_1_1"))
    res = resolve_scale(np.full((400, 400, 3), 120, np.uint8),
                        cfg=_MARK_ON, road_category="IV")
    assert res.type == "manhole_gost3634_cover"   # база — узкая полоса
    assert res.mm_per_px == 2.0            # значение базы не менялось...
    assert res.error_band_pct == 18.0      # ...значит и полоса не сужается
    # (до фикса: _narrowed_band(18, 25) ≈ 14.6 — точность росла без слияния)
    assert res.confidence == "medium"      # выгода — в уверенности
    assert res.cross_checked is True


def test_curb_exclude_checks_candidate_position_not_frame_center():
    # Ревью 2026-07-02: exclude_boxes проверялся по центру КАДРА, а не кандидата.
    # Борт в ВЕРХНЕЙ части кадра (y≈200), bbox дефекта накрывает борт → отказ;
    # bbox в центре кадра (y≈400), борта не касается → борт принимается.
    img = np.full((800, 1200, 3), 130, np.uint8)
    cv2.line(img, (100, 150), (1100, 150), (40, 40, 40), 3)
    cv2.line(img, (100, 250), (1100, 250), (40, 40, 40), 3)
    cfg = config.InferenceConfig(allow_curb_reference=True)

    box_on_curb = [(100.0, 100.0, 1000.0, 200.0)]      # накрывает кромки
    ref = scale_from_curb(img, cfg=cfg, exclude_boxes=box_on_curb)
    assert ref.available is False
    assert "внутри bbox дефекта" in ref.note

    box_frame_center = [(500.0, 350.0, 200.0, 100.0)]  # центр кадра, не борт
    ref = scale_from_curb(img, cfg=cfg, exclude_boxes=box_frame_center)
    assert ref.available is True           # раньше ложно отклонялся


def test_curb_disabled_by_default():
    # Две тёмные параллельные линии (имитация борта): без флага борт не считается.
    img = np.full((800, 1200, 3), 130, np.uint8)
    cv2.line(img, (200, 400), (1000, 400), (40, 40, 40), 3)
    cv2.line(img, (200, 470), (1000, 470), (40, 40, 40), 3)
    res = resolve_scale(img)
    assert res.type != "curb_gost6665"


def test_curb_single_line_rejected():
    # Одна тёмная линия НЕ становится эталоном даже при включённом борте
    # (защита от ложного эталона «тень под бордюром», PROJECT.md §4.4 п.5).
    img = np.full((800, 1200, 3), 130, np.uint8)
    cv2.line(img, (200, 430), (1000, 430), (40, 40, 40), 3)
    cfg = config.InferenceConfig(allow_curb_reference=True)
    assert scale_from_curb(img, cfg=cfg).available is False


def test_curb_gap_is_true_separation_not_sign_flipped():
    # Регрессия знака на склейке 0/180°: две параллельные кромки в ~100 px
    # должны дать gap ≈ 100 px (а не ~14× из-за переворота знака cos).
    img = np.full((800, 1200, 3), 130, np.uint8)
    cv2.line(img, (100, 350), (1100, 350), (40, 40, 40), 3)
    cv2.line(img, (100, 450), (1100, 450), (40, 40, 40), 3)
    cfg = config.InferenceConfig(allow_curb_reference=True)
    ref = scale_from_curb(img, cfg=cfg)
    assert ref.available
    assert ref.type == "curb_gost6665"
    assert abs(ref.measured_px - 100) < 40      # истинное расстояние, не фантом
    assert 0.5 < ref.mm_per_px < 5.0


def test_scale_to_dict_additive_keys():
    d = ReferenceMeasurement(available=False).to_dict()
    # старый контракт §8 сохранён (подмножество)...
    assert {"available", "mm_per_px", "reference",
            "homography_applied", "error_band_pct"} <= set(d)
    assert {"type", "known_mm", "measured_px"} <= set(d["reference"])
    # ...плюс аддитивные поля мульти-эталона
    assert "subtype" in d["reference"]
    assert {"cross_checked", "agreeing_types", "candidates_n"} <= set(d)
    # геометрия overlay по-прежнему не утекает в JSON
    assert "circle_px" not in d and "ellipse_px" not in d and "polylines_px" not in d
