"""Регрессы отложенных находок аудита 2026-07-07 (docs/HANDOFF_FABLE.md, блок ⏸️):
№7 — порог площади маски зависел от разрешения кадра (только ВНИЗ, не вверх);
№8 — пороги бакета глубины мимо InferenceConfig (правило «пороги через config»);
№10 — пофайловые overlay в --pairs-dir теряли id пары (счётчик дедупа ≠ папка);
№19 — второй проход ансамбля молча пустел при дрейфе ярлыка класса pothole.
Каждая находка — падающий тест до фикса, зелёный после (процесс проекта).
"""
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from road_defect import cli, config
from road_defect import depth as depth_mod
from road_defect import detect as detect_mod
from road_defect import imgio
from road_defect import scale as scale_mod
from road_defect.detect import Detection
from road_defect.pipeline import DefectPipeline


# --- P3 #7: порог площади маски масштабируется ТОЛЬКО вниз -------------------
def test_mask_floor_unchanged_at_and_above_reference_resolution():
    cfg = config.InferenceConfig()
    # Все текущие данные проекта (городские демо 1152x2560, пары 4096x1844,
    # 12-МП кадры) обязаны сохранить прежние абсолютные пороги.
    assert cfg.scaled_mask_min_area(cfg.mask_min_area_px, (2560, 1152)) == 200
    assert cfg.scaled_mask_min_area(cfg.mask_min_area_px, (1844, 4096)) == 200
    assert cfg.scaled_mask_min_area(cfg.mask_min_area_px, (3000, 4000)) == 200
    # Наивная «доля площади кадра» ПОВЫСИЛА бы порог на 48 МП и срезала бы
    # дальние мелкие ямы — порог вверх не растёт никогда.
    assert cfg.scaled_mask_min_area(cfg.mask_min_area_px, (6000, 8000)) == 200


def test_mask_floor_lowered_proportionally_on_small_frames():
    cfg = config.InferenceConfig()
    # 1024x768 (~0.79 МП): та же маска после даунскейла теряет площадь
    # пропорционально площади кадра — абсолютный порог резал бы валидное.
    assert cfg.scaled_mask_min_area(200, (768, 1024)) == 54
    assert cfg.scaled_mask_min_area(60, (768, 1024)) == 16
    assert cfg.scaled_mask_min_area(200, (2, 2)) == 1   # пол 1 px, не 0


class _ThinCrackDetector:
    """Уверенная трещина 4x12=48 px на кадре 400x400 (< линейного пола 60)."""

    is_fallback = False

    def load(self):
        return self

    def detect(self, img):
        H, W = img.shape[:2]
        mask = np.zeros((H, W), bool)
        mask[200:204, 100:112] = True
        return [Detection(cls_name="longitudinal_crack", raw_label="D00",
                          confidence=0.9, bbox_xywh=(100.0, 200.0, 12.0, 4.0),
                          mask=mask)]


def test_small_frame_keeps_thin_crack_that_absolute_floor_rejected(tmp_path, monkeypatch):
    img = np.full((400, 400, 3), 120, np.uint8)
    path = tmp_path / "frame.jpg"
    assert imgio.write_image(path, img)
    monkeypatch.setattr(scale_mod, "scale_from_manhole",
                        lambda *a, **k: scale_mod.ReferenceMeasurement(available=False))

    pipe = DefectPipeline(use_depth=False)
    pipe.detector = _ThinCrackDetector()
    rep, _, _ = pipe.analyze_image(path)

    # До фикса: 48 px < 60 -> детекция отброшена целиком (только warning).
    assert [d["class"] for d in rep["defects"]] == ["longitudinal_crack"]


class _SmallPotholeDetector:
    """Яма 10x10=100 px на кадре 400x400 (< площадного пола 200)."""

    is_fallback = False

    def load(self):
        return self

    def detect(self, img):
        H, W = img.shape[:2]
        mask = np.zeros((H, W), bool)
        mask[200:210, 100:110] = True
        return [Detection(cls_name="pothole", raw_label="D40", confidence=0.9,
                          bbox_xywh=(100.0, 200.0, 10.0, 10.0), mask=mask)]


def test_small_frame_keeps_small_pothole_that_absolute_floor_rejected(tmp_path, monkeypatch):
    # Пин areal-ветки №7 (ревью: интеграция покрывала только линейную ветку —
    # мутация «скейлить лишь трещины» проходила незамеченной).
    img = np.full((400, 400, 3), 120, np.uint8)
    path = tmp_path / "frame.jpg"
    assert imgio.write_image(path, img)
    monkeypatch.setattr(scale_mod, "scale_from_manhole",
                        lambda *a, **k: scale_mod.ReferenceMeasurement(available=False))

    pipe = DefectPipeline(use_depth=False)
    pipe.detector = _SmallPotholeDetector()
    rep, _, _ = pipe.analyze_image(path)

    assert [d["class"] for d in rep["defects"]] == ["pothole"]


def test_new_config_knobs_are_validated_at_construction():
    # Ревью: вырожденные значения новых ручек падали бы делением на нуль на
    # КАЖДОМ кадре (ворота линейки / scaled_mask_min_area) — отказ сразу.
    with pytest.raises(ValueError):
        config.InferenceConfig(depth_bucket_hi=0.0)
    with pytest.raises(ValueError):
        config.InferenceConfig(depth_bucket_lo=0.4, depth_bucket_hi=0.2)
    with pytest.raises(ValueError):
        config.InferenceConfig(mask_min_area_ref_px=0)
    with pytest.raises(ValueError):
        config.InferenceConfig(mask_min_area_ref_px=-100)
    with pytest.raises(ValueError):
        config.InferenceConfig(depth_noise_gate_sigma=-1.0)


# --- P3 #8: пороги бакета глубины живут в InferenceConfig --------------------
def _road_scene(pit_depth, H=400, W=400, noise=0.003, seed=7,
                cx=200, cy=260, r=40, grad=1.5):
    """Мини-копия сцены из test_depth_bucket: наклонная дорога + вмятина."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    depth = 2.0 + grad * (yy / H)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    depth -= pit_depth * np.exp(-dist2 / (2 * (r * 0.6) ** 2))
    depth += rng.normal(0.0, noise, size=depth.shape)
    return depth, dist2 <= r * r


def test_bucket_thresholds_live_in_config_with_compat_aliases():
    d = config.DEFAULT_INFERENCE
    # Канон — config; модульные имена BUCKET_* сохранены для импортов
    # (tests/test_depth_bucket.py, scripts/calibrate_depth_bucket.py).
    assert depth_mod.BUCKET_LO == d.depth_bucket_lo == 0.08
    assert depth_mod.BUCKET_HI == d.depth_bucket_hi == 0.35
    assert d.depth_noise_gate_sigma == 3.0


def test_ring_plane_bucket_honors_threshold_overrides():
    depth, mask = _road_scene(pit_depth=0.09)
    base = depth_mod.ring_plane_bucket(depth, mask)
    assert base["depth_bucket"] == "medium"
    rn = base["_rel_norm"]

    deep = depth_mod.ring_plane_bucket(depth, mask, hi=rn * 0.9)
    assert deep["depth_bucket"] == "deep"
    shallow = depth_mod.ring_plane_bucket(depth, mask, lo=rn * 1.1, hi=rn * 2.0)
    assert shallow["depth_bucket"] == "shallow"

    # Ворота линейки обязаны считаться от ПАРАМЕТРА hi, не от модульного
    # BUCKET_HI (пин ревью): крошечный hi ужесточает ворота до честного отказа.
    refused = depth_mod.ring_plane_bucket(depth, mask, hi=0.02)
    assert refused["depth_bucket"] is None
    assert refused["method"] == "degenerate_local_ruler"


def test_relative_depth_threads_config_thresholds():
    depth, mask = _road_scene(pit_depth=0.09)
    img = np.zeros((*depth.shape, 3), np.uint8)

    def _fake_engine(cfg=None):
        rd = (depth_mod.RelativeDepth(cfg) if cfg is not None
              else depth_mod.RelativeDepth())
        rd._tried, rd.available = True, True          # модель не поднимаем
        rd._frame_depth = lambda _img: depth.astype(np.float32)
        return rd

    assert _fake_engine().relative_bucket(img, mask)["depth_bucket"] == "medium"

    cfg = config.InferenceConfig(depth_bucket_lo=0.01, depth_bucket_hi=0.05)
    out = _fake_engine(cfg).relative_bucket(img, mask)
    assert out["depth_bucket"] == "deep"              # hi дошёл до бакетизации
    assert out["depth_cm"] is None                    # честность не тронута

    # Пины ревью: lo и noise_gate тоже должны ДОХОДИТЬ (не только hi).
    out_lo = _fake_engine(config.InferenceConfig(
        depth_bucket_lo=0.30, depth_bucket_hi=0.32)).relative_bucket(img, mask)
    assert out_lo["depth_bucket"] == "shallow"        # rel_norm ~0.24 < lo
    out_ng = _fake_engine(config.InferenceConfig(
        depth_noise_gate_sigma=1e6)).relative_bucket(img, mask)
    assert out_ng["method"].endswith("below_local_noise")


def test_pipeline_threads_cfg_into_depth_engine():
    # Пин ревью: откат RelativeDepth(cfg) -> RelativeDepth() в конструкторе
    # pipeline был невидим (никто не строил DefectPipeline(use_depth=True)).
    c = config.InferenceConfig(depth_bucket_hi=0.123)
    assert DefectPipeline(cfg=c, use_depth=True).depth.cfg is c


# --- P3 #10: пофайловые overlay пары несут id пары ---------------------------
def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


class _FakePairPipe:
    def analyze_pair(self, a, b):
        overlay = np.zeros((8, 8, 3), dtype=np.uint8)
        report = {
            "image": Path(a).name,
            "mode": "two_view_unmatched",
            "image_size_px": [8, 8],
            "scale": {"available": False},
            "defects": [],
            "warnings": [],
            "fusion": {"fused": False, "matched": False},
        }
        return report, overlay, overlay.copy(), []


def test_pairs_dir_per_view_overlays_carry_pair_id(tmp_path):
    # Папка 002 пропущена намеренно: старая схема нумеровала overlay счётчиком
    # дедупа (front_2_annotated.jpg), который при любом пропуске расходился
    # с id папки — привязать кадр к паре было нельзя.
    root = tmp_path / "pairs"
    for pid in ("001", "003"):
        _touch(root / pid / "front.jpg")
        _touch(root / pid / "back.jpg")
    out = tmp_path / "out"

    assert cli._run_pairs_dir(_FakePairPipe(), str(root), out) == 0

    names = {p.name for p in out.glob("*_annotated.jpg")}
    assert names == {
        "001__front_annotated.jpg", "001__back_annotated.jpg",
        "001__front__back_pair_annotated.jpg",
        "003__front_annotated.jpg", "003__back_annotated.jpg",
        "003__front__back_pair_annotated.jpg",
    }


def test_single_pair_names_stay_unprefixed(tmp_path):
    # Без --pairs-dir префикса нет — одиночный --pair сохраняет старые имена.
    a, b = tmp_path / "front.jpg", tmp_path / "back.jpg"
    _touch(a)
    _touch(b)
    out = tmp_path / "out"

    assert cli._save_pair_result(_FakePairPipe(), a, b, out) is not None
    names = {p.name for p in out.glob("*_annotated.jpg")}
    assert names == {"front_annotated.jpg", "back_annotated.jpg",
                     "front__back_pair_annotated.jpg"}


def test_pairs_dir_warns_about_stale_reports_of_old_convention(tmp_path, capsys):
    # Миграционный гейт: файл старой конвенции прогон не перезапишет —
    # смешанная папка двоит сводки/глобы, CLI обязан предупредить.
    root = tmp_path / "pairs"
    _touch(root / "022" / "front.jpg")
    _touch(root / "022" / "back.jpg")
    out = tmp_path / "out"
    out.mkdir()
    (out / "022__front_22__back_22_pair.json").write_text("{}", encoding="utf-8")

    assert cli._run_pairs_dir(_FakePairPipe(), str(root), out) == 0
    assert "не перезапишет" in capsys.readouterr().err


def test_pairs_dir_rerun_into_same_folder_is_silent(tmp_path, capsys):
    # Идемпотентный повторный прогон (те же имена) шуметь не должен.
    root = tmp_path / "pairs"
    _touch(root / "001" / "front.jpg")
    _touch(root / "001" / "back.jpg")
    out = tmp_path / "out"

    assert cli._run_pairs_dir(_FakePairPipe(), str(root), out) == 0
    capsys.readouterr()
    assert cli._run_pairs_dir(_FakePairPipe(), str(root), out) == 0
    assert "не перезапишет" not in capsys.readouterr().err


# --- Латентная ловушка #19: ярлык класса второго прохода ---------------------
def test_pothole_label_matcher_is_case_and_taxonomy_aware():
    assert detect_mod._is_pothole_label("pothole") is True
    assert detect_mod._is_pothole_label("Pothole") is True   # дрейф регистра
    assert detect_mod._is_pothole_label("D40") is True       # RDD-таксономия
    assert detect_mod._is_pothole_label("d40") is True       # регистр кода тоже
    assert detect_mod._is_pothole_label("D40 ") is True      # хвостовой пробел
    assert detect_mod._is_pothole_label("crack") is False
    assert detect_mod._is_pothole_label("Repair") is False   # заплатка — не яма


def test_second_pass_normalizes_odd_case_and_filters_foreign():
    dets = [Detection("Pothole", "Pothole", 0.9, (0.0, 0.0, 10.0, 10.0)),
            Detection("crack", "crack", 0.9, (5.0, 5.0, 10.0, 10.0))]
    got = detect_mod._second_pass_potholes(dets)
    assert [d.cls_name for d in got] == ["pothole"]  # каноничен для downstream


def _second_pass_detector(monkeypatch, names: dict):
    import ultralytics

    class _FakeYOLO:
        def __init__(self, weights):
            self.names = names

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    monkeypatch.setattr(detect_mod, "_acquire_weights", lambda c: "w.pt")
    det = detect_mod.Detector(config.InferenceConfig(ensemble_pothole=True))
    det._load_pothole_second_pass()
    return det


def test_second_pass_without_pothole_class_fails_loudly(monkeypatch):
    # Веса без класса pothole -> ensemble_active было бы True при 0 детекций
    # НАВСЕГДА (молчаливый пустой второй проход). Теперь честный отказ,
    # который pipeline поднимает в warnings отчёта.
    det = _second_pass_detector(monkeypatch, {0: "crack", 1: "patch"})
    assert det.ensemble_failed is True
    assert det.ensemble_active is False
    assert det._pothole_model is None


def test_second_pass_with_pothole_class_activates_case_insensitively(monkeypatch):
    det = _second_pass_detector(monkeypatch, {0: "Pothole"})
    assert det.ensemble_failed is False
    assert det.ensemble_active is True
    assert det._pothole_model is not None


def test_second_pass_with_unverifiable_names_still_activates(monkeypatch):
    # Пустые/отсутствующие names — «нечего проверить», НЕ повод валить ансамбль
    # (гейт бьёт только «ярлыки есть, но ямы среди них нет»). Пин ветки ревью.
    det = _second_pass_detector(monkeypatch, {})
    assert det.ensemble_failed is False
    assert det.ensemble_active is True


def test_detect_call_site_normalizes_and_merges_with_cfg_iou(monkeypatch):
    # Пин ревью: сам detect() не был покрыт — откат колл-сайта к жёсткому
    # `cls_name == "pothole"` или к захардкоженному iou_thr=0.5 проходил тесты.
    det = detect_mod.Detector(config.InferenceConfig(ensemble_pothole=True))
    det._model, det._pothole_model = "PRIMARY", "SECOND"
    det._tried_second_pass, det.ensemble_active = True, True

    def fake_predict(self, model, img):
        if model == "PRIMARY":
            return [Detection("pothole", "D40", 0.8, (0.0, 0.0, 100.0, 100.0))]
        return [
            # IoU ~0.43 с первичной: дубль при cfg 0.35, «новая» при старом 0.5
            Detection("Pothole", "Pothole", 0.9, (40.0, 0.0, 100.0, 100.0)),
            Detection("Pothole", "Pothole", 0.9, (300.0, 300.0, 50.0, 50.0)),
            Detection("crack", "crack", 0.9, (0.0, 0.0, 10.0, 10.0)),
        ]

    monkeypatch.setattr(detect_mod.Detector, "_predict", fake_predict)
    out = det.detect(np.zeros((8, 8, 3), np.uint8))

    # дубль подавлен порогом из cfg, новая яма канонизирована, чужой класс убран
    assert [d.cls_name for d in out] == ["pothole", "pothole"]
    assert out[1].raw_label == "Pothole"
