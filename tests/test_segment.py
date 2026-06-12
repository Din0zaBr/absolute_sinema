"""Логика Segmenter для бэкенда mobile_sam_ultralytics (предиктор замокан):
выбор маски с лучшим скором, кэш эмбеддинга кадра, честный фолбэк в GrabCut
при пустой/кривой маске. Сеть и реальные веса не нужны.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from road_defect.segment import Segmenter


class FakeUltralyticsPredictor:
    """Мимикрирует ultralytics.models.sam.Predictor после setup_model()."""

    def __init__(self, results_factory):
        self._results_factory = results_factory
        self.set_image_calls = 0

    def set_image(self, image):
        self.set_image_calls += 1

    def __call__(self, bboxes, multimask_output=False):
        return [self._results_factory()]


def _results(masks_hw: list[np.ndarray] | None, scores: list[float]):
    if masks_hw is None:
        return SimpleNamespace(masks=None, boxes=None)
    data = torch.stack([torch.as_tensor(m, dtype=torch.bool) for m in masks_hw])
    return SimpleNamespace(
        masks=SimpleNamespace(data=data),
        boxes=SimpleNamespace(conf=torch.as_tensor(scores)),
    )


def _segmenter(fake) -> Segmenter:
    s = Segmenter()
    s._tried = True  # не лезть в сеть за реальными весами
    s.backend = "mobile_sam_ultralytics"
    s._predictor = fake
    return s


FRAME = np.full((60, 80, 3), 120, np.uint8)
BBOX = (10.0, 10.0, 30.0, 20.0)


def test_picks_best_score_mask():
    weak = np.zeros((60, 80), bool); weak[0:5, 0:5] = True
    strong = np.zeros((60, 80), bool); strong[12:28, 12:38] = True
    fake = FakeUltralyticsPredictor(lambda: _results([weak, strong], [0.4, 0.9]))
    out = _segmenter(fake).segment(FRAME, BBOX)
    assert out.dtype == bool and out.shape == (60, 80)
    assert np.array_equal(out, strong)


def test_frame_embedding_cached_between_defects():
    m = np.zeros((60, 80), bool); m[12:28, 12:38] = True
    fake = FakeUltralyticsPredictor(lambda: _results([m], [0.9]))
    seg = _segmenter(fake)
    seg.segment(FRAME, BBOX)
    seg.segment(FRAME, (40.0, 30.0, 20.0, 20.0))  # тот же кадр, другой бокс
    assert fake.set_image_calls == 1
    other = np.full((50, 70, 3), 90, np.uint8)  # другой кадр — новый эмбеддинг
    seg.segment(other, BBOX)
    assert fake.set_image_calls == 2


@pytest.mark.parametrize("factory", [
    lambda: _results(None, []),                                  # масок нет
    lambda: _results([np.ones((30, 40), bool)], [0.9]),          # не размер кадра
])
def test_falls_back_to_grabcut_inside_bbox(factory):
    seg = _segmenter(FakeUltralyticsPredictor(factory))
    out = seg.segment(FRAME, BBOX)
    assert out.dtype == bool and out.shape == (60, 80)
    outside = out.copy()
    outside[10:30, 10:40] = False  # всё, что вне bbox, должно быть пустым
    assert not outside.any()
    assert seg.last_method == "grabcut"


# --- крэк-экстрактор (black-hat, полное разрешение) --------------------------

def _crack_scene():
    """Серый асфальт с текстурным шумом и тёмной трещиной 3 px."""
    import cv2
    rng = np.random.default_rng(7)
    base = rng.integers(125, 145, (300, 400), np.uint8)
    cv2.line(base, (40, 150), (360, 130), 55, 3)
    line = np.zeros((300, 400), np.uint8)
    cv2.line(line, (40, 150), (360, 130), 255, 3)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), line > 0


def test_crack_mask_follows_thin_line():
    bgr, line_px = _crack_scene()
    mask, method = Segmenter._crack_mask(bgr, (30, 100, 340, 80))
    assert mask is not None and mask.shape == (300, 400)
    assert method == "crack_blackhat"
    # покрывает большую часть трещины...
    in_box = line_px.copy()
    in_box[:90, :] = in_box[170:, :] = False  # часть линии вне bbox+pad не в счёт
    assert (mask & in_box).sum() / in_box.sum() > 0.5
    # ...и при этом тонкая, а не заливка bbox (главный симптом SAM/GrabCut)
    assert mask.sum() < 0.2 * 340 * 80


def test_light_dust_filled_crack_via_tophat():
    # Трещина, забитая светлой пылью: светлее асфальта — blackhat слеп,
    # выручает симметричный tophat (реальный случай hwfU5q, QA 2026-06-12).
    import cv2
    rng = np.random.default_rng(11)
    base = rng.integers(110, 130, (300, 400), np.uint8)
    cv2.line(base, (40, 150), (360, 130), 210, 3)
    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    mask, method = Segmenter._crack_mask(bgr, (30, 100, 340, 80))
    assert mask is not None and method == "crack_tophat"
    assert mask.sum() < 0.2 * 340 * 80


def test_round_shadow_blobs_rejected_by_elongation():
    # Округлые тёмные пятна МЕЛЬЧЕ ядра (камешки, мелкая пятнистая тень) —
    # не трещина: компонент компактный, фильтр вытянутости его режет.
    # Пятна ШИРЕ ядра дают дугообразные (вытянутые) отклики по границе и
    # фильтром не ловятся — известное ограничение, см. STATUS (лечится только
    # специализированной crack-seg моделью).
    import cv2
    rng = np.random.default_rng(13)
    base = rng.integers(125, 145, (300, 400), np.uint8)
    for cx, cy in [(80, 140), (160, 170), (240, 130), (300, 160), (120, 200)]:
        cv2.circle(base, (cx, cy), 5, 70, -1)
    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    mask, method = Segmenter._crack_mask(bgr, (30, 100, 340, 80))
    assert mask is None and method is None


def test_crack_mask_rejects_featureless_crop():
    flat = np.full((200, 200, 3), 128, np.uint8)
    assert Segmenter._crack_mask(flat, (20, 20, 120, 120)) == (None, None)


def test_linear_class_routes_to_blackhat_not_sam():
    bgr, _ = _crack_scene()
    fake = FakeUltralyticsPredictor(lambda: _results(None, []))
    seg = _segmenter(fake)
    out = seg.segment(bgr, (30.0, 100.0, 340.0, 80.0),
                      cls_name="longitudinal_crack")
    assert out.any()
    assert seg.last_method == "crack_blackhat"
    assert fake.set_image_calls == 0  # SAM не трогали


def test_tiny_crack_mask_falls_back_to_sam_not_lost():
    # Крэк-экстрактор дал < min_area: уверенная детекция не должна пропасть —
    # уходим в SAM (маска грубее, но дефект остаётся в отчёте).
    bgr, _ = _crack_scene()
    sam_mask = np.zeros(bgr.shape[:2], bool); sam_mask[110:160, 50:350] = True
    fake = FakeUltralyticsPredictor(lambda: _results([sam_mask], [0.9]))
    seg = _segmenter(fake)
    # порог заведомо больше тонкой маски трещины (~1-2 тыс. px)
    out = seg.segment(bgr, (30.0, 100.0, 340.0, 80.0),
                      cls_name="transverse_crack", min_area=5000)
    assert np.array_equal(out, sam_mask)
    assert seg.last_method == "mobile_sam_ultralytics"


def test_linear_class_falls_back_to_sam_on_flat_crop():
    m = np.zeros((60, 80), bool); m[12:28, 12:38] = True
    fake = FakeUltralyticsPredictor(lambda: _results([m], [0.9]))
    seg = _segmenter(fake)
    out = seg.segment(FRAME, BBOX, cls_name="transverse_crack")  # FRAME однотонный
    assert np.array_equal(out, m)
    assert seg.last_method == "mobile_sam_ultralytics"
