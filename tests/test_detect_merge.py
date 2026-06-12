"""Слияние детекций ансамбля (merge_detections): второй проход дополняет
основной детектор, не дублируя то, что оба видят.
"""
from road_defect.detect import Detection, merge_detections, _iou_xywh


def _det(cls, x, y, w, h, conf=0.8):
    return Detection(cls_name=cls, raw_label=cls, confidence=conf,
                     bbox_xywh=(float(x), float(y), float(w), float(h)))


def test_iou_basics():
    assert _iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou_xywh((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert 0.3 < _iou_xywh((0, 0, 10, 10), (5, 0, 10, 10)) < 0.4


def test_overlapping_same_class_not_duplicated():
    primary = [_det("pothole", 100, 100, 50, 50)]
    secondary = [_det("pothole", 105, 102, 50, 50, conf=0.9)]  # IoU ~0.8
    merged = merge_detections(primary, secondary)
    assert len(merged) == 1 and merged[0] is primary[0]  # авторитет — основной


def test_new_pothole_from_second_pass_added():
    primary = [_det("longitudinal_crack", 10, 10, 200, 30)]
    secondary = [_det("pothole", 400, 400, 120, 90, conf=0.81)]  # пропуск rezzzq
    merged = merge_detections(primary, secondary)
    assert len(merged) == 2
    assert any(d.cls_name == "pothole" for d in merged)


def test_overlap_with_other_class_still_added():
    # Яма поверх трещины — разные классы, не дубль.
    primary = [_det("longitudinal_crack", 100, 100, 60, 60)]
    secondary = [_det("pothole", 100, 100, 60, 60)]
    merged = merge_detections(primary, secondary)
    assert len(merged) == 2
