"""Тесты правил уведомлений (подсистема 5). Без сети и моделей."""
import pytest

from road_defect import config
from road_defect.platform import db, notify


def defect_row(**over):
    row = {"class": "pothole", "non_conforming": "indeterminate_without_depth",
           "severity_note": "", "depth_bucket": None,
           "depth_est_cm": None, "depth_est_high_cm": None}
    row.update(over)
    return row


def rules_of(row):
    return [rule for rule, _msg in notify.rules_for_defect(row)]


def test_non_conforming_yes_triggers_gost_rule():
    rules = notify.rules_for_defect(defect_row(non_conforming="yes",
                                               severity_note="превышены пороги"))
    assert ("gost_non_conforming", ) == tuple(r for r, _ in rules)[:1]
    message = dict(rules)["gost_non_conforming"]
    assert "ГОСТ" in message and "превышены пороги" in message


def test_deep_bucket_triggers_suspect():
    assert "deep_pothole_suspect" in rules_of(defect_row(depth_bucket="deep"))


def test_depth_estimate_above_threshold_triggers_suspect():
    est = config.GOST50597_MAX_DEPTH_CM + 1.0  # оценка 6 см при пороге 5
    rules = notify.rules_for_defect(defect_row(depth_est_cm=est))
    assert dict(rules).keys() == {"deep_pothole_suspect"}
    # честная оговорка: это оценка, требуется выезд/замер
    msg = dict(rules)["deep_pothole_suspect"]
    assert "оценка" in msg and ("выезд" in msg or "замер" in msg)


def test_high_bound_used_over_point():
    # точечная оценка мала, но верхняя граница достигает порога — подозрение есть
    row = defect_row(depth_est_cm=3.0,
                     depth_est_high_cm=config.GOST50597_MAX_DEPTH_CM)
    assert "deep_pothole_suspect" in rules_of(row)


def test_upper_bound_wording_is_honest():
    # Ревью 2026-07-15: below-noise верхняя граница — предел разрешимости карты;
    # текст обязан говорить «не исключает до X», а не «может достигать X».
    row = defect_row(depth_est_cm=29.2,
                     depth_est_method="ring_plane_v1|upper_bound")
    rules = dict(notify.rules_for_defect(row))
    msg = rules["deep_pothole_suspect"]
    assert "не исключает" in msg and "29.2" in msg
    assert "может достигать" not in msg


def test_shallow_estimate_no_suspect():
    row = defect_row(depth_bucket="shallow", depth_est_cm=2.0,
                     depth_est_high_cm=3.0)
    assert rules_of(row) == []


def test_suspect_only_for_potholes():
    row = defect_row(**{"class": "patch", "depth_bucket": "deep"})
    assert rules_of(row) == []


def test_no_rules_for_conforming_defect():
    assert rules_of(defect_row(non_conforming="no")) == []


# --- create_notifications: сквозной через БД ----------------------------------

def _report(cls="pothole", non_conforming="yes", bucket="deep"):
    return {
        "image": "x.jpg", "mode": "single", "image_size_px": [32, 32],
        "scale": {"available": False, "reference": {}, "homography_applied": False},
        "defects": [{
            "id": 1, "class": cls, "confidence": 0.9, "bbox_px": [1, 1, 10, 10],
            "mask_rle": {"size": [32, 32], "counts": [1024]},
            "shape": {"length_px": 10.0, "width_px": 8.0},
            "metric": {"available": False, "depth_cm": None,
                       "depth_bucket": bucket},
            "severity": {"non_conforming": non_conforming, "note": "тест"},
        }],
        "location": {"available": False}, "warnings": [],
        "engine_version": "0.1.0-test",
    }


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "n.db")
    db.init_db(c)
    yield c
    c.close()


def test_create_notifications_idempotent(conn):
    db.import_report(conn, _report(), "r1")
    # яма: и несоответствие ГОСТ, и deep-бакет → два правила
    assert notify.create_notifications(conn) == 2
    assert notify.create_notifications(conn) == 0  # повтор не плодит дубликатов
    rows = conn.execute("SELECT * FROM notifications ORDER BY rule").fetchall()
    assert [r["rule"] for r in rows] == ["deep_pothole_suspect", "gost_non_conforming"]
    assert all(r["status"] == "new" for r in rows)


def test_create_notifications_none_needed(conn):
    db.import_report(conn, _report(cls="patch", non_conforming="no",
                                   bucket=None), "r2")
    assert notify.create_notifications(conn) == 0
