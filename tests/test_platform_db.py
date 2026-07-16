"""Тесты SQLite-хранилища платформы (подсистема 2). Без сети и моделей."""
import json
from pathlib import Path

import pytest

from road_defect.platform import db


def make_report(image="photo.jpg", cls="pothole", non_conforming="yes",
                depth_estimate="point", location=None, pair_id=None,
                depth_bucket="deep"):
    """Минимально полный отчёт по контракту report.py (см. build_report)."""
    metric = {
        "available": False, "equivalent_diameter_cm": None, "length_cm": None,
        "width_cm": None, "area_cm2": None, "area_m2": None,
        "depth_cm": None,                      # сертифицированной глубины нет — принцип
        "depth_bucket": depth_bucket, "depth_method": None,
        "depth_certifiable": False, "confidence": None,
        "error_band_pct": None, "view_tilt_deg": None,
    }
    if depth_estimate == "point":
        metric["depth_cm_estimate"] = {
            "available": True, "point_cm": 6.0, "low_cm": 3.0, "high_cm": 9.0,
            "upper_bound_cm": None, "method": "two_view_v0",
            "confidence": "low", "certifiable": False,
            "vs_gost_5cm": "above", "reason": None,
            "note": "оценка, не измерение"}
    elif depth_estimate == "upper_bound":
        metric["depth_cm_estimate"] = {
            "available": True, "point_cm": None, "low_cm": None, "high_cm": None,
            "upper_bound_cm": 2.0, "method": "two_view_v0",
            "confidence": "low", "certifiable": False,
            "vs_gost_5cm": "below", "reason": None,
            "note": "оценка, не измерение"}
    elif depth_estimate == "none":
        metric["depth_cm_estimate"] = None
    # depth_estimate == "absent": ключа нет вовсе (старые отчёты)

    report = {
        "image": image,
        "mode": "single",
        "image_size_px": [32, 32],
        "scale": {"available": False, "mm_per_px": None, "reference": {},
                  "homography_applied": False, "note": "эталон не найден"},
        "defects": [{
            "id": 1, "class": cls, "raw_label": "D40", "confidence": 0.9,
            "bbox_px": [1, 1, 10, 10],
            "mask_rle": {"size": [32, 32], "counts": [0, 4, 28, 4, 988]},
            "shape": {"length_px": 10.0, "width_px": 8.0,
                      "equivalent_diameter_px": 9.0},
            "metric": metric,
            "severity": {"standard": "ГОСТ Р 50597-2017",
                         "length_exceeds": None, "area_exceeds": None,
                         "depth_exceeds": None,
                         "non_conforming": non_conforming,
                         "repair_deadline_days": None,
                         "hazard_signing_required": None,
                         "note": "тестовый вердикт"},
        }],
        "location": location or {"available": False, "source": "manual_later"},
        "warnings": [],
        "engine_version": "0.1.0-test",
    }
    if pair_id is not None:
        report["pair_id"] = pair_id
    return report


@pytest.fixture()
def conn(tmp_path):
    # родительской папки нет — connect обязан её создать
    c = db.connect(tmp_path / "вложенная" / "platform.db")
    db.init_db(c)
    yield c
    c.close()


def test_connect_creates_parent_and_row_factory(tmp_path, conn):
    assert (tmp_path / "вложенная" / "platform.db").is_file()
    conn.execute("INSERT INTO reports (stem, created_utc, json) VALUES ('a','t','{}')")
    row = conn.execute("SELECT stem FROM reports").fetchone()
    assert row["stem"] == "a"  # sqlite3.Row: доступ по имени колонки


def test_init_db_idempotent(conn):
    db.init_db(conn)  # повторный вызов не падает и не стирает данные
    db.init_db(conn)


def test_import_report_flattens_defect(conn):
    rid = db.import_report(conn, make_report(), "r001")
    row = conn.execute("SELECT * FROM defects").fetchone()
    assert row["report_id"] == rid
    assert row["class"] == "pothole"
    assert row["confidence"] == pytest.approx(0.9)
    assert row["length_cm"] is None            # нет эталона — нет сантиметров
    assert row["depth_bucket"] == "deep"
    assert row["depth_est_cm"] == pytest.approx(6.0)
    assert row["depth_est_low_cm"] == pytest.approx(3.0)
    assert row["depth_est_high_cm"] == pytest.approx(9.0)
    assert row["depth_est_method"] == "two_view_v0"
    assert row["non_conforming"] == "yes"
    # исходный JSON сохранён целиком
    saved = json.loads(conn.execute("SELECT json FROM reports").fetchone()["json"])
    assert saved["defects"][0]["metric"]["depth_cm"] is None


def test_depth_estimate_upper_bound_only(conn):
    db.import_report(conn, make_report(depth_estimate="upper_bound"), "r_ub")
    row = conn.execute("SELECT * FROM defects").fetchone()
    assert row["depth_est_cm"] == pytest.approx(2.0)
    assert "upper_bound" in row["depth_est_method"]


@pytest.mark.parametrize("variant", ["absent", "none"])
def test_depth_estimate_absent_or_none(conn, variant):
    # старые отчёты без ключа depth_cm_estimate и отчёты с None — не падаем
    db.import_report(conn, make_report(depth_estimate=variant), f"r_{variant}")
    row = conn.execute("SELECT * FROM defects").fetchone()
    assert row["depth_est_cm"] is None
    assert row["depth_est_method"] is None


def test_import_report_upsert_replaces_and_cascades(conn):
    rid1 = db.import_report(conn, make_report(), "r001")
    defect_id = conn.execute("SELECT id FROM defects").fetchone()["id"]
    conn.execute(
        "INSERT INTO notifications (report_id, defect_id, rule, message, created_utc)"
        " VALUES (?, ?, 'x', 'm', 't')", (rid1, defect_id))
    conn.commit()

    rid2 = db.import_report(conn, make_report(cls="patch"), "r001")
    assert db.get_report(conn, rid2)["stem"] == "r001"
    assert conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 1
    defects = conn.execute("SELECT * FROM defects").fetchall()
    assert len(defects) == 1 and defects[0]["class"] == "patch"
    # каскад: старый дефект удалён → его уведомления тоже
    assert conn.execute("SELECT COUNT(*) FROM notifications").fetchone()[0] == 0


def test_stats_and_map_points(conn):
    db.import_report(conn, make_report(), "r1",
                     lat=47.27743, lon=39.70409, gps_source="journal")
    db.import_report(conn, make_report(cls="patch", non_conforming="no",
                                       depth_estimate="absent"), "r2")
    s = db.stats(conn)
    assert s["reports"] == 2
    assert s["defects"] == 2
    assert s["potholes"] == 1
    assert s["by_class"] == {"pothole": 1, "patch": 1}
    assert s["non_conforming"] == 1
    assert s["with_coords"] == 1
    assert s["new_notifications"] == 0

    points = db.map_points(conn)
    assert len(points) == 1  # только отчёт с координатами
    p = points[0]
    assert p["lat"] == pytest.approx(47.27743)
    assert p["lon"] == pytest.approx(39.70409)
    assert p["class"] == "pothole"
    assert p["non_conforming"] == "yes"
    assert p["report_id"] == 1


def test_list_reports_and_get_report(conn):
    rid = db.import_report(conn, make_report(), "r1")
    rows = db.list_reports(conn, limit=10)
    assert len(rows) == 1 and rows[0]["n_defects"] == 1
    assert db.get_report(conn, rid)["stem"] == "r1"
    assert db.get_report(conn, 999) is None


def test_status_lifecycle(conn):
    rid = db.import_report(conn, make_report(), "r1")
    assert db.get_report(conn, rid)["status"] == "new"      # дефолт учёта

    assert db.set_report_status(conn, rid, "in_progress") is True
    assert db.get_report(conn, rid)["status"] == "in_progress"
    assert db.set_report_status(conn, rid, "fixed") is True
    assert db.get_report(conn, rid)["status"] == "fixed"

    # whitelist: значение приходит из HTTP-формы, чужое — ошибка, не запись
    with pytest.raises(ValueError):
        db.set_report_status(conn, rid, "готово")
    assert db.set_report_status(conn, 999, "fixed") is False


def test_status_preserved_on_reimport(conn):
    rid = db.import_report(conn, make_report(), "r1")
    db.set_report_status(conn, rid, "fixed")
    # повторный импорт того же stem (upsert) не обнуляет работу оператора
    rid2 = db.import_report(conn, make_report(cls="patch"), "r1")
    assert db.get_report(conn, rid2)["status"] == "fixed"
    # а новый stem начинает с «new»
    rid3 = db.import_report(conn, make_report(), "r2")
    assert db.get_report(conn, rid3)["status"] == "new"


def test_init_db_migrates_old_schema(tmp_path):
    # БД, созданная до появления статуса учёта: reports без колонки status
    c = db.connect(tmp_path / "old.db")
    c.executescript("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY,
            stem TEXT UNIQUE NOT NULL,
            created_utc TEXT NOT NULL,
            json TEXT NOT NULL
        );""")
    c.execute("INSERT INTO reports (stem, created_utc, json)"
              " VALUES ('старый', 't', '{}')")
    c.commit()
    db.init_db(c)  # обязана добавить колонку, не тронув данные
    row = c.execute("SELECT * FROM reports").fetchone()
    assert row["stem"] == "старый"
    assert row["status"] == "new"
    db.init_db(c)  # и остаться идемпотентной
    c.close()


def test_stats_status_counts(conn):
    for stem in ("r1", "r2", "r3"):
        db.import_report(conn, make_report(), stem)
    db.set_report_status(conn, 1, "in_progress")
    db.set_report_status(conn, 2, "fixed")
    s = db.stats(conn)
    assert s["in_progress"] == 1
    assert s["fixed"] == 1


def test_list_reports_status_filter(conn):
    db.import_report(conn, make_report(), "r1")
    db.import_report(conn, make_report(), "r2")
    db.set_report_status(conn, 1, "fixed")
    assert [r["stem"] for r in db.list_reports(conn, status="fixed")] == ["r1"]
    assert [r["stem"] for r in db.list_reports(conn, status="new")] == ["r2"]
    assert len(db.list_reports(conn)) == 2


def test_delete_report_cascades(conn):
    rid = db.import_report(conn, make_report(), "r1")
    defect_id = conn.execute("SELECT id FROM defects").fetchone()["id"]
    conn.execute(
        "INSERT INTO notifications (report_id, defect_id, rule, message, created_utc)"
        " VALUES (?, ?, 'x', 'm', 't')", (rid, defect_id))
    conn.commit()

    assert db.delete_report(conn, rid) is True
    for table in ("reports", "defects", "notifications"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert db.delete_report(conn, rid) is False  # повторно — уже нечего


def test_export_rows_honest_nulls(conn):
    db.import_report(conn, make_report(), "r1",
                     lat=47.27743, lon=39.70409, gps_source="journal")
    db.import_report(conn, make_report(cls="patch", non_conforming="no",
                                       depth_estimate="absent"), "r2")
    rows = db.export_rows(conn)
    assert [r["stem"] for r in rows] == ["r1", "r2"]
    first = rows[0]
    assert first["class"] == "pothole"
    assert first["length_cm"] is None          # нет эталона — NULL, не ноль
    assert first["depth_est_cm"] == pytest.approx(6.0)
    assert first["status"] == "new"
    assert rows[1]["depth_est_cm"] is None


def test_import_reports_directory(tmp_path, conn, capsys):
    out = tmp_path / "outputs"
    out.mkdir()
    journal = tmp_path / "journal.csv"
    journal.write_text("id,street_or_gps\n006,47.27932 39.70642\n",
                       encoding="utf-8")

    # a: без координат, с оверлеем рядом; с оценкой глубины
    (out / "a.json").write_text(json.dumps(make_report()), encoding="utf-8")
    (out / "a_annotated.jpg").write_bytes(b"jpg")
    # b: location из самого отчёта (движок/оператор)
    (out / "b.json").write_text(json.dumps(make_report(
        depth_estimate="absent",
        location={"available": True, "lat": 47.5, "lon": 39.5,
                  "source": "manual"})), encoding="utf-8")
    # c: координаты по pair_id из журнала
    (out / "c.json").write_text(json.dumps(make_report(pair_id="006")),
                                encoding="utf-8")
    # битый JSON — пропуск с печатью, не падение
    (out / "broken.json").write_text("{не json", encoding="utf-8")

    n = db.import_reports(conn, out, journal_csv=journal)
    assert n == 3
    assert "broken.json" in capsys.readouterr().out

    rows = {r["stem"]: r for r in conn.execute("SELECT * FROM reports").fetchall()}
    assert set(rows) == {"a", "b", "c"}
    assert rows["a"]["overlay_path"].endswith("a_annotated.jpg")
    assert rows["a"]["lat"] is None
    assert rows["b"]["lat"] == pytest.approx(47.5)
    assert rows["b"]["gps_source"] == "manual"
    assert rows["c"]["lat"] == pytest.approx(47.27932)
    assert rows["c"]["gps_source"] == "journal"

    # повторный импорт идемпотентен по числу отчётов (upsert по stem)
    assert db.import_reports(conn, out, journal_csv=journal) == 3
    assert conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 3
