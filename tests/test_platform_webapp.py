"""Тесты веб-интерфейса платформы: живой сервер на порту 0, фейковый пайплайн.

Без сети (только 127.0.0.1) и без моделей: pipeline_factory подменяется
объектом, который возвращает канонический отчёт и крошечный overlay.
"""
import base64
import json
import sqlite3
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from road_defect.platform import db
from road_defect.platform.webapp import PlatformApp

TIMEOUT = 10
# настоящий 1x1 PNG — содержимое сервер не разбирает, но пусть будет честным
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def make_report(image="photo.png"):
    """Канонический отчёт по контракту report.py (минимально полный)."""
    return {
        "image": image, "mode": "single", "image_size_px": [32, 32],
        "scale": {"available": False, "mm_per_px": None, "reference": {},
                  "homography_applied": False, "note": "эталон не найден"},
        "defects": [{
            "id": 1, "class": "pothole", "raw_label": "D40", "confidence": 0.9,
            "bbox_px": [1, 1, 10, 10],
            "mask_rle": {"size": [32, 32], "counts": [0, 4, 28, 4, 988]},
            "shape": {"length_px": 10.0, "width_px": 8.0,
                      "equivalent_diameter_px": 9.0},
            "metric": {"available": False, "equivalent_diameter_cm": None,
                       "length_cm": None, "width_cm": None, "area_cm2": None,
                       "area_m2": None, "depth_cm": None, "depth_bucket": "deep",
                       "depth_method": None, "depth_certifiable": False,
                       "confidence": None, "error_band_pct": None,
                       "view_tilt_deg": None,
                       "depth_cm_estimate": {
                           "available": True, "point_cm": 6.0, "low_cm": 3.0,
                           "high_cm": 9.0, "upper_bound_cm": None,
                           "method": "two_view_v0", "confidence": "low",
                           "certifiable": False, "vs_gost_5cm": "above",
                           "reason": None, "note": "оценка, не измерение"}},
            "severity": {"standard": "ГОСТ Р 50597-2017",
                         "length_exceeds": None, "area_exceeds": None,
                         "depth_exceeds": None, "non_conforming": "yes",
                         "repair_deadline_days": None,
                         "hazard_signing_required": None,
                         "note": "тестовый вердикт"},
        }],
        "location": {"available": False, "source": "manual_later"},
        "warnings": ["тестовое предупреждение"],
        "engine_version": "0.1.0-test",
    }


class FakePipeline:
    """Заменяет DefectPipeline: канонический отчёт + overlay 32x32x3."""

    def analyze_image(self, path):
        return (make_report(image=Path(path).name),
                np.zeros((32, 32, 3), dtype=np.uint8), [])


def _start(app):
    server = app.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture()
def served(tmp_path):
    app = PlatformApp(tmp_path / "data", pipeline_factory=FakePipeline)
    server, thread, base = _start(app)
    yield app, base
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def fetch(url, data=None, headers=None):
    """(status, body:str, final_url). 4xx/5xx не бросают исключение."""
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read().decode("utf-8", "replace"), resp.url
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace"), url


def multipart(filename, payload):
    boundary = "----platformtestboundary"
    body = (
        (f"--{boundary}\r\n"
         f'Content-Disposition: form-data; name="photo"; filename="{filename}"\r\n'
         "Content-Type: application/octet-stream\r\n\r\n").encode("utf-8")
        + payload + f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


# --- страницы ------------------------------------------------------------------

def test_index_ok_and_russian(served):
    _app, base = served
    status, body, _ = fetch(base + "/")
    assert status == 200
    assert "Дашборд" in body
    assert "Отчётов" in body


def test_health(served):
    _app, base = served
    status, body, _ = fetch(base + "/health")
    assert status == 200
    assert json.loads(body) == {"ok": True}


def test_unknown_route_404(served):
    _app, base = served
    status, _, _ = fetch(base + "/nope")
    assert status == 404
    status, _, _ = fetch(base + "/report/999")
    assert status == 404


def test_map_empty_placeholder(served):
    _app, base = served
    status, body, _ = fetch(base + "/map")
    assert status == 200
    assert "Точек с координатами пока нет" in body


# --- загрузка → анализ → отчёт ---------------------------------------------------

def test_upload_full_flow(served):
    app, base = served
    body, headers = multipart("photo.png", TINY_PNG)
    status, page, final_url = fetch(base + "/upload", data=body, headers=headers)

    # редирект после POST приводит на страницу отчёта
    assert status == 200
    assert "/report/" in final_url
    assert "pothole" in page and "яма" in page
    # честность: нет эталона → размеры в px; глубина — только «оценка»
    assert "нет эталона масштаба" in page
    # шаблон капитализирует первую букву depth_text (фикс «Глубина: глубина:»)
    assert "Глубина ≈ 6.0 см (3.0–9.0, оценка)" in page
    assert "не соответствует ГОСТ" in page
    assert "тестовое предупреждение" in page

    # отчёт лёг в БД, JSON и overlay записаны в data_dir
    with closing(sqlite3.connect(app.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM reports").fetchall()
        assert len(rows) == 1
        stem = rows[0]["stem"]
        n_notif = conn.execute(
            "SELECT COUNT(*) FROM notifications WHERE status='new'").fetchone()[0]
    assert (app.outputs_dir / f"{stem}.json").is_file()
    assert (app.outputs_dir / f"{stem}_annotated.jpg").is_file()
    # non_conforming=yes → уведомление создано (плюс deep-подозрение)
    assert n_notif >= 1

    # overlay отдаётся через /media
    status, _, _ = fetch(base + f"/media/outputs/{stem}_annotated.jpg")
    assert status == 200

    # уведомление видно на странице и закрывается кнопкой
    status, page, _ = fetch(base + "/notifications")
    assert status == 200 and "Несоответствие ГОСТ" in page
    with closing(sqlite3.connect(app.db_path)) as conn:
        nid = conn.execute("SELECT id FROM notifications").fetchone()[0]
    status, page, _ = fetch(base + f"/notifications/{nid}/done", data=b"")
    assert status == 200
    with closing(sqlite3.connect(app.db_path)) as conn:
        st = conn.execute("SELECT status FROM notifications WHERE id=?",
                          (nid,)).fetchone()[0]
    assert st == "done"


def test_upload_same_name_makes_unique_stem(served):
    app, base = served
    body, headers = multipart("photo.png", TINY_PNG)
    fetch(base + "/upload", data=body, headers=headers)
    fetch(base + "/upload", data=body, headers=headers)
    with closing(sqlite3.connect(app.db_path)) as conn:
        stems = [r[0] for r in conn.execute("SELECT stem FROM reports")]
    assert len(stems) == 2 and len(set(stems)) == 2


def test_upload_bad_extension_rejected(served):
    _app, base = served
    body, headers = multipart("report.txt", b"hello")
    status, page, _ = fetch(base + "/upload", data=body, headers=headers)
    assert status == 400
    assert "не поддерживается" in page


def test_upload_not_multipart(served):
    _app, base = served
    status, page, _ = fetch(base + "/upload", data=b"raw",
                            headers={"Content-Type": "text/plain"})
    assert status == 400


def test_no_model_mode_503(tmp_path):
    app = PlatformApp(tmp_path / "data2", pipeline_factory=None)
    server, thread, base = _start(app)
    try:
        body, headers = multipart("photo.png", TINY_PNG)
        status, page, _ = fetch(base + "/upload", data=body, headers=headers)
        assert status == 503
        assert "без CV-моделей" in page
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- /media: защита от traversal --------------------------------------------------

@pytest.mark.parametrize("path", [
    "/media/..%2f..%2fCLAUDE.md",
    "/media/../CLAUDE.md",
    "/media/..%5c..%5cCLAUDE.md",       # windows-обход обратными слэшами
])
def test_media_traversal_blocked(served, path):
    _app, base = served
    status, body, _ = fetch(base + path)
    assert status in (403, 404)
    assert "Road Defect Engine" not in body  # содержимое CLAUDE.md не утекло


def test_media_missing_file_404(served):
    _app, base = served
    # кириллическое имя файла уходит в URL перекодированным (как из браузера)
    rel = urllib.parse.quote("outputs/нет_такого.jpg")
    status, _, _ = fetch(base + "/media/" + rel)
    assert status == 404


# --- карта с точками ----------------------------------------------------------------

def test_map_with_points(served):
    app, base = served
    rep = make_report()
    with closing(db.connect(app.db_path)) as conn:
        db.import_report(conn, rep, "geo1", lat=47.27743, lon=39.70409,
                         gps_source="journal")
    status, body, _ = fetch(base + "/map")
    assert status == 200
    assert "leaflet" in body
    assert "47.27743" in body
    assert "интернета для тайлов" in body  # честная оговорка про офлайн
    assert "status_label" in body          # статус учёта попадает в попап


# --- жизненный цикл отчёта: статус, удаление, экспорт --------------------------------

FORM = {"Content-Type": "application/x-www-form-urlencoded"}


def seed_report(app, stem="r1", **kwargs):
    # image = stem: дашборд показывает r.image or r.stem, тестам нужно
    # различать отчёты по видимому тексту страницы
    with closing(db.connect(app.db_path)) as conn:
        return db.import_report(conn, make_report(image=f"{stem}.jpg"),
                                stem, **kwargs)


def test_report_status_flow(served):
    app, base = served
    rid = seed_report(app)

    status, page, final_url = fetch(base + f"/report/{rid}/status",
                                    data=b"status=in_progress", headers=FORM)
    assert status == 200 and f"/report/{rid}" in final_url  # 303 → страница отчёта
    assert "Статус устранения" in page and "в работе" in page

    fetch(base + f"/report/{rid}/status", data=b"status=fixed", headers=FORM)
    with closing(db.connect(app.db_path)) as conn:
        st = conn.execute("SELECT status FROM reports WHERE id=?",
                          (rid,)).fetchone()[0]
    assert st == "fixed"

    # whitelist статусов и несуществующий отчёт
    status, page, _ = fetch(base + f"/report/{rid}/status",
                            data=b"status=hacked", headers=FORM)
    assert status == 400 and "Недопустимый статус" in page
    status, _, _ = fetch(base + "/report/999/status",
                         data=b"status=fixed", headers=FORM)
    assert status == 404


def test_index_status_filter(served):
    app, base = served
    seed_report(app, "первый")
    rid2 = seed_report(app, "второй")
    fetch(base + f"/report/{rid2}/status", data=b"status=fixed", headers=FORM)

    status, body, _ = fetch(base + "/?status=fixed")
    assert status == 200
    assert "второй" in body and "первый" not in body
    # чужое значение фильтра игнорируется, показываются все
    status, body, _ = fetch(base + "/?status=%27drop%27")
    assert status == 200
    assert "второй" in body and "первый" in body


def test_delete_report_flow(served):
    app, base = served
    rid = seed_report(app)
    with closing(db.connect(app.db_path)) as conn:
        conn.execute(
            "INSERT INTO notifications (report_id, defect_id, rule, message,"
            " created_utc) SELECT report_id, id, 'x', 'm', 't' FROM defects")
        conn.commit()

    status, _, final_url = fetch(base + f"/report/{rid}/delete", data=b"",
                                 headers=FORM)
    assert status == 200 and final_url == base + "/"  # 303 → дашборд
    with closing(db.connect(app.db_path)) as conn:
        for table in ("reports", "defects", "notifications"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    status, _, _ = fetch(base + f"/report/{rid}")
    assert status == 404
    status, _, _ = fetch(base + f"/report/{rid}/delete", data=b"", headers=FORM)
    assert status == 404


def test_huge_id_is_404_not_500(served):
    # Ревью 2026-07-15: id > int64 ронял sqlite3 OverflowError → 500.
    app, base = served
    rid = seed_report(app)
    huge = "9" * 25
    assert fetch(base + f"/report/{huge}")[0] == 404
    assert fetch(base + f"/report/{huge}/status",
                 data=b"status=fixed", headers=FORM)[0] == 404
    assert fetch(base + f"/report/{huge}/delete", data=b"", headers=FORM)[0] == 404
    # done-маршрут исторически не 404-ит на несуществующем id — но и не 500
    assert fetch(base + f"/notifications/{huge}/done", data=b"")[0] == 200
    # живой отчёт не задет
    assert fetch(base + f"/report/{rid}")[0] == 200


def test_export_csv_escapes_formulas_and_ceiling(served):
    app, base = served
    # имя файла с формуло-опасным префиксом + оценка «верхняя граница»
    rep = make_report(image="=2+2.png")
    rep["defects"][0]["metric"]["depth_cm_estimate"] = {
        "available": True, "point_cm": None, "low_cm": None, "high_cm": None,
        "upper_bound_cm": 2.0, "method": "ring_plane_v1", "confidence": "low",
        "certifiable": False, "vs_gost_5cm": "below", "reason": None,
        "note": "оценка, не измерение"}
    with closing(db.connect(app.db_path)) as conn:
        db.import_report(conn, rep, "формулы")
    _status, text, _ = fetch(base + "/export.csv")
    row = [ln for ln in text.splitlines() if "формулы" in ln][0]
    assert "'=2+2.png" in row          # экранирование формулы для Excel
    # потолок 2.0 — в своей колонке, «точечная» пуста: …,бакет,точка,потолок,…
    assert ",deep,,2.0," in row
    assert "upper_bound" in row


def test_export_csv(served):
    app, base = served
    seed_report(app, "экспорт1", lat=47.27743, lon=39.70409, gps_source="journal")

    req = urllib.request.Request(base + "/export.csv")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/csv")
        assert "attachment" in resp.headers.get("Content-Disposition", "")
        raw = resp.read()

    assert raw.startswith(b"\xef\xbb\xbf")  # BOM: Excel/Windows открывает кириллицу
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    assert lines[0].startswith("отчёт_id,файл,stem")
    row = lines[1]
    assert "экспорт1" in row and "pothole" in row and "яма" in row
    assert ",новый," in row                  # статус учёта по-русски
    assert ",,," not in lines[0]             # заголовок без пустот
    assert ",6.0," in row                    # оценка глубины попала
    # честность: сантиметров без эталона нет — пустые клетки, не нули
    assert ",0,0," not in row and ",0.0,0.0," not in row
