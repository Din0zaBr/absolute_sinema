"""SQLite-хранилище отчётов CV-движка (подсистема 2).

Источник правды — JSON-отчёт движка (report.py): он целиком кладётся в колонку
reports.json, а таблица defects — лишь плоская проекция для выборок/карты/
уведомлений. При расхождении верить JSON.

Честность измерений сохраняется на уровне схемы: length_cm/width_cm/area_m2
допускают NULL (нет ИСТОЧНИКА масштаба → нет сантиметров), depth_est_* — только
ОЦЕНКА глубины (depth_cm_estimate из metric), никогда не «измерение»;
сертифицированная глубина (metric.depth_cm) в этой версии всегда null и колонки
не имеет. scale_source/metric_confidence (2026-07-16) хранят происхождение
размеров: reference — эталон; camera_height/mixed — ОЦЕНКА от высоты камеры
(confidence=low, не для актирования) — без этих колонок оценки в реестре были
неотличимы от эталонных замеров.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import geo

# Жизненный цикл учёта: new → in_progress → fixed. Статус — атрибут УЧЁТА
# (работа дорожной службы), а не измерения, поэтому живёт в reports, а не в JSON.
REPORT_STATUSES = ("new", "in_progress", "fixed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY,
    stem TEXT UNIQUE NOT NULL,
    image TEXT,
    mode TEXT,
    created_utc TEXT NOT NULL,
    engine_version TEXT,
    lat REAL,
    lon REAL,
    gps_source TEXT,
    source_path TEXT,
    overlay_path TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY,
    report_id INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    defect_index INTEGER,
    class TEXT,
    confidence REAL,
    length_cm REAL,
    width_cm REAL,
    area_m2 REAL,
    scale_source TEXT,
    metric_confidence TEXT,
    depth_bucket TEXT,
    depth_est_cm REAL,
    depth_est_low_cm REAL,
    depth_est_high_cm REAL,
    depth_est_method TEXT,
    non_conforming TEXT,
    severity_note TEXT
);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY,
    report_id INTEGER NOT NULL,
    defect_id INTEGER NOT NULL REFERENCES defects(id) ON DELETE CASCADE,
    rule TEXT NOT NULL,
    message TEXT NOT NULL,
    created_utc TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',
    UNIQUE(defect_id, rule)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Открыть БД (создав родительскую папку). foreign_keys=ON обязателен:
    без него каскадное удаление дефектов/уведомлений молча не работает."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Создать схему. Идемпотентно (CREATE TABLE IF NOT EXISTS + миграции)."""
    conn.executescript(_SCHEMA)
    # Миграция БД, созданных до появления статуса учёта: CREATE TABLE IF NOT
    # EXISTS существующую таблицу не трогает, колонку добавляем отдельно.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(reports)")}
    if "status" not in cols:
        conn.execute("ALTER TABLE reports"
                     " ADD COLUMN status TEXT NOT NULL DEFAULT 'new'")
    # Миграция 2026-07-16: происхождение размеров (масштаб от высоты камеры).
    dcols = {row[1] for row in conn.execute("PRAGMA table_info(defects)")}
    for col in ("scale_source", "metric_confidence"):
        if col not in dcols:
            conn.execute(f"ALTER TABLE defects ADD COLUMN {col} TEXT")
    conn.commit()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _depth_estimate_columns(metric: dict) -> dict:
    """Плоские колонки оценки глубины из metric.depth_cm_estimate.

    Ключ может ПОЛНОСТЬЮ отсутствовать (старые отчёты), быть None, либо dict,
    где задан point_cm ИЛИ только upper_bound_cm («глубина < X см»). Во втором
    случае в depth_est_cm кладём верхнюю границу, а метод помечаем upper_bound —
    потребитель обязан видеть, что это граница, а не точечная оценка.
    """
    cols = {"depth_est_cm": None, "depth_est_low_cm": None,
            "depth_est_high_cm": None, "depth_est_method": None}
    est = metric.get("depth_cm_estimate")
    if not isinstance(est, dict) or not est.get("available"):
        return cols
    cols["depth_est_low_cm"] = est.get("low_cm")
    cols["depth_est_high_cm"] = est.get("high_cm")
    method = est.get("method")
    if est.get("point_cm") is not None:
        cols["depth_est_cm"] = est["point_cm"]
        cols["depth_est_method"] = method
    elif est.get("upper_bound_cm") is not None:
        cols["depth_est_cm"] = est["upper_bound_cm"]
        cols["depth_est_method"] = f"{method}|upper_bound" if method else "upper_bound"
    else:
        cols["depth_est_method"] = method
    return cols


def import_report(conn: sqlite3.Connection, report: dict, stem: str,
                  source_path: str | None = None, overlay_path: str | None = None,
                  lat: float | None = None, lon: float | None = None,
                  gps_source: str | None = None) -> int:
    """Записать отчёт в БД (upsert по stem) и вернуть report_id.

    Перезапись реализована как DELETE старой строки + INSERT: каскад чистит
    дефекты и их уведомления, поэтому повторный импорт не плодит дубликатов
    (закрытые уведомления при этом теряются — осознанная плата за простоту).
    Статус учёта (new/in_progress/fixed) при перезаписи СОХРАНЯЕТСЯ: это работа
    оператора, повторный импорт того же stem не должен её обнулять.
    """
    prev = conn.execute("SELECT status FROM reports WHERE stem = ?",
                        (stem,)).fetchone()
    status = prev["status"] if prev is not None else "new"
    conn.execute("DELETE FROM reports WHERE stem = ?", (stem,))
    cur = conn.execute(
        "INSERT INTO reports (stem, image, mode, created_utc, engine_version,"
        " lat, lon, gps_source, source_path, overlay_path, status, json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (stem, report.get("image"), report.get("mode"), _utc_now(),
         report.get("engine_version"), lat, lon, gps_source,
         str(source_path) if source_path else None,
         str(overlay_path) if overlay_path else None,
         status, json.dumps(report, ensure_ascii=False)))
    report_id = cur.lastrowid
    for idx, d in enumerate(report.get("defects") or [], start=1):
        metric = d.get("metric") or {}
        severity = d.get("severity") or {}
        est = _depth_estimate_columns(metric)
        conn.execute(
            "INSERT INTO defects (report_id, defect_index, class, confidence,"
            " length_cm, width_cm, area_m2, scale_source, metric_confidence,"
            " depth_bucket, depth_est_cm,"
            " depth_est_low_cm, depth_est_high_cm, depth_est_method,"
            " non_conforming, severity_note)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (report_id, d.get("id", idx), d.get("class"), d.get("confidence"),
             metric.get("length_cm"), metric.get("width_cm"),
             metric.get("area_m2"), metric.get("scale_source"),
             metric.get("confidence"), metric.get("depth_bucket"),
             est["depth_est_cm"], est["depth_est_low_cm"],
             est["depth_est_high_cm"], est["depth_est_method"],
             severity.get("non_conforming"), severity.get("note")))
    conn.commit()
    return report_id


def _report_latlon(report: dict, stem: str,
                   journal_csv: Path | None) -> tuple[float | None, float | None, str | None]:
    """Координаты отчёта: location из JSON → журнал по pair_id → нет.

    Порядок отражает доверие к источникам: location пишет сам движок (или
    оператор), журнал — ручной, но привязан к id пары; выдумать третий
    источник нельзя (принцип честности)."""
    loc = report.get("location") or {}
    if loc.get("available"):
        pair = geo.parse_latlon(f"{loc.get('lat')} {loc.get('lon')}")
        if pair is not None:
            return pair[0], pair[1], str(loc.get("source") or "report_location")
    if journal_csv is not None:
        pair_id = report.get("pair_id") or stem.split("_", 1)[0]
        pair = geo.journal_latlon(journal_csv, str(pair_id))
        if pair is not None:
            return pair[0], pair[1], "journal"
    return None, None, None


def import_reports(conn: sqlite3.Connection, out_dir: Path,
                   journal_csv: Path | None = None) -> int:
    """Импортировать все *.json из папки результатов движка. Вернуть число
    импортированных. Битый JSON пропускается с печатью — один плохой файл
    не должен ронять импорт остальных."""
    out_dir = Path(out_dir)
    count = 0
    for path in sorted(out_dir.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            print(f"[пропуск] {path.name}: не читается как JSON ({exc})")
            continue
        if not isinstance(report, dict) or "defects" not in report:
            print(f"[пропуск] {path.name}: не похож на отчёт движка")
            continue
        stem = path.stem
        overlay = path.with_name(f"{stem}_annotated.jpg")
        lat, lon, gps_source = _report_latlon(report, stem, journal_csv)
        # Пути храним абсолютными: относительный путь зависел бы от cwd
        # запуска импорта и позже не резолвился бы (миниатюры/медиа).
        import_report(conn, report, stem,
                      source_path=str(path.resolve()),
                      overlay_path=str(overlay.resolve()) if overlay.is_file() else None,
                      lat=lat, lon=lon, gps_source=gps_source)
        count += 1
    return count


# --- Жизненный цикл учёта ------------------------------------------------------

def set_report_status(conn: sqlite3.Connection, report_id: int,
                      status: str) -> bool:
    """Сменить статус устранения отчёта. Неизвестный статус — ValueError
    (whitelist REPORT_STATUSES, значение приходит из HTTP-формы).
    Возвращает False, если отчёта с таким id нет."""
    if status not in REPORT_STATUSES:
        raise ValueError(f"неизвестный статус: {status!r}")
    cur = conn.execute("UPDATE reports SET status = ? WHERE id = ?",
                       (status, int(report_id)))
    conn.commit()
    return cur.rowcount > 0


def delete_report(conn: sqlite3.Connection, report_id: int) -> bool:
    """Удалить отчёт из УЧЁТА (строки reports/defects/notifications — каскадом).
    Файлы на диске (загрузка, JSON, оверлей) намеренно не трогаются: БД — не
    единственный их потребитель, а удаление записи должно быть дёшево обратимо
    повторным импортом."""
    cur = conn.execute("DELETE FROM reports WHERE id = ?", (int(report_id),))
    conn.commit()
    return cur.rowcount > 0


# --- Выборки для UI ----------------------------------------------------------

def list_reports(conn: sqlite3.Connection, limit: int = 50,
                 status: str | None = None) -> list[sqlite3.Row]:
    """Последние отчёты с количеством дефектов (для таблицы дашборда).
    status — необязательный фильтр по статусу устранения."""
    where = ""
    args: list = []
    if status is not None:
        where = " WHERE r.status = ?"
        args.append(status)
    args.append(int(limit))
    return conn.execute(
        "SELECT r.*, (SELECT COUNT(*) FROM defects d WHERE d.report_id = r.id)"
        f" AS n_defects FROM reports r{where} ORDER BY r.id DESC LIMIT ?",
        args).fetchall()


def get_report(conn: sqlite3.Connection, report_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM reports WHERE id = ?",
                        (int(report_id),)).fetchone()


def list_defects(conn: sqlite3.Connection,
                 report_id: int | None = None) -> list[sqlite3.Row]:
    """Дефекты (плоская проекция), при report_id — только одного отчёта."""
    if report_id is None:
        return conn.execute(
            "SELECT * FROM defects ORDER BY report_id, defect_index").fetchall()
    return conn.execute(
        "SELECT * FROM defects WHERE report_id = ? ORDER BY defect_index",
        (int(report_id),)).fetchall()


def stats(conn: sqlite3.Connection) -> dict:
    """Сводка для плиток дашборда. Считает по плоской проекции defects."""
    one = lambda sql, *args: conn.execute(sql, args).fetchone()[0]  # noqa: E731
    by_class = {row["class"]: row["n"] for row in conn.execute(
        "SELECT class, COUNT(*) AS n FROM defects GROUP BY class ORDER BY n DESC")}
    return {
        "reports": one("SELECT COUNT(*) FROM reports"),
        "defects": one("SELECT COUNT(*) FROM defects"),
        "potholes": by_class.get("pothole", 0),
        "by_class": by_class,
        "non_conforming": one(
            "SELECT COUNT(*) FROM defects WHERE non_conforming = 'yes'"),
        "with_coords": one(
            "SELECT COUNT(*) FROM reports WHERE lat IS NOT NULL AND lon IS NOT NULL"),
        "new_notifications": one(
            "SELECT COUNT(*) FROM notifications WHERE status = 'new'"),
        "in_progress": one(
            "SELECT COUNT(*) FROM reports WHERE status = 'in_progress'"),
        "fixed": one("SELECT COUNT(*) FROM reports WHERE status = 'fixed'"),
    }


def map_points(conn: sqlite3.Connection) -> list[dict]:
    """Точки для карты: по дефекту на маркер, только отчёты с координатами."""
    rows = conn.execute(
        "SELECT d.id AS defect_id, d.class, d.non_conforming, d.depth_bucket,"
        " d.confidence, r.id AS report_id, r.lat, r.lon, r.image, r.status"
        " FROM defects d JOIN reports r ON r.id = d.report_id"
        " WHERE r.lat IS NOT NULL AND r.lon IS NOT NULL").fetchall()
    return [dict(row) for row in rows]


def export_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Реестр дефектов для экспорта в CSV: дефект + атрибуты его отчёта.
    NULL остаются NULL (нет эталона — пустая клетка, не ноль)."""
    return conn.execute(
        "SELECT r.id AS report_id, r.stem, r.image, r.created_utc, r.status,"
        " r.lat, r.lon, r.gps_source,"
        " d.defect_index, d.class, d.confidence, d.length_cm, d.width_cm,"
        " d.area_m2, d.scale_source, d.metric_confidence,"
        " d.depth_bucket, d.depth_est_cm, d.depth_est_low_cm,"
        " d.depth_est_high_cm, d.depth_est_method, d.non_conforming,"
        " d.severity_note"
        " FROM defects d JOIN reports r ON r.id = d.report_id"
        " ORDER BY r.id, d.defect_index").fetchall()
