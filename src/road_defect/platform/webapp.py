"""Веб-интерфейс платформы (подсистемы 3–5) на stdlib http.server.

Почему stdlib, а не фреймворк: платформа офлайновая и однопользовательская
(оператор на своей машине), новых pip-зависимостей проект не добавляет.
ThreadingHTTPServer обслуживает статику/страницы параллельно, но сам АНАЛИЗ
сериализован threading.Lock — модели (YOLO/SAM/DepthAnything) не потокобезопасны.

Запуск:
    $env:PYTHONPATH="src"
    .\\.venv\\Scripts\\python.exe -m road_defect.platform.webapp

Принцип честности наследуется от движка: сантиметры от эталона — как есть,
от высоты камеры — только с явной пометкой «оценка» и источником
(metric.scale_source); оценка глубины — только со словом «оценка».
"""
from __future__ import annotations

import argparse
import csv
import email
import email.policy
import io
import json
import mimetypes
import re
import sys
import threading
import traceback
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from . import db, geo, notify
from . import templates as templates_mod

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Ограничение тела запроса: защита от случайной загрузки видео/архива,
# а не «порог качества» — потому константа модуля, не config.py.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def _route_id(digits: str) -> int:
    """id из URL. За пределами int64 SQLite кладёт OverflowError (ревью
    2026-07-15: /report/9…9 давал 500) — возвращаем сентинель -1, который
    гарантированно не найдётся и уйдёт штатной веткой 404."""
    value = int(digits)
    return value if value <= 2**63 - 1 else -1


def _csv_safe(value):
    """Экранирование формул для Excel: ячейка, начинающаяся с = + - @ или
    таба/CR, трактуется как формула (CSV-инъекция). Строкам добавляем
    апостроф-префикс; числа и None не трогаем (ревью 2026-07-15)."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _default_pipeline_factory():
    """Ленивая фабрика боевого пайплайна: тяжёлые импорты (torch/ultralytics)
    происходят только при первой загрузке фото, а не при старте сервера."""
    from ..pipeline import DefectPipeline

    return DefectPipeline(use_depth=True)


class PlatformApp:
    """Состояние приложения: пути данных, БД, ленивый пайплайн, шаблоны.

    pipeline_factory — точка инъекции для тестов (фейковый пайплайн без сети
    и моделей); None означает режим --no-model: загрузка честно отвечает 503.
    SQLite-соединение НЕ разделяется между потоками — каждый запрос открывает
    своё (sqlite3 по умолчанию привязывает соединение к потоку создания).
    """

    def __init__(self, data_dir: str | Path,
                 pipeline_factory=_default_pipeline_factory):
        self.data_dir = Path(data_dir).resolve()
        self.uploads_dir = self.data_dir / "uploads"
        self.outputs_dir = self.data_dir / "outputs"
        self.db_path = self.data_dir / "platform.db"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_factory = pipeline_factory
        self._pipeline = None
        self._pipeline_lock = threading.Lock()
        self.analysis_lock = threading.Lock()
        self.env = templates_mod.build_env()
        with closing(db.connect(self.db_path)) as conn:
            db.init_db(conn)

    # --- инфраструктура ------------------------------------------------------
    def open_db(self):
        return db.connect(self.db_path)

    def get_pipeline(self):
        """Пайплайн создаётся один раз под замком; None = моделей нет."""
        if self.pipeline_factory is None:
            return None
        with self._pipeline_lock:
            if self._pipeline is None:
                self._pipeline = self.pipeline_factory()
        return self._pipeline

    def make_server(self, host: str = "127.0.0.1",
                    port: int = 8000) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer((host, port), _Handler)
        server.app = self  # хендлеры достают приложение через self.server.app
        return server

    def media_url(self, stored_path: str | None) -> str | None:
        """URL /media/... для пути из БД. Файлы вне data_dir не публикуются:
        /media отдаёт ТОЛЬКО из data_dir (см. _serve_media), поэтому для
        импортированных извне отчётов миниатюры честно отсутствуют."""
        if not stored_path:
            return None
        p = Path(stored_path)
        if not p.is_absolute():
            p = self.data_dir / p
        try:
            rel = p.resolve().relative_to(self.data_dir)
        except (ValueError, OSError):
            return None
        return "/media/" + quote(rel.as_posix())


class _Handler(BaseHTTPRequestHandler):
    """Маршрутизация. Любая необработанная ошибка → страница 500 без
    stack trace в браузере (детали печатаются в консоль сервера)."""

    server_version = "RoadDefectPlatform/0.1"

    @property
    def app(self) -> PlatformApp:
        return self.server.app

    # --- отправка ответов ----------------------------------------------------
    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        # 303: после POST браузер обязан перейти GET'ом (upload → страница отчёта).
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _render(self, template: str, status: int = 200, **ctx) -> None:
        html = self.app.env.get_template(template).render(**ctx)
        self._send_html(status, html)

    def _message(self, status: int, title: str, text: str) -> None:
        self._render("message.html", status=status, title=title, text=text)

    # --- HTTP-методы ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (имя диктует BaseHTTPRequestHandler)
        self._safe(self._route_get)

    def do_POST(self) -> None:  # noqa: N802
        self._safe(self._route_post)

    def _safe(self, route) -> None:
        try:
            route()
        except (BrokenPipeError, ConnectionResetError):
            pass  # клиент ушёл — не наша ошибка
        except Exception:
            traceback.print_exc()
            try:
                self._message(500, "Внутренняя ошибка",
                              "Запрос не выполнен из-за внутренней ошибки "
                              "сервера. Подробности — в консоли сервера.")
            except Exception:
                pass

    # --- маршруты -------------------------------------------------------------
    def _route_get(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            return self._page_index()
        if path == "/upload":
            return self._render("upload.html", error=None)
        m = re.fullmatch(r"/report/(\d+)", path)
        if m:
            return self._page_report(_route_id(m.group(1)))
        if path == "/map":
            return self._page_map()
        if path == "/notifications":
            return self._page_notifications()
        if path == "/export.csv":
            return self._serve_export()
        if path.startswith("/media/"):
            return self._serve_media(path[len("/media/"):])
        if path == "/health":
            return self._send_json(200, {"ok": True})
        self._message(404, "Страница не найдена",
                      f"Маршрута «{path}» не существует.")

    def _route_post(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/upload":
            return self._handle_upload()
        m = re.fullmatch(r"/notifications/(\d+)/done", path)
        if m:
            return self._handle_notification_done(_route_id(m.group(1)))
        m = re.fullmatch(r"/report/(\d+)/status", path)
        if m:
            return self._handle_report_status(_route_id(m.group(1)))
        m = re.fullmatch(r"/report/(\d+)/delete", path)
        if m:
            return self._handle_report_delete(_route_id(m.group(1)))
        self._message(404, "Страница не найдена",
                      f"Маршрута «{path}» не существует.")

    # --- страницы ---------------------------------------------------------------
    def _page_index(self) -> None:
        # ?status=new|in_progress|fixed — фильтр таблицы; чужое значение
        # игнорируем молча (ссылка руками), показываем все отчёты.
        qs = parse_qs(urlsplit(self.path).query)
        status = (qs.get("status") or [None])[0]
        if status not in db.REPORT_STATUSES:
            status = None
        with closing(self.app.open_db()) as conn:
            stats = db.stats(conn)
            rows = db.list_reports(conn, limit=20, status=status)
        reports = []
        for r in rows:
            d = dict(r)
            d["thumb_url"] = self.app.media_url(r["overlay_path"])
            reports.append(d)
        self._render("index.html", stats=stats, reports=reports,
                     status_filter=status,
                     new_notifications=stats["new_notifications"])

    def _page_report(self, report_id: int) -> None:
        with closing(self.app.open_db()) as conn:
            row = db.get_report(conn, report_id)
            new_count = db.stats(conn)["new_notifications"]
        if row is None:
            return self._message(404, "Отчёт не найден",
                                 f"Отчёта №{report_id} нет в базе.")
        report = json.loads(row["json"])
        defects = []
        for d in report.get("defects") or []:
            severity = d.get("severity") or {}
            defects.append({
                "cls": d.get("class"),
                "confidence": d.get("confidence") or 0.0,
                "non_conforming": severity.get("non_conforming"),
                "severity_note": severity.get("note"),
                "size_text": templates_mod.size_text(d),
                "depth_text": templates_mod.depth_text(d.get("metric") or {}),
            })
        self._render(
            "report.html", r=dict(row), defects=defects,
            warnings=report.get("warnings") or [],
            scale_available=(report.get("scale") or {}).get("available", False),
            # camera_height/mixed: сантиметры есть, но это ОЦЕНКА — баннер
            # обязан говорить это, а не «сантиметры не выдаются» (ревью 2026-07-16)
            camera_scale_used=any(
                ((d.get("metric") or {}).get("scale_source")
                 in ("camera_height", "mixed"))
                for d in (report.get("defects") or [])),
            overlay_url=self.app.media_url(row["overlay_path"]),
            new_notifications=new_count)

    def _page_map(self) -> None:
        with closing(self.app.open_db()) as conn:
            points = db.map_points(conn)
            new_count = db.stats(conn)["new_notifications"]
        for p in points:
            p["status_label"] = templates_mod.status_label(p.get("status"))
        if points:
            # Инлайн-JSON в <script>: экранируем «<», чтобы имя файла вида
            # </script> не могло разорвать страницу.
            esc = lambda obj: json.dumps(obj, ensure_ascii=False).replace("<", "\\u003c")  # noqa: E731
            present = {p["class"] for p in points if p["class"]}
            # порядок легенды = фиксированный порядок палитры, не частота
            classes_present = [c for c in templates_mod.CLASS_COLORS if c in present]
            classes_present += sorted(present - set(classes_present))
            self._render("map.html",
                         points_json=esc(points),
                         colors_json=esc(templates_mod.CLASS_COLORS),
                         labels_json=esc(notify.CLASS_LABELS_RU),
                         classes_present=classes_present,
                         new_notifications=new_count)
        else:
            self._render("map.html", points_json=None, classes_present=[],
                         new_notifications=new_count)

    def _page_notifications(self) -> None:
        with closing(self.app.open_db()) as conn:
            items = [dict(r) for r in conn.execute(
                "SELECT * FROM notifications"
                " ORDER BY (status = 'new') DESC, id DESC").fetchall()]
            new_count = sum(1 for n in items if n["status"] == "new")
        self._render("notifications.html", items=items,
                     new_notifications=new_count)

    def _handle_notification_done(self, notification_id: int) -> None:
        with closing(self.app.open_db()) as conn:
            conn.execute("UPDATE notifications SET status = 'done' WHERE id = ?",
                         (notification_id,))
            conn.commit()
        self._redirect("/notifications")

    # --- жизненный цикл отчёта ---------------------------------------------------
    def _read_form(self) -> dict[str, str]:
        """Тело application/x-www-form-urlencoded → {поле: первое значение}."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if 0 < length <= 65536 else b""
        pairs = parse_qs(raw.decode("utf-8", "replace"))
        return {k: v[0] for k, v in pairs.items() if v}

    def _handle_report_status(self, report_id: int) -> None:
        status = self._read_form().get("status", "")
        try:
            with closing(self.app.open_db()) as conn:
                found = db.set_report_status(conn, report_id, status)
        except ValueError:
            return self._message(400, "Недопустимый статус",
                                 f"Статуса «{status}» не существует. Допустимы: "
                                 f"{', '.join(db.REPORT_STATUSES)}.")
        if not found:
            return self._message(404, "Отчёт не найден",
                                 f"Отчёта №{report_id} нет в базе.")
        self._redirect(f"/report/{report_id}")

    def _handle_report_delete(self, report_id: int) -> None:
        """Удалить отчёт из учёта (БД). Файлы на диске не трогаем — удаление
        обратимо повторным импортом JSON."""
        with closing(self.app.open_db()) as conn:
            found = db.delete_report(conn, report_id)
        if not found:
            return self._message(404, "Отчёт не найден",
                                 f"Отчёта №{report_id} нет в базе.")
        self._redirect("/")

    def _serve_export(self) -> None:
        """Реестр дефектов в CSV: передать ответственному за участок.
        utf-8-sig — чтобы Excel на Windows открывал кириллицу без вопросов;
        None остаются пустыми клетками (нет источника масштаба — пустая
        клетка). масштаб_источник/уверенность_размеров отличают эталонный
        замер (reference) от оценки по высоте камеры (camera_height/mixed,
        low — не для актирования); без них оценки в реестре были неотличимы
        от замеров (ревью 2026-07-16)."""
        with closing(self.app.open_db()) as conn:
            rows = db.export_rows(conn)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "отчёт_id", "файл", "stem", "добавлен_utc", "статус", "статус_рус",
            "широта", "долгота", "источник_gps", "дефект_№", "класс",
            "класс_рус", "уверенность", "длина_см", "ширина_см", "площадь_м2",
            "масштаб_источник", "уверенность_размеров",
            "бакет_глубины", "глубина_оценка_см", "глубина_оценка_потолок_см",
            "глубина_оценка_низ_см", "глубина_оценка_верх_см", "метод_оценки",
            "несоответствие_гост", "примечание_гост"])
        for r in rows:
            # Верхняя граница (метод «…|upper_bound») — потолок различимости,
            # а не точечная оценка: в «точечной» колонке ей не место (ревью
            # 2026-07-15) — иначе сортировка по колонке читает «< 2 см» как «2 см».
            is_ceiling = "upper_bound" in (r["depth_est_method"] or "")
            writer.writerow([_csv_safe(v) for v in (
                r["report_id"], r["image"], r["stem"], r["created_utc"],
                r["status"], templates_mod.status_label(r["status"]),
                r["lat"], r["lon"], r["gps_source"], r["defect_index"],
                r["class"], notify.CLASS_LABELS_RU.get(r["class"] or "",
                                                       r["class"] or ""),
                r["confidence"], r["length_cm"], r["width_cm"], r["area_m2"],
                r["scale_source"], r["metric_confidence"],
                r["depth_bucket"],
                None if is_ceiling else r["depth_est_cm"],
                r["depth_est_cm"] if is_ceiling else None,
                r["depth_est_low_cm"], r["depth_est_high_cm"],
                r["depth_est_method"], r["non_conforming"],
                r["severity_note"])])
        body = buf.getvalue().encode("utf-8-sig")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition",
                         'attachment; filename="defects.csv"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --- загрузка и анализ ------------------------------------------------------
    def _handle_upload(self) -> None:
        part, err = self._read_multipart_file()
        if err is not None:
            return self._render("upload.html", status=400, error=err)
        filename, payload = part

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return self._render(
                "upload.html", status=400,
                error=f"Расширение «{suffix or 'без расширения'}» не поддерживается. "
                      f"Допустимы: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")

        pipeline = self.app.get_pipeline()
        if pipeline is None:
            return self._message(
                503, "Анализ недоступен",
                "Сервер запущен без CV-моделей (--no-model), анализ фото "
                "невозможен. Перезапустите сервер без этого флага.")

        upload_path, stem = self._save_upload(filename, payload)

        try:
            # Модели не потокобезопасны — анализ строго по одному фото за раз.
            with self.app.analysis_lock:
                report, overlay, _masks = pipeline.analyze_image(upload_path)
        except Exception:
            traceback.print_exc()
            return self._message(
                500, "Анализ не выполнен",
                f"Движок не смог обработать «{filename}» (битый файл или сбой "
                "моделей). Подробности — в консоли сервера.")

        from .. import imgio
        from .. import report as report_mod

        report_mod.save_report(report, self.app.outputs_dir, stem)
        overlay_rel = None
        if overlay is not None:
            overlay_name = f"{stem}_annotated.jpg"
            if imgio.write_image(self.app.outputs_dir / overlay_name, overlay):
                overlay_rel = f"outputs/{overlay_name}"
            else:
                print(f"[ошибка] не записался overlay {overlay_name}", file=sys.stderr)

        # GPS: location отчёта (если движок/оператор его дал) → EXIF снимка → нет.
        lat = lon = gps_source = None
        loc = report.get("location") or {}
        if loc.get("available"):
            pair = geo.parse_latlon(f"{loc.get('lat')} {loc.get('lon')}")
            if pair is not None:
                lat, lon = pair
                gps_source = str(loc.get("source") or "report_location")
        if lat is None:
            pair = geo.exif_gps(upload_path)
            if pair is not None:
                lat, lon = pair
                gps_source = "exif"

        with closing(self.app.open_db()) as conn:
            report_id = db.import_report(
                conn, report, stem,
                source_path=f"uploads/{upload_path.name}",
                overlay_path=overlay_rel,
                lat=lat, lon=lon, gps_source=gps_source)
            notify.create_notifications(conn)
        self._redirect(f"/report/{report_id}")

    def _read_multipart_file(self):
        """Достать (имя файла, байты) из multipart/form-data без модуля cgi
        (он deprecated): приклеиваем заголовок Content-Type к телу и разбираем
        стандартным email-парсером. Возвращает ((name, bytes), None) или
        (None, текст ошибки)."""
        ctype = self.headers.get("Content-Type", "")
        if not ctype.lower().startswith("multipart/form-data"):
            return None, "Ожидалась форма multipart/form-data."
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return None, "Пустое тело запроса."
        if length > MAX_UPLOAD_BYTES:
            return None, (f"Файл больше лимита "
                          f"{MAX_UPLOAD_BYTES // (1024 * 1024)} МБ.")
        body = self.rfile.read(length)
        msg = email.message_from_bytes(
            b"Content-Type: " + ctype.encode("latin-1")
            + b"\r\nMIME-Version: 1.0\r\n\r\n" + body,
            policy=email.policy.default)
        if not msg.is_multipart():
            return None, "Не удалось разобрать multipart-форму."
        for part in msg.iter_parts():
            filename = part.get_filename()
            if not filename:
                continue
            payload = part.get_payload(decode=True)
            if payload:
                return (filename, payload), None
        return None, "В форме нет файла (поле photo пустое)."

    def _save_upload(self, filename: str, payload: bytes) -> tuple[Path, str]:
        """Сохранить файл под безопасным уникальным именем; вернуть (путь, stem).

        Имя чистится до [букв/цифр/._-] (кириллица легальна — imgio её умеет),
        уникальность против uploads/ И БД: одинаковые имена не должны молча
        перезаписывать чужие отчёты (upsert по stem — только осознанный)."""
        base = re.sub(r"[^\w.\-]", "_", Path(filename).stem) or "photo"
        suffix = Path(filename).suffix.lower()
        with closing(self.app.open_db()) as conn:
            taken = {row["stem"] for row in
                     conn.execute("SELECT stem FROM reports").fetchall()}
        stem, n = base, 2
        while stem in taken or (self.app.uploads_dir / f"{stem}{suffix}").exists():
            stem = f"{base}_{n}"
            n += 1
        path = self.app.uploads_dir / f"{stem}{suffix}"
        path.write_bytes(payload)
        return path, stem

    # --- статика ------------------------------------------------------------------
    def _serve_media(self, rel: str) -> None:
        """Отдать файл ТОЛЬКО из data_dir. Защита от traversal: resolve() +
        is_relative_to — «../», «..%2f» и абсолютные пути получают 403."""
        data_dir = self.app.data_dir.resolve()
        try:
            target = (data_dir / rel).resolve()
        except (OSError, ValueError):
            return self._message(403, "Доступ запрещён", "Недопустимый путь.")
        if not target.is_relative_to(data_dir):
            return self._message(403, "Доступ запрещён",
                                 "Путь выходит за пределы папки данных.")
        if not target.is_file():
            return self._message(404, "Файл не найден",
                                 "Такого файла в папке данных нет.")
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(argv=None) -> None:
    # На Windows перенаправленный stdout кодируется cp1251 — русские сообщения
    # не должны ронять print (тот же приём, что в cli.py).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 — нестандартный поток (тесты, embed)
            pass
    parser = argparse.ArgumentParser(
        prog="python -m road_defect.platform.webapp",
        description="Веб-интерфейс платформы учёта дорожных дефектов "
                    "(загрузка фото → анализ → отчёт/карта/уведомления).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data", default="platform_data",
                        help="папка данных (uploads/, outputs/, platform.db)")
    parser.add_argument("--import-outputs", metavar="ПАПКА",
                        help="одноразово импортировать готовые JSON-отчёты "
                             "движка в БД и выйти (сервер не поднимается)")
    parser.add_argument("--journal", default="datasets/journal.csv",
                        help="журнал ям для GPS по pair_id (CSV)")
    parser.add_argument("--no-model", action="store_true",
                        help="поднять интерфейс без CV-моделей "
                             "(загрузка фото будет честно отвечать 503)")
    args = parser.parse_args(argv)

    factory = None if args.no_model else _default_pipeline_factory
    app = PlatformApp(Path(args.data), pipeline_factory=factory)

    if args.import_outputs:
        journal = Path(args.journal)
        with closing(app.open_db()) as conn:
            n = db.import_reports(conn, Path(args.import_outputs),
                                  journal_csv=journal if journal.is_file() else None)
            created = notify.create_notifications(conn)
        print(f"Импортировано отчётов: {n}; новых уведомлений: {created}. "
              f"БД: {app.db_path}")
        return

    server = app.make_server(args.host, args.port)
    print(f"Открой http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
