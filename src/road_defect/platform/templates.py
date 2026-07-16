"""Jinja2-шаблоны веб-интерфейса (подсистема 3) + презентационные хелперы.

Всё автономно: inline-CSS, светлая/тёмная тема через prefers-color-scheme,
никаких внешних ресурсов, кроме Leaflet на странице карты (тайлам карты нужен
интернет — об этом честно написано на самой странице).

Палитра классов — валидированная категориальная палитра (dataviz-скилл,
проверена скриптом validate_palette.js для обеих тем: CVD-разнесение и
контраст в норме). Назначение цветов КЛАССАМ фиксированное (цвет следует за
сущностью, не за позицией в легенде); рядом с цветной точкой всегда есть
текстовая подпись — идентичность никогда не кодируется одним цветом.
"""
from __future__ import annotations

from .notify import CLASS_LABELS_RU

# Фиксированное назначение: класс → слот валидированной палитры (light, dark).
CLASS_COLORS = {
    "pothole":            ("#2a78d6", "#3987e5"),   # слот 1, синий
    "patch":              ("#1baf7a", "#199e70"),   # слот 2, аква
    "alligator_crack":    ("#eda100", "#c98500"),   # слот 3, жёлтый
    "longitudinal_crack": ("#008300", "#008300"),   # слот 4, зелёный
    "transverse_crack":   ("#4a3aa7", "#9085e9"),   # слот 5, фиолетовый
}
_FALLBACK_COLOR = ("#898781", "#898781")            # неизвестный класс — нейтральный

# Статусный цвет «critical» (зарезервирован, не переиспользуется под классы).
CRITICAL_COLOR = ("#d03b3b", "#d03b3b")

# Жизненный цикл учёта (db.REPORT_STATUSES) — русские подписи и стиль бейджа.
STATUS_LABELS_RU = {"new": "новый", "in_progress": "в работе", "fixed": "устранён"}
STATUS_BADGE_CLASS = {"new": "", "in_progress": "progress", "fixed": "good"}


def class_label(cls: str | None) -> str:
    return CLASS_LABELS_RU.get(cls or "", cls or "дефект")


def status_label(status: str | None) -> str:
    return STATUS_LABELS_RU.get(status or "", status or "—")


def class_color(cls: str | None, dark: bool = False) -> str:
    return CLASS_COLORS.get(cls or "", _FALLBACK_COLOR)[1 if dark else 0]


def depth_text(metric: dict) -> str:
    """Честная строка про глубину дефекта.

    Приоритет: оценка в см (двухвидовая и т.п., ВСЕГДА со словом «оценка») →
    относительный бакет → «не определена». metric.depth_cm не показывается
    никогда: сертифицированная глубина в этой версии невозможна и всегда null.
    Терпит отсутствие ключа depth_cm_estimate (старые отчёты) и None.
    """
    est = metric.get("depth_cm_estimate")
    est = est if isinstance(est, dict) else None
    bucket = metric.get("depth_bucket")

    if est and est.get("available"):
        if est.get("point_cm") is not None:
            low, high = est.get("low_cm"), est.get("high_cm")
            if low is not None and high is not None:
                return (f"глубина ≈ {est['point_cm']:.1f} см "
                        f"({low:.1f}–{high:.1f}, оценка)")
            return f"глубина ≈ {est['point_cm']:.1f} см (оценка)"
        if est.get("upper_bound_cm") is not None:
            return f"глубина < {est['upper_bound_cm']:g} см (оценка)"

    if bucket:
        reason = (est or {}).get("reason")
        suffix = f"см недоступны: {reason}" if reason else "см недоступны"
        return f"глубина: бакет {bucket} ({suffix})"
    return "глубина не определена"


def size_text(defect: dict) -> str:
    """Размеры дефекта: сантиметры от эталона — как есть; от высоты камеры
    (scale_source camera_height/mixed) — с явной пометкой «оценка» и полосой
    (ревью 2026-07-16: без пометки оценка читалась как замер); иначе честные
    пиксели с оговоркой."""
    metric = defect.get("metric") or {}
    shape = defect.get("shape") or {}
    if metric.get("available"):
        parts = []
        if metric.get("length_cm") is not None:
            parts.append(f"длина {metric['length_cm']:.1f} см")
        if metric.get("width_cm") is not None:
            parts.append(f"ширина {metric['width_cm']:.1f} см")
        if metric.get("area_m2") is not None:
            parts.append(f"площадь {metric['area_m2']:.3f} м²")
        if parts:
            text = ", ".join(parts)
            if metric.get("scale_source") in ("camera_height", "mixed"):
                band = metric.get("error_band_pct")
                suffix = (f", ±{band:.0f}%" if isinstance(band, (int, float))
                          and not isinstance(band, bool) else "")
                text += (f" (ОЦЕНКА от высоты камеры{suffix} — размер маски, "
                         "не для актирования)")
            return text
    length_px = shape.get("length_px")
    width_px = shape.get("width_px")
    if length_px is not None and width_px is not None:
        return (f"{length_px:.0f} × {width_px:.0f} px "
                "(нет эталона масштаба, размеры в px)")
    return "размеры недоступны (нет эталона масштаба)"


def build_env():
    """Собрать Jinja2-окружение. autoescape по имени шаблона (*.html)."""
    from jinja2 import DictLoader, Environment, select_autoescape

    env = Environment(loader=DictLoader(TEMPLATES),
                      autoescape=select_autoescape())
    # class_colors нужен базовому шаблону (CSS-точки классов) на каждой странице.
    env.globals.update(class_label=class_label, class_color=class_color,
                       class_colors=CLASS_COLORS, status_label=status_label,
                       status_labels=STATUS_LABELS_RU,
                       status_badge_class=STATUS_BADGE_CLASS)
    return env


# --- Шаблоны -----------------------------------------------------------------

_BASE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{% block title %}Дорожные дефекты{% endblock %} — платформа</title>
{% block head %}{% endblock %}
<style>
:root {
  color-scheme: light dark;
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --hairline: #e1e0d9; --border: rgba(11,11,11,0.10);
  --critical: #d03b3b; --good: #0ca30c;
  --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --hairline: #2c2c2a; --border: rgba(255,255,255,0.10);
    --accent: #3987e5;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
header { background: var(--surface); border-bottom: 1px solid var(--hairline);
         padding: 10px 20px; display: flex; gap: 18px; align-items: baseline;
         flex-wrap: wrap; }
header .brand { font-weight: 700; color: var(--ink); }
header nav { display: flex; gap: 14px; }
main { max-width: 1080px; margin: 0 auto; padding: 20px; }
h1 { font-size: 20px; margin: 6px 0 16px; }
h2 { font-size: 16px; margin: 20px 0 10px; }
.card { background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
         gap: 12px; margin-bottom: 18px; }
.tile { background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 12px 14px; }
.tile .label { font-size: 12px; color: var(--ink-2); }
.tile .value { font-size: 28px; font-weight: 700; color: var(--ink); }
.tile .note { font-size: 12px; color: var(--muted); }
.tile .flag { font-size: 12px; color: var(--critical); }
table { border-collapse: collapse; width: 100%; background: var(--surface); }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--hairline);
         vertical-align: middle; }
th { font-size: 12px; color: var(--ink-2); font-weight: 600; }
td.num { font-variant-numeric: tabular-nums; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
       margin-right: 6px; vertical-align: baseline; background: var(--dot, var(--muted)); }
{% for cls, pair in class_colors.items() %}
.dot-{{ cls }} { --dot: {{ pair[0] }}; }
{% endfor %}
@media (prefers-color-scheme: dark) {
{% for cls, pair in class_colors.items() %}
  .dot-{{ cls }} { --dot: {{ pair[1] }}; }
{% endfor %}
}
.thumb { height: 48px; border-radius: 4px; border: 1px solid var(--border); display: block; }
.muted { color: var(--muted); font-size: 13px; }
.warn { color: var(--critical); }
.badge { display: inline-block; font-size: 12px; padding: 1px 8px;
         border-radius: 10px; border: 1px solid var(--border); color: var(--ink-2); }
.badge.critical { color: var(--critical); border-color: var(--critical); }
.badge.good { color: var(--good); border-color: var(--good); }
.badge.progress { color: var(--accent); border-color: var(--accent); }
button.danger { color: var(--critical); border-color: var(--critical); background: transparent; }
.inline-forms { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.inline-forms form { margin: 0; }
button { font: inherit; padding: 5px 12px; border-radius: 6px; cursor: pointer;
         border: 1px solid var(--border); background: var(--surface); color: var(--ink); }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
input[type=file] { font: inherit; color: var(--ink-2); }
img.overlay { max-width: 100%; border-radius: 8px; border: 1px solid var(--border); }
ul.plain { padding-left: 18px; margin: 6px 0; }
</style>
</head>
<body>
<header>
  <span class="brand">Дорожные дефекты</span>
  <nav>
    <a href="/">Дашборд</a>
    <a href="/upload">Загрузить фото</a>
    <a href="/map">Карта</a>
    <a href="/notifications">Уведомления{% if new_notifications %} ({{ new_notifications }}){% endif %}</a>
  </nav>
</header>
<main>
{% block content %}{% endblock %}
<p class="muted">Принцип честности: система не показывает числа, которых физически
не может знать. Нет эталона — нет сертифицируемых сантиметров (оценки, например
от высоты камеры, — только с явной пометкой); глубина в см — только как «оценка».</p>
</main>
</body>
</html>
"""

_INDEX = """{% extends "base.html" %}
{% block title %}Дашборд{% endblock %}
{% block content %}
<h1>Дашборд</h1>
<div class="tiles">
  <div class="tile"><div class="label">Отчётов</div>
    <div class="value">{{ stats.reports }}</div></div>
  <div class="tile"><div class="label">Дефектов</div>
    <div class="value">{{ stats.defects }}</div></div>
  <div class="tile"><div class="label">Ям</div>
    <div class="value">{{ stats.potholes }}</div></div>
  <div class="tile"><div class="label">Несоответствий ГОСТ</div>
    <div class="value">{{ stats.non_conforming }}</div>
    {% if stats.non_conforming %}<div class="flag">&#9888; требуют устранения</div>
    {% else %}<div class="note">подтверждённых нет</div>{% endif %}</div>
  <div class="tile"><div class="label">Новых уведомлений</div>
    <div class="value">{{ stats.new_notifications }}</div></div>
  <div class="tile"><div class="label">Отчётов с координатами</div>
    <div class="value">{{ stats.with_coords }}</div>
    <div class="note"><a href="/map">на карте</a></div></div>
  <div class="tile"><div class="label">В работе</div>
    <div class="value">{{ stats.in_progress }}</div></div>
  <div class="tile"><div class="label">Устранено</div>
    <div class="value">{{ stats.fixed }}</div>
    <div class="note">отчётов со статусом «устранён»</div></div>
</div>

{% if stats.by_class %}
<div class="card">
  <h2 style="margin-top:0">Дефекты по классам</h2>
  {% for cls, n in stats.by_class.items() %}
    <span style="margin-right:16px; white-space:nowrap">
      <span class="dot dot-{{ cls }}"></span>{{ class_label(cls) }} — {{ n }}</span>
  {% endfor %}
</div>
{% endif %}

<h2>Последние отчёты</h2>
<p class="muted">Статус:
  {% if status_filter %}<a href="/">все</a>{% else %}<strong>все</strong>{% endif %}
  {% for st, label in status_labels.items() %}
  · {% if status_filter == st %}<strong>{{ label }}</strong>
    {% else %}<a href="/?status={{ st }}">{{ label }}</a>{% endif %}
  {% endfor %}
  &nbsp;·&nbsp; <a href="/export.csv">скачать реестр дефектов (CSV)</a></p>
{% if reports %}
<table>
  <tr><th></th><th>Файл</th><th>Режим</th><th>Дефектов</th><th>Статус</th><th>GPS</th><th>Добавлен (UTC)</th></tr>
  {% for r in reports %}
  <tr>
    <td>{% if r.thumb_url %}<a href="/report/{{ r.id }}"><img class="thumb"
        src="{{ r.thumb_url }}" alt="оверлей {{ r.stem }}"></a>{% else %}—{% endif %}</td>
    <td><a href="/report/{{ r.id }}">{{ r.image or r.stem }}</a></td>
    <td class="muted">{{ r.mode or "—" }}</td>
    <td class="num">{{ r.n_defects }}</td>
    <td><span class="badge {{ status_badge_class.get(r.status, '') }}">{{ status_label(r.status) }}</span></td>
    <td class="muted">{% if r.lat is not none %}{{ "%.5f"|format(r.lat) }},
      {{ "%.5f"|format(r.lon) }}{% else %}нет{% endif %}</td>
    <td class="muted">{{ r.created_utc }}</td>
  </tr>
  {% endfor %}
</table>
{% elif status_filter %}
<p class="muted">Отчётов со статусом «{{ status_label(status_filter) }}» нет.</p>
{% else %}
<p class="muted">Отчётов пока нет. <a href="/upload">Загрузите фото</a> или импортируйте
готовые JSON: <code>--import-outputs &lt;папка&gt;</code>.</p>
{% endif %}
{% endblock %}
"""

_UPLOAD = """{% extends "base.html" %}
{% block title %}Загрузка фото{% endblock %}
{% block content %}
<h1>Загрузка фото</h1>
<div class="card">
  <form method="post" action="/upload" enctype="multipart/form-data">
    <p><input type="file" name="photo" accept=".jpg,.jpeg,.png,.bmp,.webp" required></p>
    <p><button class="primary" type="submit">Проанализировать</button></p>
  </form>
  <p class="muted">Форматы: JPG, PNG, BMP, WebP. Анализ на CPU занимает от долей
  секунды до нескольких секунд; при первом запуске модели загружаются дольше.
  Координаты берутся из EXIF снимка, если они там есть.</p>
</div>
{% if error %}<div class="card"><p class="warn">{{ error }}</p></div>{% endif %}
{% endblock %}
"""

_REPORT = """{% extends "base.html" %}
{% block title %}Отчёт {{ r.stem }}{% endblock %}
{% block content %}
<h1>Отчёт: {{ r.image or r.stem }}</h1>
<p class="muted">Режим: {{ r.mode or "—" }} · Движок: {{ r.engine_version or "—" }}
 · Добавлен: {{ r.created_utc }} (UTC)
 {% if r.lat is not none %} · GPS: {{ "%.5f"|format(r.lat) }}, {{ "%.5f"|format(r.lon) }}
   ({{ r.gps_source or "источник неизвестен" }}) — <a href="/map">на карте</a>
 {% else %} · GPS: нет данных{% endif %}</p>

<div class="card inline-forms">
  <span>Статус устранения:
    <span class="badge {{ status_badge_class.get(r.status, '') }}">{{ status_label(r.status) }}</span></span>
  {% for st, label in status_labels.items() %}{% if st != r.status %}
  <form method="post" action="/report/{{ r.id }}/status">
    <input type="hidden" name="status" value="{{ st }}">
    <button type="submit">&rarr; {{ label }}</button>
  </form>
  {% endif %}{% endfor %}
  <form method="post" action="/report/{{ r.id }}/delete" style="margin-left:auto"
        onsubmit="return confirm('Удалить отчёт из учёта? Записи о дефектах и уведомления будут удалены; файлы на диске останутся.')">
    <button type="submit" class="danger">Удалить из учёта</button>
  </form>
</div>

{% if overlay_url %}
<p><img class="overlay" src="{{ overlay_url }}" alt="размеченное фото {{ r.stem }}"></p>
{% else %}<p class="muted">Размеченное изображение недоступно.</p>{% endif %}

{% if not scale_available %}
{% if camera_scale_used %}
<div class="card"><p class="warn" style="margin:0">Эталон масштаба в кадре не
подтверждён — размеры в сантиметрах получены ОЦЕНКОЙ от высоты камеры
(confidence=low, не для актирования); полоса ошибки указана у дефекта.</p></div>
{% else %}
<div class="card"><p class="warn" style="margin:0">Эталон масштаба в кадре не
подтверждён — сантиметры не выдаются, размеры указаны в пикселях.</p></div>
{% endif %}
{% endif %}

<h2>Дефекты ({{ defects|length }})</h2>
{% for d in defects %}
<div class="card">
  <p style="margin:0 0 6px"><span class="dot dot-{{ d.cls }}"></span>
    <strong>{{ class_label(d.cls) }}</strong>
    <span class="muted">({{ d.cls }}, уверенность {{ "%.2f"|format(d.confidence) }})</span>
    {% if d.non_conforming == "yes" %}<span class="badge critical">&#9888; не соответствует ГОСТ</span>
    {% elif d.non_conforming == "no" %}<span class="badge good">в пределах нормы</span>
    {% else %}<span class="badge">вердикт не определён</span>{% endif %}</p>
  <ul class="plain">
    <li>Размеры: {{ d.size_text }}</li>
    {# depth_text сам начинается со слова «глубина» — без префикса, иначе
       на странице выходило «Глубина: глубина: бакет …» (живая проверка). #}
    <li>{{ d.depth_text | capitalize }}</li>
    <li>Вердикт ГОСТ: {{ d.severity_note or "—" }}</li>
  </ul>
</div>
{% else %}
<p class="muted">Дефекты не обнаружены.</p>
{% endfor %}

{% if warnings %}
<h2>Предупреждения движка</h2>
<div class="card"><ul class="plain">
  {% for w in warnings %}<li class="muted">{{ w }}</li>{% endfor %}
</ul></div>
{% endif %}
{% endblock %}
"""

_MAP = """{% extends "base.html" %}
{% block title %}Карта{% endblock %}
{% block head %}
{% if points_json %}
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
{% endif %}
{% endblock %}
{% block content %}
<h1>Карта дефектов</h1>
{% if points_json %}
<div class="card" style="padding:0"><div id="map" style="height:520px; border-radius:8px"></div></div>
<p class="muted">Карта требует интернета для тайлов (OpenStreetMap), маркеры —
локальные, из БД платформы. Красная обводка — несоответствие ГОСТ.</p>
<p>
{% for cls in classes_present %}
  <span style="margin-right:16px; white-space:nowrap">
    <span class="dot dot-{{ cls }}"></span>{{ class_label(cls) }}</span>
{% endfor %}
</p>
<script>
const POINTS = {{ points_json | safe }};
const COLORS = {{ colors_json | safe }};
const DARK = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
const LABELS = {{ labels_json | safe }};
const map = L.map("map");
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  { maxZoom: 19, attribution: "&copy; OpenStreetMap" }).addTo(map);
const group = L.featureGroup();
for (const p of POINTS) {
  const pair = COLORS[p.class] || ["#898781", "#898781"];
  const m = L.circleMarker([p.lat, p.lon], {
    radius: 8, weight: 2, fillOpacity: 0.85,
    fillColor: pair[DARK ? 1 : 0],
    color: p.non_conforming === "yes" ? "#d03b3b" : "#fcfcfb",
  });
  const label = LABELS[p.class] || p.class;
  m.bindPopup(label +
    (p.non_conforming === "yes" ? " — <b>не соответствует ГОСТ</b>" : "") +
    (p.depth_bucket ? "<br>бакет глубины: " + p.depth_bucket : "") +
    (p.status_label ? "<br>статус: " + p.status_label : "") +
    '<br><a href="/report/' + p.report_id + '">открыть отчёт</a>');
  group.addLayer(m);
}
group.addTo(map);
map.fitBounds(group.getBounds().pad(0.2));
</script>
{% else %}
<div class="card"><p class="muted" style="margin:0">Точек с координатами пока нет —
ни у одного отчёта нет GPS (EXIF/журнал/location). Карта появится, когда появятся
координаты; выдумывать их система не будет.</p></div>
{% endif %}
{% endblock %}
"""

_NOTIFICATIONS = """{% extends "base.html" %}
{% block title %}Уведомления{% endblock %}
{% block content %}
<h1>Уведомления</h1>
{% if items %}
<table>
  <tr><th>Статус</th><th>Правило</th><th>Сообщение</th><th>Отчёт</th><th>Создано (UTC)</th><th></th></tr>
  {% for n in items %}
  <tr>
    <td>{% if n.status == "new" %}<span class="badge critical">новое</span>
        {% else %}<span class="badge good">обработано</span>{% endif %}</td>
    <td class="muted">{{ n.rule }}</td>
    <td>{{ n.message }}</td>
    <td><a href="/report/{{ n.report_id }}">#{{ n.report_id }}</a></td>
    <td class="muted">{{ n.created_utc }}</td>
    <td>{% if n.status == "new" %}
      <form method="post" action="/notifications/{{ n.id }}/done" style="margin:0">
        <button type="submit">Отметить обработанным</button>
      </form>{% endif %}</td>
  </tr>
  {% endfor %}
</table>
{% else %}
<p class="muted">Уведомлений нет. Они создаются автоматически по правилам:
несоответствие ГОСТ и подозрение на глубокую яму (по оценке, требуется выезд).</p>
{% endif %}
{% endblock %}
"""

_MESSAGE = """{% extends "base.html" %}
{% block title %}{{ title }}{% endblock %}
{% block content %}
<h1>{{ title }}</h1>
<div class="card"><p style="margin:0">{{ text }}</p></div>
<p><a href="/">&larr; на дашборд</a></p>
{% endblock %}
"""

TEMPLATES = {
    "base.html": _BASE,
    "index.html": _INDEX,
    "upload.html": _UPLOAD,
    "report.html": _REPORT,
    "map.html": _MAP,
    "notifications.html": _NOTIFICATIONS,
    "message.html": _MESSAGE,
}
