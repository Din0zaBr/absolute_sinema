"""Форма для ручного заполнения замеров журнала (DATASET.md §4) по кадрам gt_photos.

Собирает ОДНУ автономную HTML-страницу: на каждую яму — её кадр(ы) с рулеткой
крупно (по ним читается замер) + миниатюра front для контекста + поля
length/width/depth/reference_in_frame/type/weather/notes, предзаполненные из
текущего журнала. Кнопка «Скачать journal.csv» собирает заполненный журнал с
тем же порядком колонок и utf-8-sig (Excel откроет корректно).

Замер рулеткой руками этим не заменить — но форма убирает ручное открытие
десятков фото и правку CSV вслепую: смотришь линейку, вводишь число, скачиваешь
готовый журнал и заменяешь им datasets\\journal.csv.

Картинки ужимаются (рулетки до --ruler-px, миниатюры front до --thumb-px) и
встраиваются base64 — страница переносимая и работает офлайн.

Запуск:
  .\\.venv\\Scripts\\python.exe scripts\\build_measure_form.py
  # открыть outputs_measure\\measure_form.html, заполнить, «Скачать journal.csv»,
  # заменить им datasets\\journal.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

from PIL import Image

# Порядок колонок журнала — ДОЛЖЕН совпадать с ingest_local_pairs.JOURNAL_FIELDS.
FIELDS = [
    "id", "scene", "date", "street_or_gps", "type", "length_cm", "width_cm",
    "depth_cm", "reference_in_frame", "weather", "notes",
]
FIXED_FIELDS = ["id", "scene", "date", "street_or_gps"]  # не редактируются в форме
TYPE_OPTIONS = ["pothole", "long_crack", "trans_crack", "alligator", "patch"]
REF_OPTIONS = ["", "none", "manhole", "marking", "curb"]


def _data_uri(path: Path, max_px: int, quality: int) -> str | None:
    """JPEG-миниатюра пути как data: URI, либо None если файл не читается."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_px, max_px))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
    except Exception:  # noqa: BLE001 - нечитаемый кадр не должен ронять всю форму
        return None
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _load_journal(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f) if row.get("id")]


def _select(field: str, value: str, options: list[str]) -> str:
    value = value or ""
    opts_list = list(options)
    if value not in opts_list:
        # значение журнала вне предопределённого списка (напр. type='crack')
        # НЕЛЬЗЯ терять: без selected dropdown отдал бы первую опцию, и выгрузка
        # затёрла бы поле. Добавляем его первым и помечаем (ревью 2026-07-05).
        opts_list = [value] + opts_list
    opts = []
    for o in opts_list:
        sel = " selected" if o == value else ""
        label = o or "—"
        opts.append(f'<option value="{html.escape(o)}"{sel}>{html.escape(label)}</option>')
    return (f'<select data-field="{field}">' + "".join(opts) + "</select>")


def _pid_forms(pid: str) -> list[str]:
    """id и его zero-padded вариант — журнал ingest'а даёт '001', «сырой» прогон
    '1'; кадры gt_photos/пары названы zero-padded. Ищем по обоим (ревью 2026-07-05)."""
    forms = [pid]
    if pid.zfill(3) != pid:
        forms.append(pid.zfill(3))
    return forms


def _rel_href(path: Path, base: Path) -> str:
    """Относительная ссылка path от base; '' если пути на разных дисках Windows
    (os.path.relpath там кидает ValueError — оригинал остаётся без ссылки)."""
    try:
        return quote(os.path.relpath(path, base).replace(os.sep, "/"))
    except ValueError:
        return ""


def _text(field: str, value: str, placeholder: str = "", inputmode: str = "") -> str:
    im = f' inputmode="{inputmode}"' if inputmode else ""
    return (f'<input data-field="{field}" type="text" value="{html.escape(value or "")}"'
            f' placeholder="{html.escape(placeholder)}"{im}>')


def _pit_card(pit: dict) -> str:
    fixed = {k: pit.get(k, "") for k in FIXED_FIELDS}
    fixed_attr = html.escape(json.dumps(fixed, ensure_ascii=False), quote=True)
    scene = pit.get("scene") or pit["id"]
    scene_badge = ("" if scene == pit["id"]
                   else f'<span class="scene">сцена {html.escape(scene)} (общий кадр)</span>')
    loc = pit.get("street_or_gps") or "—"
    date = pit.get("date") or "—"

    rulers = "".join(
        (f'<a href="{html.escape(r["href"])}" target="_blank" title="открыть оригинал">'
         f'<img class="ruler" src="{r["uri"]}" loading="lazy" alt="рулетка {i}"></a>'
         if r.get("href") else
         f'<img class="ruler" src="{r["uri"]}" loading="lazy" alt="рулетка {i}">')
        for i, r in enumerate(pit.get("rulers") or [], 1))
    if not rulers:
        rulers = '<div class="noimg">нет кадров с рулеткой</div>'
    thumb = (f'<img class="thumb" src="{pit["front_thumb"]}" loading="lazy" alt="front">'
             if pit.get("front_thumb") else '<div class="noimg thumb">нет front</div>')

    fields = (
        '<label>длина, см' + _text("length_cm", pit.get("length_cm", ""), "напр. 42", "decimal") + '</label>'
        '<label>ширина, см' + _text("width_cm", pit.get("width_cm", ""), "напр. 31", "decimal") + '</label>'
        '<label>глубина, см' + _text("depth_cm", pit.get("depth_cm", ""), "рейка+рулетка", "decimal") + '</label>'
        '<label>эталон в кадре' + _select("reference_in_frame", pit.get("reference_in_frame", ""), REF_OPTIONS) + '</label>'
        '<label>тип' + _select("type", pit.get("type", "pothole"), TYPE_OPTIONS) + '</label>'
        '<label>погода' + _text("weather", pit.get("weather", ""), "sunny / wet / …") + '</label>'
        '<label class="wide">заметки' + _text("notes", pit.get("notes", "")) + '</label>'
    )

    return (
        f'<section class="pit" data-fixed="{fixed_attr}">'
        f'<header><h2>Яма {html.escape(pit["id"])}</h2>{scene_badge}'
        f'<span class="meta">{html.escape(date)} · {html.escape(loc)}</span></header>'
        f'<div class="body"><div class="imgs">{rulers}{thumb}</div>'
        f'<div class="form">{fields}</div></div>'
        f'</section>'
    )


def build_html(pits: list[dict], generated_note: str = "") -> str:
    cards = "\n".join(_pit_card(p) for p in pits)
    fields_js = json.dumps(FIELDS)
    note = html.escape(generated_note)
    # CSS/JS инлайном — страница автономна и работает по file://.
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Замеры журнала — {len(pits)} ям</title>
<style>
  :root {{ color-scheme: light dark; --bg:#fff; --fg:#1a1a1a; --card:#f6f6f7;
           --line:#dcdce0; --accent:#1b7f4b; --muted:#6b6b70; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#16171a; --fg:#e8e8ea; --card:#212227; --line:#34353b;
             --accent:#43c07d; --muted:#9a9aa2; }} }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:15px/1.45 system-ui,Segoe UI,Arial,sans-serif;
          background:var(--bg); color:var(--fg); }}
  .topbar {{ position:sticky; top:0; z-index:5; display:flex; gap:16px;
             align-items:center; padding:12px 18px; background:var(--bg);
             border-bottom:1px solid var(--line); }}
  .topbar h1 {{ font-size:16px; margin:0; }}
  #progress {{ color:var(--muted); }}
  button {{ font:inherit; padding:8px 16px; border:0; border-radius:8px;
            background:var(--accent); color:#fff; cursor:pointer; }}
  .hint {{ color:var(--muted); font-size:13px; margin-left:auto; max-width:46ch; }}
  main {{ padding:18px; max-width:1200px; margin:0 auto; }}
  .pit {{ border:1px solid var(--line); border-radius:12px; background:var(--card);
          margin-bottom:18px; overflow:hidden; }}
  .pit.done {{ border-color:var(--accent); }}
  header {{ display:flex; gap:12px; align-items:baseline; flex-wrap:wrap;
            padding:10px 14px; border-bottom:1px solid var(--line); }}
  header h2 {{ margin:0; font-size:16px; }}
  .scene {{ color:var(--accent); font-size:12px; }}
  .meta {{ color:var(--muted); font-size:13px; margin-left:auto; }}
  .body {{ display:grid; grid-template-columns:minmax(0,1.4fr) minmax(0,1fr); gap:14px; padding:14px; }}
  @media (max-width:820px) {{ .body {{ grid-template-columns:1fr; }} }}
  .imgs {{ display:flex; flex-wrap:wrap; gap:8px; align-content:start; }}
  .ruler {{ width:100%; max-width:100%; border-radius:8px; border:1px solid var(--line); }}
  .thumb {{ width:160px; height:auto; border-radius:8px; border:1px solid var(--line); }}
  .noimg {{ color:var(--muted); font-size:13px; padding:10px; border:1px dashed var(--line); border-radius:8px; }}
  .form {{ display:grid; grid-template-columns:1fr 1fr; gap:10px 12px; align-content:start; }}
  .form label {{ display:flex; flex-direction:column; gap:4px; font-size:13px; color:var(--muted); }}
  .form label.wide {{ grid-column:1 / -1; }}
  .form input, .form select {{ font:inherit; padding:7px 9px; border:1px solid var(--line);
                               border-radius:7px; background:var(--bg); color:var(--fg); }}
</style></head><body>
<div class="topbar">
  <h1>Замеры журнала</h1>
  <span id="progress">Заполнено: 0 / {len(pits)}</span>
  <button id="dl">Скачать journal.csv</button>
  <span class="hint">Смотри линейку → впиши см → «Скачать journal.csv» → замени им datasets\\journal.csv. {note}</span>
</div>
<main>
{cards}
</main>
<script>
const FIELDS = {fields_js};
function csvCell(v) {{ v = (v==null?'':String(v)); return /[",\\n\\r]/.test(v) ? '"'+v.replace(/"/g,'""')+'"' : v; }}
function buildCsv() {{
  const lines = [FIELDS.join(',')];
  document.querySelectorAll('.pit').forEach(p => {{
    const fixed = JSON.parse(p.dataset.fixed);
    const row = FIELDS.map(f => {{
      const inp = p.querySelector('[data-field="'+f+'"]');
      return csvCell(inp ? inp.value : (fixed[f] !== undefined ? fixed[f] : ''));
    }});
    lines.push(row.join(','));
  }});
  return '\\ufeff' + lines.join('\\r\\n') + '\\r\\n';
}}
document.getElementById('dl').addEventListener('click', () => {{
  const blob = new Blob([buildCsv()], {{ type: 'text/csv;charset=utf-8' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'journal.csv'; a.click();
  URL.revokeObjectURL(a.href);
}});
function updateProgress() {{
  let done = 0; const pits = document.querySelectorAll('.pit');
  pits.forEach(p => {{
    const has = ['length_cm','width_cm','depth_cm'].some(f => {{
      const i = p.querySelector('[data-field="'+f+'"]'); return i && i.value.trim(); }});
    if (has) {{ done++; p.classList.add('done'); }} else p.classList.remove('done');
  }});
  document.getElementById('progress').textContent = 'Заполнено: ' + done + ' / ' + pits.length;
}}
document.addEventListener('input', updateProgress);
window.addEventListener('load', updateProgress);
</script>
</body></html>
"""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--journal", default="datasets/journal.csv")
    ap.add_argument("--gt", default="datasets/gt_photos", help="папка кадров с рулеткой")
    ap.add_argument("--pairs", default="datasets/local_pairs", help="папка пар (для миниатюр front)")
    ap.add_argument("--out", default="outputs_measure/measure_form.html")
    ap.add_argument("--ruler-px", type=int, default=1200, help="макс. сторона кадра рулетки")
    ap.add_argument("--thumb-px", type=int, default=360, help="макс. сторона миниатюры front")
    args = ap.parse_args()

    rows = _load_journal(Path(args.journal))
    if not rows:
        print(f"Журнал пуст или не найден: {args.journal} — сначала разложи фото ingest'ом.")
        return 1

    gt_dir, pairs_dir = Path(args.gt), Path(args.pairs)
    out = Path(args.out)
    out_parent = out.parent
    pits, n_rulers = [], 0
    for row in rows:
        pid = row["id"]
        ruler_files: list[Path] = []
        seen: set[Path] = set()
        for form in _pid_forms(pid):
            for r in sorted(gt_dir.glob(f"{form}_ruler_*.jpg")):
                if r not in seen:
                    seen.add(r)
                    ruler_files.append(r)
        rulers = []
        for r in ruler_files:
            uri = _data_uri(r, args.ruler_px, 80)
            if uri:
                # относительная ссылка на ОРИГИНАЛ для точного чтения по клику
                # (встраивание ужато; оригинал не дублируется в файл формы)
                rulers.append({"uri": uri, "href": _rel_href(r, out_parent)})
        n_rulers += len(rulers)
        # у ямы-дубля пары в local_pairs нет (ingest дедуплицирует по сценам,
        # 2026-07-16) — миниатюра берётся из папки сцены-представителя
        thumb_ids = list(_pid_forms(pid)) + list(_pid_forms(row.get("scene") or pid))
        front = next((pairs_dir / f / "front.jpg" for f in thumb_ids
                      if (pairs_dir / f / "front.jpg").is_file()), None)
        thumb = _data_uri(front, args.thumb_px, 75) if front else None
        pits.append({**row, "rulers": rulers, "front_thumb": thumb})

    out.parent.mkdir(parents=True, exist_ok=True)
    note = f"Ям: {len(pits)}, кадров рулеток: {n_rulers}."
    out.write_text(build_html(pits, note), encoding="utf-8")

    size_mb = out.stat().st_size / 1e6
    print(f"Форма: {out}  ({size_mb:.1f} МБ, ям {len(pits)}, рулеток {n_rulers})")
    print("Открой в браузере, заполни замеры, нажми «Скачать journal.csv» и "
          "замени им datasets/journal.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
