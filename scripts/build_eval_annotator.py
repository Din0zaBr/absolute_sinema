"""Разметчик eval-набора (P0): одна автономная HTML-страница с canvas-боксами.

Читает `datasets/eval_v1/manifest.csv` + кадры, встраивает их (ужатые, base64)
и собирает офлайн-страницу: рисование боксов дефектов и конфузоров, приём/
отклонение ПРЕДЛОЖЕНИЙ движка (--proposals: папка отчётов CLI по кадрам),
пометка кадра «размечен», автосохранение в localStorage и кнопка «Скачать
annotations.json» (схема — docs/EVAL.md, координаты в ПИКСЕЛЯХ ОРИГИНАЛА).

Честность: предложения движка — НЕ разметка. Они рисуются серым пунктиром и в
экспорт не попадают, пока человек их явно не принял (Enter/кнопка). В метрики
(scripts/eval_detection.py) идут только кадры, помеченные «размечен».

Запуск:
  .\\.venv\\Scripts\\python.exe scripts\\build_eval_annotator.py --proposals outputs_eval/reports
  # открыть outputs_eval/annotator.html, разметить, «Скачать annotations.json»,
  # положить файл в datasets/eval_v1/annotations.json
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_defect import config  # noqa: E402

DEFECT_LABELS = sorted(set(config.DEFECT_CLASSES.values()))
CONFUSOR_CATEGORIES = ["car", "off_road", "shadow_curb", "other"]


def _thumb(path: Path, max_px: int, quality: int):
    """(data URI, orig_w, orig_h, disp_w, disp_h) или None, если не читается."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            ow, oh = im.size
            im.thumbnail((max_px, max_px))
            dw, dh = im.size
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
    except Exception:  # noqa: BLE001 — нечитаемый кадр не роняет страницу
        return None
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return uri, ow, oh, dw, dh


def load_manifest(eval_dir: Path) -> list[dict]:
    mpath = eval_dir / "manifest.csv"
    if not mpath.is_file():
        raise FileNotFoundError(
            f"Нет {mpath} — сначала собери набор: scripts/make_eval_set.py")
    with mpath.open(encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f) if row.get("frame_id")]


def load_proposals(proposals_dir: Path | None, frame_ids: list[str]) -> dict:
    """frame_id -> [{label, conf, bbox}] из отчётов движка (bbox_px оригинала).

    Классы вне таксономии дефектов не предлагаем (COCO-фолбэк и пр.)."""
    out: dict[str, list] = {}
    if proposals_dir is None:
        return out
    for fid in frame_ids:
        rp = proposals_dir / f"{fid}.json"
        if not rp.is_file():
            continue
        try:
            rep = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — битый отчёт не роняет разметчик
            continue
        props = []
        for d in rep.get("defects", []):
            if d.get("class") in DEFECT_LABELS and d.get("bbox_px"):
                props.append({"label": d["class"],
                              "conf": round(float(d.get("confidence") or 0), 3),
                              "bbox": [float(v) for v in d["bbox_px"]]})
        if props:
            out[fid] = props
    return out


def build_page(frames: list[dict], initial: dict | None, proposals: dict) -> str:
    import hashlib

    # id набора = состав кадров: localStorage-черновики разных eval-наборов
    # на одной машине не должны перетирать друг друга (ревью 2026-07-07)
    set_id = hashlib.md5("|".join(f["id"] for f in frames).encode()).hexdigest()[:8]
    data = {"setId": set_id, "defectLabels": DEFECT_LABELS,
            "confusorCategories": CONFUSOR_CATEGORIES,
            "frames": frames, "initial": initial, "proposals": proposals}
    # </ внутри JSON не должен закрыть <script> раньше времени
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    n = len(frames)
    return """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка eval_v1 — """ + str(n) + """ кадров</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#1a1a1a; --card:#f6f6f7;
          --line:#dcdce0; --accent:#1b7f4b; --muted:#6b6b70; --warn:#b7791f; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16171a; --fg:#e8e8ea; --card:#212227; --line:#34353b;
            --accent:#43c07d; --muted:#9a9aa2; --warn:#e0a83a; } }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.4 system-ui,Segoe UI,Arial,sans-serif;
         background:var(--bg); color:var(--fg); display:flex; height:100vh; }
  #side { width:230px; overflow-y:auto; border-right:1px solid var(--line);
          padding:10px; flex-shrink:0; }
  #side h1 { font-size:15px; margin:2px 0 8px; }
  #side .fr { display:flex; gap:6px; align-items:center; padding:4px 6px;
              border-radius:6px; cursor:pointer; }
  #side .fr:hover { background:var(--card); }
  #side .fr.cur { background:var(--card); outline:1px solid var(--accent); }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--line); flex-shrink:0; }
  .dot.rev { background:var(--accent); } .dot.part { background:var(--warn); }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  #bar { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
         padding:8px 12px; border-bottom:1px solid var(--line); }
  #bar select,#bar button,#bar label { font:inherit; }
  #bar select { padding:5px 7px; border:1px solid var(--line); border-radius:6px;
                background:var(--bg); color:var(--fg); }
  button { padding:6px 12px; border:0; border-radius:7px; cursor:pointer;
           background:var(--accent); color:#fff; }
  button.sec { background:var(--card); color:var(--fg); border:1px solid var(--line); }
  #notice { color:var(--warn); font-size:12px; }
  #wrap { flex:1; overflow:auto; padding:10px; }
  canvas { border:1px solid var(--line); border-radius:6px; cursor:crosshair;
           max-width:none; }
  #foot { padding:6px 12px; border-top:1px solid var(--line); color:var(--muted);
          font-size:12.5px; display:flex; gap:14px; flex-wrap:wrap; }
  .rev-toggle { display:flex; gap:6px; align-items:center; font-weight:600; }
</style></head><body>
<div id="side"><h1>Кадры <span id="prog"></span></h1><div id="frames"></div></div>
<div id="main">
  <div id="bar">
    <button class="sec" id="prev">←</button><button class="sec" id="next">→</button>
    <b id="title"></b>
    <select id="mode"><option value="defect">дефект</option>
      <option value="confusor">конфузор</option></select>
    <select id="label"></select>
    <button id="accept" style="display:none">Принять предложение (Enter)</button>
    <label class="rev-toggle"><input type="checkbox" id="reviewed">кадр размечен</label>
    <button id="dl">Скачать annotations.json</button>
    <label class="sec" style="padding:6px 10px;border:1px solid var(--line);border-radius:7px;cursor:pointer">
      импорт<input type="file" id="imp" accept=".json" style="display:none"></label>
    <span id="notice"></span>
  </div>
  <div id="wrap"><canvas id="cv"></canvas></div>
  <div id="foot">
    <span>рисовать: тянуть мышью</span><span>выбрать: клик</span>
    <span>двигать: тянуть выбранный</span><span>удалить/отклонить: Del</span>
    <span>принять предложение: Enter</span><span>кадры: ←/→ (PgUp/PgDn)</span>
    <span>серый пунктир = предложение движка, в экспорт НЕ идёт, пока не принято</span>
  </div>
</div>
<script type="application/json" id="data">""" + payload + """</script>
<script>
'use strict';
const DATA = JSON.parse(document.getElementById('data').textContent);
const LS_KEY = 'road_defect_eval_annotations_' + DATA.setId;
const COLORS = {pothole:'#e5484d', patch:'#9a5cd0', alligator_crack:'#2f9e73',
                longitudinal_crack:'#3b82d9', transverse_crack:'#0ea5b7'};
const CONF_COLOR = '#e08a2e';

// --- состояние ---------------------------------------------------------------
// proposals_resolved: сигнатуры уже РЕШЁННЫХ предложений движка (приняты или
// отклонены) — живут в state и переживают перезагрузку: иначе принятое
// предложение «воскресало» серым пунктиром и повторный Enter плодил дубль-GT
// (ревью 2026-07-07). В метрики поле не идёт (служебное).
function emptyFrame(f) { return {image_size_px:[f.w,f.h], status:'draft',
                                 defects:[], confusors:[], note:'',
                                 proposals_resolved:[]}; }
function propSig(p) { return p.label + '|' + p.bbox.map(v => Math.round(v)).join(','); }
let state = {version:1, frames:{}};
let proposals = {};   // fid -> [{label,conf,bbox,rejected}]
(function init() {
  const stored = localStorage.getItem(LS_KEY);
  if (stored) {
    try { state = JSON.parse(stored);
      document.getElementById('notice').textContent =
        'восстановлено из localStorage (импортируй файл, чтобы заменить)';
    } catch (e) { state = {version:1, frames:{}}; }
  } else if (DATA.initial) { state = DATA.initial; }
  if (!state.frames) state.frames = {version:1, frames:{}}.frames;
  for (const f of DATA.frames)
    if (!state.frames[f.id]) state.frames[f.id] = emptyFrame(f);
  for (const [fid, props] of Object.entries(DATA.proposals || {})) {
    const done = (state.frames[fid] && state.frames[fid].proposals_resolved) || [];
    proposals[fid] = props.map(p => ({...p, rejected: done.includes(propSig(p))}));
  }
})();

let cur = 0, sel = null, drag = null;   // sel={kind,index}
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const IMG = new Image();

function frameMeta() { return DATA.frames[cur]; }
function frameState() { return state.frames[frameMeta().id]; }
function save() { localStorage.setItem(LS_KEY, JSON.stringify(state)); renderSide(); }

// --- список кадров -----------------------------------------------------------
function renderSide() {
  const box = document.getElementById('frames'); box.innerHTML = '';
  let done = 0;
  DATA.frames.forEach((f, i) => {
    const st = state.frames[f.id];
    const div = document.createElement('div');
    div.className = 'fr' + (i === cur ? ' cur' : '');
    const dot = document.createElement('span');
    dot.className = 'dot' + (st.status === 'reviewed' ? ' rev' :
      (st.defects.length + st.confusors.length ? ' part' : ''));
    if (st.status === 'reviewed') done++;
    div.appendChild(dot);
    div.appendChild(document.createTextNode(f.id));
    div.onclick = () => { cur = i; sel = null; openFrame(); };
    box.appendChild(div);
  });
  document.getElementById('prog').textContent = done + '/' + DATA.frames.length;
}

// --- канвас ------------------------------------------------------------------
function openFrame() {
  const f = frameMeta();
  document.getElementById('title').textContent = f.id;
  document.getElementById('reviewed').checked = frameState().status === 'reviewed';
  IMG.onload = draw; IMG.src = f.uri;
  cv.width = f.dw; cv.height = f.dh;
  renderSide(); updateBar(); draw();
}
function toDisp(v) { return v / frameMeta().scale; }
function toOrig(v) { return v * frameMeta().scale; }
function drawBox(b, color, dash, width, label) {
  ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.lineWidth = width;
  const [x, y, w, h] = b.map(toDisp);
  ctx.strokeRect(x, y, w, h);
  if (label) { ctx.setLineDash([]); ctx.font = '12px system-ui';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color; ctx.fillRect(x, Math.max(0, y - 16), tw + 8, 16);
    ctx.fillStyle = '#fff'; ctx.fillText(label, x + 4, Math.max(12, y - 4)); }
}
function draw() {
  if (!IMG.complete) return;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(IMG, 0, 0, cv.width, cv.height);
  const st = frameState();
  (proposals[frameMeta().id] || []).forEach((p, i) => {
    if (p.rejected) return;
    const on = sel && sel.kind === 'proposal' && sel.index === i;
    drawBox(p.bbox, on ? '#888' : '#9a9aa2', [3, 4], on ? 3 : 1.5,
            p.label + ' ' + p.conf + '?');
  });
  st.confusors.forEach((c, i) => {
    const on = sel && sel.kind === 'confusor' && sel.index === i;
    drawBox(c.bbox_xywh, CONF_COLOR, [7, 5], on ? 3.5 : 2, c.category);
  });
  st.defects.forEach((d, i) => {
    const on = sel && sel.kind === 'defect' && sel.index === i;
    drawBox(d.bbox_xywh, COLORS[d.label] || '#888', [], on ? 3.5 : 2, d.label);
  });
  if (drag && drag.kind === 'new')
    drawBox(drag.bbox, '#43c07d', [4, 3], 2, null);
}

// --- боксы: поиск/операции -----------------------------------------------------
function hit(px, py) {   // px,py в ОРИГИНАЛЬНЫХ px; сверху вниз по слоям
  const st = frameState();
  const inb = (b) => px >= b[0] && py >= b[1] && px <= b[0] + b[2] && py <= b[1] + b[3];
  for (let i = st.defects.length - 1; i >= 0; i--)
    if (inb(st.defects[i].bbox_xywh)) return {kind:'defect', index:i};
  for (let i = st.confusors.length - 1; i >= 0; i--)
    if (inb(st.confusors[i].bbox_xywh)) return {kind:'confusor', index:i};
  const props = proposals[frameMeta().id] || [];
  for (let i = props.length - 1; i >= 0; i--)
    if (!props[i].rejected && inb(props[i].bbox)) return {kind:'proposal', index:i};
  return null;
}
function selBox() {
  if (!sel) return null;
  const st = frameState();
  if (sel.kind === 'defect') return st.defects[sel.index].bbox_xywh;
  if (sel.kind === 'confusor') return st.confusors[sel.index].bbox_xywh;
  return (proposals[frameMeta().id] || [])[sel.index].bbox;
}
function updateBar() {
  const lab = document.getElementById('label');
  const mode = document.getElementById('mode').value;
  const opts = mode === 'defect' ? DATA.defectLabels : DATA.confusorCategories;
  lab.innerHTML = opts.map(o => '<option>' + o + '</option>').join('');
  if (sel && sel.kind === 'defect') lab.value = frameState().defects[sel.index].label;
  if (sel && sel.kind === 'confusor') lab.value = frameState().confusors[sel.index].category;
  document.getElementById('accept').style.display =
    (sel && sel.kind === 'proposal') ? '' : 'none';
}
function resolveProposal(p) {   // решение персистится в state (см. emptyFrame)
  const st = frameState();
  if (!st.proposals_resolved) st.proposals_resolved = [];
  if (!st.proposals_resolved.includes(propSig(p))) st.proposals_resolved.push(propSig(p));
  p.rejected = true;
}
function acceptProposal() {
  if (!sel || sel.kind !== 'proposal') return;
  const p = proposals[frameMeta().id][sel.index];
  frameState().defects.push({label: p.label,
                             bbox_xywh: p.bbox.map(v => Math.round(v * 10) / 10)});
  resolveProposal(p);                      // из предложений ушло в разметку
  sel = {kind:'defect', index: frameState().defects.length - 1};
  save(); updateBar(); draw();
}
function deleteSel() {
  if (!sel) return;
  const st = frameState();
  if (sel.kind === 'defect') st.defects.splice(sel.index, 1);
  else if (sel.kind === 'confusor') st.confusors.splice(sel.index, 1);
  else resolveProposal(proposals[frameMeta().id][sel.index]);
  sel = null; save(); updateBar(); draw();
}

// --- мышь ----------------------------------------------------------------------
cv.addEventListener('mousedown', (e) => {
  const ox = toOrig(e.offsetX), oy = toOrig(e.offsetY);
  const h = hit(ox, oy);
  if (h && sel && h.kind === sel.kind && h.index === sel.index && h.kind !== 'proposal') {
    const b = selBox(); drag = {kind:'move', dx: ox - b[0], dy: oy - b[1]};
  } else if (h) { sel = h; drag = null; updateBar(); draw(); }
  else { sel = null; drag = {kind:'new', x0: ox, y0: oy, bbox:[ox, oy, 0, 0]};
         updateBar(); draw(); }
});
cv.addEventListener('mousemove', (e) => {
  if (!drag) return;
  const ox = toOrig(e.offsetX), oy = toOrig(e.offsetY);
  if (drag.kind === 'new') {
    drag.bbox = [Math.min(drag.x0, ox), Math.min(drag.y0, oy),
                 Math.abs(ox - drag.x0), Math.abs(oy - drag.y0)];
  } else {
    const b = selBox(); b[0] = ox - drag.dx; b[1] = oy - drag.dy;
  }
  draw();
});
window.addEventListener('mouseup', () => {
  if (!drag) return;
  if (drag.kind === 'new') {
    const minPx = 4 * frameMeta().scale;   // случайный клик — не бокс
    if (drag.bbox[2] >= minPx && drag.bbox[3] >= minPx) {
      const bb = drag.bbox.map(v => Math.round(v * 10) / 10);
      const mode = document.getElementById('mode').value;
      const val = document.getElementById('label').value;
      const st = frameState();
      if (mode === 'defect') { st.defects.push({label: val, bbox_xywh: bb});
        sel = {kind:'defect', index: st.defects.length - 1}; }
      else { st.confusors.push({category: val, bbox_xywh: bb});
        sel = {kind:'confusor', index: st.confusors.length - 1}; }
    }
  }
  drag = null; save(); updateBar(); draw();
});

// --- клавиатура/кнопки -----------------------------------------------------------
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'Delete' || e.key === 'Backspace') { deleteSel(); e.preventDefault(); }
  if (e.key === 'Enter') { acceptProposal(); e.preventDefault(); }
  if (e.key === 'ArrowLeft' || e.key === 'PageUp') document.getElementById('prev').click();
  if (e.key === 'ArrowRight' || e.key === 'PageDown') document.getElementById('next').click();
});
document.getElementById('prev').onclick = () => {
  cur = (cur - 1 + DATA.frames.length) % DATA.frames.length; sel = null; openFrame(); };
document.getElementById('next').onclick = () => {
  cur = (cur + 1) % DATA.frames.length; sel = null; openFrame(); };
document.getElementById('accept').onclick = acceptProposal;
document.getElementById('mode').onchange = () => { sel = null; updateBar(); draw(); };
document.getElementById('label').onchange = () => {
  // Значение применяется к выбранному боксу ТОЛЬКО из «его» списка: в режиме
  // «конфузор» с выбранным дефектом запись 'car' в label дефекта валила бы
  // потом весь замер валидацией разметки (ревью 2026-07-07).
  const v = document.getElementById('label').value;
  if (sel && sel.kind === 'defect' && DATA.defectLabels.includes(v))
    frameState().defects[sel.index].label = v;
  if (sel && sel.kind === 'confusor' && DATA.confusorCategories.includes(v))
    frameState().confusors[sel.index].category = v;
  save(); draw();
};
document.getElementById('reviewed').onchange = (e) => {
  frameState().status = e.target.checked ? 'reviewed' : 'draft'; save();
};
document.getElementById('dl').onclick = () => {
  const blob = new Blob([JSON.stringify(state, null, 1)],
                        {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'annotations.json'; a.click();
  URL.revokeObjectURL(a.href);
};
document.getElementById('imp').addEventListener('change', (e) => {
  const file = e.target.files[0]; if (!file) return;
  file.text().then(t => {
    const parsed = JSON.parse(t);
    if (parsed.version !== 1 || !parsed.frames) { alert('не annotations.json v1'); return; }
    state = parsed;
    for (const f of DATA.frames)
      if (!state.frames[f.id]) state.frames[f.id] = emptyFrame(f);
    sel = null; save(); openFrame();
    document.getElementById('notice').textContent = 'импортировано: ' + file.name;
  }).catch(() => alert('файл не прочитался как JSON'));
});

renderSide(); updateBar(); openFrame();
</script>
</body></html>
"""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", default="datasets/eval_v1")
    ap.add_argument("--out", default="outputs_eval/annotator.html")
    ap.add_argument("--proposals", default=None,
                    help="папка отчётов движка (*.json) для предзаполнения")
    ap.add_argument("--max-px", type=int, default=1600,
                    help="макс. сторона встроенного кадра")
    ap.add_argument("--quality", type=int, default=78)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    try:
        manifest = load_manifest(eval_dir)
    except FileNotFoundError as e:
        print(e)
        return 2

    frames, skipped = [], []
    for row in manifest:
        fid = row["frame_id"]
        img = eval_dir / "frames" / f"{fid}.jpg"
        t = _thumb(img, args.max_px, args.quality) if img.is_file() else None
        if t is None:
            skipped.append(fid)
            continue
        uri, ow, oh, dw, dh = t
        frames.append({"id": fid, "uri": uri, "w": ow, "h": oh,
                       "dw": dw, "dh": dh,
                       # прямой и обратный масштаб: боксы храним в ОРИГИНАЛЬНЫХ px
                       "scale": ow / dw})
    if not frames:
        print(f"В {eval_dir / 'frames'} нет читаемых кадров из manifest.csv")
        return 2

    initial = None
    ann_path = eval_dir / "annotations.json"
    if ann_path.is_file():
        try:
            initial = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            print(f"[!] {ann_path} не читается как JSON — страница без предзагрузки")

    proposals = load_proposals(Path(args.proposals) if args.proposals else None,
                               [f["id"] for f in frames])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(frames, initial, proposals), encoding="utf-8")

    n_props = sum(len(v) for v in proposals.values())
    print(f"Разметчик: {out}  ({out.stat().st_size / 1e6:.1f} МБ, кадров "
          f"{len(frames)}, предложений движка {n_props})")
    if skipped:
        print(f"[!] пропущены кадры без файла/битые: {', '.join(skipped)}")
    print("Открой в браузере, разметь, «Скачать annotations.json» и положи в "
          f"{ann_path}. Затем: python scripts/eval_detection.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
