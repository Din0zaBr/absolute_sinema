# -*- coding: utf-8 -*-
"""Собирает автономный HTML-отчёт (одна прокручиваемая страница, картинки
встроены base64). Драйвится манифестом MANIFEST в конце файла."""
import base64, io, json, html, os
from PIL import Image

def _load(path):
    return Image.open(path).convert("RGB")

def _crop(img, bbox, pad=0.6):
    """bbox = [x,y,w,h] (px). Возвращает кроп с полями pad*размер вокруг."""
    W, H = img.size
    x, y, w, h = bbox
    cx, cy = x + w/2, y + h/2
    side_w = w * (1 + 2*pad)
    side_h = h * (1 + 2*pad)
    # держим не слишком вытянутым: минимум квадрат вокруг
    side = max(side_w, side_h, 320)
    half = side/2
    l = max(0, int(cx - half)); t = max(0, int(cy - half))
    r = min(W, int(cx + half)); b = min(H, int(cy + half))
    return img.crop((l, t, r, b))

def _b64(img, maxw, q=82):
    w, h = img.size
    if w > maxw:
        img = img.resize((maxw, int(h*maxw/w)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def img_uri(path, crop=None, pad=0.6, maxw=1100, q=82):
    im = _load(path)
    if crop is not None:
        im = _crop(im, crop, pad)
    return _b64(im, maxw, q)

def esc(s): return html.escape(str(s))

# ---------- рендер секций ----------
def render_gallery(items):
    cards = []
    for it in items:
        uri = img_uri(it["src"], crop=it.get("crop"), pad=it.get("pad",0.6),
                      maxw=it.get("maxw",1000), q=it.get("q",82))
        badges = "".join(f'<span class="badge {b.get("cls","")}">{esc(b["t"])}</span>'
                          for b in it.get("badges", []))
        cap = f'<div class="cap">{esc(it["caption"])}</div>' if it.get("caption") else ""
        cards.append(f'''<figure class="card">
      <div class="imgwrap"><img loading="lazy" src="{uri}" alt="{esc(it.get('caption',''))}"></div>
      <figcaption>{badges}{cap}</figcaption>
    </figure>''')
    return '<div class="gallery">\n' + "\n".join(cards) + "\n</div>"

def render_pairs(pairs):
    blocks = []
    def viewcap(v):
        bits = []
        if v.get("conf") is not None: bits.append(f'уверенность {v["conf"]:.2f}')
        if v.get("ecc") is not None: bits.append(f'экс {v["ecc"]:.2f}')
        if v.get("sol") is not None: bits.append(f'сол {v["sol"]:.2f}')
        return " · ".join(bits)
    for p in pairs:
        ff = img_uri(p["front"]["full"], maxw=940, q=84)
        fb = img_uri(p["back"]["full"], maxw=940, q=84)
        zf = img_uri(p["front"]["zoom"], maxw=560, q=86)
        zb = img_uri(p["back"]["zoom"], maxw=560, q=86)
        blocks.append(f'''<div class="pair">
      <div class="pairhead"><span class="pid">Яма {esc(p['label'])}</span>
        <span class="matched">● одна и та же яма — найдена в обоих ракурсах</span></div>
      <div class="views">
        <figure class="view"><span class="vtag">спереди · весь кадр</span>
          <div class="imgwrap"><img loading="lazy" src="{ff}"></div></figure>
        <div class="link"><div class="linkdot"></div><span>=</span></div>
        <figure class="view"><span class="vtag">сзади · весь кадр</span>
          <div class="imgwrap"><img loading="lazy" src="{fb}"></div></figure>
      </div>
      <div class="zooms">
        <figure class="zoom"><div class="imgwrap"><img loading="lazy" src="{zf}"></div>
          <figcaption>крупно · {esc(viewcap(p['front']))}</figcaption></figure>
        <figure class="zoom"><div class="imgwrap"><img loading="lazy" src="{zb}"></div>
          <figcaption>крупно · {esc(viewcap(p['back']))}</figcaption></figure>
      </div>
      <div class="agree">{esc(p.get("agree_note",""))}</div>
    </div>''')
    return "\n".join(blocks)

def render_diffpairs(items):
    blocks = []
    def cap(v):
        b = []
        if v.get("cls"): b.append(v["cls"])
        if v.get("conf") is not None: b.append(f'уверенность {v["conf"]:.2f}')
        return " · ".join(b)
    for p in items:
        fu = img_uri(p["front"]["full"], maxw=940, q=84)
        bu = img_uri(p["back"]["full"], maxw=940, q=84)
        blocks.append(f'''<div class="pair">
      <div class="pairhead"><span class="pid">{esc(p['label'])}</span>
        <span class="matched diff">● разные ямы — не сопоставлены как одна</span></div>
      <div class="views">
        <figure class="view"><span class="vtag">спереди · весь кадр</span>
          <div class="imgwrap"><img loading="lazy" src="{fu}"></div>
          <figcaption>{esc(cap(p['front']))}</figcaption></figure>
        <div class="link diff"><span>≠</span></div>
        <figure class="view"><span class="vtag">сзади · весь кадр</span>
          <div class="imgwrap"><img loading="lazy" src="{bu}"></div>
          <figcaption>{esc(cap(p['back']))}</figcaption></figure>
      </div>
      <div class="agree">{esc(p.get("caption",""))}</div>
    </div>''')
    return "\n".join(blocks)

def render_missedpairs(items):
    blocks = []
    for p in items:
        fu = img_uri(p["found"]["full"], maxw=940, q=84)
        mu = img_uri(p["missed"]["full"], maxw=940, q=84)
        blocks.append(f'''<div class="pair">
      <div class="pairhead"><span class="pid">{esc(p['label'])}</span>
        <span class="matched miss">● видит только с одного ракурса</span></div>
      <div class="views">
        <figure class="view"><span class="vtag okv">{esc(p['found'].get('view',''))} · НАЙДЕНО</span>
          <div class="imgwrap"><img loading="lazy" src="{fu}"></div>
          <figcaption>яма · уверенность {p['found']['conf']:.2f}</figcaption></figure>
        <div class="link miss"><span>→</span></div>
        <figure class="view"><span class="vtag warnv">{esc(p['missed'].get('view',''))} · ПРОПУЩЕНО</span>
          <div class="imgwrap"><img loading="lazy" src="{mu}"></div>
          <figcaption>та же яма — не выделена</figcaption></figure>
      </div>
      <div class="agree">{esc(p.get("caption",""))}</div>
    </div>''')
    return "\n".join(blocks)

def render_beforeafter(items):
    blocks = []
    for it in items:
        bu = img_uri(it["before"]["src"], maxw=740, q=85)
        au = img_uri(it["after"]["src"], maxw=740, q=85)
        blocks.append(f'''<div class="ba">
      <div class="baviews">
        <figure class="view"><span class="vtag">оригинал</span>
          <div class="imgwrap"><img loading="lazy" src="{bu}"></div></figure>
        <div class="link arrow"><span>→</span></div>
        <figure class="view"><span class="vtag good">нашла система</span>
          <div class="imgwrap"><img loading="lazy" src="{au}"></div></figure>
      </div>
      <div class="agree">{esc(it["caption"])}</div>
    </div>''')
    return "\n".join(blocks)

def render_notes(notes):
    cards = "".join(f'<div class="note"><div class="ni">{esc(n["icon"])}</div>'
                    f'<div><b>{esc(n["t"])}</b><p>{esc(n["d"])}</p></div></div>'
                    for n in notes)
    return f'<div class="notes">{cards}</div>'

CSS = """
:root{--bg:#0e1116;--card:#171b22;--card2:#1d222b;--tx:#e7ecf3;--mut:#98a2b3;
--acc:#ff5a4d;--ok:#39d98a;--line:#232a35;--warn:#f5a623;}
@media (prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--card2:#f0f2f5;
--tx:#151a21;--mut:#5b6673;--line:#e3e7ec;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px 80px}
header.hero{padding:54px 0 26px;border-bottom:1px solid var(--line);margin-bottom:8px}
h1{font-size:34px;margin:0 0 8px;letter-spacing:-.5px}
.sub{color:var(--mut);font-size:17px;max-width:760px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:12px 16px;min-width:120px}
.stat b{font-size:24px;display:block;line-height:1.1}
.stat span{color:var(--mut);font-size:13px}
section{margin-top:46px}
h2{font-size:22px;margin:0 0 4px;display:flex;align-items:center;gap:10px}
h2 .n{width:26px;height:26px;border-radius:7px;background:var(--acc);color:#fff;
font-size:14px;display:grid;place-items:center;flex:none}
.lede{color:var(--mut);margin:2px 0 20px;max-width:820px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.card{margin:0;background:var(--card);border:1px solid var(--line);border-radius:14px;
overflow:hidden}
.imgwrap{background:#000;line-height:0}
.imgwrap img{width:100%;height:auto;display:block}
figcaption{padding:11px 13px}
.badge{display:inline-block;font-size:12px;font-weight:600;padding:2px 8px;border-radius:20px;
background:var(--card2);color:var(--mut);margin:0 6px 6px 0}
.badge.ok{background:rgba(57,217,138,.15);color:var(--ok)}
.badge.acc{background:rgba(255,90,77,.14);color:var(--acc)}
.badge.warn{background:rgba(245,166,35,.15);color:var(--warn)}
.cap{color:var(--tx);font-size:14px;margin-top:2px}
.pair{background:var(--card);border:1px solid var(--line);border-radius:16px;
padding:16px 16px 14px;margin-bottom:18px}
.pairhead{display:flex;justify-content:space-between;align-items:center;gap:12px;
margin-bottom:12px;flex-wrap:wrap}
.pid{font-weight:700;font-size:15px}
.matched{color:var(--ok);font-size:13px;font-weight:700;background:rgba(57,217,138,.13);
border:1px solid rgba(57,217,138,.35);border-radius:20px;padding:4px 11px}
.matched.diff{color:var(--warn);background:rgba(245,166,35,.13);border-color:rgba(245,166,35,.4)}
.matched.miss{color:var(--warn);background:rgba(245,166,35,.13);border-color:rgba(245,166,35,.4)}
.link.diff,.link.miss{color:var(--warn)}
.link.diff span,.link.miss span{font-size:24px}
.vtag.okv{color:var(--ok)}
.vtag.warnv{color:var(--warn)}
.tip{display:flex;gap:12px;align-items:flex-start;background:rgba(57,217,138,.10);
border:1px solid rgba(57,217,138,.35);border-radius:12px;padding:14px 16px;margin:0 0 20px;font-size:14.5px}
.tip .ti{font-size:20px;flex:none;line-height:1.3}
.tip b{color:var(--ok)}
.views{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px}
.view{margin:0}
.view .imgwrap{border-radius:11px;overflow:hidden;border:1px solid var(--line)}
.vtag{display:inline-block;font-size:12px;color:var(--mut);margin-bottom:6px;
text-transform:uppercase;letter-spacing:.06em}
.view figcaption{padding:8px 2px 0;color:var(--mut);font-size:13px;font-variant-numeric:tabular-nums}
.link{display:flex;flex-direction:column;align-items:center;gap:6px;color:var(--ok)}
.linkdot{width:12px;height:12px;border-radius:50%;border:2px solid var(--ok)}
.link span{font-size:20px;font-weight:700}
.zooms{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.zoom{margin:0}
.zoom .imgwrap{border-radius:10px;overflow:hidden;border:1px solid var(--line)}
.zoom figcaption{padding:7px 2px 0;color:var(--mut);font-size:12.5px;font-variant-numeric:tabular-nums}
.agree{margin-top:12px;color:var(--mut);font-size:13.5px;border-top:1px dashed var(--line);
padding-top:10px}
.ba{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;margin-bottom:16px}
.baviews{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px}
.baviews .view .imgwrap{border-radius:11px;overflow:hidden;border:1px solid var(--line)}
.vtag.good{color:var(--ok)}
.link.arrow span{font-size:26px}
.notes{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.note{display:flex;gap:12px;background:var(--card);border:1px solid var(--line);
border-radius:14px;padding:15px}
.note .ni{font-size:22px;flex:none}
.note b{font-size:15px}.note p{margin:4px 0 0;color:var(--mut);font-size:13.5px}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);
color:var(--mut);font-size:13px}
@media(max-width:640px){.views,.baviews{grid-template-columns:1fr;gap:10px}
.link{flex-direction:row;justify-content:center}h1{font-size:27px}}
"""

def build(manifest, out_path):
    m = manifest
    stats = "".join(f'<div class="stat"><b>{esc(s["v"])}</b><span>{esc(s["l"])}</span></div>'
                    for s in m["stats"])
    parts = [f'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(m["title"])}</title><style>{CSS}</style></head><body><div class="wrap">
<header class="hero"><h1>{esc(m["title"])}</h1>
<div class="sub">{esc(m["subtitle"])}</div>
<div class="stats">{stats}</div></header>''']
    for sec in m["sections"]:
        parts.append(f'<section><h2><span class="n">{esc(sec["n"])}</span>{esc(sec["h"])}</h2>')
        if sec.get("lede"): parts.append(f'<div class="lede">{esc(sec["lede"])}</div>')
        if sec.get("tip"): parts.append(f'<div class="tip"><span class="ti">📸</span><div>{sec["tip"]}</div></div>')
        t = sec["type"]
        if t == "gallery": parts.append(render_gallery(sec["items"]))
        elif t == "pairs": parts.append(render_pairs(sec["items"]))
        elif t == "beforeafter": parts.append(render_beforeafter(sec["items"]))
        elif t == "diffpairs": parts.append(render_diffpairs(sec["items"]))
        elif t == "missedpairs": parts.append(render_missedpairs(sec["items"]))
        elif t == "notes": parts.append(render_notes(sec["items"]))
        parts.append('</section>')
    parts.append(f'<footer>{esc(m["footer"])}</footer></div></body></html>')
    html_str = "\n".join(parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print("WROTE", out_path, round(len(html_str.encode())/1e6, 2), "MB")

if __name__ == "__main__":
    print("library ready")
