"""Демо-презентация движка: outputs_demo/*.json + картинки → один автономный HTML-дек.

В отличие от make_demo_report.py (подробный отчёт-таблица для чтения) этот скрипт
собирает ПРЕЗЕНТАЦИЮ в стиле docs/presentation.html: тёмный слайд-дек, много
картинок, мало текста — чтобы показывать, а не читать. Клавиши ← → листают, F —
полный экран.

Ядро истории — съёмка одной ямы с ДВУХ сторон (режим --pair): движок находит
дефект в каждом виде и сопоставляет их. Показаны две честные пары:
  1) реальная городская пара без эталона → two_view_unmatched, см НЕ выдумываются;
  2) контролируемый кадр с люком известного размера → two_view_fused: косой ракурс
     занижает площадь, слияние снимает ракурс, глубина в см всё равно null.

Все картинки уменьшаются и встраиваются base64 — файл самодостаточен.

Запуск:  python scripts/make_demo_deck.py
         [--outputs outputs_demo] [--synth outputs_demo_synth]
         [--out outputs_demo/demo_report.html]
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Цвета overlay (BGR-палитра report.py → CSS), сверено с _annotated.jpg.
COLOR = {
    "pothole": "#ff3b30",
    "patch": "#ff9500",
    "alligator_crack": "#ffd60a",
    "longitudinal_crack": "#00d3e0",
    "transverse_crack": "#ff2dca",
}
CLASS_RU = {
    "pothole": "яма / выбоина",
    "patch": "заплатка / ремонт",
    "alligator_crack": "сетка трещин",
    "longitudinal_crack": "продольная трещина",
    "transverse_crack": "поперечная трещина",
}


# --------------------------------------------------------------------------- #
#  Встраивание картинок
# --------------------------------------------------------------------------- #
def _encode(img, quality: int) -> str:
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def embed(path: Path, max_side: int = 1280, quality: int = 82) -> str:
    """Файл изображения → data-URI (уменьшенный JPEG)."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    return _encode(img, quality)


def embed_crop(path: Path, bbox_xywh, pad: float = 1.9, aspect: float = 1.5,
               max_side: int = 1100, quality: int = 84) -> str:
    """Кадрирование вокруг bbox дефекта (x, y, w, h) с полем `pad`, приведение к
    соотношению сторон `aspect` (w/h) и встраивание. Показывает дефект крупно."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    H, W = img.shape[:2]
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    half_w, half_h = w * pad / 2.0, h * pad / 2.0
    # привести к нужному соотношению сторон, только расширяя (чтобы дефект влез)
    if half_w / half_h < aspect:
        half_w = half_h * aspect
    else:
        half_h = half_w / aspect
    x0 = max(0, int(cx - half_w)); x1 = min(W, int(cx + half_w))
    y0 = max(0, int(cy - half_h)); y1 = min(H, int(cy + half_h))
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    s = min(1.0, max_side / max(ch, cw))
    if s < 1.0:
        crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))),
                          interpolation=cv2.INTER_AREA)
    return _encode(crop, quality)


# --------------------------------------------------------------------------- #
#  Поиск артефактов и загрузка данных
# --------------------------------------------------------------------------- #
def annotated(d: Path, prefix: str, kind: str = "single") -> Path | None:
    for p in sorted(d.glob(prefix + "*_annotated.jpg")):
        n = p.name
        single = "__" not in n and "_ensemble_" not in n
        if kind == "single" and single:
            return p
        if kind == "pair" and "_pair_annotated" in n:
            return p
        if kind == "ensemble" and "_ensemble_annotated" in n:
            return p
    return None


def load_json(d: Path, prefix: str, kind: str = "single") -> dict | None:
    for p in sorted(d.glob(prefix + "*.json")):
        n = p.name
        if kind == "single" and "__" not in n and "_ensemble" not in n:
            return json.loads(p.read_text(encoding="utf-8"))
        if kind == "ensemble" and "_ensemble.json" in n:
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def main_pothole_bbox(rep: dict):
    """bbox самой уверенной ямы (x, y, w, h)."""
    pots = [x for x in rep.get("defects", []) if x.get("class") == "pothole"]
    if not pots:
        return None
    return max(pots, key=lambda x: x.get("confidence", 0)).get("bbox_px")


# --------------------------------------------------------------------------- #
#  CSS (тёмная тема presentation.html + добавки для дека-отчёта)
# --------------------------------------------------------------------------- #
CSS = """
  :root{
    --bg:#191b1e; --bg2:#141619; --card:#26292e; --card2:#2e3237;
    --text:#ECEAE4; --muted:#9CA0A6; --line:#3a3e44;
    --paint:#F4C20D; --paint-dim:rgba(244,194,13,.13);
    --ok:#54B083; --warn:#E2A33A; --no:#DA5A4B; --info:#5AA9E6;
    --sans:"Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
    --mono:"Cascadia Code",Consolas,"SF Mono",ui-monospace,Menlo,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--text);font-family:var(--sans);
    -webkit-font-smoothing:antialiased;overflow:hidden;}
  .deck{position:fixed;inset:0}
  .slide{position:absolute;inset:0;display:none;flex-direction:column;justify-content:center;
    padding:min(6vh,60px) min(7vw,92px);
    background:radial-gradient(1200px 640px at 82% -12%,rgba(244,194,13,.06),transparent 60%),
      radial-gradient(1000px 720px at -8% 112%,rgba(255,255,255,.03),transparent 55%),var(--bg);
    animation:fade .32s ease;}
  .slide.active{display:flex}
  @keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
  .slide.center{align-items:center;text-align:center}
  .wrap{width:100%;max-width:1180px;margin:0 auto}
  h1{font-size:clamp(30px,5.4vw,60px);line-height:1.04;margin:0;letter-spacing:-.02em;font-weight:800}
  h2{font-size:clamp(22px,3.3vw,38px);line-height:1.1;margin:0 0 .45em;letter-spacing:-.015em;font-weight:750}
  .kicker{color:var(--paint);font-weight:700;letter-spacing:.14em;text-transform:uppercase;
    font-size:clamp(11px,1.5vw,14px);margin-bottom:14px}
  p{font-size:clamp(14px,1.75vw,20px);line-height:1.5;color:#DEDCD6;margin:.4em 0}
  .lead{font-size:clamp(16px,2.2vw,24px);color:#EFEEE9}
  .muted{color:var(--muted)}
  b,strong{color:#fff;font-weight:700}
  .paint{color:var(--paint)} .mono{font-family:var(--mono)}
  .dash{height:5px;border-radius:3px;margin:20px 0;
    background:repeating-linear-gradient(90deg,var(--paint) 0 34px,transparent 34px 60px)}
  ul{margin:.3em 0;padding-left:0;list-style:none}
  li{position:relative;padding-left:28px;margin:.5em 0;font-size:clamp(14px,1.8vw,20px);line-height:1.4;color:#DEDCD6}
  li::before{content:"";position:absolute;left:2px;top:.6em;width:11px;height:11px;border-radius:2px;
    background:var(--paint);transform:rotate(45deg)}
  li.ok::before{background:var(--ok)} li.warn::before{background:var(--warn)} li.no::before{background:var(--no)}
  .grid{display:grid;gap:18px}
  .g2{grid-template-columns:1fr 1fr} .g3{grid-template-columns:repeat(3,1fr)}
  .card{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
    border-radius:16px;padding:18px 22px}
  .card h3{margin:0 0 6px;font-size:clamp(15px,1.9vw,21px)}
  .card p{font-size:clamp(13px,1.55vw,17px);margin:.25em 0}
  .authors{margin-top:28px;padding-top:16px;border-top:1px solid var(--line);display:inline-block;
    font-size:clamp(14px,1.9vw,20px);color:#DEDCD6}
  .gloss{color:var(--muted);font-weight:400;font-size:.92em}
  .hero{max-height:76vh;border-radius:14px;border:1px solid var(--line);box-shadow:0 18px 50px #0008;width:auto}

  /* stat tiles */
  .statrow{display:flex;gap:13px;flex-wrap:wrap;justify-content:center;margin-top:24px}
  .stile{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
    border-radius:14px;padding:13px 20px;min-width:132px}
  .stile b{display:block;font-size:clamp(24px,3.6vw,40px);font-weight:800;color:var(--paint);line-height:1}
  .stile span{color:var(--muted);font-size:12.5px}

  /* legend */
  .legend{display:flex;flex-direction:column;gap:12px}
  .lchip{display:flex;align-items:center;gap:12px;font-size:clamp(14px,1.7vw,19px);color:#EFEEE9}
  .sw{width:22px;height:22px;border-radius:5px;flex:none;box-shadow:0 0 0 1px #0006 inset}

  /* badges */
  .badges{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}
  .badge{display:inline-block;border-radius:999px;padding:5px 13px;font-size:clamp(12px,1.4vw,15.5px);
    font-weight:700;border:1px solid transparent}
  .badge.ok{background:rgba(84,176,131,.16);color:var(--ok);border-color:rgba(84,176,131,.42)}
  .badge.warn{background:rgba(226,163,58,.15);color:var(--warn);border-color:rgba(226,163,58,.42)}
  .badge.no{background:rgba(218,90,75,.15);color:var(--no);border-color:rgba(218,90,75,.42)}
  .badge.mono{font-family:var(--mono);background:#00000033;color:#cfcfcf;border-color:var(--line);font-weight:600}

  /* pair / two-view */
  .pairrow{display:grid;grid-template-columns:1fr auto 1fr;gap:16px;align-items:center;margin-top:8px}
  .vframe{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;
    display:flex;flex-direction:column}
  .vframe img{display:block;width:100%;height:auto;max-height:52vh;object-fit:contain;
    background:#0c0d0f;border-bottom:1px solid var(--line)}
  .vframe figcaption{padding:8px 13px;font-size:clamp(11px,1.3vw,14.5px);color:#D8D6D0;line-height:1.3}
  .vframe .vtag{display:block;font-family:var(--mono);color:var(--paint);font-size:11px;
    letter-spacing:.05em;text-transform:uppercase;margin-bottom:2px}
  .conn{display:flex;flex-direction:column;align-items:center;gap:4px;color:var(--paint);
    font-family:var(--mono);font-size:12px;text-align:center;min-width:96px}
  .conn .a{font-size:30px;line-height:.7}
  .numstrip{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;justify-content:center;
    margin-top:14px;font-family:var(--mono);font-size:clamp(13px,1.7vw,20px)}
  .numstrip .u{color:var(--muted)} .numstrip .v{color:#fff;font-weight:700}
  .numstrip .hi{color:var(--paint);font-weight:800} .numstrip .op{color:var(--paint);padding:0 2px}
  .ctrltag{display:inline-block;font-family:var(--mono);font-size:12px;color:var(--warn);
    border:1px dashed var(--warn);border-radius:8px;padding:4px 10px;margin-bottom:6px}

  /* gallery */
  .gallery{display:flex;gap:13px;justify-content:center;align-items:flex-start;flex-wrap:nowrap}
  .gshot{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;
    flex:1 1 0;display:flex;flex-direction:column;align-items:center}
  .gshot img{display:block;width:auto;max-width:100%;max-height:50vh;object-fit:contain;
    background:#0c0d0f;border-bottom:1px solid var(--line)}
  .gshot figcaption{padding:8px 11px;font-size:clamp(10.5px,1.15vw,13.5px);color:#D8D6D0;line-height:1.28;width:100%}
  .gshot .cf{font-family:var(--mono);color:var(--ok);font-weight:700}

  /* chrome */
  .bar{position:fixed;left:0;top:0;height:3px;background:var(--paint);width:0;z-index:20;transition:width .3s}
  .pnum{position:fixed;right:18px;bottom:14px;font-family:var(--mono);font-size:13px;color:var(--muted);z-index:20}
  .brand{position:fixed;left:20px;bottom:12px;font-size:12px;color:var(--muted);z-index:20;letter-spacing:.03em}
  .brand b{color:var(--paint)}
  .zone{position:fixed;top:0;bottom:0;width:22%;z-index:15;cursor:pointer} #prev{left:0} #next{right:0}
  .hint{position:fixed;right:16px;top:14px;font-size:12px;color:var(--muted);z-index:20}
  kbd{font-family:var(--mono);background:#000a;border:1px solid var(--line);border-bottom-width:2px;
    border-radius:5px;padding:1px 6px;font-size:11px;color:#cfcfcf}
  @media (max-width:820px){ .g2,.g3{grid-template-columns:1fr} .pairrow{grid-template-columns:1fr}
    .conn{flex-direction:row}.gallery{flex-wrap:wrap} .zone{display:none} .hint{display:none} }
"""

NAV = """
  const slides=[...document.querySelectorAll('.slide')];
  const bar=document.getElementById('bar'), pnum=document.getElementById('pnum');
  let i=0;
  function show(n){
    i=Math.max(0,Math.min(slides.length-1,n));
    slides.forEach((s,k)=>s.classList.toggle('active',k===i));
    pnum.textContent=(i+1)+' / '+slides.length;
    bar.style.width=(i/(slides.length-1)*100)+'%';
    history.replaceState(null,'','#'+(i+1));
  }
  addEventListener('keydown',e=>{
    if(['ArrowRight','PageDown',' '].includes(e.key)){show(i+1);e.preventDefault();}
    else if(['ArrowLeft','PageUp'].includes(e.key)){show(i-1);e.preventDefault();}
    else if(e.key==='Home'){show(0);e.preventDefault();}
    else if(e.key==='End'){show(slides.length-1);e.preventDefault();}
    else if(e.key.toLowerCase()==='f'){const d=document;
      (d.fullscreenElement?d.exitFullscreen():d.documentElement.requestFullscreen())?.catch(()=>{});}
  });
  document.getElementById('next').onclick=()=>show(i+1);
  document.getElementById('prev').onclick=()=>show(i-1);
  let x0=null;
  addEventListener('touchstart',e=>x0=e.touches[0].clientX,{passive:true});
  addEventListener('touchend',e=>{if(x0==null)return;const dx=e.changedTouches[0].clientX-x0;
    if(Math.abs(dx)>50)show(i+(dx<0?1:-1));x0=null;},{passive:true});
  show((parseInt(location.hash.slice(1))||1)-1);
"""


def build(demo: Path, synth: Path) -> str:
    # --- собрать картинки (все гейтированы визуально при подготовке дека) ---
    A = lambda pre, kind="single": annotated(demo, pre, kind)
    img_hero = embed(A("JO7VcIsR"), max_side=1500, quality=84)          # флагман, 6 дефектов
    img_q5 = embed(A("Q5pslz"), max_side=1000)                          # яма 0.89 (близко)
    img_cw = embed(A("CWSCOk"), max_side=1000)                          # две ямы + люк не-эталон
    img_hw = embed(A("hwfU5q"), max_side=1000)                          # две трещины
    img_2i = embed(A("2_IdWsB"), max_side=1000)                         # яма+трещина, широкий план

    # пара 1 — реальная, без эталона (виды одной ямы: спереди Q5pslz, сзади 2_IdWs)
    pair_front = embed(A("Q5pslz"), max_side=1050)
    pair_back = embed(A("2_IdWsB"), max_side=1050)

    # пара 2 — контролируемый кадр с люком (синтетика): слияние работает
    synth_front = embed(annotated(synth, "yama_vpered", "pair"), max_side=1200)  # прямой вид (люк-круг)
    synth_back = embed(annotated(synth, "yama_nazad"), max_side=1200)            # косой вид (люк-эллипс)

    # 6dyN — базовый детектор пропустил → --ensemble нашёл (before/after одним кадром)
    ens_rep = load_json(demo, "6dyN", "ensemble")
    bbox = main_pothole_bbox(ens_rep) or [970, 238, 805, 579]
    miss_before = embed_crop(A("6dyN"), bbox, pad=1.9, aspect=1.5)               # base annotated = без разметки
    miss_after = embed_crop(A("6dyN", "ensemble"), bbox, pad=1.9, aspect=1.5)

    def legend_chip(cls):
        return (f'<div class="lchip"><span class="sw" style="background:{COLOR[cls]}"></span>'
                f'{CLASS_RU[cls]}</div>')

    legend = "".join(legend_chip(c) for c in
                     ["pothole", "longitudinal_crack", "transverse_crack",
                      "alligator_crack", "patch"])

    slides = []

    # 1 — титул
    slides.append(f"""
  <section class="slide center active">
    <div class="wrap">
      <div class="kicker">Демонстрация · прогон движка на реальных фото города</div>
      <h1>Road&nbsp;Defect&nbsp;Engine — что увидел движок</h1>
      <p class="lead" style="margin-top:16px;max-width:880px;margin-inline:auto">
        Шесть снимков города прошли через движок автоматически: дефекты найдены,
        обведены контуром, классифицированы. <b>Ни одного придуманного числа.</b></p>
      <div class="statrow">
        <div class="stile"><b>6</b><span>фотографий</span></div>
        <div class="stile"><b>14</b><span>дефектов найдено</span></div>
        <div class="stile"><b>2</b><span>вида одной ямы сопоставлены</span></div>
        <div class="stile"><b>0</b><span>ложных сантиметров</span></div>
      </div>
      <div class="authors">
        <span class="muted">Проект подготовили:</span>
        <b>Заболотный И. А.</b> &nbsp;·&nbsp; <b>Картошкин А. М.</b>
        <span class="muted">&nbsp;— кафедра КБИС</span>
      </div>
    </div>
  </section>""")

    # 2 — как читать (легенда + честность)
    slides.append(f"""
  <section class="slide">
    <div class="wrap">
      <div class="kicker">Как читать разметку</div>
      <h2>Цвет — класс дефекта, число — уверенность модели</h2>
      <div class="grid" style="grid-template-columns:auto 1fr;gap:40px;align-items:center;margin-top:6px">
        <div class="legend">{legend}</div>
        <div class="grid g3">
          <div class="card" style="border-color:var(--ok)"><h3 class="paint">Всегда</h3>
            <p>Дефект, класс и контур — на любом фото.</p></div>
          <div class="card" style="border-color:var(--warn)"><h3 class="paint">С эталоном</h3>
            <p>Сантиметры — только если в кадре есть объект известного размера (люк).</p></div>
          <div class="card" style="border-color:var(--no)"><h3 class="paint">Никогда по 1 фото</h3>
            <p>Глубина в см — физически неопределима; выдаётся <span class="mono">null</span>.</p></div>
        </div>
      </div>
      <div class="dash"></div>
      <p class="muted">Полупрозрачная маска — точный контур дефекта. Синий эллипс (если есть) — эталон-люк.
        Размер в <span class="mono">px</span> означает: эталона в кадре нет, сантиметры честно не выданы.</p>
    </div>
  </section>""")

    # 3 — флагман JO7VcI (hero)
    slides.append(f"""
  <section class="slide">
    <div class="wrap">
      <div class="kicker">Одно фото → размеченный отчёт</div>
      <div class="grid" style="grid-template-columns:auto 1fr;align-items:center;gap:34px">
        <img class="hero" src="{img_hero}" alt="Городская сцена: яма и трещины, 6 дефектов">
        <div>
          <h2>Шесть дефектов, четыре класса — за один проход</h2>
          <ul>
            <li class="ok"><b>Яма</b> в центре проезда — <span class="mono">pothole 0.79</span></li>
            <li><b>Поперечные трещины</b> — <span class="mono">0.65 / 0.61</span></li>
            <li><b>Продольные трещины</b> в зоне под тенью дерева</li>
            <li>Каждый контур — своим методом: ямы обводит <b>MobileSAM</b>,
              трещины — классика в полном разрешении</li>
            <li class="warn">Размеры в пикселях: эталона в кадре нет — сантиметры не выдаются</li>
          </ul>
          <p class="muted" style="margin-top:8px">Это не макет — прямой вывод программы:
            <span class="mono">_annotated.jpg</span> + машиночитаемый <span class="mono">.json</span>.</p>
        </div>
      </div>
    </div>
  </section>""")

    # 4 — галерея города
    slides.append(f"""
  <section class="slide">
    <div class="wrap">
      <div class="kicker">Практический результат · дороги города</div>
      <h2 style="margin-bottom:.3em">Разные дефекты, разные сцены</h2>
      <div class="gallery">
        <figure class="gshot"><img src="{img_q5}" alt="Яма крупным планом">
          <figcaption><span class="cf">pothole 0.89</span> · маска точно по краю выбоины</figcaption></figure>
        <figure class="gshot"><img src="{img_cw}" alt="Две ямы, люк не принят за эталон">
          <figcaption><span class="cf">pothole 0.82 / 0.81</span> · люк в кадре <b>не</b> принят за эталон</figcaption></figure>
        <figure class="gshot"><img src="{img_hw}" alt="Две продольные трещины">
          <figcaption><span class="cf">crack 0.80 / 0.74</span> · продольные трещины</figcaption></figure>
        <figure class="gshot"><img src="{img_2i}" alt="Яма и трещина, широкий план">
          <figcaption><span class="cf">pothole 0.81</span> · яма + продольная трещина, широкий план</figcaption></figure>
      </div>
    </div>
  </section>""")

    # 5 — ПАРА 1: реальная, без эталона (сопоставление видов)
    slides.append(f"""
  <section class="slide">
    <div class="wrap">
      <div class="kicker">Съёмка одной ямы с двух сторон · режим --pair</div>
      <h2>Модель находит яму в обоих видах и сопоставляет их</h2>
      <div class="pairrow">
        <figure class="vframe"><img src="{pair_front}" alt="Вид спереди: яма 0.89">
          <figcaption><span class="vtag">вид спереди</span>яма найдена, <span class="mono">pothole 0.89</span></figcaption></figure>
        <div class="conn"><div>matched</div><div class="a">↔</div><div>сопоставлено</div></div>
        <figure class="vframe"><img src="{pair_back}" alt="Вид сзади: яма 0.81 + трещина">
          <figcaption><span class="vtag">вид сзади</span>яма найдена, <span class="mono">pothole 0.81</span> + трещина</figcaption></figure>
      </div>
      <div class="badges">
        <span class="badge ok">виды сопоставлены · matched: true</span>
        <span class="badge warn">слияние в см НЕ выполнено — эталона нет ни в одном виде</span>
        <span class="badge mono">two_view_unmatched</span>
      </div>
      <p class="muted" style="margin-top:8px">Честно: без объекта известного размера в кадре
        сантиметры не выдумываются. Виды связаны, но размеры остаются одиночными.</p>
    </div>
  </section>""")

    # 6 — ПАРА 2: контролируемый кадр с люком → слияние работает
    slides.append(f"""
  <section class="slide">
    <div class="wrap">
      <div class="kicker">Что даёт эталон в кадре · контролируемый прогон</div>
      <span class="ctrltag">контролируемый эталонный кадр (синтетика): люк ГОСТ 3634 = 646 мм · масштаб и геометрия настоящие</span>
      <h2 style="margin-top:4px">Два вида + люк известного размера → сантиметры со слиянием</h2>
      <div class="pairrow">
        <figure class="vframe"><img src="{synth_front}" alt="Прямой вид: люк-круг">
          <figcaption><span class="vtag">прямой вид · наклон 0.6°</span>люк почти круглый → масштаб надёжный</figcaption></figure>
        <div class="conn"><div>fused</div><div class="a">⇄</div><div>слияние</div></div>
        <figure class="vframe"><img src="{synth_back}" alt="Косой вид: люк-эллипс">
          <figcaption><span class="vtag">косой вид · наклон 43°</span>люк сплющен в эллипс → площадь занижена</figcaption></figure>
      </div>
      <div class="numstrip">
        <span class="u">косой вид</span> <span class="v">703&nbsp;см²</span>
        <span class="u">·</span> <span class="u">прямой</span> <span class="v">949&nbsp;см²</span>
        <span class="op">→</span> <span class="u">слияние</span> <span class="hi">951&nbsp;см²</span>
        <span class="u">(согласие&nbsp;1.6%)</span>
      </div>
      <div class="badges">
        <span class="badge ok">two_view_fused · длина 43 см · площадь 0.095 м² · ±30%</span>
        <span class="badge warn">глубина = null — два косых кадра ≠ серия SfM</span>
      </div>
    </div>
  </section>""")

    # 7 — честная слабость и её устранение (6dyN)
    slides.append(f"""
  <section class="slide">
    <div class="wrap">
      <div class="kicker">Слабость показана и устранена</div>
      <h2>Яму, засыпанную светлым бетоном, базовый детектор не увидел</h2>
      <div class="pairrow">
        <figure class="vframe"><img src="{miss_before}" alt="Базовый детектор: пусто">
          <figcaption><span class="vtag">базовый детектор</span>0 откликов даже при пороге 0.01 —
            яма не похожа на тёмные ямы датасета RDD2022</figcaption></figure>
        <div class="conn"><div>--ensemble</div><div class="a">→</div><div>2-й проход</div></div>
        <figure class="vframe"><img src="{miss_after}" alt="Ensemble: яма найдена 0.81">
          <figcaption><span class="vtag">второй проход</span>яма найдена: <span class="mono">pothole 0.81</span>,
            маска точно по краю</figcaption></figure>
      </div>
      <div class="badges">
        <span class="badge ok">--ensemble нашёл пропуск</span>
        <span class="badge mono">инфраструктура дообучения на местных данных готова</span>
      </div>
      <p class="muted" style="margin-top:8px">Показывать слабость и её устранение — честнее, чем скрывать.
        Диагностика (<span class="mono">detect_sweep.py</span>) подтвердила ноль откликов до второго прохода.</p>
    </div>
  </section>""")

    # 8 — финал
    slides.append(f"""
  <section class="slide center">
    <div class="wrap">
      <h1>Каждое число — движок физически&nbsp;может&nbsp;его&nbsp;знать</h1>
      <div class="dash" style="max-width:520px;margin:26px auto"></div>
      <p class="lead" style="max-width:820px;margin-inline:auto">
        Детекция и контур — на любом фото. Сантиметры — только с эталоном.
        Глубина — из серии кадров, а не из одного снимка.</p>
      <p class="muted" style="margin-top:22px">Дорожная карта: фотограмметрия серии кадров
        (ГОСТ-глубина и объём) → ONNX-рантайм → backend, карта и автоотчёты ответственным.</p>
      <p class="muted" style="margin-top:18px">Проект подготовили:
        <b>Заболотный И. А.</b>, <b>Картошкин А. М.</b> — кафедра КБИС.</p>
    </div>
  </section>""")

    body = "".join(slides)
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Road Defect Engine — демо-презентация</title>
<style>{CSS}</style>
</head>
<body>
<div class="bar" id="bar"></div>
<div class="zone" id="prev"></div>
<div class="zone" id="next"></div>
<div class="hint"><kbd>←</kbd> <kbd>→</kbd> листать · <kbd>F</kbd> во весь экран</div>
<div class="brand"><b>Road Defect Engine</b> · демо-отчёт по фото города</div>
<div class="pnum" id="pnum"></div>
<div class="deck">{body}
</div>
<script>{NAV}</script>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Сборка демо-презентации (тёмный слайд-дек).")
    ap.add_argument("--outputs", default=str(ROOT / "outputs_demo"))
    ap.add_argument("--synth", default=str(ROOT / "outputs_demo_synth"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    demo = Path(args.outputs)
    synth = Path(args.synth)
    out_path = Path(args.out) if args.out else demo / "demo_report.html"

    page = build(demo, synth)
    out_path.write_text(page, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"Готово: {out_path} ({size_mb:.1f} МБ, автономный слайд-дек)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
