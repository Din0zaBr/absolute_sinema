# -*- coding: utf-8 -*-
"""Сборка docs/detection_report.pptx из docs/detection_report.html.

Зачем: руководству нужен тот же отчёт в формате презентации. Источник правды —
сам HTML-отчёт: фотографии берутся из его base64 (байт-в-байт), схемы — рендер
ЕГО ЖЕ inline-SVG (подмножество: rect/line/circle/ellipse/polygon/polyline/
path M-L-H-V-Q-Z/text + классы из <style>) через matplotlib. Тексты слайдов —
краткие выжимки тех же формулировок (честность сохранена дословно).

Без новых pip-зависимостей: .pptx — это zip с OOXML, собираем stdlib-зипом;
matplotlib и PIL уже стоят (зависимости ultralytics).

Запуск:  .\\.venv\\Scripts\\python.exe scripts\\make_report_pptx.py
         [--html docs/detection_report.html] [--out docs/detection_report.pptx]
"""
from __future__ import annotations

import argparse
import base64
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

# Кириллица + спецглифы (⚠ ₁ ₂ Ø Δ ᵢ ⊥): rcParams-фолбэк в 3.10 не всегда
# включает per-glyph подбор — список передаётся КАЖДОМУ ax.text явно.
FONT_STACK = ["Arial", "Segoe UI", "Segoe UI Symbol", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = FONT_STACK

ROOT = Path(__file__).resolve().parents[1]
EMU_CM = 360000
SLIDE_W, SLIDE_H = 12192000, 6858000          # 16:9, 33.87 × 19.05 см
INK, MUT, ACC = "151A21", "5B6673", "FF5A4D"
BOX_FILL, BOX_LINE = "F4F5F8", "D7DBE1"

# --- разбор HTML ---------------------------------------------------------------

def extract_photos(html: str) -> list[tuple[str, bytes]]:
    """Все data-изображения по порядку: [(alt или '', jpeg-байты)]."""
    out = []
    for m in re.finditer(
            r'src="data:image/(?:jpeg|png);base64,([^"]+)"(?:\s+alt="([^"]*)")?',
            html):
        out.append((m.group(2) or "", base64.b64decode(m.group(1))))
    return out


def extract_svgs(html: str) -> list[str]:
    """Inline-SVG блоки отчёта по порядку документа."""
    return re.findall(r"<svg .*?</svg>", html, flags=re.S)


# --- рендер SVG-подмножества через matplotlib -----------------------------------

def _svg_classes(style_text: str) -> dict[str, dict[str, str]]:
    classes: dict[str, dict[str, str]] = {}
    for name, body in re.findall(r"([.\w]+)\{([^}]*)\}", style_text):
        props = dict(p.split(":", 1) for p in body.split(";") if ":" in p)
        classes[name.lstrip(".")] = {k.strip(): v.strip() for k, v in props.items()}
    return classes


def _props(el, classes, inherited):
    """Свойства элемента: база text → классы по порядку → атрибуты."""
    p = dict(inherited)
    if el.tag == "text":
        p.update(classes.get("text", {}))
    for cls in (el.get("class") or "").split():
        p.update(classes.get(cls, {}))
    for attr in ("fill", "stroke", "stroke-width", "stroke-dasharray",
                 "font-size", "font-weight", "text-anchor"):
        if el.get(attr) is not None:
            p[attr] = el.get(attr)
    return p


def _dash(p):
    d = p.get("stroke-dasharray")
    if not d:
        return "solid"
    parts = [float(x) * 0.72 for x in re.split(r"[ ,]+", d.strip())]
    return (0, tuple(parts))


def _path_points(d: str) -> tuple[list[tuple[float, float]], bool]:
    """M/L/H/V/Q/Z (только абсолютные — ровно то, что есть в отчёте)."""
    tokens = re.findall(r"[MLHVQZ]|-?\d+(?:\.\d+)?", d)
    pts: list[tuple[float, float]] = []
    closed = False
    i, cmd = 0, None
    while i < len(tokens):
        t = tokens[i]
        if t in "MLHVQZ":
            cmd = t
            i += 1
            if cmd == "Z":
                closed = True
            continue
        if cmd in ("M", "L"):
            pts.append((float(t), float(tokens[i + 1]))); i += 2
        elif cmd == "H":
            pts.append((float(t), pts[-1][1])); i += 1
        elif cmd == "V":
            pts.append((pts[-1][0], float(t))); i += 1
        elif cmd == "Q":
            cx, cy, x1, y1 = (float(tokens[i + k]) for k in range(4))
            x0, y0 = pts[-1]
            for s in range(1, 25):
                u = s / 24.0
                pts.append(((1 - u) ** 2 * x0 + 2 * (1 - u) * u * cx + u * u * x1,
                            (1 - u) ** 2 * y0 + 2 * (1 - u) * u * cy + u * u * y1))
            i += 4
        else:
            raise ValueError(f"неожиданный токен пути: {t!r}")
    return pts, closed


_ANCHOR = {"start": "left", "middle": "center", "end": "right"}


def render_svg_png(svg: str, out_png: Path, dpi: int = 150) -> None:
    svg = re.sub(r'\sxmlns="[^"]+"', "", svg, count=1)
    root = ET.fromstring(svg)
    vb = [float(v) for v in root.get("viewBox").split()]
    W, H = vb[2], vb[3]
    classes = _svg_classes("".join(root.itertext()) if root.find("style") is None
                           else (root.find("style").text or ""))

    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis("off"); fig.patch.set_facecolor("white")

    def draw(el, inherited):
        tag = el.tag
        if tag == "style":
            return
        p = _props(el, classes, inherited)
        stroke = p.get("stroke", "none")
        fill = p.get("fill", "none" if tag != "text" else "#222")
        lw = float(p.get("stroke-width", 1)) * 0.72
        if tag == "g":
            for ch in el:
                draw(ch, p)
        elif tag == "rect":
            ax.add_patch(Rectangle(
                (float(el.get("x")), float(el.get("y"))),
                float(el.get("width")), float(el.get("height")),
                facecolor=fill if fill != "none" else "none",
                edgecolor=stroke if stroke != "none" else "none",
                linewidth=lw, zorder=2))
        elif tag == "line":
            ax.plot([float(el.get("x1")), float(el.get("x2"))],
                    [float(el.get("y1")), float(el.get("y2"))],
                    color=stroke, linewidth=lw, linestyle=_dash(p),
                    solid_capstyle="butt", zorder=3)
        elif tag in ("circle", "ellipse"):
            rx = float(el.get("r", el.get("rx", 0)))
            ry = float(el.get("r", el.get("ry", 0)))
            ax.add_patch(Ellipse(
                (float(el.get("cx")), float(el.get("cy"))), 2 * rx, 2 * ry,
                facecolor=fill if fill != "none" else "none",
                edgecolor=stroke if stroke != "none" else "none",
                linewidth=lw, zorder=4))
        elif tag in ("polygon", "polyline"):
            pts = [tuple(map(float, xy.split(",")))
                   for xy in el.get("points").split()]
            if tag == "polygon":
                ax.add_patch(Polygon(pts, closed=True,
                                     facecolor=fill if fill != "none" else "none",
                                     edgecolor=stroke if stroke != "none" else "none",
                                     linewidth=lw, zorder=3))
            else:
                ax.plot(*zip(*pts), color=stroke, linewidth=lw,
                        linestyle=_dash(p), solid_capstyle="butt", zorder=3)
        elif tag == "path":
            pts, closed = _path_points(el.get("d"))
            if fill != "none":
                ax.add_patch(Polygon(pts, closed=True, facecolor=fill,
                                     edgecolor=stroke if stroke != "none" else "none",
                                     linewidth=lw, zorder=4))
            else:
                ax.plot(*zip(*pts), color=stroke, linewidth=lw,
                        linestyle=_dash(p), solid_capstyle="butt",
                        solid_joinstyle="miter", zorder=3)
        elif tag == "text":
            size = float(p.get("font-size", "15").replace("px", "")) * 0.72
            ax.text(float(el.get("x")), float(el.get("y")), el.text or "",
                    fontsize=size, color=p.get("fill", "#222"),
                    fontfamily=FONT_STACK,
                    fontweight=("bold" if p.get("font-weight") == "bold" else
                                "normal"),
                    ha=_ANCHOR.get(p.get("text-anchor", "start"), "left"),
                    va="baseline", zorder=5)

    for child in root:
        draw(child, {})
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)


# --- OOXML-конструктор ----------------------------------------------------------

NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')


def cm(v: float) -> int:
    return int(round(v * EMU_CM))


class Slide:
    def __init__(self):
        self.shapes: list[str] = []
        self.images: list[tuple[str, bytes, str]] = []  # (имя, байты, ext)
        self._id = 1

    def _next(self) -> int:
        self._id += 1
        return self._id

    def text(self, x, y, w, h, paras, *, anchor="t", fill=None, line=None):
        """paras: [(текст, размер_pt, bold, цвет, align, отступ_после_pt)]."""
        body = []
        for txt, size, bold, color, align, after in paras:
            body.append(
                f'<a:p><a:pPr algn="{align}"><a:spcAft><a:spcPts '
                f'val="{int(after * 100)}"/></a:spcAft><a:buNone/></a:pPr>'
                f'<a:r><a:rPr lang="ru-RU" sz="{int(size * 100)}" '
                f'b="{1 if bold else 0}" dirty="0"><a:solidFill><a:srgbClr '
                f'val="{color}"/></a:solidFill><a:latin typeface="Segoe UI"/>'
                f'<a:cs typeface="Segoe UI"/></a:rPr>'
                f'<a:t>{escape(txt)}</a:t></a:r></a:p>')
        fill_xml = (f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
                    if fill else '<a:noFill/>')
        line_xml = (f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/>'
                    f'</a:solidFill></a:ln>' if line else '')
        n = self._next()
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{n}" name="t{n}"/>'
            f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/>'
            f'</a:xfrm><a:prstGeom prst="roundRect"><a:avLst>'
            f'<a:gd name="adj" fmla="val 6000"/></a:avLst></a:prstGeom>'
            f'{fill_xml}{line_xml}</p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="91440" rIns="91440" '
            f'tIns="45720" bIns="45720" anchor="{anchor}"/><a:lstStyle/>'
            f'{"".join(body)}</p:txBody></p:sp>')

    def picture(self, data: bytes, ext: str, x, y, w, h):
        n = self._next()
        name = f"image_s{id(self)}_{n}.{ext}"
        self.images.append((name, data, ext))
        rid = f"rId{len(self.images) + 1}"          # rId1 занят layout'ом
        self.shapes.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{n}" name="p{n}"/>'
            f'<p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/>'
            f'</a:stretch></p:blipFill>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/>'
            f'</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{BOX_LINE}"/>'
            f'</a:solidFill></a:ln></p:spPr></p:pic>')

    def chip(self, num: str, y_cm: float = 0.9):
        self.text(cm(1.2), cm(y_cm), cm(1.0), cm(1.0),
                  [(num, 16, True, "FFFFFF", "ctr", 0)],
                  anchor="ctr", fill=ACC)

    def title(self, num: str, txt: str, sub: str | None = None):
        self.chip(num)
        self.text(cm(2.5), cm(0.72), cm(30.2), cm(1.4),
                  [(txt, 26, True, INK, "l", 0)])
        if sub:
            self.text(cm(2.5), cm(1.95), cm(30.2), cm(1.1),
                      [(sub, 12.5, False, MUT, "l", 0)])

    def xml(self) -> str:
        return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<p:sld {NS}><p:cSld><p:spTree>'
                f'<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/>'
                f'<p:nvPr/></p:nvGrpSpPr>'
                f'<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
                f'<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>'
                f'</p:grpSpPr>{"".join(self.shapes)}</p:spTree></p:cSld>'
                f'<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


_THEME = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="rd">
<a:themeElements>
<a:clrScheme name="rd"><a:dk1><a:srgbClr val="{INK}"/></a:dk1>
<a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="44546A"/></a:dk2>
<a:lt2><a:srgbClr val="E7E6E6"/></a:lt2>
<a:accent1><a:srgbClr val="{ACC}"/></a:accent1>
<a:accent2><a:srgbClr val="2A8F4D"/></a:accent2>
<a:accent3><a:srgbClr val="2A78D6"/></a:accent3>
<a:accent4><a:srgbClr val="EDA100"/></a:accent4>
<a:accent5><a:srgbClr val="C62828"/></a:accent5>
<a:accent6><a:srgbClr val="898781"/></a:accent6>
<a:hlink><a:srgbClr val="2A78D6"/></a:hlink>
<a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme>
<a:fontScheme name="rd">
<a:majorFont><a:latin typeface="Segoe UI"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>
<a:minorFont><a:latin typeface="Segoe UI"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>
</a:fontScheme>
<a:fmtScheme name="rd">
<a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
<a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
<a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle>
<a:effectStyle><a:effectLst/></a:effectStyle>
<a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
</a:fmtScheme></a:themeElements></a:theme>'''

_MASTER = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster {NS}><p:cSld><p:spTree>
<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>
<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
</p:spTree></p:cSld>
<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1"
 accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5"
 accent6="accent6" hlink="hlink" folHlink="folHlink"/>
<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>'''

_LAYOUT = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout {NS}><p:cSld><p:spTree>
<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>
<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'''


def write_pptx(out: Path, slides: list[Slide], title: str) -> None:
    z = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
    n = len(slides)

    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ] + [f'<Override PartName="/ppt/slides/slide{i + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
         for i in range(n)]
    z.writestr("[Content_Types].xml",
               '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
               '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
               '<Default Extension="xml" ContentType="application/xml"/>'
               '<Default Extension="png" ContentType="image/png"/>'
               '<Default Extension="jpg" ContentType="image/jpeg"/>'
               + "".join(overrides) + "</Types>")

    z.writestr("_rels/.rels",
               '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
               '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
               '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
               '</Relationships>')

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    z.writestr("docProps/core.xml",
               '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
               'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
               'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
               f'<dc:title>{escape(title)}</dc:title>'
               '<dc:creator>Road Defect Engine</dc:creator>'
               f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
               f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
               '</cp:coreProperties>')
    z.writestr("docProps/app.xml",
               '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
               'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
               '<Application>road_defect make_report_pptx</Application></Properties>')

    sld_ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{2 + i}"/>'
                      for i in range(n))
    z.writestr("ppt/presentation.xml",
               f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               f'<p:presentation {NS}>'
               f'<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/>'
               f'</p:sldMasterIdLst><p:sldIdLst>{sld_ids}</p:sldIdLst>'
               f'<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}"/>'
               f'<p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'] + \
           [f'<Relationship Id="rId{2 + i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i + 1}.xml"/>'
            for i in range(n)]
    z.writestr("ppt/_rels/presentation.xml.rels",
               '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               + "".join(rels) + '</Relationships>')

    z.writestr("ppt/theme/theme1.xml", _THEME)
    z.writestr("ppt/slideMasters/slideMaster1.xml", _MASTER)
    z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels",
               '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
               '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
               '</Relationships>')
    z.writestr("ppt/slideLayouts/slideLayout1.xml", _LAYOUT)
    z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels",
               '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
               '</Relationships>')

    for i, s in enumerate(slides, start=1):
        z.writestr(f"ppt/slides/slide{i}.xml", s.xml())
        rel = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>']
        for j, (name, data, _ext) in enumerate(s.images):
            z.writestr(f"ppt/media/{name}", data)
            rel.append(f'<Relationship Id="rId{j + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{name}"/>')
        z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + "".join(rel) + '</Relationships>')
    z.close()


# --- содержимое слайдов ----------------------------------------------------------

def img_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        return im.size


def place_photo(s: Slide, data: bytes, x_cm: float, y_cm: float,
                w_cm: float | None = None, h_cm: float | None = None) -> float:
    """Фото по ширине ИЛИ высоте (вторая сторона — по аспекту). Возвращает
    фактическую высоту в см."""
    w_px, h_px = img_size(data)
    if w_cm is not None:
        h_cm = w_cm * h_px / w_px
    else:
        w_cm = h_cm * w_px / h_px
    s.picture(data, "jpg", cm(x_cm), cm(y_cm), cm(w_cm), cm(h_cm))
    return h_cm


def schema_slide(schemas, idx, num, title, sub, note) -> Slide:
    s = Slide()
    s.title(num, title, sub)
    png = schemas[idx]
    w_px, h_px = img_size(png)
    w_cm_v = 31.4
    h_cm_v = w_cm_v * h_px / w_px
    y = 3.3 if sub else 2.6
    s.picture(png, "png", cm(1.2), cm(y), cm(w_cm_v), cm(h_cm_v))
    if note:
        s.text(cm(1.2), cm(y + h_cm_v + 0.25), cm(31.4), cm(1.7),
               [(note, 12, False, MUT, "l", 0)])
    return s


def build_slides(photos, svgs, tmp: Path) -> list[Slide]:
    by_alt = {alt: data for alt, data in photos if alt}
    schemas = []
    for i, svg in enumerate(svgs):
        png_path = tmp / f"schema{i}.png"
        render_svg_png(svg, png_path)
        schemas.append(png_path.read_bytes())

    S: list[Slide] = []

    # 1 · Титул
    s = Slide()
    s.text(cm(1.2), cm(4.6), cm(31.4), cm(2.2),
           [("Дорожные дефекты глазами модели", 40, True, INK, "ctr", 0)])
    s.text(cm(4.0), cm(7.2), cm(25.8), cm(1.8),
           [("Офлайн-движок учёта дефектов по фото: детекция, контуры, вердикт "
             "ГОСТ Р 50597 — и честный путь к глубине в сантиметрах",
             15, False, MUT, "ctr", 0)])
    stats = [("48×2", "пары «яма с двух сторон»"), ("5", "классов дефектов по ГОСТ"),
             ("0,2–5 с", "на фото, CPU"), ("0", "выдуманных см без эталона")]
    for i, (v, l) in enumerate(stats):
        x = 2.5 + i * 7.5
        s.text(cm(x), cm(10.6), cm(6.9), cm(3.0),
               [(v, 24, True, INK, "ctr", 4), (l, 11, False, MUT, "ctr", 0)],
               anchor="ctr", fill=BOX_FILL, line=BOX_LINE)
    s.text(cm(1.2), cm(17.6), cm(31.4), cm(0.9),
           [("Road Defect Engine · 15.07.2026 · всё офлайн, обычный ноутбук",
             10.5, False, MUT, "ctr", 0)])
    S.append(s)

    # 2 · Что модель видит
    s = Slide()
    s.title("1", "Что модель видит",
            "Найденные дефекты обведены прямо на фото — контур, класс, уверенность.")
    picks = [("Выбоина с оголённым щебнем; контур точно по краю.", None),
             ("Яма и сеть трещин в одном кадре — несколько классов сразу.", None),
             ("Продольная трещина: тонкая структура в полном разрешении.", None)]
    x = 1.2
    for alt, _ in picks:
        data = by_alt[next(k for k in by_alt if k.startswith(alt.split(";")[0][:20]))]
        h = place_photo(s, data, x, 3.5, w_cm=10.0)
        h = min(h, 11.5)
        s.text(cm(x), cm(3.5 + h + 0.15), cm(10.0), cm(1.6),
               [(alt, 10.5, False, MUT, "l", 0)])
        x += 10.7
    S.append(s)

    # 3 · Одна яма — два ракурса (первая чистая пара из раздела 3)
    s = Slide()
    s.title("3", "Одна яма — два ракурса",
            "Каждую яму снимают спереди и сзади: модель находит её в обоих кадрах "
            "и связывает как один объект (Яма A · чистая пара).")
    pair = [d for a, d in photos if not a][:2]      # первые 2 безальтовых = виды пары A
    xx = 1.2
    for data, tag in zip(pair, ("вид спереди · весь кадр", "вид сзади · весь кадр")):
        h = place_photo(s, data, xx, 3.7, w_cm=15.3)
        s.text(cm(xx), cm(3.7 + min(h, 12.2) + 0.15), cm(15.3), cm(0.9),
               [(tag, 11, False, MUT, "l", 0)])
        xx += 16.1
    S.append(s)

    # 4 · Глубина: что подтверждено
    s = Slide()
    s.title("6", "Глубина: что уже подтверждено",
            "Полевая проверка набора из 48 ям — реальные глубины 1–3 см.")
    data012 = by_alt[next(k for k in by_alt if k.startswith("Яма 012"))]
    h = place_photo(s, data012, 1.2, 3.4, h_cm=14.6)
    bullets = [
        ("Полевой замер ямы 012: перепад 1–3 см.", 14),
        ("Фотограмметрический прототип по двум встречным кадрам: 1,4 см "
         "[1,25–1,57] — попадание в полевой диапазон.", 14),
        ("Честная оговорка: это единственная из 95 попыток на архиве, "
         "прошедшая все проверки геометрии. Точность выигрывается протоколом "
         "съёмки, а не дорисовывается софтом.", 14),
        ("Во всех 39 парах с найденной ямой система сказала «мелкая» — "
         "полевая проверка это подтвердила. Сантиметры без основания "
         "не выдуманы ни разу.", 14),
    ]
    paras = []
    for t, sz in bullets:
        paras.append(("•  " + t, sz, False, INK, "l", 10))
    s.text(cm(1.2 + (14.6 * img_size(data012)[0] / img_size(data012)[1]) + 0.9),
           cm(3.6), cm(20.5), cm(14.0), paras)
    S.append(s)

    # 5 · Почему фото не даёт сантиметров (схема 0)
    S.append(schema_slide(
        schemas, 0, "6", "Почему фото само по себе не даёт сантиметров",
        None,
        "Слева: кадр фиксирует только углы — без эталона или известной геометрии "
        "съёмки масштаб не восстановим (у пары с рук неизвестна база B). Справа: "
        "чтение вертикальной рулетки с косого кадра завышает глубину — механизм "
        "отозванных «7 см» при реальных 1–3 см."))

    # 6 · Путь к сантиметрам — обзор трёх ступеней
    s = Slide()
    s.title("7", "Как мы придём к сантиметрам глубины",
            "Три ступени — от ручного замера к съёмке с борта. У каждой — схема, "
            "формула и полоса точности (следующие слайды).")
    cols = [
        ("Этап 1 · Рейка + рулетка", "±0,5–1 см",
         "Ближайшие выезды. Кадр точки касания читается автоматически; "
         "ненадёжный кадр получает отказ с причиной. Протокол утверждён, "
         "алгоритм спроектирован."),
        ("Этап 2 · Серия + маркерная доска", "цель ±1–2 см",
         "Целевой ручной режим (режим D ТЗ): 6–10 кадров при обходе ямы, "
         "доска даёт масштаб и позы. Глубина — без рулетки в яме."),
        ("Этап 3 · Бортовая пара с машины", "потенциал ±0,5–1 см",
         "Массовый охват без остановки автомобиля. Математика проверена "
         "стендом 21/21; нужен калиброванный риг и пилот."),
    ]
    for i, (t, acc, body) in enumerate(cols):
        x = 1.2 + i * 10.65
        s.text(cm(x), cm(3.6), cm(10.0), cm(9.6),
               [(t, 15, True, INK, "l", 6),
                (acc, 14, True, "2A8F4D", "l", 8),
                (body, 12, False, MUT, "l", 0)],
               fill=BOX_FILL, line=BOX_LINE)
    s.text(cm(1.2), cm(13.9), cm(31.4), cm(2.6),
           [("Общее правило: каждое число выходит с полосой погрешности; если "
             "геометрия кадра проверок не прошла — система отказывает с указанием "
             "причины, но не выдумывает.", 13, True, INK, "l", 6),
            ("Уже сегодня, без новых выездов: категория «мелкая / средняя / "
             "глубокая» для каждой ямы; при люке ГОСТ 3634 в кадре (лаз Ø600 мм) — "
             "оценка d ≈ rel·W⊥ с полосой и пометкой «оценка, не для акта».",
             12, False, MUT, "l", 0)])
    S.append(s)

    # 7–9 · этапы со схемами 1..3
    S.append(schema_slide(
        schemas, 1, "7", "Этап 1 · Рейка + рулетка — ближайшие выезды (±0,5–1 см)",
        None,
        "Точка касания рейки и ленты лежит в плоскости ленты — параллакса нет. "
        "Статус: протокол утверждён, алгоритм ридера спроектирован (v1), "
        "реализация — ближайший цикл; включение в продукт — после полевой сверки "
        "на 5–10 ямах (цель — расхождение с ручным чтением ≤ 1 см)."))
    S.append(schema_slide(
        schemas, 2, "7", "Этап 2 · Серия кадров + маркерная доска (цель ±1–2 см)",
        None,
        "Доска известного размера даёт и масштаб, и позу камеры в каждом кадре; "
        "лучи из разных точек триангулируют дно. Цена метода — распечатанный "
        "лист и ~минута на съёмку. Проверка честности: глубина по независимым "
        "под-парам серии обязана совпадать (расхождение ≤ 1,5 см), иначе отказ."))
    S.append(schema_slide(
        schemas, 3, "7", "Этап 3 · Бортовая пара с автомобиля — массовый охват",
        None,
        "Синтетический стенд: 21 сценарий из 21, ошибка 0,0–0,3 см на номинале; "
        "яма с физически невидимым дном получает честный отказ. Полевой кейс "
        "012 (1,4 см при факте 1–3 см) получен именно этим прототипом. Дальше — "
        "калиброванный риг на машине и обязательный пилот."))

    # 10 · Продукт: конвейер (схема 4)
    S.append(schema_slide(
        schemas, 4, "8", "Как это работает в продукте",
        None,
        "Всё офлайн: анализ на CPU, база SQLite и веб-интерфейс — на одной "
        "машине; интернет нужен только странице карты (Leaflet + тайлы "
        "OpenStreetMap). Загрузка по одному фото через браузер или партией "
        "(импорт готовых JSON-отчётов одной командой)."))

    # 11 · Экраны MVP (схема 5)
    S.append(schema_slide(
        schemas, 5, "8", "Экраны MVP — уже работает локально",
        None,
        "Это не концепт: в демо-базе 48 реальных отчётов с координатами, в "
        "реестре 96 дефектов, корректность закреплена 379 автотестами. Статусы "
        "«новый → в работе → устранён», фильтр, карта, уведомления, выгрузка "
        "реестра в CSV. Числа на схеме условные."))

    # 12 · Дальше по продукту
    s = Slide()
    s.title("8", "Дальше по продукту — за рамками MVP", None)
    cols = [
        ("Роли и участки",
         "Привязка дефектов к участкам дорог и ответственным, вход по ролям "
         "(оператор / служба / руководитель) — уведомление сразу попадает "
         "нужному человеку."),
        ("Развёртывание у службы",
         "Тот же интерфейс на сервере организации: загрузка с любых рабочих "
         "мест, общая база, регламентные выгрузки. Архитектура уже разделена — "
         "переезд без переписывания."),
        ("Съёмка по протоколу",
         "Мобильный сценарий с подсказками кадра (обзорный + замерный), а "
         "дальше — автоматический поток с бортовых камер: дефекты встают на "
         "учёт без ручной загрузки."),
    ]
    for i, (t, body) in enumerate(cols):
        x = 1.2 + i * 10.65
        s.text(cm(x), cm(3.6), cm(10.0), cm(10.4),
               [(t, 15, True, INK, "l", 8), (body, 12.5, False, MUT, "l", 0)],
               fill=BOX_FILL, line=BOX_LINE)
    S.append(s)

    # 13 · Честные границы
    s = Slide()
    s.title("9", "Честные границы", None)
    rows = [
        ("Сантиметры — только с эталоном",
         "Размер в см считается лишь при объекте известного размера в кадре "
         "(люк ГОСТ 3634, разметка, линейка). Нет эталона — размеры в пикселях, "
         "с явной оговоркой."),
        ("Глубину «голое» фото не даёт",
         "По обычному снимку выдаётся только категория «мелкая / средняя / "
         "глубокая». Сантиметры глубины — из замерного кадра «рейка + рулетка» "
         "или серии с маркерной доской."),
        ("Пара — когда яма в кадре одна",
         "Два ракурса связываются, только если в каждом кадре одна явная яма; "
         "иначе система отказывается сопоставлять. Лучше не связать, чем "
         "связать неверно."),
        ("Ложные срабатывания бывают",
         "На полном кадре модель иногда метит колесо, тень или заделку как яму. "
         "Системное лекарство — дообучение на местных фото (запущено: eval-набор "
         "и разметка)."),
    ]
    for i, (t, body) in enumerate(rows):
        x = 1.2 + (i % 2) * 16.1
        y = 3.4 + (i // 2) * 5.6
        s.text(cm(x), cm(y), cm(15.3), cm(5.2),
               [(t, 14.5, True, INK, "l", 6), (body, 12, False, MUT, "l", 0)],
               fill=BOX_FILL, line=BOX_LINE)
    s.text(cm(1.2), cm(15.1), cm(31.4), cm(1.6),
           [("Система не выдаёт числа, которых физически не может знать.",
             16, True, INK, "ctr", 0)], anchor="ctr")
    S.append(s)
    return S


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--html", default=str(ROOT / "docs" / "detection_report.html"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "detection_report.pptx"))
    args = ap.parse_args()

    html = Path(args.html).read_text(encoding="utf-8")
    photos = extract_photos(html)
    svgs = extract_svgs(html)
    assert len(svgs) == 6, f"ожидалось 6 SVG-схем, найдено {len(svgs)}"

    tmp = Path(args.out).parent / "_pptx_tmp"
    tmp.mkdir(exist_ok=True)
    slides = build_slides(photos, svgs, tmp)
    write_pptx(Path(args.out), slides, "Дорожные дефекты глазами модели")
    for p in tmp.glob("schema*.png"):
        p.unlink()
    tmp.rmdir()
    print(f"OK: {args.out} — слайдов: {len(slides)}, "
          f"фото: {len(photos)}, схем: {len(svgs)}")


if __name__ == "__main__":
    main()
