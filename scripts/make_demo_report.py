"""Демо-отчёт для презентации: outputs/*.json + картинки → один автономный HTML.

Запуск:  python scripts/make_demo_report.py [--outputs outputs] [--originals .]
                                            [--out outputs/demo_report.html]

Картинки уменьшаются и встраиваются base64 — файл самодостаточен, его можно
открыть на любой машине без проекта. Никаких внешних зависимостей кроме cv2.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASS_RU = {
    "pothole": "яма / выбоина",
    "patch": "заплатка / ремонт",
    "alligator_crack": "сетка трещин",
    "longitudinal_crack": "продольная трещина",
    "transverse_crack": "поперечная трещина",
}
# Цвета как на overlay (report._CLASS_COLORS, BGR -> CSS)
CLASS_CSS = {
    "pothole": "#ff3b30",
    "patch": "#ff9500",
    "alligator_crack": "#ffd60a",
    "longitudinal_crack": "#00d3e0",
    "transverse_crack": "#ff2dca",
}
METHOD_RU = {
    "detector_mask": "маска seg-детектора",
    "crack_blackhat": "классика: black-hat, полное разрешение",
    "crack_tophat": "классика: top-hat (светлая трещина)",
    "mobile_sam": "MobileSAM (пакет)",
    "mobile_sam_ultralytics": "MobileSAM (ultralytics)",
    "grabcut": "GrabCut (грубый фолбэк)",
}
BUCKET_RU = {"shallow": "мелкая", "medium": "средняя", "deep": "глубокая"}
# Тип эталона масштаба (reference.type) → человекочитаемый стандарт.
REF_RU = {
    "manhole_gost3634_cover": "люк (ГОСТ 3634)",
    "marking": "разметка (ГОСТ Р 51256)",
    "curb_gost6665": "борт (ГОСТ 6665)",
}


def scale_badge(scale: dict) -> str:
    """Бейдж эталона по фактическому типу/подтипу, с уверенностью и кросс-проверкой."""
    if not scale.get("available"):
        return '<span class="badge neutral">эталон не найден — размеры в пикселях</span>'
    ref = scale.get("reference", {})
    name = REF_RU.get(ref.get("type"), ref.get("type") or "эталон")
    sub = f' / {ref["subtype"]}' if ref.get("subtype") else ""
    conf = scale.get("confidence") or "?"
    cls = {"high": "ok", "medium": "ok", "low": "warn"}.get(conf, "neutral")
    xc = " · ✓ перекрёстно подтверждён" if scale.get("cross_checked") else ""
    agree = scale.get("agreeing_types") or []
    xc += f" ({', '.join(agree)})" if agree else ""
    return (f'<span class="badge {cls}">эталон: {name}{sub}, '
            f'{ref.get("known_mm", "?")} мм → {scale.get("mm_per_px")} мм/px '
            f'(±{scale.get("error_band_pct", "?")}%, {conf}){xc}</span>')


def fusion_summary(fz: dict) -> str:
    """Блок слияния двух видов (pair-режим); пусто для одиночных отчётов."""
    if not fz:
        return ""
    if fz.get("fused"):
        agree = fz.get("agreement", "?")
        cls = {"agree": "ok", "partial": "warn", "disagree": "bad"}.get(agree, "neutral")
        head = (f'<span class="badge {cls}">слияние двух видов: {agree} '
                f'(расхожд. площади {fz.get("disagreement_pct")}%)</span>')
    else:
        head = ('<span class="badge neutral">два вида: слияние не выполнено '
                '(см. предупреждения) — фабрикованных см нет</span>')
    rows = ""
    for v in fz.get("per_view", []):
        rows += (f"<li>{html.escape(str(v.get('image', '?'))[:28])}: яма "
                 f"{'найдена' if v.get('pothole_found') else 'нет'}, "
                 f"эталон {'есть' if v.get('scale_available') else 'нет'}</li>")
    return f'<p>{head}</p><ul class="small">{rows}</ul>'


def embed_image(path: Path, max_side: int = 1280, quality: int = 82) -> str | None:
    """Файл изображения → data-URI (уменьшенный JPEG)."""
    import cv2
    from road_defect import imgio

    img = imgio.read_image(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def severity_badge(sev: dict) -> str:
    # Контракт severity.py: "yes" | "no" | "indeterminate_without_depth" |
    # "indeterminate_without_scale" (ревью 2026-06-12: сравнение с True/False
    # делало бейджи недостижимыми). Текст warn-бейджа обязан называть
    # НЕДОСТАЮЩЕЕ измерение честно (ревью 2026-07-02).
    nc = sev.get("non_conforming")
    if nc == "yes":
        return '<span class="badge bad">НЕ соответствует ГОСТ</span>'
    if nc == "no":
        return '<span class="badge ok">соответствует ГОСТ</span>'
    if nc == "indeterminate_without_scale":
        return ('<span class="badge warn">нет эталона масштаба — '
                'размеры не подтверждены</span>')
    return ('<span class="badge warn">глубина не подтверждена — '
            'вердикт ГОСТ требует серийной съёмки</span>')


def metric_cell(d: dict) -> str:
    m = d.get("metric", {})
    if m.get("available") and m.get("equivalent_diameter_cm") is not None:
        parts = [f"⌀ ~{m['equivalent_diameter_cm']} см"]
        if m.get("length_cm") is not None:
            parts.append(f"длина {m['length_cm']} см")
        if m.get("area_m2") is not None:
            parts.append(f"{m['area_m2']} м²")
        err = m.get("error_band_pct")
        if err:
            parts.append(f"±{err:.0f}%")
        return "<br>".join(parts)
    shape = d.get("shape", {})
    return f"⌀ {shape.get('equivalent_diameter_px', 0):.0f} px <i>(см недоступны: нет эталона)</i>"


def defect_rows(defects: list) -> str:
    rows = []
    for d in defects:
        cls = d.get("class", "?")
        color = CLASS_CSS.get(cls, "#34c759")
        bucket = d.get("metric", {}).get("depth_bucket")
        depth = BUCKET_RU.get(bucket, "—") if bucket else "—"
        rows.append(f"""
        <tr>
          <td>{d.get('id')}</td>
          <td><span class="dot" style="background:{color}"></span>
              {html.escape(CLASS_RU.get(cls, cls))}</td>
          <td>{d.get('confidence', 0):.2f}</td>
          <td>{metric_cell(d)}</td>
          <td>{depth}</td>
          <td class="small">{html.escape(METHOD_RU.get(d.get('mask_method') or '', d.get('mask_method') or '—'))}</td>
          <td>{severity_badge(d.get('severity', {}))}</td>
        </tr>""")
    return "".join(rows)


def image_card(rep: dict, original_uri: str | None, annotated_uri: str | None) -> str:
    name = html.escape(rep.get("image", "?"))
    scale = rep.get("scale", {})
    scale_txt = scale_badge(scale)
    fusion_txt = fusion_summary(rep.get("fusion"))

    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in rep.get("warnings", []))
    n = len(rep.get("defects", []))
    table = ""
    if n:
        table = f"""
      <table>
        <thead><tr><th>#</th><th>класс</th><th>conf</th><th>размер</th>
                   <th>глубина (относит.)</th><th>метод маски</th><th>ГОСТ Р 50597</th></tr></thead>
        <tbody>{defect_rows(rep["defects"])}</tbody>
      </table>"""

    def img_tag(uri, caption):
        if not uri:
            return f'<div class="noimg">{caption}: файл не найден</div>'
        return f'<figure><img src="{uri}" loading="lazy"><figcaption>{caption}</figcaption></figure>'

    return f"""
  <section class="card">
    <h2>{name} <span class="count">{n} дефект(ов)</span></h2>
    <p>{scale_txt}</p>
    {fusion_txt}
    <div class="pair">
      {img_tag(original_uri, "Оригинал")}
      {img_tag(annotated_uri, "Результат движка")}
    </div>
    {table}
    {f'<details><summary>Предупреждения движка ({len(rep.get("warnings", []))})</summary><ul>{warnings}</ul></details>' if warnings else ''}
  </section>"""


CSS = """
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 0 0 60px;
         background: #f4f5f7; color: #1c1c1e; }
  header { background: #101828; color: #fff; padding: 36px 48px; }
  header h1 { margin: 0 0 6px; font-size: 28px; }
  header p { margin: 4px 0; color: #cdd5e0; max-width: 980px; }
  .stats { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 18px; }
  .stat { background: #1d2939; border-radius: 10px; padding: 12px 18px; min-width: 130px; }
  .stat b { display: block; font-size: 24px; }
  .stat span { color: #98a2b3; font-size: 13px; }
  main { max-width: 1280px; margin: 0 auto; padding: 0 24px; }
  .principle { background: #fff7e6; border: 1px solid #ffd591; border-radius: 12px;
               padding: 14px 20px; margin: 24px 0; }
  .card { background: #fff; border-radius: 14px; padding: 22px 26px; margin: 26px 0;
          box-shadow: 0 1px 4px rgba(16,24,40,.08); }
  .card h2 { margin: 0 0 8px; font-size: 17px; word-break: break-all; }
  .count { color: #667085; font-weight: 400; font-size: 14px; }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 12px 0; }
  figure { margin: 0; }
  figure img { width: 100%; border-radius: 10px; display: block; }
  figcaption { text-align: center; color: #667085; font-size: 13px; margin-top: 6px; }
  .noimg { background: #f2f4f7; border-radius: 10px; display: flex; align-items: center;
           justify-content: center; color: #98a2b3; min-height: 200px; }
  table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }
  th { text-align: left; color: #667085; font-weight: 600; font-size: 12.5px;
       text-transform: uppercase; letter-spacing: .03em; }
  th, td { padding: 8px 10px; border-bottom: 1px solid #eaecf0; vertical-align: top; }
  .dot { display: inline-block; width: 11px; height: 11px; border-radius: 3px;
         margin-right: 7px; vertical-align: -1px; }
  .badge { display: inline-block; border-radius: 999px; padding: 3px 11px;
           font-size: 12.5px; font-weight: 600; }
  .badge.ok { background: #e6f6ec; color: #117a37; }
  .badge.bad { background: #fdebec; color: #c01048; }
  .badge.warn { background: #fff4e5; color: #b54708; }
  .badge.neutral { background: #eef2f6; color: #475467; }
  .small { font-size: 12.5px; color: #475467; }
  details { margin-top: 10px; color: #475467; font-size: 13.5px; }
  footer { max-width: 1280px; margin: 30px auto 0; padding: 0 24px; color: #667085;
           font-size: 13.5px; }
  footer h3 { color: #344054; }
"""


def build_html(reports: list, generated: str) -> str:
    total_defects = sum(len(r["rep"].get("defects", [])) for r in reports)
    by_class: dict = {}
    cm_count = 0
    for r in reports:
        for d in r["rep"].get("defects", []):
            by_class[d.get("class", "?")] = by_class.get(d.get("class", "?"), 0) + 1
            if d.get("metric", {}).get("available"):
                cm_count += 1
    class_chips = " · ".join(
        f"{CLASS_RU.get(k, k)}: {v}" for k, v in sorted(by_class.items(), key=lambda t: -t[1]))
    engine_version = reports[0]["rep"].get("engine_version", "?") if reports else "?"

    cards = "".join(image_card(r["rep"], r["orig"], r["annot"]) for r in reports)
    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Road Defect Engine — демо-отчёт</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{CSS}</style></head>
<body>
<header>
  <h1>Road Defect Engine — анализ дорожных дефектов по фото</h1>
  <p>Детекция и классификация дефектов покрытия, контур/форма, размеры в см при
     наличии эталона в кадре, вердикт по ГОСТ Р 50597-2017. CPU-only, готовые
     открытые модели. Версия движка {html.escape(str(engine_version))}, отчёт от {generated}.</p>
  <div class="stats">
    <div class="stat"><b>{len(reports)}</b><span>фотографий</span></div>
    <div class="stat"><b>{total_defects}</b><span>дефектов найдено</span></div>
    <div class="stat"><b>{cm_count}</b><span>с размерами в см</span></div>
    <div class="stat"><b>0</b><span>ложных чисел: глубина в см не выдаётся без серийной съёмки</span></div>
  </div>
  <p style="margin-top:14px">{html.escape(class_chips)}</p>
</header>
<main>
  <div class="principle">
    <b>Принцип честности измерений.</b> Движок никогда не выдаёт числа, которых
    физически не может знать: глубина в сантиметрах по одному фото невозможна
    (физика монокулярного зрения) — выдаётся только относительная оценка
    «мелкая/средняя/глубокая»; размеры в см появляются только при подтверждённом
    эталоне масштаба (люк ГОСТ 3634 — основной; разметка ГОСТ Р 51256 и борт
    ГОСТ 6665 — опционально, low-confidence, требуют подтверждения). Неподтверждённый
    или конфликтующий эталон отбрасывается, потому что ложный масштаб хуже его
    отсутствия. Слияние двух видов одной ямы («впереди» + «позади») уточняет размеры
    обратно-дисперсионным усреднением и поправкой площади за наклон, но глубину в см
    по-прежнему не выдаёт — два косых кадра не дают сертифицируемую глубину.
  </div>
  {cards}
</main>
<footer>
  <h3>Как читать отчёт</h3>
  <p>Слева оригинал, справа разметка движка: цветные полупрозрачные маски — контур
  дефекта (красный — яма, голубой — продольная трещина, пурпурный — поперечная,
  жёлтый — сетка, оранжевый — заплатка), синий эллипс — эталон-люк. «Метод маски»
  показывает, чем получен контур: нейросеть (MobileSAM) для площадных дефектов,
  классическая экстракция тонких структур в полном разрешении — для трещин.</p>
  <h3>Известные ограничения (зафиксированы в docs/STATUS.md)</h3>
  <p>Серые крышки люков на сером асфальте пока не подтверждаются как эталон —
  масштаб честно не выдаётся; детектор может пропускать нетипичные дефекты
  (засыпанные ямы) — лечится дообучением или вторым проходом; маски трещин в
  сложных тенях могут фрагментироваться. Дорожная карта: фотограмметрия серии
  кадров (ГОСТ-глубина и объём), ONNX-рантайм, backend + карта + уведомления.</p>
</footer>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Сборка демо-отчёта HTML.")
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--originals", default=str(ROOT))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out_dir = Path(args.outputs)
    originals = Path(args.originals)
    out_path = Path(args.out) if args.out else out_dir / "demo_report.html"

    jsons = sorted(out_dir.glob("*.json"))
    if not jsons:
        print(f"Нет *.json в {out_dir}")
        return 1

    reports = []
    for jp in jsons:
        rep = json.loads(jp.read_text(encoding="utf-8"))
        orig_path = originals / rep.get("image", "")
        annot_path = out_dir / f"{jp.stem}_annotated.jpg"
        reports.append({
            "rep": rep,
            "orig": embed_image(orig_path) if orig_path.exists() else None,
            "annot": embed_image(annot_path) if annot_path.exists() else None,
        })
        print(f"  + {jp.stem[:24]:24} дефектов: {len(rep.get('defects', []))}")

    from datetime import date
    page = build_html(reports, generated=date.today().isoformat())
    out_path.write_text(page, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"Готово: {out_path} ({size_mb:.1f} МБ, автономный файл)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
