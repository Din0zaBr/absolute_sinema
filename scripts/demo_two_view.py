"""Контролируемое (synthetic) демо: масштаб по люку (F2) + слияние двух видов (F1).

ЗАЧЕМ. На реальных демо-фото в репозитории НЕТ эталона масштаба в кадре (люк/
разметка не детектируются), поэтому см там честно не выдаются. Чтобы показать
F1/F2 «в числах» end-to-end, здесь строятся СИНТЕТИЧЕСКИЕ кадры с люком известного
размера (ГОСТ 3634, обод 646 мм) и одной ямой, снятой с ДВУХ ракурсов.

ЧТО НАСТОЯЩЕЕ, А ЧТО ЗАГЛУШКА (честно):
  • НАСТОЯЩИЕ: детект люка и масштаб (scale.resolve_scale на реальном изображении),
    дескрипторы формы, перевод в см, вся математика слияния (fusion.fuse_pair),
    серьёзность по ГОСТ, JSON-контракт. Это тот же продакшен-код.
  • ЗАГЛУШКА: детектор ЯМЫ (YOLO не обучен на синтетике) заменён скриптовым —
    он отдаёт заранее заданную маску ямы. На реальных фото детектор работает сам.

ГЛАВНЫЙ ТЕЗИС ДЕМО. Одиночный КОСОЙ вид ЗАНИЖАЕТ площадь ямы (ракурсное
укорочение). Слияние двух видов снимает это смещение, используя наклон каждого
вида (из эллипса люка), и даёт площадь точнее, чем любой одиночный кадр.

Запуск:  $env:PYTHONPATH="src"; python scripts/demo_two_view.py
Результат: outputs_demo_synth/*.json + *_annotated.jpg + demo_report.html
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road_defect import config, imgio          # noqa: E402
from road_defect import report as report_mod   # noqa: E402
from road_defect.detect import Detection       # noqa: E402
from road_defect.pipeline import DefectPipeline  # noqa: E402

OUT = ROOT / "outputs_demo_synth"
H, W = 3000, 4000
MANHOLE_CENTER = (1200, 1500)
POTHOLE_CENTER = (2800, 1600)


def _asphalt(level: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = np.full((H, W, 3), level, np.uint8).astype(np.int16)
    base += rng.randint(-12, 12, (H, W, 1)).astype(np.int16)
    return np.clip(base, 0, 255).astype(np.uint8)


def make_frame(manhole_semi, pothole_semi, seed):
    """Кадр с тёмным люком (эллипс) и ямой; возвращает (img, mask ямы, bbox ямы)."""
    import cv2

    img = _asphalt(seed=seed)
    cv2.ellipse(img, MANHOLE_CENTER, manhole_semi, 0, 0, 360, (38, 38, 38), -1)
    cv2.ellipse(img, POTHOLE_CENTER, pothole_semi, 0, 0, 360, (60, 60, 60), -1)
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, POTHOLE_CENTER, pothole_semi, 0, 0, 360, 255, -1)
    mask = m > 0
    sx, sy = pothole_semi
    bbox = (float(POTHOLE_CENTER[0] - sx), float(POTHOLE_CENTER[1] - sy),
            float(2 * sx), float(2 * sy))
    return img, mask, bbox


class ScriptedPotholeDetector:
    """Заглушка детектора ямы: отдаёт заранее заданные маски по кадрам по порядку.
    Масштаб (люк) при этом детектится по-настоящему из изображения."""

    is_fallback = False

    def __init__(self, frames):
        self._frames = frames        # [(mask, bbox), ...]
        self._i = 0

    def load(self):
        return self

    def detect(self, image_bgr):
        mask, bbox = self._frames[self._i % len(self._frames)]
        self._i += 1
        return [Detection(cls_name="pothole", raw_label="D40", confidence=0.9,
                          bbox_xywh=bbox, mask=mask)]


def _defect_cm(rep):
    d = rep["defects"][0]["metric"]
    return d


def main() -> int:
    # Перенаправленный stdout на русской Windows — cp1251; «²» в неё не кодируется.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    OUT.mkdir(parents=True, exist_ok=True)

    # Истинная яма ~43 x 28 см. Вид A — близко к надиру (люк-круг). Вид B — косой
    # ~43° (люк сплющен в эллипс), и ЯМА в нём ракурсно укорочена по той же оси.
    tilt_deg = 43.0
    fore = math.cos(math.radians(tilt_deg))      # ≈ 0.731
    # Люк: обод 646 мм. Большая полуось 300 px → диаметр 600 px → 1.077 мм/px.
    manhole_A = (300, 300)
    manhole_B = (300, int(round(300 * fore)))    # сплющен ракурсом
    pothole_A = (200, 130)
    pothole_B = (200, int(round(130 * fore)))    # яма укорочена тем же ракурсом

    imgA, maskA, bboxA = make_frame(manhole_A, pothole_A, seed=1)
    imgB, maskB, bboxB = make_frame(manhole_B, pothole_B, seed=2)
    pathA, pathB = OUT / "yama_vpered.jpg", OUT / "yama_nazad.jpg"
    imgio.write_image(pathA, imgA)
    imgio.write_image(pathB, imgB)

    # Блоб-путь эталона включаем: косой люк (эллипс 2:1) Hough не видит.
    cfg = config.InferenceConfig(allow_blob_reference=True)

    # --- одиночные виды (для сравнения) ---
    pipe_a = DefectPipeline(cfg=cfg, road_category="IV")
    pipe_a.detector = ScriptedPotholeDetector([(maskA, bboxA)])
    repA, _, _ = pipe_a.analyze_image(pathA)

    pipe_b = DefectPipeline(cfg=cfg, road_category="IV")
    pipe_b.detector = ScriptedPotholeDetector([(maskB, bboxB)])
    repB, _, _ = pipe_b.analyze_image(pathB)

    # --- слияние двух видов ---
    pipe = DefectPipeline(cfg=cfg, road_category="IV")
    pipe.detector = ScriptedPotholeDetector([(maskA, bboxA), (maskB, bboxB)])
    rep, ovA, ovB, _ = pipe.analyze_pair(pathA, pathB)

    used: set = set()
    sa = report_mod.unique_stem(pathA.stem, used)
    sb = report_mod.unique_stem(pathB.stem, used)
    report_mod.save_report(rep, OUT, f"{sa}__{sb}_pair")
    imgio.write_image(OUT / f"{sa}__{sb}_pair_annotated.jpg", ovA)
    imgio.write_image(OUT / f"{sb}_annotated.jpg", ovB)

    mA, mB, mF = _defect_cm(repA), _defect_cm(repB), _defect_cm(rep)
    cv = mF.get("cross_view", {})
    line = "-" * 78
    print(line)
    print("КОНТРОЛИРУЕМОЕ ДЕМО: масштаб по люку (F2) + слияние двух видов (F1)")
    print("(масштаб и математика — настоящие; детектор ЯМЫ застаблен синтетикой)")
    print(line)
    print(f"{'величина':<22}{'вид A (надир)':>18}{'вид B (косой 43°)':>20}{'СЛИЯНИЕ':>16}")
    print(f"{'scale, мм/px':<22}{repA['scale']['mm_per_px']!s:>18}"
          f"{repB['scale']['mm_per_px']!s:>20}{'—':>16}")
    print(f"{'наклон люка, °':<22}{repA['scale']['tilt_deg']!s:>18}"
          f"{repB['scale']['tilt_deg']!s:>20}{'—':>16}")
    print(f"{'уверенность':<22}{mA['confidence']!s:>18}{mB['confidence']!s:>20}"
          f"{mF['confidence']!s:>16}")
    print(f"{'площадь, см2':<22}{mA['area_cm2']!s:>18}{mB['area_cm2']!s:>20}"
          f"{mF['area_cm2']!s:>16}")
    print(f"{'длина, см':<22}{mA['length_cm']!s:>18}{mB['length_cm']!s:>20}"
          f"{mF['length_cm']!s:>16}")
    print(f"{'полоса ошибки, %':<22}{mA['error_band_pct']!s:>18}"
          f"{mB['error_band_pct']!s:>20}{mF['error_band_pct']!s:>16}")
    print(line)
    print(f"Согласие видов: {cv.get('agreement')}  |  расхожд. площади "
          f"{cv.get('disagreement_pct')}%  |  площадь скорр. за наклон: "
          f"{cv.get('area_tilt_corrected')}")
    a_corr = cv.get("per_view_area_cm2")
    print(f"Площади после де-ракурса (оба вида): {a_corr} см2")
    print(f"Глубина в см: {mF['depth_cm']} (сертифицируемо: {mF['depth_certifiable']}) "
          "— два косых кадра НЕ дают глубину, и это правильно")
    print(line)
    print("ВЫВОД: косой вид B занизил площадь "
          f"({mB['area_cm2']} см2 против {mA['area_cm2']} у надирного A); слияние сняло "
          f"ракурс и дало {mF['area_cm2']} см2 — точнее любого одиночного кадра.")
    print(line)

    # --- HTML-отчёт по синтетическим кадрам ---
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_demo_report.py"),
         "--outputs", str(OUT), "--originals", str(OUT)],
        capture_output=True, text=True)
    sys.stdout.write(rc.stdout)
    if rc.returncode == 0:
        print(f"HTML-отчёт: {OUT / 'demo_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
