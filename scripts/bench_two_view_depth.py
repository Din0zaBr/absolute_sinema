"""Стенд валидации two-view глубины (photogrammetry.py) на синтетике.

Отвечает на вопрос пилота §8 docs/VEHICLE_CAPTURE.md ДО покупки железа:
выживает ли математика plane+parallax в коде — восстанавливает ли оценщик
ИЗВЕСТНУЮ глубину чаши, честно ли отказывает на затенении/рассинхроне,
не фабрикует ли глубину на плоской заплатке.

Acceptance-критерии (вердикт по dz_p90):
  value            |dz_p90 - dz_true| <= max(tol_abs, tol_rel*dz_true)
  null             отказ ИЛИ dz_p90 <= 1.2 см (заплатка не «углубляется»)
  refuse           только честный отказ (метод != two_view_plane_parallax)
  refuse_or_value  отказ ИЛИ значение в допуске (пограничная геометрия)

Запуск:
  $env:PYTHONPATH="src"
  .\\.venv\\Scripts\\python.exe scripts\\bench_two_view_depth.py [--quick] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from road_defect import config                                  # noqa: E402
from road_defect.photogrammetry import estimate_two_view_depth  # noqa: E402
from synth_road3d import build_two_view_case                    # noqa: E402


def case(name, dz=0.05, d1=4.0, d2=4.0, h=1.30, h_b=None,
         pe_f=0.0, pe_b=0.0, h_nom_b=None, kind="pothole", seed=0,
         mask_morph=0, reg="affine", expect="value",
         tol_abs=0.010, tol_rel=0.20, flat_core=0.45,
         pit_semi=(0.30, 0.22), note=""):
    return dict(name=name, dz=dz, d1=d1, d2=d2, h=h,
                h_b=h if h_b is None else h_b, pe_f=pe_f, pe_b=pe_b,
                h_nom_b=h_nom_b, kind=kind, seed=seed, mask_morph=mask_morph,
                reg=reg, expect=expect, tol_abs=tol_abs, tol_rel=tol_rel,
                flat_core=flat_core, pit_semi=pit_semi, note=note)


CASES = [
    # --- номинал: сетка глубина x дистанция ----------------------------------
    case("dz3_D4", dz=0.03, note="номинал"),
    case("dz5_D4", dz=0.05, note="номинал"),
    case("dz8_D4", dz=0.08, tol_rel=0.25,
         note="глубокая на D=4: взаимно видимо 44.5% дна (ray-check ревью) — "
              "ядро в истинных координатах обязано это измерить"),
    case("dz3_D25", dz=0.03, d1=2.5, d2=2.5, note="номинал, ближе"),
    case("dz5_D25", dz=0.05, d1=2.5, d2=2.5, note="номинал, ближе"),
    case("dz8_D25", dz=0.08, d1=2.5, d2=2.5, note="глубокая на рабочей дистанции"),
    case("dz5_D4_seed7", dz=0.05, seed=7, note="другая текстура/шум"),
    case("dz5_asym_D3_5", dz=0.05, d1=3.0, d2=5.0, note="несимметричные дистанции"),
    case("dz6_elong", dz=0.06, d1=2.5, d2=2.5, pit_semi=(0.45, 0.10),
         note="вытянутая колейная яма 0.9x0.2 м: регресс BLOCKER'а ревью "
              "(эрозия кругового диаметра опустошала ядро)"),
    # --- честность -----------------------------------------------------------
    case("patch_null", dz=0.0, kind="patch", expect="null",
         note="плоская тёмная заплатка: глубины НЕТ"),
    case("dz8_D6_occl", dz=0.08, d1=6.0, d2=6.0, expect="refuse_or_value",
         tol_rel=0.30, note="взаимное затенение дна: отказ лучше лжи"),
    case("dz8_D6_steep", dz=0.08, d1=6.0, d2=6.0, flat_core=0.80,
         expect="refuse_or_value", tol_rel=0.30,
         note="крутые стенки: взаимной зоны дна почти нет"),
    case("dz10_D4_steep", dz=0.10, d1=4.0, d2=4.0, flat_core=0.80,
         expect="refuse_or_value", tol_rel=0.30,
         note="глубокая с крутыми стенками на рабочей дистанции"),
    case("dz5_wrong_hb", dz=0.05, h_nom_b=1.05, expect="refuse",
         note="рассинхрон калибровки высоты: обязан отказать"),
    case("dz22_ceiling", dz=0.22, d1=2.5, d2=2.5, pit_semi=(0.70, 0.45),
         flat_core=0.30, expect="refuse",
         note="дно глубже потолка скана dz_max=0.20: отказ, а не "
              "цензурированная статистика"),
    # --- возмущения рига -----------------------------------------------------
    case("dz5_pitch05", dz=0.05, pe_f=+0.5, pe_b=-0.35, tol_rel=0.30,
         note="тангаж ±0.5° (подвеска)"),
    case("dz5_pitch1", dz=0.05, pe_f=+1.0, pe_b=-0.7, expect="refuse_or_value",
         tol_rel=0.35, note="тангаж ~1°"),
    case("dz5_pitch05_sim", dz=0.05, pe_f=+0.5, pe_b=-0.35, reg="similarity",
         expect="refuse_or_value", tol_rel=0.30,
         note="similarity-регистрация под тангажом"),
    case("dz5_hb105", dz=0.05, h_b=1.05, note="разные высоты камер (обе известны)"),
    # --- несовершенство сегментации ------------------------------------------
    case("dz5_mask_erode", dz=0.05, mask_morph=-9, note="маска на 9 px уже"),
    case("dz5_mask_dilate", dz=0.05, mask_morph=+9, note="маска на 9 px шире"),
]

QUICK = {"dz5_D4", "patch_null", "dz5_wrong_hb"}


def _morph_mask(mask, k):
    if k == 0:
        return mask
    import cv2
    import numpy as np
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(k), abs(k)))
    op = cv2.dilate if k > 0 else cv2.erode
    return op(mask.astype(np.uint8), kern).astype(bool)


def run_case(c: dict) -> dict:
    t0 = time.perf_counter()
    built = build_two_view_case(
        dz=c["dz"], d1=c["d1"], d2=c["d2"], h_f=c["h"], h_b=c["h_b"],
        pitch_err_f=c["pe_f"], pitch_err_b=c["pe_b"], h_nom_b=c["h_nom_b"],
        kind=c["kind"], seed=c["seed"], flat_core=c["flat_core"],
        pit_semi=c["pit_semi"])
    cfg = config.TwoViewDepthConfig(reg_model=c["reg"])
    res = estimate_two_view_depth(
        built["front"]["image"], _morph_mask(built["front"]["mask"], c["mask_morph"]),
        built["view_front"],
        built["back"]["image"], _morph_mask(built["back"]["mask"], c["mask_morph"]),
        built["view_back"], cfg)
    res["_elapsed_s"] = round(time.perf_counter() - t0, 1)
    res["_true_baseline_m"] = built["true_baseline_m"]
    return res


def verdict(c: dict, res: dict) -> tuple[str, str]:
    ok_method = res.get("method") == "two_view_plane_parallax"
    p90 = res.get("depth_cm_p90")
    err = None if p90 is None else abs(p90 / 100.0 - c["dz"])
    tol = max(c["tol_abs"], c["tol_rel"] * c["dz"])
    if c["expect"] == "value":
        if not ok_method:
            return "FAIL", f"отказ ({res.get('method')}) вместо значения"
        return ("PASS", f"err={err * 100:.1f}см (tol {tol * 100:.1f})") \
            if err <= tol else ("FAIL", f"err={err * 100:.1f}см > tol {tol * 100:.1f}")
    if c["expect"] == "null":
        if not ok_method:
            return "PASS", f"честный отказ: {res.get('method')}"
        return ("PASS", f"p90={p90}см ~ 0") if p90 <= 1.2 \
            else ("FAIL", f"фабрикация глубины на заплатке: p90={p90}см")
    if c["expect"] == "refuse":
        return ("PASS", f"отказ: {res.get('method')}") if not ok_method \
            else ("FAIL", f"выдал {p90}см при рассинхроне калибровки")
    if c["expect"] == "refuse_or_value":
        if not ok_method:
            return "PASS", f"честный отказ: {res.get('method')}"
        return ("PASS", f"err={err * 100:.1f}см (tol {tol * 100:.1f})") \
            if err <= tol else ("FAIL", f"уверенно врёт: err={err * 100:.1f}см")
    return "FAIL", f"неизвестный expect={c['expect']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="только смок-подмножество кейсов")
    ap.add_argument("--out", default=str(ROOT / "outputs_bench" / "two_view_depth"),
                    help="куда писать CSV с полными результатами")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.quick or c["name"] in QUICK]
    rows, n_pass = [], 0
    print(f"Стенд two-view глубины: {len(cases)} кейсов\n")
    hdr = (f"{'кейс':<18} {'dz,см':>5} {'D1+D2':>6} {'p90,см':>7} {'med,см':>7} "
           f"{'ядро':>6} {'B,м':>6} {'метод':<34} {'вердикт':<5}")
    print(hdr)
    print("-" * len(hdr))
    for c in cases:
        res = run_case(c)
        v, why = verdict(c, res)
        n_pass += v == "PASS"
        b = res.get("baseline_m")
        print(f"{c['name']:<18} {c['dz'] * 100:>5.1f} "
              f"{c['d1'] + c['d2']:>6.1f} "
              f"{res.get('depth_cm_p90') if res.get('depth_cm_p90') is not None else '—':>7} "
              f"{res.get('depth_cm_median') if res.get('depth_cm_median') is not None else '—':>7} "
              f"{res.get('coverage_core', 0.0):>6.2f} "
              f"{b if b is not None else '—':>6} "
              f"{res.get('method', ''):<34} {v}  {why}")
        rows.append({**{k: c[k] for k in ('name', 'dz', 'd1', 'd2', 'expect',
                                          'reg', 'note')},
                     **{k: res.get(k) for k in
                        ('depth_cm_p90', 'depth_cm_median', 'coverage',
                         'coverage_core', 'n_matches', 'n_core_matches',
                         'saturated_frac', 'mask_iou', 'baseline_m',
                         'n_reg_inliers', 'method', '_elapsed_s',
                         '_true_baseline_m')},
                     "verdict": v, "why": why})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bench_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nИтог: {n_pass}/{len(cases)} PASS. CSV: {csv_path}")
    return 0 if n_pass == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
