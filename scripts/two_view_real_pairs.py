"""Экспериментальный прогон two-view глубины (photogrammetry.py v0) на РЕАЛЬНЫХ
ручных парах перед/за (datasets/local_pairs) — режим C из docs/TZ_MEASUREMENT.md.

ЧЕСТНОСТЬ (главный принцип проекта): ручные пары — НЕ калиброванный риг.
  * f_px восстановлен из EXIF: все кадры сняты одним realme 13+ 5G,
    FocalLength=4.84 мм, зум 1.0. Два независимых пересчёта дают
    f_px ≈ 3025 (4.84 мм / 1.6 мкм биннинг) и ≈ 2958 (26 мм экв. × 4096/36) —
    расхождение ~2% входит в полосу ошибки.
  * Высота камеры и тангаж НЕИЗВЕСТНЫ → перебор по сетке тангажей при
    номинальной высоте; гейты оценщика (регистрация кольца, рассинхрон
    масштаба, анизотропия, покрытие дна) отбраковывают ложные калибровки.
  * Результат пары — диапазон dz по ПЛАТО прошедших конфигураций плюс
    мультипликативная полоса (высота ±10%, f ±2%). Нет плато или плато
    нестабильно → отказ с причиной. depth_certifiable=False ВСЕГДА.

Запуск (venv):
  $env:PYTHONPATH="src"
  .\\.venv\\Scripts\\python.exe scripts\\two_view_real_pairs.py [--pairs 006 012] [--out FILE]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_defect import config  # noqa: E402
from road_defect import photogrammetry  # noqa: E402
from road_defect.photogrammetry import RigView, estimate_two_view_depth  # noqa: E402
from road_defect.pipeline import DefectPipeline, _select_dominant_pothole  # noqa: E402
from road_defect import imgio  # noqa: E402

# --- калибровка, восстановимая из EXIF (одна модель телефона на весь набор) ---
F_PX = 3025.0          # 4.84 мм / 1.6 мкм (Sony LYT-600, биннинг 2x2)
F_BAND_PCT = 2.0       # расхождение двух пересчётов f_px
H_NOM_M = 1.60         # высота съёмки с рук: номинал…
H_BAND_PCT = 10.0      # …и честная неопределённость (1.45–1.75 м)
# Сетка тангажей (вниз от горизонта). По композиции кадров — 10–40°.
PITCH_GRID = (10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 35.0, 40.0)
# Плато стабильно, если разброс медиан dz по прошедшим конфигурациям укладывается
# в max(1 см, 35% от медианы) — иначе калибровка неопределима по данным.
PLATEAU_ABS_CM = 1.0
PLATEAU_REL = 0.35
# Плато из 1–2 конфигураций — не плато, а точка: разброс 0 тривиален и ничего
# не подтверждает. Единственная прошедшая конфигурация пары 012 (1.4 см)
# совпала с полевым фактом «ямы 1–3 см» (DEPTH_RESULTS.md, ред. 2026-07-15),
# но одиночный проход из 64 не воспроизводим — доказательством не является.
PLATEAU_MIN_CONFIGS = 3
# Ослабленный порог инлаеров для РУЧНЫХ пар (у рига 25): встречный вечерний свет
# инвертирует контраст микротекстуры, выживают только альбедо-фичи (швы, пятна).
# Компенсируется ECC-доводкой + сохранёнными гейтами масштаба/анизотропии +
# требованием стабильного плато по сетке тангажей.
REG_MIN_INLIERS_REAL = 12
_FEAT_CACHE: dict = {}


def register_views_on_ring_robust(bev_f, bev_b, excl_f, excl_b, cfg):
    """Замена photogrammetry.register_views_on_ring для ручных пар.

    AKAZE+SIFT общий пул -> RANSAC-аффин -> ECC-доводка по полосовой яркости.
    Сигнатура/контракт оригинала: (A 2x3, n_inliers, rms) | (None, n, причина).
    Гейты отражения/масштаба/анизотропии — те же, что в оригинале.
    """
    import cv2

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    u8f = clahe.apply(photogrammetry._to_u8(bev_f))
    u8b = clahe.apply(photogrammetry._to_u8(bev_b))
    m_f = (~excl_f).astype(np.uint8) * 255
    m_b = (~excl_b).astype(np.uint8) * 255

    def _features(u8, m):
        """Кэш фич: BEV зависит только от (вид, тангаж) — в сетке 8x8 всего
        16 уникальных BEV, без кэша фичи считались бы 64 раза."""
        key = (u8.shape, hashlib.md5(u8.tobytes()).hexdigest(),
               hashlib.md5(m.tobytes()).hexdigest())
        if key not in _FEAT_CACHE:
            out = []
            for name, det in (("akaze", cv2.AKAZE_create(threshold=5e-4)),
                              ("sift", cv2.SIFT_create())):
                kp, desc = det.detectAndCompute(u8, m)
                out.append((name, [k.pt for k in (kp or [])], desc))
            _FEAT_CACHE[key] = out
        return _FEAT_CACHE[key]

    src_all, dst_all = [], []
    for (name_f, pts_f, df), (name_b, pts_b, db) in zip(_features(u8f, m_f),
                                                        _features(u8b, m_b)):
        norm = cv2.NORM_HAMMING if name_f == "akaze" else cv2.NORM_L2
        if df is None or db is None or len(pts_f) < 8 or len(pts_b) < 8:
            continue
        knn = cv2.BFMatcher(norm).knnMatch(db, df, k=2)
        good = [m for m, n in (p for p in knn if len(p) == 2)
                if m.distance < 0.8 * n.distance]
        src_all += [pts_b[m.queryIdx] for m in good]
        dst_all += [pts_f[m.trainIdx] for m in good]
    if len(src_all) < REG_MIN_INLIERS_REAL:
        return None, len(src_all), "registration_failed:few_matches"
    src = np.float32(src_all).reshape(-1, 1, 2)
    dst = np.float32(dst_all).reshape(-1, 1, 2)
    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                  ransacReprojThreshold=3.0,
                                  maxIters=10000, confidence=0.999)
    n_inl = int(inl.sum()) if inl is not None else 0
    if A is None or n_inl < REG_MIN_INLIERS_REAL:
        return None, n_inl, "registration_failed:ransac"

    # ECC-доводка по полосовой ЯРКОСТИ (альбедо: швы/пятна сохраняют полярность
    # при встречном свете; мм-рельеф — нет, он и так подавлен полосой).
    def _band(x):
        f = x.astype(np.float32)
        return cv2.GaussianBlur(f, (0, 0), 3) - cv2.GaussianBlur(f, (0, 0), 25)

    try:
        # ECC ищет warp координат шаблона(f)->вход(b); инициализация - инверсия A
        A_h = np.vstack([A, [0.0, 0.0, 1.0]])
        init = np.linalg.inv(A_h)[:2].astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        _cc, W = cv2.findTransformECC(_band(bev_f), _band(bev_b),
                                      np.ascontiguousarray(init),
                                      cv2.MOTION_AFFINE, crit,
                                      inputMask=m_f, gaussFiltSize=5)
        A = np.linalg.inv(np.vstack([W, [0.0, 0.0, 1.0]]))[:2]
    except cv2.error:
        pass  # доводка не сошлась — остаёмся на RANSAC-аффине

    sel = inl.ravel().astype(bool)
    pred = src.reshape(-1, 2)[sel] @ A[:, :2].T + A[:, 2]
    rms = float(np.sqrt(np.mean(np.sum(
        (pred - dst.reshape(-1, 2)[sel]) ** 2, axis=1))))
    d2 = float(np.linalg.det(A[:, :2]))
    if d2 <= 0.0:
        return None, n_inl, "registration_failed:reflection"
    sv = np.linalg.svd(A[:, :2], compute_uv=False)
    if abs(float(np.sqrt(d2)) - 1.0) > cfg.reg_scale_tol:
        return None, n_inl, "registration_scale_mismatch"
    if np.max(np.abs(sv - 1.0)) > cfg.reg_aniso_tol:
        return None, n_inl, "registration_distorted"
    return A, n_inl, rms


photogrammetry.register_views_on_ring = register_views_on_ring_robust


def candidate_masks(pipe: DefectPipeline, path: Path):
    """(список масок-кандидатов ямы, статус, изображение).

    'ok' — один доминирующий кандидат (как в пайплайне). При неоднозначности
    (топ-2 в пределах 0.10 уверенности) НЕ пасуем молча, а отдаём обоих
    кандидатов: выбор сделает ГЕОМЕТРИЯ (гейты регистрации/перекрытия дна
    photogrammetry), а не внешний вид — это не re-ID.
    """
    report, _overlay, masks = pipe.analyze_image(path)
    defect, status = _select_dominant_pothole(report["defects"])
    pots = sorted((d for d in report["defects"] if d.get("class") == "pothole"),
                  key=lambda d: d.get("confidence", 0.0), reverse=True)
    if not pots:
        return [], status, None
    if defect is not None:
        chosen = [defect]
    else:  # ambiguous_multiple_potholes: топ-2 кандидата на геометрическую проверку
        chosen = pots[:2]
    img = imgio.read_image(path)
    return [masks[report["defects"].index(d)] for d in chosen], status, img


def run_pair(pipe: DefectPipeline, name: str, path_a: Path, path_b: Path,
             pitch_grid=PITCH_GRID, h_nom: float = H_NOM_M,
             cfg: config.TwoViewDepthConfig = config.DEFAULT_TWO_VIEW) -> dict:
    row: dict = {"pair": name}
    _FEAT_CACHE.clear()
    t0 = time.time()
    cands_f, st_f, img_f = candidate_masks(pipe, path_a)
    cands_b, st_b, img_b = candidate_masks(pipe, path_b)
    row["front_status"], row["back_status"] = st_f, st_b
    if not cands_f or not cands_b:
        row["outcome"] = "no_input_masks"
        return row

    def view(img, pitch):
        h, w = img.shape[:2]
        return RigView(f_px=F_PX, cx=w / 2.0, cy=h / 2.0,
                       cam_height_m=h_nom, pitch_deg=pitch)

    def grid_for(img):
        """Сетка тангажей вида. Кортеж-как-есть, либо 'auto' (режим etalon):
        портретный кадр — почти надир, альбомный — косой обзорный."""
        if pitch_grid != "auto":
            return pitch_grid
        h, w = img.shape[:2]
        return (60.0, 70.0, 80.0, 88.0) if h > w else (15.0, 22.0, 30.0, 40.0, 50.0)

    refusals = Counter()
    combo_pass: dict[tuple[int, int], list] = {}
    for ci, m_f in enumerate(cands_f):
        for cj, m_b in enumerate(cands_b):
            for pf in grid_for(img_f):
                for pb in grid_for(img_b):
                    res = estimate_two_view_depth(
                        img_f, m_f, view(img_f, pf), img_b, m_b,
                        view(img_b, pb), cfg=cfg)
                    if res["method"] == "two_view_plane_parallax":
                        combo_pass.setdefault((ci, cj), []).append(
                            {"pitch_f": pf, "pitch_b": pb,
                             "dz_med": res["depth_cm_median"],
                             "dz_p90": res["depth_cm_p90"],
                             "baseline_m": res["baseline_m"],
                             "n_matches": res["n_matches"],
                             "coverage": round(res.get("coverage", 0.0), 3)})
                    else:
                        refusals[res["method"]] += 1

    row["n_candidates"] = [len(cands_f), len(cands_b)]
    if len(combo_pass) > 1:
        # ≥2 комбинации кандидатов геометрически согласованы — молчаливый выбор
        # недопустим (память проекта: matched ≠ re-ID) → честный отказ.
        row["outcome"] = "candidate_geometry_ambiguous"
        row["combo_details"] = {f"{k}": v for k, v in combo_pass.items()}
        return row
    passing = next(iter(combo_pass.values())) if combo_pass else []
    if combo_pass:
        row["candidate_combo"] = list(next(iter(combo_pass.keys())))
    row["n_passing"] = len(passing)
    row["refusal_counts"] = dict(refusals.most_common())
    row["elapsed_s"] = round(time.time() - t0, 1)
    if not passing:
        row["outcome"] = "all_configs_refused"
        return row

    meds = np.array([p["dz_med"] for p in passing])
    med = float(np.median(meds))
    spread = float(meds.max() - meds.min())
    row["passing"] = passing
    row["dz_med_cm"] = round(med, 2)
    row["dz_range_cm"] = [round(float(meds.min()), 2), round(float(meds.max()), 2)]
    row["plateau_spread_cm"] = round(spread, 2)
    # полная полоса: плато × (высота ± 10%) × (f ± 2%)
    scale = (1.0 + H_BAND_PCT / 100.0) * (1.0 + F_BAND_PCT / 100.0)
    row["dz_band_cm"] = [round(float(meds.min()) / scale, 2),
                         round(float(meds.max()) * scale, 2)]
    if len(passing) < PLATEAU_MIN_CONFIGS:
        row["outcome"] = "insufficient_plateau"
        row["depth_certifiable"] = False
        return row
    stable = spread <= max(PLATEAU_ABS_CM, PLATEAU_REL * max(med, 1e-6))
    row["outcome"] = "estimate" if stable else "unstable_plateau"
    row["depth_certifiable"] = False
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="ID пар (например 006 012); по умолчанию все")
    ap.add_argument("--root", default="datasets/local_pairs")
    ap.add_argument("--out", default=None, help="JSONL результатов")
    ap.add_argument("--mode", choices=("fb", "etalon"), default="fb",
                    help="fb: front/back из local_pairs; etalon: два близких "
                         "кадра из 'Проэкт/.../Яма N/Эталоны' (малая база, "
                         "почти надир: дно видно целиком в обоих видах)")
    args = ap.parse_args()

    if args.mode == "etalon":
        root = Path("Проэкт/Проэкт/Ямки")
        jobs = []
        for d in sorted(root.iterdir(), key=lambda p: (len(p.name), p.name)):
            et = sorted((d / "Эталоны").glob("*.jpg")) if (d / "Эталоны").is_dir() else []
            if len(et) >= 2:
                jobs.append((d.name, et[0], et[1]))
        # сетка тангажей по ориентации кадра (портрет=надир, альбом=косой);
        # база между близкими кадрами бывает < 0.5 м — ослабляем порог
        # вырожденной базы (чувствительность падает, гейты остальные те же)
        pitch_grid, h_nom = "auto", 1.45
        cfg_run = dataclasses.replace(config.DEFAULT_TWO_VIEW,
                                      min_baseline_m=0.25)
    else:
        root = Path(args.root)
        jobs = [(d.name, d / "front.jpg", d / "back.jpg")
                for d in sorted(root.iterdir()) if d.is_dir()]
        pitch_grid, h_nom = PITCH_GRID, H_NOM_M
        cfg_run = config.DEFAULT_TWO_VIEW
    if args.pairs:
        want = set(args.pairs)
        jobs = [j for j in jobs if j[0] in want or j[0].split()[-1].zfill(3) in want]
    out_path = Path(args.out) if args.out else None

    pipe = DefectPipeline(use_depth=False)
    results = []
    for name, pa, pb in jobs:
        row = run_pair(pipe, name, pa, pb, pitch_grid=pitch_grid, h_nom=h_nom,
                       cfg=cfg_run)
        results.append(row)
        band = row.get("dz_band_cm")
        print(f"[{row['pair']}] {row['outcome']}"
              + (f" dz_med={row.get('dz_med_cm')} см, полоса {band[0]}–{band[1]} см,"
                 f" плато {row.get('n_passing')} конфиг., разброс {row.get('plateau_spread_cm')} см"
                 if band else f" ({row.get('front_status')}/{row.get('back_status')}"
                              f" {row.get('refusal_counts', {})})"),
              flush=True)
        if out_path:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    done = [r for r in results if r["outcome"] == "estimate"]
    unst = [r for r in results if r["outcome"] == "unstable_plateau"]
    print(f"\nИтого: {len(results)} пар; оценка получена: {len(done)}; "
          f"нестабильное плато: {len(unst)}; отказ: "
          f"{len(results) - len(done) - len(unst)}")


if __name__ == "__main__":
    main()
