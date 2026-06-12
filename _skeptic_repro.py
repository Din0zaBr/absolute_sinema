# -*- coding: utf-8 -*-
"""Skeptic repro: does describe_mask understate fragmented crack masks
that the real crack extractor produces?"""
import sys
sys.path.insert(0, r"C:\Users\Kordon\doroga_govna\src")

import numpy as np
import cv2

from road_defect.shape import describe_mask, apply_scale
from road_defect.segment import Segmenter
from road_defect import config

print("=== Part 1: reviewer's synthetic two-fragment mask ===")
mask = np.zeros((200, 500), bool)
mask[100:103, 10:110] = True    # fragment 3x100 = 300 px
mask[100:103, 150:400] = True   # fragment 3x250 = 750 px
sd = describe_mask(mask)
print("mask.sum() =", int(mask.sum()))
print("area_px2   =", sd.area_px2)
print("major_axis_px =", round(sd.major_axis_px, 1))
print("bbox_px    =", sd.bbox_px, "(full defect spans cols 10..400 = 390 px)")
scaled = apply_scale(sd, mm_per_px=2.0)
print("length_cm @2mm/px =", scaled["length_cm"],
      " (full-extent would be ~", round(390 * 2.0 / 10.0, 1), "cm)")
print("area_m2 =", scaled["area_m2"],
      " (sum-based would be", round(int(mask.sum()) * 4.0 / 1e6, 4), ")")

print()
print("=== Part 2: realistic dashed crack through the REAL extractor ===")
# Asphalt-like noise + dark crack drawn as dashes (crack partially filled /
# interrupted -- common on real photos).
rng = np.random.default_rng(7)
base = rng.integers(125, 145, (300, 400), np.uint8)
# dashed dark line y~140, x from 40 to 360, dashes 45 px with 25 px gaps
for x0 in range(40, 360, 70):
    x1 = min(x0 + 45, 360)
    cv2.line(base, (x0, 142), (x1, 140), 55, 3)
bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

m, method = Segmenter._crack_mask(bgr, (30, 100, 340, 80))
if m is None:
    print("extractor returned None -> repro FAILED on this scene")
else:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        m.astype(np.uint8), connectivity=8)
    comp_areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)
    print("method =", method)
    print("components =", n - 1, "areas =", comp_areas)
    print("mask.sum() =", int(m.sum()))
    sd2 = describe_mask(m)
    print("area_px2 (largest comp only) =", sd2.area_px2)
    print("major_axis_px =", round(sd2.major_axis_px, 1))
    ys, xs = np.nonzero(m)
    full_extent = xs.max() - xs.min() + 1
    print("full mask x-extent =", full_extent, "px")
    min_area = config.DEFAULT_INFERENCE.mask_min_area_linear_px
    print("passes pipeline min_area filter (sum >= %d)?" % min_area,
          int(m.sum()) >= min_area)
    sc = apply_scale(sd2, mm_per_px=2.0)
    print("reported length_cm @2mm/px =", sc["length_cm"],
          "vs full-extent ~", round(full_extent * 2.0 / 10.0, 1), "cm")
    print("reported area_m2 =", sc["area_m2"],
          "vs sum-based", round(int(m.sum()) * 4.0 / 1e6, 4))

print()
print("=== Part 3: GOST area_exceeds flip ===")
# Fragmented defect whose TOTAL area exceeds 0.06 m2 but largest fragment doesn't.
# At 5 mm/px: 0.06 m2 = 60000 mm2 = 2400 px. Two fragments of 1500 px each.
mask3 = np.zeros((300, 600), bool)
mask3[100:130, 50:100] = True    # 30x50 = 1500 px
mask3[150:180, 200:250] = True   # 30x50 = 1500 px
sd3 = describe_mask(mask3)
sc3 = apply_scale(sd3, mm_per_px=5.0)
total_m2 = int(mask3.sum()) * 25.0 / 1e6
print("total area_m2 =", total_m2, "(>= 0.06:", total_m2 >= 0.06, ")")
print("reported area_m2 =", sc3["area_m2"],
      "(area_exceeds:", sc3["area_m2"] >= 0.06, ")")
