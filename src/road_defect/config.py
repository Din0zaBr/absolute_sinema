"""Единая точка правды: константы ГОСТ, классы дефектов, пороги, реестр моделей.

Числа ГОСТ подтверждены ресёрчем (см. дизайн §6). Не дублировать эти значения по
коду — импортировать отсюда.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Пути -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# --- Классы дефектов --------------------------------------------------------
# RDD2022-таксономия + наши прикладные классы. Эталоны (люк/разметка) детектятся
# отдельно и в этот список «дефектов» не входят.
DEFECT_CLASSES = {
    "D00": "longitudinal_crack",   # продольная трещина
    "D10": "transverse_crack",     # поперечная трещина
    "D20": "alligator_crack",      # сетка трещин
    "D40": "pothole",              # выбоина / яма
    "Repair": "patch",             # заделка / ямочный ремонт
}

# Классы, для которых имеет смысл мерить «диаметр/глубину» (площадные дефекты).
AREAL_DEFECT_CLASSES = {"pothole", "patch", "alligator_crack"}


# --- ГОСТ 3634: люк как эталон масштаба -------------------------------------
# Стабилен только лаз 600 мм (суффикс «-60», одинаков для Л(А15) и Т(С250)).
GOST3634_CLEAR_OPENING_MM = 600       # лаз — первичный якорь масштаба
GOST3634_COVER_OUTER_MM = 646         # обод крышки (сверять с изданием!)
GOST3634_BASE_DIAMETER_MM = (750, 840)  # опорная база по классу нагрузки


# --- ГОСТ Р 51256-2018: разметка как вторичный эталон -----------------------
# Зависит от категории дороги; допуск ±0.01 м.
GOST51256_LINE_1_1_MM = 100           # узкая продольная (0.08 кат IV; 0.15 кат IA)
GOST51256_EDGE_LINE_MM = (100, 200)   # краевая 1.2 по категории
GOST51256_STOP_LINE_MM = 400          # СТОП-линия 1.12


# --- ГОСТ Р 50597-2017: пороги серьёзности (п. 5.2.4 / табл. 5.3) -----------
# Одиночный дефект НЕ соответствует норме при выполнении ВСЕХ трёх условий.
GOST50597_MAX_LENGTH_CM = 15.0        # предельная длина
GOST50597_MAX_DEPTH_CM = 5.0          # предельная глубина
GOST50597_MAX_AREA_M2 = 0.06          # предельная площадь

# Сроки устранения одиночного дефекта по категории дороги, сутки (табл. 5.3).
GOST50597_REPAIR_DEADLINE_DAYS = {
    "IA": 1, "IB": 3, "IV_cat": 3, "II": 5, "III": 7, "IV": 10, "V": 12,
}
# Обозначить/оградить опасный участок — в течение 2 часов (п. 4.4).
GOST50597_HAZARD_SIGNING_HOURS = 2


# --- Реестр моделей (приоритетные кандидаты для скачивания) -----------------
@dataclass
class ModelCandidate:
    """Кандидат весов: откуда тянуть и как опознать."""
    name: str
    source: str          # 'hf_hub' | 'ultralytics' | 'local'
    repo_id: str = ""
    filename: str = ""
    note: str = ""


# Детекторы дорожных дефектов — пробуем по порядку, берём первый загрузившийся.
DETECTOR_CANDIDATES = [
    ModelCandidate(
        name="rezzzq-yolo12s-rdd2022", source="hf_hub",
        repo_id="rezzzq/yolo12s-road-damage-rdd2022", filename="best.pt",
        note="5 классов вкл. Repair; полный RDD2022 (ближе к РФ)",
    ),
    ModelCandidate(
        name="keremberke-yolov8m-pothole-seg", source="hf_hub",
        repo_id="keremberke/yolov8m-pothole-segmentation", filename="best.pt",
        note="instance-seg маски ям",
    ),
    # Фолбэк: COCO-сегментация (нет классов дефектов) — только чтобы доказать
    # работоспособность конвейера сегментации/формы/overlay.
    ModelCandidate(
        name="yolo11n-seg-coco", source="ultralytics",
        filename="yolo11n-seg.pt", note="фолбэк COCO; не детектит ямы",
    ),
]

# Сегментатор формы (box-prompt).
SEGMENTER = ModelCandidate(
    name="mobile_sam", source="hf_hub",
    repo_id="dhkim2810/MobileSAM", filename="mobile_sam.pt",
    note="Apache-2.0; box-prompt; CPU-friendly",
)

# Относительная глубина (опционально).
DEPTH_MODEL = ModelCandidate(
    name="depth-anything-v2-small", source="hf_hub",
    repo_id="depth-anything/Depth-Anything-V2-Small",
    filename="depth_anything_v2_vits.pth",
    note="Apache-2.0 (только Small)",
)


# --- Пороги инференса -------------------------------------------------------
@dataclass
class InferenceConfig:
    det_conf: float = 0.25            # порог уверенности детектора
    det_iou: float = 0.45             # NMS IoU
    imgsz: int = 1024                 # длинная сторона при инференсе
    # Hough по люку гоняется на копии с длинной стороной <= imgsz, поэтому
    # радиусы задаются долей короткой стороны кадра, а не пикселями.
    manhole_min_radius_frac: float = 0.03   # мин. радиус люка (доля кадра)
    manhole_max_radius_frac: float = 0.45   # макс. радиус люка (доля кадра)
    depth_buckets_cm: tuple = (2.0, 5.0)  # границы мелкая|средняя|глубокая (относит.)
    mask_min_area_px: int = 200       # отбраковка крошечных масок


DEFAULT_INFERENCE = InferenceConfig()
