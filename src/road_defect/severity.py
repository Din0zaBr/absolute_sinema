"""Классификация серьёзности дефекта по ГОСТ Р 50597-2017.

Чистый модуль. Ключевая честность: нога «глубина ≥ 5 см» решается только при
наличии измеренной глубины (Сценарий C / датчик). Без неё вердикт по
несоответствию — `indeterminate_without_depth`, а не «соответствует».
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from . import config


@dataclass
class SeverityVerdict:
    standard: str
    length_exceeds: bool | None      # длина ≥ 15 см
    area_exceeds: bool | None        # площадь ≥ 0.06 м²
    depth_exceeds: bool | None       # глубина ≥ 5 см (None = не измерена)
    non_conforming: str              # "yes" | "no" | "indeterminate_without_depth"
    repair_deadline_days: int | None
    hazard_signing_required: bool | None
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def classify(
    length_cm: float | None,
    area_m2: float | None,
    depth_cm: float | None,
    road_category: str = "IV",
) -> SeverityVerdict:
    """Вердикт по ГОСТ Р 50597-2017.

    Дефект несоответствует норме при ВСЕХ трёх условиях:
    длина ≥ 15 см И глубина ≥ 5 см И площадь ≥ 0.06 м².
    """
    length_exc = None if length_cm is None else length_cm >= config.GOST50597_MAX_LENGTH_CM
    area_exc = None if area_m2 is None else area_m2 >= config.GOST50597_MAX_AREA_M2
    depth_exc = None if depth_cm is None else depth_cm >= config.GOST50597_MAX_DEPTH_CM

    deadline = config.GOST50597_REPAIR_DEADLINE_DAYS.get(road_category)

    # Все три измерены → однозначный вердикт.
    if None not in (length_exc, area_exc, depth_exc):
        if length_exc and area_exc and depth_exc:
            return SeverityVerdict(
                standard="ГОСТ Р 50597-2017",
                length_exceeds=length_exc, area_exceeds=area_exc, depth_exceeds=depth_exc,
                non_conforming="yes",
                repair_deadline_days=deadline,
                hazard_signing_required=True,
                note=(f"Несоответствие норме (длина/глубина/площадь превышены). "
                      f"Срок устранения для кат. {road_category}: {deadline} сут; "
                      f"обозначить/оградить в течение {config.GOST50597_HAZARD_SIGNING_HOURS} ч."),
            )
        return SeverityVerdict(
            standard="ГОСТ Р 50597-2017",
            length_exceeds=length_exc, area_exceeds=area_exc, depth_exceeds=depth_exc,
            non_conforming="no", repair_deadline_days=None, hazard_signing_required=False,
            note="В пределах допустимого по совокупности условий.",
        )

    # Глубина не измерена (типичный одиночный кадр) — честно «не определено».
    if depth_exc is None:
        partial = []
        if length_exc is not None:
            partial.append(f"длина {'превышает' if length_exc else 'в норме'}")
        if area_exc is not None:
            partial.append(f"площадь {'превышает' if area_exc else 'в норме'}")
        return SeverityVerdict(
            standard="ГОСТ Р 50597-2017",
            length_exceeds=length_exc, area_exceeds=area_exc, depth_exceeds=None,
            non_conforming="indeterminate_without_depth",
            repair_deadline_days=None, hazard_signing_required=None,
            note=("Глубину нельзя подтвердить по этому входу — нужна серийная съёмка. "
                  + ("; ".join(partial) if partial else "размеры не измерены (нет эталона).")),
        )

    # Размеры не измерены (нет эталона), но глубина есть — редкий случай.
    return SeverityVerdict(
        standard="ГОСТ Р 50597-2017",
        length_exceeds=length_exc, area_exceeds=area_exc, depth_exceeds=depth_exc,
        non_conforming="indeterminate_without_depth",
        repair_deadline_days=None, hazard_signing_required=None,
        note="Недостаточно измерений для полного вердикта (нет эталона масштаба).",
    )
