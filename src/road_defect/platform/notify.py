"""Уведомления по дефектам (подсистема 5): правила поверх плоской проекции БД.

Правила намеренно консервативные и честные: уведомление — это «требуется
внимание/выезд», а не «система измерила нарушение». Оценка глубины (двух-
видовая или иная) в тексте всегда помечена словом «оценка» — по одному фото
сертифицированной глубины не существует (принцип честности, CLAUDE.md).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .. import config

# Русские подписи классов — используются и здесь, и в шаблонах UI.
CLASS_LABELS_RU = {
    "pothole": "яма",
    "patch": "заплатка",
    "alligator_crack": "сетка трещин",
    "longitudinal_crack": "продольная трещина",
    "transverse_crack": "поперечная трещина",
}


def rules_for_defect(row: dict) -> list[tuple[str, str]]:
    """Список (rule, message) для одной строки defects.

    - gost_non_conforming: движок вынес вердикт «yes» (все три условия ГОСТ
      измерены и превышены) — это самый сильный сигнал.
    - deep_pothole_suspect: яма с бакетом deep ИЛИ оценкой глубины, чья верхняя
      граница достигает порога ГОСТ — подозрение, не вердикт: требуется выезд.
    """
    rules: list[tuple[str, str]] = []
    label = CLASS_LABELS_RU.get(row.get("class") or "", row.get("class") or "дефект")

    if row.get("non_conforming") == "yes":
        note = row.get("severity_note") or ""
        rules.append((
            "gost_non_conforming",
            f"Несоответствие ГОСТ Р 50597-2017: {label}. {note}".strip()))

    if row.get("class") == "pothole":
        # Верхняя граница оценки честнее точечного значения: если ДАЖЕ она
        # ниже порога, подозрения нет; если достигает — нужен замер на месте.
        est = row.get("depth_est_high_cm")
        if est is None:
            est = row.get("depth_est_cm")
        method = row.get("depth_est_method") or ""
        deep_bucket = row.get("depth_bucket") == "deep"
        deep_estimate = est is not None and est >= config.GOST50597_MAX_DEPTH_CM
        if deep_bucket or deep_estimate:
            if deep_estimate and "upper_bound" in method:
                # Ревью 2026-07-15: верхняя граница below-noise — предел
                # разрешимости карты, а не оценка глубины; «может достигать X»
                # терял эту семантику и провоцировал ложные выезды.
                detail = (f"просадка ниже разрешимости карты, но оценка не "
                          f"исключает глубину до {est:.1f} см (порог ГОСТ "
                          f"{config.GOST50597_MAX_DEPTH_CM:g} см)")
            elif deep_estimate:
                detail = (f"по оценке глубина может достигать {est:.1f} см "
                          f"(порог ГОСТ {config.GOST50597_MAX_DEPTH_CM:g} см)")
            else:
                detail = "относительная глубина в бакете «deep»"
            rules.append((
                "deep_pothole_suspect",
                f"Подозрение на глубокую яму: {detail}. Это оценка, не измерение — "
                "требуется выезд и контрольный замер."))
    return rules


def create_notifications(conn: sqlite3.Connection) -> int:
    """Создать уведомления по всем дефектам. Идемпотентно: INSERT OR IGNORE
    по UNIQUE(defect_id, rule) — повторный вызов не плодит дубликатов.
    Возвращает число НОВЫХ уведомлений."""
    created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = 0
    rows = conn.execute("SELECT * FROM defects").fetchall()
    for row in rows:
        for rule, message in rules_for_defect(dict(row)):
            cur = conn.execute(
                "INSERT OR IGNORE INTO notifications"
                " (report_id, defect_id, rule, message, created_utc)"
                " VALUES (?, ?, ?, ?, ?)",
                (row["report_id"], row["id"], rule, message, created_utc))
            new += cur.rowcount
    conn.commit()
    return new
