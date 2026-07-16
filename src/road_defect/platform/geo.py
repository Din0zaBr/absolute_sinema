"""Геопривязка отчётов: разбор координат журнала и EXIF-GPS.

Три независимых источника координат (в порядке доверия): location из JSON-отчёта,
журнал ям (datasets/journal.csv, колонка street_or_gps «lat lon»), EXIF снимка.
Модуль не выдумывает координаты: любой мусор/ошибка → None, не «(0, 0)» —
точка на нулевом меридиане у Гвинейского залива хуже честного отсутствия.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

# Числа с необязательным знаком и дробной частью; разделители (пробел/запятая/
# точка с запятой) и посторонний текст вокруг допустимы — журнал заполняют люди.
_FLOAT_RE = re.compile(r"[-+]?\d{1,3}(?:\.\d+)?")

# Теги GPS IFD (EXIF): номера стабильны по спецификации EXIF 2.3.
_GPS_LAT_REF, _GPS_LAT = 1, 2
_GPS_LON_REF, _GPS_LON = 3, 4
_GPS_IFD_POINTER = 0x8825


def parse_latlon(text: str | None) -> tuple[float, float] | None:
    """Разобрать «lat lon» из свободного текста журнала.

    Терпит запятую-разделитель («47.27, 39.70») и мусор вокруг, но требует
    РОВНО двух чисел в допустимых диапазонах: три и более числа — это уже не
    координаты, а что-то другое (адрес с номером дома), угадывать нельзя.
    """
    if not text:
        return None
    nums = _FLOAT_RE.findall(text)
    if len(nums) != 2:
        return None
    try:
        lat, lon = float(nums[0]), float(nums[1])
    except ValueError:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def journal_latlon(journal_csv: Path, pair_id: str) -> tuple[float, float] | None:
    """Координаты пары по журналу ям (колонки id, street_or_gps).

    id в журнале трёхзначный («006»), а в отчётах/именах может быть «6» —
    ищем и как есть, и с zfill(3). Любая проблема файла → None.
    utf-8-sig: реальный datasets/journal.csv начинается с BOM, и с чистым
    utf-8 заголовок первой колонки был бы «\\ufeffid» — id бы не находился.
    """
    journal_csv = Path(journal_csv)
    if not journal_csv.is_file():
        return None
    wanted = {str(pair_id).strip()}
    wanted.add(str(pair_id).strip().zfill(3))
    try:
        with journal_csv.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if (row.get("id") or "").strip() in wanted:
                    return parse_latlon(row.get("street_or_gps"))
    except (OSError, csv.Error, UnicodeDecodeError):
        return None
    return None


def _dms_to_deg(dms, ref: str) -> float:
    """Градусы/минуты/секунды EXIF → десятичные градусы со знаком по N/S/E/W.

    PIL отдаёт компоненты как IFDRational — приводим через float(), иначе
    арифметика может дать Fraction-подобный тип, который не сериализуется.
    """
    parts = [float(v) for v in dms]
    while len(parts) < 3:
        parts.append(0.0)
    deg = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    if str(ref).strip().upper() in ("S", "W"):
        deg = -deg
    return deg


def exif_gps(image_path: Path) -> tuple[float, float] | None:
    """Координаты из EXIF снимка (GPS IFD). Любая ошибка → None.

    Смартфонные EXIF бывают битыми (обрезанный IFD, нулевые рационалы) —
    ловим всё подряд: отсутствие координат честнее падения загрузки.
    """
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            gps = im.getexif().get_ifd(_GPS_IFD_POINTER)
        if not gps:
            return None
        lat_dms, lat_ref = gps.get(_GPS_LAT), gps.get(_GPS_LAT_REF)
        lon_dms, lon_ref = gps.get(_GPS_LON), gps.get(_GPS_LON_REF)
        if lat_dms is None or lon_dms is None:
            return None
        lat = _dms_to_deg(lat_dms, lat_ref or "N")
        lon = _dms_to_deg(lon_dms, lon_ref or "E")
    except Exception:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon
