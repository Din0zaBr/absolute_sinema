# report_gen — генераторы демо-отчёта детекции (reference-снимок)

Этими скриптами собран **`docs/detection_report.html`** — одностраничный
прокручиваемый демо-отчёт детекции для руководства (2026-07-07).

- `build_report.py` — **движок**: crop + base64-встраивание JPEG + сборка HTML,
  вся CSS, типы секций `gallery / pairs / beforeafter / diffpairs / missedpairs / notes`,
  рисование рамок на кадре (`full_marked`, `full_dashed`).
- `build_final.py` — **манифест**: данные секций, bbox’ы, подписи, вызов `build()`.

## ⚠️ Это снимок, а не turnkey-скрипт
Скрипты были заточены под временную scratchpad-сессию и ожидают:
- env-переменные `SCR` (temp-папка) и `REPO` (корень репозитория);
- ПРЕДсозданные в `$SCR` промежуточные артефакты, которых в чистой сессии НЕТ:
  `crops/` (зум-кропы пар 006/012/018), `regen_out/` (повторные прогоны CLI ради
  bbox; относится к уже отключённой секции-плитке), `gal_crops/`.

**Source of truth — сам готовый `docs/detection_report.html`** (автономный, с
встроенными картинками). Скрипты держим как эталон структуры/CSS/логики секций,
чтобы не пересобирать отчёт с нуля.

## Данные, откуда тянется отчёт
`outputs_pairs/`, `outputs_demo/`, `datasets/local_pairs/NNN/{front,back}.jpg`,
корневые демо-фото. Per-view bbox’ы получают повторным прогоном CLI на парах
(см. `docs/PAIR_WORKFLOW.md`).

## Правки отчёта малой кровью
1. Мелочь (текст/подписи) — редактировать готовый `docs/detection_report.html`.
2. Структурная пересборка — восстановить промежуточные кропы (прогон CLI на нужных
   `datasets/local_pairs/*`), затем:
   ```bash
   SCR=<temp-dir> REPO=<repo-root> python scripts/report_gen/build_final.py
   ```
   Учесть session-зависимые пути выше (строки `SCR/crops`, `SCR/regen_out`).
