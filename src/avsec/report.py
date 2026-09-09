"""Build a Markdown report from whatever run artifacts exist in a directory.

The report separates, explicitly and by section:

* **[ЗАПУСК]** - a confirmed result of an actual run in this repository;
* **[МОДЕЛЬ]** - a conclusion that holds inside the simulation model only;
* **[ІНЖЕНЕРНЕ ПРИПУЩЕННЯ]** - a design assumption we made;
* **[ГІПОТЕЗА]** - a research hypothesis, not yet established;
* **[НЕ ВИКОНАНО]** - a check that was not performed.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from avsec import evaluation as ev
from avsec.utils import ensure_dir

TAG_RUN = "**[ЗАПУСК]**"
TAG_MODEL = "**[МОДЕЛЬ]**"
TAG_ASSUMPTION = "**[ІНЖЕНЕРНЕ ПРИПУЩЕННЯ]**"
TAG_HYPOTHESIS = "**[ГІПОТЕЗА]**"
TAG_NOTDONE = "**[НЕ ВИКОНАНО]**"


def _load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _table(rows: Sequence[Dict[str, Any]], columns: Sequence[str],
           headers: Optional[Sequence[str]] = None) -> str:
    if not rows:
        return "_(немає даних)_\n"
    head = list(headers or columns)
    out = ["| " + " | ".join(head) + " |",
           "| " + " | ".join("---" for _ in head) + " |"]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append("n/a" if not np.isfinite(v) else f"{v:.3f}")
            elif isinstance(v, bool):
                cells.append("так" if v else "ні")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def build_report(input_dir: str, output_dir: str, title: str = "Звіт avsec") -> str:
    """Collect every artifact found under ``input_dir`` into one Markdown report."""
    ensure_dir(output_dir)
    lines: List[str] = [f"# {title}", ""]

    comparison = _load(os.path.join(input_dir, "comparison.json"))
    demo = _load(os.path.join(input_dir, "demo.json"))
    attacks = _load(os.path.join(input_dir, "attacks.json"))
    tuning = _load(os.path.join(input_dir, "tuning.json"))
    ablations = _load(os.path.join(input_dir, "ablations.json"))
    sweeps = _load(os.path.join(input_dir, "sweeps.json"))
    cvbs = _load(os.path.join(input_dir, "cvbs.json"))
    budget = _load(os.path.join(input_dir, "budget.json"))

    any_run = next((r for r in (comparison, demo, attacks, tuning, ablations, sweeps, cvbs)
                    if r), None)
    if any_run and "record" in any_run:
        rec = any_run["record"]
        env = rec.get("environment", {})
        lines += [
            "## Умови запуску", "",
            f"- конфігурація: `{rec.get('config_id')}`",
            f"- Python {env.get('python')}, {env.get('platform')}",
            f"- пакети: " + ", ".join(f"{k} {v}" for k, v in
                                      sorted(env.get("packages", {}).items())),
            f"- git commit: `{env.get('git_commit') or 'репозиторій git відсутній'}`",
            f"- seed (невідтворювані секрети не входять): "
            f"`{rec.get('config', {}).get('seed')}`",
            f"- ключі: режим `{rec.get('config', {}).get('key_mode')}` "
            "(lab = відтворюваний ключ лише для бенчмарків)",
            "",
        ]

    # -------------------------------------------------------------- budget
    if budget:
        lines += ["## Бюджет каналу та затримка", "",
                  f"{TAG_RUN} Розрахунок за фактичними параметрами профілю.", ""]
        lines += [f"- нестиснений опорний потік: "
                  f"{budget['raw_grayscale_reference_bps']/1e6:.2f} Мбіт/с "
                  f"({budget['raw_grayscale_reference_note']})", ""]
        rows = []
        for name, d in budget.get("methods", {}).items():
            rows.append({
                "method": name,
                "unit_payload": d["unit_plain_bytes"],
                "unit_wire": d["unit_wire_bytes"],
                "units_raster": d.get("units_per_raster_after_placement",
                                      d["units_per_raster"]),
                "payload_bps": round(d["payload_bitrate_bps"] / 1e6, 3),
                "gross_bps": round(d["gross_bitrate_bps"] / 1e6, 3),
                "efficiency": d["payload_efficiency"],
                "accum_rows": d.get("accumulation_rows"),
                "latency_ms": round(d["latency"]["virtual_total_s"] * 1e3, 1),
            })
        lines += [_table(rows, ["method", "unit_payload", "unit_wire", "units_raster",
                                "payload_bps", "gross_bps", "efficiency", "accum_rows",
                                "latency_ms"],
                         ["метод", "корисних Б/одиницю", "Б у каналі/одиницю",
                          "одиниць/растр", "корисна Мбіт/с", "повна Мбіт/с",
                          "ефективність", "накопич. рядків", "модельна затримка, мс"]), ""]
        lines += [f"{TAG_MODEL} Наведена затримка - віртуальний розклад моделі, "
                  "а не вимірювання апаратури.", ""]

    # ---------------------------------------------------------- comparison
    if comparison:
        lines += ["## Порівняння методів", ""]
        if comparison.get("provenance_warning"):
            lines += [f"> **Увага.** {comparison['provenance_warning']}: висновки "
                      "стосуються синтетичного стенда, а не реального відеотракту.", ""]
        lines += [f"{TAG_RUN} Спільний бюджет: "
                  f"{json.dumps(comparison.get('budget', {}), ensure_ascii=False)}", ""]
        rows = []
        for s in comparison.get("summary", []):
            rows.append({
                "method": s["method"],
                "auth": s.get("authenticated"),
                "sequences": s.get("sequences"),
                "psnr": s.get("psnr_mean"),
                "psnr_ci": f"[{s.get('psnr_lo', float('nan')):.2f}; "
                           f"{s.get('psnr_hi', float('nan')):.2f}]",
                "coverage": s.get("coverage_mean"),
            })
        lines += [_table(rows, ["method", "auth", "sequences", "psnr", "psnr_ci",
                                "coverage"],
                         ["метод", "автентифіковано", "послідовностей", "PSNR, дБ",
                          "95% ДІ", "перевірене покриття"]), ""]
        if comparison.get("paired"):
            lines += ["### Парні порівняння (PSNR, за послідовностями)", ""]
            prows = [{"pair": f"{p['a']} - {p['b']}", "mean": p.get("mean"),
                      "ci": f"[{p.get('lo', float('nan')):.2f}; "
                            f"{p.get('hi', float('nan')):.2f}]",
                      "sig": p.get("significant_at_95")}
                     for p in comparison["paired"]]
            lines += [_table(prows, ["pair", "mean", "ci", "sig"],
                             ["пара", "середня різниця, дБ", "95% ДІ",
                              "значуще на рівні 95%"]), ""]
        if comparison.get("errors"):
            lines += ["### Методи, що не вмістилися у спільний бюджет", ""]
            for k, v in comparison["errors"].items():
                lines.append(f"- `{k}`: {v}")
            lines.append("")
        lines += [f"{TAG_ASSUMPTION} Для аналогових базових методів `coverage = 1` "
                  "означає лише «зображення відображено», а не «дані перевірені»: "
                  "у B0a/B1/B2 криптографічної перевірки немає.", ""]

    # ------------------------------------------------------------- attacks
    if attacks:
        lines += ["## Атаки на перестановочні базові методи", "",
                  f"{TAG_RUN} Параметри LFSR - наша реконструкція: "
                  f"`{json.dumps(attacks.get('lfsr', {}).get('lfsr', {}), ensure_ascii=False)}`",
                  ""]
        rows = []
        for r in attacks.get("results", []):
            m = r.get("metrics", {})
            rows.append({
                "attack": r["name"], "content": r.get("content", "-"),
                "success": r.get("success"),
                "acc": m.get("permutation_accuracy", m.get("direct_accuracy", float("nan"))),
                "neigh": m.get("neighbour_accuracy", float("nan")),
                "psnr": m.get("reconstruction_psnr_db", float("nan")),
                "sec": round(r.get("seconds", 0.0), 3),
            })
        lines += [_table(rows, ["attack", "content", "success", "acc", "neigh", "psnr",
                                "sec"],
                         ["атака", "вміст", "успіх", "точність перестановки",
                          "суміжність", "PSNR відновлення, дБ", "с"]), ""]
        lines += [f"> {attacks.get('caveat', '')}", ""]
        if attacks.get("similarity_table"):
            lines += ["### «Схожість» у сенсі статті 2021 року", "",
                      f"{TAG_RUN} Кореляція оригіналу зі скремблованим і з відновленим "
                      "зображенням, |r|·100%.", ""]
            lines += [_table(attacks["similarity_table"],
                             ["seed", "content", "similarity_scrambled_pct",
                              "similarity_decrypted_pct", "exact_recovery"],
                             ["seed", "вміст", "оригінал vs скрембл, %",
                              "оригінал vs відновлено, %", "точне відновлення"]), ""]
            lines += [f"{TAG_MODEL} Низька кореляція між кадрами не є доказом "
                      "конфіденційності: атака за сумісністю меж і атака за відомим "
                      "кадром працюють попри неї.", ""]
        if attacks.get("protocol_checks"):
            lines += ["### Перевірки протоколу AEAD", ""]
            lines += [_table(attacks["protocol_checks"],
                             ["check", "expected", "outcome", "passed"],
                             ["перевірка", "очікувано", "результат", "пройдено"]), ""]

    # -------------------------------------------------------------- tuning
    if tuning:
        lines += ["## Добір параметрів", "",
                  f"{TAG_RUN} {tuning.get('discipline', '')}", "",
                  f"- кандидатів: {tuning.get('n_candidates')}, "
                  f"допустимих: {tuning.get('n_admissible')}",
                  f"- розподіл даних: "
                  f"{json.dumps(tuning.get('splits', {}), ensure_ascii=False)}", ""]
        if tuning.get("validation"):
            lines += [_table(tuning["validation"],
                             ["key", "psnr_full", "calibration_psnr", "coverage",
                              "rasters_per_frame", "virtual_latency_s"],
                             ["конфігурація", "PSNR (валідація)", "PSNR (калібрування)",
                              "покриття", "растрів/кадр", "модельна затримка, с"]), ""]
        if tuning.get("selected"):
            lines += [f"**Обрана конфігурація:** `{tuning['selected'].get('key')}`", ""]

    # ----------------------------------------------------------- ablations
    if ablations:
        lines += ["## Абляції запропонованого методу", "",
                  f"{TAG_RUN} {ablations.get('note', '')}", ""]
        lines += [_table(ablations.get("rows", []),
                         ["variant", "admissible", "psnr_mean", "coverage_mean",
                          "rasters_mean", "reason"],
                         ["варіант", "допустимий", "PSNR, дБ", "покриття",
                          "растрів/кадр", "причина відхилення"]), ""]

    # -------------------------------------------------------------- sweeps
    if sweeps:
        lines += ["## Залежність від сили спотворень", ""]
        lines += [_table(sweeps.get("rows", []),
                         ["channel", "method", "psnr_mean", "coverage_mean",
                          "rasters_mean", "n"],
                         ["канал", "метод", "PSNR, дБ", "покриття", "растрів/кадр",
                          "кадрів"]), ""]
        for name, path in (sweeps.get("plots") or {}).items():
            lines += [f"![{name}]({os.path.relpath(path, output_dir).replace(os.sep, '/')})",
                      ""]

    # ---------------------------------------------------------------- CVBS
    if cvbs:
        lines += ["## Наскрізна демонстрація через модель CVBS (рівень B)", "",
                  f"{TAG_RUN} Профіль каналу: `{cvbs.get('preset')}`, метод "
                  f"`{cvbs.get('method')}`.", "",
                  f"- якість: PSNR {cvbs.get('quality', {}).get('psnr_full', float('nan')):.2f} дБ, "
                  f"перевірене покриття "
                  f"{cvbs.get('quality', {}).get('coverage', 0):.3f}",
                  f"- статуси одиниць: "
                  f"{json.dumps(cvbs.get('status_counts', {}), ensure_ascii=False)}", ""]
        for st in cvbs.get("cvbs_stats", []):
            lines.append(f"- растр: відновлено рядків "
                         f"{st.get('lines_recovered')}/{st.get('lines_expected')}, "
                         f"полів {st.get('fields_found')}, "
                         f"фронтів синхронізації {st.get('sync_edges_found')}")
        lines += ["", f"{TAG_ASSUMPTION} {cvbs.get('compliance_note', '')}", ""]
        prof = cvbs.get("profile", {})
        if prof.get("simplifications"):
            lines += ["Спрощення моделі CVBS:", ""]
            lines += [f"- {s}" for s in prof["simplifications"]]
            lines.append("")

    # ---------------------------------------------------------------- demo
    if demo:
        lines += ["## Демонстрація на одному кадрі", ""]
        lines += [_table(demo.get("rows", []),
                         ["method", "psnr_full", "ssim_full", "coverage",
                          "units_verified", "units_sent", "rasters"],
                         ["метод", "PSNR, дБ", "SSIM", "покриття", "перевірено одиниць",
                          "надіслано одиниць", "растрів"]), ""]
        for method, imgs in (demo.get("images") or {}).items():
            lines.append(f"### {method}")
            for kind, path in imgs.items():
                rel = os.path.relpath(path, output_dir).replace(os.sep, "/")
                lines.append(f"- {kind}: ![{method}-{kind}]({rel})")
            lines.append("")

    # ------------------------------------------------------------ closing
    lines += [
        "## Що НЕ підтверджено цим звітом", "",
        f"{TAG_NOTDONE} Вимірювання на фізичному передавачі/приймачі "
        "(TS5828, video grabber, Raspberry Pi) не проводилися.",
        f"{TAG_NOTDONE} Радіочастотний FM-тракт не змодельовано; результати рівня B "
        "стосуються лише композитного відеосигналу в основній смузі.",
        f"{TAG_NOTDONE} Енергоспоживання, вартість і маса апаратури не вимірювалися.",
        f"{TAG_HYPOTHESIS} Твердження про перевагу узгодженого добору параметрів "
        "дійсне лише в межах перевірених конфігурацій і моделей каналу; "
        "за їх межами воно не встановлене.",
        f"{TAG_MODEL} Правильна інтеграція AEAD, підтверджена перевірками, не доводить "
        "відсутності інших помилок протоколу.",
        "",
    ]
    text = "\n".join(lines)
    out_path = os.path.join(output_dir, "report.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out_path


__all__ = ["build_report"]
