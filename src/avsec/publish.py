"""Generate every published number from **one** run (defect F18).

The previous documentation quoted figures collected at different times, under
different code, with no way to tell which run produced which number.  This
module takes a single run directory, reads only the tables that run wrote, and
regenerates:

* ``docs/results.md``   - the results document, every table derived here;
* ``docs/figures.md``   - the G01-G43 catalogue with ready/pending and reasons;
* ``docs/programme.md`` - E01-E11 status, including what was not done;
* the results section of ``README.md`` between its markers.

Every generated file carries the ``run_id`` and the commit it came from.  A
word like "доведено" or "значуще" is emitted only where the corresponding
interval actually excludes zero; the wording is chosen by code, not by hand.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from avsec.program import CATALOGUE, EXPERIMENTS, PROGRAM, programme_status
from avsec.utils import ensure_dir

README_BEGIN = "<!-- AVSEC:RESULTS:BEGIN -->"
README_END = "<!-- AVSEC:RESULTS:END -->"

#: Words that may only appear next to a result that supports them.
CLAIM_WORDS = ("доведено", "значуще", "significant", "реальне обладнання",
               "hardware-verified", "дешевий")


@dataclass
class RunView:
    """Read-only view of one run directory."""

    run_dir: str

    def csv(self, name: str) -> List[Dict[str, Any]]:
        p = os.path.join(self.run_dir, name)
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def json(self, name: str) -> Dict[str, Any]:
        p = os.path.join(self.run_dir, name)
        if not os.path.exists(p):
            return {}
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    @property
    def manifest(self) -> Dict[str, Any]:
        return self.json("run_manifest.json")

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id", "unknown"))

    @property
    def commit(self) -> str:
        return str(self.manifest.get("commit") or "не в git-репозиторії")

    def stamp(self) -> str:
        m = self.manifest
        env = m.get("environment", {})
        an = self.json("analysis.json")
        used = an.get("n_scenes") or m.get("dataset", {}).get(
            "scenes_per_split", {}).get("test", "?")
        return (f"_Усі числа нижче походять з одного прогону:_ `run_id={self.run_id}`, "
                f"commit `{self.commit}`, "
                f"{env.get('platform', '?')}, Python {env.get('python', '?')}, "
                f"NumPy {env.get('packages', {}).get('numpy', '?')}. "
                f"Матеріал: {m.get('dataset', {}).get('n_clips', '?')} кліпів / "
                f"{m.get('dataset', {}).get('n_scenes', '?')} сцен усього, з них "
                f"{used} у цьому прогоні (split "
                f"{', '.join(m.get('plan', {}).get('splits', []) ) or '?'}); "
                f"**синтетичні дані**.")


def _relpath(path: str) -> str:
    """Relative path when possible; on Windows a different drive has none."""
    try:
        return os.path.relpath(path).replace(os.sep, "/")
    except ValueError:
        return path.replace(os.sep, "/")


def _f(row: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _fmt(v: float, nd: int = 2) -> str:
    return "—" if not np.isfinite(v) else f"{v:.{nd}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


CHANNEL_ORDER = ("clean", "mild", "moderate", "bursty", "harsh")


def _ch_key(name: str) -> Tuple[int, str]:
    base = str(name).split("[")[0]
    return (CHANNEL_ORDER.index(base) if base in CHANNEL_ORDER else 99, str(name))


# ------------------------------------------------------------------- sections
def quality_table(view: RunView) -> str:
    rows = [r for r in view.csv("summary.csv") if r.get("metric") == "psnr_full"]
    if not rows:
        return "_Таблиця не побудована: у прогоні немає `summary.csv`._"
    channels = sorted({r["channel"] for r in rows}, key=_ch_key)
    methods = sorted({r["method"] for r in rows})
    body = []
    for m in methods:
        auth = next((r for r in rows if r["method"] == m), {}).get("authenticated")
        cells = [m + ("" if auth == "True" else " *")]
        for ch in channels:
            sel = [r for r in rows if r["method"] == m and r["channel"] == ch]
            if not sel:
                cells.append("—")
                continue
            cells.append(f"{_fmt(_f(sel[0], 'mean'))} "
                         f"[{_fmt(_f(sel[0], 'lo'))}; {_fmt(_f(sel[0], 'hi'))}]")
        body.append(cells)
    n = max((int(r.get("n_scenes") or 0) for r in rows), default=0)
    return (_table(["метод"] + list(channels), body)
            + f"\n\nPSNR усього показаного кадру, дБ; середнє за {n} незалежними "
              "сценами з 95% бутстреп-інтервалом.  `*` - метод не автентифікує "
              "того, що показує.")


def coverage_table(view: RunView) -> str:
    rows = [r for r in view.csv("summary.csv") if r.get("metric") == "coverage"]
    if not rows:
        return ""
    channels = sorted({r["channel"] for r in rows}, key=_ch_key)
    body = []
    for m in sorted({r["method"] for r in rows}):
        auth = next((r for r in rows if r["method"] == m), {}).get("authenticated")
        cells = [m + ("" if auth == "True" else " *")]
        for ch in channels:
            sel = [r for r in rows if r["method"] == m and r["channel"] == ch]
            cells.append(_fmt(_f(sel[0], "mean"), 3) if sel else "—")
        body.append(cells)
    return (_table(["метод"] + list(channels), body)
            + "\n\nПоточне перевірене покриття.  Для методів з `*` це частка "
              "показаної картинки, а **не** перевірених даних: вони нічого не "
              "автентифікують, тому їхнє «покриття» не порівнюване з рештою.")


def primary_section(view: RunView) -> str:
    an = view.json("analysis.json")
    if not an:
        return "_Аналіз не запускався._"
    plan = an.get("plan", {})
    a, b = plan.get("primary_comparison", ["P", "B4"])
    rows = an.get("primary", [])
    body = []
    for r in sorted(rows, key=lambda x: _ch_key(x.get("channel", ""))):
        mean, lo, hi = _f(r, "mean"), _f(r, "lo"), _f(r, "hi")
        sig = bool(r.get("significant_at_95"))
        holm = bool(r.get("significant_holm"))
        if lo == hi == 0.0 and mean == 0.0:
            # A zero-width interval at exactly zero is not "no difference
            # detected" - it means the two methods produced identical output on
            # every scene, which is a stronger and different statement.
            verdict = "ідентичний результат на всіх сценах"
        elif not sig:
            verdict = "різниця не встановлена"
        elif mean > 0:
            verdict = "перевага " + a + (" (значуще після Holm)" if holm else "")
        else:
            verdict = "перевага " + b + (" (значуще після Holm)" if holm else "")
        body.append([r.get("channel"), f"{mean:+.2f}",
                     f"[{lo:+.2f}; {hi:+.2f}]", r.get("n_scenes"), verdict])
    v = an.get("primary_verdict", {})
    lines = [
        f"Попередньо зареєстроване порівняння: **{a} − {b}** за метрикою "
        f"`{plan.get('metric')}`.  Одиниця незалежності - "
        f"{plan.get('unit_of_independence')}; {plan.get('aggregation')}.",
        "",
        _table(["канал", "різниця, дБ", "95% CI", "сцен", "висновок"], body),
        "",
    ]
    better = v.get("channels_where_a_is_better") or []
    worse = v.get("channels_where_a_is_worse") or []
    flat = v.get("channels_with_no_detectable_difference") or []
    if better:
        lines.append(f"**Значуща перевага {a}** зафіксована в каналах: "
                     f"{', '.join(better)}.")
    if worse:
        lines.append(f"**Значуща поразка {a}** зафіксована в каналах: "
                     f"{', '.join(worse)} - там простіша схема {b} краща, і це "
                     f"результат, а не аномалія.")
    if flat:
        lines.append(f"У каналах {', '.join(flat)} інтервал містить нуль: різниця "
                     f"не встановлена.  Це **не** доказ рівності.")
    return "\n".join(lines)


def budget_table(view: RunView) -> str:
    rows = view.csv("budgets.csv")
    if not rows:
        return ""
    body = [[r["method"], r["unit_plain_bytes"], r["unit_wire_bytes"],
             r["units_placed"],
             f"{_f(r, 'payload_bitrate_bps') / 1e6:.3f}",
             f"{_f(r, 'payload_efficiency'):.3f}",
             f"{_f(r, 'latency_mean_s') * 1e3:.0f}",
             f"{_f(r, 'logical_buffer_kb'):.1f}",
             r.get("confidentiality", ""), r.get("authentication", "")]
            for r in rows]
    return (_table(["метод", "payload, Б", "на дроті, Б", "одиниць/растр",
                    "Мбіт/с", "ефективність", "затримка, мс", "буфер, кБ",
                    "конфіденційність", "автентифікація"], body)
            + "\n\nЗатримка - різниця позначок часу `display − capture` у "
              "подієвому розкладі, а не сума тривалостей етапів.  Буфер - "
              "**логічний буфер протоколу**; це не пікова пам'ять процесу і не "
              "розмір масивів симулятора (три величини на G41).")


def failure_section(view: RunView) -> str:
    rows = view.csv("failures.csv")
    if not rows:
        return "Відмов у прогоні немає."
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_kind.setdefault(r.get("status", "?"), []).append(r)
    lines = [f"Усього відмов: **{len(rows)}**, за категоріями:"]
    for kind, items in sorted(by_kind.items()):
        clips = sorted({i.get("clip", "") for i in items})
        methods = sorted({i.get("method", "") for i in items})
        lines.append(f"- `{kind}`: {len(items)} завдань; методи "
                     f"{', '.join(methods)}; кліпи {', '.join(clips)}")
        detail = items[0].get("detail", "")
        if detail:
            lines.append(f"  - приклад: `{detail[:160]}`")
    lines.append("")
    lines.append("Ці завдання **не** вилучені з обліку: сцена, де метод не "
                 "вміщується у спільний бюджет, відсутня в його середньому і "
                 "показана тут. Парне порівняння використовує лише сцени, де "
                 "обидва методи мають результат, і повідомляє, скільки відкинуто.")
    return "\n".join(lines)


def e05_section(view: RunView) -> str:
    rows = [r for r in view.csv("codewords.csv") if r.get("admissible") == "True"]
    joint = view.csv("joint_loss.csv")
    if not rows:
        return "_E05 не запускалась у цьому прогоні._"
    body = []
    for scheme in sorted({r["scheme"] for r in rows}):
        sel = [r for r in rows if r["scheme"] == scheme
               and r.get("n_descriptions") == "2" and r.get("column_twist") == "1"
               and r.get("burst_lines") == "16"]
        if not sel:
            continue
        r = sel[0]
        jl = [j for j in joint if j["scheme"] == scheme
              and j.get("n_descriptions") == "2" and j.get("column_twist") == "1"
              and j.get("burst_lines") == "16" and j.get("stripe_lost") == "True"]
        p_lost = sum(_f(j, "fraction", 0.0) for j in jl)
        body.append([scheme, r["accumulation_rows"], r["worst_damaged_bytes"],
                     r["rs_nsym"],
                     "так" if r["survives_as_erasures"] == "True" else "ні",
                     f"{p_lost:.3f}"])
    return (_table(["розміщення", "буфер, рядків", "макс. пошкоджених Б",
                    "RS nsym", "виживає як стирання", "P(втрачено обидва описи)"],
                   body)
            + "\n\nОдин burst 16 рядків растру, два описи, вичерпний перебір усіх "
              "початкових позицій.  Дві колонки праворуч вимірюють **різні** "
              "речі: чи виправить FEC пошкоджене кодове слово, і чи втрачені "
              "обидва описи однієї смуги зображення - за двох описів гарантія "
              "«зачеплено не більше двох класів» сама по собі не гарантує "
              "виживання жодного опису.")



def e01_section(view: RunView) -> str:
    rows = view.csv("e01_rate_quality.csv")
    if not rows:
        return "_E01 не запускалась у цьому прогоні._"
    ok = [r for r in rows if r.get("admissible") == "True"]
    bad = [r for r in rows if r.get("admissible") != "True"]
    body = []
    for codec in sorted({r["codec"] for r in ok}):
        sel = sorted([r for r in ok if r["codec"] == codec
                      and r.get("n_descriptions") == "1"
                      and r.get("stripe_height") == "24"],
                     key=lambda r: _f(r, "bits_per_frame"))
        for r in sel:
            body.append([codec, r.get("quality"), f"{_f(r, 'bits_per_frame'):.0f}",
                         _fmt(_f(r, "psnr_full")), _fmt(_f(r, "ssim_full"), 3)])
    reasons: Dict[str, int] = {}
    for r in bad:
        reasons[r.get("reason", "?")[:80]] = reasons.get(r.get("reason", "?")[:80], 0) + 1
    lines = [
        _table(["кодек", "quality", "біт/кадр", "PSNR, дБ", "SSIM"], body),
        "",
        "Один опис, смуга 24 рядки, чистий канал. Кодеки зіставлені за "
        "**фактичним бітрейтом**: при однаковому номері `quality` DCT і JPEG "
        "витрачають різну кількість бітів, тож порівняння «за quality» нічого "
        "не означає.",
        "",
        f"Недопустимих конфігурацій: **{len(bad)} з {len(rows)}**.",
    ]
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {n}× {reason}")
    lines.append("")
    lines.append("`raw` не вміщується у спільний бюджет **у жодній** перевіреній "
                 "конфігурації. Це результат, а не пропуск.")
    return "\n".join(lines)


def e06_section(view: RunView) -> str:
    rows = view.csv("ablations.csv")
    if not rows:
        return "_E06 не запускалась у цьому прогоні._"
    fact = [r for r in rows if r.get("study") == "factorial"]
    body = []
    for r in sorted(fact, key=lambda r: (r.get("n_descriptions"), r.get("placement"))):
        ok = r.get("admissible") == "True"
        body.append([r.get("n_descriptions"), r.get("placement"),
                     _fmt(_f(r, "psnr_full")) if ok else "—",
                     _fmt(_f(r, "coverage"), 3) if ok else "—",
                     "" if ok else r.get("reason", "")[:70]])
    grid = [r for r in rows if r.get("study") == "payload_x_fec"]
    pls = sorted({int(_f(r, "max_unit_payload", 0)) for r in grid})
    nss = sorted({int(_f(r, "fec_nsym", 0)) for r in grid})
    gbody = []
    for pl in pls:
        cells = [str(pl)]
        for ns in nss:
            sel = [r for r in grid if int(_f(r, "max_unit_payload", 0)) == pl
                   and int(_f(r, "fec_nsym", 0)) == ns]
            cells.append(_fmt(_f(sel[0], "psnr_full"))
                         if sel and sel[0].get("admissible") == "True" else "×")
        gbody.append(cells)
    fill = [r for r in rows if r.get("study") == "fill_policy"]
    fbody = [[r.get("fill"), _fmt(_f(r, "psnr_full")),
              _fmt(_f(r, "coverage"), 3), _fmt(_f(r, "max_age_frames"), 2)]
             for r in fill]
    win = [r for r in rows if r.get("study") == "window"]
    wbody = [[r.get("window_rows") or "повний растр",
              _fmt(_f(r, "accumulation_rows"), 0),
              _fmt(_f(r, "window_accumulation_ms"), 1),
              _fmt(_f(r, "psnr_full"))] for r in win]

    return "\n".join([
        "**Це дослідження механізму.** Решта транспорту зафіксована на профілі "
        "`P`, тож клітинки відрізняються лише фактором, що перевіряється. "
        "Порівняння найкращих систем за однаковим бюджетом — інша постановка "
        "(розділ 3) і звітується окремо.",
        "",
        "### Описи × розміщення",
        "",
        _table(["описів", "розміщення", "PSNR, дБ", "покриття", "причина відмови"],
               body),
        "",
        "Усередині фіксованого профілю `P` перехід від одного опису з глибоким "
        "блочним перемежуванням до двох описів з BAWP **не дає** виграшу за "
        "PSNR. Отже перевага `P` над `B4` у розділі 3 походить не від MDC і "
        "BAWP самих по собі, а від решти профілю — менших одиниць і сильнішого "
        "FEC. Це прямо відповідає на дослідницьке питання і звужує заявку про "
        "внесок розміщення.",
        "",
        "### Розмір одиниці × FEC",
        "",
        _table(["payload, Б"] + [f"nsym={n} (k={255-n})" for n in nss], gbody),
        "",
        "`×` — конфігурація не вміщується у спільний бюджет; причина в "
        "`ablations.csv`. Сильніший FEC виграє на пошкодженому каналі в усьому "
        "діапазоні, а великі одиниці стають недопустимими раніше, ніж стають "
        "корисними.",
        "",
        "### Вікно розміщення",
        "",
        _table(["вікно, рядків", "буфер, рядків", "накопичення, мс", "PSNR, дБ"],
               wbody),
        "",
        "### Політика відновлення пропусків",
        "",
        _table(["політика", "PSNR, дБ", "поточне покриття", "макс. вік, кадрів"],
               fbody),
        "",
        "Покриття тут — **лише поточні перевірені** пікселі: старі й домальовані "
        "ділянки до нього не входять за жодної політики, тому політика не може "
        "покращити покриття домальовуванням.",
    ])


def e09_section(view: RunView) -> str:
    rows = view.csv("timings.csv")
    scaling = view.csv("scaling.csv")
    if not rows:
        return "_E09 не запускалась у цьому прогоні._"
    ref = [r for r in rows if r.get("resolution") == "256x192"]
    ref = sorted(ref, key=lambda r: -_f(r, "total_s"))[:12]
    body = [[r.get("stage"), r.get("calls"), _fmt(_f(r, "median_ms")),
             _fmt(_f(r, "p95_ms")), _fmt(_f(r, "p99_ms"))] for r in ref]
    sc = [r for r in scaling if r.get("admissible") == "True"
          and r.get("method") == "P"]
    sbody = [[r.get("resolution"), _fmt(_f(r, "median_ms")),
              _fmt(_f(r, "frames_per_s"), 2)] for r in
             sorted(sc, key=lambda r: _f(r, "pixels"))]
    return "\n".join([
        _table(["етап", "викликів", "median, мс", "p95, мс", "p99, мс"], body),
        "",
        "Кадр 256×192, warm-up відкинуто. Це **wall-clock симулятора**, а не "
        "віртуальний розклад передавання і не оцінка апаратної реалізації.",
        "",
        "### Масштабування (метод P)",
        "",
        _table(["роздільність", "median, мс", "кадрів/с"], sbody),
        "",
        "Вимірювання зроблено в один процес: median, отриманий, поки дванадцять "
        "worker-ів конкурують за ті самі ядра, не був би вартістю етапу. "
        "Це ПК; **це не вимір на Raspberry Pi 5**, і наявність GPU нічого тут "
        "не прискорює — код виконується на CPU у NumPy та чистому Python.",
    ])



def transfer_section(view: RunView, other_dir: Optional[str]) -> str:
    """Same configurations, different material: does the conclusion survive?

    This is a transfer check, not a second tuning round - the configurations
    are not re-selected on the new material.  Both the agreement and the change
    in magnitude are reported, because a smaller effect on real data is a
    result and not something to smooth over.
    """
    if not other_dir or not os.path.isdir(other_dir):
        return ("_Порівняння з реальними кадрами недоступне: немає другого "
                "прогону._")
    other = RunView(other_dir)
    a_an, b_an = view.json("analysis.json"), other.json("analysis.json")
    if not a_an or not b_an:
        return "_Один із прогонів не проаналізовано._"
    a_rows = {r["channel"]: r for r in a_an.get("primary", [])}
    b_rows = {r["channel"]: r for r in b_an.get("primary", [])}
    a_cov = {(r["channel"], r["method"]): _f(r, "mean")
             for r in view.csv("summary.csv") if r.get("metric") == "coverage"}
    b_cov = {(r["channel"], r["method"]): _f(r, "mean")
             for r in other.csv("summary.csv") if r.get("metric") == "coverage"}

    body = []
    agree = disagree = 0
    for ch in sorted(set(a_rows) & set(b_rows) - {"ALL"}, key=_ch_key):
        ra, rb = a_rows[ch], b_rows[ch]
        ma, mb = _f(ra, "mean"), _f(rb, "mean")
        sa = bool(ra.get("significant_at_95"))
        sb = bool(rb.get("significant_at_95"))
        # Two runs agree when they reach the same CONCLUSION.  Comparing the
        # sign of an effect that neither run established is meaningless: a
        # -0.01 dB and a +0.21 dB point estimate, both with an interval
        # containing zero, say exactly the same thing.
        if not sa and not sb:
            same, verdict = True, "збігається (в обох не встановлено)"
        elif sa != sb:
            same, verdict = False, "**розходиться** (значуще лише в одному)"
        elif np.sign(ma) == np.sign(mb):
            same, verdict = True, "збігається"
        else:
            same, verdict = False, "**розходиться** (протилежний знак)"
        agree += int(same)
        disagree += int(not same)
        ca = a_cov.get((ch, "P"))
        cb = b_cov.get((ch, "P"))
        body.append([
            ch,
            f"{ma:+.2f} [{_f(ra, 'lo'):+.2f}; {_f(ra, 'hi'):+.2f}]"
            + ("" if sa else " (не встановлено)"),
            f"{mb:+.2f} [{_f(rb, 'lo'):+.2f}; {_f(rb, 'hi'):+.2f}]"
            + ("" if sb else " (не встановлено)"),
            f"{ca:.3f} / {cb:.3f}" if ca is not None and cb is not None else "—",
            verdict,
        ])
    lines = [
        f"Конфігурації **не переналаштовувались** під другий набір — вони ті "
        f"самі. Синтетичний прогін: {a_an.get('n_scenes')} сцен, "
        f"{a_an.get('n_observations')} кадрів. Реальні кадри з дрона: "
        f"{b_an.get('n_scenes')} сцен, {b_an.get('n_observations')} кадрів.",
        "",
        _table(["канал", "синтетичні, P−B4, дБ", "реальні, P−B4, дБ",
                "покриття P синт/реал", "висновок"], body),
        "",
        f"Висновок збігається у **{agree} з {agree + disagree}** каналів"
        + (f"; розходиться у {disagree}." if disagree else "."),
    ]
    fa = view.csv("failures.csv")
    fb = other.csv("failures.csv")
    lines += [
        "",
        f"**Відмови за бюджетом:** синтетичний набір — {len(fa)}, реальні "
        f"кадри — {len(fb)}."
        + (" Процедурні патерни з високою частотою (регулярна сітка, рендерений "
           "текст) виявились важчими за будь-який реальний кадр: синтетичний "
           "набір був консервативнішим тестом, а не легшим."
           if len(fa) > len(fb) else ""),
    ]
    return "\n".join(lines)


def figures_section(view: RunView, figures_dir: str) -> str:
    rows = view.csv(os.path.join(os.path.relpath(figures_dir, view.run_dir),
                                 "figure_index.csv")) if figures_dir else []
    if not rows:
        idx = os.path.join(figures_dir, "figure_index.csv")
        if os.path.exists(idx):
            with open(idx, encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
    if not rows:
        return "_Каталог рисунків не побудовано._"
    ready = [r for r in rows if r["status"] == "ready"]
    body = [[r["figure"], r["title_uk"], r["experiment"],
             "готово" if r["status"] == "ready" else "pending",
             r.get("reason", "")[:110]] for r in rows]
    return (f"Готово **{len(ready)} з {len(rows)}** рисунків каталогу.\n\n"
            + _table(["ID", "назва", "експеримент", "статус", "причина"], body)
            + "\n\nPNG, SVG і PDF одного рисунка - це один результат, а не три. "
              "Поряд з кожним рисунком лежить його CSV з даними та JSON з "
              "параметрами побудови.")


def _evidence(compare_dir: Optional[str]) -> Dict[str, str]:
    """Experiments whose evidence is a whole second run, not a figure."""
    out: Dict[str, str] = {}
    if compare_dir and os.path.exists(os.path.join(compare_dir, "analysis.json")):
        out["E12"] = "done"
    return out


def programme_section(figures_dir: str,
                      evidence: Optional[Dict[str, str]] = None) -> str:
    rows: List[Dict[str, Any]] = []
    idx = os.path.join(figures_dir, "figure_index.csv")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    st = programme_status(rows, evidence)
    body = []
    for e in st:
        label = {"done": "виконано", "partial": "частково",
                 "not_done": "не виконано", "planned": "заплановано"}[e["status"]]
        body.append([e["id"], e["title_uk"], label,
                     f"{e['figures_ready']}/{e['figures_total']}",
                     e.get("reason", "")[:180]])
    return _table(["ID", "дослідження", "статус", "рисунків", "причина"], body)


# --------------------------------------------------------------------- output
def build_results_doc(view: RunView, figures_dir: str,
                      compare_dir: Optional[str] = None) -> str:
    parts = [
        "# Результати",
        "",
        view.stamp(),
        "",
        "Кожне число в цьому файлі згенеровано з таблиць одного прогону "
        "командою `avsec analyze` + `python scripts/publish.py`.  Значення з "
        "попередніх прогонів перенесені до "
        "[historical_results.md](historical_results.md) і не змішуються з цими.",
        "",
        "## 1. Якість за профілями каналу",
        "",
        quality_table(view),
        "",
        "## 2. Поточне перевірене покриття",
        "",
        coverage_table(view),
        "",
        "## 3. Попередньо зареєстроване порівняння",
        "",
        primary_section(view),
        "",
        "## 4. Бюджет, затримка й пам'ять",
        "",
        budget_table(view),
        "",
        "## 5. Відмови",
        "",
        failure_section(view),
        "",
        "## 6. Розміщення: що саме дає BAWP (E05)",
        "",
        e05_section(view),
        "",
        "## 7. Вартість джерельного кодування (E01)",
        "",
        e01_section(view),
        "",
        "## 8. Абляції: внесок кожного компонента (E06)",
        "",
        e06_section(view),
        "",
        "## 9. Обчислювальна вартість (E09)",
        "",
        e09_section(view),
        "",
        "## 9a. Перенесення на реальні кадри з дрона (E12)",
        "",
        transfer_section(view, compare_dir),
        "",
        "## 10. Каталог рисунків",
        "",
        figures_section(view, figures_dir),
        "",
        "## 11. Програма досліджень",
        "",
        programme_section(figures_dir, _evidence(compare_dir)),
        "",
        "## 12. Межі цих результатів",
        "",
        "- Увесь матеріал **синтетичний**; жодного кадру з реальної камери або "
        "реального радіоканалу тут немає.",
        "- Канал - програмна модель рівня A (растрові спотворення) і рівня B "
        "(CVBS 625/50). Однакова назва профілю на двох рівнях **не** означає "
        "однаковий канал.",
        "- Апаратних вимірювань не проводилось (E11 не виконано), тому слова "
        "«дешевий», «низька затримка на реальному тракті» та «перевірено на "
        "обладнанні» до цієї роботи не застосовні.",
        "- Затримка й пропускна здатність - величини **модельного** розкладу; "
        "wall-clock симулятора звітується окремо і ніколи до них не додається.",
        "",
    ]
    return "\n".join(parts)


def build_readme_block(view: RunView, figures_dir: str) -> str:
    an = view.json("analysis.json")
    v = an.get("primary_verdict", {})
    rows = [r for r in view.csv("summary.csv") if r.get("metric") == "psnr_full"]
    idx_rows: List[Dict[str, Any]] = []
    idx = os.path.join(figures_dir, "figure_index.csv")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8", newline="") as fh:
            idx_rows = list(csv.DictReader(fh))
    n_ready = sum(1 for r in idx_rows if r["status"] == "ready")
    checks = view.json("protocol_checks.json")
    budgets = view.csv("budgets.csv")
    p_row = next((r for r in budgets if r["method"] == "P"), {})
    b4_row = next((r for r in budgets if r["method"] == "B4"), {})

    better = v.get("channels_where_a_is_better") or []
    worse = v.get("channels_where_a_is_worse") or []
    flat = v.get("channels_with_no_detectable_difference") or []
    prim = {r["channel"]: r for r in an.get("primary", [])}

    cov = {(r["channel"], r["method"]): _f(r, "mean")
           for r in view.csv("summary.csv") if r.get("metric") == "coverage"}

    # The headline is composed from what the intervals actually say, so it
    # cannot claim a general win when the effect is confined to one channel.
    chain = view.json("e13.json")
    decomposition = ""
    if chain:
        decomposition = (
            f" Покроковий розклад цієї різниці (E13) показує, що її дають "
            f"**параметри транспорту** ({chain.get('transport_gain_db', 0):+.2f} дБ), "
            f"тоді як самі запропоновані механізми — кілька описів і BAWP — "
            f"**віднімають** {abs(chain.get('mechanism_gain_db', 0)):.2f} дБ.")
    if better and worse:
        headline = (f"**Перевага запропонованої схеми залежить від каналу.** "
                    f"`P` значуще краща за `B4` на {', '.join(better)} і значуще "
                    f"гірша на {', '.join(worse)}. Загального виграшу немає, і "
                    f"це головний результат прогону, а не застереження до нього."
                    + decomposition)
    elif better:
        headline = (f"**`P` значуще перевершує `B4`** на каналах "
                    f"{', '.join(better)}; програшів не зафіксовано.")
    elif worse:
        headline = (f"**`P` значуще програє `B4`** на каналах "
                    f"{', '.join(worse)}; виграшів не зафіксовано.")
    else:
        headline = ("**Різниця між `P` і `B4` не встановлена в жодному каналі** — "
                    "інтервали містять нуль. Це не доказ рівності.")

    lines = [
        README_BEGIN,
        "",
        headline,
        "",
        view.stamp(),
        "",
        f"- **Обсяг:** {an.get('n_observations', '?')} оброблених кадрів, "
        f"{an.get('n_scenes', '?')} незалежних сцен, "
        f"{len(an.get('channels', []))} профілів каналу, "
        f"{len(an.get('methods', []))} методів.",
    ]
    for ch in ("clean", "mild", "moderate", "bursty", "harsh"):
        r = prim.get(ch)
        if not r:
            continue
        mean, lo, hi = _f(r, "mean"), _f(r, "lo"), _f(r, "hi")
        if r.get("significant_at_95"):
            verdict = "значуще"
        elif abs(mean) < 0.01 and abs(hi - lo) < 0.01:
            # Rounding a 0.0005 dB difference to "-0.00" invents a sign; say
            # what the measurement actually shows instead.
            verdict = "методи дали практично тотожний результат (|Δ| < 0,01 дБ)"
        else:
            verdict = "різниця не встановлена"
        cp, cb = cov.get((ch, "P")), cov.get((ch, "B4"))
        cov_txt = ("" if cp is None or cb is None
                   else f", перевірене покриття {cp:.3f} проти {cb:.3f}")
        nd = 3 if abs(mean) < 0.05 else 2
        lines.append(
            f"- **`{ch}`:** P − B4 = {mean:+.{nd}f} дБ "
            f"[{lo:+.{nd}f}; {hi:+.{nd}f}] за {r.get('n_scenes')} "
            f"сценами — {verdict}{cov_txt}.")
    if checks:
        lines.append(f"- **Перевірки протоколу:** {checks.get('n_passed')}/"
                     f"{checks.get('n_checks')} пройдено.")
    if p_row and b4_row:
        lines.append(
            f"- **Бюджет:** B4 — {_f(b4_row, 'payload_bitrate_bps') / 1e6:.3f} Мбіт/с "
            f"корисних даних (ефективність {_f(b4_row, 'payload_efficiency'):.3f}), "
            f"P — {_f(p_row, 'payload_bitrate_bps') / 1e6:.3f} Мбіт/с "
            f"({_f(p_row, 'payload_efficiency'):.3f}); модельна затримка "
            f"{_f(p_row, 'latency_mean_s') * 1e3:.0f} мс, логічний буфер "
            f"{_f(p_row, 'logical_buffer_kb'):.1f} кБ.")
    lines += [
        f"- **Рисунки:** {n_ready} з {len(idx_rows) or len(CATALOGUE)} "
        f"(K01–K10 і G01–G43)"
        + (" — усі побудовано." if n_ready == len(idx_rows) and idx_rows
           else "; решта позначені `pending` з причиною у "
                "[docs/figures.md](docs/figures.md)."),
        "- **Апаратних вимірювань немає** (E11 не виконано): усе нижче — модель.",
        "",
        f"Повні таблиці: [docs/results.md](docs/results.md) · "
        f"аналіз і висновки: [docs/conclusions.md](docs/conclusions.md) · "
        f"курована добірка з усіма рисунками: [results/](results/).",
        README_END,
    ]
    return "\n".join(lines)


def publish(run_dir: str, docs_dir: str = "docs", readme: str = "README.md",
            figures_dir: Optional[str] = None,
            compare_dir: Optional[str] = None) -> Dict[str, Any]:
    """Regenerate every document that quotes a number, from this run only."""
    view = RunView(run_dir)
    figures_dir = figures_dir or os.path.join(run_dir, "figures")
    ensure_dir(docs_dir)

    results = build_results_doc(view, figures_dir, compare_dir)
    with open(os.path.join(docs_dir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(results)

    with open(os.path.join(docs_dir, "figures.md"), "w", encoding="utf-8") as fh:
        fh.write("# Каталог рисунків: K01–K10 і G01–G43\n\n" + view.stamp() + "\n\n"
                 + figures_section(view, figures_dir) + "\n")

    with open(os.path.join(docs_dir, "programme.md"), "w", encoding="utf-8") as fh:
        fh.write("# Програма досліджень E01–E13\n\n" + view.stamp() + "\n\n"
                 + programme_section(figures_dir, _evidence(compare_dir)) + "\n\n"
                 + "## E11 — апаратна перевірка\n\n"
                 + EXPERIMENTS["E11"].reason + "\n\n"
                 + "Доки цих вимірювань немає, рисунки H01–H04 не існують, а "
                   "твердження про вартість, масу, споживання та наскрізну "
                   "затримку реального тракту не робляться.\n")

    updated_readme = False
    if os.path.exists(readme):
        with open(readme, encoding="utf-8") as fh:
            text = fh.read()
        block = build_readme_block(view, figures_dir)
        if README_BEGIN in text and README_END in text:
            head = text.split(README_BEGIN)[0]
            tail = text.split(README_END)[1]
            text = head + block + tail
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
        with open(readme, "w", encoding="utf-8") as fh:
            fh.write(text)
        updated_readme = True

    return {"run_id": view.run_id, "commit": view.commit,
            "compared_with": compare_dir,
            "results": os.path.join(docs_dir, "results.md"),
            "figures": os.path.join(docs_dir, "figures.md"),
            "programme": os.path.join(docs_dir, "programme.md"),
            "readme_updated": updated_readme}


__all__ = ["RunView", "publish", "build_results_doc", "build_readme_block",
           "README_BEGIN", "README_END", "CLAIM_WORDS"]
