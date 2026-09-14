"""The research programme E01-E11 and the figure catalogue G01-G43.

This module is the single place that says *what was planned*.  Every runner and
every report reads it, so a figure that was never produced shows up as
``pending`` with a reason instead of quietly disappearing from the catalogue,
and an experiment that was not run cannot be described as if it had been.

Nothing here executes anything; it is the plan, plus the rule for deciding
whether an artifact exists.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------- experiments
@dataclass(frozen=True)
class Experiment:
    eid: str
    title_uk: str
    title_en: str
    question: str
    outputs: Tuple[str, ...]
    status: str = "planned"          # planned | done | partial | not_done
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.eid, "title_uk": self.title_uk, "title_en": self.title_en,
                "question": self.question, "outputs": list(self.outputs),
                "status": self.status, "reason": self.reason}


PROGRAM: Tuple[Experiment, ...] = (
    Experiment(
        "E01", "Вартість джерельного кодування на чистому каналі",
        "Source-coding cost on a clean channel",
        "Скільки якості втрачається до каналу через стиснення, фрагментацію "
        "та кілька описів?",
        ("e01_rate_quality.csv", "G11", "G12")),
    Experiment(
        "E02", "Основне порівняння систем", "Main system comparison",
        "Чи дає запропонована схема кращу якість у тому самому бюджеті, ніж "
        "сегментований AEAD без MDC і BAWP?",
        ("frames.csv", "observations.csv", "paired_effects.csv",
         "G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10")),
    Experiment(
        "E03", "Однофакторні залежності від спотворень",
        "Single-factor impairment sweeps",
        "Як якість і покриття залежать від кожного окремого параметра каналу?",
        ("sweeps.csv", "G13", "G17", "G18", "G19", "G20", "G21")),
    Experiment(
        "E04", "Взаємодія спотворень і межа працездатності",
        "Impairment interaction and the operating boundary",
        "Де межа, за якою жодна конфігурація не виконує робочий критерій?",
        ("interaction.csv", "G22", "G23")),
    Experiment(
        "E05", "Ізольоване дослідження BAWP", "BAWP in isolation",
        "Що саме дає розміщення, якщо все інше зафіксовано?",
        ("codewords.csv", "G27", "G28", "G29", "G30")),
    Experiment(
        "E06", "Абляції та взаємодія параметрів", "Ablations and parameter interaction",
        "Який внесок кожного компонента окремо і в комбінації?",
        ("ablations.csv", "G31", "G32", "G33", "G34")),
    Experiment(
        "E07", "Динаміка, втрата синхронізації та затримка",
        "Dynamics, loss of sync and latency",
        "Як швидко система відновлюється і наскільки старі дані показує?",
        ("dynamics.csv", "G24", "G25", "G26", "G43")),
    Experiment(
        "E08", "Атаки й перевірки протоколу", "Attacks and protocol checks",
        "Чи витримують бази 2021 року відомі атаки і чи покриває AEAD усі поля?",
        ("attacks.json", "protocol_checks.json", "G35", "G36", "G37", "G38")),
    Experiment(
        "E09", "Обчислювальна вартість і компроміси", "Computational cost and trade-offs",
        "Скільки коштує кожен етап і які межі Парето якість-швидкість-затримка-пам'ять?",
        ("timings.csv", "budgets.csv", "G14", "G15", "G16", "G39", "G40", "G41")),
    Experiment(
        "E10", "Перенесення між моделями", "Transfer between models",
        "Чи переносяться зафіксовані на рівні A конфігурації на рівень B "
        "без переналаштування?",
        ("cvbs.json", "G42")),
    Experiment(
        "E12", "Реальні кадри з дрона", "Real UAV imagery",
        "Чи переносяться висновки з процедурного матеріалу на справжні кадри "
        "з дрона, без переналаштування конфігурацій?",
        ("frames.csv", "paired_effects.csv", "K01", "K02", "K07", "K08")),
    Experiment(
        "E13", "Розклад переваги по кроках", "Decomposition of the advantage",
        "Яку частину різниці B4 -> P дають параметри транспорту, доступні будь-"
        "якій схемі, а яку - два запропоновані механізми?",
        ("chain.csv", "chain_scenes.csv", "K05", "K11")),
    Experiment(
        "E14", "Незалежні природні джерела", "Independent natural sources",
        "Чи зберігаються висновки, коли одиниця незалежності - окрема "
        "фотографія з БпЛА, а не вирізка з однієї?",
        ("frames.csv", "paired_effects.csv", "dataset_manifest.csv", "K12")),
    Experiment(
        "E15", "Карта області працездатності", "Operating region map",
        "За яких довжин і частот пакетних спотворень запропонована схема "
        "перевершує переналаштовану базу, і де різниця не встановлена?",
        ("operating_map.csv", "operating_map_scenes.csv", "K13")),
    Experiment(
        "E11", "Апаратна перевірка", "Hardware validation",
        "Які реальні місткість, затримка, споживання і вартість фізичного тракту?",
        ("H01", "H02", "H03", "H04"),
        status="not_done",
        reason="фізичний CVBS-тракт (генератор/ЦАП + відеозахоплення) недоступний; "
               "жодного апаратного вимірювання не виконано, і слово «дешевий» "
               "лишається метою, а не підтвердженим результатом"),
)

EXPERIMENTS: Dict[str, Experiment] = {e.eid: e for e in PROGRAM}


# ------------------------------------------------------------------- figures
@dataclass(frozen=True)
class Figure:
    gid: str
    title_uk: str
    title_en: str
    axes: str
    source: str                      # which experiment's data it needs
    tables: Tuple[str, ...] = ()     # input tables it reads
    panels: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.gid, "title_uk": self.title_uk, "title_en": self.title_en,
                "axes": self.axes, "experiment": self.source,
                "tables": list(self.tables), "panels": self.panels}


CATALOGUE: Tuple[Figure, ...] = (
    Figure("G01", "Наскрізні приклади", "End-to-end examples",
           "оригінал / переданий растр / прийнятий растр / реконструкція",
           "E02", ("examples/",), panels=4),
    Figure("G02", "Карти доступності", "Availability maps",
           "поточне / старе / інтерпольоване", "E02", ("examples/",), panels=3),
    Figure("G03", "Якість у п'яти профілях", "Quality across five profiles",
           "профіль -> PSNR усього показаного кадру, CI", "E02", ("observations.csv",)),
    Figure("G04", "Структурна якість", "Structural quality",
           "профіль -> SSIM, CI", "E02", ("observations.csv",)),
    Figure("G05", "Поточне автентифіковане покриття", "Current authenticated coverage",
           "профіль -> частка; неавтентифіковані бази відділені", "E02",
           ("observations.csv",)),
    Figure("G06", "Працездатність", "Operability",
           "профіль -> частка показів, що виконують критерій", "E02",
           ("observations.csv",)),
    Figure("G07", "Парна перевага P над B4", "Paired advantage of P over B4",
           "профіль -> різниця PSNR з CI", "E02", ("paired_effects.csv",)),
    Figure("G08", "Результати окремих сцен", "Per-scene results",
           "PSNR B4 -> PSNR P, діагональ рівності", "E02", ("observations.csv",)),
    Figure("G09", "Залежність від вмісту", "Content dependence",
           "категорія x канал -> середня різниця P-B4", "E02", ("observations.csv",)),
    Figure("G10", "Розподіл виграшу", "Distribution of the gain",
           "різниця P-B4 за сценами -> ECDF", "E02", ("observations.csv",)),
    Figure("G11", "Стиснення на чистому каналі", "Compression on a clean channel",
           "фактичні біти/кадр -> PSNR", "E01", ("e01_rate_quality.csv",)),
    Figure("G12", "Накладні витрати", "Overhead breakdown",
           "метод -> payload/header/tag/FEC/padding/filler/службові комірки",
           "E01", ("budgets.csv",)),
    Figure("G13", "Корисні дані, отримані вчасно", "Useful data delivered in time",
           "рівень шуму -> автентифіковані корисні Б/с", "E03", ("sweeps.csv",)),
    Figure("G14", "Межа Парето за швидкістю", "Rate Pareto front",
           "фактичний бітрейт -> якість за фіксованої затримки", "E09",
           ("budgets.csv", "observations.csv")),
    Figure("G15", "Межа Парето за затримкою", "Latency Pareto front",
           "затримка -> якість, однаковий ресурс каналу", "E09",
           ("budgets.csv", "observations.csv")),
    Figure("G16", "Межа Парето за буфером", "Buffer Pareto front",
           "логічний буфер -> якість; RAM симулятора окремо", "E09",
           ("budgets.csv", "observations.csv")),
    Figure("G17", "Чутливість до шуму", "Noise sensitivity",
           "sigma шуму -> PSNR і покриття", "E03", ("sweeps.csv",), panels=2),
    Figure("G18", "Чутливість до довжини burst", "Burst-length sensitivity",
           "пошкоджені рядки -> якість/покриття", "E03", ("sweeps.csv",), panels=2),
    Figure("G19", "Чутливість до частоти burst", "Burst-rate sensitivity",
           "burst на растр -> якість/покриття", "E03", ("sweeps.csv",), panels=2),
    Figure("G20", "Чутливість до джитеру", "Jitter sensitivity",
           "sigma зсуву, px -> якість/синхронізація", "E03", ("sweeps.csv",), panels=2),
    Figure("G21", "Чутливість до фільтрації", "Low-pass sensitivity",
           "параметр фільтра -> SER і якість", "E03", ("sweeps.csv",), panels=2),
    Figure("G22", "Підсилення та зміщення", "Gain and offset",
           "gain x offset -> покриття; області clipping", "E04", ("interaction.csv",)),
    Figure("G23", "Спільний вплив шуму й burst", "Noise and burst interaction",
           "sigma x довжина -> P-B4; невизначеність окремо", "E04",
           ("interaction.csv",), panels=2),
    Figure("G24", "Час відновлення", "Recovery time",
           "час після завади -> частка відновлених запусків", "E07", ("dynamics.csv",)),
    Figure("G25", "Динаміка якості", "Quality dynamics",
           "час -> PSNR і покриття, позначені завади", "E07", ("dynamics.csv",), panels=2),
    Figure("G26", "Актуальність показу", "Freshness of the display",
           "час -> вік даних, затримка, довжина черги", "E07", ("dynamics.csv",), panels=3),
    Figure("G27", "Геометрія перемежувачів", "Interleaver geometry",
           "рядок x стовпець, кольори RS-слів і описів", "E05", ("placement/",)),
    Figure("G28", "Пошкодження кодових слів", "Codeword damage",
           "початок burst x RS-слово -> помилки/стирання", "E05", ("codewords.csv",)),
    Figure("G29", "Перевірка аналітичної межі", "Analytic bound check",
           "довжина burst -> фактичний максимум і заявлена межа", "E05",
           ("codewords.csv",)),
    Figure("G30", "Спільні втрати описів", "Joint description loss",
           "втрачені описи однієї смуги -> частота", "E05", ("codewords.csv",)),
    Figure("G31", "Абляція MDC x розміщення", "MDC x placement ablation",
           "1/2 описи x block/BAWP -> якість і покриття", "E06", ("ablations.csv",)),
    Figure("G32", "Розмір одиниці та FEC", "Unit size and FEC",
           "payload x nsym -> якість; недопустимі клітинки позначені", "E06",
           ("ablations.csv",)),
    Figure("G33", "Вікно й затримка", "Window and latency",
           "висота вікна -> якість та модельна затримка", "E06", ("ablations.csv",)),
    Figure("G34", "Політика відновлення пропусків", "Gap-fill policy",
           "neutral/interpolate/previous -> якість, вік, покриття", "E06",
           ("ablations.csv",)),
    Figure("G35", "Атака за сумісністю меж", "Boundary-consistency attack",
           "сітка блоків / категорія -> точність відновлення", "E08", ("attacks.json",)),
    Figure("G36", "Відомі або обрані кадри", "Known or chosen frames",
           "число кадрів/запитів -> успіх відновлення перестановки", "E08",
           ("attacks.json",)),
    Figure("G37", "Перебір лабораторного LFSR", "Lab LFSR brute force",
           "ширина seed -> виміряний час; тайм-аути позначені", "E08", ("attacks.json",)),
    Figure("G38", "Перевірки протоколу", "Protocol checks",
           "сценарій x очікуваний/отриманий статус", "E08", ("protocol_checks.json",)),
    Figure("G39", "Час етапів конвеєра", "Pipeline stage timing",
           "етап -> median/p95/p99 wall-clock", "E09", ("timings.csv",)),
    Figure("G40", "Масштабування", "Scaling",
           "роздільність -> кадри/с та вартість кадру", "E09", ("timings.csv",)),
    Figure("G41", "Пам'ять", "Memory",
           "конфігурація -> peak RAM і логічні буфери, окремі панелі", "E09",
           ("budgets.csv",), panels=2),
    Figure("G42", "Перенесення рівень A -> B", "Transfer level A -> B",
           "зіставлені умови -> якість/покриття без повторного добору", "E10",
           ("cvbs.json",)),
    Figure("G43", "Подієвий розклад", "Event schedule",
           "часова шкала накопичення, кодування, передавання, приймання, показу",
           "E07", ("budgets.csv",)),
)

#: The ten figures that carry the argument.  The G catalogue is complete by
#: construction - one number per research question - but completeness is not the
#: same as saying something, and several of those figures are bookkeeping.  These
#: are the ones a reader should look at first; they are built from exactly the
#: same tables.
KEY: Tuple[Figure, ...] = (
    Figure("K01", "Захист і якість за методами", "Protection and quality by method",
           "захист × метод, поруч якість і покриття", "E02",
           ("summary.csv",), panels=2),
    Figure("K02", "Точка перелому", "The crossover",
           "довжина пакета → якість і покриття, точка перетину", "E03",
           ("sweeps.csv",), panels=2),
    Figure("K03", "Карта режимів", "Operating map",
           "шум × пакет → хто виграє; «≈» там, де різниця не встановлена", "E04",
           ("interaction.csv",)),
    Figure("K04", "Компроміс схем розміщення", "Placement trade-off",
           "пошкодження RS-слова × втрата обох описів", "E05",
           ("codewords.csv", "joint_loss.csv")),
    Figure("K05", "Джерела виміряної переваги", "Where the advantage comes from",
           "кроки B4 → P → внесок кожного, дБ", "E13",
           ("chain.csv",), panels=2),
    Figure("K06", "Ціна стиснення", "The price of compression",
           "фактичний бітрейт → якість, межа місткості каналу", "E01",
           ("e01_rate_quality.csv",)),
    Figure("K07", "Наскрізний приклад", "End-to-end example",
           "кадр → растр → прийнято → реконструкція, B4 і P", "E02",
           ("frames.csv",), panels=8),
    Figure("K08", "Склад показаного зображення", "What the viewer is shown",
           "перевірене / старе / домальоване", "E02", ("frames.csv",), panels=3),
    Figure("K09", "Бюджет, затримка, пам'ять", "Budget, latency, memory",
           "місткість → корисні дані; розклад подій; три величини пам'яті", "E09",
           ("budgets.json",), panels=3),
    Figure("K10", "Атаки на реконструкцію B1 зі статичною перестановкою",
           "Attacks on the reconstructed B1 with a static permutation",
           "оригінал / передане / відновлене атакою + успіх атак", "E08",
           ("attacks.json",), panels=3),
    Figure("K11", "Залежність результату від порядку кроків",
           "Dependence of the result on the order of steps",
           "прямий і зворотний розклад B4 → P, з парними інтервалами", "E13",
           ("chain.csv",), panels=2),
    Figure("K12", "24 незалежні джерела", "Twenty-four independent sources",
           "ефект по сценах і по вихідних фотографіях, з інтервалами", "E14",
           ("paired_effects.csv",), panels=2),
    Figure("K13", "Карта області працездатності", "Operating region map",
           "довжина × частота пакетів: де P кращий за переналаштований B4", "E15",
           ("operating_map.csv",), panels=2),
)

#: The figures of the paper itself.  Everything else - the rest of the K series
#: and the whole G catalogue - is supplementary material (R12).
#:
#: Eight, chosen because each answers a question the others do not:
#:   K01  protection and quality of each method, side by side
#:   K04  what the placement rule buys and what it costs
#:   K05  where the measured advantage comes from
#:   K11  how much of that decomposition is an artefact of the step order
#:   K12  the effect with an interval over independent recordings
#:   K13  the operating region, as a map rather than a threshold
#:   K09  the budget: capacity, latency and memory in one place
#:   K10  attacks on the reconstructed B1 with a static permutation
MAIN_FIGURES: Tuple[str, ...] = ("K01", "K04", "K05", "K11", "K12", "K13",
                                 "K09", "K10")

FIGURES: Dict[str, Figure] = {f.gid: f for f in CATALOGUE + KEY}

assert len(CATALOGUE) == 43, "the catalogue is defined as 43 figures"
assert len(KEY) == 13, "there are thirteen key figures"
assert 6 <= len(MAIN_FIGURES) <= 8, "a paper carries 6-8 figures"
assert all(g in FIGURES for g in MAIN_FIGURES)


# ---------------------------------------------------------------- readiness
#: Export formats a "ready" figure must have.  PNG, SVG and PDF of the same
#: figure are one result, not three, so all three are required together.
EXPORT_FORMATS = ("png", "svg", "pdf")


def figure_paths(directory: str, gid: str) -> Dict[str, str]:
    return {fmt: os.path.join(directory, f"{gid}.{fmt}") for fmt in EXPORT_FORMATS}


def figure_status(directory: str, gid: str, reason: str = "") -> Dict[str, Any]:
    """Is this figure actually on disk, with its data table next to it?"""
    paths = figure_paths(directory, gid)
    present = {fmt: os.path.exists(p) for fmt, p in paths.items()}
    data = os.path.join(directory, f"{gid}.csv")
    meta = os.path.join(directory, f"{gid}.json")
    ready = all(present.values()) and os.path.exists(data)
    fig = FIGURES.get(gid)
    return {
        "figure": gid,
        "title_uk": fig.title_uk if fig else "",
        "title_en": fig.title_en if fig else "",
        "experiment": fig.source if fig else "",
        "status": "ready" if ready else "pending",
        "reason": "" if ready else (reason or _default_reason(gid, present, data)),
        "files": ";".join(p for fmt, p in paths.items() if present[fmt]),
        "data_table": data if os.path.exists(data) else "",
        "params": meta if os.path.exists(meta) else "",
    }


def _default_reason(gid: str, present: Dict[str, bool], data: str) -> str:
    fig = FIGURES.get(gid)
    exp = EXPERIMENTS.get(fig.source) if fig else None
    if exp is not None and exp.status == "not_done":
        return f"{exp.eid} не виконано: {exp.reason}"
    if not any(present.values()):
        return f"дані {fig.source if fig else '?'} відсутні або експеримент не запускався"
    missing = [f for f, ok in present.items() if not ok]
    if missing:
        return f"немає форматів: {', '.join(missing)}"
    if not os.path.exists(data):
        return "немає CSV з даними рисунка"
    return "невідома причина"


def catalogue_status(directory: str, reasons: Optional[Dict[str, str]] = None,
                     include_key: bool = True) -> List[Dict[str, Any]]:
    """The whole table, ready or pending, in catalogue order.

    The ten key figures come first: a reader who stops after the first screen
    should have seen the ones that carry the argument.
    """
    reasons = reasons or {}
    figs = (list(KEY) if include_key else []) + list(CATALOGUE)
    return [figure_status(directory, f.gid, reasons.get(f.gid, "")) for f in figs]


def programme_status(figure_rows: Sequence[Dict[str, Any]],
                     evidence: Optional[Dict[str, str]] = None
                     ) -> List[Dict[str, Any]]:
    """Derive each experiment's status from the figures that depend on it.

    ``evidence`` overrides that for experiments whose result is not a figure of
    its own.  E12, for instance, is a second full run on different material: its
    figures are the same K-numbers rebuilt from that run's tables, so counting
    figures in one directory would report it as never performed.
    """
    evidence = evidence or {}
    out: List[Dict[str, Any]] = []
    for exp in PROGRAM:
        mine = [r for r in figure_rows if r.get("experiment") == exp.eid]
        ready = sum(1 for r in mine if r["status"] == "ready")
        if exp.status == "not_done":
            status = "not_done"
        elif exp.eid in evidence:
            status = evidence[exp.eid]
        elif not mine:
            status = exp.status
        elif ready == len(mine):
            status = "done"
        elif ready:
            status = "partial"
        else:
            status = "not_done"
        d = exp.to_dict()
        d.update({"status": status, "figures_ready": ready,
                  "figures_total": len(mine)})
        if status != "done" and not d["reason"]:
            missing = [r["figure"] for r in mine if r["status"] != "ready"]
            d["reason"] = ("не побудовано: " + ", ".join(missing)) if missing else ""
        out.append(d)
    return out


__all__ = ["Experiment", "PROGRAM", "EXPERIMENTS", "Figure", "CATALOGUE",
           "KEY", "FIGURES",
           "EXPORT_FORMATS", "figure_paths", "figure_status", "catalogue_status",
           "programme_status"]
