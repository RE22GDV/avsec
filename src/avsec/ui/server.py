"""Standard-library HTTP server backing the avsec web interface.

Long jobs run on a worker thread and report progress; the page polls
``/api/job/<id>``.  Every endpoint calls the same library functions as the CLI,
so the UI cannot drift from the command line results.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np

from avsec import __version__
from avsec.utils import NpEncoder, sanitize_json

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()
_RUNS_DIR = "runs"


# ------------------------------------------------------------------ helpers
def png_data_url(img: np.ndarray, max_width: int = 720) -> str:
    import cv2

    a = np.asarray(img)
    if a.ndim == 2:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if a.shape[1] > max_width:
        scale = max_width / a.shape[1]
        a = cv2.resize(a, (max_width, max(1, int(a.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", a)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def plot_data_url(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _new_job() -> str:
    jid = uuid.uuid4().hex[:12]
    with _LOCK:
        _JOBS[jid] = {"id": jid, "state": "running", "progress": 0.0, "stage": "starting",
                      "result": None, "error": None, "started": time.time()}
        if len(_JOBS) > 40:  # bounded memory
            for k in sorted(_JOBS, key=lambda k: _JOBS[k]["started"])[:10]:
                _JOBS.pop(k, None)
    return jid


def _set(jid: str, **kw: Any) -> None:
    with _LOCK:
        if jid in _JOBS:
            _JOBS[jid].update(kw)


def run_async(fn: Callable[[Callable[[str, float, Dict[str, Any]], None]], Any]) -> str:
    jid = _new_job()

    def _progress(stage: str, frac: float, info: Dict[str, Any]) -> None:
        _set(jid, stage=stage, progress=max(0.0, min(1.0, float(frac))))

    def _worker() -> None:
        try:
            res = fn(_progress)
            _set(jid, state="done", progress=1.0, stage="done", result=res)
        except Exception as exc:  # surfaced to the page, never swallowed
            _set(jid, state="error", error=f"{type(exc).__name__}: {exc}",
                 traceback=traceback.format_exc()[-4000:])

    threading.Thread(target=_worker, daemon=True).start()
    return jid


def _config(payload: Dict[str, Any]):
    from avsec.config import config_from_dict

    return config_from_dict(payload.get("config") or {})


# ------------------------------------------------------------------- actions
def _drone_patterns() -> List[str]:
    """Real UAV scenes, offered next to the procedural patterns.

    They are prefixed so nobody can confuse a real frame with a generated one:
    the provenance field says the same thing, but the dropdown is what a user
    actually reads.
    """
    try:
        from avsec.sources.drone import DRONE_PHOTO, DRONE_SCENES
    except Exception:
        return []
    if not os.path.exists(DRONE_PHOTO):
        return []
    return [f"uav:{name}" for name, _c, _s, _e, _m, split in DRONE_SCENES
            if split == "test"]


def action_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    from avsec.channel import PRESETS
    from avsec.config import ALL_METHODS, DEFAULT_PROFILES, ExperimentConfig
    from avsec.interleaving import SCHEMES
    from avsec.modem.cvbs import CVBS_PRESETS
    from avsec.sources import PATTERNS
    from avsec.utils import environment_record

    cfg = ExperimentConfig()
    return {
        "version": __version__,
        "environment": environment_record(),
        "channel_presets": {k: v.describe() for k, v in PRESETS.items()},
        "cvbs_presets": sorted(CVBS_PRESETS),
        "patterns": sorted(PATTERNS) + _drone_patterns(),
        # the canonical list, not a copy of it: a scheme added to the matrix
        # used to appear in the command line and silently not here
        "methods": list(ALL_METHODS),
        "interleaver_schemes": list(SCHEMES),
        "codecs": ["raw", "dct", "jpeg"],
        "default_config": cfg.to_dict(),
        "default_profiles": {k: v.to_dict() for k, v in DEFAULT_PROFILES.items()},
        "configs_dir": sorted(
            f for f in os.listdir("configs") if f.endswith(".yaml")
        ) if os.path.isdir("configs") else [],
        "method_notes": {
            "B0a": "незахищене зображення через аналоговий растр (опора для втрат якості)",
            "B0a-R": ("те саме передавання, але кадр повторюється в усі три слоти, "
                      "а приймач поєднує копії за медіаною - чесна аналогова опора"),
            "B0d": "той самий цифровий транспорт БЕЗ криптографії - діагностика, нічого не автентифікує",
            "B0d-W": ("цифровий транспорт із публічним вибілюванням потоку - "
                      "відокремлює вирівнювання статистики від шифрування"),
            "B1": "перестановка блоків на LFSR (реконструкція статті 2021)",
            "B2": "та сама перестановка з криптогенератором - вміст блоків НЕ шифрується",
            "B3": "AEAD цілого кадру однією одиницею; потрібні всі фрагменти",
            "B4": "незалежно захищені смуги, один опис - сильний базовий метод",
            "B4t": ("B4 з параметрами транспорту схеми P, але БЕЗ обох механізмів - "
                    "контрольна схема, яка відокремлює внесок транспорту"),
            "P": "кілька описів + BAWP + спільний добір параметрів (запропоноване)",
        },
        # these two are not in the matrix: they have their own benches, their
        # own tabs and their own documents, so they are named rather than
        # silently missing from the list above
        "own_bench_methods": {
            "B2s": "перестановка блоків разом із ключовою заміною значень пікселів",
            "B2l": "шифротекст зі сталою яскравістю: монохромний спостерігач бачить сіре поле",
        },
    }


def action_demo(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    """One frame through the selected methods, returning images inline."""
    from avsec import evaluation as ev
    from avsec import sources as S
    from avsec.experiments import build_methods, build_sources
    from avsec.utils import StageTimer, experiment_rng

    cfg = _config(payload)
    timer = StageTimer()
    if payload.get("image_b64"):
        import cv2

        raw = base64.b64decode(payload["image_b64"].split(",")[-1])
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError("не вдалося прочитати завантажене зображення")
        if arr.shape != (cfg.frame_height, cfg.frame_width):
            arr = cv2.resize(arr, (cfg.frame_width, cfg.frame_height),
                             interpolation=cv2.INTER_AREA)
        source = S.FrameSource("uploaded", [arr], S.PROV_LOCAL_FILE,
                               "image uploaded through the web interface")
    else:
        source = build_sources(cfg)[0]

    methods = build_methods(cfg, timer)
    frame = source.frames[0]
    from avsec.channel import ChannelTrace

    trace = ChannelTrace(seed=cfg.seed, scene=source.name, repetition=0,
                         profile=cfg.channel_preset,
                         rasters_per_frame=cfg.budget.rasters_per_frame)
    out: List[Dict[str, Any]] = []
    for i, (name, m) in enumerate(methods.items()):
        progress(f"метод {name}", i / max(len(methods), 1), {})
        m.reset()
        try:
            res = m.process(frame, 0, trace)
        except Exception as exc:
            out.append({"method": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        imgs = res.images()
        out.append({
            "method": name,
            "metrics": res.metrics.to_row(),
            "notes": res.notes,
            "authenticated": bool(getattr(m, "authenticated", False)),
            "images": {k: png_data_url(v) for k, v in imgs.items()},
            "truth": res.truth.summary() if res.truth else None,
        })
    return {"kind": "demo", "source": source.summary(), "results": out,
            "timing": timer.summary(), "config": cfg.to_dict()}


def action_scramble(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    """Interactive B1/B2 playground plus the attacks on them."""
    from avsec import attacks as A
    from avsec import sources as S
    from avsec.baselines import CryptoPermutationScrambler
    from avsec.crypto import derive_session_keys, new_session_id
    from avsec.lfsr import BlockScrambler, correlation_similarity

    cfg = _config(payload)
    H, W = cfg.frame_height, cfg.frame_width
    if payload.get("image_b64"):
        import cv2

        raw = base64.b64decode(payload["image_b64"].split(",")[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    else:
        img = S.PATTERNS[payload.get("pattern", "edges")](H, W)

    progress("перестановка", 0.15, {})
    sc = BlockScrambler(cfg.lfsr)
    scrambled = sc.scramble(img, 0)
    restored = sc.descramble(scrambled, 0, (H, W))
    r1, sim1 = correlation_similarity(sc._fit(img), scrambled)
    r2, sim2 = correlation_similarity(img, restored)

    rows, cols = cfg.lfsr.grid_rows, cfg.lfsr.grid_cols
    result: Dict[str, Any] = {
        "kind": "scramble",
        "lfsr": cfg.lfsr.describe(),
        "similarity": {
            "corr_original_vs_scrambled": r1, "similarity_scrambled_pct": sim1,
            "corr_original_vs_restored": r2, "similarity_restored_pct": sim2,
            "exact_recovery": bool(np.array_equal(img, restored)),
        },
        "images": {
            "original": png_data_url(img),
            "scrambled": png_data_url(scrambled),
            "restored": png_data_url(restored),
        },
        "attacks": [],
    }

    if payload.get("run_attacks", True):
        progress("атака за відомим кадром", 0.4, {})
        cp = A.attack_chosen_plaintext(sc, 0, H, W)
        result["attacks"].append(cp.to_dict())
        kp = A.attack_known_pair(sc._fit(img), scrambled, rows, cols)
        kp.metrics["permutation_accuracy"] = float(
            (kp.recovered_permutation == sc.permutation(0)).mean())
        result["attacks"].append(kp.to_dict())
        progress("атака за сумісністю меж", 0.65, {})
        ba = A.attack_boundary_reassembly(scrambled, rows, cols, sc.permutation(0),
                                          sc._fit(img))
        result["attacks"].append(ba.to_dict())
        if ba.reconstructed is not None:
            result["images"]["boundary_attack"] = png_data_url(ba.reconstructed)
        if payload.get("run_bruteforce"):
            progress("перебір seed", 0.85, {})
            bf = A.attack_lfsr_bruteforce(sc._fit(img), scrambled, cfg.lfsr,
                                          max_states=int(payload.get("max_states", 70000)))
            result["attacks"].append(bf.to_dict())

    if payload.get("run_b2"):
        progress("B2 криптоперестановка", 0.92, {})
        sid = new_session_id()
        keys = derive_session_keys(cfg.master_secret(), sid)
        cs = CryptoPermutationScrambler(rows, cols, keys.key, sid, per_frame=True)
        s2 = cs.scramble(img, 0)
        result["images"]["b2_scrambled"] = png_data_url(s2)
        ba2 = A.attack_boundary_reassembly(s2, rows, cols, cs.permutation(0), img)
        d = ba2.to_dict()
        d["name"] = "boundary_compatibility_reassembly_on_B2"
        result["attacks"].append(d)
        if ba2.reconstructed is not None:
            result["images"]["b2_boundary_attack"] = png_data_url(ba2.reconstructed)
    return result


def action_comparison(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    from avsec.experiments import run_comparison

    cfg = _config(payload)
    out_dir = os.path.join(_RUNS_DIR, f"ui_comparison_{int(time.time())}")
    res = run_comparison(cfg, out_dir, progress)
    res["output_dir"] = out_dir
    res.pop("record", None)
    return res


def action_budget(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    from avsec.experiments import run_budget

    return run_budget(_config(payload))


def action_attacks(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    from avsec.experiments import run_attacks

    cfg = _config(payload)
    out_dir = os.path.join(_RUNS_DIR, f"ui_attacks_{int(time.time())}")
    res = run_attacks(cfg, out_dir, progress)
    res["output_dir"] = out_dir
    res.pop("record", None)
    return res


def action_cvbs(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    """Level-B run plus a waveform plot of one transmitted line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from avsec.experiments import build_sources, run_cvbs
    from avsec.modem.cvbs import CVBSChannel, CVBSGenerator, CVBSProfile, cvbs_preset
    from avsec.utils import experiment_rng

    cfg = _config(payload)
    preset_name = payload.get("preset", "mild")
    out_dir = os.path.join(_RUNS_DIR, f"ui_cvbs_{int(time.time())}")
    res = run_cvbs(cfg, out_dir, preset_name, payload.get("method", "B4"), progress)
    res["output_dir"] = out_dir
    res.pop("record", None)

    progress("осцилограма", 0.9, {})
    prof = CVBSProfile()
    gen = CVBSGenerator(prof)
    from avsec.config import MethodProfile

    mp = cfg.profile("B4")
    tr = mp.transport_config(cfg.budget)
    from avsec.modem import RasterModem

    modem = RasterModem(tr.modem)
    demo_syms = np.zeros(tr.modem.capacity_symbols, dtype=np.uint8)
    demo_syms[::3] = tr.modem.levels - 1
    raster = modem.modulate(demo_syms)
    sig = gen.generate_frame(raster)
    chan = CVBSChannel(cvbs_preset(preset_name), prof)
    rx, _ = chan.apply(sig, experiment_rng(cfg.seed, "ui-cvbs"))
    start = prof.line_samples * 30
    n = prof.line_samples * 2
    t = np.arange(n) / prof.sample_rate_hz * 1e6
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, sig[start:start + n], lw=0.9, label="передано")
    ax.plot(t, rx[start:start + n], lw=0.7, alpha=0.75, label="прийнято")
    ax.axhline(prof.sync_level, ls=":", lw=0.8, color="gray")
    ax.axhline(prof.blank_level, ls=":", lw=0.8, color="gray")
    ax.axhline(prof.white_level, ls=":", lw=0.8, color="gray")
    ax.set_xlabel("час, мкс")
    ax.set_ylabel("рівень, В")
    ax.set_title(f"CVBS 625/50: два рядки, профіль каналу «{preset_name}»")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    res["waveform"] = plot_data_url(fig)
    plt.close(fig)
    res["profile_description"] = prof.describe()
    return res


def action_tune(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    from avsec.experiments import run_tuning
    from avsec.optimization import ParameterSpace

    cfg = _config(payload)
    sp = payload.get("space") or {}
    space = ParameterSpace(
        stripe_height=tuple(sp.get("stripe_height", (16, 24))),
        n_descriptions=tuple(sp.get("n_descriptions", (1, 2))),
        codec=tuple(sp.get("codec", ("dct",))),
        quality=tuple(sp.get("quality", (15, 25))),
        max_unit_payload=tuple(sp.get("max_unit_payload", (320, 640))),
        fec_nsym=tuple(sp.get("fec_nsym", (96, 128))),
        modulation=tuple(tuple(m) for m in sp.get("modulation", [[4, 8, 2]])),
        interleaver=tuple(tuple(i) for i in sp.get(
            "interleaver", [["block", 278, 0], ["bawp", 0, 12]])),
    )
    out_dir = os.path.join(_RUNS_DIR, f"ui_tuning_{int(time.time())}")
    res = run_tuning(cfg, out_dir, space, int(payload.get("frames", 1)), progress)
    res["output_dir"] = out_dir
    res.pop("record", None)
    return res


def action_ablate(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    from avsec.experiments import run_ablations

    cfg = _config(payload)
    out_dir = os.path.join(_RUNS_DIR, f"ui_ablations_{int(time.time())}")
    res = run_ablations(cfg, out_dir, progress)
    res["output_dir"] = out_dir
    res.pop("record", None)
    return res


def action_sweep(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    from avsec.experiments import run_sweeps

    cfg = _config(payload)
    out_dir = os.path.join(_RUNS_DIR, f"ui_sweeps_{int(time.time())}")
    presets = payload.get("presets") or ["clean", "mild", "moderate", "bursty", "harsh"]
    res = run_sweeps(cfg, out_dir, presets, progress)
    imgs = {}
    for k, p in (res.get("plots") or {}).items():
        try:
            import cv2

            imgs[k] = png_data_url(cv2.imread(p, cv2.IMREAD_COLOR), max_width=980)
        except Exception:
            pass
    res["plot_images"] = imgs
    res["output_dir"] = out_dir
    res.pop("record", None)
    return res


def action_report(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    from avsec.report import build_report

    src = payload.get("input_dir") or _RUNS_DIR
    out = payload.get("output_dir") or os.path.join("reports", "ui")
    path = build_report(src, out, payload.get("title", "Звіт avsec"))
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return {"path": path, "markdown": text}


def action_list_runs(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []
    if os.path.isdir(_RUNS_DIR):
        for name in sorted(os.listdir(_RUNS_DIR), reverse=True)[:60]:
            d = os.path.join(_RUNS_DIR, name)
            if os.path.isdir(d):
                out.append({"name": name, "path": d,
                            "files": sorted(os.listdir(d))[:20]})
    return {"runs": out, "runs_dir": _RUNS_DIR}


def action_load_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    from avsec.config import load_config

    path = payload.get("path")
    if not path:
        raise ValueError("не вказано шлях до конфігурації")
    if not os.path.isabs(path):
        path = os.path.join("configs", os.path.basename(path))
    return {"config": load_config(path).to_dict(), "path": path}


# ------------------------------------------------------- saved-run explorer
# Every one of these reads what `avsec analyze` / `avsec plots` already wrote.
# None of them runs an experiment, so the UI can never show a number the CLI
# would not produce.
def action_explorer_runs(payload: Dict[str, Any]) -> Any:
    from avsec.ui import explorer

    return {"runs": explorer.list_runs(payload.get("runs_dir") or _RUNS_DIR)}


def _run_dir(payload: Dict[str, Any]) -> str:
    d = payload.get("run") or payload.get("run_dir") or ""
    if not d:
        raise ValueError("не вказано каталог прогону")
    if not os.path.isdir(d):
        raise ValueError(f"каталог {d} не існує")
    return d


def action_explorer_overview(payload: Dict[str, Any]) -> Any:
    from avsec.ui import explorer

    return explorer.run_overview(_run_dir(payload))


def action_explorer_filters(payload: Dict[str, Any]) -> Any:
    from avsec.ui import explorer

    return explorer.filters(_run_dir(payload))


def action_explorer_summary(payload: Dict[str, Any]) -> Any:
    from avsec.ui import explorer

    return explorer.summary(_run_dir(payload), payload.get("methods"),
                            payload.get("channels"),
                            payload.get("metric", "psnr_full"))


def action_explorer_paired(payload: Dict[str, Any]) -> Any:
    from avsec.ui import explorer

    return explorer.paired(_run_dir(payload), payload.get("a", "P"),
                           payload.get("b", "B4"), payload.get("channel"),
                           payload.get("metric", "psnr_full"))


def action_explorer_scenes(payload: Dict[str, Any]) -> Any:
    from avsec.ui import explorer

    return explorer.per_scene(_run_dir(payload),
                              payload.get("methods") or ["P", "B4"],
                              payload.get("channel", "bursty"),
                              payload.get("metric", "psnr_full"))


def action_explorer_failures(payload: Dict[str, Any]) -> Any:
    from avsec.ui import explorer

    return explorer.failures(_run_dir(payload))


def action_explorer_figures(payload: Dict[str, Any]) -> Any:
    from avsec.ui import explorer

    return explorer.figure_catalogue(_run_dir(payload))


def action_explorer_figure(payload: Dict[str, Any]) -> Any:
    from avsec.ui import explorer

    return explorer.figure_image(_run_dir(payload), payload.get("figure", "G03"))


def action_explorer_export(payload: Dict[str, Any]) -> Any:
    from avsec.ui import explorer

    return explorer.figure_export(_run_dir(payload), payload.get("figure", "G03"),
                                  payload.get("format", "svg"))


def action_explorer_agemap(payload: Dict[str, Any]) -> Any:
    from avsec.ui import explorer

    return explorer.age_map(_run_dir(payload), payload.get("clip", ""),
                            payload.get("method", "P"),
                            payload.get("channel", "bursty"),
                            int(payload.get("frame_id", 0)))



def _frames_for(payload, cfg, n: int = 1) -> List[np.ndarray]:
    """``n`` frames of what the tab is pointed at: upload, UAV scene or pattern.

    Some measurements need a *second* frame of the same material - the
    multi-frame reuse attack, for one - so the frames come from one call rather
    than from two unrelated sources.  An upload is a still, so it repeats.
    """
    from avsec import sources as S

    H, W = cfg.frame_height, cfg.frame_width
    if payload.get("image_b64"):
        import cv2

        raw = base64.b64decode(payload["image_b64"].split(",")[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return [img] * n
    name = payload.get("pattern", "edges")
    if name.startswith("uav:"):
        from avsec.sources.drone import DRONE_SCENES, drone_suite

        want = name.split(":", 1)[1]
        picked = [d for d in DRONE_SCENES if d[0] == want] or None
        frames = drone_suite(H, W, n_frames=max(2, n), scenes=picked)[0].frames
        return [np.asarray(f) for f in frames[:n]]
    if name not in S.PATTERNS:
        raise ValueError(f"unknown pattern {name!r}")
    # a procedural pattern is deterministic, so a moving crop gives a second
    # frame that differs the way consecutive frames of a scene differ
    base = S.PATTERNS[name](H + 8, W + 8)
    return [np.ascontiguousarray(base[i * 4:i * 4 + H, i * 4:i * 4 + W])
            for i in range(n)]


def _frame_for(payload, cfg):
    """The single frame the interactive tabs work on."""
    return _frames_for(payload, cfg, 1)[0]


def _table_grid(table) -> Dict[str, Any]:
    """A 16x16 substitution table in the layout every cipher spec prints."""
    a = np.asarray(table, dtype=np.int64).reshape(16, 16)
    return {
        "rows": [[f"{int(v):02X}" for v in row] for row in a],
        "values": [int(v) for v in a.ravel()],
    }


# ------------------------------------------------- B2s: substitution + permutation
def action_substitution(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    """One frame through ``B2s``, the tables it used, and the attacks on it.

    The stages are shown separately because the point of the scheme is that two
    primitives act in different spaces: substitution changes values and leaves
    positions, permutation changes positions and leaves values.
    """
    from avsec.crypto import derive_session_keys, new_session_id
    from avsec.subst_lab import (_b2, _sanity, _stage_images, attack_table,
                                 error_amplification, histogram_facts,
                                 sbox_tables)
    from avsec.substitution import (SUBSTITUTION_MODES,
                                    SubstitutionPermutationScrambler)

    cfg = _config(payload)
    rows, cols = cfg.b2_grid
    mode = payload.get("mode", "block")
    source = payload.get("source", "random")
    if mode not in SUBSTITUTION_MODES:
        raise ValueError(f"mode must be one of {SUBSTITUTION_MODES}")

    sid = new_session_id()
    key = derive_session_keys(cfg.master_secret(), sid).key
    img = _frame_for(payload, cfg)

    progress("таблиця замін", 0.1, {})
    box, inv = sbox_tables(key, sid)
    sc = SubstitutionPermutationScrambler(rows, cols, key, sid, mode=mode,
                                          source=source)
    b2 = _b2(cfg, key, sid)

    progress("стадії перетворення", 0.3, {})
    stages = _stage_images(sc, b2, img)
    fit = sc._fit(img)
    glob = SubstitutionPermutationScrambler(rows, cols, key, sid, mode="session",
                                            source=source)

    result: Dict[str, Any] = {
        "kind": "substitution",
        "mode": mode,
        "source": source,
        "grid": {"rows": rows, "cols": cols, "blocks": rows * cols},
        "sbox": _table_grid(box),
        "inverse": _table_grid(inv),
        "sanity": _sanity(box, inv),
        "error_amplification": error_amplification(box),
        "histogram": histogram_facts(fit, b2.scramble(img, 0),
                                     glob.substitute(img, 0),
                                     sc.substitute(img, 0)),
        "images": {k: png_data_url(v) for k, v in stages.items()},
        "exact_recovery": bool(np.array_equal(fit, stages["5_restored"])),
    }

    if payload.get("run_attacks", True):
        progress("атаки на B2 і B2s", 0.55, {})
        pair = _frames_for(payload, cfg, 2)
        result["attacks"] = attack_table(cfg, key, sid,
                                         [sc._fit(f) for f in pair],
                                         source=source)
    return result


# ------------------------------------------------------- the computed S-box
def action_sbox(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    """The GF(2^8) table: checked against AES, measured, swept over the field.

    The table is *computed*, not stored, so the interesting questions are
    whether the computation reproduces the published table byte for byte and
    what its measured resistance is next to a keyed random table.
    """
    from avsec.crypto import derive_session_keys, new_session_id
    from avsec.galois import (GF256, KeyedAlgebraicSbox, algebraic_sbox,
                              inverse_algebraic_sbox)
    from avsec.galois_lab import polynomial_sweep, property_table, verify_against_aes
    from avsec.substitution import crypto_sbox

    sid = new_session_id()
    cfg = _config(payload)
    key = derive_session_keys(cfg.master_secret(), sid).key

    progress("звірка з AES", 0.15, {})
    gf = GF256()
    box = algebraic_sbox(gf)
    inv = inverse_algebraic_sbox(gf)
    keyed = KeyedAlgebraicSbox(key, gf).table(b"|" + sid + b"|session")
    rnd = crypto_sbox(key, b"|" + sid + b"|session")

    progress("виміряні властивості", 0.45, {})
    res: Dict[str, Any] = {
        "kind": "sbox",
        "verification": verify_against_aes(),
        "field": gf.describe(),
        "properties": property_table(key, sid),
        "tables": {
            "algebraic": _table_grid(box),
            "inverse": _table_grid(inv),
            "keyed": _table_grid(keyed),
            "random": _table_grid(rnd),
        },
    }
    if payload.get("run_sweep", True):
        progress("розгортка за незвідними многочленами", 0.7, {})
        res["polynomials"] = polynomial_sweep()
    return res


# ------------------------------------------ B2l: constant-luminance ciphertext
def action_luma(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    """``B2l`` on one frame: what the monochrome observer gets, and the tract.

    The luminance plane of the ciphertext is the whole point, so it is shown as
    an image next to its statistics: a reader can see the flat grey field and
    read the entropy that says it carries nothing.
    """
    from avsec.crypto import derive_session_keys, new_session_id
    from avsec.luma_balance import (BEST_LEVEL, LumaBalancedMono, chroma_planes,
                                    luma_statistics, palette_capacity)
    from avsec.luma_lab import _to_grey, plane_attacks, tract_example

    cfg = _config(payload)
    sid = new_session_id()
    key = derive_session_keys(cfg.master_secret(), sid).key
    level = int(payload.get("level", BEST_LEVEL))
    img = _frame_for(payload, cfg)

    progress("шифрування зі сталою яскравістю", 0.15, {})
    mono = LumaBalancedMono(key, sid, level)
    ct = mono.encrypt(img, 0)
    back = mono.decrypt(ct, 0)
    pl = chroma_planes(ct)
    stats = luma_statistics(ct)

    res: Dict[str, Any] = {
        "kind": "luma",
        "level": level,
        "capacity": palette_capacity(level),
        "statistics": stats,
        "exact_recovery": bool(np.array_equal(img, back)),
        "images": {
            "1_original": png_data_url(img),
            "2_ciphertext": png_data_url(ct),
            "3_cipher_luma": png_data_url(
                np.clip(np.rint(pl["Y"]), 0, 255).astype(np.uint8)),
            "4_cipher_cb": png_data_url(_to_grey(pl["Cb"])),
            "5_restored": png_data_url(back),
        },
    }

    if payload.get("run_attacks", True):
        progress("атаки за площинами з контролем", 0.45, {})
        res["plane_attacks"] = plane_attacks(cfg, key, sid, img, level)

    if payload.get("run_tract", True):
        progress("наскрізний прогін через тракт", 0.7, {})
        images, rows = tract_example(cfg, cfg.master_secret(), img, "village",
                                     level=level,
                                     source_bits=int(payload.get("source_bits", 8)))
        res["tract"] = rows
        res["tract_images"] = {k: png_data_url(v, max_width=320)
                               for k, v in images.items()
                               if not k.endswith("_damage")}
    return res


# ------------------------------------------------------------------- checks
def action_checks(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    """Every check the project can run, in one place, with its own verdict.

    These existed only on the command line.  Each group reports how many checks
    passed out of how many ran, and the failures are listed rather than
    summarised, so a red group can be acted on without leaving the page.
    """
    import subprocess
    import sys as _sys

    groups: List[Dict[str, Any]] = []

    def group(name: str, cmd: str, doc: str, ok: bool, passed: int, total: int,
              failures: Optional[List[str]] = None, note: str = "") -> None:
        groups.append({"name": name, "command": cmd, "purpose": doc,
                       "ok": bool(ok), "passed": int(passed), "total": int(total),
                       "failures": failures or [], "note": note})

    cfg = _config(payload)

    progress("перевірки протоколу", 0.1, {})
    try:
        from avsec.experiments import run_protocol_checks

        checks = run_protocol_checks(cfg)
        n_ok = sum(1 for c in checks if c.get("passed"))
        group("Протокол AEAD і кадрування", "avsec protocol-check",
              "мутація кожного семантичного поля, межі парсера, повтори, "
              "зрив стану сеансу", n_ok == len(checks), n_ok, len(checks),
              [f"{c.get('name')}: очікувалось {c.get('expected')!r}, "
               f"отримано {c.get('observed')!r}"
               for c in checks if not c.get("passed")])
    except Exception as exc:
        group("Протокол AEAD і кадрування", "avsec protocol-check",
              "перевірки протоколу", False, 0, 0, [f"{type(exc).__name__}: {exc}"])

    progress("паспорт набору даних", 0.35, {})
    try:
        from avsec.dataset import build_manifest
        from avsec.experiments import build_sources

        report = build_manifest(build_sources(cfg), seed=cfg.seed).validate()
        probs = [str(x) for x in (report.get("problems") or [])]
        # warnings do not fail the check, but hiding them would misrepresent
        # the manifest: they are listed under the verdict, marked as such
        probs += [f"попередження: {w}" for w in (report.get("warnings") or [])]
        group("Паспорт набору даних", "avsec dataset",
              "дублікати, протікання між поділами, походження кожного джерела",
              bool(report.get("ok")),
              0 if report.get("problems") else 1, 1, probs,
              note=(f"джерел: {report.get('n_sources', '—')}, "
                    f"сцен: {report.get('n_scenes', '—')}, "
                    f"кліпів: {report.get('n_clips', '—')}"))
    except Exception as exc:
        group("Паспорт набору даних", "avsec dataset", "перевірка паспорта",
              False, 0, 0, [f"{type(exc).__name__}: {exc}"])

    run_dir = payload.get("input_dir") or "results/main"
    progress("перерахунок опублікованих чисел", 0.6, {})
    if os.path.isdir(run_dir):
        try:
            from avsec.verify import verify

            res = verify(run_dir, output=None)
            checks = res.get("checks") or []
            n_ok = sum(1 for c in checks if c.get("ok"))
            group("Опубліковані числа", f"avsec verify --input {run_dir}",
                  "кожне опубліковане число перераховано з таблиць прогону",
                  bool(res.get("ok")), n_ok, len(checks),
                  [str(c.get("kind")) + ": " + str(c.get("channel", ""))
                   for c in checks if not c.get("ok")],
                  note=f"метрика {res.get('plan', {}).get('metric', '—')}")
        except Exception as exc:
            group("Опубліковані числа", f"avsec verify --input {run_dir}",
                  "перерахунок чисел", False, 0, 0,
                  [f"{type(exc).__name__}: {exc}"])
    else:
        group("Опубліковані числа", f"avsec verify --input {run_dir}",
              "перерахунок чисел", False, 0, 0,
              [f"каталог {run_dir} не знайдено"])

    progress("перевірки документації", 0.85, {})
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    for name, script, doc in (
            ("Посилання в документації", "check_links.py",
             "кожне відносне посилання й кожен якір мають вести кудись"),
            ("Формули в документації", "check_math.py",
             "формули мають дійти до читача в будь-якому переглядачі")):
        path = os.path.join(root, "scripts", script)
        if not os.path.exists(path):
            continue
        try:
            r = subprocess.run([_sys.executable, path], capture_output=True,
                               text=True, cwd=root, timeout=180)
            tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()]
            group(name, f"python scripts/{script}", doc, r.returncode == 0,
                  1 if r.returncode == 0 else 0, 1,
                  [] if r.returncode == 0 else tail[:20],
                  note=tail[-1] if tail else "")
        except Exception as exc:
            group(name, f"python scripts/{script}", doc, False, 0, 0,
                  [f"{type(exc).__name__}: {exc}"])

    return {
        "kind": "checks",
        "groups": groups,
        "all_passed": all(g["ok"] for g in groups),
        "n_groups": len(groups),
        "n_groups_passed": sum(1 for g in groups if g["ok"]),
    }


# --------------------------------------------------- stored bench results
#: Where each bench writes, and what the tab should show from it.
STORED_LABS: Dict[str, Dict[str, Any]] = {
    "b2s": {"dir": "results/b2s", "file": "b2s.json", "command": "run.bat subst-lab"},
    "gf": {"dir": "results/gf_sbox", "file": "gf_sbox.json", "command": "run.bat gf-lab"},
    "luma": {"dir": "results/luma", "file": "luma.json", "command": "run.bat luma-lab"},
}


def action_stored_lab(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Read a bench result already on disk, without running anything.

    A full bench takes minutes.  The tabs open on the stored run so the page
    has content immediately, and the run button recomputes it.  Provenance is
    passed through untouched: the reader sees which commit produced the numbers
    and whether the tree was clean.
    """
    spec = STORED_LABS.get(str(payload.get("which", "")))
    if spec is None:
        raise ValueError("unknown bench")
    path = os.path.join(spec["dir"], spec["file"])
    if not os.path.exists(path):
        return {"present": False, "path": path, "command": spec["command"]}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.pop("figures", None)
    return {"present": True, "path": path, "command": spec["command"],
            "data": sanitize_json(data)}


def action_full_lab(payload: Dict[str, Any], progress) -> Dict[str, Any]:
    """Run one of the three benches exactly as the command line runs it."""
    which = str(payload.get("which", ""))
    cfg = _config(payload)
    if which == "b2s":
        from avsec.subst_lab import run_subst_lab

        res = run_subst_lab(cfg, "results/b2s", progress=progress)
    elif which == "gf":
        from avsec.galois_lab import run_galois_lab

        res = run_galois_lab(cfg, "results/gf_sbox", progress=progress)
    elif which == "luma":
        from avsec.luma_lab import run_luma_lab

        res = run_luma_lab(cfg, "results/luma", progress=progress)
    else:
        raise ValueError("unknown bench")
    res.pop("figures", None)
    return {"which": which, "data": sanitize_json(res)}


ASYNC_ACTIONS: Dict[str, Callable[..., Any]] = {
    "demo": action_demo,
    "scramble": action_scramble,
    "comparison": action_comparison,
    "attacks": action_attacks,
    "cvbs": action_cvbs,
    "tune": action_tune,
    "ablate": action_ablate,
    "sweep": action_sweep,
    "report": action_report,
    "budget": action_budget,
    "substitution": action_substitution,
    "sbox": action_sbox,
    "luma": action_luma,
    "checks": action_checks,
    "full_lab": action_full_lab,
}

#: The long actions are also reachable synchronously.  The page uses this path
#: when it is driven by a script (``?sync=1``): a polling loop is fine for a
#: human watching a progress bar, but a headless browser fast-forwards timers
#: and screenshots the page mid-run.  One blocking request is deterministic.
def _sync(fn: Callable[..., Any]) -> Callable[[Dict[str, Any]], Any]:
    # a no-op progress sink, not None: the actions call it directly
    return lambda payload: fn(payload, lambda *a, **k: None)


SYNC_ACTIONS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "defaults": action_defaults,
    "runs": action_list_runs,
    "load_config": action_load_config,
    "stored_lab": action_stored_lab,
    # saved-run explorer
    "explorer_runs": action_explorer_runs,
    "explorer_overview": action_explorer_overview,
    "explorer_filters": action_explorer_filters,
    "explorer_summary": action_explorer_summary,
    "explorer_paired": action_explorer_paired,
    "explorer_scenes": action_explorer_scenes,
    "explorer_failures": action_explorer_failures,
    "explorer_figures": action_explorer_figures,
    "explorer_figure": action_explorer_figure,
    "explorer_export": action_explorer_export,
    "explorer_agemap": action_explorer_agemap,
    # blocking variants of the long actions, for scripted use
    "demo": _sync(action_demo),
    "budget": _sync(action_budget),
    "scramble": _sync(action_scramble),
    "attacks": _sync(action_attacks),
    "cvbs": _sync(action_cvbs),
    "substitution": _sync(action_substitution),
    "sbox": _sync(action_sbox),
    "luma": _sync(action_luma),
    "checks": _sync(action_checks),
}


# ------------------------------------------------------------------ handler
class Handler(BaseHTTPRequestHandler):
    server_version = f"avsec/{__version__}"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        pass

    # -- utils --------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(sanitize_json(obj), ensure_ascii=False, cls=NpEncoder,
                          allow_nan=False).encode("utf-8")
        self._send(code, body)

    # -- routes -------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._file(os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
        if path.startswith("/static/"):
            name = os.path.basename(path)
            full = os.path.join(STATIC_DIR, name)
            if os.path.isfile(full):
                ctype = ("text/css" if name.endswith(".css")
                         else "application/javascript" if name.endswith(".js")
                         else "application/octet-stream")
                return self._file(full, ctype)
            return self._json(404, {"error": "not found"})
        if path.startswith("/api/job/"):
            jid = path.rsplit("/", 1)[-1]
            with _LOCK:
                job = _JOBS.get(jid)
            if job is None:
                return self._json(404, {"error": "unknown job"})
            return self._json(200, job)
        if path == "/api/health":
            return self._json(200, {"ok": True, "version": __version__})
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except Exception as exc:
            return self._json(400, {"error": f"bad request body: {exc}"})

        if path.startswith("/api/sync/"):
            action = path.rsplit("/", 1)[-1]
            fn = SYNC_ACTIONS.get(action)
            if fn is None:
                return self._json(404, {"error": f"unknown action {action}"})
            try:
                return self._json(200, fn(payload))
            except Exception as exc:
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}",
                                        "traceback": traceback.format_exc()[-3000:]})

        if path.startswith("/api/run/"):
            action = path.rsplit("/", 1)[-1]
            fn = ASYNC_ACTIONS.get(action)
            if fn is None:
                return self._json(404, {"error": f"unknown action {action}"})
            jid = run_async(lambda prog: fn(payload, prog))
            return self._json(202, {"job": jid})

        return self._json(404, {"error": "not found"})

    def _file(self, full: str, ctype: str) -> None:
        if not os.path.isfile(full):
            return self._json(404, {"error": f"missing {os.path.basename(full)}"})
        with open(full, "rb") as fh:
            self._send(200, fh.read(), ctype)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          runs_dir: str = "runs") -> None:
    global _RUNS_DIR

    _RUNS_DIR = runs_dir
    os.makedirs(runs_dir, exist_ok=True)

    class _Server(ThreadingHTTPServer):
        # Windows lets a second process bind the same port with SO_REUSEADDR and
        # then silently serve stale code; fail loudly instead.
        allow_reuse_address = False

    try:
        httpd = _Server((host, port), Handler)
    except OSError as exc:
        raise SystemExit(f"port {port} is already in use ({exc}); stop the other "
                         f"avsec UI or pass --port") from exc
    url = f"http://{host}:{port}/"
    print(f"avsec UI: {url}")
    print("Ctrl+C щоб зупинити.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nзупинено")
    finally:
        httpd.server_close()


__all__ = ["serve", "Handler", "png_data_url"]
