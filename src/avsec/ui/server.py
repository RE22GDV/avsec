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
def action_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    from avsec.channel import PRESETS
    from avsec.config import DEFAULT_PROFILES, ExperimentConfig
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
        "patterns": sorted(PATTERNS),
        "methods": ["B0a", "B0d", "B1", "B2", "B3", "B4", "P"],
        "interleaver_schemes": list(SCHEMES),
        "codecs": ["raw", "dct", "jpeg"],
        "default_config": cfg.to_dict(),
        "default_profiles": {k: v.to_dict() for k, v in DEFAULT_PROFILES.items()},
        "configs_dir": sorted(
            f for f in os.listdir("configs") if f.endswith(".yaml")
        ) if os.path.isdir("configs") else [],
        "method_notes": {
            "B0a": "незахищене зображення через аналоговий растр (опора для втрат якості)",
            "B0d": "той самий цифровий транспорт БЕЗ криптографії - діагностика, нічого не автентифікує",
            "B1": "перестановка блоків на LFSR (реконструкція статті 2021)",
            "B2": "та сама перестановка з криптогенератором - вміст блоків НЕ шифрується",
            "B3": "AEAD цілого кадру однією одиницею; потрібні всі фрагменти",
            "B4": "незалежно захищені смуги, один опис - сильний базовий метод",
            "P": "кілька описів + BAWP + спільний добір параметрів (запропоноване)",
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
}

SYNC_ACTIONS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "defaults": action_defaults,
    "runs": action_list_runs,
    "load_config": action_load_config,
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
