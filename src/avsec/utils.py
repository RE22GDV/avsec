"""Small shared helpers: timing, hashing, bit packing, deterministic non-secret RNG."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- timing
class Stopwatch:
    """Wall-clock stopwatch based on perf_counter (never used as a virtual schedule)."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def reset(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0


class StageTimer:
    """Accumulates wall-clock time per named pipeline stage.

    Wall-clock cost of the simulator is a *different* metric from the virtual
    transmission schedule computed by :mod:`avsec.budget`; both are reported.
    """

    def __init__(self) -> None:
        self._acc: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._samples: Dict[str, List[float]] = defaultdict(list)
        self._stack: List[tuple] = []
        self._pending = "unnamed"

    def __call__(self, name: str) -> "StageTimer":
        self._pending = name
        return self

    def __enter__(self) -> "StageTimer":
        self._stack.append((self._pending, time.perf_counter()))
        return self

    def __exit__(self, *exc: Any) -> None:
        name, t0 = self._stack.pop()
        dt = time.perf_counter() - t0
        self._acc[name] += dt
        self._counts[name] += 1
        if len(self._samples[name]) < 100000:
            self._samples[name].append(dt)

    def merge(self, other: "StageTimer") -> None:
        for k, v in other._acc.items():
            self._acc[k] += v
            self._counts[k] += other._counts[k]
            self._samples[k].extend(other._samples[k][:10000])

    def totals(self) -> Dict[str, float]:
        return dict(self._acc)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k, v in self._acc.items():
            s = np.asarray(self._samples[k], dtype=float)
            out[k] = {
                "total_s": float(v),
                "calls": int(self._counts[k]),
                "mean_ms": float(s.mean() * 1e3) if s.size else 0.0,
                "median_ms": float(np.median(s) * 1e3) if s.size else 0.0,
                "p95_ms": float(np.percentile(s, 95) * 1e3) if s.size else 0.0,
                "p99_ms": float(np.percentile(s, 99) * 1e3) if s.size else 0.0,
                "max_ms": float(s.max() * 1e3) if s.size else 0.0,
                "n_samples": int(s.size),
            }
        return out


# --------------------------------------------------------------------------- hashing
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------------- RNG
def experiment_rng(seed: int, *tags: Any) -> np.random.Generator:
    """Deterministic RNG for the *non-secret* parts of an experiment.

    Never use this for key material - see :mod:`avsec.crypto`.  The tags make
    independent streams (channel realisation, noise, source selection, ...).
    """
    material = ("|".join(str(t) for t in tags)).encode("utf-8")
    digest = hashlib.sha256(int(seed).to_bytes(8, "big", signed=False) + material).digest()
    return np.random.default_rng(np.frombuffer(digest, dtype=np.uint32).copy())


# --------------------------------------------------------------------------- bits
def bytes_to_symbols(data: bytes, bits_per_symbol: int) -> np.ndarray:
    """MSB-first packing of bytes into `bits_per_symbol`-bit symbols."""
    if bits_per_symbol not in (1, 2, 4, 8):
        raise ValueError("bits_per_symbol must be 1, 2, 4 or 8")
    arr = np.frombuffer(data, dtype=np.uint8)
    if bits_per_symbol == 8:
        return arr.astype(np.uint8).copy()
    bits = np.unpackbits(arr)
    grouped = bits.reshape(-1, bits_per_symbol).astype(np.uint16)
    weights = (1 << np.arange(bits_per_symbol - 1, -1, -1)).astype(np.uint16)
    return (grouped * weights).sum(axis=1).astype(np.uint8)


def symbols_to_bytes(symbols: np.ndarray, bits_per_symbol: int) -> bytes:
    if bits_per_symbol == 8:
        return np.asarray(symbols, dtype=np.uint8).tobytes()
    sym = np.asarray(symbols, dtype=np.uint8)
    nbits = sym.size * bits_per_symbol
    usable = (nbits // 8) * 8
    bits = np.zeros((sym.size, bits_per_symbol), dtype=np.uint8)
    for i in range(bits_per_symbol):
        bits[:, i] = (sym >> (bits_per_symbol - 1 - i)) & 1
    flat = bits.reshape(-1)[:usable]
    return np.packbits(flat).tobytes()


def public_whiten(data: bytes, seq: int) -> bytes:
    """XOR with a PUBLIC keystream derived from the unit sequence number.

    Diagnostic control only (defect F15).  Both endpoints can compute it and so
    can an eavesdropper: it equalises symbol statistics between an encrypted and
    an unencrypted transport so that a measured difference cannot be explained
    by "one of them looks random".  It provides no confidentiality whatsoever.
    """
    if not data:
        return data
    seed = hashlib.sha256(b"avsec/public-whitening|" + int(seq).to_bytes(8, "big")).digest()
    stream = bytearray()
    block = seed
    while len(stream) < len(data):
        block = hashlib.sha256(block).digest()
        stream += block
    return bytes(a ^ b for a, b in zip(data, stream[: len(data)]))


def symbols_needed(n_bytes: int, bits_per_symbol: int) -> int:
    return int(np.ceil(n_bytes * 8 / bits_per_symbol))


# --------------------------------------------------------------------------- env
def environment_record() -> Dict[str, Any]:
    """Platform / dependency record stored with every experiment run."""
    versions: Dict[str, str] = {}
    for mod in ("numpy", "scipy", "cv2", "cryptography", "reedsolo", "PIL", "matplotlib", "yaml"):
        try:
            m = __import__(mod)
            versions[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:  # pragma: no cover - optional deps
            versions[mod] = "missing"
    rec = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
        "avsec_version": __import__("avsec").__version__,
    }
    rec["git_commit"] = _git_commit()
    rec.update(_git_worktree())
    rec["pip_freeze"] = _pip_freeze()
    return rec


def _git(args: Sequence[str], timeout: int = 10) -> Optional[str]:
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             timeout=timeout, cwd=root)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _git_commit() -> Optional[str]:
    return _git(["rev-parse", "HEAD"])


def _git_worktree() -> Dict[str, Any]:
    """Whether the code that produced a result was actually the committed code.

    A commit hash alone does not pin a run: if the working tree had uncommitted
    edits, the run used code that exists nowhere else.  The status is recorded
    so a reader can tell the difference (defect R11).
    """
    status = _git(["status", "--porcelain"])
    if status is None:
        return {"git_worktree": "unknown (no git)", "git_dirty": None,
                "git_dirty_files": []}
    files = [line[3:].strip() for line in status.splitlines() if line.strip()]
    return {
        "git_worktree": "clean" if not files else "DIRTY",
        "git_dirty": bool(files),
        "git_dirty_files": files[:100],
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_describe": _git(["describe", "--tags", "--always", "--dirty"]),
        "git_dirty_note": ("" if not files else
                           "УВАГА: робоче дерево змінене; результат отримано "
                           "кодом, якого немає в жодному коміті"),
    }


def _pip_freeze() -> List[str]:
    """Exact versions of everything installed, not only what we import."""
    out = None
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "freeze", "--local"],
                             capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            out = res.stdout
    except Exception:
        pass
    return sorted(l.strip() for l in (out or "").splitlines() if l.strip())



def open_table(path: str):
    """Open a CSV that may be stored gzipped, transparently.

    Evidence directories keep the per-frame tables as ``.csv.gz``: every row
    is there, which is what ``avsec verify`` needs, at a size a repository can
    carry.  Callers should not have to care which form they got.
    """
    import gzip

    if os.path.exists(path):
        return open(path, encoding="utf-8", newline="")
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt", encoding="utf-8", newline="")
    raise FileNotFoundError(path)


def table_exists(path: str) -> bool:
    return os.path.exists(path) or os.path.exists(path + ".gz")

# --------------------------------------------------------------------------- io
def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


class NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, bytes):
            return o.hex()
        return super().default(o)


def sanitize_json(obj: Any) -> Any:
    """Replace NaN/Inf with ``None`` so the result is valid JSON for any parser.

    Python's ``json`` emits bare ``NaN``/``Infinity``, which ``JSON.parse`` in a
    browser rejects; the web interface uses this before answering a request.
    """
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    return obj


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, cls=NpEncoder)


def append_jsonl(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, cls=NpEncoder) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> None:
    import csv

    if not rows:
        return
    cols = columns or sorted({k for r in rows for k in r})
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def percentiles(values: Iterable[float]) -> Dict[str, float]:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "min": float(a.min()),
    }


def mean_ci95(values: Iterable[float]) -> Dict[str, float]:
    """Mean with a normal-approximation 95% CI over *independent* samples."""
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    if a.size == 1:
        return {"n": 1, "mean": float(a[0]), "lo": float(a[0]), "hi": float(a[0]), "sd": 0.0}
    m = float(a.mean())
    sd = float(a.std(ddof=1))
    se = sd / float(np.sqrt(a.size))
    return {"n": int(a.size), "mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "sd": sd}


@dataclass
class RunRecord:
    """Bookkeeping written next to every experiment output."""

    config_id: str
    config: Dict[str, Any]
    environment: Dict[str, Any] = field(default_factory=environment_record)
    inputs: Dict[str, str] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "config": self.config,
            "environment": self.environment,
            "inputs": self.inputs,
            "counters": dict(self.counters),
        }
