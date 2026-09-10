"""Event-based virtual time and honest memory accounting (defect F16).

Latency used to be a **sum** of stage durations.  That is wrong whenever stages
overlap: while raster *k* is being serialised, stripe *k+1* is already being
accumulated, so adding the two double-counts.  Here every stage records a
timestamp on a virtual clock and the latency of a frame is simply

    display_time(frame) - capture_time(frame)

which is a difference of timestamps, not a sum of durations.

Wall-clock simulator time is never added to this clock: it is a separate metric
with its own units (see :class:`avsec.utils.StageTimer`).

Memory is reported in three separate quantities, because they measure different
things and only one of them is "RAM":

* ``process_peak_bytes`` - peak allocation actually observed for the process;
* ``logical_buffer_bytes`` - what the *protocol* requires to be buffered
  (reassembly, interleaver window, display queue);
* ``model_array_bytes`` - what the *simulator* holds (rasters, CVBS sample
  arrays), which a real implementation would not need.

A limit expressed on one of them is never reported as a limit on another.
"""
from __future__ import annotations

import os
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Event kinds, in the order they normally occur for one frame.
EVENT_KINDS = (
    "capture",           # the source frame exists at the transmitter
    "stripe_ready",      # one stripe has been accumulated
    "unit_ready",        # one protected unit is sealed and FEC-encoded
    "tx_start",          # first symbol of a raster leaves
    "tx_end",            # last symbol of a raster leaves
    "rx_start",          # first symbol of a raster arrives
    "rx_end",            # last symbol arrives; demodulation may begin
    "unit_verified",     # AEAD verified a unit
    "unit_rejected",     # a unit was rejected, with a reason
    "assembled",         # the picture was assembled
    "display",           # the picture is shown
    "deadline_miss",     # the display deadline passed without new data
)


@dataclass
class Event:
    t: float                     # virtual seconds since the start of the run
    kind: str
    frame_id: int = -1
    raster: int = -1
    unit: int = -1
    detail: str = ""

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimingProfile:
    """Rates that anchor the virtual clock.  Level A and CVBS differ, so both
    are stated explicitly instead of being assumed the same."""

    raster_rate_hz: float = 25.0
    total_lines: int = 576          # lines carried per raster period
    active_lines: int = 560
    rasters_per_frame: int = 3
    source_fps: float = 8.333
    symbol_height: int = 2
    label: str = "level-A raster"

    @property
    def raster_period_s(self) -> float:
        return 1.0 / max(self.raster_rate_hz, 1e-9)

    @property
    def line_period_s(self) -> float:
        return self.raster_period_s / max(self.total_lines, 1)

    @property
    def frame_period_s(self) -> float:
        return 1.0 / max(self.source_fps, 1e-9)

    def symbol_row_period_s(self) -> float:
        return self.line_period_s * self.symbol_height

    def describe(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "raster_rate_hz": self.raster_rate_hz,
            "raster_period_ms": round(self.raster_period_s * 1e3, 4),
            "total_lines": self.total_lines, "active_lines": self.active_lines,
            "line_period_us": round(self.line_period_s * 1e6, 4),
            "rasters_per_frame": self.rasters_per_frame,
            "source_fps": self.source_fps,
            "frame_period_ms": round(self.frame_period_s * 1e3, 4),
        }


CVBS_625_50 = TimingProfile(raster_rate_hz=25.0, total_lines=625, active_lines=576,
                            label="level-B CVBS 625/50")


class Timeline:
    """Virtual-time event log for one clip."""

    def __init__(self, profile: Optional[TimingProfile] = None) -> None:
        self.profile = profile or TimingProfile()
        self.events: List[Event] = []

    def add(self, t: float, kind: str, **kw: Any) -> Event:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind!r}; have {EVENT_KINDS}")
        e = Event(float(t), kind, **kw)
        self.events.append(e)
        return e

    # ------------------------------------------------------- frame schedule
    def schedule_frame(self, frame_id: int, stripes: int, units: int,
                       rasters: int, interleaver_rows: int,
                       verified_units: Optional[Sequence[int]] = None,
                       display_deadline_s: Optional[float] = None) -> Dict[str, float]:
        """Lay one frame on the virtual clock and return its timestamps.

        Overlap is explicit: stripe accumulation of a frame runs *during* the
        camera's frame period, and raster serialisation starts as soon as the
        interleaver window is full - it does not wait for the last stripe.
        """
        p = self.profile
        t_capture = frame_id * p.frame_period_s
        self.add(t_capture, "capture", frame_id=frame_id)

        stripe_dt = p.frame_period_s / max(stripes, 1)
        for i in range(stripes):
            self.add(t_capture + (i + 1) * stripe_dt, "stripe_ready",
                     frame_id=frame_id, unit=i)

        # a unit is ready once its stripe is ready (coding cost is measured
        # separately as wall-clock, not charged to the virtual schedule)
        units_per_stripe = max(1, units // max(stripes, 1))
        for u in range(units):
            stripe = min(stripes - 1, u // max(units_per_stripe, 1))
            self.add(t_capture + (stripe + 1) * stripe_dt, "unit_ready",
                     frame_id=frame_id, unit=u)

        window_s = interleaver_rows * p.symbol_row_period_s()
        t_tx0 = t_capture + min(window_s, p.frame_period_s)
        for r in range(rasters):
            s = t_tx0 + r * p.raster_period_s
            self.add(s, "tx_start", frame_id=frame_id, raster=r)
            self.add(s + p.raster_period_s, "tx_end", frame_id=frame_id, raster=r)
            self.add(s, "rx_start", frame_id=frame_id, raster=r)
            self.add(s + p.raster_period_s, "rx_end", frame_id=frame_id, raster=r)

        t_last_rx = t_tx0 + rasters * p.raster_period_s
        for u in (verified_units or []):
            self.add(t_last_rx, "unit_verified", frame_id=frame_id, unit=int(u))
        self.add(t_last_rx, "assembled", frame_id=frame_id)
        t_display = t_last_rx
        self.add(t_display, "display", frame_id=frame_id)

        if display_deadline_s is not None and (t_display - t_capture) > display_deadline_s:
            self.add(t_display, "deadline_miss", frame_id=frame_id,
                     detail=f"latency {t_display - t_capture:.3f}s exceeds "
                            f"{display_deadline_s:.3f}s")
        return {"capture": t_capture, "tx_start": t_tx0, "last_rx": t_last_rx,
                "display": t_display, "latency_s": t_display - t_capture}

    # ------------------------------------------------------------- queries
    def latency(self, frame_id: int) -> float:
        cap = self._first(frame_id, "capture")
        disp = self._first(frame_id, "display")
        if cap is None or disp is None:
            return float("nan")
        return disp.t - cap.t

    def latencies(self) -> Dict[int, float]:
        frames = sorted({e.frame_id for e in self.events if e.frame_id >= 0})
        return {f: self.latency(f) for f in frames}

    def deadline_misses(self) -> int:
        return sum(1 for e in self.events if e.kind == "deadline_miss")

    def _first(self, frame_id: int, kind: str) -> Optional[Event]:
        for e in self.events:
            if e.frame_id == frame_id and e.kind == kind:
                return e
        return None

    def summary(self) -> Dict[str, Any]:
        lat = np.asarray(list(self.latencies().values()), dtype=float)
        lat = lat[np.isfinite(lat)]
        return {
            "profile": self.profile.describe(),
            "n_events": len(self.events),
            "n_frames": int(lat.size),
            "latency_s": {
                "mean": float(lat.mean()) if lat.size else float("nan"),
                "median": float(np.median(lat)) if lat.size else float("nan"),
                "p95": float(np.percentile(lat, 95)) if lat.size else float("nan"),
                "max": float(lat.max()) if lat.size else float("nan"),
            },
            "deadline_misses": self.deadline_misses(),
            "note": "virtual schedule from event timestamps; wall-clock simulator "
                    "time is a different metric and is never added to it",
        }

    def to_rows(self) -> List[Dict[str, Any]]:
        return [e.to_row() for e in self.events]


# ------------------------------------------------------------------- memory
def process_peak_bytes() -> Tuple[int, str]:
    """Best-effort peak memory of this process, with the method named.

    Returns ``(bytes, method)``.  The method string is reported alongside the
    number so nobody mistakes a Python-allocation figure for process RSS.
    """
    try:  # Windows
        import ctypes
        import ctypes.wintypes as wt

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        # HANDLE is pointer sized; leaving the default c_int restype truncates
        # the pseudo-handle and the call silently fails.
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(_PMC), wt.DWORD]
        psapi.GetProcessMemoryInfo.restype = wt.BOOL
        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                      ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize), "GetProcessMemoryInfo/PeakWorkingSetSize"
    except Exception:
        pass
    try:  # Unix
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if os.uname().sysname == "Darwin" else 1024
        return int(ru * scale), "getrusage/ru_maxrss"
    except Exception:
        pass
    if tracemalloc.is_tracing():
        return int(tracemalloc.get_traced_memory()[1]), "tracemalloc (python allocations only)"
    return 0, "unavailable"


@dataclass
class MemoryReport:
    """Three separate quantities; none of them stands in for another."""

    process_peak_bytes: int = 0
    process_method: str = "unavailable"
    python_peak_bytes: int = 0
    logical_buffers: Dict[str, int] = field(default_factory=dict)
    model_arrays: Dict[str, int] = field(default_factory=dict)

    @property
    def logical_total_bytes(self) -> int:
        return int(sum(self.logical_buffers.values()))

    @property
    def model_total_bytes(self) -> int:
        return int(sum(self.model_arrays.values()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_peak_bytes": self.process_peak_bytes,
            "process_peak_mb": round(self.process_peak_bytes / 1e6, 3),
            "process_method": self.process_method,
            "python_peak_bytes": self.python_peak_bytes,
            "python_peak_mb": round(self.python_peak_bytes / 1e6, 3),
            "logical_buffers": dict(self.logical_buffers),
            "logical_total_kb": round(self.logical_total_bytes / 1024, 2),
            "model_arrays": dict(self.model_arrays),
            "model_total_mb": round(self.model_total_bytes / 1e6, 3),
            "note": "the protocol buffer limit applies to logical_total_kb only; "
                    "it is NOT a limit on process memory, and the model arrays "
                    "are simulator overhead a real implementation would not hold",
        }


class MemoryProbe:
    """Context manager measuring Python peak allocation around a block."""

    def __init__(self) -> None:
        self.report = MemoryReport()
        self._started_here = False

    def __enter__(self) -> "MemoryProbe":
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_here = True
        else:
            tracemalloc.reset_peak()
        return self

    def __exit__(self, *exc: Any) -> None:
        _, peak = tracemalloc.get_traced_memory()
        self.report.python_peak_bytes = int(peak)
        if self._started_here:
            tracemalloc.stop()
        self.report.process_peak_bytes, self.report.process_method = process_peak_bytes()


def logical_buffers(unit_wire_bytes: int, units_per_raster: int,
                    rasters_per_frame: int, interleaver_rows: int,
                    n_data_cols: int, bits_per_symbol: int,
                    display_queue_frames: int, frame_bytes: int) -> Dict[str, int]:
    """What the *protocol* must hold, itemised.

    This is the quantity the shared budget constrains.  It deliberately excludes
    the simulator's own raster and CVBS arrays.
    """
    interleave_bits = interleaver_rows * n_data_cols * bits_per_symbol
    return {
        "reassembly_units": int(unit_wire_bytes * units_per_raster * rasters_per_frame),
        "interleaver_window": int(np.ceil(interleave_bits / 8)),
        "display_queue": int(frame_bytes * max(1, display_queue_frames)),
    }


def model_arrays(raster_h: int, raster_w: int, rasters_per_frame: int,
                 cvbs_samples: int = 0) -> Dict[str, int]:
    return {
        "transmitted_rasters": int(raster_h * raster_w * rasters_per_frame),
        "received_rasters": int(raster_h * raster_w * rasters_per_frame),
        "cvbs_samples_float32": int(cvbs_samples * 4),
    }


__all__ = [
    "EVENT_KINDS", "Event", "TimingProfile", "CVBS_625_50", "Timeline",
    "MemoryReport", "MemoryProbe", "process_peak_bytes", "logical_buffers",
    "model_arrays",
]
