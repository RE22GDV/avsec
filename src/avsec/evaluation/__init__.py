"""Metrics, aggregation and plots.

Quality is always reported twice, as required by the experimental protocol:

* over the **whole displayed picture**, with the fill policy that produced it;
* over the **verified, current** area only, together with that area's coverage.

Reporting only the second number would let a method inflate its score by
discarding hard regions, so a coverage-weighted view is mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------- quality
def mse(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    d = (x - y) ** 2
    if mask is not None:
        if not mask.any():
            return float("nan")
        return float(d[mask].mean())
    return float(d.mean())


def psnr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None,
         peak: float = 255.0) -> float:
    """PSNR in dB, or ``+inf`` when the reconstruction is bit-exact.

    A zero MSE means the frame came back exactly; substituting a finite
    stand-in such as 99 dB would report a measurement that was never made and
    would pull an average toward an arbitrary constant.  Callers therefore see
    ``inf``, record ``bit_exact`` alongside the MSE, and the statistics layer
    counts those frames separately instead of averaging them in.
    """
    m = mse(a, b, mask)
    if not np.isfinite(m):
        return float("nan")
    if m <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(peak * peak / m))


def ssim(a: np.ndarray, b: np.ndarray, win: int = 7, peak: float = 255.0,
         mask: Optional[np.ndarray] = None) -> float:
    """Uniform-window SSIM (Wang et al. 2004) with the standard constants."""
    from scipy import ndimage

    x = a.astype(np.float64)
    y = b.astype(np.float64)
    c1 = (0.01 * peak) ** 2
    c2 = (0.03 * peak) ** 2
    mu_x = ndimage.uniform_filter(x, win, mode="nearest")
    mu_y = ndimage.uniform_filter(y, win, mode="nearest")
    xx = ndimage.uniform_filter(x * x, win, mode="nearest") - mu_x * mu_x
    yy = ndimage.uniform_filter(y * y, win, mode="nearest") - mu_y * mu_y
    xy = ndimage.uniform_filter(x * y, win, mode="nearest") - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (xx + yy + c2)
    smap = num / np.maximum(den, 1e-12)
    if mask is not None:
        return float(smap[mask].mean()) if mask.any() else float("nan")
    return float(smap.mean())


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    n = min(x.size, y.size)
    x, y = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else 0.0


def signal_statistics(raster, active=None) -> Dict[str, float]:
    """Level, variance, RMS and crest factor of what is actually transmitted.

    Defect F15: an identical *amplitude range* is not the same as identical
    signal power.  Two schemes can use the window [40, 216] and still differ in
    mean level, variance and peak-to-average ratio, so those are measured and
    reported instead of being assumed equal.
    """
    a = np.asarray(raster, dtype=np.float64)
    if active is not None:
        x0, y0, x1, y1 = active
        a = a[y0:y1, x0:x1]
    flat = a.ravel()
    if flat.size == 0:
        return {}
    mean = float(flat.mean())
    rms = float(np.sqrt((flat ** 2).mean()))
    ac = flat - mean
    return {
        "mean_level": mean,
        "min_level": float(flat.min()),
        "max_level": float(flat.max()),
        "peak_to_peak": float(flat.max() - flat.min()),
        "rms": rms,
        "variance": float(flat.var()),
        "ac_rms": float(np.sqrt((ac ** 2).mean())),
        "crest_factor": float(np.abs(flat).max() / rms) if rms > 0 else float("nan"),
        "n_samples": int(flat.size),
    }


# ---------------------------------------------------- illustrative statistics
def entropy_bits(img: np.ndarray) -> float:
    h = np.bincount(np.asarray(img, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    p = h / max(h.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def npcr_uaci(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """NPCR / UACI.  Illustrative only - not a substitute for a threat model."""
    x = a.astype(np.int32)
    y = b.astype(np.int32)
    npcr = float((x != y).mean() * 100.0)
    uaci = float((np.abs(x - y) / 255.0).mean() * 100.0)
    return npcr, uaci


def adjacent_correlation(img: np.ndarray, direction: str = "horizontal",
                         n_pairs: int = 4096, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    h, w = img.shape
    if direction == "horizontal":
        ys = rng.integers(0, h, n_pairs)
        xs = rng.integers(0, w - 1, n_pairs)
        a, b = img[ys, xs], img[ys, xs + 1]
    elif direction == "vertical":
        ys = rng.integers(0, h - 1, n_pairs)
        xs = rng.integers(0, w, n_pairs)
        a, b = img[ys, xs], img[ys + 1, xs]
    else:
        ys = rng.integers(0, h - 1, n_pairs)
        xs = rng.integers(0, w - 1, n_pairs)
        a, b = img[ys, xs], img[ys + 1, xs + 1]
    return correlation(a, b)


# --------------------------------------------- descriptions: three questions
#: "Damaged" and "lost" are not the same thing, and a single joint-loss count
#: conflates them (defect R04).  A description can be hit by a burst and come
#: back perfectly, because that is what the FEC is for.  Three separate
#: questions are asked of every description of every stripe:
#:
#: 1. ``touched``      - distortion reached at least one of its symbols;
#: 2. ``fec_failed``   - the FEC could not reconstruct data it needed;
#: 3. ``unusable``     - after authentication and decoding, the reconstruction
#:                       cannot use it (a required segment never verified).
#:
#: Only the third is a loss.  Band ("stripe") loss is then split the same way,
#: because "the band is gone" can mean three different things.
BAND_OUTCOMES = ("complete", "partial_segments", "reduced_descriptions",
                 "no_usable_description")


@dataclass
class DescriptionOutcome:
    """What happened to one description of one stripe."""

    stripe_id: int
    desc_id: int
    n_segs: int = 0
    n_touched: int = 0
    n_fec_failed: int = 0
    n_verified: int = 0

    @property
    def touched(self) -> bool:
        return self.n_touched > 0

    @property
    def fec_failed(self) -> bool:
        return self.n_fec_failed > 0

    @property
    def usable(self) -> bool:
        """Every segment it needs verified, so the decoder may use it."""
        return self.n_segs > 0 and self.n_verified >= self.n_segs

    @property
    def partial(self) -> bool:
        return 0 < self.n_verified < self.n_segs


def description_accounting(outcomes: Iterable[DescriptionOutcome],
                           n_descriptions: int) -> Dict[str, float]:
    """Turn per-description outcomes into the three metrics and band states.

    The old ``joint_loss`` counted a band as lost when *both* descriptions were
    touched by a burst.  That is the wrong event: two touched descriptions that
    the FEC repaired lose nothing at all.  What the plot must count is the band
    being unusable for reconstruction, which is the last row here.
    """
    per_stripe: Dict[int, List[DescriptionOutcome]] = {}
    for o in outcomes:
        per_stripe.setdefault(o.stripe_id, []).append(o)

    n_desc_total = n_touched = n_fec = n_unusable = n_partial = 0
    bands = {k: 0 for k in BAND_OUTCOMES}
    for _stripe, descs in sorted(per_stripe.items()):
        usable = [d for d in descs if d.usable]
        n_desc_total += len(descs)
        n_touched += sum(1 for d in descs if d.touched)
        n_fec += sum(1 for d in descs if d.fec_failed)
        n_unusable += sum(1 for d in descs if not d.usable)
        n_partial += sum(1 for d in descs if d.partial)
        if not usable:
            bands["no_usable_description"] += 1
        elif any(d.partial for d in descs):
            bands["partial_segments"] += 1
        elif len(usable) < max(1, min(n_descriptions, len(descs))):
            bands["reduced_descriptions"] += 1
        else:
            bands["complete"] += 1

    n_bands = max(1, len(per_stripe))
    return {
        "desc_total": n_desc_total,
        "desc_touched": n_touched,
        "desc_fec_failed": n_fec,
        "desc_unusable": n_unusable,
        "desc_partial": n_partial,
        "desc_touched_frac": n_touched / max(1, n_desc_total),
        "desc_fec_failed_frac": n_fec / max(1, n_desc_total),
        "desc_unusable_frac": n_unusable / max(1, n_desc_total),
        "bands_total": len(per_stripe),
        **{f"band_{k}": v for k, v in bands.items()},
        "band_lost_frac": bands["no_usable_description"] / n_bands,
        "band_degraded_frac": (bands["partial_segments"]
                               + bands["reduced_descriptions"]) / n_bands,
    }


# ------------------------------------------------------------------ containers
@dataclass
class FrameMetrics:
    """Everything measured for one displayed frame."""

    frame_id: int
    method: str
    psnr_full: float = float("nan")
    ssim_full: float = float("nan")
    psnr_verified: float = float("nan")
    ssim_verified: float = float("nan")
    coverage: float = 0.0
    mse_full: float = float("nan")
    #: True when the displayed frame matched the source exactly.  Reported as a
    #: count, never converted into a finite PSNR stand-in (defect F18).
    bit_exact: bool = False
    stale_fraction: float = 0.0
    estimated_fraction: float = 0.0
    max_age_frames: float = 0.0
    units_sent: int = 0
    units_verified: int = 0
    units_rejected: int = 0
    status_counts: Dict[str, int] = field(default_factory=dict)
    symbol_errors_pre_fec: float = float("nan")
    symbol_error_denominator: int = 0
    corrected_symbols: int = 0
    payload_bytes: int = 0
    wire_bytes: int = 0
    rasters: int = 0
    sync_found: bool = True
    resynchronised: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        row = {k: v for k, v in self.__dict__.items() if k not in ("status_counts", "extra")}
        for k, v in self.status_counts.items():
            row[f"status_{k}"] = v
        for k, v in self.extra.items():
            row[f"x_{k}"] = v
        return row


def quality_pair(original: np.ndarray, rendered: np.ndarray,
                 available: np.ndarray) -> Dict[str, float]:
    """The mandatory two-view quality report, with MSE and exactness kept.

    ``mse_full`` is always finite and is the metric to use when a comparison
    must include bit-exact frames; ``bit_exact`` says whether this frame was
    reproduced without a single differing sample.
    """
    m_full = mse(original, rendered)
    return {
        "psnr_full": psnr(original, rendered),
        "ssim_full": ssim(original, rendered),
        "psnr_verified": psnr(original, rendered, mask=available),
        "ssim_verified": ssim(original, rendered, mask=available),
        "coverage": float(available.mean()),
        "mse_full": m_full,
        "bit_exact": bool(m_full <= 1e-12),
    }


def aggregate(rows: Sequence[Dict[str, Any]], by: str = "method",
              metrics: Sequence[str] = ("psnr_full", "ssim_full", "coverage")
              ) -> List[Dict[str, Any]]:
    """Group rows and summarise each metric with a mean and a 95% CI."""
    from avsec.utils import mean_ci95

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r.get(by), []).append(r)
    out: List[Dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rec: Dict[str, Any] = {by: key, "n_rows": len(items)}
        for m in metrics:
            vals = [it[m] for it in items if m in it and it[m] is not None]
            ci = mean_ci95(vals)
            rec[f"{m}_mean"] = ci.get("mean")
            rec[f"{m}_lo"] = ci.get("lo")
            rec[f"{m}_hi"] = ci.get("hi")
            rec[f"{m}_n"] = ci.get("n")
        out.append(rec)
    return out


def paired_difference(rows_a: Sequence[float], rows_b: Sequence[float]) -> Dict[str, float]:
    """Paired comparison over matched independent sequences."""
    from avsec.utils import mean_ci95

    a = np.asarray(list(rows_a), dtype=float)
    b = np.asarray(list(rows_b), dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return {"n": 0}
    d = a[:n] - b[:n]
    ci = mean_ci95(d)
    ci["significant_at_95"] = bool(np.isfinite(ci.get("lo", np.nan))
                                   and (ci["lo"] > 0 or ci["hi"] < 0))
    return ci


# ---------------------------------------------------------------------- plots
def save_line_plot(path: str, x: Sequence[float], series: Dict[str, Sequence[float]],
                   xlabel: str, ylabel: str, title: str,
                   errors: Optional[Dict[str, Sequence[float]]] = None) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from avsec.utils import ensure_dir
    import os

    ensure_dir(os.path.dirname(os.path.abspath(path)))
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=140)
    for name, ys in series.items():
        if errors and name in errors:
            ax.errorbar(x, ys, yerr=errors[name], marker="o", capsize=3, label=name)
        else:
            ax.plot(x, ys, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_bar_plot(path: str, labels: Sequence[str], values: Sequence[float],
                  ylabel: str, title: str,
                  errors: Optional[Sequence[float]] = None) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    from avsec.utils import ensure_dir

    ensure_dir(os.path.dirname(os.path.abspath(path)))
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=140)
    ax.bar(range(len(labels)), values, yerr=errors, capsize=3, color="#3b6ea5")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_image(path: str, img: np.ndarray) -> str:
    import cv2
    import os

    from avsec.utils import ensure_dir

    ensure_dir(os.path.dirname(os.path.abspath(path)))
    cv2.imwrite(path, img)
    return path


def availability_overlay(image: np.ndarray, available: np.ndarray,
                         stale: Optional[np.ndarray] = None) -> np.ndarray:
    """Colour map of the frame: green = verified, amber = stale, red = estimated."""
    import cv2

    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float64)
    overlay = np.zeros_like(bgr)
    overlay[..., 1] = 255.0                       # green base
    est = ~available
    if stale is not None:
        overlay[stale] = (0, 190, 255)            # amber (BGR)
        est = est & ~stale
    overlay[est] = (0, 0, 255)                    # red
    return np.clip(0.65 * bgr + 0.35 * overlay, 0, 255).astype(np.uint8)


__all__ = [
    "mse", "psnr", "ssim", "correlation", "signal_statistics",
    "entropy_bits", "npcr_uaci", "BAND_OUTCOMES", "DescriptionOutcome",
    "description_accounting",
    "adjacent_correlation", "FrameMetrics", "quality_pair", "aggregate",
    "paired_difference", "save_line_plot", "save_bar_plot", "save_image",
    "availability_overlay",
]
