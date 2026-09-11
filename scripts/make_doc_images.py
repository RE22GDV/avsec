"""Build the illustrations the README uses that are not research figures.

These are documentation assets, not results: a contact sheet of the real scenes
and a hero strip showing one frame travelling through the whole path.  They are
rendered from the same code and the same run seeds as everything else, so what
the README shows is what the pipeline actually produces.

Usage::

    python scripts/make_doc_images.py [run_dir]
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT = os.path.join(ROOT, "docs", "img")

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"savefig.dpi": 200, "font.size": 9,
                         "axes.grid": False})
    return plt


def _strip(panels, path: str, height: float = 2.5, colour_flags=None) -> str:
    """One row of captioned frames.

    Matplotlib is used rather than OpenCV because the captions are Cyrillic and
    OpenCV's bundled Hershey fonts have no glyphs for them - they render as
    question marks.  It also keeps the typography identical to the research
    figures.
    """
    plt = _plt()
    n = len(panels)
    widths = [p[0].shape[1] / p[0].shape[0] for p in panels]
    fig, axes = plt.subplots(1, n, figsize=(sum(widths) * height, height * 1.22),
                             gridspec_kw={"width_ratios": widths})
    axes = axes if n > 1 else [axes]
    for ax, (img, cap) in zip(axes, panels):
        if img.ndim == 3:
            ax.imshow(img[..., ::-1], interpolation="nearest")
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#cfd8dc")
        ax.set_xlabel(cap, fontsize=8.5, labelpad=5)
    fig.subplots_adjust(wspace=0.03)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    import cv2

    a = cv2.imread(path)
    return f"{path}  {a.shape[1]}x{a.shape[0]}"


def scene_sheet() -> str:
    """The ten real test scenes, so a reader sees what the material is."""
    plt = _plt()
    from avsec.sources.drone import drone_suite

    srcs = [s for s in drone_suite(192, 256, 4) if s.meta["split"] == "test"][:10]
    fig, axes = plt.subplots(2, 5, figsize=(13.0, 4.4))
    for ax, s in zip(axes.ravel(), srcs):
        ax.imshow(s.frames[0], cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#cfd8dc")
        ax.set_xlabel(f"{s.name.replace('uav_', '')}  ·  {s.meta['category']}",
                      fontsize=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.22)
    path = os.path.join(OUT, "drone_scenes.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    import cv2

    a = cv2.imread(path)
    return f"{path}  {a.shape[1]}x{a.shape[0]}"


def hero(run_dir: str, clip: str = "uav_dune_ridge", channel: str = "bursty",
         frame: int = 6) -> str:
    """One real frame through the whole path, for the top of the README."""
    from avsec.channel import ChannelTrace
    from avsec.config import load_config
    from avsec.evaluation import availability_overlay
    from avsec.experiments import build_methods, build_sources

    cfg_path = os.path.join(run_dir, "config.yaml")
    cfg = load_config(cfg_path if os.path.exists(cfg_path)
                      else os.path.join(ROOT, "configs", "research_drone.yaml"))
    sub = dataclasses.replace(cfg, methods=("P",), channel_preset=channel,
                              channel_overrides={})
    srcs = {s.name: s for s in build_sources(sub)}
    src = srcs.get(clip) or list(srcs.values())[0]
    m = build_methods(sub)["P"]
    trace = ChannelTrace(seed=cfg.channel_seed_value,
                         scene=f"{src.name}|{channel}", repetition=0,
                         profile=channel,
                         rasters_per_frame=cfg.budget.rasters_per_frame)
    m.reset()
    res = None
    for fi in range(min(frame, len(src.frames) - 1) + 1):
        res = m.process(src.frames[fi], fi, trace)
    stale = getattr(res, "stale", np.zeros_like(res.available))

    panels = [
        (res.original, "1. кадр з дрона"),
        (res.transmitted, "2. переданий растр:\nзашифровані дані як рівні яскравості"),
        (res.received, "3. прийнятий растр\nпісля аналогового каналу"),
        (res.reconstructed,
         f"4. реконструкція\nPSNR {res.metrics.psnr_full:.1f} дБ"),
        (availability_overlay(res.reconstructed, res.available, stale),
         f"5. зелене = перевірено ({res.metrics.coverage:.2f}),\nчервоне = домальовано"),
    ]
    out = _strip(panels, os.path.join(OUT, "hero_drone.png"), height=2.35)
    return (f"{out}  PSNR={res.metrics.psnr_full:.2f} "
            f"cov={res.metrics.coverage:.3f}")


def photo_preview() -> str:
    """The source photograph itself, downscaled for the documentation."""
    import cv2

    from avsec.sources.drone import load_photo

    img = load_photo(os.path.join(ROOT, "data", "real",
                                  "curonian_spit_epha_dune.jpg"))
    h, w = img.shape[:2]
    small = cv2.resize(img, (1100, int(round(h * 1100 / w))),
                       interpolation=cv2.INTER_AREA)
    path = os.path.join(OUT, "drone_photo.jpg")
    cv2.imwrite(path, small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f"{path}  {small.shape[1]}x{small.shape[0]}"


def main(argv: list) -> int:
    os.makedirs(OUT, exist_ok=True)
    run_dir = argv[0] if argv else os.path.join(ROOT, "runs", "drone")
    for fn, args in ((scene_sheet, ()), (photo_preview, ()), (hero, (run_dir,))):
        try:
            print("ok  ", fn(*args))
        except Exception as exc:
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
