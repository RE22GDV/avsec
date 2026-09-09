"""YAML/dict configuration shared by the library, the CLI and the web UI.

One schema, one loader, no duplicated logic: the UI builds the same
:class:`ExperimentConfig` objects that ``avsec ...`` builds from a YAML file.
"""
from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from avsec.channel import PRESETS, RasterChannelConfig, preset as channel_preset
from avsec.crypto import CryptoProfile
from avsec.fec import FECConfig
from avsec.interleaving import InterleaverConfig
from avsec.lfsr import DEFAULT_TAPS, LFSRConfig, ScramblerConfig
from avsec.optimization import SharedBudget
from avsec.source_coding import SourceCodingConfig
from avsec.transmitter import TransportConfig

ALL_METHODS = ("B0a", "B0d", "B1", "B2", "B3", "B4", "P")


@dataclass
class MethodProfile:
    """A frozen per-method configuration of the digital pipeline."""

    stripe_height: int = 24
    n_descriptions: int = 1
    codec: str = "dct"
    quality: int = 12
    max_unit_payload: int = 640
    fec_nsym: int = 96
    fec_header_nsym: int = 48
    modulation: Tuple[int, int, int] = (4, 8, 2)
    interleaver: Tuple[str, int, int] = ("block", 278, 0)

    def source_config(self) -> SourceCodingConfig:
        return SourceCodingConfig(
            stripe_height=self.stripe_height, n_descriptions=self.n_descriptions,
            codec=self.codec, quality=self.quality,
            max_unit_payload=self.max_unit_payload)

    def transport_config(self, budget: SharedBudget) -> TransportConfig:
        lv, cw, chh = self.modulation
        scheme, depth, burst = self.interleaver
        return TransportConfig(
            modem=budget.modem(lv, cw, chh),
            fec_payload=FECConfig(k=max(1, 255 - self.fec_nsym), nsym=self.fec_nsym),
            fec_header=FECConfig(k=52, nsym=self.fec_header_nsym),
            interleaver=InterleaverConfig(
                scheme=scheme, depth=max(1, depth), window_rows=0,
                burst_rows=max(1, burst) if scheme == "bawp" else 4),
            raster_rate_hz=budget.raster_rate_hz,
            max_rasters_per_frame=budget.rasters_per_frame,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["modulation"] = list(self.modulation)
        d["interleaver"] = list(self.interleaver)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MethodProfile":
        d = dict(d or {})
        if "modulation" in d:
            d["modulation"] = tuple(d["modulation"])
        if "interleaver" in d:
            il = list(d["interleaver"])
            while len(il) < 3:
                il.append(0)
            d["interleaver"] = (str(il[0]), int(il[1]), int(il[2]))
        known = {f for f in MethodProfile.__dataclass_fields__}
        return MethodProfile(**{k: v for k, v in d.items() if k in known})


# Selected by `avsec tune` on the calibration/validation splits of
# configs/tuning.yaml (bursty channel).  See runs/_tuning/families.json and
# docs/experiments.md for the search budget used for each family.
_TUNED_SINGLE = dict(stripe_height=24, n_descriptions=1, codec="dct", quality=12,
                     max_unit_payload=640, fec_nsym=96, fec_header_nsym=48,
                     modulation=(4, 8, 2), interleaver=("block", 278, 0))
_TUNED_MDC = dict(stripe_height=24, n_descriptions=2, codec="dct", quality=8,
                  max_unit_payload=320, fec_nsym=128, fec_header_nsym=48,
                  modulation=(4, 8, 2), interleaver=("bawp", 0, 8))

DEFAULT_PROFILES: Dict[str, MethodProfile] = {
    "B0d": MethodProfile(**_TUNED_SINGLE),
    "B3": MethodProfile(**_TUNED_SINGLE),
    "B4": MethodProfile(**_TUNED_SINGLE),
    "P": MethodProfile(**_TUNED_MDC),
}


@dataclass
class ExperimentConfig:
    """Everything a reproducible run needs."""

    name: str = "smoke"
    seed: int = 20240909
    frame_width: int = 256
    frame_height: int = 192
    n_frames: int = 4
    sources: Dict[str, Any] = field(default_factory=lambda: {"kind": "synthetic"})
    channel_preset: str = "moderate"
    channel_overrides: Dict[str, Any] = field(default_factory=dict)
    budget: SharedBudget = field(default_factory=SharedBudget)
    methods: Sequence[str] = ALL_METHODS
    profiles: Dict[str, MethodProfile] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_PROFILES))
    crypto: CryptoProfile = field(default_factory=CryptoProfile)
    lfsr: ScramblerConfig = field(default_factory=lambda: ScramblerConfig(
        grid_rows=12, grid_cols=16, lfsr=LFSRConfig(seed=44257), per_frame=False))
    b2_grid: Tuple[int, int] = (12, 16)
    fill: str = "interpolate"
    key_mode: str = "lab"          # 'lab' (reproducible benchmark) | 'secure'
    key_file: Optional[str] = None
    save_images: bool = True
    max_frames_per_source: int = 4
    channel_realisations: int = 1
    notes: str = ""

    # -- derived ------------------------------------------------------
    def channel_config(self) -> RasterChannelConfig:
        cfg = channel_preset(self.channel_preset)
        for k, v in (self.channel_overrides or {}).items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown channel field {k!r}")
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
        return cfg

    def profile(self, method: str) -> MethodProfile:
        return self.profiles.get(method, DEFAULT_PROFILES.get(method, MethodProfile()))

    def master_secret(self):
        from avsec.crypto import lab_master_secret, load_master_secret

        if self.key_mode == "lab":
            return lab_master_secret(self.seed)
        return load_master_secret(self.key_file)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "seed": self.seed,
            "frame": {"width": self.frame_width, "height": self.frame_height,
                      "n_frames": self.n_frames},
            "sources": self.sources,
            "channel": {"preset": self.channel_preset,
                        "overrides": self.channel_overrides},
            "budget": self.budget.describe(),
            "methods": list(self.methods),
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
            "crypto": self.crypto.describe(),
            "lfsr": self.lfsr.describe(),
            "b2_grid": list(self.b2_grid),
            "fill": self.fill,
            "key_mode": self.key_mode,
            "max_frames_per_source": self.max_frames_per_source,
            "channel_realisations": self.channel_realisations,
            "notes": self.notes,
        }


def _budget_from(d: Dict[str, Any]) -> SharedBudget:
    b = SharedBudget()
    for k, v in (d or {}).items():
        if hasattr(b, k):
            setattr(b, k, v)
    return b


def config_from_dict(d: Dict[str, Any]) -> ExperimentConfig:
    d = dict(d or {})
    cfg = ExperimentConfig()
    cfg.name = d.get("name", cfg.name)
    cfg.seed = int(d.get("seed", cfg.seed))
    fr = d.get("frame", {})
    cfg.frame_width = int(fr.get("width", cfg.frame_width))
    cfg.frame_height = int(fr.get("height", cfg.frame_height))
    cfg.n_frames = int(fr.get("n_frames", cfg.n_frames))
    cfg.sources = d.get("sources", cfg.sources)
    ch = d.get("channel", {})
    cfg.channel_preset = ch.get("preset", cfg.channel_preset)
    cfg.channel_overrides = ch.get("overrides", {}) or {}
    cfg.budget = _budget_from(d.get("budget", {}))
    cfg.methods = d.get("methods", list(cfg.methods))
    profs = copy.deepcopy(DEFAULT_PROFILES)
    for k, v in (d.get("profiles") or {}).items():
        profs[k] = MethodProfile.from_dict(v)
    cfg.profiles = profs
    cr = d.get("crypto", {}) or {}
    cfg.crypto = CryptoProfile(
        algorithm=cr.get("algorithm", "chacha20poly1305"),
        replay_window=int(cr.get("replay_window", 4096)),
        direction=cr.get("direction", "uplink"),
        stream_id=int(cr.get("stream_id", 0)))
    lf = d.get("lfsr", {}) or {}
    width = int(lf.get("width", 16))
    taps = tuple(lf.get("taps", DEFAULT_TAPS.get(width, DEFAULT_TAPS[16])))
    cfg.lfsr = ScramblerConfig(
        grid_rows=int(lf.get("grid_rows", 12)), grid_cols=int(lf.get("grid_cols", 16)),
        lfsr=LFSRConfig(width=width, taps=taps, seed=int(lf.get("seed", 44257)),
                        form=lf.get("form", "fibonacci")),
        variant=lf.get("variant", "fisher_yates"),
        size_policy=lf.get("size_policy", "pad"),
        per_frame=bool(lf.get("per_frame", False)))
    cfg.b2_grid = tuple(d.get("b2_grid", [12, 16]))
    cfg.fill = d.get("fill", cfg.fill)
    cfg.key_mode = d.get("key_mode", cfg.key_mode)
    cfg.key_file = d.get("key_file")
    cfg.save_images = bool(d.get("save_images", True))
    cfg.max_frames_per_source = int(d.get("max_frames_per_source", cfg.max_frames_per_source))
    cfg.channel_realisations = int(d.get("channel_realisations", cfg.channel_realisations))
    cfg.notes = d.get("notes", "")
    return cfg


def load_config(path: str) -> ExperimentConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = config_from_dict(data)
    cfg.notes = (cfg.notes + f" [loaded from {os.path.basename(path)}]").strip()
    return cfg


def dump_config(cfg: ExperimentConfig, path: str) -> str:
    import yaml

    from avsec.utils import ensure_dir

    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, allow_unicode=True, sort_keys=False)
    return path


__all__ = [
    "ALL_METHODS", "MethodProfile", "DEFAULT_PROFILES", "ExperimentConfig",
    "config_from_dict", "load_config", "dump_config",
]
