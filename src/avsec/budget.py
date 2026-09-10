"""Channel budget, virtual transmission schedule and latency accounting.

Two different quantities are kept apart everywhere:

* the **virtual schedule** - how long the modelled signal would take on the
  wire, derived from the raster rate and the profile;
* the **wall-clock cost** - how long this simulator took, measured by
  :class:`avsec.utils.StageTimer`.

A simulation that runs for one second may model a completely different signal
interval, so the two are never added together or interchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from avsec.crypto import TAG_LEN
from avsec.fec import FECConfig
from avsec.framing import HEADER_LEN
from avsec.modem import ModemConfig


@dataclass
class TransportBudget:
    """Full accounting of one transport unit and of one raster."""

    unit_plain_bytes: int
    header_bytes: int
    tag_bytes: int
    header_encoded_bytes: int
    payload_encoded_bytes: int
    unit_wire_bytes: int
    unit_symbols: int
    raster_capacity_symbols: int
    units_per_raster: int
    unused_symbols: int
    bits_per_symbol: int
    raster_rate_hz: float

    # -- derived ----------------------------------------------------------
    @property
    def gross_bitrate_bps(self) -> float:
        return self.raster_capacity_symbols * self.bits_per_symbol * self.raster_rate_hz

    @property
    def wire_bitrate_bps(self) -> float:
        return self.units_per_raster * self.unit_wire_bytes * 8 * self.raster_rate_hz

    @property
    def payload_bitrate_bps(self) -> float:
        return self.units_per_raster * self.unit_plain_bytes * 8 * self.raster_rate_hz

    @property
    def efficiency(self) -> float:
        return self.payload_bitrate_bps / self.gross_bitrate_bps if self.gross_bitrate_bps else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "unit_plain_bytes": self.unit_plain_bytes,
            "header_bytes": self.header_bytes,
            "tag_bytes": self.tag_bytes,
            "header_encoded_bytes": self.header_encoded_bytes,
            "payload_encoded_bytes": self.payload_encoded_bytes,
            "unit_wire_bytes": self.unit_wire_bytes,
            "unit_symbols": self.unit_symbols,
            "raster_capacity_symbols": self.raster_capacity_symbols,
            "units_per_raster": self.units_per_raster,
            "unused_symbols": self.unused_symbols,
            "raster_fill_fraction": round(
                1.0 - self.unused_symbols / max(self.raster_capacity_symbols, 1), 4),
            "bits_per_symbol": self.bits_per_symbol,
            "raster_rate_hz": self.raster_rate_hz,
            "gross_bitrate_bps": round(self.gross_bitrate_bps, 1),
            "wire_bitrate_bps": round(self.wire_bitrate_bps, 1),
            "payload_bitrate_bps": round(self.payload_bitrate_bps, 1),
            "payload_efficiency": round(self.efficiency, 4),
            "overhead_breakdown_per_unit": {
                "payload": self.unit_plain_bytes,
                "aead_tag": self.tag_bytes,
                "header": self.header_bytes,
                "header_parity": self.header_encoded_bytes - self.header_bytes,
                "payload_parity": self.payload_encoded_bytes
                                  - (self.unit_plain_bytes + self.tag_bytes),
            },
        }


def compute_budget(
    modem: ModemConfig,
    fec_payload: FECConfig,
    fec_header: FECConfig,
    unit_plain_bytes: int,
    raster_rate_hz: float = 25.0,
) -> TransportBudget:
    """Exact per-unit and per-raster accounting for a fixed-size unit profile."""
    hdr_enc = fec_header.encoded_len(HEADER_LEN)
    pay_enc = fec_payload.encoded_len(unit_plain_bytes + TAG_LEN)
    wire = hdr_enc + pay_enc
    bps = modem.bits_per_symbol
    unit_symbols = int(np.ceil(wire * 8 / bps))
    cap = modem.capacity_symbols
    upr = cap // unit_symbols if unit_symbols else 0
    return TransportBudget(
        unit_plain_bytes=unit_plain_bytes,
        header_bytes=HEADER_LEN,
        tag_bytes=TAG_LEN,
        header_encoded_bytes=hdr_enc,
        payload_encoded_bytes=pay_enc,
        unit_wire_bytes=wire,
        unit_symbols=unit_symbols,
        raster_capacity_symbols=cap,
        units_per_raster=upr,
        unused_symbols=cap - upr * unit_symbols,
        bits_per_symbol=bps,
        raster_rate_hz=raster_rate_hz,
    )


@dataclass
class LatencyBudget:
    """Virtual latency contributions, in seconds of *modelled signal time*."""

    stripe_accumulation_s: float
    interleaver_accumulation_s: float
    serialisation_s: float
    reception_s: float
    display_hold_s: float

    @property
    def total_s(self) -> float:
        return (self.stripe_accumulation_s + self.interleaver_accumulation_s
                + self.serialisation_s + self.reception_s + self.display_hold_s)

    def to_dict(self) -> Dict[str, float]:
        return {
            "stripe_accumulation_s": round(self.stripe_accumulation_s, 6),
            "interleaver_accumulation_s": round(self.interleaver_accumulation_s, 6),
            "serialisation_s": round(self.serialisation_s, 6),
            "reception_s": round(self.reception_s, 6),
            "display_hold_s": round(self.display_hold_s, 6),
            "virtual_total_s": round(self.total_s, 6),
        }


def compute_latency(
    modem: ModemConfig,
    interleaver_rows: int,
    rasters_per_frame: int,
    source_frame_interval_s: float,
    stripe_height: int,
    source_height: int,
    raster_rate_hz: float = 25.0,
) -> LatencyBudget:
    """Virtual end-to-end latency of the modelled schedule.

    * stripe accumulation - the camera time needed to have one stripe available;
    * interleaver accumulation - the raster time spanned by the placement window;
    * serialisation - the time to transmit the rasters of one frame;
    * reception - one raster time before the last symbol of a unit is available;
    * display hold - one source frame interval.
    """
    line_time = 1.0 / (raster_rate_hz * modem.raster_height)
    stripe_s = source_frame_interval_s * (stripe_height / max(source_height, 1))
    inter_s = interleaver_rows * modem.symbol_height * line_time
    ser_s = rasters_per_frame / raster_rate_hz
    return LatencyBudget(
        stripe_accumulation_s=stripe_s,
        interleaver_accumulation_s=inter_s,
        serialisation_s=ser_s,
        reception_s=1.0 / raster_rate_hz,
        display_hold_s=source_frame_interval_s,
    )


def raster_overhead(modem: ModemConfig) -> Dict[str, object]:
    """Where every pixel of a raster goes.

    Defect F16 asks the budget diagram to account for *everything*: blanking,
    synchronisation rows, pilot columns, the symbol grid that is left unused
    because the raster is not an exact multiple of a cell, and the filler that
    pads the last partially used unit slot.  Only what remains is payload.
    """
    total_px = modem.raster_width * modem.raster_height
    active_px = ((modem.active_x1 - modem.active_x0)
                 * (modem.active_y1 - modem.active_y0))
    cell_px = modem.symbol_width * modem.symbol_height
    grid_px = modem.n_rows * modem.n_cols * cell_px
    sync_px = modem.sync_rows * modem.n_cols * cell_px
    pilot_px = 2 * modem.pilot_cols * modem.n_data_rows * cell_px
    return {
        "raster_pixels": total_px,
        "blanking_pixels": total_px - active_px,
        "active_pixels": active_px,
        "grid_remainder_pixels": active_px - grid_px,
        "sync_pixels": sync_px,
        "pilot_pixels": pilot_px,
        "data_cell_pixels": grid_px - sync_px - pilot_px,
        "data_symbols": modem.capacity_symbols,
        "fractions": {
            "blanking": round((total_px - active_px) / total_px, 4),
            "grid_remainder": round((active_px - grid_px) / total_px, 4),
            "sync": round(sync_px / total_px, 4),
            "pilots": round(pilot_px / total_px, 4),
            "data_cells": round((grid_px - sync_px - pilot_px) / total_px, 4),
        },
    }


def budget_waterfall(budget: "TransportBudget", modem: ModemConfig,
                     units_placed: Optional[int] = None) -> Dict[str, object]:
    """Symbol-level waterfall from raster capacity down to delivered payload.

    ``units_placed`` is the number of units the interleaver can actually place,
    which may be below ``units_per_raster`` - the difference is real lost
    capacity and appears here as ``placement_loss_symbols`` instead of being
    quietly dropped.
    """
    placed = budget.units_per_raster if units_placed is None else int(units_placed)
    placed = max(0, min(placed, budget.units_per_raster))
    cap = budget.raster_capacity_symbols
    bps = max(budget.bits_per_symbol, 1)
    used = placed * budget.unit_symbols
    sym_per_byte = 8.0 / bps
    hdr_sym = placed * budget.header_bytes * sym_per_byte
    hpar_sym = placed * (budget.header_encoded_bytes - budget.header_bytes) * sym_per_byte
    tag_sym = placed * budget.tag_bytes * sym_per_byte
    ppar_sym = placed * (budget.payload_encoded_bytes
                         - budget.unit_plain_bytes - budget.tag_bytes) * sym_per_byte
    pay_sym = placed * budget.unit_plain_bytes * sym_per_byte
    pack_sym = used - (hdr_sym + hpar_sym + tag_sym + ppar_sym + pay_sym)
    steps = [
        ("raster capacity", cap),
        ("placement loss", -((budget.units_per_raster - placed) * budget.unit_symbols)),
        ("unused tail (filler)", -(cap - budget.units_per_raster * budget.unit_symbols)),
        ("header", -hdr_sym),
        ("header parity", -hpar_sym),
        ("payload parity", -ppar_sym),
        ("AEAD tag", -tag_sym),
        ("byte->symbol padding", -pack_sym),
    ]
    return {
        "steps": [{"label": k, "symbols": round(float(v), 2)} for k, v in steps],
        "payload_symbols": round(float(pay_sym), 2),
        "units_placed": placed,
        "units_per_raster_nominal": budget.units_per_raster,
        "payload_fraction_of_capacity": round(float(pay_sym) / cap, 4) if cap else 0.0,
        "raster_overhead": raster_overhead(modem),
        "note": "symbols, not pixels; the raster_overhead block accounts for the "
                "pixels that never reach the symbol grid at all",
    }


def timeline_latency(modem: ModemConfig, interleaver_rows: int,
                     rasters_per_frame: int, source_fps: float,
                     stripe_height: int, source_height: int,
                     units_per_frame: int, raster_rate_hz: float = 25.0,
                     n_frames: int = 8, deadline_s: Optional[float] = None
                     ) -> Dict[str, object]:
    """Latency from **event timestamps**, not from a sum of stage durations.

    Defect F16: adding stripe accumulation, interleaver accumulation and
    serialisation double-counts the intervals during which those stages overlap.
    Here each frame is placed on a virtual clock and its latency is the
    difference ``display - capture``.  The old additive figure is returned next
    to it as ``additive_upper_bound`` so the two can be compared honestly.
    """
    from avsec.timeline import Timeline, TimingProfile

    stripes = max(1, int(np.ceil(source_height / max(stripe_height, 1))))
    profile = TimingProfile(
        raster_rate_hz=raster_rate_hz, total_lines=modem.raster_height,
        active_lines=modem.active_y1 - modem.active_y0,
        rasters_per_frame=rasters_per_frame, source_fps=source_fps,
        symbol_height=modem.symbol_height, label="level-A raster")
    tl = Timeline(profile)
    for f in range(int(n_frames)):
        tl.schedule_frame(f, stripes=stripes, units=max(1, int(units_per_frame)),
                          rasters=rasters_per_frame,
                          interleaver_rows=interleaver_rows,
                          verified_units=range(max(1, int(units_per_frame))),
                          display_deadline_s=deadline_s)
    additive = compute_latency(modem, interleaver_rows, rasters_per_frame,
                               1.0 / max(source_fps, 1e-9), stripe_height,
                               source_height, raster_rate_hz)
    summary = tl.summary()
    summary["additive_upper_bound"] = additive.to_dict()
    summary["overlap_saving_s"] = round(
        additive.total_s - summary["latency_s"]["mean"], 6)
    summary["stripes_per_frame"] = stripes
    summary["events"] = tl.to_rows()
    return summary


def raw_video_bitrate(width: int, height: int, fps: float, bits: int = 8) -> float:
    """Sanity check used in the documentation: uncompressed grayscale bitrate."""
    return width * height * fps * bits


def capacity_check(budget: TransportBudget, units_needed_per_frame: int,
                   fps: float) -> Dict[str, object]:
    """Does one frame's data fit in the declared capacity at the target rate?"""
    rasters_needed = int(np.ceil(units_needed_per_frame / max(budget.units_per_raster, 1)))
    rasters_available = budget.raster_rate_hz / max(fps, 1e-9)
    return {
        "units_needed_per_frame": units_needed_per_frame,
        "units_per_raster": budget.units_per_raster,
        "rasters_needed_per_frame": rasters_needed,
        "rasters_available_per_frame": round(rasters_available, 3),
        "fits": rasters_needed <= rasters_available,
        "achievable_fps": round(budget.raster_rate_hz / max(rasters_needed, 1), 3),
    }


__all__ = [
    "TransportBudget", "compute_budget", "LatencyBudget", "compute_latency",
    "raster_overhead", "budget_waterfall", "timeline_latency",
    "raw_video_bitrate", "capacity_check",
]
