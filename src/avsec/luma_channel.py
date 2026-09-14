"""The luminance-balanced scheme over the project's own analog path.

Why this module exists
----------------------
``docs/luma_balance.md`` measured the scheme against perturbations applied
directly to the ciphertext array.  That answers "how much amplitude error can
the codebook absorb", but not "what happens on the tract this project models",
which is the question that decides whether the scheme is usable here.

The obstacle is that the modelled path is monochrome: it carries one raster of
luma per slot and has no colour subcarrier.  A constant-luminance ciphertext is
a colour image, so it has to be carried some other way.  The honest way inside
the existing budget is to send the three colour planes in the **three raster
slots** the shared budget already provides - exactly the slots ``B0a-R`` uses
for repetition and the digital methods use for units.  The comparison then
stays fair: the same number of slots, the same channel traces, the same scenes.

What this changes about the measurement
---------------------------------------
Each plane travels in its own slot, so each meets an **independent** realisation
of the impairment.  A pixel's three components are therefore perturbed
independently, which moves the point in all three directions of the RGB space
at once.  That is harsher than the single-array perturbation of the earlier
section, and it is what a three-slot transmission would really do.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from avsec import evaluation as ev
from avsec.baselines import AnalogPictureMethod, MethodResult
from avsec.channel import resolve_trace
from avsec.evaluation import FrameMetrics
from avsec.luma_balance import BEST_LEVEL, LumaBalancedMono, luma

#: Colour planes are sent in this slot order; the schedule is public.
PLANE_SLOTS = (0, 1, 2)


class AnalogColourPlanesMethod(AnalogPictureMethod):
    """Carries an RGB frame as three luma rasters, one plane per slot.

    The receiver reassembles the three planes into a colour frame and hands it
    to the inverse transform.  A slot that produced no raster leaves its plane
    unavailable, and the pixels that depend on it are marked rather than
    guessed.
    """

    def __init__(self, name: str, transport, channel_cfg, frame_h: int,
                 frame_w: int, transform=None, inverse=None, timer=None,
                 notes: str = "") -> None:
        super().__init__(name, transport, channel_cfg, frame_h, frame_w,
                         transform=transform, inverse=inverse,
                         authenticated=False, timer=timer, notes=notes,
                         rasters_per_frame=len(PLANE_SLOTS))

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update({
            "family": "analog transport of three colour planes",
            "slots_used": len(PLANE_SLOTS),
            "confidentiality": "constant-luminance codebook",
            "note": ("одна площина на слот; кожна зустрічає НЕЗАЛЕЖНУ "
                     "реалізацію спотворення"),
        })
        return d

    def process(self, frame: np.ndarray, frame_id: int, trace: Any) -> MethodResult:
        t = self.timer
        tr = resolve_trace(trace, len(PLANE_SLOTS))
        with t(f"{self.name}.transform"):
            payload = self._transform(frame, frame_id)
        if payload.ndim != 3 or payload.shape[2] != 3:
            raise ValueError("the transform must produce an RGB frame")

        planes: List[Optional[np.ndarray]] = []
        first_tx = first_rx = None
        truth0 = None
        scores: List[float] = []
        for slot in PLANE_SLOTS:
            with t(f"{self.name}.modulation"):
                tx = self.carrier.transmit(
                    np.ascontiguousarray(payload[..., slot]))
            if first_tx is None:
                first_tx = tx
            got_plane = None
            for rx_raster, tru in self.channel.apply_stream(tx, tr.rng(frame_id, slot)):
                if truth0 is None:
                    truth0 = tru
                if rx_raster is None:
                    continue
                if first_rx is None:
                    first_rx = rx_raster
                with t(f"{self.name}.demodulation"):
                    got, info = self.carrier.receive(rx_raster,
                                                     (self.frame_h, self.frame_w))
                got_plane = got
                scores.append(float(info["sync_score"]))
                break
            planes.append(got_plane)

        received = sum(p is not None for p in planes)
        if received < len(PLANE_SLOTS):
            # a missing plane makes the codeword undecodable everywhere
            blank = np.full_like(frame, 128)
            avail = np.zeros_like(frame, dtype=bool)
            q = ev.quality_pair(frame, blank, avail)
            met = FrameMetrics(frame_id=frame_id, method=self.name,
                               rasters=len(PLANE_SLOTS),
                               sync_found=bool(scores and max(scores) > 0.3), **q)
            met.extra.update({
                "authenticated": False, "channel_trace": tr.trace_id,
                "planes_received": received,
                "coverage_meaning": "плоскість кольору втрачена, кадр не декодовано",
            })
            return MethodResult(self.name, frame_id, frame,
                                first_tx if first_tx is not None else frame,
                                first_rx if first_rx is not None else frame,
                                blank, avail, np.zeros_like(avail), met, truth0,
                                self.notes)

        rx_rgb = np.stack([np.asarray(p, dtype=np.uint8) for p in planes], axis=2)
        with t(f"{self.name}.inverse"):
            recon = self._inverse(rx_rgb, frame_id)
        recon = np.asarray(recon)[: self.frame_h, : self.frame_w]
        avail = np.ones_like(frame, dtype=bool)
        q = ev.quality_pair(frame, recon, avail)
        met = FrameMetrics(frame_id=frame_id, method=self.name,
                           rasters=len(PLANE_SLOTS),
                           sync_found=bool(scores and max(scores) > 0.3), **q)
        met.extra.update({
            "authenticated": False, "channel_trace": tr.trace_id,
            "planes_received": received,
            "cipher_luma_std": float(np.asarray(luma(rx_rgb)).std()),
            "coverage_meaning": "picture displayed, not verified",
        })
        return MethodResult(self.name, frame_id, frame, first_tx, first_rx,
                            recon, avail, np.zeros_like(avail), met, truth0,
                            self.notes)


def make_b2l(transport, channel_cfg, frame_h: int, frame_w: int, master,
             session_id: Optional[bytes] = None, level: int = BEST_LEVEL,
             timer=None, session_ids=None,
             source_bits: int = 8) -> AnalogColourPlanesMethod:
    """``B2l``: the monochrome luminance-balanced path over three raster slots."""
    from avsec.baselines import SecureSessionIds
    from avsec.crypto import derive_session_keys

    sid = session_id or (session_ids or SecureSessionIds()).next("B2l")
    keys = derive_session_keys(master, sid)
    sc = LumaBalancedMono(keys.key, sid, level, source_bits=source_bits)
    return AnalogColourPlanesMethod(
        f"B2l-{source_bits}bit", transport, channel_cfg, frame_h, frame_w,
        transform=lambda img, fid: sc.encrypt(img, fid),
        # the channel perturbs values, so the receiver decodes by nearest
        # codeword; exact table lookup collapses under any amplitude error
        inverse=lambda rgb, fid: sc.decrypt(rgb, fid, nearest=True),
        timer=timer,
        notes=(f"constant-luminance codebook at level {level}, {source_bits} "
               f"source bits, three colour planes over three raster slots; "
               f"no integrity tag"))


__all__ = ["PLANE_SLOTS", "AnalogColourPlanesMethod", "make_b2l"]
