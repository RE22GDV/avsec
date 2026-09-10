"""Baselines B0-B4 and the proposed method P behind one common interface.

============ ===============================================================
Method       What it is
============ ===============================================================
``B0a``      Unprotected picture carried as an analog raster (reference for
             quality loss and added latency of the *analog* path).
``B0d``      The same digital transport with **no** cryptography - separates
             the price of the modem and the FEC from the price of crypto.
             Diagnostic only; it authenticates nothing.
``B1``       LFSR block permutation of the 2021 paper (our reconstruction),
             carried as an analog raster.
``B2``       The same permutation driven by a cryptographic generator with a
             unique per-frame context.  A stronger generator does **not** hide
             the content of the blocks; this baseline exists to show that.
``B3``       Authenticated encryption of a whole coded frame as a single AEAD
             unit, fragmented over the digital transport.  Authentication
             succeeds only after the complete AEAD unit is reassembled.
``B4``       Independently protected regular stripes (one description), tuned
             FEC and interleaving.  The mandatory strong baseline.
``P``        Multiple independent descriptions of a short stripe, independent
             AEAD per unit, and joint parameter selection with the proposed
             burst-aware placement (BAWP).
============ ===============================================================

``B4`` and ``P`` are the *same* pipeline class with different configurations,
which is what makes the comparison fair: the difference is the configuration
under test, not the implementation quality.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec import evaluation as ev
from avsec.channel import (
    ChannelTrace,
    ChannelTruth,
    RasterChannel,
    RasterChannelConfig,
    resolve_trace,
)
from avsec.crypto import (
    TAG_LEN,
    SecureSessionIds,
    SessionIdSource,
    AuthenticationFailed,
    CryptoProfile,
    MasterSecret,
    Opener,
    ReplayDetected,
    Sealer,
    derive_session_keys,
    new_session_id,
)
from avsec.evaluation import FrameMetrics
from avsec.fec import RSCodec
from avsec.framing import HEADER_LEN, FramingError, Geometry, UnitHeader, parse_header
from avsec.interleaving import Interleaver
from avsec.lfsr import BlockScrambler, ScramblerConfig, invert_permutation
from avsec.modem import RasterModem, prbs_symbols
from avsec.receiver import Receiver, UnitStatus, VerifiedUnit
from avsec.source_coding import (
    FILL_INTERPOLATE,
    AssembledFrame,
    FrameAssembler,
    SourceCodingConfig,
    StripeCoder,
)
from avsec.transmitter import CapacityExceeded, Transmitter, TransportConfig
from avsec.utils import StageTimer, bytes_to_symbols, symbols_to_bytes


# --------------------------------------------------------------------- result
@dataclass
class MethodResult:
    method: str
    frame_id: int
    original: np.ndarray
    transmitted: np.ndarray
    received: np.ndarray
    reconstructed: np.ndarray
    available: np.ndarray
    stale: np.ndarray
    metrics: FrameMetrics
    truth: Optional[ChannelTruth] = None
    notes: str = ""

    def images(self) -> Dict[str, np.ndarray]:
        return {
            "original": self.original,
            "transmitted": self.transmitted,
            "received": self.received,
            "reconstructed": self.reconstructed,
            "availability": ev.availability_overlay(self.reconstructed, self.available,
                                                    self.stale),
        }


class Method:
    """Common interface for every comparable method."""

    name = "base"
    authenticated = False

    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def process(self, frame: np.ndarray, frame_id: int,
                rng: np.random.Generator) -> MethodResult:
        raise NotImplementedError


# ----------------------------------------------------------- analog carrier
class AnalogCarrier:
    """Carry a picture in the raster's active area with a public alignment strip.

    Amplitude use is deliberately identical to the digital modem
    (``[level_low, level_high]``) so no scheme silently receives extra signal
    power.  The alignment strip is public, and the receiver *searches* for it
    exactly like the digital receiver does.
    """

    def __init__(self, raster_h: int, raster_w: int, x0: int, x1: int, y0: int, y1: int,
                 level_low: int = 40, level_high: int = 216, sync_lines: int = 16,
                 block: int = 8, guard_lines: int = 4) -> None:
        self.raster_h, self.raster_w = raster_h, raster_w
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.level_low, self.level_high = level_low, level_high
        self.sync_lines = sync_lines
        self.block = block
        # A blanking guard band under the strip (and the blanking above the active
        # window) makes the vertical offset uniquely observable.
        self.guard_lines = guard_lines
        self.img_y0 = y0 + sync_lines + guard_lines
        self.img_h = y1 - self.img_y0
        self.img_w = x1 - x0
        n_blocks = self.img_w // block
        pat = prbs_symbols(n_blocks, 2, 0x5A5A)
        strip = np.repeat(pat, block)[: self.img_w]
        self._strip_sym = strip
        self._strip = np.where(strip > 0, level_high, level_low).astype(np.float64)

    def _factors(self, src_h: int, src_w: int) -> Tuple[int, int]:
        """Integer pixel/line replication factors, as an analog line doubler does."""
        return max(1, self.img_h // max(src_h, 1)), max(1, self.img_w // max(src_w, 1))

    def transmit(self, image: np.ndarray) -> np.ndarray:
        import cv2

        raster = np.full((self.raster_h, self.raster_w), 16, dtype=np.uint8)
        src_h, src_w = image.shape
        fy, fx = self._factors(src_h, src_w)
        if fy * src_h <= self.img_h and fx * src_w <= self.img_w:
            scaled = np.repeat(np.repeat(image, fy, axis=0), fx, axis=1)
        else:  # source larger than the active area: fall back to area resampling
            scaled = cv2.resize(image, (self.img_w, self.img_h),
                                interpolation=cv2.INTER_AREA)
        lo, hi = self.level_low, self.level_high
        body = lo + scaled.astype(np.float64) * (hi - lo) / 255.0
        raster[self.y0 : self.y0 + self.sync_lines, self.x0 : self.x1] = np.tile(
            self._strip, (self.sync_lines, 1)).astype(np.uint8)
        h = min(body.shape[0], self.y1 - self.img_y0)
        w = min(body.shape[1], self.x1 - self.x0)
        raster[self.img_y0 : self.img_y0 + h, self.x0 : self.x0 + w] = \
            np.round(body[:h, :w]).astype(np.uint8)
        return raster

    def receive(self, raster: np.ndarray, out_shape: Tuple[int, int],
                search_x: int = 12, search_y: int = 8) -> Tuple[np.ndarray, Dict[str, float]]:
        import cv2

        # Per-row correlation with the known strip.  Averaging rows *before*
        # correlating would make the vertical offset unobservable (a constant row
        # does not change a correlation coefficient), so correlate first and
        # average the coefficients afterwards.
        ref = self._strip - self._strip.mean()
        ref_n = float(np.sqrt((ref * ref).sum())) or 1.0
        # The scoring window is *taller* than the strip, so rows that fall outside
        # it contribute zero correlation.  Without that the vertical offset would
        # be ambiguous by the strip margin and the picture would land a few lines
        # off even on a clean channel.
        guard = max(2, self.guard_lines)
        y_lo = self.y0 - search_y - guard
        y_hi = self.y0 + self.sync_lines + search_y + guard
        rows = np.clip(np.arange(y_lo, y_hi), 0, raster.shape[0] - 1)
        best = (-2.0, 0, 0)
        for dx in range(-search_x, search_x + 1):
            xs = np.clip(np.arange(self.x0 + dx, self.x1 + dx), 0, raster.shape[1] - 1)
            band = raster[np.ix_(rows, xs)].astype(np.float64)
            b = band - band.mean(axis=1, keepdims=True)
            den = np.sqrt((b * b).sum(axis=1)) * ref_n
            corr = np.where(den > 0, (b @ ref) / np.where(den > 0, den, 1.0), 0.0)
            for dy in range(-search_y, search_y + 1):
                a = (self.y0 + dy) - y_lo
                z = a + self.sync_lines
                lo_a, hi_z = a - guard, z + guard
                if lo_a < 0 or hi_z > corr.size:
                    continue
                inside = float(corr[a:z].mean())
                outside = float(np.concatenate([corr[lo_a:a], corr[z:hi_z]]).mean())
                # Differential matched score: the strip must correlate and its
                # neighbourhood must not.  A plain coverage score would let strongly
                # structured picture content win a few lines away from the strip.
                s = inside - outside
                if s > best[0]:
                    best = (s, dy, dx)
        score, dy, dx = best

        ys = np.clip(np.arange(self.y0 + dy + 2, self.y0 + dy + self.sync_lines - 2),
                     0, raster.shape[0] - 1)
        xs = np.clip(np.arange(self.x0 + dx, self.x1 + dx), 0, raster.shape[1] - 1)
        obs = raster[np.ix_(ys, xs)].astype(np.float64).mean(axis=0)
        gain, offset = _affine_fit(self._strip, obs)

        src_h, src_w = out_shape
        fy, fx = self._factors(src_h, src_w)
        use_h, use_w = fy * src_h, fx * src_w
        integer_path = use_h <= self.img_h and use_w <= self.img_w
        h_span = use_h if integer_path else self.img_h
        w_span = use_w if integer_path else self.img_w
        yy = np.clip(np.arange(self.img_y0 + dy, self.img_y0 + dy + h_span),
                     0, raster.shape[0] - 1)
        xs2 = np.clip(np.arange(self.x0 + dx, self.x0 + dx + w_span),
                      0, raster.shape[1] - 1)
        body = raster[np.ix_(yy, xs2)].astype(np.float64)
        body = (body - offset) / (gain if abs(gain) > 1e-6 else 1.0)
        norm = (body - self.level_low) * 255.0 / max(self.level_high - self.level_low, 1)
        norm = np.clip(norm, 0, 255)
        if integer_path:
            out = norm.reshape(src_h, fy, src_w, fx).mean(axis=(1, 3))
            out = np.clip(np.round(out), 0, 255).astype(np.uint8)
        else:
            out = cv2.resize(np.clip(norm, 0, 255).astype(np.uint8),
                             (src_w, src_h), interpolation=cv2.INTER_AREA)
        return out, {"sync_score": score, "dx": float(dx), "dy": float(dy),
                     "gain": gain, "offset": offset}


def _affine_fit(ref: np.ndarray, obs: np.ndarray) -> Tuple[float, float]:
    r = np.asarray(ref, dtype=np.float64).ravel()
    o = np.asarray(obs, dtype=np.float64).ravel()
    n = min(r.size, o.size)
    r, o = r[:n], o[:n]
    if n < 2 or r.std() < 1e-9:
        return 1.0, 0.0
    den = float(((r - r.mean()) ** 2).sum())
    if den < 1e-9:
        return 1.0, 0.0
    g = float(((r - r.mean()) * (o - o.mean())).sum() / den)
    if abs(g) < 1e-3:
        return 1.0, 0.0
    return g, float(o.mean() - g * r.mean())


# -------------------------------------------------- crypto permutation (B2)
def crypto_permutation(n: int, key: bytes, context: bytes) -> np.ndarray:
    """Fisher-Yates shuffle driven by a ChaCha20 keystream (unique per context).

    The keystream comes from the standard library implementation; only the
    sampling loop is ours.  Rejection sampling avoids modulo bias.
    """
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

    nonce = hashlib.sha256(context).digest()[:16]
    sub_key = hashlib.sha256(b"avsec/perm-key" + key + context).digest()
    enc = Cipher(algorithms.ChaCha20(sub_key, nonce), mode=None).encryptor()
    stream = enc.update(bytes(max(64, 8 * n + 256)))
    words = np.frombuffer(stream[: (len(stream) // 4) * 4], dtype="<u4")
    perm = np.arange(n, dtype=np.int64)
    wi = 0
    span = 1 << 32
    for i in range(n - 1, 0, -1):
        m = i + 1
        limit = (span // m) * m
        j = i
        while wi < words.size:
            v = int(words[wi])
            wi += 1
            if v < limit:
                j = v % m
                break
        perm[i], perm[j] = perm[j], perm[i]
    return perm


class CryptoPermutationScrambler:
    """Block permutation with a cryptographic generator and per-frame context."""

    def __init__(self, grid_rows: int, grid_cols: int, key: bytes,
                 session_id: bytes, per_frame: bool = True) -> None:
        self.rows, self.cols = grid_rows, grid_cols
        self.key = key
        self.session_id = session_id
        self.per_frame = per_frame
        self._cache: Dict[int, np.ndarray] = {}

    def permutation(self, frame_id: int) -> np.ndarray:
        key = frame_id if self.per_frame else 0
        if key not in self._cache:
            ctx = b"avsec/b2|" + self.session_id + b"|" + int(key).to_bytes(8, "big")
            self._cache[key] = crypto_permutation(self.rows * self.cols, self.key, ctx)
        return self._cache[key]

    def _tiles(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        bh, bw = h // self.rows, w // self.cols
        img = img[: bh * self.rows, : bw * self.cols]
        return img.reshape(self.rows, bh, self.cols, bw).swapaxes(1, 2).reshape(-1, bh, bw)

    def _join(self, tiles: np.ndarray) -> np.ndarray:
        bh, bw = tiles.shape[1], tiles.shape[2]
        return tiles.reshape(self.rows, self.cols, bh, bw).swapaxes(1, 2).reshape(
            self.rows * bh, self.cols * bw)

    def scramble(self, img: np.ndarray, frame_id: int) -> np.ndarray:
        return self._join(self._tiles(img)[self.permutation(frame_id)])

    def descramble(self, img: np.ndarray, frame_id: int) -> np.ndarray:
        return self._join(self._tiles(img)[invert_permutation(self.permutation(frame_id))])


# ------------------------------------------------------------- analog methods
class AnalogPictureMethod(Method):
    """B0a / B1 / B2: the picture itself travels through the analog path."""

    def __init__(self, name: str, transport: TransportConfig,
                 channel_cfg: RasterChannelConfig, frame_h: int, frame_w: int,
                 transform=None, inverse=None, authenticated: bool = False,
                 timer: Optional[StageTimer] = None, notes: str = "",
                 rasters_per_frame: int = 1) -> None:
        self.name = name
        self.rasters_per_frame = max(1, int(rasters_per_frame))
        self.authenticated = authenticated
        self.cfg = transport
        m = transport.modem
        self.carrier = AnalogCarrier(m.raster_height, m.raster_width, m.active_x0,
                                     m.active_x1, m.active_y0, m.active_y1,
                                     m.level_low, m.level_high)
        self.channel = RasterChannel(channel_cfg)
        self.frame_h, self.frame_w = frame_h, frame_w
        self._transform = transform
        self._inverse = inverse
        self.timer = timer or StageTimer()
        self.notes = notes

    def describe(self) -> Dict[str, Any]:
        return {
            "method": self.name, "family": "analog picture transport",
            "authenticated": self.authenticated,
            "confidentiality": "none" if self._transform is None else "permutation only",
            "notes": self.notes,
        }

    def reset(self) -> None:
        self.channel.reset_stream()

    def process(self, frame: np.ndarray, frame_id: int,
                trace: Any) -> MethodResult:
        t = self.timer
        tr = resolve_trace(trace, self.rasters_per_frame)
        with t(f"{self.name}.transform"):
            payload = self._transform(frame, frame_id) if self._transform else frame
        with t(f"{self.name}.modulation"):
            tx = self.carrier.transmit(payload)
        # The analog picture occupies the first raster slot of the shared budget,
        # so it meets the same damage as slot 0 of any digital method (F08).
        deliveries = self.channel.apply_stream(tx, tr.rng(frame_id, 0))
        rx, truth = None, None
        for img, tru in deliveries:
            truth = truth or tru
            if img is not None and rx is None:
                rx = img
        if rx is None:                       # raster dropped: nothing is displayed
            blank = np.full_like(frame, 128)
            avail = np.zeros_like(frame, dtype=bool)
            q = ev.quality_pair(frame, blank, avail)
            met = FrameMetrics(frame_id=frame_id, method=self.name, rasters=1,
                               sync_found=False, **q)
            met.extra.update({"authenticated": False, "channel_trace": tr.trace_id,
                              "coverage_meaning": "raster dropped, nothing displayed"})
            return MethodResult(self.name, frame_id, frame, tx, tx, blank, avail,
                                np.zeros_like(avail), met, truth, self.notes)
        with t(f"{self.name}.demodulation"):
            got, info = self.carrier.receive(rx, (self.frame_h, self.frame_w))
        with t(f"{self.name}.inverse"):
            recon = self._inverse(got, frame_id) if self._inverse else got
        recon = recon[: self.frame_h, : self.frame_w]
        if recon.shape != frame.shape:
            out = np.full(frame.shape, 128, dtype=np.uint8)
            out[: recon.shape[0], : recon.shape[1]] = recon
            recon = out
        avail = np.ones_like(frame, dtype=bool)     # a picture is always displayed
        stale = np.zeros_like(frame, dtype=bool)
        q = ev.quality_pair(frame, recon, avail)
        met = FrameMetrics(
            frame_id=frame_id, method=self.name, rasters=1,
            sync_found=info["sync_score"] > 0.3, **q,
        )
        met.extra.update({"authenticated": False, "sync_score": info["sync_score"],
                          "est_dx": info["dx"], "est_dy": info["dy"],
                          "coverage_meaning": "picture displayed, not verified"})
        return MethodResult(self.name, frame_id, frame, tx, rx, recon, avail, stale,
                            met, truth, self.notes)


def make_b0_analog(transport: TransportConfig, channel_cfg: RasterChannelConfig,
                   frame_h: int, frame_w: int, timer=None) -> AnalogPictureMethod:
    return AnalogPictureMethod(
        "B0a", transport, channel_cfg, frame_h, frame_w, None, None, False, timer,
        notes="unprotected analog picture; reference for quality loss and latency",
    )


def make_b1_lfsr(transport: TransportConfig, channel_cfg: RasterChannelConfig,
                 frame_h: int, frame_w: int, scr_cfg: ScramblerConfig,
                 timer=None) -> AnalogPictureMethod:
    sc = BlockScrambler(scr_cfg)
    return AnalogPictureMethod(
        "B1", transport, channel_cfg, frame_h, frame_w,
        transform=lambda img, fid: sc.scramble(img, fid),
        inverse=lambda img, fid: sc.descramble(img, fid, (frame_h, frame_w)),
        authenticated=False, timer=timer,
        notes=("reconstruction of the 2021 LFSR block permutation; "
               f"per_frame={scr_cfg.per_frame}, grid={scr_cfg.grid_rows}x{scr_cfg.grid_cols}"),
    )


def make_b2_cryptoperm(transport: TransportConfig, channel_cfg: RasterChannelConfig,
                       frame_h: int, frame_w: int, grid_rows: int, grid_cols: int,
                       master: MasterSecret, session_id: Optional[bytes] = None,
                       timer=None,
                       session_ids: Optional[SessionIdSource] = None
                       ) -> AnalogPictureMethod:
    sid = session_id or (session_ids or SecureSessionIds()).next("B2")
    keys = derive_session_keys(master, sid)
    sc = CryptoPermutationScrambler(grid_rows, grid_cols, keys.key, sid, per_frame=True)
    return AnalogPictureMethod(
        "B2", transport, channel_cfg, frame_h, frame_w,
        transform=lambda img, fid: sc.scramble(img, fid),
        inverse=lambda img, fid: sc.descramble(img, fid),
        authenticated=False, timer=timer,
        notes=("cryptographic permutation generator with a unique per-frame context; "
               "block contents are NOT encrypted - this is not full encryption"),
    )


class AnalogRepetitionMethod(AnalogPictureMethod):
    """B0a-R: the analog picture repeated over the slots the budget allows.

    Defect F15 asks for a *strong* analog control.  Sending one raster and
    leaving the other two slots idle wastes budget that every digital method
    spends, so this control repeats the same picture in all available slots and
    combines the received copies at the receiver.  The combining rule
    (``mean`` or ``median``) is a profile parameter chosen on the validation
    split, and the repetition schedule is public and agreed in advance.
    """

    def __init__(self, name: str, transport: TransportConfig,
                 channel_cfg: RasterChannelConfig, frame_h: int, frame_w: int,
                 repetitions: int = 3, combine: str = "median",
                 transform=None, inverse=None,
                 timer: Optional[StageTimer] = None, notes: str = "") -> None:
        super().__init__(name, transport, channel_cfg, frame_h, frame_w,
                         transform=transform, inverse=inverse, authenticated=False,
                         timer=timer, notes=notes, rasters_per_frame=repetitions)
        if combine not in ("mean", "median"):
            raise ValueError("combine must be 'mean' or 'median'")
        self.repetitions = max(1, int(repetitions))
        self.combine = combine

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update({"family": "analog picture transport with repetition",
                  "repetitions": self.repetitions, "combine": self.combine})
        return d

    def process(self, frame: np.ndarray, frame_id: int, trace: Any) -> MethodResult:
        t = self.timer
        tr = resolve_trace(trace, self.repetitions)
        with t(f"{self.name}.transform"):
            payload = self._transform(frame, frame_id) if self._transform else frame
        with t(f"{self.name}.modulation"):
            tx = self.carrier.transmit(payload)

        copies: List[np.ndarray] = []
        first_rx = None
        truth0 = None
        scores: List[float] = []
        for slot in range(self.repetitions):
            for rx_raster, tru in self.channel.apply_stream(tx, tr.rng(frame_id, slot)):
                if truth0 is None:
                    truth0 = tru
                if rx_raster is None:
                    continue
                if first_rx is None:
                    first_rx = rx_raster
                with t(f"{self.name}.demodulation"):
                    got, info = self.carrier.receive(rx_raster, (self.frame_h, self.frame_w))
                copies.append(got.astype(np.float64))
                scores.append(float(info["sync_score"]))

        if not copies:
            blank = np.full_like(frame, 128)
            avail = np.zeros_like(frame, dtype=bool)
            q = ev.quality_pair(frame, blank, avail)
            met = FrameMetrics(frame_id=frame_id, method=self.name,
                               rasters=self.repetitions, sync_found=False, **q)
            met.extra.update({"authenticated": False, "channel_trace": tr.trace_id,
                              "repetitions_received": 0})
            return MethodResult(self.name, frame_id, frame, tx, tx, blank, avail,
                                np.zeros_like(avail), met, truth0, self.notes)

        stack = np.stack(copies)
        combined = np.median(stack, axis=0) if self.combine == "median" else stack.mean(axis=0)
        with t(f"{self.name}.inverse"):
            got = np.clip(np.round(combined), 0, 255).astype(np.uint8)
            recon = self._inverse(got, frame_id) if self._inverse else got
        recon = recon[: self.frame_h, : self.frame_w]
        avail = np.ones_like(frame, dtype=bool)
        q = ev.quality_pair(frame, recon, avail)
        met = FrameMetrics(frame_id=frame_id, method=self.name,
                           rasters=self.repetitions,
                           sync_found=bool(scores and max(scores) > 0.3), **q)
        met.extra.update({
            "authenticated": False, "channel_trace": tr.trace_id,
            "repetitions_sent": self.repetitions,
            "repetitions_received": len(copies),
            "combine": self.combine,
            "coverage_meaning": "picture displayed, not verified",
        })
        return MethodResult(self.name, frame_id, frame, tx,
                            first_rx if first_rx is not None else tx,
                            recon, avail, np.zeros_like(avail), met, truth0, self.notes)


def make_b0a_repetition(transport: TransportConfig, channel_cfg: RasterChannelConfig,
                        frame_h: int, frame_w: int, repetitions: int = 3,
                        combine: str = "median", timer=None) -> AnalogRepetitionMethod:
    return AnalogRepetitionMethod(
        "B0a-R", transport, channel_cfg, frame_h, frame_w, repetitions, combine,
        notes=("unprotected analog picture repeated over the slots the shared "
               "budget allows, combined at the receiver; strong analog control"))


# ------------------------------------------------------------ digital methods
class DigitalMethod(Method):
    """B0d / B4 / P: independently protected units over the digital transport."""

    def __init__(self, name: str, source_cfg: SourceCodingConfig,
                 transport: TransportConfig, channel_cfg: RasterChannelConfig,
                 master: MasterSecret, frame_h: int, frame_w: int,
                 crypto_profile: Optional[CryptoProfile] = None, secure: bool = True,
                 fill: str = FILL_INTERPOLATE, max_frame_age: int = 2,
                 timer: Optional[StageTimer] = None, notes: str = "",
                 session_ids: Optional[SessionIdSource] = None) -> None:
        self.name = name
        self.authenticated = secure
        self.session_ids = session_ids or SecureSessionIds()
        self.source_cfg = source_cfg
        self.cfg = transport
        self.timer = timer or StageTimer()
        self.tx = Transmitter(source_cfg, transport, master, crypto_profile,
                              session_id=self.session_ids.next(name),
                              frame_width=frame_w, frame_height=frame_h,
                              timer=self.timer, secure=secure)
        self.rx = Receiver(source_cfg, transport, master, crypto_profile,
                           frame_width=frame_w, frame_height=frame_h,
                           max_frame_age=max_frame_age, timer=self.timer, secure=secure)
        self.channel = RasterChannel(channel_cfg)
        self.assembler = FrameAssembler(frame_h, frame_w, fill)
        self.frame_h, self.frame_w = frame_h, frame_w
        self.notes = notes

    def reset(self) -> None:
        """Begin an independent sequence: new session, clean sync and channel."""
        self.assembler.reset()
        self.tx.new_session(self.session_ids.next(self.name))
        self.rx.reset_state()
        self.channel.reset_stream()

    def describe(self) -> Dict[str, Any]:
        d = self.tx.describe()
        d.update({"method": self.name, "family": "digital transport",
                  "authenticated": self.authenticated, "notes": self.notes})
        return d

    def process(self, frame: np.ndarray, frame_id: int,
                trace: Any) -> MethodResult:
        tr = resolve_trace(trace, self.cfg.max_rasters_per_frame)
        txf = self.tx.encode_frame(frame, frame_id)
        units: List[Any] = []
        status_counts: Dict[str, int] = {}
        n_ok = n_rej = 0
        corrected = 0
        sym_err = 0
        sym_den = 0
        first_tx = txf.rasters[0]
        first_rx: Optional[np.ndarray] = None
        truth0: Optional[ChannelTruth] = None
        resync = False
        sync_ok = True
        n_delivered = 0

        for ri, raster in enumerate(txf.rasters):
            # One shared trace indexed by absolute raster time: every method that
            # occupies this slot meets the same damage (defect F08).
            rng = tr.rng(frame_id, ri)
            deliveries = self.channel.apply_stream(raster, rng)
            ref = txf.symbols[ri]
            used = np.concatenate(self.tx.placement[: txf.occupied_slots[ri]]) \
                if txf.occupied_slots[ri] else np.empty(0, dtype=np.int64)
            for rx_raster, truth in deliveries:
                if first_rx is None and rx_raster is not None:
                    first_rx, truth0 = rx_raster, truth
                if truth0 is None:
                    truth0 = truth
                if rx_raster is None:           # dropped on the way: nothing arrives
                    status_counts["raster_dropped"] = \
                        status_counts.get("raster_dropped", 0) + 1
                    sync_ok = False
                    continue
                n_delivered += 1
                out = self.rx.receive_raster(rx_raster)
                resync = resync or out.resynchronised
                sync_ok = sync_ok and out.demod.sync_found
                if used.size:
                    sym_err += int((out.demod.symbols[used] != ref[used]).sum())
                    sym_den += int(used.size)
                for o in out.outcomes:
                    status_counts[o.status.value] = status_counts.get(o.status.value, 0) + 1
                    corrected += o.payload_corrected + o.header_corrected
                    if o.ok and o.unit is not None:
                        units.append(o.unit)
                        n_ok += 1
                    elif o.status in (UnitStatus.FILLER, UnitStatus.IDLE):
                        pass
                    else:
                        n_rej += 1

        with self.timer(f"{self.name}.assembly"):
            # Assembly is keyed by the authenticated identity, so a late frame
            # cannot be counted as the current one (defect F04).
            current = [u for u in units if u.frame_id == frame_id]
            n_ok = len(current)
            asm = self.assembler.assemble_verified(current)
        q = ev.quality_pair(frame, asm.image, asm.available)
        met = FrameMetrics(
            frame_id=frame_id, method=self.name,
            stale_fraction=float(asm.from_previous.mean()),
            estimated_fraction=float(1.0 - asm.available.mean()),
            max_age_frames=float(asm.age_map.max()) if asm.age_map.size else 0.0,
            units_sent=txf.n_units, units_verified=n_ok, units_rejected=n_rej,
            status_counts=status_counts,
            symbol_errors_pre_fec=(sym_err / sym_den) if sym_den else float("nan"),
            symbol_error_denominator=sym_den, corrected_symbols=corrected,
            payload_bytes=txf.payload_bytes, wire_bytes=txf.wire_bytes,
            rasters=len(txf.rasters), sync_found=sync_ok, resynchronised=resync,
            **q,
        )
        met.extra.update({
            "authenticated": self.authenticated,
            "segments": txf.n_segments, "filler_units": txf.n_filler,
            "units_per_raster": self.tx.budget.units_per_raster,
            "rasters_delivered": n_delivered,
            "channel_trace": tr.trace_id,
            "units_from_other_frames": len(units) - n_ok,
        })
        return MethodResult(self.name, frame_id, frame, first_tx,
                            first_rx if first_rx is not None else first_tx,
                            asm.image, asm.available, asm.from_previous, met, truth0,
                            self.notes)


# ------------------------------------------------------- B3: whole-frame AEAD
@dataclass
class _B3Fragment:
    """One recovered fragment, described only by what the receiver could read."""

    header: UnitHeader
    payload: bytes


class B3Transmitter:
    """Codes a frame, seals it as ONE AEAD unit, fragments it over the transport."""

    def __init__(self, source_cfg: SourceCodingConfig, transport: TransportConfig,
                 master: MasterSecret, frame_h: int, frame_w: int,
                 crypto_profile: Optional[CryptoProfile] = None,
                 timer: Optional[StageTimer] = None,
                 session_ids: Optional[SessionIdSource] = None) -> None:
        from avsec import budget as budget_mod

        self.source_cfg = source_cfg
        self.cfg = transport
        self.crypto_profile = crypto_profile or CryptoProfile()
        self._master = master
        self.frame_h, self.frame_w = frame_h, frame_w
        self.timer = timer or StageTimer()
        self.session_ids = session_ids or SecureSessionIds()
        self.session_id = self.session_ids.next("B3")
        self.session_epoch = 0
        self._rebuild()

        self.modem = RasterModem(transport.modem)
        self.rs_payload = RSCodec(transport.fec_payload)
        self.rs_header = RSCodec(transport.fec_header)
        self.budget = budget_mod.compute_budget(
            transport.modem, transport.fec_payload, transport.fec_header,
            source_cfg.max_unit_payload, transport.raster_rate_hz)
        self.interleaver = Interleaver(transport.interleaver, transport.modem.n_data_rows,
                                       transport.modem.n_data_cols)
        from avsec.transmitter import _apply_placement_capacity

        _apply_placement_capacity(self.budget, self.interleaver, 1)
        u = self.budget.units_per_raster
        self.placement = self.interleaver.place([self.budget.unit_symbols] * u, [0] * u, 1)
        self.frame_coder = StripeCoder(SourceCodingConfig(
            stripe_height=frame_h, n_descriptions=1, codec=source_cfg.codec,
            quality=source_cfg.quality, max_unit_payload=1 << 24, max_segments=1))

    def _rebuild(self) -> None:
        self.keys = derive_session_keys(
            self._master, self.session_id, self.crypto_profile.direction,
            self.crypto_profile.stream_id, self.crypto_profile.algorithm,
            self.session_epoch)
        self.sealer = Sealer(self.keys)

    def new_session(self) -> None:
        self.session_id = self.session_ids.next("B3")
        self.session_epoch = 0
        self._rebuild()

    def frame_header(self, frame_id: int, seq: int, n_frags: int, frag: int,
                     coded_len: int) -> UnitHeader:
        """Header of one fragment.  ``seg_id = 0`` is the canonical frame AAD."""
        return UnitHeader(
            profile_id=self.cfg.profile_id, session_id=self.session_id,
            session_epoch=self.session_epoch,
            stream_id=self.crypto_profile.stream_id,
            codec_id=self.source_cfg.codec_id, frame_id=frame_id, stripe_id=0,
            desc_id=0, seg_id=min(frag, 255), n_segs=min(n_frags, 255), n_descs=1,
            unit_seq=seq,
            geometry=Geometry(0, 0, self.frame_w, self.frame_h, 1, 1, 0, 0),
            payload_len=min(coded_len, 0xFFFF), flags=0)

    def encode_frame(self, frame: np.ndarray, frame_id: int):
        t = self.timer
        with t("B3.source_coding"):
            segs = self.frame_coder.encode_frame(frame, frame_id)
            coded = b"".join(s.payload for s in segs)
        if len(coded) > 0xFFFF:
            raise CapacityExceeded(
                f"B3 coded frame is {len(coded)} bytes; the header carries a "
                "16-bit length, so the receiver could not recover it")

        frag_size = self.source_cfg.max_unit_payload
        n_frags = int(np.ceil((len(coded) + TAG_LEN) / frag_size))
        # The counter is allocated by the sealer and the AAD is built inside the
        # allocation, so no caller can pick or repeat a nonce (defect F03).
        built: Dict[str, int] = {}

        def _aad(counter: int) -> bytes:
            built["seq"] = counter
            return self.frame_header(frame_id, counter, n_frags, 0, len(coded)).core_bytes()

        with t("B3.crypto"):
            seq, ct = self.sealer.seal(coded, _aad)
        n_frags = int(np.ceil(len(ct) / frag_size))

        u = self.budget.units_per_raster
        n_rasters = int(np.ceil(n_frags / u))
        if n_rasters > self.cfg.max_rasters_per_frame:
            raise CapacityExceeded(
                f"B3 needs {n_rasters} rasters for frame {frame_id}; the profile "
                f"allows {self.cfg.max_rasters_per_frame}")

        rasters: List[np.ndarray] = []
        ref_syms: List[np.ndarray] = []
        occupied: List[int] = []
        fi = 0
        for _ in range(n_rasters):
            sym = np.zeros(self.cfg.modem.capacity_symbols, dtype=np.uint8)
            n_occ = 0
            for slot in range(u):
                if fi >= n_frags:
                    break
                chunk = ct[fi * frag_size : (fi + 1) * frag_size]
                chunk = chunk + bytes(frag_size - len(chunk))
                hdr = self.frame_header(frame_id, seq, n_frags, fi, len(coded))
                wire = self.rs_header.encode(hdr.to_bytes()) + self.rs_payload.encode(
                    chunk + bytes(TAG_LEN))
                s = bytes_to_symbols(wire, self.cfg.modem.bits_per_symbol)
                cells = self.placement[slot]
                sym[cells[: s.size]] = s[: cells.size]
                fi += 1
                n_occ += 1
            rasters.append(self.modem.modulate(sym))
            ref_syms.append(sym)
            occupied.append(n_occ)
        return rasters, ref_syms, occupied, len(coded), n_frags


class B3Receiver:
    """Recovers the AEAD unit from the signal alone.

    Defect F05: this side never sees the transmitter's counter, associated data,
    ciphertext length or slot occupancy.  Everything is read out of the received
    headers, bounded before use, and checked for mutual consistency; the AEAD tag
    over the canonical frame header is what finally decides.
    """

    def __init__(self, source_cfg: SourceCodingConfig, transport: TransportConfig,
                 master: MasterSecret, frame_h: int, frame_w: int,
                 crypto_profile: Optional[CryptoProfile] = None,
                 timer: Optional[StageTimer] = None, max_sessions: int = 4) -> None:
        from avsec import budget as budget_mod

        self.source_cfg = source_cfg
        self.cfg = transport
        self.crypto_profile = crypto_profile or CryptoProfile()
        self.master = master
        self.frame_h, self.frame_w = frame_h, frame_w
        self.timer = timer or StageTimer()
        self.max_sessions = max_sessions

        self.modem = RasterModem(transport.modem)
        self.rs_payload = RSCodec(transport.fec_payload)
        self.rs_header = RSCodec(transport.fec_header)
        self.budget = budget_mod.compute_budget(
            transport.modem, transport.fec_payload, transport.fec_header,
            source_cfg.max_unit_payload, transport.raster_rate_hz)
        self.interleaver = Interleaver(transport.interleaver, transport.modem.n_data_rows,
                                       transport.modem.n_data_cols)
        from avsec.transmitter import _apply_placement_capacity

        _apply_placement_capacity(self.budget, self.interleaver, 1)
        u = self.budget.units_per_raster
        self.placement = self.interleaver.place([self.budget.unit_symbols] * u, [0] * u, 1)
        self.frame_coder = StripeCoder(SourceCodingConfig(
            stripe_height=frame_h, n_descriptions=1, codec=source_cfg.codec,
            quality=source_cfg.quality, max_unit_payload=1 << 24, max_segments=1))
        self._openers: Dict[Tuple[bytes, int, int], Opener] = {}
        self._fragments: Dict[Tuple, Dict[int, bytes]] = {}
        self.status_counts: Dict[str, int] = {}

    def _bump(self, k: str, n: int = 1) -> None:
        self.status_counts[k] = self.status_counts.get(k, 0) + n

    def reset(self) -> None:
        self._fragments.clear()

    def _opener(self, ctx: Tuple[bytes, int, int]) -> Optional[Opener]:
        op = self._openers.get(ctx)
        if op is not None:
            return op
        try:
            keys = derive_session_keys(
                self.master, ctx[0], self.crypto_profile.direction, ctx[2],
                self.crypto_profile.algorithm, ctx[1])
        except Exception:
            return None
        return Opener(keys, self.crypto_profile.replay_window)   # not cached yet

    def _commit(self, ctx: Tuple[bytes, int, int], opener: Opener) -> None:
        if ctx not in self._openers and len(self._openers) >= self.max_sessions:
            self._openers.pop(next(iter(self._openers)))
        self._openers[ctx] = opener

    def receive_raster(self, raster: np.ndarray) -> List[_B3Fragment]:
        """Recover whatever fragments this raster carried.  No transmitter input."""
        t = self.timer
        with t("B3.demodulation"):
            demod = self.modem.demodulate(raster)
        if not demod.sync_found:
            self._bump("no_sync")
            return []

        frag_size = self.source_cfg.max_unit_payload
        hdr_enc = self.cfg.fec_header.encoded_len(HEADER_LEN)
        pay_len = frag_size + TAG_LEN
        pay_enc = self.cfg.fec_payload.encoded_len(pay_len)
        sym_per_byte = 8 // self.cfg.modem.bits_per_symbol

        out: List[_B3Fragment] = []
        # every slot of the public schedule is examined; the receiver does not
        # know which of them the transmitter actually filled
        for slot, cells in enumerate(self.placement):
            wire = symbols_to_bytes(demod.symbols[cells], self.cfg.modem.bits_per_symbol)
            er = demod.erasures[cells]
            byte_er = np.flatnonzero(
                er[: (er.size // sym_per_byte) * sym_per_byte]
                .reshape(-1, sym_per_byte).any(axis=1))
            hdr_bytes, _ = self.rs_header.try_decode(
                wire[:hdr_enc], HEADER_LEN, [int(b) for b in byte_er if b < hdr_enc])
            if hdr_bytes is None:
                self._bump("header_unrecoverable")
                continue
            try:
                hdr = parse_header(hdr_bytes, accepted_profiles=(self.cfg.profile_id,),
                                   max_payload_len=0xFFFF)
            except FramingError:
                self._bump("header_invalid")
                continue
            # bound every field that will drive an allocation, before using it
            if hdr.n_segs < 1 or hdr.seg_id >= hdr.n_segs:
                self._bump("header_invalid")
                continue
            if hdr.payload_len + TAG_LEN > hdr.n_segs * frag_size:
                self._bump("header_invalid")
                continue
            if hdr.geometry.width != self.frame_w or hdr.geometry.height != self.frame_h:
                self._bump("header_invalid")
                continue
            pay, _ = self.rs_payload.try_decode(
                wire[hdr_enc : hdr_enc + pay_enc], pay_len,
                [int(b) - hdr_enc for b in byte_er if hdr_enc <= b < hdr_enc + pay_enc])
            if pay is None:
                self._bump("payload_unrecoverable")
                continue
            self._bump("fragment_recovered")
            out.append(_B3Fragment(hdr, pay[:frag_size]))
        return out

    def offer(self, fragments: Sequence[_B3Fragment]
              ) -> List[Tuple[UnitHeader, np.ndarray]]:
        """Collect fragments and try to open any AEAD unit that is now complete."""
        frames: List[Tuple[UnitHeader, np.ndarray]] = []
        for f in fragments:
            h = f.header
            # the group key uses only authenticated-to-be fields; if any of them
            # was tampered with, the group simply will not verify
            key = (h.session_id, h.session_epoch, h.stream_id, h.frame_id,
                   h.n_segs, h.payload_len, h.unit_seq)
            self._fragments.setdefault(key, {})[h.seg_id] = f.payload
            if len(self._fragments) > 8:                      # bounded memory
                self._fragments.pop(next(iter(self._fragments)))
            group = self._fragments.get(key)
            if group is None or len(group) < h.n_segs:
                continue
            got = self._try_open(h, group)
            if got is not None:
                frames.append(got)
            self._fragments.pop(key, None)
        return frames

    def _try_open(self, h: UnitHeader, group: Dict[int, bytes]
                  ) -> Optional[Tuple[UnitHeader, np.ndarray]]:
        blob = b"".join(group[i] for i in range(h.n_segs))
        ct_len = h.payload_len + TAG_LEN            # derived from the header, not the TX
        if len(blob) < ct_len:
            self._bump("frame_incomplete")
            return None
        ct = blob[:ct_len]
        ctx = (h.session_id, h.session_epoch, h.stream_id)
        opener = self._opener(ctx)
        if opener is None:
            self._bump("session_limit")
            return None
        # canonical frame AAD: the same header with seg_id = 0
        aad = replace(h, seg_id=0).core_bytes()
        try:
            with self.timer("B3.aead"):
                plain = opener.open(h.unit_seq, ct, aad)
        except ReplayDetected:
            self._bump("replay")
            return None
        except AuthenticationFailed:
            self._bump("auth_failed")
            return None
        self._commit(ctx, opener)
        try:
            samples = self.frame_coder.decode_segment(plain, h.geometry)
        except Exception:
            self._bump("decode_failed")
            return None
        self._bump("frame_verified")
        return h, samples


class WholeFrameAEADMethod(Method):
    """B3: one AEAD unit per frame, fragmented over the same digital transport.

    Authentication can only succeed after every fragment of the AEAD unit has
    been recovered, so a single unrecoverable fragment costs the whole frame.
    Transmitter and receiver are separate objects and share nothing but the
    signal, the public profile and the legitimate key material (defect F05).
    """

    name = "B3"
    authenticated = True

    def __init__(self, source_cfg: SourceCodingConfig, transport: TransportConfig,
                 channel_cfg: RasterChannelConfig, master: MasterSecret,
                 frame_h: int, frame_w: int,
                 crypto_profile: Optional[CryptoProfile] = None,
                 fill: str = FILL_INTERPOLATE,
                 timer: Optional[StageTimer] = None,
                 session_ids: Optional[SessionIdSource] = None) -> None:
        self.source_cfg = source_cfg
        self.cfg = transport
        self.timer = timer or StageTimer()
        self.session_ids = session_ids or SecureSessionIds()
        self.tx = B3Transmitter(source_cfg, transport, master, frame_h, frame_w,
                                crypto_profile, self.timer, self.session_ids)
        self.rx = B3Receiver(source_cfg, transport, master, frame_h, frame_w,
                             crypto_profile, self.timer)
        self.channel = RasterChannel(channel_cfg)
        self.assembler = FrameAssembler(frame_h, frame_w, fill)
        self.frame_h, self.frame_w = frame_h, frame_w
        self.budget = self.tx.budget

    def reset(self) -> None:
        self.assembler.reset()
        self.tx.new_session()
        self.rx.reset()
        self.channel.reset_stream()

    def describe(self) -> Dict[str, Any]:
        return {
            "method": "B3", "family": "digital transport, whole-frame AEAD",
            "authenticated": True,
            "session_id": self.tx.session_id.hex(),
            "session_epoch": self.tx.session_epoch,
            "fragment_plain_bytes": self.source_cfg.max_unit_payload,
            "budget": self.tx.budget.to_dict(),
            "notes": "one AEAD unit per frame; all fragments are required; the "
                     "receiver recovers length, counter and AAD from the signal",
        }

    def process(self, frame: np.ndarray, frame_id: int, trace: Any) -> MethodResult:
        tr = resolve_trace(trace, self.cfg.max_rasters_per_frame)
        rasters, ref_syms, occupied, coded_len, n_frags = self.tx.encode_frame(
            frame, frame_id)

        before = dict(self.rx.status_counts)
        sym_err = sym_den = 0
        first_rx = None
        truth0 = None
        sync_ok = True
        verified: List[Tuple[UnitHeader, np.ndarray]] = []
        n_recovered = 0

        for ri, raster in enumerate(rasters):
            for rx_raster, truth in self.channel.apply_stream(raster, tr.rng(frame_id, ri)):
                if truth0 is None:
                    truth0 = truth
                if rx_raster is None:
                    sync_ok = False
                    continue
                if first_rx is None:
                    first_rx = rx_raster
                frags = self.rx.receive_raster(rx_raster)
                n_recovered += len(frags)
                verified.extend(self.rx.offer(frags))
                used = np.concatenate(self.tx.placement[: occupied[ri]]) \
                    if occupied[ri] else np.empty(0, dtype=np.int64)
                if used.size:
                    d = self.rx.modem.demodulate(rx_raster)
                    sync_ok = sync_ok and d.sync_found
                    sym_err += int((d.symbols[used] != ref_syms[ri][used]).sum())
                    sym_den += int(used.size)

        units: List[VerifiedUnit] = []
        note = ""
        for h, samples in verified:
            if h.frame_id != frame_id:
                continue
            units.append(VerifiedUnit(
                session_id=h.session_id, session_epoch=h.session_epoch,
                stream_id=h.stream_id, frame_id=h.frame_id, stripe_id=0,
                desc_id=0, seg_id=0, geometry=h.geometry, samples=samples,
                received_at=float(frame_id)))
        if not units:
            note = (f"the AEAD unit of frame {frame_id} did not verify: "
                    f"{n_recovered}/{n_frags} fragments recovered")

        with self.timer("B3.assembly"):
            asm = self.assembler.assemble_verified(units)
        q = ev.quality_pair(frame, asm.image, asm.available)
        status_counts = {k: self.rx.status_counts[k] - before.get(k, 0)
                         for k in self.rx.status_counts
                         if self.rx.status_counts[k] - before.get(k, 0) > 0}
        met = FrameMetrics(
            frame_id=frame_id, method="B3",
            stale_fraction=float(asm.from_previous.mean()),
            estimated_fraction=float(1.0 - asm.available.mean()),
            units_sent=n_frags, units_verified=n_recovered,
            units_rejected=max(0, n_frags - n_recovered), status_counts=status_counts,
            symbol_errors_pre_fec=(sym_err / sym_den) if sym_den else float("nan"),
            symbol_error_denominator=sym_den,
            payload_bytes=coded_len,
            wire_bytes=n_frags * self.tx.budget.unit_wire_bytes,
            rasters=len(rasters), sync_found=sync_ok, **q)
        met.extra.update({"authenticated": True, "fragments": n_frags,
                          "fragments_recovered": n_recovered,
                          "channel_trace": tr.trace_id,
                          "frame_verified": bool(units)})
        return MethodResult("B3", frame_id, frame, rasters[0],
                            first_rx if first_rx is not None else rasters[0],
                            asm.image, asm.available, asm.from_previous, met, truth0, note)


__all__ = [
    "MethodResult", "Method", "AnalogCarrier", "AnalogPictureMethod",
    "CryptoPermutationScrambler", "crypto_permutation",
    "make_b0_analog", "make_b1_lfsr", "make_b2_cryptoperm",
    "AnalogRepetitionMethod", "make_b0a_repetition",
    "DigitalMethod", "WholeFrameAEADMethod", "B3Transmitter", "B3Receiver",
]
