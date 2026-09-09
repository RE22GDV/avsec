"""Stripes, multiple descriptions, independent source coding and reassembly.

Design rules enforced here
--------------------------
* A **segment** is the atomic source-coding entity: it is coded on its own and
  can be decoded on its own, with no shared state and no dependency on any
  other segment or on any previous frame.
* A segment always fits into one independently protected transport unit, so
  "one unit lost" never means "a description is undecodable".
* Multiple descriptions of a stripe are regular polyphase sub-lattices; each
  description alone reconstructs a coarse version of the stripe, and more
  descriptions add known samples.
* Reconstruction of missing samples is deterministic (nearest available sample
  plus an optional smoothing pass) and is always reported through an
  availability map, so estimated pixels are never counted as received data.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec.framing import Geometry

CODEC_RAW = 0
CODEC_DCT = 1
CODEC_JPEG = 2
CODEC_FILLER = 0xFF   # authenticated padding unit; carries no picture data
CODEC_NAMES = {CODEC_RAW: "raw", CODEC_DCT: "dct", CODEC_JPEG: "jpeg", CODEC_FILLER: "filler"}
CODEC_IDS = {"raw": CODEC_RAW, "dct": CODEC_DCT, "jpeg": CODEC_JPEG}


class SourceCodingError(Exception):
    pass


class BudgetExceeded(SourceCodingError):
    """A segment cannot be made to fit into the configured unit payload."""


# ------------------------------------------------------------------- lattices
def lattice(n_descriptions: int, desc_id: int) -> Tuple[int, int, int, int]:
    """Return ``(step_x, step_y, phase_x, phase_y)`` for one description.

    * 1 description  - the full lattice.
    * 2 descriptions - column parity split (step_x = 2).
    * 4 descriptions - the four phases of a 2x2 grid.
    """
    if n_descriptions == 1:
        return 1, 1, 0, 0
    if n_descriptions == 2:
        if not 0 <= desc_id < 2:
            raise SourceCodingError("desc_id out of range")
        return 2, 1, desc_id, 0
    if n_descriptions == 4:
        if not 0 <= desc_id < 4:
            raise SourceCodingError("desc_id out of range")
        return 2, 2, desc_id % 2, desc_id // 2
    raise SourceCodingError("n_descriptions must be 1, 2 or 4")


# --------------------------------------------------------------------- codecs
_ZIGZAG8 = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
], dtype=np.int64)

# Standard JPEG luminance quantisation table (ITU-T T.81, Annex K, Table K.1).
_QTAB = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)


def _quality_scale(quality: int) -> float:
    q = int(np.clip(quality, 1, 100))
    return (5000.0 / q if q < 50 else 200.0 - 2.0 * q) / 100.0


def _qtable(quality: int) -> np.ndarray:
    return np.clip(np.round(_QTAB * _quality_scale(quality)), 1, 255)


def encode_raw(block: np.ndarray) -> bytes:
    return np.ascontiguousarray(block, dtype=np.uint8).tobytes()


def decode_raw(payload: bytes, h: int, w: int) -> np.ndarray:
    if len(payload) != h * w:
        raise SourceCodingError(f"raw payload {len(payload)} != {h*w}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(h, w).copy()


def _dct2(a: np.ndarray) -> np.ndarray:
    from scipy.fft import dctn

    return dctn(a, type=2, norm="ortho", axes=(-2, -1))


def _idct2(a: np.ndarray) -> np.ndarray:
    from scipy.fft import idctn

    return idctn(a, type=2, norm="ortho", axes=(-2, -1))


def _zigzag_varint(value: int, out: bytearray) -> None:
    """Zig-zag mapped LEB128, the standard compact signed-integer encoding."""
    v = (value << 1) ^ (value >> 63) if value < 0 else (value << 1)
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    shift = 0
    v = 0
    while True:
        if pos >= len(buf):
            raise SourceCodingError("truncated varint in dct payload")
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
        if shift > 63:
            raise SourceCodingError("varint too long in dct payload")
    return ((v >> 1) ^ -(v & 1)), pos


def encode_dct(block: np.ndarray, quality: int) -> bytes:
    """8x8 DCT, JPEG-style quantisation, zigzag, run-length tokens, DEFLATE.

    Token stream, per 8x8 block, in zigzag order::

        0x00                 end of block (every remaining coefficient is zero)
        r (1..255)           skip r-1 zeros, then a zig-zag varint coefficient

    Self-contained: the decoder derives every layout parameter from the
    authenticated geometry and the profile quality, so the payload carries no
    private header of its own and no state is shared between segments.
    """
    h, w = block.shape
    ph, pw = int(np.ceil(h / 8) * 8), int(np.ceil(w / 8) * 8)
    pad = np.zeros((ph, pw), dtype=np.float64)
    pad[:h, :w] = block
    if ph > h:
        pad[h:, :w] = pad[h - 1 : h, :w]
    if pw > w:
        pad[:, w:] = pad[:, w - 1 : w]
    tiles = pad.reshape(ph // 8, 8, pw // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8) - 128.0
    q = _qtable(quality)
    qc = np.round(_dct2(tiles) / q).astype(np.int32)
    zz = qc.reshape(-1, 64)[:, _ZIGZAG8]

    out = bytearray()
    for row in zz:
        nz = np.flatnonzero(row)
        prev = -1
        for idx in nz:
            gap = int(idx) - prev
            while gap > 255:
                out.append(255)
                _zigzag_varint(0, out)
                gap -= 255
            out.append(gap)
            _zigzag_varint(int(row[idx]), out)
            prev = int(idx)
        out.append(0)
    return zlib.compress(bytes(out), 6)


def decode_dct(payload: bytes, h: int, w: int, quality: int) -> np.ndarray:
    ph, pw = int(np.ceil(h / 8) * 8), int(np.ceil(w / 8) * 8)
    nblocks = (ph // 8) * (pw // 8)
    try:
        raw = zlib.decompress(payload)
    except zlib.error as exc:
        raise SourceCodingError(f"dct payload is not valid DEFLATE: {exc}") from exc

    zz = np.zeros((nblocks, 64), dtype=np.float64)
    pos = 0
    for b in range(nblocks):
        idx = -1
        while True:
            if pos >= len(raw):
                raise SourceCodingError("truncated dct token stream")
            r = raw[pos]
            pos += 1
            if r == 0:
                break
            val, pos = _read_varint(raw, pos)
            idx += r
            if idx >= 64:
                raise SourceCodingError("dct token stream overruns a block")
            zz[b, idx] = val
    qc = np.empty_like(zz)
    qc[:, _ZIGZAG8] = zz
    coef = qc.reshape(-1, 8, 8) * _qtable(quality)
    tiles = _idct2(coef) + 128.0
    img = tiles.reshape(ph // 8, pw // 8, 8, 8).swapaxes(1, 2).reshape(ph, pw)
    return np.clip(np.round(img[:h, :w]), 0, 255).astype(np.uint8)


def encode_jpeg(block: np.ndarray, quality: int) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", block, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise SourceCodingError("cv2 JPEG encoding failed")
    return buf.tobytes()


def decode_jpeg(payload: bytes, h: int, w: int) -> np.ndarray:
    import cv2

    arr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise SourceCodingError("cv2 JPEG decoding failed")
    if arr.shape != (h, w):
        raise SourceCodingError(f"jpeg geometry mismatch {arr.shape} != {(h, w)}")
    return arr


# --------------------------------------------------------------------- config
@dataclass
class SourceCodingConfig:
    stripe_height: int = 8
    n_descriptions: int = 1
    codec: str = "dct"
    quality: int = 55
    max_unit_payload: int = 512     # plaintext bytes per independently protected unit
    max_segments: int = 32

    def validate(self) -> None:
        if self.codec not in CODEC_IDS:
            raise SourceCodingError(f"codec must be one of {sorted(CODEC_IDS)}")
        if self.n_descriptions not in (1, 2, 4):
            raise SourceCodingError("n_descriptions must be 1, 2 or 4")
        if self.stripe_height < 1:
            raise SourceCodingError("stripe_height must be >= 1")
        if self.max_unit_payload < 32:
            raise SourceCodingError("max_unit_payload must be >= 32")

    @property
    def codec_id(self) -> int:
        return CODEC_IDS[self.codec]

    def describe(self) -> Dict[str, object]:
        return {
            "stripe_height": self.stripe_height,
            "n_descriptions": self.n_descriptions,
            "codec": self.codec,
            "quality": self.quality,
            "max_unit_payload": self.max_unit_payload,
            "lossless": self.codec == "raw",
        }


@dataclass
class Segment:
    """One independently coded, independently protectable piece of a frame."""

    frame_id: int
    stripe_id: int
    desc_id: int
    seg_id: int
    n_segs: int
    n_descs: int
    geometry: Geometry
    payload: bytes
    codec_id: int

    @property
    def size(self) -> int:
        return len(self.payload)


# ------------------------------------------------------------------- encoder
class StripeCoder:
    """Turn a frame into independently decodable segments and back."""

    def __init__(self, cfg: SourceCodingConfig) -> None:
        cfg.validate()
        self.cfg = cfg

    # -- geometry ---------------------------------------------------------
    def stripe_bounds(self, height: int) -> List[Tuple[int, int]]:
        sh = self.cfg.stripe_height
        return [(y, min(y + sh, height)) for y in range(0, height, sh)]

    def n_stripes(self, height: int) -> int:
        return len(self.stripe_bounds(height))

    # -- encode -----------------------------------------------------------
    def _encode_block(self, block: np.ndarray) -> bytes:
        c = self.cfg.codec
        if c == "raw":
            return encode_raw(block)
        if c == "dct":
            return encode_dct(block, self.cfg.quality)
        return encode_jpeg(block, self.cfg.quality)

    def _decode_block(self, payload: bytes, h: int, w: int) -> np.ndarray:
        c = self.cfg.codec
        if c == "raw":
            return decode_raw(payload, h, w)
        if c == "dct":
            return decode_dct(payload, h, w, self.cfg.quality)
        return decode_jpeg(payload, h, w)

    def encode_frame(self, frame: np.ndarray, frame_id: int = 0) -> List[Segment]:
        h, w = frame.shape
        out: List[Segment] = []
        nd = self.cfg.n_descriptions
        for sid, (y0, y1) in enumerate(self.stripe_bounds(h)):
            for d in range(nd):
                sx, sy, px, py = lattice(nd, d)
                rows = np.arange(y0 + py, y1, sy)
                cols = np.arange(px, w, sx)
                if rows.size == 0 or cols.size == 0:
                    continue
                sub = frame[np.ix_(rows, cols)]
                out.extend(self._segment(sub, frame_id, sid, d, nd, y0, sx, sy, px, py))
        return out

    def _segment(
        self, sub: np.ndarray, frame_id: int, stripe_id: int, desc_id: int, n_descs: int,
        y0: int, sx: int, sy: int, px: int, py: int,
    ) -> List[Segment]:
        """Split one description of one stripe into column bands that fit the budget."""
        sh, sw = sub.shape
        budget = self.cfg.max_unit_payload
        for n_segs in range(1, self.cfg.max_segments + 1):
            edges = np.linspace(0, sw, n_segs + 1).astype(int)
            if any(edges[i + 1] - edges[i] < 1 for i in range(n_segs)):
                break
            payloads = []
            ok = True
            for i in range(n_segs):
                band = sub[:, edges[i] : edges[i + 1]]
                p = self._encode_block(band)
                if len(p) > budget:
                    ok = False
                    break
                payloads.append(p)
            if ok:
                segs = []
                for i, p in enumerate(payloads):
                    g = Geometry(
                        x0=int(edges[i] * sx), y0=int(y0),
                        width=int(edges[i + 1] - edges[i]), height=int(sh),
                        step_x=sx, step_y=sy, phase_x=px, phase_y=py,
                    )
                    segs.append(Segment(frame_id, stripe_id, desc_id, i, n_segs, n_descs,
                                        g, p, self.cfg.codec_id))
                return segs
        raise BudgetExceeded(
            f"cannot fit a {sh}x{sw} description into {budget}-byte units with at most "
            f"{self.cfg.max_segments} segments (codec={self.cfg.codec}, "
            f"quality={self.cfg.quality}); lower the quality, the stripe height or "
            f"raise max_unit_payload"
        )

    # -- decode -----------------------------------------------------------
    def decode_segment(self, payload: bytes, geometry: Geometry) -> np.ndarray:
        return self._decode_block(payload, geometry.height, geometry.width)


# ------------------------------------------------------------------ assembly
FILL_NEUTRAL = "neutral"
FILL_INTERPOLATE = "interpolate"
FILL_PREVIOUS = "previous"


@dataclass
class AssembledFrame:
    image: np.ndarray                 # rendered picture (received + estimated)
    available: np.ndarray             # bool: pixel carries verified received data
    from_previous: np.ndarray         # bool: pixel copied from an older frame
    age_map: np.ndarray               # int: frames since this pixel was received
    fill_mode: str
    n_units_used: int = 0

    @property
    def coverage(self) -> float:
        return float(self.available.mean())

    def summary(self) -> Dict[str, float]:
        return {
            "coverage": self.coverage,
            "stale_fraction": float(self.from_previous.mean()),
            "estimated_fraction": float(1.0 - self.available.mean()),
            "max_age_frames": float(self.age_map.max()) if self.age_map.size else 0.0,
        }


class FrameAssembler:
    """Place verified samples on a canvas and render with an explicit fill policy."""

    def __init__(self, height: int, width: int, fill: str = FILL_INTERPOLATE,
                 max_age: int = 3, smoothing_passes: int = 1) -> None:
        self.h, self.w = height, width
        self.fill = fill
        self.max_age = max_age
        self.smoothing_passes = smoothing_passes
        self._prev: Optional[np.ndarray] = None
        self._prev_age: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None
        self._prev_age = None

    def assemble(self, placements: Sequence[Tuple[Geometry, np.ndarray]]) -> AssembledFrame:
        canvas = np.zeros((self.h, self.w), dtype=np.float64)
        avail = np.zeros((self.h, self.w), dtype=bool)
        for g, samples in placements:
            rows = g.rows()
            cols = g.cols()
            rows = rows[rows < self.h]
            cols = cols[cols < self.w]
            if rows.size == 0 or cols.size == 0:
                continue
            sub = samples[: rows.size, : cols.size]
            canvas[np.ix_(rows, cols)] = sub
            avail[np.ix_(rows, cols)] = True

        from_prev = np.zeros((self.h, self.w), dtype=bool)
        age = np.zeros((self.h, self.w), dtype=np.int32)
        img = canvas.copy()
        missing = ~avail

        if missing.any():
            if self.fill == FILL_NEUTRAL:
                img[missing] = 128.0
            elif self.fill == FILL_PREVIOUS and self._prev is not None:
                prev_age = (self._prev_age if self._prev_age is not None
                            else np.zeros((self.h, self.w), dtype=np.int32))
                usable = missing & (prev_age + 1 <= self.max_age)
                img[usable] = self._prev[usable]
                from_prev = usable
                age[usable] = prev_age[usable] + 1
                rest = missing & ~usable
                if rest.any():
                    img = self._interpolate(img, avail | usable)
            else:
                img = self._interpolate(img, avail)

        out = np.clip(np.round(img), 0, 255).astype(np.uint8)
        self._prev = out.astype(np.float64)
        self._prev_age = age
        return AssembledFrame(out, avail, from_prev, age, self.fill)

    def _interpolate(self, img: np.ndarray, known: np.ndarray) -> np.ndarray:
        """Deterministic nearest-known-sample fill plus optional smoothing.

        Uses an exact Euclidean distance transform, so the result depends only
        on the availability mask - never on any information the receiver does
        not have.
        """
        from scipy import ndimage

        if not known.any():
            return np.full_like(img, 128.0)
        idx = ndimage.distance_transform_edt(~known, return_distances=False,
                                             return_indices=True)
        filled = img[tuple(idx)]
        out = np.where(known, img, filled)
        for _ in range(max(0, self.smoothing_passes)):
            blurred = ndimage.uniform_filter(out, size=3, mode="nearest")
            out = np.where(known, out, blurred)
        return out


__all__ = [
    "CODEC_RAW", "CODEC_DCT", "CODEC_JPEG", "CODEC_FILLER", "CODEC_NAMES", "CODEC_IDS",
    "SourceCodingError", "BudgetExceeded", "lattice",
    "encode_raw", "decode_raw", "encode_dct", "decode_dct", "encode_jpeg", "decode_jpeg",
    "SourceCodingConfig", "Segment", "StripeCoder",
    "FILL_NEUTRAL", "FILL_INTERPOLATE", "FILL_PREVIOUS",
    "AssembledFrame", "FrameAssembler",
]
