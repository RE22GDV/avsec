"""Versioned binary wire format with a strict, bounded parser.

Canonical header (profile v1), big-endian, fixed 48 bytes plus a 4-byte CRC32::

    off size field            meaning
    ---------------------------------------------------------------------
      0    2  magic           0xA5 0x53
      2    1  version         protocol version (1)
      3    1  profile_id      transport/codec profile identifier
      4    8  session_id      random per-session identifier
     12    1  stream_id       logical stream inside the session
     13    1  codec_id        source coding profile of this payload
     14    4  frame_id        source frame number
     18    2  stripe_id       stripe index inside the frame
     20    1  desc_id         description index inside the stripe
     21    1  seg_id          segment index inside the description
     22    1  n_segs          number of segments of this description
     23    1  n_descs         number of descriptions of this stripe
     24    8  unit_seq        AEAD counter == anti-replay sequence number
     32    2  x0              geometry: left column in the source frame
     34    2  y0              geometry: top row in the source frame
     36    2  width           geometry: number of *sampled* columns
     38    2  height          geometry: number of *sampled* rows
     40    1  step_x          sub-lattice horizontal step
     41    1  step_y          sub-lattice vertical step
     42    1  phase_x         sub-lattice horizontal phase
     43    1  phase_y         sub-lattice vertical phase
     44    2  payload_len     ciphertext length in bytes (tag excluded)
     46    1  flags           bit0 last-segment, bit1 stripe-complete marker
     47    1  reserved        must be 0
     48    4  header_crc32    CRC-32 of bytes[0:48]

Bytes ``[0:48]`` are passed verbatim as AEAD associated data, so every field
above is cryptographically bound to the payload.  The CRC is a transport-level
aid for *finding and pre-parsing* a header before authentication; it is never
treated as a substitute for the tag.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

MAGIC = b"\xa5\x53"
VERSION = 1
HEADER_CORE_LEN = 48
HEADER_LEN = HEADER_CORE_LEN + 4
_STRUCT = struct.Struct(">2sBB8sBBIHBBBBQHHHHBBBBHBB")

MAX_PAYLOAD_LEN = 16384        # hard bound applied before any allocation
MAX_FRAME_DIM = 4096

FLAG_LAST_SEGMENT = 0x01
FLAG_STRIPE_COMPLETE = 0x02


class FramingError(Exception):
    """Malformed, unsupported or out-of-range header."""


class UnknownVersion(FramingError):
    pass


class UnknownProfile(FramingError):
    pass


@dataclass(frozen=True)
class Geometry:
    """Where a payload's samples live in the source frame.

    The unit covers source pixels
    ``(y0 + phase_y + j*step_y, x0 + phase_x + i*step_x)`` for
    ``0 <= j < height`` and ``0 <= i < width``.
    """

    x0: int
    y0: int
    width: int
    height: int
    step_x: int = 1
    step_y: int = 1
    phase_x: int = 0
    phase_y: int = 0

    def n_samples(self) -> int:
        return self.width * self.height

    def validate(self, frame_w: int, frame_h: int) -> None:
        if not (0 <= self.x0 < MAX_FRAME_DIM and 0 <= self.y0 < MAX_FRAME_DIM):
            raise FramingError("geometry origin out of range")
        if not (1 <= self.width <= MAX_FRAME_DIM and 1 <= self.height <= MAX_FRAME_DIM):
            raise FramingError("geometry size out of range")
        if not (1 <= self.step_x <= 16 and 1 <= self.step_y <= 16):
            raise FramingError("geometry step out of range")
        if not (0 <= self.phase_x < self.step_x and 0 <= self.phase_y < self.step_y):
            raise FramingError("geometry phase inconsistent with step")
        last_x = self.x0 + self.phase_x + (self.width - 1) * self.step_x
        last_y = self.y0 + self.phase_y + (self.height - 1) * self.step_y
        if last_x >= frame_w or last_y >= frame_h:
            raise FramingError("geometry exceeds the declared frame size")

    def rows(self):
        import numpy as np

        return self.y0 + self.phase_y + np.arange(self.height) * self.step_y

    def cols(self):
        import numpy as np

        return self.x0 + self.phase_x + np.arange(self.width) * self.step_x


@dataclass(frozen=True)
class UnitHeader:
    """One independently protected transport unit's public header."""

    profile_id: int
    session_id: bytes
    stream_id: int
    codec_id: int
    frame_id: int
    stripe_id: int
    desc_id: int
    seg_id: int
    n_segs: int
    n_descs: int
    unit_seq: int
    geometry: Geometry
    payload_len: int
    flags: int = 0
    version: int = VERSION

    # -- serialisation -------------------------------------------------
    def core_bytes(self) -> bytes:
        """Canonical 48-byte encoding used verbatim as AEAD associated data."""
        if len(self.session_id) != 8:
            raise FramingError("session id must be 8 bytes")
        g = self.geometry
        return _STRUCT.pack(
            MAGIC, self.version & 0xFF, self.profile_id & 0xFF, self.session_id,
            self.stream_id & 0xFF, self.codec_id & 0xFF,
            self.frame_id & 0xFFFFFFFF, self.stripe_id & 0xFFFF,
            self.desc_id & 0xFF, self.seg_id & 0xFF, self.n_segs & 0xFF, self.n_descs & 0xFF,
            self.unit_seq & 0xFFFFFFFFFFFFFFFF,
            g.x0 & 0xFFFF, g.y0 & 0xFFFF, g.width & 0xFFFF, g.height & 0xFFFF,
            g.step_x & 0xFF, g.step_y & 0xFF, g.phase_x & 0xFF, g.phase_y & 0xFF,
            self.payload_len & 0xFFFF, self.flags & 0xFF, 0,
        )

    def to_bytes(self) -> bytes:
        core = self.core_bytes()
        return core + struct.pack(">I", zlib.crc32(core) & 0xFFFFFFFF)

    def with_seq(self, seq: int) -> "UnitHeader":
        return replace(self, unit_seq=seq)

    def with_payload_len(self, n: int) -> "UnitHeader":
        return replace(self, payload_len=n)

    def key(self) -> Tuple[int, int, int, int]:
        return (self.frame_id, self.stripe_id, self.desc_id, self.seg_id)

    def to_dict(self) -> Dict[str, object]:
        g = self.geometry
        return {
            "version": self.version, "profile_id": self.profile_id,
            "session_id": self.session_id.hex(), "stream_id": self.stream_id,
            "codec_id": self.codec_id, "frame_id": self.frame_id,
            "stripe_id": self.stripe_id, "desc_id": self.desc_id, "seg_id": self.seg_id,
            "n_segs": self.n_segs, "n_descs": self.n_descs, "unit_seq": self.unit_seq,
            "geometry": {"x0": g.x0, "y0": g.y0, "width": g.width, "height": g.height,
                         "step_x": g.step_x, "step_y": g.step_y,
                         "phase_x": g.phase_x, "phase_y": g.phase_y},
            "payload_len": self.payload_len, "flags": self.flags,
        }


def parse_header(
    data: bytes,
    accepted_versions: Tuple[int, ...] = (VERSION,),
    accepted_profiles: Optional[Tuple[int, ...]] = None,
    max_payload_len: int = MAX_PAYLOAD_LEN,
    check_crc: bool = True,
) -> UnitHeader:
    """Strictly bounded header parser.

    Every field is range-checked *before* it is used for allocation or
    indexing.  A header that parses here is still **untrusted** until the AEAD
    tag over its canonical bytes verifies.
    """
    if len(data) < HEADER_LEN:
        raise FramingError(f"header needs {HEADER_LEN} bytes, got {len(data)}")
    core = data[:HEADER_CORE_LEN]
    (magic, version, profile_id, session_id, stream_id, codec_id, frame_id, stripe_id,
     desc_id, seg_id, n_segs, n_descs, unit_seq, x0, y0, width, height,
     step_x, step_y, phase_x, phase_y, payload_len, flags, reserved) = _STRUCT.unpack(core)

    if magic != MAGIC:
        raise FramingError("bad magic")
    if check_crc:
        (crc,) = struct.unpack(">I", data[HEADER_CORE_LEN:HEADER_LEN])
        if crc != (zlib.crc32(core) & 0xFFFFFFFF):
            raise FramingError("header CRC mismatch")
    if version not in accepted_versions:
        raise UnknownVersion(f"unsupported protocol version {version}")
    if accepted_profiles is not None and profile_id not in accepted_profiles:
        raise UnknownProfile(f"unsupported profile id {profile_id}")
    if reserved != 0:
        raise FramingError("reserved byte must be zero")
    if not 0 < payload_len <= max_payload_len:
        raise FramingError(f"payload_len {payload_len} outside (0, {max_payload_len}]")
    if not (1 <= step_x <= 16 and 1 <= step_y <= 16):
        raise FramingError("geometry step out of range")
    if not (phase_x < step_x and phase_y < step_y):
        raise FramingError("geometry phase inconsistent with step")
    if not (1 <= width <= MAX_FRAME_DIM and 1 <= height <= MAX_FRAME_DIM):
        raise FramingError("geometry size out of range")
    if n_segs == 0 or seg_id >= n_segs:
        raise FramingError("segment indices inconsistent")
    if n_descs == 0 or desc_id >= n_descs:
        raise FramingError("description indices inconsistent")

    return UnitHeader(
        profile_id=profile_id, session_id=session_id, stream_id=stream_id, codec_id=codec_id,
        frame_id=frame_id, stripe_id=stripe_id, desc_id=desc_id, seg_id=seg_id,
        n_segs=n_segs, n_descs=n_descs, unit_seq=unit_seq,
        geometry=Geometry(x0, y0, width, height, step_x, step_y, phase_x, phase_y),
        payload_len=payload_len, flags=flags, version=version,
    )


def header_overhead_bytes() -> int:
    return HEADER_LEN


__all__ = [
    "MAGIC", "VERSION", "HEADER_LEN", "HEADER_CORE_LEN", "MAX_PAYLOAD_LEN",
    "FLAG_LAST_SEGMENT", "FLAG_STRIPE_COMPLETE",
    "FramingError", "UnknownVersion", "UnknownProfile",
    "Geometry", "UnitHeader", "parse_header", "header_overhead_bytes",
]
