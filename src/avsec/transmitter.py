"""Transmitter: frame -> segments -> AEAD units -> FEC -> placement -> raster.

Fixed-size profile (v1)
-----------------------
Every transport unit carries exactly ``unit_plain_bytes`` of plaintext (the
real length is an authenticated header field, the rest is padding), so all
units have the same wire size and the raster schedule is fully deterministic
and public.  Slot ``i`` of every raster is publicly defined to carry
description ``i mod n_descriptions``; the receiver recomputes the same
placement table from the profile alone and is never handed packet boundaries,
sequence numbers or distortion parameters through a side channel.

If a frame's data does not fit into the declared capacity the transmitter
raises :class:`CapacityExceeded`.  It never truncates ciphertext and never
sends beyond the declared capacity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec import budget as budget_mod
from avsec.crypto import (
    TAG_LEN,
    CryptoProfile,
    MasterSecret,
    NullSealer,
    Sealer,
    SessionKeys,
    derive_session_keys,
    new_session_id,
)
from avsec.fec import FECConfig, RSCodec
from avsec.framing import (
    FLAG_LAST_SEGMENT,
    FLAG_STRIPE_COMPLETE,
    HEADER_LEN,
    Geometry,
    UnitHeader,
)
from avsec.interleaving import Interleaver, InterleaverConfig
from avsec.modem import ModemConfig, RasterModem
from avsec.source_coding import (
    CODEC_FILLER,
    Segment,
    SourceCodingConfig,
    StripeCoder,
)
from avsec.utils import StageTimer, bytes_to_symbols, public_whiten

FLAG_FILLER = 0x04


class CapacityExceeded(Exception):
    """The configured profile cannot carry this frame; nothing was truncated."""


def _apply_placement_capacity(budget, interleaver, n_descriptions: int) -> None:
    """Reduce the slot count to what the placement scheme can really carry.

    Called identically by the transmitter and the receiver, so the public
    schedule stays identical on both sides.
    """
    placeable = interleaver.max_units(budget.unit_symbols, n_descriptions,
                                      budget.units_per_raster)
    if placeable < budget.units_per_raster:
        budget.units_per_raster = placeable
        budget.unused_symbols = (budget.raster_capacity_symbols
                                 - placeable * budget.unit_symbols)


@dataclass
class TransportConfig:
    """Public transport profile shared by both endpoints."""

    modem: ModemConfig = field(default_factory=ModemConfig)
    fec_payload: FECConfig = field(default_factory=lambda: FECConfig(k=191, nsym=64))
    fec_header: FECConfig = field(default_factory=lambda: FECConfig(k=52, nsym=32))
    interleaver: InterleaverConfig = field(default_factory=InterleaverConfig)
    profile_id: int = 1
    raster_rate_hz: float = 25.0
    send_filler: bool = True
    max_rasters_per_frame: int = 8
    # Diagnostic control (F15): XOR the wire payload with a PUBLIC pseudo-random
    # sequence.  It equalises the symbol statistics of an unencrypted transport
    # with those of an encrypted one, so a difference cannot be attributed to
    # "the ciphertext looks random".  It is NOT protection: the sequence is
    # public and both endpoints derive it from the unit sequence number.
    public_whitening: bool = False

    def describe(self) -> Dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "raster_rate_hz": self.raster_rate_hz,
            "send_filler_units": self.send_filler,
            "public_whitening": self.public_whitening,
            "public_whitening_note": "diagnostic control, provides no confidentiality",
            "modem": self.modem.describe(),
            "fec_payload": self.fec_payload.describe(),
            "fec_header": self.fec_header.describe(),
            "interleaver": self.interleaver.describe(),
        }


@dataclass
class TxFrame:
    frame_id: int
    rasters: List[np.ndarray]
    n_units: int
    n_filler: int
    n_segments: int
    payload_bytes: int
    wire_bytes: int
    unit_headers: List[UnitHeader] = field(default_factory=list)
    symbols: List[np.ndarray] = field(default_factory=list)   # evaluator-only reference
    occupied_slots: List[int] = field(default_factory=list)
    #: ``(raster_index, slot_index)`` of each entry of ``unit_headers``, so the
    #: evaluator can say which received slot carried which description and
    #: segment without guessing the ordering (R04).
    unit_slots: List[Tuple[int, int]] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        return {
            "frame_id": self.frame_id, "rasters": len(self.rasters),
            "units": self.n_units, "filler_units": self.n_filler,
            "segments": self.n_segments, "payload_bytes": self.payload_bytes,
            "wire_bytes": self.wire_bytes,
        }


class Transmitter:
    """Turns frames into transmitted rasters under a fixed, public schedule."""

    def __init__(
        self,
        source_cfg: SourceCodingConfig,
        transport_cfg: TransportConfig,
        master: MasterSecret,
        crypto_profile: Optional[CryptoProfile] = None,
        session_id: Optional[bytes] = None,
        session_epoch: int = 0,
        frame_width: int = 320,
        frame_height: int = 240,
        timer: Optional[StageTimer] = None,
        secure: bool = True,
    ) -> None:
        source_cfg.validate()
        self.source_cfg = source_cfg
        self.cfg = transport_cfg
        self.crypto_profile = crypto_profile or CryptoProfile()
        self._master = master
        self.session_id = session_id or new_session_id()
        self.session_epoch = int(session_epoch)
        self.keys: SessionKeys = derive_session_keys(
            master, self.session_id, self.crypto_profile.direction,
            self.crypto_profile.stream_id, self.crypto_profile.algorithm,
            self.session_epoch,
        )
        self.secure = bool(secure)
        self.sealer = Sealer(self.keys) if secure else NullSealer(self.keys)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.timer = timer or StageTimer()

        self.coder = StripeCoder(source_cfg)
        self.modem = RasterModem(transport_cfg.modem)
        self.rs_payload = RSCodec(transport_cfg.fec_payload)
        self.rs_header = RSCodec(transport_cfg.fec_header)
        self.budget = budget_mod.compute_budget(
            transport_cfg.modem, transport_cfg.fec_payload, transport_cfg.fec_header,
            source_cfg.max_unit_payload, transport_cfg.raster_rate_hz,
        )
        if self.budget.units_per_raster < 1:
            raise CapacityExceeded(
                f"one transport unit needs {self.budget.unit_symbols} symbols but the "
                f"raster holds only {self.budget.raster_capacity_symbols}"
            )
        self.interleaver = Interleaver(
            transport_cfg.interleaver, transport_cfg.modem.n_data_rows,
            transport_cfg.modem.n_data_cols,
        )
        _apply_placement_capacity(self.budget, self.interleaver,
                                  source_cfg.n_descriptions)
        if self.budget.units_per_raster < 1:
            raise CapacityExceeded(
                "the configured placement scheme cannot fit a single transport unit "
                "into one raster")
        self._placement = self._build_placement()

    def _rebuild_sealer(self) -> None:
        self.keys = derive_session_keys(
            self._master, self.session_id, self.crypto_profile.direction,
            self.crypto_profile.stream_id, self.crypto_profile.algorithm,
            self.session_epoch,
        )
        self.sealer = Sealer(self.keys) if self.secure else NullSealer(self.keys)

    def new_session(self, session_id: Optional[bytes] = None) -> bytes:
        """Start a fresh session: new random id, epoch 0, counter 0."""
        self.session_id = session_id or new_session_id()
        self.session_epoch = 0
        self._rebuild_sealer()
        return self.session_id

    def new_epoch(self) -> int:
        """Advance the epoch of the current session id (autonomous restart).

        The epoch is authenticated in the header and mixed into the key
        derivation, so the counter may safely restart at zero: a different epoch
        is a different key, hence a different nonce space (defect F02/F03).
        """
        self.session_epoch += 1
        self._rebuild_sealer()
        return self.session_epoch

    # -- public schedule ---------------------------------------------------
    def _build_placement(self) -> List[np.ndarray]:
        """Placement table for the slots of one raster (public, receiver-computable)."""
        u = self.budget.units_per_raster
        d = self.source_cfg.n_descriptions
        return self.interleaver.place(
            [self.budget.unit_symbols] * u, [i % d for i in range(u)], d
        )

    @property
    def placement(self) -> List[np.ndarray]:
        return self._placement

    def slots_per_raster(self) -> int:
        return self.budget.units_per_raster

    def slot_description(self, slot: int) -> int:
        return slot % self.source_cfg.n_descriptions

    # -- unit construction -------------------------------------------------
    def _make_unit(self, header: UnitHeader, payload: bytes) -> Tuple[UnitHeader, bytes]:
        """Pad, seal and FEC-encode one unit; returns the final header and wire bytes."""
        fixed = self.source_cfg.max_unit_payload
        if len(payload) > fixed:
            raise CapacityExceeded(
                f"payload {len(payload)} exceeds the fixed unit size {fixed}"
            )
        plain = payload + bytes(fixed - len(payload))
        # The sealer allocates the counter and hands it to this builder; there is
        # no way to pick or repeat a counter from outside (defect F03).
        built: Dict[str, UnitHeader] = {}

        def _aad(counter: int) -> bytes:
            h = header.with_seq(counter).with_payload_len(len(payload))
            built["header"] = h
            return h.core_bytes()

        _, ct = self.sealer.seal(plain, _aad)
        hdr = built["header"]
        if self.cfg.public_whitening:
            ct = public_whiten(ct, hdr.unit_seq)
        wire = self.rs_header.encode(hdr.to_bytes()) + self.rs_payload.encode(ct)
        return hdr, wire

    def _filler_header(self, frame_id: int) -> UnitHeader:
        return UnitHeader(
            profile_id=self.cfg.profile_id, session_id=self.session_id,
            session_epoch=self.session_epoch,
            stream_id=self.crypto_profile.stream_id, codec_id=CODEC_FILLER,
            frame_id=frame_id, stripe_id=0, desc_id=0, seg_id=0, n_segs=1,
            n_descs=self.source_cfg.n_descriptions, unit_seq=0,
            geometry=Geometry(0, 0, 1, 1, 1, 1, 0, 0), payload_len=1, flags=FLAG_FILLER,
        )

    def _segment_header(self, seg: Segment) -> UnitHeader:
        flags = FLAG_LAST_SEGMENT if seg.seg_id == seg.n_segs - 1 else 0
        if seg.desc_id == seg.n_descs - 1 and (flags & FLAG_LAST_SEGMENT):
            flags |= FLAG_STRIPE_COMPLETE
        return UnitHeader(
            profile_id=self.cfg.profile_id, session_id=self.session_id,
            session_epoch=self.session_epoch,
            stream_id=self.crypto_profile.stream_id, codec_id=seg.codec_id,
            frame_id=seg.frame_id, stripe_id=seg.stripe_id, desc_id=seg.desc_id,
            seg_id=seg.seg_id, n_segs=seg.n_segs, n_descs=seg.n_descs, unit_seq=0,
            geometry=seg.geometry, payload_len=len(seg.payload), flags=flags,
        )

    # -- frame -------------------------------------------------------------
    def encode_frame(self, frame: np.ndarray, frame_id: int) -> TxFrame:
        t = self.timer
        with t("tx.source_coding"):
            segments = self.coder.encode_frame(frame, frame_id)

        d = self.source_cfg.n_descriptions
        queues: List[List[Segment]] = [[] for _ in range(d)]
        for s in segments:
            queues[s.desc_id % d].append(s)

        u = self.budget.units_per_raster
        slots_for_desc = [sum(1 for i in range(u) if i % d == dd) for dd in range(d)]
        n_rasters = 1
        for dd in range(d):
            if slots_for_desc[dd] == 0:
                if queues[dd]:
                    raise CapacityExceeded(
                        f"description {dd} has no slot in the public schedule "
                        f"({u} slots, {d} descriptions)"
                    )
                continue
            n_rasters = max(n_rasters, int(np.ceil(len(queues[dd]) / slots_for_desc[dd])))
        if n_rasters > self.cfg.max_rasters_per_frame:
            raise CapacityExceeded(
                f"frame {frame_id} needs {n_rasters} rasters, the profile allows "
                f"{self.cfg.max_rasters_per_frame}; lower the quality/resolution or "
                "raise the capacity"
            )

        cursors = [0] * d
        rasters: List[np.ndarray] = []
        headers: List[UnitHeader] = []
        n_units = n_filler = 0
        payload_bytes = wire_bytes = 0

        symbol_frames: List[np.ndarray] = []
        occupied: List[int] = []
        unit_slots: List[Tuple[int, int]] = []
        for ri in range(n_rasters):
            sym = np.zeros(self.cfg.modem.capacity_symbols, dtype=np.uint8)
            n_occ = 0
            for slot in range(u):
                dd = slot % d
                seg: Optional[Segment] = None
                if cursors[dd] < len(queues[dd]):
                    seg = queues[dd][cursors[dd]]
                    cursors[dd] += 1
                if seg is None and not self.cfg.send_filler:
                    continue
                with t("tx.crypto"):
                    if seg is None:
                        hdr, wire = self._make_unit(self._filler_header(frame_id), b"\x00")
                        n_filler += 1
                    else:
                        hdr, wire = self._make_unit(self._segment_header(seg), seg.payload)
                        n_units += 1
                        payload_bytes += len(seg.payload)
                wire_bytes += len(wire)
                headers.append(hdr)
                unit_slots.append((ri, slot))
                with t("tx.modulation"):
                    s = bytes_to_symbols(wire, self.cfg.modem.bits_per_symbol)
                    cells = self._placement[slot]
                    if s.size != cells.size:
                        raise CapacityExceeded(
                            f"unit produced {s.size} symbols, the slot holds {cells.size}"
                        )
                    sym[cells] = s
                n_occ += 1
            with t("tx.raster"):
                rasters.append(self.modem.modulate(sym))
            symbol_frames.append(sym)
            occupied.append(n_occ)

        return TxFrame(
            frame_id=frame_id, rasters=rasters, n_units=n_units, n_filler=n_filler,
            n_segments=len(segments), payload_bytes=payload_bytes, wire_bytes=wire_bytes,
            unit_headers=headers, symbols=symbol_frames, occupied_slots=occupied,
            unit_slots=unit_slots,
        )

    # -- reporting ---------------------------------------------------------
    def describe(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id.hex(),
            "session_epoch": self.session_epoch,
            "key_origin": self.keys.origin,
            "secure": self.secure,
            "source_coding": self.source_cfg.describe(),
            "transport": self.cfg.describe(),
            "crypto": self.crypto_profile.describe(),
            "budget": self.budget.to_dict(),
            "visible_metadata": [
                "raster sync pattern and pilot cells (fixed public pattern)",
                "unit boundaries and slot schedule (fixed size, public)",
                "header fields: version, profile, session id, stream, frame/stripe/"
                "description/segment indices, geometry, payload length, flags",
                "total traffic volume and timing (constant by construction when "
                "filler units are enabled)",
            ],
        }


__all__ = ["TransportConfig", "TxFrame", "Transmitter", "CapacityExceeded", "FLAG_FILLER"]
