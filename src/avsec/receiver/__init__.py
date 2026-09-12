"""Receiver: raster -> symbols -> FEC -> strict parse -> AEAD -> verified samples.

Hard rules enforced here
------------------------
* The receiver's only inputs are the received raster, the public profile and
  the legitimate key material.  It never receives original pixels, plaintext,
  the true damage mask, true offsets or hidden frame indices.
* Nothing reaches the image decoder before its AEAD tag verifies.
* A correct CRC is never treated as authentication.
* The anti-replay window advances only after a successful authentication.
* Every rejection is reported with a structured status, never as a bare
  ``None`` or empty array.
* Queues, allocations and waiting times are bounded - but a bound is never met
  by *forgetting freshness*.  The key cache and the session admissibility state
  are two different objects (:mod:`avsec.receiver.sessions`): evicting a key
  costs one HKDF derivation, while displacing a session **closes** it, and a
  closed session can never be reopened from a recording.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from avsec import budget as budget_mod
from avsec.crypto import (
    TAG_LEN,
    SessionKeys,
    AuthenticationFailed,
    CryptoProfile,
    MasterSecret,
    NullOpener,
    Opener,
    ReplayDetected,
    derive_session_keys,
)
from avsec.fec import RSCodec
from avsec.framing import (
    HEADER_CORE_LEN,
    HEADER_LEN,
    VERSION as PROTOCOL_VERSION,
    FramingError,
    Geometry,
    UnitHeader,
    UnknownProfile,
    UnknownVersion,
    parse_header,
)
from avsec.interleaving import Interleaver
from avsec.modem import DemodResult, RasterModem
from avsec.receiver.sessions import (
    Admission,
    KeyCache,
    SessionContext,
    SessionLedger,
    SessionState,
)
from avsec.source_coding import CODEC_FILLER, SourceCodingConfig, StripeCoder
from avsec.transmitter import TransportConfig
from avsec.utils import StageTimer, public_whiten, symbols_to_bytes


class UnitStatus(str, Enum):
    VERIFIED = "verified"
    FILLER = "filler"
    IDLE = "idle"
    HEADER_UNRECOVERABLE = "header_unrecoverable"
    HEADER_INVALID = "header_invalid"
    UNKNOWN_VERSION = "unknown_version"
    UNKNOWN_PROFILE = "unknown_profile"
    PAYLOAD_UNRECOVERABLE = "payload_unrecoverable"
    AUTH_FAILED = "auth_failed"
    REPLAY = "replay"
    STALE = "stale"
    STALE_EPOCH = "stale_epoch"
    DECODE_FAILED = "decode_failed"
    NO_SYNC = "no_sync"
    SESSION_LIMIT = "session_limit"
    SESSION_CLOSED = "session_closed"


TERMINAL_REJECTIONS = {
    UnitStatus.HEADER_UNRECOVERABLE, UnitStatus.HEADER_INVALID, UnitStatus.UNKNOWN_VERSION,
    UnitStatus.UNKNOWN_PROFILE, UnitStatus.PAYLOAD_UNRECOVERABLE, UnitStatus.AUTH_FAILED,
    UnitStatus.REPLAY, UnitStatus.STALE, UnitStatus.STALE_EPOCH, UnitStatus.DECODE_FAILED,
    UnitStatus.SESSION_CLOSED,
}

#: How an admissibility verdict becomes a unit status.
_ADMISSION_STATUS = {
    Admission.CLOSED: UnitStatus.SESSION_CLOSED,
    Admission.STALE_EPOCH: UnitStatus.STALE_EPOCH,
    Admission.STALE_FRAME: UnitStatus.STALE,
}


@dataclass(frozen=True)
class VerifiedUnit:
    """Picture samples together with the identity that AEAD actually attested.

    Frame assembly consumes these, never bare ``(geometry, samples)`` pairs, so
    segments of two different frames, sessions or epochs cannot be merged into
    one picture and then reported as full coverage (defect F04).
    """

    session_id: bytes
    session_epoch: int
    stream_id: int
    frame_id: int
    stripe_id: int
    desc_id: int
    seg_id: int
    geometry: "Geometry"
    samples: np.ndarray
    received_at: float

    @property
    def identity(self) -> Tuple[bytes, int, int, int]:
        return (self.session_id, self.session_epoch, self.stream_id, self.frame_id)

    @property
    def unit_key(self) -> Tuple[int, int, int, int]:
        return (self.frame_id, self.stripe_id, self.desc_id, self.seg_id)


@dataclass
class UnitOutcome:
    slot: int
    status: UnitStatus
    detail: str = ""
    header: Optional[UnitHeader] = None
    samples: Optional[np.ndarray] = None
    unit: Optional[VerifiedUnit] = None
    header_corrected: int = 0
    payload_corrected: int = 0
    erasures_used: int = 0

    @property
    def ok(self) -> bool:
        return self.status is UnitStatus.VERIFIED

    def to_dict(self) -> Dict[str, object]:
        return {
            "slot": self.slot, "status": self.status.value, "detail": self.detail,
            "header": self.header.to_dict() if self.header else None,
            "header_corrected": self.header_corrected,
            "payload_corrected": self.payload_corrected,
            "erasures_used": self.erasures_used,
        }


@dataclass
class RasterOutcome:
    outcomes: List[UnitOutcome]
    demod: DemodResult
    resynchronised: bool = False

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for o in self.outcomes:
            out[o.status.value] = out.get(o.status.value, 0) + 1
        return out


class Receiver:
    """Stateless per raster, stateful per session (keys, replay, staleness)."""

    def __init__(
        self,
        source_cfg: SourceCodingConfig,
        transport_cfg: TransportConfig,
        master: MasterSecret,
        crypto_profile: Optional[CryptoProfile] = None,
        frame_width: int = 320,
        frame_height: int = 240,
        max_sessions: int = 4,
        max_frame_age: int = 2,
        sync_threshold: float = 0.35,
        search_x: int = 12,
        search_y: int = 8,
        timer: Optional[StageTimer] = None,
        secure: bool = True,
        retired_sessions: int = 256,
        key_cache_size: int = 32,
        cold_start: str = "tofu",
        session_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        source_cfg.validate()
        self.source_cfg = source_cfg
        self.cfg = transport_cfg
        self.master = master
        self.crypto_profile = crypto_profile or CryptoProfile()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_sessions = max_sessions
        self.max_frame_age = max_frame_age
        self.secure = bool(secure)
        self.sync_threshold = sync_threshold
        self.search_x = search_x
        self.search_y = search_y
        self.timer = timer or StageTimer()

        self.coder = StripeCoder(source_cfg)
        self.modem = RasterModem(transport_cfg.modem)
        self.rs_payload = RSCodec(transport_cfg.fec_payload)
        self.rs_header = RSCodec(transport_cfg.fec_header)
        self.budget = budget_mod.compute_budget(
            transport_cfg.modem, transport_cfg.fec_payload, transport_cfg.fec_header,
            source_cfg.max_unit_payload, transport_cfg.raster_rate_hz,
        )
        self.interleaver = Interleaver(
            transport_cfg.interleaver, transport_cfg.modem.n_data_rows,
            transport_cfg.modem.n_data_cols,
        )
        from avsec.transmitter import _apply_placement_capacity

        _apply_placement_capacity(self.budget, self.interleaver,
                                  source_cfg.n_descriptions)
        d = source_cfg.n_descriptions
        u = self.budget.units_per_raster
        self._placement = self.interleaver.place(
            [self.budget.unit_symbols] * u, [i % d for i in range(u)], d
        )
        # Two objects, two jobs (see avsec.receiver.sessions):
        #   * the key cache is a performance optimisation and may be evicted
        #     freely - a miss costs one HKDF derivation;
        #   * the ledger is the authoritative freshness state - displacing a
        #     session *closes* it, so an old recording stays rejected.
        self._keys = KeyCache(key_cache_size)
        self.sessions = SessionLedger(
            active_capacity=max_sessions, retired_capacity=retired_sessions,
            replay_window=self.crypto_profile.replay_window,
            max_frame_age=max_frame_age, cold_start=cold_start,
        )
        if session_state:
            self.sessions.import_state(session_state)
        self._had_sync = False
        self.stats: Dict[str, int] = {}
        self.clock: float = 0.0        # raster index; advanced by receive_raster

    # -- sessions ---------------------------------------------------------
    # Defect F01: nothing about session state may change before a unit has
    # authenticated.  Every method here is read-only except _commit_session,
    # which runs only after a successful AEAD open.
    #
    # Defect F19: the replay window used to live in the opener, which lived in
    # a bounded LRU cache, so displacing a session silently discarded its
    # freshness state and a recorded raster authenticated again.  The window now
    # belongs to the ledger; the cache holds keys only.
    def _session_keys(self, ctx: SessionContext) -> Optional[SessionKeys]:
        """Derived keys for a context, cached.  Purely a performance path."""
        keys = self._keys.get(ctx)
        if keys is not None:
            return keys
        session_id, epoch, stream_id = ctx
        try:
            keys = derive_session_keys(
                self.master, session_id, self.crypto_profile.direction,
                stream_id, self.crypto_profile.algorithm, epoch,
            )
        except Exception:
            return None
        self._keys.put(ctx, keys)
        return keys

    def _opener_for(self, ctx: SessionContext) -> Optional[Opener]:
        """An opener bound to the ledger's replay window for this context.

        For an ``ACTIVE`` context the window is the stored one, so a replayed
        sequence number is rejected.  For an ``UNKNOWN`` context the window is
        fresh and is installed only if :meth:`_commit_session` is reached.
        """
        keys = self._session_keys(ctx)
        if keys is None:
            return None
        window = self.sessions.window_for(ctx)
        cls = Opener if self.secure else NullOpener
        return cls(keys, self.crypto_profile.replay_window, window=window)

    def _commit_session(self, ctx: SessionContext, opener: Opener,
                        frame_id: int = 0) -> None:
        """Admit a context after a unit under it authenticated successfully."""
        self.sessions.commit(ctx, opener.replay, frame_id, self.clock)

    # Read-only helpers kept for callers and tests that inspect session state.
    def _lookup_opener(self, ctx: SessionContext) -> Optional[Opener]:
        if self.sessions.state_of(ctx) is not SessionState.ACTIVE:
            return None
        return self._opener_for(ctx)

    def _provisional_opener(self, ctx: SessionContext) -> Optional[Opener]:
        return self._opener_for(ctx)

    def export_session_state(self) -> Dict[str, Any]:
        """Freshness state a restarted receiver can resume from."""
        return self.sessions.export_state()

    # -- one raster -------------------------------------------------------
    def receive_raster(self, raster: np.ndarray) -> RasterOutcome:
        t = self.timer
        with t("rx.demodulation"):
            demod = self.modem.demodulate(raster, self.search_x, self.search_y,
                                          self.sync_threshold)
        if not demod.sync_found:
            self._bump(UnitStatus.NO_SYNC.value, self.budget.units_per_raster)
            had = self._had_sync
            self._had_sync = False
            return RasterOutcome(
                [UnitOutcome(i, UnitStatus.NO_SYNC, "sync pattern not found in this raster")
                 for i in range(self.budget.units_per_raster)],
                demod, resynchronised=False,
            )
        resync = not self._had_sync
        self._had_sync = True
        self.clock += 1.0

        bps = self.cfg.modem.bits_per_symbol
        sym_per_byte = 8 // bps
        hdr_enc = self.cfg.fec_header.encoded_len(HEADER_LEN)
        pay_len = self.source_cfg.max_unit_payload + TAG_LEN
        pay_enc = self.cfg.fec_payload.encoded_len(pay_len)

        outcomes: List[UnitOutcome] = []
        for slot, cells in enumerate(self._placement):
            outcomes.append(self._decode_slot(slot, cells, demod, sym_per_byte,
                                              hdr_enc, pay_len, pay_enc))
        for o in outcomes:
            self._bump(o.status.value)
        return RasterOutcome(outcomes, demod, resynchronised=resync)

    def _decode_slot(self, slot: int, cells: np.ndarray, demod: DemodResult,
                     sym_per_byte: int, hdr_enc: int, pay_len: int,
                     pay_enc: int) -> UnitOutcome:
        t = self.timer
        with t("rx.deinterleave"):
            sym = demod.symbols[cells]
            er = demod.erasures[cells]
            wire = symbols_to_bytes(sym, self.cfg.modem.bits_per_symbol)
            byte_er = np.flatnonzero(
                er[: (er.size // sym_per_byte) * sym_per_byte]
                .reshape(-1, sym_per_byte).any(axis=1)
            )

        with t("rx.fec_header"):
            hdr_block = wire[:hdr_enc]
            hdr_er = [int(b) for b in byte_er if b < hdr_enc]
            hdr_bytes, hstats = self.rs_header.try_decode(hdr_block, HEADER_LEN, hdr_er)
        if hdr_bytes is None:
            hdr_bytes, hstats = self.rs_header.try_decode(hdr_block, HEADER_LEN, None)
        if hdr_bytes is None:
            return UnitOutcome(slot, UnitStatus.HEADER_UNRECOVERABLE,
                               str(hstats.get("error", "")))

        try:
            header = parse_header(
                hdr_bytes, accepted_versions=(PROTOCOL_VERSION,),
                accepted_profiles=(self.cfg.profile_id,),
                max_payload_len=self.source_cfg.max_unit_payload,
            )
        except UnknownVersion as exc:
            return UnitOutcome(slot, UnitStatus.UNKNOWN_VERSION, str(exc))
        except UnknownProfile as exc:
            return UnitOutcome(slot, UnitStatus.UNKNOWN_PROFILE, str(exc))
        except FramingError as exc:
            return UnitOutcome(slot, UnitStatus.HEADER_INVALID, str(exc))

        # the header is parsed but still untrusted; bound-check it against the
        # public profile before it drives any allocation or indexing
        if header.codec_id != CODEC_FILLER:
            try:
                header.geometry.validate(self.frame_width, self.frame_height)
            except FramingError as exc:
                return UnitOutcome(slot, UnitStatus.HEADER_INVALID, str(exc), header)
            if header.n_descs != self.source_cfg.n_descriptions:
                return UnitOutcome(slot, UnitStatus.HEADER_INVALID,
                                   "description count differs from the agreed profile", header)

        # Freshness is tracked per authenticated context, not globally: a new
        # session or a new epoch legitimately restarts frame numbering at zero
        # (defect F02).  These pre-authentication checks may only *reject*; they
        # never advance any watermark, admit a context or close one (F01, F19).
        ctx: SessionContext = (header.session_id, header.session_epoch, header.stream_id)
        verdict, why = self.sessions.classify(ctx, header.frame_id)
        if verdict is not Admission.ADMIT:
            return UnitOutcome(slot, _ADMISSION_STATUS[verdict], why, header)

        opener = self._opener_for(ctx)
        if opener is None:
            return UnitOutcome(slot, UnitStatus.SESSION_LIMIT,
                               "no key material for this session context", header)

        with t("rx.fec_payload"):
            pay_block = wire[hdr_enc : hdr_enc + pay_enc]
            pay_er = [int(b) - hdr_enc for b in byte_er if hdr_enc <= b < hdr_enc + pay_enc]
            ct, pstats = self.rs_payload.try_decode(pay_block, pay_len, pay_er)
        if ct is None:
            ct, pstats = self.rs_payload.try_decode(pay_block, pay_len, None)
        if ct is None:
            return UnitOutcome(slot, UnitStatus.PAYLOAD_UNRECOVERABLE,
                               str(pstats.get("error", "")), header,
                               header_corrected=int(hstats.get("corrected_symbols", 0)))

        if self.cfg.public_whitening:
            ct = public_whiten(ct, header.unit_seq)
        with t("rx.aead"):
            try:
                plain = opener.open(header.unit_seq, ct, hdr_bytes[:HEADER_CORE_LEN])
            except ReplayDetected as exc:
                return UnitOutcome(slot, UnitStatus.REPLAY, str(exc), header)
            except AuthenticationFailed as exc:
                # Nothing was mutated: a forged unit cannot evict a live session,
                # advance a watermark or consume replay history (defect F01).
                return UnitOutcome(slot, UnitStatus.AUTH_FAILED, str(exc), header)

        # ---- authenticated from here on; only now may state advance ----
        # One call: admission, the epoch watermark, the frame watermark and the
        # replay window all live in the ledger and move together.
        self._commit_session(ctx, opener, header.frame_id)

        if header.codec_id == CODEC_FILLER or (header.flags & 0x04):
            return UnitOutcome(slot, UnitStatus.FILLER, "authenticated filler unit", header)

        payload = plain[: header.payload_len]
        with t("rx.source_decoding"):
            try:
                samples = self.coder.decode_segment(payload, header.geometry)
            except Exception as exc:
                return UnitOutcome(slot, UnitStatus.DECODE_FAILED, f"{type(exc).__name__}: {exc}",
                                   header)
        unit = VerifiedUnit(
            session_id=header.session_id, session_epoch=header.session_epoch,
            stream_id=header.stream_id, frame_id=header.frame_id,
            stripe_id=header.stripe_id, desc_id=header.desc_id, seg_id=header.seg_id,
            geometry=header.geometry, samples=samples, received_at=self.clock,
        )
        return UnitOutcome(
            slot, UnitStatus.VERIFIED, "", header, samples, unit,
            header_corrected=int(hstats.get("corrected_symbols", 0)),
            payload_corrected=int(pstats.get("corrected_symbols", 0)),
            erasures_used=int(pstats.get("erasures_used", 0)),
        )

    def _bump(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    def reset_sync(self) -> None:
        self._had_sync = False

    def reset_state(self) -> None:
        """Forget synchronisation only.

        Session, epoch and freshness state is keyed by the authenticated
        context, so a new session or epoch already restarts frame numbering
        safely (defect F02) and there is nothing to clear here.  Wiping the
        replay history between clips would be wrong: it would hide replays.
        """
        self._had_sync = False

    def session_report(self) -> Dict[str, Any]:
        """What the receiver's session state is, and on what assumptions."""
        return {"ledger": self.sessions.describe(), "key_cache": self._keys.describe()}


__all__ = ["UnitStatus", "UnitOutcome", "RasterOutcome", "Receiver",
           "TERMINAL_REJECTIONS", "VerifiedUnit", "SessionContext",
           "SessionLedger", "KeyCache", "Admission", "SessionState"]
