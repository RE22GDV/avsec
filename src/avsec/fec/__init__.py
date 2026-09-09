"""Reed-Solomon forward error correction with explicit, auditable accounting.

The implementation of the code itself comes from ``reedsolo`` (GF(256), RS(255, k)).
This module only fixes the blocking, the shortening rule, the padding and the
erasure interface, and reports the *real* overhead so the channel budget cannot
be understated.

Blocking rule (profile v1)
--------------------------
A message of ``L`` bytes is split into ``ceil(L / k)`` blocks.  All blocks but
the last carry exactly ``k`` bytes; the last block is a **shortened** codeword
carrying ``L - (nblocks-1)*k`` bytes.  Every block is encoded to
``payload_bytes + nsym`` symbols, so the encoded length is ``L + nblocks*nsym``
and both sides can compute the block layout from ``L`` alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import reedsolo


class FECError(Exception):
    pass


class Uncorrectable(FECError):
    """The block could not be corrected; the caller must discard or mark erased."""


@dataclass(frozen=True)
class FECConfig:
    """RS(n, k) over GF(256); ``nsym = n - k`` parity symbols per block."""

    k: int = 191
    nsym: int = 64

    def validate(self) -> None:
        if self.nsym < 0 or self.nsym > 254:
            raise FECError("nsym must be in [0, 254]")
        if self.k < 1 or self.k + self.nsym > 255:
            raise FECError(f"k + nsym must be <= 255 (got {self.k} + {self.nsym})")

    @property
    def n(self) -> int:
        return self.k + self.nsym

    @property
    def max_errors(self) -> int:
        """Correctable symbol errors per block when positions are unknown."""
        return self.nsym // 2

    @property
    def max_erasures(self) -> int:
        """Correctable symbol erasures per block when positions are known."""
        return self.nsym

    @property
    def code_rate(self) -> float:
        return self.k / self.n if self.n else 1.0

    def n_blocks(self, message_len: int) -> int:
        return max(1, int(np.ceil(message_len / self.k))) if message_len else 0

    def encoded_len(self, message_len: int) -> int:
        return message_len + self.n_blocks(message_len) * self.nsym

    def block_layout(self, message_len: int) -> List[Tuple[int, int, int, int]]:
        """``[(msg_start, msg_len, enc_start, enc_len), ...]`` for a message length."""
        out: List[Tuple[int, int, int, int]] = []
        nb = self.n_blocks(message_len)
        ms = es = 0
        for i in range(nb):
            mlen = min(self.k, message_len - ms)
            elen = mlen + self.nsym
            out.append((ms, mlen, es, elen))
            ms += mlen
            es += elen
        return out

    def describe(self) -> Dict[str, object]:
        return {
            "code": f"RS({self.n}, {self.k}) over GF(256)",
            "k": self.k, "nsym": self.nsym, "n": self.n,
            "code_rate": round(self.code_rate, 4),
            "max_symbol_errors_per_block": self.max_errors,
            "max_symbol_erasures_per_block": self.max_erasures,
            "library": f"reedsolo {getattr(reedsolo, '__version__', 'unknown')}",
        }


class RSCodec:
    """Blocked RS encoder/decoder with erasure support and per-block status."""

    def __init__(self, cfg: FECConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self._rs = reedsolo.RSCodec(cfg.nsym) if cfg.nsym > 0 else None

    # -- encode -----------------------------------------------------------
    def encode(self, message: bytes) -> bytes:
        if self.cfg.nsym == 0:
            return bytes(message)
        out = bytearray()
        for ms, mlen, _, _ in self.cfg.block_layout(len(message)):
            block = message[ms : ms + mlen]
            out += bytes(self._rs.encode(bytearray(block)))
        return bytes(out)

    # -- decode -----------------------------------------------------------
    def decode(
        self,
        received: bytes,
        message_len: int,
        erasures: Optional[Sequence[int]] = None,
    ) -> Tuple[bytes, Dict[str, object]]:
        """Decode ``received`` back to ``message_len`` bytes.

        ``erasures`` are byte positions in ``received`` that the *receiver*
        itself flagged as unreliable (e.g. a symbol far from every level, or a
        raster line it never observed).  The true error mask is never supplied
        by the simulator.

        Raises :class:`Uncorrectable` if any block fails.  The returned stats
        record how many symbols each block actually corrected.
        """
        layout = self.cfg.block_layout(message_len)
        expected = self.cfg.encoded_len(message_len)
        if len(received) != expected:
            raise Uncorrectable(
                f"expected {expected} encoded bytes for a {message_len}-byte message, "
                f"got {len(received)}"
            )
        if self.cfg.nsym == 0:
            return bytes(received), {"blocks": len(layout), "corrected_symbols": 0,
                                     "erasures_used": 0}

        er = sorted(set(int(e) for e in (erasures or []) if 0 <= int(e) < len(received)))
        out = bytearray()
        corrected = 0
        used_er = 0
        for bi, (_, mlen, es, elen) in enumerate(layout):
            block = bytearray(received[es : es + elen])
            block_er = [e - es for e in er if es <= e < es + elen]
            if len(block_er) > self.cfg.max_erasures:
                raise Uncorrectable(
                    f"block {bi}: {len(block_er)} erasures exceed the code capability "
                    f"({self.cfg.max_erasures})"
                )
            try:
                dec, _, errata = self._rs.decode(block, erase_pos=block_er or None)
            except reedsolo.ReedSolomonError as exc:
                raise Uncorrectable(f"block {bi}: {exc}") from exc
            if len(dec) != mlen:
                raise Uncorrectable(f"block {bi}: decoded length {len(dec)} != {mlen}")
            corrected += len(errata)
            used_er += len(block_er)
            out += bytes(dec)
        return bytes(out), {"blocks": len(layout), "corrected_symbols": corrected,
                            "erasures_used": used_er}

    def try_decode(
        self, received: bytes, message_len: int, erasures: Optional[Sequence[int]] = None
    ) -> Tuple[Optional[bytes], Dict[str, object]]:
        """Non-raising variant used by the receiver's per-unit loop."""
        try:
            msg, stats = self.decode(received, message_len, erasures)
            stats["ok"] = True
            return msg, stats
        except Uncorrectable as exc:
            return None, {"ok": False, "error": str(exc)}


def overhead_report(cfg: FECConfig, message_len: int) -> Dict[str, object]:
    """Real accounting for a given message length (not the nominal code rate)."""
    nb = cfg.n_blocks(message_len)
    enc = cfg.encoded_len(message_len)
    return {
        "message_bytes": message_len,
        "blocks": nb,
        "parity_bytes": nb * cfg.nsym,
        "encoded_bytes": enc,
        "effective_rate": (message_len / enc) if enc else 1.0,
        "nominal_rate": cfg.code_rate,
        "last_block_shortened_to": message_len - (nb - 1) * cfg.k if nb else 0,
    }


__all__ = ["FECError", "Uncorrectable", "FECConfig", "RSCodec", "overhead_report"]
