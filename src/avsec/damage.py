"""From a burst on the raster to damaged Reed-Solomon symbols, in the right units.

What was wrong before
---------------------
The placement study counted how many **modem cells** of a unit fell inside a
burst and compared that count against ``nsym``, the number of parity *bytes*
per RS block.  Three separate mistakes were stacked on top of one another
(defect R03):

1. **Units.**  A modem cell carries ``bits_per_symbol`` bits, so with 2 bits per
   cell four damaged cells are at most one damaged byte.  Counting cells as
   bytes overstates the damage by ``8 / bits_per_symbol``.
2. **Double counting.**  Several damaged cells of the same byte are one damaged
   byte, not several.
3. **Blocking.**  A unit is not one codeword.  It is the FEC-encoded header
   followed by the FEC-encoded payload, and the payload is itself split into
   ``ceil(L / k)`` RS blocks, the last one shortened.  Reed-Solomon corrects
   ``nsym/2`` errors *per block*; comparing a whole unit's damage against one
   block's budget answers a question nobody asked.

This module does the conversion explicitly, in one place, so that "damaged
bytes" and "correctable" are measured in the same units for once.

The layout of a transport unit on the wire
------------------------------------------
::

    byte 0                      hdr_enc                     hdr_enc + pay_enc
    |  RS(header)  |            |  RS block 0 | RS block 1 | ... | last (short) |
    <-- one block, HEADER_LEN --><------------ payload, ceil(L/k) blocks ------->

Symbol ``i`` of a unit lives in byte ``i // (8 / bits_per_symbol)``; that byte
lives in exactly one of the blocks above.

Errors versus erasures
----------------------
They are different quantities and are never mixed here.  A block survives
``e`` **unknown** errors when ``2e <= nsym``, and ``e`` **erasures** - positions
the receiver itself flagged - when ``e <= nsym``.  Which of the two applies is a
property of the receiver, not of the channel, so both are reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec.fec import FECConfig
from avsec.framing import HEADER_LEN


@dataclass(frozen=True)
class UnitLayout:
    """Where the RS blocks of one transport unit sit inside its wire bytes."""

    bits_per_symbol: int
    fec_header: FECConfig
    fec_payload: FECConfig
    payload_len: int                      # ciphertext + tag, before FEC
    header_len: int = HEADER_LEN

    @property
    def symbols_per_byte(self) -> int:
        return 8 // int(self.bits_per_symbol)

    @property
    def header_encoded(self) -> int:
        return self.fec_header.encoded_len(self.header_len)

    @property
    def payload_encoded(self) -> int:
        return self.fec_payload.encoded_len(self.payload_len)

    @property
    def wire_bytes(self) -> int:
        return self.header_encoded + self.payload_encoded

    @property
    def wire_symbols(self) -> int:
        return self.wire_bytes * self.symbols_per_byte

    def blocks(self) -> List[Tuple[str, int, int, FECConfig]]:
        """``(kind, first_byte, length, code)`` for every RS block of the unit."""
        out: List[Tuple[str, int, int, FECConfig]] = []
        for _ms, mlen, es, elen in self.fec_header.block_layout(self.header_len):
            out.append(("header", es, elen, self.fec_header))
        base = self.header_encoded
        for _ms, mlen, es, elen in self.fec_payload.block_layout(self.payload_len):
            out.append(("payload", base + es, elen, self.fec_payload))
        return out

    def describe(self) -> Dict[str, Any]:
        blocks = self.blocks()
        return {
            "bits_per_symbol": self.bits_per_symbol,
            "symbols_per_byte": self.symbols_per_byte,
            "header_bytes": self.header_len,
            "header_encoded_bytes": self.header_encoded,
            "payload_bytes": self.payload_len,
            "payload_encoded_bytes": self.payload_encoded,
            "wire_bytes": self.wire_bytes,
            "wire_symbols": self.wire_symbols,
            "n_rs_blocks": len(blocks),
            "n_payload_blocks": sum(1 for b in blocks if b[0] == "payload"),
            "block_lengths": [b[2] for b in blocks],
            "header_nsym": self.fec_header.nsym,
            "payload_nsym": self.fec_payload.nsym,
            "correctable_errors_per_payload_block": self.fec_payload.max_errors,
            "correctable_erasures_per_payload_block": self.fec_payload.max_erasures,
        }


@dataclass
class UnitDamage:
    """The damage one burst did to one unit, in every unit of measurement."""

    damaged_symbols: int = 0
    damaged_bytes: int = 0
    header_damaged_bytes: int = 0
    payload_damaged_bytes: int = 0
    per_block: List[int] = field(default_factory=list)
    block_kinds: List[str] = field(default_factory=list)
    worst_block_bytes: int = 0
    survives_as_errors: bool = True
    survives_as_erasures: bool = True
    limiting_block: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "damaged_symbols": self.damaged_symbols,
            "damaged_bytes": self.damaged_bytes,
            "header_damaged_bytes": self.header_damaged_bytes,
            "payload_damaged_bytes": self.payload_damaged_bytes,
            "worst_block_damaged_bytes": self.worst_block_bytes,
            "survives_as_errors": self.survives_as_errors,
            "survives_as_erasures": self.survives_as_erasures,
            "limiting_block": self.limiting_block,
            "per_block_damaged_bytes": list(self.per_block),
        }


def damaged_bytes_of(symbol_ordinals: Sequence[int], symbols_per_byte: int
                     ) -> np.ndarray:
    """Byte indices touched by the given symbol ordinals, each counted once."""
    if len(symbol_ordinals) == 0:
        return np.empty(0, dtype=np.int64)
    ords = np.asarray(symbol_ordinals, dtype=np.int64)
    return np.unique(ords // int(symbols_per_byte))


def assess_unit(symbol_ordinals: Sequence[int], layout: UnitLayout) -> UnitDamage:
    """Damage of one unit, given which of *its own* symbols were hit.

    ``symbol_ordinals`` are positions **within the unit** (0 = its first
    symbol), not raster cell indices.
    """
    bytes_hit = damaged_bytes_of(symbol_ordinals, layout.symbols_per_byte)
    dmg = UnitDamage(damaged_symbols=int(len(symbol_ordinals)),
                     damaged_bytes=int(bytes_hit.size))
    dmg.header_damaged_bytes = int(np.count_nonzero(bytes_hit < layout.header_encoded))
    dmg.payload_damaged_bytes = dmg.damaged_bytes - dmg.header_damaged_bytes

    worst = 0
    limiting = -1
    for bi, (kind, first, length, code) in enumerate(layout.blocks()):
        n = int(np.count_nonzero((bytes_hit >= first) & (bytes_hit < first + length)))
        dmg.per_block.append(n)
        dmg.block_kinds.append(kind)
        if 2 * n > code.nsym:
            dmg.survives_as_errors = False
        if n > code.nsym:
            dmg.survives_as_erasures = False
        if n > worst:
            worst, limiting = n, bi
    dmg.worst_block_bytes = worst
    dmg.limiting_block = limiting
    return dmg


def burst_symbol_ordinals(cells: np.ndarray, n_cols: int, first_row: int,
                          n_rows: int) -> np.ndarray:
    """Which ordinals of a placed unit fall into rows ``[first_row, +n_rows)``."""
    rows = np.asarray(cells, dtype=np.int64) // int(n_cols)
    return np.flatnonzero((rows >= first_row) & (rows < first_row + n_rows))


def sweep_burst(cells_per_unit: Sequence[np.ndarray], n_cols: int, n_rows: int,
                burst_rows: int, layout: UnitLayout,
                descs: Optional[Sequence[int]] = None,
                n_descriptions: int = 1) -> Dict[str, Any]:
    """Every burst position against every placed unit, counted in RS bytes.

    Returns the worst and mean damage per unit, how many bursts leave at least
    one unit uncorrectable, and - separately - how many leave a whole image
    stripe without a usable description.  "Touched" and "lost" are different
    columns here for the same reason they are in :mod:`avsec.evaluation`: a
    description the FEC repaired was not lost (R04).
    """
    starts = max(1, int(n_rows) - int(burst_rows) + 1)
    n_units = len(cells_per_unit)
    descs = list(descs if descs is not None else [0] * n_units)
    n_desc = max(1, int(n_descriptions))
    stripe_of = [i // n_desc for i in range(n_units)]

    worst_bytes = np.zeros(n_units, dtype=np.int64)
    sum_bytes = np.zeros(n_units, dtype=np.float64)
    worst_symbols = np.zeros(n_units, dtype=np.int64)
    n_uncorrectable_positions = 0
    n_positions_with_stripe_lost = 0
    n_positions_with_desc_touched = 0
    touched_hist: Dict[int, int] = {}
    lost_hist: Dict[int, int] = {}

    rows_of = [np.asarray(c, dtype=np.int64) // int(n_cols) for c in cells_per_unit]
    for s0 in range(starts):
        s1 = s0 + burst_rows
        touched_per_stripe: Dict[int, set] = {}
        lost_per_stripe: Dict[int, set] = {}
        any_uncorrectable = False
        for u, rows in enumerate(rows_of):
            hit = np.flatnonzero((rows >= s0) & (rows < s1))
            if hit.size == 0:
                continue
            d = assess_unit(hit, layout)
            if d.damaged_bytes > worst_bytes[u]:
                worst_bytes[u] = d.damaged_bytes
            if hit.size > worst_symbols[u]:
                worst_symbols[u] = hit.size
            sum_bytes[u] += d.damaged_bytes
            touched_per_stripe.setdefault(stripe_of[u], set()).add(int(descs[u]))
            if not d.survives_as_errors:
                any_uncorrectable = True
                lost_per_stripe.setdefault(stripe_of[u], set()).add(int(descs[u]))
        n_uncorrectable_positions += int(any_uncorrectable)
        touched = max((len(v) for v in touched_per_stripe.values()), default=0)
        lost = max((len(v) for v in lost_per_stripe.values()), default=0)
        touched_hist[touched] = touched_hist.get(touched, 0) + 1
        lost_hist[lost] = lost_hist.get(lost, 0) + 1
        n_positions_with_desc_touched += int(touched > 0)
        if lost >= n_desc:
            n_positions_with_stripe_lost += 1

    return {
        "n_burst_positions": int(starts),
        "worst_damaged_bytes": int(worst_bytes.max()) if n_units else 0,
        "worst_damaged_symbols": int(worst_symbols.max()) if n_units else 0,
        "worst_unit": int(worst_bytes.argmax()) if n_units else -1,
        "mean_damaged_bytes": float(sum_bytes.sum() / max(1, starts * max(1, n_units))),
        "n_positions_uncorrectable": n_uncorrectable_positions,
        "frac_positions_uncorrectable": n_uncorrectable_positions / max(1, starts),
        "n_positions_stripe_lost": n_positions_with_stripe_lost,
        "frac_positions_stripe_lost": n_positions_with_stripe_lost / max(1, starts),
        "n_positions_any_desc_touched": n_positions_with_desc_touched,
        "touched_histogram": touched_hist,
        "lost_histogram": lost_hist,
    }


def decode_check(symbol_ordinals: Sequence[int], layout: UnitLayout,
                 seed: int = 0, as_erasures: bool = False) -> Dict[str, Any]:
    """Run a real RS decode over the same damage and compare the verdicts.

    The analytic count above is only useful if it agrees with the decoder.  This
    encodes a random message, corrupts exactly the bytes the model says a burst
    touched, and asks :class:`avsec.fec.RSCodec` whether it comes back.  The
    caller can then assert that prediction and reality match (R03).
    """
    from avsec.fec import RSCodec

    rng = np.random.default_rng(seed)
    model = assess_unit(symbol_ordinals, layout)
    bytes_hit = damaged_bytes_of(symbol_ordinals, layout.symbols_per_byte)

    header = bytes(rng.integers(0, 256, layout.header_len, dtype=np.uint8))
    payload = bytes(rng.integers(0, 256, layout.payload_len, dtype=np.uint8))
    rs_h, rs_p = RSCodec(layout.fec_header), RSCodec(layout.fec_payload)
    wire = bytearray(rs_h.encode(header) + rs_p.encode(payload))
    for b in bytes_hit.tolist():
        wire[b] ^= int(rng.integers(1, 256))

    base = layout.header_encoded
    hdr_er = [int(b) for b in bytes_hit if b < base] if as_erasures else None
    pay_er = ([int(b) - base for b in bytes_hit if b >= base]
              if as_erasures else None)
    got_h, _ = rs_h.try_decode(bytes(wire[:base]), layout.header_len, hdr_er)
    got_p, _ = rs_p.try_decode(bytes(wire[base:]), layout.payload_len, pay_er)

    decoded_ok = bool(got_h == header and got_p == payload)
    predicted = (model.survives_as_erasures if as_erasures
                 else model.survives_as_errors)
    return {
        "damaged_bytes": model.damaged_bytes,
        "per_block_damaged_bytes": model.per_block,
        "predicted_recoverable": predicted,
        "decoder_recovered": decoded_ok,
        "agree": predicted == decoded_ok,
        "mode": "erasures" if as_erasures else "unknown errors",
    }


def layout_for(transport_cfg: Any, max_unit_payload: int,
               tag_len: int = 16) -> UnitLayout:
    """Build the layout straight from a transport profile."""
    return UnitLayout(
        bits_per_symbol=int(transport_cfg.modem.bits_per_symbol),
        fec_header=transport_cfg.fec_header,
        fec_payload=transport_cfg.fec_payload,
        payload_len=int(max_unit_payload) + int(tag_len),
    )


__all__ = ["UnitLayout", "UnitDamage", "assess_unit", "damaged_bytes_of",
           "burst_symbol_ordinals", "sweep_burst", "decode_check", "layout_for"]
