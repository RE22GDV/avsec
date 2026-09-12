"""Session admissibility: the receiver's authoritative freshness state.

Why this is a separate module
-----------------------------
Before this, one bounded ``OrderedDict`` of :class:`~avsec.crypto.Opener`
objects did two unrelated jobs at once:

* it **cached derived keys**, purely to avoid running HKDF again, and
* it **held the anti-replay window and the frame watermark**, which are the
  only reasons an old recording is rejected.

Mixing them makes the cache's eviction policy a security decision.  With
``max_sessions = 4``, anything that makes the receiver see four other session
contexts displaces the genuine one; its replay window disappears with it, and a
raster recorded earlier authenticates again, because the receiver no longer
remembers that those sequence numbers were ever used.

The two are separated here:

``KeyCache``
    pure performance.  Holds :class:`~avsec.crypto.SessionKeys`.  Evicting an
    entry costs one HKDF derivation and changes no decision.

``SessionLedger``
    the authoritative record of which session contexts are *admissible*.  A
    context is ``ACTIVE`` (with a live replay window and frame watermark) or
    ``CLOSED`` (permanently rejected).  Nothing here is ever silently
    forgotten: when a context is displaced from the active table it is
    **closed**, not dropped, so a replayed raster is rejected rather than
    re-accepted.

Life cycle of a context ``(session_id, epoch, stream_id)``
----------------------------------------------------------
``UNKNOWN`` -> ``ACTIVE``
    only when a unit carrying that context **authenticates**.  Admission is
    refused up front for a context that is already ``CLOSED`` or whose epoch
    was retired.  :meth:`SessionLedger.classify` is read-only, so a forged or
    corrupted header can never move a context along this path.

``ACTIVE`` -> ``CLOSED`` (final)
    on any of three events, all of which require an authenticated unit:

    1. a strictly higher epoch of the same ``(session_id, stream_id)`` becomes
       active - the old epoch is retired by the protocol;
    2. the active table is full and this is the least recently authenticated
       context - it is displaced by a newer session;
    3. :meth:`close` is called explicitly (end of session).

    ``CLOSED`` is final.  A closed context is rejected with
    ``Admission.CLOSED`` for as long as the ledger remembers it.

Behaviour across a receiver restart
-----------------------------------
Freshness is state, and a process that starts with no state cannot distinguish
a live session from a recording of one.  The policy is therefore explicit
rather than implied:

``cold_start="tofu"`` (default)
    trust-on-first-use *per session context*.  An unknown context is admitted
    and its first authenticated unit establishes the watermark.  This is sound
    only if the transmitter starts a fresh ``session_id`` or advances
    ``session_epoch`` after every receiver restart - which
    :meth:`avsec.transmitter.Transmitter.new_session` and
    :meth:`~avsec.transmitter.Transmitter.new_epoch` do.  The mode is recorded
    in :meth:`SessionLedger.describe` so no report can quote a replay result
    without it.

``cold_start="persisted"``
    the ledger is restored with :meth:`import_state` before the first raster,
    so contexts closed before the restart stay closed and watermarks survive.
    This is the mode a deployment would use; it needs somewhere to persist
    state, which the lab harness does not have by default.

Bounded memory, stated honestly
-------------------------------
``retired_capacity`` closed contexts are remembered exactly.  Session ids are
8 random bytes, so an unbounded stream of genuine sessions cannot be remembered
exactly in bounded memory by anybody; when a tombstone is dropped the ledger
increments ``forgotten_closures`` and :meth:`describe` reports it.  A number of
sessions below the capacity - which is what every experiment in this repository
runs - is remembered exactly.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from avsec.crypto import ReplayWindow, SessionKeys

#: ``(session_id, session_epoch, stream_id)`` - the context AEAD attests to.
SessionContext = Tuple[bytes, int, int]
#: ``(session_id, stream_id)`` - the scope over which the epoch is monotone.
EpochKey = Tuple[bytes, int]


class SessionState(str, Enum):
    UNKNOWN = "unknown"
    ACTIVE = "active"
    CLOSED = "closed"


class Admission(str, Enum):
    """Verdict of the pre-authentication admissibility check."""

    ADMIT = "admit"
    CLOSED = "closed"
    STALE_EPOCH = "stale_epoch"
    STALE_FRAME = "stale"


@dataclass
class SessionRecord:
    """Everything the receiver knows about one admitted context."""

    ctx: SessionContext
    replay: ReplayWindow
    newest_frame: int = -1
    first_seen: float = 0.0
    last_seen: float = 0.0
    n_units: int = 0

    def to_dict(self) -> Dict[str, Any]:
        sid, epoch, stream = self.ctx
        return {"session_id": sid.hex(), "epoch": int(epoch), "stream_id": int(stream),
                "newest_frame": int(self.newest_frame),
                "first_seen": float(self.first_seen), "last_seen": float(self.last_seen),
                "n_units": int(self.n_units), "replay": self.replay.to_dict()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SessionRecord":
        ctx = (bytes.fromhex(str(d["session_id"])), int(d["epoch"]), int(d["stream_id"]))
        return SessionRecord(
            ctx=ctx, replay=ReplayWindow.from_dict(d.get("replay") or {}),
            newest_frame=int(d.get("newest_frame", -1)),
            first_seen=float(d.get("first_seen", 0.0)),
            last_seen=float(d.get("last_seen", 0.0)),
            n_units=int(d.get("n_units", 0)))


class KeyCache:
    """Bounded LRU of derived session keys.  Holds no security state.

    Evicting from here only costs an HKDF derivation; it can never change
    whether a unit is accepted.  That is the entire point of it being separate
    from :class:`SessionLedger`.
    """

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = max(1, int(capacity))
        self._keys: "OrderedDict[SessionContext, SessionKeys]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, ctx: SessionContext) -> Optional[SessionKeys]:
        keys = self._keys.get(ctx)
        if keys is None:
            self.misses += 1
            return None
        self.hits += 1
        self._keys.move_to_end(ctx)
        return keys

    def put(self, ctx: SessionContext, keys: SessionKeys) -> None:
        if ctx not in self._keys:
            while len(self._keys) >= self.capacity:
                self._keys.popitem(last=False)
                self.evictions += 1
        self._keys[ctx] = keys
        self._keys.move_to_end(ctx)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, ctx: object) -> bool:
        return ctx in self._keys

    def describe(self) -> Dict[str, Any]:
        return {"capacity": self.capacity, "held": len(self._keys),
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions,
                "note": ("кеш ключів: витіснення коштує одного HKDF і не "
                         "впливає на жодне рішення про допустимість")}


class SessionLedger:
    """Authoritative admissibility state: active contexts and closed ones."""

    COLD_START_MODES = ("tofu", "persisted")

    def __init__(self, active_capacity: int = 4, retired_capacity: int = 256,
                 replay_window: int = 4096, max_frame_age: int = 2,
                 cold_start: str = "tofu") -> None:
        if cold_start not in self.COLD_START_MODES:
            raise ValueError(f"cold_start must be one of {self.COLD_START_MODES}")
        self.active_capacity = max(1, int(active_capacity))
        self.retired_capacity = max(self.active_capacity, int(retired_capacity))
        self.replay_window = int(replay_window)
        self.max_frame_age = int(max_frame_age)
        self.cold_start = cold_start
        self._active: "OrderedDict[SessionContext, SessionRecord]" = OrderedDict()
        self._closed: "OrderedDict[SessionContext, str]" = OrderedDict()
        self._highest_epoch: Dict[EpochKey, int] = {}
        self.restored = False
        self.counters: Dict[str, int] = {
            "admitted": 0, "closed_by_epoch": 0, "closed_by_displacement": 0,
            "closed_explicitly": 0, "rejected_closed": 0, "rejected_stale_epoch": 0,
            "rejected_stale_frame": 0, "forgotten_closures": 0,
        }

    # ------------------------------------------------------------- read-only
    def state_of(self, ctx: SessionContext) -> SessionState:
        if ctx in self._active:
            return SessionState.ACTIVE
        if ctx in self._closed:
            return SessionState.CLOSED
        return SessionState.UNKNOWN

    def classify(self, ctx: SessionContext, frame_id: int) -> Tuple[Admission, str]:
        """Pre-authentication verdict.  Mutates nothing but the counters.

        Counters are diagnostics, never inputs to a decision, so traffic that
        moves them changes no outcome.
        """
        sid, epoch, stream = ctx
        if ctx in self._closed:
            self.counters["rejected_closed"] += 1
            return (Admission.CLOSED,
                    f"session {sid.hex()[:8]} epoch {epoch} is closed "
                    f"({self._closed[ctx]}); it cannot be reopened")
        seen = self._highest_epoch.get((sid, stream))
        if seen is not None and epoch < seen:
            self.counters["rejected_stale_epoch"] += 1
            return (Admission.STALE_EPOCH, f"epoch {epoch} retired (current {seen})")
        rec = self._active.get(ctx)
        if rec is not None and rec.newest_frame >= 0 and \
                frame_id < rec.newest_frame - self.max_frame_age:
            self.counters["rejected_stale_frame"] += 1
            return (Admission.STALE_FRAME,
                    f"frame {frame_id} older than the display deadline "
                    f"(newest {rec.newest_frame} in this session/epoch)")
        return Admission.ADMIT, ""

    def window_for(self, ctx: SessionContext) -> ReplayWindow:
        """The replay window a unit of ``ctx`` must be checked against.

        For an active context this is the *stored* window - the whole point of
        the ledger.  For an unknown context it is a fresh window that is only
        installed if :meth:`commit` is reached, i.e. after authentication.
        """
        rec = self._active.get(ctx)
        if rec is not None:
            return rec.replay
        return ReplayWindow(self.replay_window)

    # --------------------------------------------------- post-authentication
    def commit(self, ctx: SessionContext, window: ReplayWindow, frame_id: int,
               clock: float) -> SessionRecord:
        """Record an authenticated unit.  Only ever called after AEAD succeeds."""
        sid, epoch, stream = ctx
        rec = self._active.get(ctx)
        if rec is None:
            self._retire_lower_epochs(sid, stream, epoch)
            self._make_room()
            rec = SessionRecord(ctx=ctx, replay=window, first_seen=clock)
            self._active[ctx] = rec
            self.counters["admitted"] += 1
        rec.last_seen = clock
        rec.n_units += 1
        if frame_id > rec.newest_frame:
            rec.newest_frame = frame_id
        key = (sid, stream)
        if epoch > self._highest_epoch.get(key, -1):
            self._highest_epoch[key] = epoch
        self._active.move_to_end(ctx)
        return rec

    def close(self, ctx: SessionContext,
              reason: str = "closed by the application") -> None:
        self._active.pop(ctx, None)
        self._tombstone(ctx, reason)
        self.counters["closed_explicitly"] += 1

    # ------------------------------------------------------------- internals
    def _retire_lower_epochs(self, sid: bytes, stream: int, epoch: int) -> None:
        for ctx in [c for c in self._active
                    if c[0] == sid and c[2] == stream and c[1] < epoch]:
            self._active.pop(ctx, None)
            self._tombstone(ctx, f"superseded by epoch {epoch}")
            self.counters["closed_by_epoch"] += 1

    def _make_room(self) -> None:
        while len(self._active) >= self.active_capacity:
            ctx, _rec = self._active.popitem(last=False)
            self._tombstone(ctx, "displaced from the active table by a newer session")
            self.counters["closed_by_displacement"] += 1

    def _tombstone(self, ctx: SessionContext, reason: str) -> None:
        if ctx in self._closed:
            self._closed.move_to_end(ctx)
            return
        while len(self._closed) >= self.retired_capacity:
            self._closed.popitem(last=False)
            self.counters["forgotten_closures"] += 1
        self._closed[ctx] = reason

    # ------------------------------------------------------------ inspection
    def __len__(self) -> int:
        return len(self._active)

    @property
    def active(self) -> "OrderedDict[SessionContext, SessionRecord]":
        return self._active

    @property
    def closed(self) -> "OrderedDict[SessionContext, str]":
        return self._closed

    def newest_frame(self, ctx: SessionContext) -> int:
        rec = self._active.get(ctx)
        return rec.newest_frame if rec else -1

    def describe(self) -> Dict[str, Any]:
        return {
            "active_capacity": self.active_capacity,
            "retired_capacity": self.retired_capacity,
            "n_active": len(self._active),
            "n_closed_remembered": len(self._closed),
            "cold_start": self.cold_start,
            "restored_from_state": self.restored,
            "counters": dict(self.counters),
            "freshness_after_restart": (
                "стан відновлено з файла" if self.restored else
                "tofu: перший автентифікований блок невідомого контексту "
                "встановлює позначку свіжості; це безпечно лише якщо передавач "
                "після перезапуску приймача починає новий session_id або "
                "піднімає session_epoch"),
            "exactness": (
                "усі закриті контексти запам'ятані точно"
                if not self.counters["forgotten_closures"] else
                f"{self.counters['forgotten_closures']} закритих контекстів "
                "витіснено з памʼяті: для них свіжість більше не доводиться"),
        }

    # ----------------------------------------------------------- persistence
    def export_state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "active_capacity": self.active_capacity,
            "retired_capacity": self.retired_capacity,
            "replay_window": self.replay_window,
            "max_frame_age": self.max_frame_age,
            "active": [r.to_dict() for r in self._active.values()],
            "closed": [{"session_id": c[0].hex(), "epoch": c[1], "stream_id": c[2],
                        "reason": why} for c, why in self._closed.items()],
            "highest_epoch": [{"session_id": k[0].hex(), "stream_id": k[1],
                               "epoch": v} for k, v in self._highest_epoch.items()],
            "counters": dict(self.counters),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Restore a persisted ledger.  Freshness survives the restart."""
        self._active.clear()
        self._closed.clear()
        self._highest_epoch.clear()
        for d in state.get("active", []):
            rec = SessionRecord.from_dict(d)
            self._active[rec.ctx] = rec
        for d in state.get("closed", []):
            ctx = (bytes.fromhex(str(d["session_id"])), int(d["epoch"]),
                   int(d["stream_id"]))
            self._closed[ctx] = str(d.get("reason", "closed before the restart"))
        for d in state.get("highest_epoch", []):
            self._highest_epoch[(bytes.fromhex(str(d["session_id"])),
                                 int(d["stream_id"]))] = int(d["epoch"])
        for k, v in (state.get("counters") or {}).items():
            if k in self.counters:
                self.counters[k] = int(v)
        self.restored = True
        self.cold_start = "persisted"


__all__ = ["SessionContext", "EpochKey", "SessionState", "Admission",
           "SessionRecord", "KeyCache", "SessionLedger"]
