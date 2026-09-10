"""Keys, sessions, AEAD and freshness.

Primitives come from ``cryptography`` (ChaCha20-Poly1305 / AES-GCM / HKDF-SHA256).
Nothing cryptographic is re-implemented here; this module only integrates the
standard constructions and enforces the nonce / freshness discipline.

Nonce construction (profile v2)
-------------------------------
``nonce = nonce_prefix(4 B) || counter(8 B, big endian)`` for the 96-bit nonce of
both ChaCha20-Poly1305 and AES-GCM.  ``nonce_prefix`` is derived per session,
**epoch**, direction and stream with HKDF; ``counter`` is strictly monotone
inside one epoch.  Two encryptions therefore share a nonce only if they share a
session key *and* a counter value.

Counters are handed out by :meth:`Sealer.seal`, which allocates the value itself
and consumes it before the caller can observe it.  There is no API that lets a
caller choose or repeat a counter, so a nonce cannot be reused through misuse of
this module (defect F03).

Session lifecycle and restart (profile v2)
-----------------------------------------
A session identifier is 8 random bytes from ``os.urandom``.  Every session also
carries a 32-bit **epoch** that is authenticated in the header *and* mixed into
the key derivation, so a different epoch means a different key.  That makes a
restart safe in a way a persisted counter alone cannot:

* preferred - start a new session id (fresh keys, epoch 0);
* long-term key, autonomous restart - keep the session id and advance the epoch,
  persisting it **before** any data of the new epoch is sent
  (:class:`SessionState.begin_epoch`).  Because the key changes with the epoch,
  restarting the counter at zero cannot repeat a nonce.

The receiver keeps the highest accepted epoch per session and rejects units of a
retired epoch, so a recording of a finished epoch cannot reopen it.  Both the
epoch and the frame watermark advance **only after** a successful
authentication.
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PROTOCOL_LABEL = b"avsec/v2"
TAG_LEN = 16          # full 128-bit Poly1305 / GCM tag, never truncated
KEY_LEN = 32
NONCE_LEN = 12
SESSION_ID_LEN = 8
COUNTER_LEN = 8
COUNTER_MAX = (1 << (8 * COUNTER_LEN)) - 1
EPOCH_MAX = (1 << 32) - 1

AEAD_ALGORITHMS = ("chacha20poly1305", "aes256gcm")

DIRECTION_UPLINK = "uplink"       # aircraft -> ground
DIRECTION_DOWNLINK = "downlink"   # ground -> aircraft (unused by default)


class CryptoError(Exception):
    """Base class for integration errors detected by this module."""


class NonceExhausted(CryptoError):
    """The per-session counter reached its limit; a new session is required."""


class AuthenticationFailed(CryptoError):
    """AEAD tag verification failed."""


class ReplayDetected(CryptoError):
    """A unit sequence number was already accepted under this session."""


# ------------------------------------------------------------------ key material
@dataclass(frozen=True)
class MasterSecret:
    """Pre-shared secret shared by the two endpoints.

    ``origin`` is either ``'secure'`` (os.urandom or an operator supplied file)
    or ``'lab'`` (deterministically derived from a *published* benchmark seed).
    Lab secrets are reproducible on purpose and must never be used to protect
    anything real; the flag travels with the object so reports cannot confuse
    the two.
    """

    key: bytes
    origin: str = "secure"

    def __post_init__(self) -> None:
        if len(self.key) != KEY_LEN:
            raise CryptoError(f"master secret must be {KEY_LEN} bytes, got {len(self.key)}")
        if self.origin not in ("secure", "lab"):
            raise CryptoError("origin must be 'secure' or 'lab'")

    @property
    def is_lab(self) -> bool:
        return self.origin == "lab"

    def __repr__(self) -> str:  # never print key bytes
        return f"MasterSecret(origin={self.origin!r}, key=<{KEY_LEN} bytes redacted>)"


def generate_master_secret() -> MasterSecret:
    """Fresh operational secret from the OS CSPRNG."""
    return MasterSecret(os.urandom(KEY_LEN), origin="secure")


def lab_master_secret(benchmark_seed: int) -> MasterSecret:
    """Reproducible secret for benchmarks only.

    The benchmark seed of an experiment must not silently drive operational
    keys, so this constructor is the *only* deterministic path and it stamps
    ``origin='lab'``.
    """
    hkdf = HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=b"avsec-lab-salt",
                info=PROTOCOL_LABEL + b"/lab-benchmark-key")
    return MasterSecret(hkdf.derive(b"lab:" + str(int(benchmark_seed)).encode()), origin="lab")


def load_master_secret(path: Optional[str] = None, env_var: str = "AVSEC_MASTER_KEY") -> MasterSecret:
    """Load an operational secret from a file (32 raw or 64 hex bytes) or env var."""
    if path:
        with open(path, "rb") as fh:
            raw = fh.read().strip()
        key = bytes.fromhex(raw.decode()) if len(raw) == 2 * KEY_LEN else raw
        return MasterSecret(key, origin="secure")
    val = os.environ.get(env_var)
    if not val:
        raise CryptoError(
            f"no master secret: pass a key file or set {env_var} (64 hex chars). "
            "Keys are never stored in this repository."
        )
    return MasterSecret(bytes.fromhex(val.strip()), origin="secure")


# ---------------------------------------------------------------------- session
def _hkdf(secret: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(secret)


@dataclass
class SessionKeys:
    session_id: bytes
    direction: str
    stream_id: int
    algorithm: str
    key: bytes = field(repr=False)
    nonce_prefix: bytes = field(repr=False)
    origin: str = "secure"
    epoch: int = 0

    @property
    def context(self) -> Tuple[bytes, int, str, int, str]:
        """The full key context.  Nonce uniqueness is required within it."""
        return (self.session_id, self.epoch, self.direction, self.stream_id,
                self.algorithm)

    def aead(self):
        if self.algorithm == "chacha20poly1305":
            return ChaCha20Poly1305(self.key)
        if self.algorithm == "aes256gcm":
            return AESGCM(self.key)
        raise CryptoError(f"unknown AEAD algorithm {self.algorithm!r}")

    def nonce(self, counter: int) -> bytes:
        if not 0 <= counter <= COUNTER_MAX:
            raise NonceExhausted(f"counter {counter} outside [0, {COUNTER_MAX}]")
        return self.nonce_prefix + counter.to_bytes(COUNTER_LEN, "big")


def derive_session_keys(
    master: MasterSecret,
    session_id: bytes,
    direction: str = DIRECTION_UPLINK,
    stream_id: int = 0,
    algorithm: str = "chacha20poly1305",
    epoch: int = 0,
) -> SessionKeys:
    """HKDF-SHA256 session derivation.

    ``salt = session_id``; ``info`` binds protocol label, direction, stream id,
    algorithm and **epoch**, so two directions, two streams or two epochs never
    share a key.  Because the epoch changes the key, a counter that restarts at
    zero in a new epoch cannot repeat a nonce.
    """
    if algorithm not in AEAD_ALGORITHMS:
        raise CryptoError(f"algorithm must be one of {AEAD_ALGORITHMS}")
    if len(session_id) != SESSION_ID_LEN:
        raise CryptoError(f"session id must be {SESSION_ID_LEN} bytes")
    if not 0 <= int(epoch) <= EPOCH_MAX:
        raise CryptoError(f"epoch must be in [0, {EPOCH_MAX}]")
    info_base = b"|".join([PROTOCOL_LABEL, direction.encode(), bytes([stream_id & 0xFF]),
                           algorithm.encode(), int(epoch).to_bytes(4, "big")])
    okm = _hkdf(master.key, session_id, info_base + b"|key+nonceprefix", KEY_LEN + 4)
    return SessionKeys(
        session_id=session_id,
        direction=direction,
        stream_id=stream_id,
        algorithm=algorithm,
        key=okm[:KEY_LEN],
        nonce_prefix=okm[KEY_LEN:],
        origin=master.origin,
        epoch=int(epoch),
    )


def new_session_id() -> bytes:
    return secrets.token_bytes(SESSION_ID_LEN)


class SessionIdSource:
    """Where session identifiers come from.

    Defect F09: in ``lab`` mode a benchmark must be bit-for-bit reproducible, and
    a session id drawn from ``os.urandom`` breaks that - two identical runs
    produced different ciphertext and therefore different channel outcomes on the
    margin.  Operational mode must keep using the OS CSPRNG.  The two are
    separate classes so a report can never confuse them.
    """

    origin = "secure"

    def next(self, context: str = "") -> bytes:
        raise NotImplementedError

    def describe(self) -> Dict[str, object]:
        return {"origin": self.origin, "class": type(self).__name__}


class SecureSessionIds(SessionIdSource):
    """Operational: fresh, unpredictable identifiers from the OS CSPRNG."""

    origin = "secure"

    def next(self, context: str = "") -> bytes:
        return secrets.token_bytes(SESSION_ID_LEN)


class LabSessionIds(SessionIdSource):
    """Benchmark only: deterministic identifiers derived from a recorded seed.

    Derived with HKDF from ``(crypto_seed, run_identity, context, counter)``, so
    two identical lab runs produce identical ciphertext, identical signals and
    identical metrics.  Never use this to protect anything: the identifiers are
    predictable by construction, and the flag ``origin='lab'`` says so.
    """

    origin = "lab"

    def __init__(self, crypto_seed: int, run_identity: str = "") -> None:
        self.crypto_seed = int(crypto_seed)
        self.run_identity = str(run_identity)
        self._counters: Dict[str, int] = {}

    def next(self, context: str = "") -> bytes:
        n = self._counters.get(context, 0)
        self._counters[context] = n + 1
        material = "|".join([self.run_identity, context, str(n)]).encode("utf-8")
        return _hkdf(
            int(self.crypto_seed).to_bytes(8, "big", signed=False),
            b"avsec-lab-session-id", PROTOCOL_LABEL + b"|sid|" + material,
            SESSION_ID_LEN)

    def reset(self) -> None:
        self._counters.clear()

    def describe(self) -> Dict[str, object]:
        d = super().describe()
        d.update({"crypto_seed": self.crypto_seed, "run_identity": self.run_identity,
                  "warning": "predictable identifiers; benchmarks only"})
        return d


def make_session_id_source(mode: str, crypto_seed: int = 0,
                           run_identity: str = "") -> SessionIdSource:
    if mode == "lab":
        return LabSessionIds(crypto_seed, run_identity)
    if mode == "secure":
        return SecureSessionIds()
    raise CryptoError(f"session id mode must be 'lab' or 'secure', got {mode!r}")


# ------------------------------------------------------------------ transmitter
@dataclass
class SessionState:
    """Persistent transmitter state for one (session id, epoch).

    ``reserved_through`` is the highest counter value that has been *durably
    reserved*.  Counters are handed out only below that watermark, and the
    watermark is persisted **before** the counters it covers are used, so a
    crash between encrypting and saving can never lead to reuse.
    """

    session_id_hex: str
    epoch: int
    next_counter: int
    reserved_through: int

    def to_dict(self) -> Dict[str, object]:
        return {"session_id": self.session_id_hex, "epoch": self.epoch,
                "next_counter": self.next_counter,
                "reserved_through": self.reserved_through}

    def save(self, path: str) -> None:
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)          # atomic; a crash leaves the old state intact

    @staticmethod
    def load(path: str) -> "SessionState":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return SessionState(d["session_id"], int(d.get("epoch", 0)),
                            int(d["next_counter"]),
                            int(d.get("reserved_through", d["next_counter"])))

    @staticmethod
    def restore(path: str, session_id: bytes, epoch: int = 0) -> "SessionState":
        """Resume the *same* epoch.  Only safe if a forward reservation survived.

        Resuming at the last counter the process happened to write would repeat
        every counter issued after the last save - exactly the crash-between-
        encrypt-and-persist hole F03 warns about.  Resuming is therefore allowed
        only above a durable reservation; without one the caller must move to a
        new epoch.
        """
        st = SessionState.load(path)
        if st.session_id_hex != session_id.hex():
            raise CryptoError(
                "refusing to reuse a session id without its persisted state "
                "(would risk nonce reuse); start a new session or a new epoch"
            )
        if st.epoch != int(epoch):
            raise CryptoError(
                f"persisted state belongs to epoch {st.epoch}, not {epoch}; "
                "use begin_epoch() to move to a fresh epoch"
            )
        if st.reserved_through <= st.next_counter:
            raise CryptoError(
                "the persisted state carries no forward counter reservation, so "
                "resuming this epoch could repeat a nonce; use begin_epoch() "
                "(a new epoch derives a new key, so the counter may restart)"
            )
        # resume above the durable reservation, never at the last used counter
        st.next_counter = st.reserved_through
        return st

    @staticmethod
    def begin_epoch(path: str, session_id: bytes) -> "SessionState":
        """Advance to the next epoch for this session id and persist it first.

        This is the safe autonomous restart: a new epoch means a new key, so the
        counter may legitimately start at zero.
        """
        try:
            prev = SessionState.load(path)
            epoch = prev.epoch + 1 if prev.session_id_hex == session_id.hex() else 0
        except (FileNotFoundError, KeyError, ValueError):
            epoch = 0
        if epoch > EPOCH_MAX:
            raise CryptoError("session epoch exhausted; negotiate a new session id")
        st = SessionState(session_id.hex(), epoch, 0, 0)
        st.save(path)
        return st


class Sealer:
    """Encrypt-and-authenticate; the sealer alone allocates nonce counters.

    The counter is consumed *before* the associated data is built and before the
    AEAD call, and it is never returned to the pool - not even if building the
    header or encrypting raises.  There is deliberately no way for a caller to
    supply a counter (defect F03).
    """

    def __init__(self, keys: SessionKeys, start_counter: int = 0,
                 reserve_chunk: int = 4096,
                 persist: Optional[Callable[["SessionState"], None]] = None) -> None:
        self.keys = keys
        self._aead = keys.aead()
        self._counter = int(start_counter)
        self._issued: set[int] = set()
        self._reserve_chunk = max(1, int(reserve_chunk))
        self._persist = persist
        self._reserved_through = int(start_counter)
        if persist is not None:
            self._extend_reservation()

    @property
    def counter(self) -> int:
        """Next counter that would be issued (diagnostic only)."""
        return self._counter

    @property
    def reserved_through(self) -> int:
        return self._reserved_through

    def _extend_reservation(self) -> None:
        self._reserved_through = self._counter + self._reserve_chunk
        if self._persist is not None:
            self._persist(self.state())

    def _take(self) -> int:
        """Allocate one counter.  Never returns the same value twice."""
        if self._counter > COUNTER_MAX:
            raise NonceExhausted(
                "session counter exhausted; move to a new epoch or session id")
        if self._persist is not None and self._counter >= self._reserved_through:
            self._extend_reservation()
        c = self._counter
        self._counter += 1          # consumed now: an exception below cannot reuse it
        return c

    def seal(self, plaintext: bytes,
             aad: Union[bytes, Callable[[int], bytes]]) -> Tuple[int, bytes]:
        """Return ``(counter, ciphertext||tag)``.

        ``aad`` may be raw bytes, or a callable receiving the freshly allocated
        counter and returning the associated data.  The callable form exists so
        a header that must carry its own sequence number can be built *inside*
        the allocation, instead of the caller choosing a counter itself.
        """
        c = self._take()
        assert c not in self._issued, "internal error: counter reuse"
        self._issued.add(c)
        if len(self._issued) > 1 << 20:          # bounded bookkeeping
            self._issued = {x for x in self._issued if x > c - (1 << 19)}
        data = aad(c) if callable(aad) else aad
        return c, self._aead.encrypt(self.keys.nonce(c), plaintext, data)

    def state(self) -> SessionState:
        return SessionState(self.keys.session_id.hex(), self.keys.epoch,
                            self._counter, self._reserved_through)


class ReplayWindow:
    """Sliding-window anti-replay over unit sequence numbers.

    The window is only advanced **after** a successful authentication, so a
    forged packet can never push the window forward and lock out real data.
    """

    def __init__(self, size: int = 4096) -> None:
        if size < 1:
            raise ValueError("window size must be >= 1")
        self.size = size
        self.highest = -1
        self._seen: set[int] = set()

    def check(self, seq: int) -> str:
        """Pre-authentication classification: 'new', 'replay' or 'too_old'."""
        if seq in self._seen:
            return "replay"
        if self.highest >= 0 and seq <= self.highest - self.size:
            return "too_old"
        return "new"

    def accept(self, seq: int) -> None:
        """Record an *authenticated* sequence number."""
        self._seen.add(seq)
        if seq > self.highest:
            self.highest = seq
        cut = self.highest - self.size
        if cut > 0 and len(self._seen) > 4 * self.size:
            self._seen = {s for s in self._seen if s > cut}


class Opener:
    """Verify-and-decrypt with replay protection.

    The caller must not touch the plaintext unless :meth:`open` returns it.
    """

    def __init__(self, keys: SessionKeys, replay_window: int = 4096) -> None:
        self.keys = keys
        self._aead = keys.aead()
        self.replay = ReplayWindow(replay_window)

    def open(self, counter: int, ciphertext: bytes, aad: bytes) -> bytes:
        status = self.replay.check(counter)
        if status == "replay":
            raise ReplayDetected(f"unit sequence {counter} already accepted")
        if status == "too_old":
            raise ReplayDetected(f"unit sequence {counter} is outside the replay window")
        try:
            pt = self._aead.decrypt(self.keys.nonce(counter), ciphertext, aad)
        except Exception as exc:  # InvalidTag and friends
            raise AuthenticationFailed(str(exc) or "tag verification failed") from exc
        self.replay.accept(counter)  # only after successful authentication
        return pt


class NullSealer(Sealer):
    """DIAGNOSTIC ONLY - no confidentiality and no authentication.

    Exists so that the *cost* of the modem and the FEC can be separated from the
    cost of cryptography (baseline B0-digital).  It keeps the wire size
    identical by appending a 16-byte CRC-derived checksum in place of the tag.
    It is never selected by a secured profile, and every report that uses it is
    labelled ``secure=False``.
    """

    def __init__(self, keys: SessionKeys, start_counter: int = 0,
                 reserve_chunk: int = 4096, persist=None) -> None:
        self.keys = keys
        self._counter = int(start_counter)
        self._issued = set()
        self._reserve_chunk = max(1, int(reserve_chunk))
        self._persist = persist
        self._reserved_through = int(start_counter)

    def seal(self, plaintext: bytes,
             aad: Union[bytes, Callable[[int], bytes]]) -> Tuple[int, bytes]:
        import hashlib

        c = self._take()
        data = aad(c) if callable(aad) else aad
        checksum = hashlib.sha256(data + plaintext).digest()[:TAG_LEN]
        return c, plaintext + checksum


class NullOpener(Opener):
    """DIAGNOSTIC ONLY counterpart of :class:`NullSealer` (no authentication)."""

    def __init__(self, keys: SessionKeys, replay_window: int = 4096) -> None:
        self.keys = keys
        self.replay = ReplayWindow(replay_window)

    def open(self, counter: int, ciphertext: bytes, aad: bytes) -> bytes:
        import hashlib

        if len(ciphertext) < TAG_LEN:
            raise AuthenticationFailed("truncated diagnostic unit")
        body, checksum = ciphertext[:-TAG_LEN], ciphertext[-TAG_LEN:]
        if hashlib.sha256(aad + body).digest()[:TAG_LEN] != checksum:
            raise AuthenticationFailed("diagnostic checksum mismatch (NOT authentication)")
        self.replay.accept(counter)
        return body


@dataclass
class CryptoProfile:
    """Everything the two endpoints must agree on before a session starts."""

    algorithm: str = "chacha20poly1305"
    replay_window: int = 4096
    direction: str = DIRECTION_UPLINK
    stream_id: int = 0

    def describe(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "tag_bytes": TAG_LEN,
            "key_bytes": KEY_LEN,
            "nonce_bytes": NONCE_LEN,
            "nonce_layout": "prefix(4B, HKDF per session+epoch+direction+stream) "
                            "|| counter(8B BE)",
            "kdf": "HKDF-SHA256(salt=session_id, info=label|direction|stream|alg|epoch)",
            "epoch_bits": 32,
            "counter_allocation": "sealer-allocated, one-shot; no caller-supplied counter",
            "replay_window": self.replay_window,
            "direction": self.direction,
            "stream_id": self.stream_id,
        }


def make_session(
    master: MasterSecret,
    profile: Optional[CryptoProfile] = None,
    session_id: Optional[bytes] = None,
    epoch: int = 0,
) -> Tuple[SessionKeys, Sealer, Opener]:
    """Convenience constructor used by the transmitter/receiver pair in tests."""
    profile = profile or CryptoProfile()
    sid = session_id or new_session_id()
    keys = derive_session_keys(master, sid, profile.direction, profile.stream_id,
                               profile.algorithm, epoch)
    return keys, Sealer(keys), Opener(keys, profile.replay_window)


__all__ = [
    "PROTOCOL_LABEL", "TAG_LEN", "KEY_LEN", "NONCE_LEN", "SESSION_ID_LEN", "COUNTER_MAX",
    "AEAD_ALGORITHMS", "DIRECTION_UPLINK", "DIRECTION_DOWNLINK", "EPOCH_MAX",
    "CryptoError", "NonceExhausted", "AuthenticationFailed", "ReplayDetected",
    "MasterSecret", "generate_master_secret", "lab_master_secret", "load_master_secret",
    "SessionKeys", "derive_session_keys", "new_session_id", "SessionState",
    "SessionIdSource", "SecureSessionIds", "LabSessionIds",
    "make_session_id_source",
    "Sealer", "Opener", "ReplayWindow", "CryptoProfile", "make_session",
    "NullSealer", "NullOpener",
]
