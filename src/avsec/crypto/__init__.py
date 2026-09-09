"""Keys, sessions, AEAD and freshness.

Primitives come from ``cryptography`` (ChaCha20-Poly1305 / AES-GCM / HKDF-SHA256).
Nothing cryptographic is re-implemented here; this module only integrates the
standard constructions and enforces the nonce / freshness discipline.

Nonce construction (profile v1)
-------------------------------
``nonce = nonce_prefix(4 B) || counter(8 B, big endian)`` for the 96-bit nonce of
both ChaCha20-Poly1305 and AES-GCM.  ``nonce_prefix`` is derived per session and
direction with HKDF; ``counter`` is strictly monotone inside a session.  Two
encryptions therefore share a nonce only if they share a session key *and* a
counter value, which the sealer refuses to produce.

Restart policy
--------------
A session identifier is 8 random bytes drawn from ``os.urandom``.  After a
restart a transmitter must either (a) start a **new** session id, or (b) restore
the persisted counter of the old session.  Reusing an old session id without its
counter is rejected by :class:`SessionState.restore`.
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PROTOCOL_LABEL = b"avsec/v1"
TAG_LEN = 16          # full 128-bit Poly1305 / GCM tag, never truncated
KEY_LEN = 32
NONCE_LEN = 12
SESSION_ID_LEN = 8
COUNTER_LEN = 8
COUNTER_MAX = (1 << (8 * COUNTER_LEN)) - 1

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
) -> SessionKeys:
    """HKDF-SHA256 session derivation.

    ``salt = session_id``; ``info`` binds protocol label, direction, stream id
    and algorithm, so two directions or two streams never share a key.
    """
    if algorithm not in AEAD_ALGORITHMS:
        raise CryptoError(f"algorithm must be one of {AEAD_ALGORITHMS}")
    if len(session_id) != SESSION_ID_LEN:
        raise CryptoError(f"session id must be {SESSION_ID_LEN} bytes")
    info_base = b"|".join([PROTOCOL_LABEL, direction.encode(), bytes([stream_id & 0xFF]),
                           algorithm.encode()])
    okm = _hkdf(master.key, session_id, info_base + b"|key+nonceprefix", KEY_LEN + 4)
    return SessionKeys(
        session_id=session_id,
        direction=direction,
        stream_id=stream_id,
        algorithm=algorithm,
        key=okm[:KEY_LEN],
        nonce_prefix=okm[KEY_LEN:],
        origin=master.origin,
    )


def new_session_id() -> bytes:
    return secrets.token_bytes(SESSION_ID_LEN)


# ------------------------------------------------------------------ transmitter
@dataclass
class SessionState:
    """Persistent transmitter state: which counter values were already used."""

    session_id_hex: str
    next_counter: int

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"session_id": self.session_id_hex, "next_counter": self.next_counter}, fh)

    @staticmethod
    def restore(path: str, session_id: bytes) -> "SessionState":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if d["session_id"] != session_id.hex():
            raise CryptoError(
                "refusing to reuse a session id without its persisted counter "
                "(would risk nonce reuse); start a new session instead"
            )
        return SessionState(d["session_id"], int(d["next_counter"]))


class Sealer:
    """Encrypt-and-authenticate with a strictly monotone counter."""

    def __init__(self, keys: SessionKeys, start_counter: int = 0) -> None:
        self.keys = keys
        self._aead = keys.aead()
        self._counter = int(start_counter)

    @property
    def counter(self) -> int:
        return self._counter

    def next_counter(self) -> int:
        if self._counter > COUNTER_MAX:
            raise NonceExhausted("session counter exhausted; rekey with a new session id")
        c = self._counter
        self._counter += 1
        return c

    def seal(self, plaintext: bytes, aad: bytes, counter: Optional[int] = None) -> Tuple[int, bytes]:
        """Return ``(counter, ciphertext||tag)``."""
        c = self.next_counter() if counter is None else int(counter)
        if c > COUNTER_MAX:
            raise NonceExhausted("counter beyond the profile limit")
        return c, self._aead.encrypt(self.keys.nonce(c), plaintext, aad)

    def state(self) -> SessionState:
        return SessionState(self.keys.session_id.hex(), self._counter)


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

    def __init__(self, keys: SessionKeys, start_counter: int = 0) -> None:
        self.keys = keys
        self._counter = int(start_counter)

    def seal(self, plaintext: bytes, aad: bytes, counter: Optional[int] = None
             ) -> Tuple[int, bytes]:
        import hashlib

        c = self.next_counter() if counter is None else int(counter)
        checksum = hashlib.sha256(aad + plaintext).digest()[:TAG_LEN]
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
            "nonce_layout": "prefix(4B, HKDF per session+direction+stream) || counter(8B BE)",
            "kdf": "HKDF-SHA256(salt=session_id, info=label|direction|stream|alg)",
            "replay_window": self.replay_window,
            "direction": self.direction,
            "stream_id": self.stream_id,
        }


def make_session(
    master: MasterSecret,
    profile: Optional[CryptoProfile] = None,
    session_id: Optional[bytes] = None,
) -> Tuple[SessionKeys, Sealer, Opener]:
    """Convenience constructor used by the transmitter/receiver pair in tests."""
    profile = profile or CryptoProfile()
    sid = session_id or new_session_id()
    keys = derive_session_keys(master, sid, profile.direction, profile.stream_id, profile.algorithm)
    return keys, Sealer(keys), Opener(keys, profile.replay_window)


__all__ = [
    "PROTOCOL_LABEL", "TAG_LEN", "KEY_LEN", "NONCE_LEN", "SESSION_ID_LEN", "COUNTER_MAX",
    "AEAD_ALGORITHMS", "DIRECTION_UPLINK", "DIRECTION_DOWNLINK",
    "CryptoError", "NonceExhausted", "AuthenticationFailed", "ReplayDetected",
    "MasterSecret", "generate_master_secret", "lab_master_secret", "load_master_secret",
    "SessionKeys", "derive_session_keys", "new_session_id", "SessionState",
    "Sealer", "Opener", "ReplayWindow", "CryptoProfile", "make_session",
    "NullSealer", "NullOpener",
]
