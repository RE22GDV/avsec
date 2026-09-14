"""Laboratory cryptanalysis of the permutation baselines and protocol checks.

Nothing here is a general break of "image scrambling"; each function states the
assumptions under which it works.  Recovering one *static* permutation says
nothing about an independent fresh permutation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from avsec.lfsr import (
    LFSR,
    LFSRConfig,
    BlockScrambler,
    ScramblerConfig,
    invert_permutation,
    lfsr_permutation,
)


# ------------------------------------------------------------------ helpers
def split_tiles(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w = img.shape
    bh, bw = h // rows, w // cols
    img = img[: bh * rows, : bw * cols]
    return img.reshape(rows, bh, cols, bw).swapaxes(1, 2).reshape(rows * cols, bh, bw)


def join_tiles(tiles: np.ndarray, rows: int, cols: int) -> np.ndarray:
    bh, bw = tiles.shape[1], tiles.shape[2]
    return tiles.reshape(rows, cols, bh, bw).swapaxes(1, 2).reshape(rows * bh, cols * bw)


def permutation_accuracy(est: np.ndarray, true: np.ndarray) -> float:
    est = np.asarray(est)
    true = np.asarray(true)
    n = min(est.size, true.size)
    return float((est[:n] == true[:n]).mean()) if n else 0.0


@dataclass
class AttackResult:
    name: str
    assumptions: str
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    recovered_permutation: Optional[np.ndarray] = None
    recovered_tables: Optional[np.ndarray] = None
    reconstructed: Optional[np.ndarray] = None
    seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "assumptions": self.assumptions,
            "success": bool(self.success),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "seconds": self.seconds,
            "notes": self.notes,
        }


# ------------------------------------------------- A1: chosen plaintext frame
def attack_chosen_plaintext(
    scrambler: BlockScrambler, frame_index: int = 0, height: int = 240, width: int = 320
) -> AttackResult:
    """Recover a *static* permutation exactly using one chosen input frame.

    Assumptions: the attacker can make the transmitter send a frame of their
    choosing (or the operator points the camera at a prepared target), the
    permutation is static, and the channel is clean enough for tile identity to
    survive.  Cost: one frame, O(N) work.
    """
    from avsec.sources import pattern_blocks

    t0 = time.perf_counter()
    rows, cols = scrambler.cfg.grid_rows, scrambler.cfg.grid_cols
    probe = pattern_blocks(height, width, rows, cols)
    scrambled = scrambler.scramble(probe, frame_index)

    ref = split_tiles(scrambler._fit(probe), rows, cols)
    got = split_tiles(scrambled, rows, cols)
    # each reference tile has a unique constant level -> identify by mean value
    ref_key = ref.reshape(ref.shape[0], -1).mean(axis=1)
    got_key = got.reshape(got.shape[0], -1).mean(axis=1)
    est = np.array([int(np.argmin(np.abs(ref_key - g))) for g in got_key], dtype=np.int64)

    true = scrambler.permutation(frame_index)
    acc = permutation_accuracy(est, true)
    return AttackResult(
        name="chosen_plaintext_permutation_recovery",
        assumptions="static permutation; attacker controls or knows one full input frame",
        success=acc == 1.0,
        metrics={"permutation_accuracy": acc, "n_blocks": float(rows * cols), "n_frames_used": 1.0},
        recovered_permutation=est,
        seconds=time.perf_counter() - t0,
        notes="Recovering a static permutation does not break an independently "
              "regenerated permutation.",
    )


# --------------------------------------------------- A2: known plaintext pair
def attack_known_pair(
    original: np.ndarray, scrambled: np.ndarray, rows: int, cols: int
) -> AttackResult:
    """Match tiles between a known original and its scrambled version.

    Assumptions: the attacker holds one (original, scrambled) pair for the same
    permutation.  Matching is by nearest tile in L2; ties are resolved greedily,
    which makes the result exact for generic natural content and degraded on
    frames with many identical tiles (e.g. large flat areas).
    """
    t0 = time.perf_counter()
    a = split_tiles(original, rows, cols).reshape(rows * cols, -1).astype(np.float64)
    b = split_tiles(scrambled, rows, cols).reshape(rows * cols, -1).astype(np.float64)
    n = rows * cols
    # cost[i, j] = distance from scrambled tile i to original tile j
    cost = ((b[:, None, :] - a[None, :, :]) ** 2).sum(axis=2)
    est = np.full(n, -1, dtype=np.int64)
    taken = np.zeros(n, dtype=bool)
    order = np.argsort(cost.min(axis=1))  # most confident first
    for i in order:
        cand = np.argsort(cost[i])
        for j in cand:
            if not taken[j]:
                est[i] = j
                taken[j] = True
                break
    return AttackResult(
        name="known_plaintext_tile_matching",
        assumptions="one (original, scrambled) pair under the same permutation",
        success=bool((est >= 0).all()),
        metrics={"n_blocks": float(n), "mean_match_cost": float(cost[np.arange(n), est].mean())},
        recovered_permutation=est,
        seconds=time.perf_counter() - t0,
    )


# ----------------------------------------------------- A3: boundary (jigsaw)
def _edge_costs(tiles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Right- and down-neighbour dissimilarity between all tile pairs.

    ``right[i, j]`` = cost of putting tile ``j`` immediately right of tile ``i``,
    measured as mean squared difference of the touching border columns plus a
    gradient-continuity term (a standard, transparent jigsaw metric).
    """
    n, bh, bw = tiles.shape
    t = tiles.astype(np.float64)
    # borders
    r_last, r_prev = t[:, :, -1], t[:, :, -2]
    l_first, l_next = t[:, :, 0], t[:, :, 1]
    d_last, d_prev = t[:, -1, :], t[:, -2, :]
    u_first, u_next = t[:, 0, :], t[:, 1, :]

    # predicted continuation across the seam (gradient extrapolation)
    pred_r = 2 * r_last - r_prev            # (n, bh)
    pred_l = 2 * l_first - l_next
    pred_d = 2 * d_last - d_prev            # (n, bw)
    pred_u = 2 * u_first - u_next

    right = ((pred_r[:, None, :] - l_first[None, :, :]) ** 2).mean(axis=2)
    right += ((pred_l[None, :, :] - r_last[:, None, :]) ** 2).mean(axis=2)
    down = ((pred_d[:, None, :] - u_first[None, :, :]) ** 2).mean(axis=2)
    down += ((pred_u[None, :, :] - d_last[:, None, :]) ** 2).mean(axis=2)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    return right, down


def attack_boundary_reassembly(
    scrambled: np.ndarray, rows: int, cols: int, true_perm: Optional[np.ndarray] = None,
    original: Optional[np.ndarray] = None,
) -> AttackResult:
    """Key-free reassembly by boundary compatibility (greedy jigsaw solver).

    Assumptions: no key, no known plaintext; only the scrambled frame and the
    public grid.  This is a classical greedy best-buddy placement with bounded
    time O(N^2 * border).  It exploits the fact that block permutation leaves
    every block's *content* untouched, so natural image continuity across seams
    is still informative.
    """
    t0 = time.perf_counter()
    tiles = split_tiles(scrambled, rows, cols)
    n = rows * cols
    right, down = _edge_costs(tiles)

    # normalised costs for the greedy frontier
    def _best_buddy(cost: np.ndarray) -> np.ndarray:
        return np.argmin(cost, axis=1)

    board: Dict[Tuple[int, int], int] = {}
    pos_of: Dict[int, Tuple[int, int]] = {}
    placed = np.zeros(n, dtype=bool)

    start = int(np.argmin(right.min(axis=1) + down.min(axis=1)))
    board[(0, 0)] = start
    pos_of[start] = (0, 0)
    placed[start] = True

    def _bounds_ok(r: int, c: int) -> bool:
        rs = [p[0] for p in board] + [r]
        cs = [p[1] for p in board] + [c]
        return (max(rs) - min(rs) < rows) and (max(cs) - min(cs) < cols)

    while placed.sum() < n:
        best = (np.inf, None, None)  # cost, position, tile
        for (r, c), tid in list(board.items()):
            for dr, dc, cost_m, transpose in (
                (0, 1, right, False), (0, -1, right, True),
                (1, 0, down, False), (-1, 0, down, True),
            ):
                pos = (r + dr, c + dc)
                if pos in board or not _bounds_ok(*pos):
                    continue
                row = cost_m[:, tid] if transpose else cost_m[tid, :]
                cand = np.where(placed, np.inf, row)
                j = int(np.argmin(cand))
                if cand[j] < best[0]:
                    best = (float(cand[j]), pos, j)
        if best[1] is None:
            break
        _, pos, j = best
        board[pos] = j
        pos_of[j] = pos
        placed[j] = True

    rs = [p[0] for p in board]
    cs = [p[1] for p in board]
    r0, c0 = min(rs), min(cs)
    grid = np.full((rows, cols), -1, dtype=np.int64)
    for (r, c), tid in board.items():
        rr, cc = r - r0, c - c0
        if 0 <= rr < rows and 0 <= cc < cols:
            grid[rr, cc] = tid

    # fill any hole with an unused tile so the picture can be rendered
    unused = [i for i in range(n) if i not in grid.ravel().tolist()]
    holes = np.argwhere(grid < 0)
    for (rr, cc), tid in zip(holes, unused):
        grid[rr, cc] = tid

    recon = join_tiles(tiles[grid.ravel()], rows, cols)

    metrics: Dict[str, float] = {"n_blocks": float(n)}
    success = False
    if true_perm is not None:
        # grid[p] = index of scrambled tile placed at position p.
        # ground truth: scrambled tile i came from original tile true_perm[i],
        # so the correct arrangement places scrambled tile i at position
        # true_perm[i].
        target = invert_permutation(np.asarray(true_perm))
        direct = float((grid.ravel() == target).mean())
        metrics["direct_accuracy"] = direct
        metrics["neighbour_accuracy"] = _neighbour_accuracy(grid, np.asarray(true_perm), rows, cols)
        success = direct > 0.5
    if original is not None:
        h = min(original.shape[0], recon.shape[0])
        w = min(original.shape[1], recon.shape[1])
        mse = float(((original[:h, :w].astype(np.float64) - recon[:h, :w].astype(np.float64)) ** 2).mean())
        metrics["reconstruction_psnr_db"] = 99.0 if mse <= 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))
    return AttackResult(
        name="boundary_compatibility_reassembly",
        assumptions="ciphertext-only; public grid; natural image continuity across seams",
        success=success,
        metrics=metrics,
        recovered_permutation=grid.ravel(),
        reconstructed=recon,
        seconds=time.perf_counter() - t0,
        notes="Greedy best-buddy jigsaw solver; no learned model is used.",
    )


def _neighbour_accuracy(grid: np.ndarray, true_perm: np.ndarray, rows: int, cols: int) -> float:
    """Fraction of adjacent pairs in the reassembly that were adjacent originally.

    ``true_perm[t]`` is the original row-major index of the content carried by
    scrambled tile ``t``.  A horizontal pair is correct when the two original
    indices differ by one *inside the same original row*; a vertical pair when
    they differ by ``cols``.
    """
    good = 0
    total = 0
    for r in range(rows):
        for c in range(cols):
            tid = int(grid[r, c])
            if tid < 0:
                continue
            o = int(true_perm[tid])
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr >= rows or cc >= cols:
                    continue
                tid2 = int(grid[rr, cc])
                if tid2 < 0:
                    continue
                o2 = int(true_perm[tid2])
                total += 1
                if dc == 1:
                    if o2 == o + 1 and (o // cols) == (o2 // cols):
                        good += 1
                else:
                    if o2 == o + cols:
                        good += 1
    return float(good / total) if total else 0.0


# --------------------------------------------------- A4: multi-frame analysis
def attack_multi_frame_reuse(
    frames: Sequence[np.ndarray], scrambler: BlockScrambler, rows: int, cols: int
) -> AttackResult:
    """Quantify the extra leverage of many frames under a reused permutation.

    Method: average the temporal variance of every scrambled tile.  Under a
    static permutation the variance map is a permuted copy of the original
    variance map, which is a strong fingerprint; we report how well tiles can be
    matched by their temporal variance alone (no known plaintext, but the
    attacker is assumed to know the *statistics* of the scene class).
    """
    t0 = time.perf_counter()
    scr = [scrambler.scramble(f, i) for i, f in enumerate(frames)]
    fitted = [scrambler._fit(f) for f in frames]
    a = np.stack([split_tiles(f, rows, cols).reshape(rows * cols, -1) for f in fitted])
    b = np.stack([split_tiles(f, rows, cols).reshape(rows * cols, -1) for f in scr])
    var_o = a.astype(np.float64).var(axis=0).mean(axis=1)
    var_s = b.astype(np.float64).var(axis=0).mean(axis=1)
    n = rows * cols
    cost = np.abs(var_s[:, None] - var_o[None, :])
    est = np.full(n, -1, dtype=np.int64)
    taken = np.zeros(n, dtype=bool)
    for i in np.argsort(cost.min(axis=1)):
        for j in np.argsort(cost[i]):
            if not taken[j]:
                est[i] = j
                taken[j] = True
                break
    true = scrambler.permutation(0)
    acc = permutation_accuracy(est, true)
    return AttackResult(
        name="multi_frame_variance_fingerprint",
        assumptions="permutation reused across frames; attacker sees many scrambled frames",
        success=acc > 0.5,
        metrics={"permutation_accuracy": acc, "n_frames": float(len(frames)), "n_blocks": float(n)},
        recovered_permutation=est,
        seconds=time.perf_counter() - t0,
        notes="Shows why a per-frame permutation is required; still not a "
              "confidentiality proof for the per-frame variant.",
    )


# ------------------------------------------------------- A5: LFSR brute force
def attack_lfsr_bruteforce(
    original: np.ndarray,
    scrambled: np.ndarray,
    cfg: ScramblerConfig,
    max_states: Optional[int] = None,
) -> AttackResult:
    """Exhaustive seed search over *our own demo* key space.

    The size of this space is a parameter of our reconstruction
    (``LFSRConfig.width``), **not** a documented property of the 2021 paper.  A
    16-bit register gives 65535 candidate seeds, which is trivially searchable.
    """
    t0 = time.perf_counter()
    rows, cols = cfg.grid_rows, cfg.grid_cols
    n = rows * cols
    space = (1 << cfg.lfsr.width) - 1
    limit = min(space, max_states or space)
    ref = split_tiles(original, rows, cols).reshape(n, -1).astype(np.int64)
    got = split_tiles(scrambled, rows, cols).reshape(n, -1).astype(np.int64)
    found: Optional[int] = None
    tried = 0
    for seed in range(1, limit + 1):
        tried += 1
        lc = LFSRConfig(cfg.lfsr.width, cfg.lfsr.taps, seed, cfg.lfsr.form,
                        cfg.lfsr.zero_state_policy)
        perm = lfsr_permutation(n, lc, cfg.variant)
        if np.array_equal(got, ref[perm]):
            found = seed
            break
    dt = time.perf_counter() - t0
    return AttackResult(
        name="lfsr_seed_bruteforce",
        assumptions="known (original, scrambled) pair; seed space = our reconstruction parameter",
        success=found is not None,
        metrics={
            "seed_found": float(found) if found is not None else -1.0,
            "seeds_tried": float(tried),
            "key_space": float(space),
            "seeds_per_second": float(tried / dt) if dt > 0 else 0.0,
        },
        seconds=dt,
        notes=f"key space 2^{cfg.lfsr.width}-1 = {space}; searched {tried}",
    )


# ------------------------------------- A6: known pair against substitution
def _sorted_histograms(tiles: np.ndarray) -> np.ndarray:
    """Per-tile value histogram, sorted.

    A substitution relabels values, so it permutes the *bins* of a histogram
    and leaves the sorted counts untouched.  The sorted histogram is therefore
    an invariant of any per-block bijection on values, which is what lets the
    block permutation be recovered even though tile contents no longer match.
    """
    n = tiles.shape[0]
    out = np.zeros((n, 256), dtype=np.int64)
    flat = tiles.reshape(n, -1).astype(np.int64)
    for i in range(n):
        out[i] = np.bincount(flat[i], minlength=256)
    return np.sort(out, axis=1)


def attack_substitution_known_pair(
    original: np.ndarray, ciphered: np.ndarray, rows: int, cols: int,
    true_perm: Optional[np.ndarray] = None,
) -> AttackResult:
    """Recover the block permutation **and** the substitution tables from one pair.

    Assumptions: the attacker holds one (original, ciphered) pair produced by
    substitution followed by block permutation, and knows the grid.  No key.

    The plain tile matching of :func:`attack_known_pair` fails here, because a
    substituted tile does not resemble its original.  This attack matches tiles
    on the sorted histogram instead - a statistic the substitution cannot
    change - and then reads the table off pointwise, since substitution acts on
    each pixel independently.

    What the result means depends on how often the tables change.  With one
    table per session the recovered tables decrypt every later frame; with a
    table per frame or per block they decrypt only the frame they came from.
    The runner measures that difference directly rather than assuming it.
    """
    t0 = time.perf_counter()
    n = rows * cols
    a = split_tiles(original, rows, cols)
    b = split_tiles(ciphered, rows, cols)
    sig_a = _sorted_histograms(a).astype(np.float64)
    sig_b = _sorted_histograms(b).astype(np.float64)

    cost = np.abs(sig_b[:, None, :] - sig_a[None, :, :]).sum(axis=2)
    est = np.full(n, -1, dtype=np.int64)
    taken = np.zeros(n, dtype=bool)
    for i in np.argsort(cost.min(axis=1)):
        for j in np.argsort(cost[i]):
            if not taken[j]:
                est[i] = j
                taken[j] = True
                break

    # Read the tables off the matched pairs.  Substitution is pointwise, so
    # cipher tile i and original tile est[i] give one (value -> value) pair per
    # pixel.  -1 marks an entry this frame does not pin down.
    tables = np.full((n, 256), -1, dtype=np.int64)
    per_block_conflicts = 0
    for i in range(n):
        src = a[est[i]].ravel().astype(np.int64)
        dst = b[i].ravel().astype(np.int64)
        for v, w in zip(src, dst):
            if tables[est[i], v] < 0:
                tables[est[i], v] = w
            elif tables[est[i], v] != w:
                per_block_conflicts += 1

    # Merging every block into one table is both an improvement and a test.
    # If one table is in use, merging pins down far more of the alphabet; if
    # the tables differ per block, merging contradicts itself - which is how
    # the attacker learns which mode is in use.
    merged = np.full(256, -1, dtype=np.int64)
    merge_conflicts = 0
    for row in tables:
        for v in np.flatnonzero(row >= 0):
            if merged[v] < 0:
                merged[v] = row[v]
            elif merged[v] != row[v]:
                merge_conflicts += 1
    is_global = merge_conflicts == 0

    metrics: Dict[str, float] = {
        "n_blocks": float(n),
        "assignment_complete": float(bool((est >= 0).all())),
        "per_block_alphabet_covered": float((tables >= 0).sum(axis=1).mean() / 256.0),
        "merged_alphabet_covered": float((merged >= 0).sum() / 256.0),
        "merge_conflicts": float(merge_conflicts),
        "substitution_is_global": float(is_global),
        "per_block_conflicts": float(per_block_conflicts),
    }
    if is_global:                      # one table: every block gets the merged one
        tables = np.tile(merged, (n, 1))
    success = False
    if true_perm is not None:
        acc = permutation_accuracy(est, np.asarray(true_perm))
        metrics["permutation_accuracy"] = acc
        success = acc > 0.5
    return AttackResult(
        name="substitution_known_pair",
        assumptions=("one (original, ciphered) pair; public grid; no key. "
                     "Matching uses the sorted per-tile histogram, which a "
                     "value substitution leaves invariant"),
        success=success,
        metrics=metrics,
        recovered_permutation=est,
        recovered_tables=tables,
        seconds=time.perf_counter() - t0,
        notes=("recovers the transform of THIS frame; whether that decrypts "
               "later frames depends on how often the tables change"),
    )


def apply_recovered(ciphered: np.ndarray, est: np.ndarray, tables: np.ndarray,
                    rows: int, cols: int, fill: int = 128) -> np.ndarray:
    """Decrypt a frame with a permutation and tables recovered by an attack.

    Entries the attack could not pin down are filled with ``fill``, so the
    resulting quality is an honest measure of what the attacker actually has.
    """
    n = rows * cols
    b = split_tiles(ciphered, rows, cols)
    out = np.full_like(b, fill)
    for i in range(n):
        j = int(est[i])
        inv = np.full(256, -1, dtype=np.int64)
        row = tables[j]
        known = np.flatnonzero(row >= 0)
        inv[row[known]] = known
        got = inv[b[i].astype(np.int64)]
        tile = np.where(got >= 0, got, fill).astype(np.uint8)
        out[j] = tile
    return join_tiles(out, rows, cols)


__all__ = [
    "AttackResult", "split_tiles", "join_tiles", "permutation_accuracy",
    "attack_chosen_plaintext", "attack_known_pair", "attack_boundary_reassembly",
    "attack_multi_frame_reuse", "attack_lfsr_bruteforce",
    "attack_substitution_known_pair", "apply_recovered",
]
