"""The web interface must not drift from the library or from its own page.

Two kinds of rot are caught here.  The first is structural: the page calls an
action by name, and nothing but a running browser would notice if that name
stopped existing.  The second is substantive: a tab is supposed to show a
particular fact - that the computed table is the AES table, that the luminance
plane of a ``B2l`` ciphertext holds one value - and a screenshot in the
documentation would keep looking plausible long after the fact stopped holding.
"""
from __future__ import annotations

import io
import os
import re

import numpy as np
import pytest

from avsec.ui import server as UI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "src", "avsec", "ui", "static", "index.html")


def _page() -> str:
    return io.open(PAGE, encoding="utf-8").read()


def _noop(*_a, **_k) -> None:
    return None


def test_every_action_the_page_calls_exists_on_the_server():
    """A renamed action would otherwise fail only in a browser, at runtime."""
    page = _page()
    called = set(re.findall(r'\bjob\("([a-z_]+)"', page))
    called |= set(re.findall(r'\bsync\("([a-z_]+)"', page))
    known = set(UI.ASYNC_ACTIONS) | set(UI.SYNC_ACTIONS)
    assert called, "the page must call at least one action"
    assert called <= known, f"page calls unknown actions: {sorted(called - known)}"


def test_every_tab_has_a_runner_and_a_hint():
    """A tab with no hint leaves the reader guessing what the button does."""
    page = _page()
    tabs = re.findall(r'\["([a-z]+)",\s*"[^"]+",\s*(run\w+)\]', page)
    assert len(tabs) >= 14, f"expected the full tab set, found {len(tabs)}"
    hints = re.search(r"const TAB_HINT = \{(.*?)\n\};", page, re.S)
    assert hints, "TAB_HINT must exist"
    for tab_id, runner in tabs:
        assert f"function {runner}(" in page, f"{runner} is not defined"
        assert re.search(rf"\b{tab_id}\s*:", hints.group(1)), f"{tab_id} has no hint"


def test_the_new_tabs_are_registered_both_ways():
    """They run as jobs for a human and synchronously for the screenshot tool."""
    for name in ("substitution", "sbox", "luma", "checks"):
        assert name in UI.ASYNC_ACTIONS, f"{name} must be runnable as a job"
        assert name in UI.SYNC_ACTIONS, f"{name} must be runnable synchronously"


def test_the_sbox_tab_reproduces_the_published_aes_table():
    """The claim the tab makes in a badge, checked here rather than believed."""
    res = UI.action_sbox({"run_sweep": False}, _noop)
    assert res["verification"]["matches_published_aes_sbox"] is True
    assert res["verification"]["inverse_is_exact"] is True
    first_row = res["tables"]["algebraic"]["rows"][0]
    assert first_row[:4] == ["63", "7C", "77", "7B"]
    assert len(res["tables"]["algebraic"]["values"]) == 256


def test_the_substitution_tab_shows_every_stage_and_recovers_exactly():
    res = UI.action_substitution({"pattern": "edges", "run_attacks": False}, _noop)
    assert res["exact_recovery"] is True
    assert set(res["images"]) == {"1_original", "2_substituted",
                                  "3_substituted_permuted",
                                  "4_permutation_only_B2", "5_restored"}
    assert res["sanity"]["bijective"] and res["sanity"]["inverse_exact"]
    # the point of the panel: a permutation keeps the histogram, a table does not
    assert res["histogram"]["hist_identical_after_permutation"] is True
    assert res["histogram"]["sorted_hist_identical_after_global_substitution"] is True


def test_the_luma_tab_shows_a_ciphertext_with_one_luminance_level():
    res = UI.action_luma({"pattern": "edges", "run_attacks": False,
                          "run_tract": False}, _noop)
    assert res["exact_recovery"] is True
    assert res["statistics"]["rounded_levels_used"] == 1
    assert abs(res["statistics"]["luma_entropy_bits"]) < 1e-9
    assert res["capacity"]["enough_for_monochrome_8_bit"] is True
    assert res["capacity"]["enough_for_colour_24_bit"] is False
    assert res["images"]["3_cipher_luma"].startswith("data:image/png;base64,")


def test_the_checks_tab_reports_a_verdict_per_group():
    """Missing inputs must read as a failed group, never as a silent pass."""
    res = UI.action_checks({"input_dir": "results/_no_such_directory"}, _noop)
    assert res["n_groups"] >= 4
    for g in res["groups"]:
        assert {"name", "command", "purpose", "ok", "failures"} <= set(g)
    missing = [g for g in res["groups"] if "verify" in g["command"]][0]
    assert missing["ok"] is False and missing["failures"]
    assert res["all_passed"] is False


def test_a_stored_bench_is_read_without_running_anything():
    res = UI.action_stored_lab({"which": "luma"})
    assert res["command"] == "run.bat luma-lab"
    if not res["present"]:
        pytest.skip("results/luma has not been produced in this checkout")
    assert res["data"]["kind"] == "luma_balance"
    assert res["data"]["git_worktree"] in ("clean", "dirty")


def test_an_unknown_bench_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        UI.action_stored_lab({"which": "../etc"})


def test_the_frame_helper_gives_distinct_frames_for_a_pattern():
    """The multi-frame reuse attack is meaningless on two identical frames."""
    from avsec.config import ExperimentConfig

    cfg = ExperimentConfig()
    frames = UI._frames_for({"pattern": "edges"}, cfg, 2)
    assert len(frames) == 2
    assert frames[0].shape == (cfg.frame_height, cfg.frame_width)
    assert not np.array_equal(frames[0], frames[1])
