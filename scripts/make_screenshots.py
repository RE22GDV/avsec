"""Capture the documentation screenshots from the real interface.

Starts the web interface on a spare port, drives a headless Chromium through
the deep links the interface exposes (``?tab=...&auto=1``), and writes the PNGs
under ``docs/img``.  Nothing is mocked: every screenshot is the interface
rendering the run directory it is pointed at.

Usage::

    python scripts/make_screenshots.py [run_dir] [--port 8799]
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT = os.path.join(ROOT, "docs", "img")

#: name -> (query string, window size, how long to let the page work)
SHOTS = {
    # the demo runs on a REAL drone frame, so the screenshot shows the
    # pipeline on the same material the published real-imagery results use
    "ui_demo": ("?tab=demo&auto=1&sync=1&pattern=uav:dune_ridge",
                (1400, 1620), 180000),
    "ui_budget": ("?tab=budget&auto=1&sync=1", (1400, 1300), 120000),
    "ui_attacks": ("?tab=scramble&auto=1&sync=1", (1400, 1350), 120000),
    # the three schemes with their own benches, and the checks; each on the
    # same real drone frame the other shots use, so the pictures compare
    "ui_substitution": ("?tab=substitution&auto=1&sync=1&pattern=uav:village",
                        (1400, 2500), 180000),
    "ui_sbox": ("?tab=sbox&auto=1&sync=1", (1400, 2450), 180000),
    "ui_luma": ("?tab=luma&auto=1&sync=1&pattern=uav:village",
                (1400, 2900), 240000),
    "ui_checks": ("?tab=checks&auto=1&sync=1", (1400, 1150), 180000),
    "ui_explorer": ("?tab=explorer", (1400, 1750), 120000),
}

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if os.path.exists(c):
            return c
    found = shutil.which("google-chrome") or shutil.which("chromium") \
        or shutil.which("chrome")
    if found:
        return found
    raise SystemExit(
        "no Chromium-based browser found - install Chrome or Edge, or take the "
        "screenshots by hand from `avsec ui`.")


def free_port(preferred: int) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])


def main(argv: list) -> int:
    run_dir = next((a for a in argv if not a.startswith("-")), "runs/main")
    port = 8799
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    port = free_port(port)
    os.makedirs(OUT, exist_ok=True)

    from avsec.ui.server import serve

    t = threading.Thread(target=serve,
                         kwargs=dict(host="127.0.0.1", port=port,
                                     open_browser=False, runs_dir="runs"),
                         daemon=True)
    t.start()
    time.sleep(1.5)

    chrome = find_chrome()
    print(f"browser : {chrome}")
    print(f"server  : http://127.0.0.1:{port}/")
    print(f"run dir : {run_dir}\n")

    for name, (query, size, budget) in SHOTS.items():
        url = f"http://127.0.0.1:{port}/{query}"
        if name == "ui_explorer":
            url += f"&run={run_dir}"
        path = os.path.join(OUT, f"{name}.png")
        cmd = [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
               "--force-device-scale-factor=1",
               f"--virtual-time-budget={budget}",
               f"--window-size={size[0]},{size[1]}",
               f"--screenshot={path}", url]
        t0 = time.time()
        subprocess.run(cmd, capture_output=True, timeout=budget / 1000 + 90)
        ok = os.path.exists(path) and os.path.getsize(path) > 8000
        print(f"{'ok ' if ok else 'FAIL'} {name:16s} "
              f"{os.path.getsize(path) / 1e3 if ok else 0:7.1f} kB  "
              f"{time.time() - t0:5.1f}s  {query}")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
