"""Regenerate the documentation from one run directory.

Usage::

    python scripts/publish.py runs/main
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from avsec.publish import publish  # noqa: E402

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/main"
    # an optional second run to compare against (the real-imagery transfer check)
    compare = sys.argv[2] if len(sys.argv) > 2 else (
        "runs/drone" if os.path.isdir("runs/drone") and run != "runs/drone" else None)
    print(json.dumps(publish(run, compare_dir=compare), indent=2,
                     ensure_ascii=False))
