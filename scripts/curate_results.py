"""Assemble the tracked evidence directory from one run.

``runs/`` is working output and is not committed: it is 200+ MB of per-frame
rows and per-cell job state.  ``results/`` is the curated subset that backs
every published number and **is** tracked, so a reader can check a table
without re-running anything.

What goes in: the analysis tables, the raw tables the plan requires, the figure
catalogue with its data and parameters, and the run manifest.  What stays out:
per-frame JSONL, per-cell job directories, and anything regenerable from the
tables that are here.

Usage::

    python scripts/curate_results.py runs/main results
"""
from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

#: Tables the plan names as required raw data, plus what the report reads.
TABLES = (
    "run_manifest.json", "dataset_manifest.csv", "dataset_manifest.json",
    "channel_traces.jsonl", "frames.csv", "units.csv", "codewords.csv",
    "joint_loss.csv", "placement_geometry.csv", "timings.csv", "scaling.csv",
    "budgets.csv", "budgets.json", "failures.csv", "failures_summary.csv",
    "paired_effects.csv", "summary.csv", "operability.csv", "analysis.json",
    "observations.csv", "jobs.csv", "matrix.json", "config.yaml",
    "e01_rate_quality.csv", "e01.json", "sweeps.csv", "e03.json",
    "interaction.csv", "e04.json", "e05.json", "ablations.csv", "e06.json",
    "dynamics.csv", "recovery.csv", "e07.json", "e09.json",
    "cvbs.json", "cvbs_transfer.csv", "attacks.json", "attacks.csv",
    "similarity.csv", "protocol_checks.json", "programme_run.json",
)

#: frames.csv is 22 MB; the per-scene aggregate is what the tables actually
#: read, so the full per-frame table is summarised instead of copied whole.
LARGE = {"frames.csv": 4_000_000, "observations.csv": 4_000_000,
         "units.csv": 4_000_000}


def _copy(src_dir: str, dst_dir: str, name: str) -> str:
    src = os.path.join(src_dir, name)
    if not os.path.exists(src):
        return ""
    dst = os.path.join(dst_dir, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    size = os.path.getsize(src)
    limit = LARGE.get(name)
    if limit and size > limit:
        return _summarise(src, dst, limit)
    shutil.copy2(src, dst)
    return f"{name} ({size / 1e6:.2f} MB)"


def _summarise(src: str, dst: str, limit: int) -> str:
    """Keep the header and a deterministic sample, and say so in a sidecar."""
    import csv

    with open(src, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    step = max(1, len(rows) * 400 // max(limit // 1000, 1) // 400)
    step = max(1, len(rows) // 20000)
    kept = rows[::step]
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(kept)
    with open(dst + ".README", "w", encoding="utf-8") as fh:
        fh.write(
            f"Це ПРОРІДЖЕНА копія: {len(kept)} з {len(rows)} рядків, кожен "
            f"{step}-й, детермінований крок.\n"
            f"Повна таблиця лишається у прогоні (`runs/`) і відтворюється "
            f"командами з README.\n"
            f"Усі опубліковані числа рахуються з ПОВНОЇ таблиці, не з цієї.\n")
    return f"{os.path.basename(dst)} (проріджено {len(kept)}/{len(rows)})"


def main(argv: list) -> int:
    run = argv[1] if len(argv) > 1 else "runs/main"
    out_root = argv[2] if len(argv) > 2 else "results"
    manifest_path = os.path.join(run, "run_manifest.json")
    run_id = "unknown"
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            run_id = json.load(fh).get("run_id", "unknown")
    dst = os.path.join(out_root, "main")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    copied = [c for c in (_copy(run, dst, n) for n in TABLES) if c]

    fig_src = os.path.join(run, "figures")
    fig_dst = os.path.join(dst, "figures")
    n_fig = 0
    if os.path.isdir(fig_src):
        os.makedirs(fig_dst, exist_ok=True)
        for name in sorted(os.listdir(fig_src)):
            if name.rsplit(".", 1)[-1] in ("png", "svg", "pdf", "csv", "json"):
                shutil.copy2(os.path.join(fig_src, name),
                             os.path.join(fig_dst, name))
                n_fig += 1

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(dst) for f in fs)
    summary = {"run_id": run_id, "source": run, "destination": dst,
               "tables": copied, "figure_files": n_fig,
               "total_mb": round(total / 1e6, 2)}
    with open(os.path.join(dst, "CURATION.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
