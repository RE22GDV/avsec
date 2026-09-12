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
    "jobs.csv", "matrix.json", "config.yaml",
    "e01_rate_quality.csv", "e01.json", "sweeps.csv", "e03.json",
    "interaction.csv", "e04.json", "e05.json", "ablations.csv", "e06.json",
    "dynamics.csv", "recovery.csv", "e07.json", "e09.json",
    "cvbs.json", "cvbs_transfer.csv", "attacks.json", "attacks.csv",
    "similarity.csv", "protocol_checks.json", "programme_run.json",
    # R03/R04: the damage model and the decoder that checks it
    "rs_cross_check.csv",
    # R09: per-scene ablation data and the paired contrasts computed from it
    "chain.csv", "chain_scenes.csv", "e13.json",
    "ablation_scenes.csv", "ablation_contrasts.csv",
    # R10: the operating map, measured nodes and their per-scene rows
    "operating_map.csv", "operating_map_scenes.csv", "e15.json",
    # R11: the recomputation of every published claim
    "verification.json", "analysis_plan.yaml",
)

#: Per-frame tables are tens of megabytes, and most of that is columns no
#: published number is computed from: eighty-one columns of per-stage counters
#: for twelve that the analysis reads.
#:
#: Earlier this was solved by keeping every n-th ROW.  That made
#: ``avsec verify --input results/main`` recompute from a different dataset
#: than the one the report published, which is exactly the check the folder
#: exists to support.  So now every row is kept and the unused COLUMNS are
#: dropped: the same size, and the verification is real.
PROJECT = {
    "frames.csv": (
        "method", "scene", "clip", "source_id", "repetition", "frame_id",
        "channel", "channel_plan", "channel_axis", "channel_value", "status",
        "detail", "job_id", "trace_id", "provenance", "profile_id",
        "availability", "psnr_displayed", "displayed_source", "gap_policy",
        "psnr_full", "ssim_full", "psnr_verified", "ssim_verified", "coverage",
        "mse_full", "bit_exact", "stale_fraction", "estimated_fraction",
        "max_age_frames", "units_sent", "units_verified", "units_rejected",
        "symbol_errors_pre_fec", "corrected_symbols", "payload_bytes",
        "wire_bytes", "rasters", "sync_found", "resynchronised",
        "x_desc_total", "x_desc_touched", "x_desc_fec_failed",
        "x_desc_unusable", "x_desc_partial", "x_bands_total",
        "x_band_complete", "x_band_partial_segments",
        "x_band_reduced_descriptions", "x_band_no_usable_description",
    ),
    "sweeps_frames.csv": (
        "method", "scene", "clip", "repetition", "frame_id", "channel",
        "channel_axis", "channel_value", "axis", "cofactors", "status",
        "availability", "psnr_displayed", "psnr_full", "ssim_full", "coverage",
        "symbol_errors_pre_fec",
    ),
}

#: Tables still kept whole, because they are already small.
LARGE: dict = {}


def _copy(src_dir: str, dst_dir: str, name: str) -> str:
    src = os.path.join(src_dir, name)
    if not os.path.exists(src):
        return ""
    dst = os.path.join(dst_dir, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    size = os.path.getsize(src)
    cols = PROJECT.get(name)
    if cols:
        return _project(src, dst, cols)
    limit = LARGE.get(name)
    if limit:
        return _summarise(src, dst, limit)
    shutil.copy2(src, dst)
    return f"{name} ({size / 1e6:.2f} MB)"


def _project(src: str, dst: str, keep) -> str:
    """Every row, only the columns any published number is computed from."""
    import csv

    with open(src, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = [c for c in (reader.fieldnames or []) if c in keep]
        rows = list(reader)
    dropped = [c for c in (rows[0] if rows else {}) if c not in fields]
    import gzip

    # Gzip, not thinning: every row survives, and the reader opens either form.
    with gzip.open(dst + ".gz", "wt", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    name = os.path.basename(dst)
    with open(dst + ".README", "w", encoding="utf-8") as fh:
        fh.write(
            f"{name}.gz: УСІ {len(rows)} рядків збережено (gzip); з "
            f"{len(fields) + len(dropped)} стовпців лишено {len(fields)}.\n\n"
            "Прорідження рядків тут НЕ застосовується: команда\n"
            "  avsec verify --input <ця тека>\n"
            "перераховує кожне опубліковане число саме з цього файла, і робити "
            "це на іншій вибірці означало б перевіряти інший набір даних.\n\n"
            "Прибрані стовпці - це посценні лічильники етапів, яких не читає "
            "жоден опублікований агрегат:\n  "
            + ", ".join(sorted(dropped)) + "\n\n"
            "Повна таблиця з усіма стовпцями відтворюється прогоном з "
            "кореневого README: у режимі key_mode: lab результат детермінований.\n")
    return (f"{name}.gz ({os.path.getsize(dst + '.gz') / 1e6:.2f} MB, "
            f"усі {len(rows)} рядків, "
            f"{len(fields)} з {len(fields) + len(dropped)} стовпців)")


def _summarise(src: str, dst: str, limit: int) -> str:
    """Keep the header and a deterministic sample, and say so in a sidecar."""
    import csv

    with open(src, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        shutil.copy2(src, dst)
        return f"{os.path.basename(dst)} (порожня)"
    step = max(1, len(rows) // max(int(limit), 1))
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
    # the destination is named after the run, so several runs can be curated
    # side by side (the synthetic one and the real-imagery one)
    name = argv[3] if len(argv) > 3 else os.path.basename(run.rstrip("/\\"))
    manifest_path = os.path.join(run, "run_manifest.json")
    run_id = "unknown"
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            run_id = json.load(fh).get("run_id", "unknown")
    dst = os.path.join(out_root, name)
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
    summary = {"run_id": run_id, "name": name, "source": run, "destination": dst,
               "tables": copied, "figure_files": n_fig,
               "total_mb": round(total / 1e6, 2)}
    with open(os.path.join(dst, "CURATION.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
