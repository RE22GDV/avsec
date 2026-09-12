"""Run the research programme E01-E15 into one run directory.

Usage::

    python scripts/run_program.py configs/research_main.yaml runs/main [workers] [E01,E03,...]

Each experiment writes its own raw table into the run directory, so the
figure engine and the report read one place.  Experiments are independent:
if one fails, the others still run and the failure is recorded rather than
aborting the programme.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from avsec.config import load_config  # noqa: E402
from avsec.utils import write_json  # noqa: E402


def _write_budgets(cfg, out: str) -> None:
    """Capacity, latency and memory tables for this configuration."""
    from avsec.experiments import run_budget
    from avsec.utils import write_csv

    o = run_budget(cfg)
    write_json(os.path.join(out, "budgets.json"), o)
    rows = []
    for m, d in o["methods"].items():
        row = {"method": m, "unit_plain_bytes": d["unit_plain_bytes"],
               "unit_wire_bytes": d["unit_wire_bytes"],
               "unit_symbols": d["unit_symbols"],
               "units_per_raster": d["units_per_raster"],
               "units_placed": d["units_per_raster_after_placement"],
               "unused_symbols": d["unused_symbols"],
               "gross_bitrate_bps": d["gross_bitrate_bps"],
               "payload_bitrate_bps": d["payload_bitrate_bps"],
               "payload_efficiency": d["payload_efficiency"],
               "latency_mean_s": d["latency"]["latency_s"]["mean"],
               "latency_additive_s":
                   d["latency"]["additive_upper_bound"]["virtual_total_s"],
               "deadline_misses": d["latency"]["deadline_misses"],
               "logical_buffer_kb": d["memory"]["logical_total_kb"],
               "process_peak_mb": d["memory"]["process_peak_mb"],
               "model_arrays_mb": d["memory"]["model_total_mb"],
               "confidentiality": d["security"].get("confidentiality"),
               "authentication": d["security"].get("authentication")}
        row.update({f"overhead_{k}": v
                    for k, v in d["overhead_breakdown_per_unit"].items()})
        row.update({f"raster_{k}": v for k, v in
                    d["waterfall"]["raster_overhead"]["fractions"].items()})
        rows.append(row)
    write_csv(os.path.join(out, "budgets.csv"), rows)


def main(argv: list) -> int:
    plan = argv[1] if len(argv) > 1 else "configs/research_main.yaml"
    out = argv[2] if len(argv) > 2 else "runs/main"
    workers = int(argv[3]) if len(argv) > 3 else 12
    only = set((argv[4] if len(argv) > 4 else "").split(",")) - {""}

    cfg = load_config(plan)
    from avsec import research
    from avsec.experiments import run_attacks, run_protocol_checks

    def _e01(): return research.run_e01(cfg, out, workers)
    def _e03(): return research.run_e03(cfg, out, workers, n_scenes=6,
                                        repetitions=3, max_frames=6)
    def _e04(): return research.run_e04(cfg, out, workers, repetitions=3,
                                        max_frames=4, n_clips=6)
    def _e05(): return research.run_e05(cfg, out)
    def _e06(): return research.run_e06(cfg, out, workers, repetitions=3,
                                        max_frames=6, n_clips=6)
    def _e07(): return research.run_e07(cfg, out, workers, n_frames=40,
                                        repetitions=3, n_clips=4)
    def _e09(): return research.run_e09(cfg, out, workers=1, warmup=1, repeats=5)
    def _e10(): return research.run_e10(cfg, out)
    def _e13(): return research.run_e13(cfg, out, workers, repetitions=3,
                                        max_frames=8)
    def _e15(): return research.run_e15(cfg, out, workers, repetitions=3,
                                        max_frames=6, n_clips=6)

    def _e08():
        att = run_attacks(cfg, out)
        write_json(os.path.join(out, "attacks.json"), att)
        checks = run_protocol_checks(cfg)
        n_pass = sum(1 for c in checks if c.get("passed"))
        write_json(os.path.join(out, "protocol_checks.json"),
                   {"checks": checks, "n_checks": len(checks), "n_passed": n_pass,
                    "all_passed": n_pass == len(checks)})
        return {"attacks": len(att.get("results", [])), "checks": len(checks),
                "checks_passed": n_pass}

    steps = [("E01", _e01), ("E03", _e03), ("E04", _e04), ("E05", _e05),
             ("E06", _e06), ("E07", _e07), ("E08", _e08), ("E09", _e09),
             ("E10", _e10), ("E13", _e13), ("E15", _e15)]
    status = {}
    for name, fn in steps:
        if only and name not in only:
            continue
        t0 = time.time()
        print(f"=== {name} starting", flush=True)
        try:
            res = fn()
            status[name] = {"status": "done", "wall_s": round(time.time() - t0, 1),
                            "summary": {k: v for k, v in (res or {}).items()
                                        if isinstance(v, (int, float, str, bool))}}
            print(f"=== {name} done in {status[name]['wall_s']} s", flush=True)
        except Exception as exc:
            status[name] = {"status": "failed", "wall_s": round(time.time() - t0, 1),
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-2000:]}
            print(f"=== {name} FAILED: {exc}", flush=True)
    # the budget tables belong to the run as much as any experiment does
    try:
        _write_budgets(cfg, out)
        status["budgets"] = {"status": "done"}
    except Exception as exc:
        status["budgets"] = {"status": "failed", "error": str(exc)}
    write_json(os.path.join(out, "programme_run.json"), status)
    print(json.dumps({k: v["status"] for k, v in status.items()}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
