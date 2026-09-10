"""Run one matrix plan to completion, resuming whatever is already there.

Usage::

    python scripts/run_matrix.py configs/research_main.yaml runs/main [workers]

The script is deliberately thin: everything it does is available through
``avsec matrix`` as well.  It exists so a long run can be started detached and
resumed after an interruption without retyping the plan.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from avsec.matrix import MatrixRunner, MatrixSpec  # noqa: E402
from avsec.utils import write_json  # noqa: E402


def main(argv: list) -> int:
    plan = argv[1] if len(argv) > 1 else "configs/research_main.yaml"
    out = argv[2] if len(argv) > 2 else "runs/main"
    cfg, spec = MatrixSpec.from_plan(plan)
    workers = int(argv[3]) if len(argv) > 3 else spec.workers
    runner = MatrixRunner(cfg, spec, out, workers=workers)
    est = runner.estimate().to_dict()
    print(f"plan={plan} out={out} workers={workers}", flush=True)
    print(f"pending jobs={est['n_jobs']} frames={est['n_frames_total']} "
          f"estimate={est['estimated_wall_human']}", flush=True)

    last = [time.time()]

    def progress(stage: str, frac: float, info: dict) -> None:
        now = time.time()
        if now - last[0] > 20.0 or frac >= 1.0:
            last[0] = now
            print(f"  {frac*100:5.1f}%  {stage}  [{info.get('status')}]", flush=True)

    t0 = time.time()
    res = runner.run(progress=progress)
    res["wall_s_measured"] = round(time.time() - t0, 1)
    res["estimate_before_run"] = est
    write_json(os.path.join(out, "matrix.json"), res)
    print(f"done: {res['n_jobs_this_call']} jobs this call, "
          f"{res['n_frames']} frame rows total, "
          f"{res['wall_s_measured']} s wall, complete="
          f"{res['completeness']['complete']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
