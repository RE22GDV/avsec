"""Command line interface.  Every command uses the same library code as the UI."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from avsec import __version__
from avsec.config import ExperimentConfig, dump_config, load_config
from avsec.utils import write_json


def _progress(stage: str, frac: float, info: Dict[str, Any]) -> None:
    bar = int(frac * 30)
    sys.stderr.write("\r[" + "#" * bar + "." * (30 - bar) + f"] {frac*100:5.1f}%  {stage[:52]:52s}")
    sys.stderr.flush()
    if frac >= 1.0:
        sys.stderr.write("\n")


def _cfg(args: argparse.Namespace) -> ExperimentConfig:
    if getattr(args, "config", None):
        return load_config(args.config)
    from avsec.config import config_from_dict

    return config_from_dict({})


def _out(args: argparse.Namespace, default: str) -> str:
    return getattr(args, "output", None) or default


def cmd_info(args: argparse.Namespace) -> int:
    from avsec.utils import environment_record

    print(json.dumps({"version": __version__, "environment": environment_record()},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_gendata(args: argparse.Namespace) -> int:
    """Write the procedural test material to disk so a run can be reproduced."""
    from avsec import evaluation as ev
    from avsec import sources as S
    from avsec.utils import ensure_dir, write_json

    out = _out(args, "data/generated")
    ensure_dir(out)
    srcs = S.default_suite(args.height, args.width, args.frames)
    srcs.append(S.chosen_plaintext_source(args.height, args.width, args.rows, args.cols))
    summary = []
    for s in srcs:
        d = ensure_dir(os.path.join(out, s.name))
        for i, f in enumerate(s.frames):
            ev.save_image(os.path.join(d, f"frame_{i:03d}.png"), f)
        summary.append(s.summary())
    write_json(os.path.join(out, "sources.json"), summary)
    print(f"wrote {len(srcs)} sources to {out}")
    for s in summary:
        print(f"  {s['name']:28s} {s['n_frames']:3d} frames  {s['provenance']}  "
              f"sha256={s['content_sha256'][:16]}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from avsec.experiments import run_demo

    cfg = _cfg(args)
    out = _out(args, "runs/demo")
    res = run_demo(cfg, out, _progress)
    for r in res["rows"]:
        print(f"{r['method']:5s} PSNR={r['psnr_full']:6.2f} dB  SSIM={r['ssim_full']:.4f}  "
              f"coverage={r['coverage']:.3f}  units={r['units_verified']}/{r['units_sent']}"
              f"  rasters={r['rasters']}")
    for k, v in res["errors"].items():
        print(f"{k:5s} FAILED: {v}")
    print(f"-> {out}")
    return 0


def cmd_scramble(args: argparse.Namespace) -> int:
    """Encrypt/recover a still image with the permutation baseline (B1)."""
    import numpy as np

    from avsec import evaluation as ev
    from avsec import sources as S
    from avsec.lfsr import BlockScrambler, correlation_similarity
    from avsec.utils import ensure_dir

    cfg = _cfg(args)
    if args.image:
        src = S.from_image_file(args.image, cfg.frame_height, cfg.frame_width)
    else:
        src = S.synthetic_still(args.pattern, cfg.frame_height, cfg.frame_width)
    img = src.frames[0]
    sc = BlockScrambler(cfg.lfsr)
    scrambled = sc.scramble(img, 0)
    restored = sc.descramble(scrambled, 0, img.shape)
    out = ensure_dir(_out(args, "runs/scramble"))
    ev.save_image(os.path.join(out, "original.png"), img)
    ev.save_image(os.path.join(out, "scrambled.png"), scrambled)
    ev.save_image(os.path.join(out, "restored.png"), restored)
    r1, s1 = correlation_similarity(sc._fit(img), scrambled)
    r2, s2 = correlation_similarity(img, restored)
    info = {
        "source": src.summary(), "scrambler": cfg.lfsr.describe(),
        "similarity_original_vs_scrambled_pct": s1,
        "similarity_original_vs_restored_pct": s2,
        "exact_recovery": bool(np.array_equal(img, restored)),
    }
    write_json(os.path.join(out, "scramble.json"), info)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def cmd_attack(args: argparse.Namespace) -> int:
    from avsec.experiments import run_attacks

    cfg = _cfg(args)
    out = _out(args, "runs/attack")
    res = run_attacks(cfg, out, _progress)
    for r in res["results"]:
        m = r["metrics"]
        extra = " ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "n_blocks")
        print(f"{r['name']:38s} success={str(r['success']):5s} {extra}")
    failed = [c for c in res["protocol_checks"] if not c.get("passed")]
    print(f"protocol checks: {len(res['protocol_checks']) - len(failed)}/"
          f"{len(res['protocol_checks'])} passed")
    for c in failed:
        print(f"  FAILED: {c['check']} -> {c['outcome']}")
    print(f"-> {out}")
    return 1 if failed else 0


def cmd_transmit(args: argparse.Namespace) -> int:
    from avsec.experiments import run_comparison

    cfg = _cfg(args)
    out = _out(args, "runs/transmit")
    res = run_comparison(cfg, out, _progress)
    for s in res["summary"]:
        print(f"{s['method']:5s} PSNR={s.get('psnr_mean', float('nan')):6.2f} dB "
              f"[{s.get('psnr_lo', float('nan')):.2f};{s.get('psnr_hi', float('nan')):.2f}] "
              f"coverage={s.get('coverage_mean', 0):.3f} "
              f"auth={s.get('authenticated')} sequences={s.get('sequences')}")
    for k, v in res["errors"].items():
        print(f"{k:5s} EXCLUDED: {v}")
    print(f"-> {out}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from avsec.experiments import run_cvbs

    cfg = _cfg(args)
    out = _out(args, "runs/cvbs")
    res = run_cvbs(cfg, out, args.preset, args.method, _progress)
    print(json.dumps({"quality": res["quality"], "status_counts": res["status_counts"],
                      "cvbs_stats": res["cvbs_stats"]}, indent=2, ensure_ascii=False))
    print(f"-> {out}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    from avsec.experiments import run_tuning
    from avsec.optimization import ParameterSpace

    cfg = _cfg(args)
    out = _out(args, "runs/tuning")
    space = ParameterSpace()
    if args.quick:
        space = ParameterSpace(
            stripe_height=(16, 24), n_descriptions=(1, 2), quality=(8, 12, 18),
            max_unit_payload=(320, 640, 896), fec_nsym=(96, 128),
            modulation=((4, 8, 2),),
            interleaver=(("block", 278, 0), ("bawp", 0, 12)))
    res = run_tuning(cfg, out, space, args.frames, _progress)
    print(f"candidates={res['n_candidates']} admissible={res['n_admissible']}")
    if res.get("selected"):
        print(f"selected: {res['selected']['key']} "
              f"validation PSNR={res['selected'].get('psnr_full', float('nan')):.2f} dB")
    else:
        print("no admissible configuration inside the shared budget")
    print(f"-> {out}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    from avsec.experiments import run_ablations

    cfg = _cfg(args)
    out = _out(args, "runs/ablations")
    res = run_ablations(cfg, out, _progress)
    for r in res["rows"]:
        if r.get("admissible"):
            print(f"{r['variant']:26s} PSNR={r['psnr_mean']:6.2f} dB "
                  f"coverage={r['coverage_mean']:.3f} rasters={r['rasters_mean']:.1f}")
        else:
            print(f"{r['variant']:26s} EXCLUDED: {r.get('reason', '')[:70]}")
    print(f"-> {out}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from avsec.experiments import run_sweeps

    cfg = _cfg(args)
    out = _out(args, "runs/sweeps")
    res = run_sweeps(cfg, out, args.presets, _progress)
    for r in res["rows"]:
        print(f"{r['channel']:9s} {r['method']:5s} PSNR={r['psnr_mean']:6.2f} dB "
              f"coverage={r['coverage_mean']:.3f}")
    print(f"-> {out}")
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    from avsec.experiments import run_budget

    cfg = _cfg(args)
    res = run_budget(cfg)
    out = _out(args, None)
    if out:
        write_json(os.path.join(out, "budget.json"), res)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Full profile run: budget, comparison, attacks, sweeps, ablations, CVBS."""
    from avsec.experiments import (
        run_ablations,
        run_attacks,
        run_comparison,
        run_cvbs,
        run_demo,
        run_sweeps,
    )
    from avsec.experiments import run_budget

    cfg = _cfg(args)
    out = _out(args, f"runs/{cfg.name}")
    write_json(os.path.join(out, "budget.json"), run_budget(cfg))
    print("== budget written")
    run_demo(cfg, out, _progress)
    print("== demo done")
    run_comparison(cfg, out, _progress)
    print("== comparison done")
    run_attacks(cfg, out, _progress)
    print("== attacks done")
    if not args.fast:
        run_sweeps(cfg, out, args.presets, _progress)
        print("== sweeps done")
        run_ablations(cfg, out, _progress)
        print("== ablations done")
    run_cvbs(cfg, out, args.cvbs_preset, "B4", _progress)
    print("== cvbs done")
    dump_config(cfg, os.path.join(out, "resolved_config.yaml"))
    print(f"-> {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from avsec.report import build_report

    path = build_report(args.input, _out(args, "reports/report"), args.title)
    print(f"-> {path}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from avsec.ui.server import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser,
          runs_dir=args.runs)
    return 0


def cmd_hardware(args: argparse.Namespace) -> int:
    """Capture from a real device.  Never falls back to synthetic frames."""
    from avsec import evaluation as ev
    from avsec import sources as S
    from avsec.utils import ensure_dir

    cfg = _cfg(args)
    try:
        src = S.from_capture_device(args.device, cfg.frame_height, cfg.frame_width,
                                    args.frames)
    except S.HardwareCaptureUnavailable as exc:
        print(f"hardware capture unavailable: {exc}", file=sys.stderr)
        print("This command does not substitute synthetic data. Connect a capture "
              "device or use 'avsec demo'.", file=sys.stderr)
        return 2
    out = ensure_dir(_out(args, "runs/hardware"))
    for i, f in enumerate(src.frames):
        ev.save_image(os.path.join(out, f"capture_{i:03d}.png"), f)
    write_json(os.path.join(out, "capture.json"), src.summary())
    print(f"captured {len(src)} frames from device {args.device} -> {out}")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    """Validate the dataset manifest: duplicates, split leakage, provenance."""
    from avsec.dataset import DatasetManifest, build_manifest
    from avsec.experiments import build_sources

    if getattr(args, "manifest", None) and os.path.exists(args.manifest):
        manifest = DatasetManifest.load(args.manifest)
        sources = None
    else:
        cfg = _cfg(args)
        sources = build_sources(cfg)
        manifest = build_manifest(sources, seed=cfg.seed)
    report = manifest.validate()
    out = _out(args, "runs/_dataset")
    manifest.save(out)
    write_json(os.path.join(out, "dataset_validation.json"), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        print(f"\nFAIL: {len(report['problems'])} problem(s); manifest written to {out}",
              file=sys.stderr)
        return 1
    print(f"\nOK: {report['n_clips']} clips, {report['n_scenes']} scenes, "
          f"no duplicates across splits; written to {out}")
    return 0


def cmd_protocol_check(args: argparse.Namespace) -> int:
    """Run the AEAD/framing protocol checks and fail loudly on any regression."""
    from avsec.experiments import run_protocol_checks

    cfg = _cfg(args)
    checks = run_protocol_checks(cfg)
    n_pass = sum(1 for c in checks if c.get("passed"))
    out = _out(args, "runs/_protocol")
    payload = {"checks": checks, "n_checks": len(checks), "n_passed": n_pass,
               "all_passed": n_pass == len(checks), "config": cfg.name}
    write_json(os.path.join(out, "protocol_checks.json"), payload)
    for c in checks:
        if not c.get("passed"):
            print(f"FAIL  {c.get('name')}: expected {c.get('expected')!r}, "
                  f"got {c.get('observed')!r}", file=sys.stderr)
    print(f"{n_pass}/{len(checks)} protocol checks passed -> {out}")
    return 0 if n_pass == len(checks) else 1


def cmd_matrix(args: argparse.Namespace) -> int:
    """Run (or estimate) one experiment matrix; resumes by default."""
    from avsec.matrix import MatrixRunner, MatrixSpec

    plan = args.plan or args.config
    if not plan:
        print("--plan is required", file=sys.stderr)
        return 2
    cfg, spec = MatrixSpec.from_plan(plan)
    out = _out(args, os.path.join("runs", spec.name))
    workers = args.workers if args.workers else spec.workers
    runner = MatrixRunner(cfg, spec, out, workers=workers)
    est = runner.estimate(measure=not args.no_measure).to_dict()
    if args.dry_run:
        print(json.dumps(est, indent=2, ensure_ascii=False))
        return 0
    if not args.resume and runner.completed():
        print(f"{len(runner.completed())} jobs already done in {out}; "
              "pass --resume to continue, or choose another --output",
              file=sys.stderr)
        return 1
    print(f"pending {est['n_jobs']} jobs, estimate {est['estimated_wall_human']}",
          file=sys.stderr)
    res = runner.run(progress=_progress, limit=args.limit)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("failures", "completeness")},
                     indent=2, ensure_ascii=False))
    c = res["completeness"]
    print(f"complete={c['complete']} missing={c['n_missing']} failed={c['n_failed']}")
    return 0 if c["complete"] else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Estimate every effect from a finished run.  Runs no new experiment."""
    from avsec.analysis import AnalysisPlan, analyse

    plan = AnalysisPlan.load(args.plan)
    res = analyse(args.input, plan, _out(args, None) or args.input)
    print(json.dumps({"n_scenes": res["n_scenes"],
                      "n_observations": res["n_observations"],
                      "channels": res["channels"], "methods": res["methods"],
                      "primary_verdict": res["primary_verdict"]},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_luma_lab(args: argparse.Namespace) -> int:
    """Luminance-balanced encryption: the monochrome and colour paths.

    Counts the colours available at one luminance level, runs both paths, and
    measures what a luminance-only receiver is left with.
    """
    from avsec.luma_lab import run_luma_lab

    cfg = _cfg(args)
    res = run_luma_lab(cfg, _out(args, "results/luma"), progress=_progress)
    print(json.dumps({
        "capacity": res["capacity"],
        "monochrome_exact": all(r["bit_exact_recovery"] for r in res["monochrome"]),
        "cipher_luma_entropy_bits":
            res["luma_only_observer"]["cipher_luma_entropy_bits"],
        "colour": [{k: r[k] for k in ("bits_kept", "bits_lost", "psnr_recovered_db")}
                   for r in res["colour"]],
        "plane_attacks": [{k: r[k] for k in ("view", "neighbour_accuracy_pct")}
                          for r in res["plane_attacks"]],
        "figures": res["figures"],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_gf_lab(args: argparse.Namespace) -> int:
    """The computed GF(2^8) substitution table against the stored one.

    Verifies the field arithmetic against the published AES S-box, measures the
    cryptographic properties of both constructions, and sweeps every irreducible
    polynomial of degree 8.
    """
    from avsec.galois_lab import run_galois_lab

    cfg = _cfg(args)
    res = run_galois_lab(cfg, _out(args, "results/gf_sbox"), progress=_progress)
    print(json.dumps({
        "matches_published_aes_sbox":
            res["verification"]["matches_published_aes_sbox"],
        "field": res["verification"]["field"]["polynomial"],
        "properties": [{k: r[k] for k in
                        ("table", "differential_uniformity", "nonlinearity",
                         "storage")} for r in res["properties"]],
        "irreducible_polynomials": len(res["polynomial_sweep"]),
        "figure": res["figures"].get("gf_sbox"),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_subst_lab(args: argparse.Namespace) -> int:
    """B2s: the substitution table, the attacks against it, and its channel cost.

    Writes the S-box and its inverse in full, the attack comparison against B2,
    the reuse matrix over both primitives, and the PSNR of every channel
    profile for B1, B2 and B2s.
    """
    from avsec.subst_lab import run_subst_lab

    cfg = _cfg(args)
    res = run_subst_lab(cfg, _out(args, "results/b2s"), progress=_progress)
    boundary = {a["target"]: a["boundary_neighbour_pct"] for a in res["attacks"]}
    clean = [r["psnr_full_db"] for r in res["channel_sweep"]
             if r["channel"] == "clean" and r["method"] == "B2s"]
    print(json.dumps({
        "grid": res["grid"],
        "sbox_bijective": res["sbox"]["bijective"],
        "sbox_inverse_exact": res["sbox"]["inverse_exact"],
        "bit_exact_recovery": all(c["bit_exact_recovery"]
                                  for c in res["correctness"]),
        "boundary_neighbour_pct": boundary,
        "b2s_clean_psnr_db": clean,
        "figures": res["figures"],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_lab(args: argparse.Namespace) -> int:
    """Reproduce the ISITIA 2021 scheme and repeat it on real photographs.

    One command, one picture, one directory: the replica of the article's own
    tables and the same measurement on crops of a UAV photograph.
    """
    from avsec.lab import run_lab

    cfg = _cfg(args)
    res = run_lab(cfg, _out(args, "results/lab"), progress=_progress)
    p1, p2 = res["part1_replica"], res["part2_natural"]
    print(json.dumps({
        "grid": res["grid"], "frame": res["frame"],
        "replica_mean_pct": p1["mean_similarity_scrambled_pct"],
        "paper_mean_pct": p1["paper_mean_similarity_scrambled_pct"],
        "replica_exact_recovery": p1["all_exact_recovery"],
        "natural_scenes": p2.get("n_scenes"),
        "natural_spread_pct": p2.get("spread_pct"),
        "figure": res["figures"].get("png"),
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Recompute every published claim from the published tables (R11).

    Runs no experiment and reads nothing but the directory it is given, so a
    reader with the results and no simulator can check each number.
    """
    from avsec.verify import render, verify

    res = verify(args.input, output=args.output or
                 os.path.join(args.input, "verification.json"))
    print(render(res))
    return 0 if res["ok"] else 1


def cmd_plots(args: argparse.Namespace) -> int:
    """Build the G01-G43 catalogue from a finished run.  Reads only."""
    from avsec.figures import FigureContext, build

    out = _out(args, os.path.join(args.input, "figures"))
    ctx = FigureContext(run_dir=args.input, out_dir=out, lang=args.lang,
                        formats=tuple(args.formats))
    rows = build(ctx, args.only.split(",") if args.only else None)
    ready = [r["figure"] for r in rows if r["status"] == "ready"]
    pending = [(r["figure"], r["reason"]) for r in rows if r["status"] != "ready"]
    print(f"ready {len(ready)}/{len(rows)}: {', '.join(ready)}")
    for gid, reason in pending:
        print(f"  pending {gid}: {reason}")
    print(f"index -> {os.path.join(out, 'figure_index.csv')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="avsec",
        description="Secure video over an existing analog composite video path "
                    "(research testbed).")
    p.add_argument("--version", action="version", version=f"avsec {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", help="YAML configuration file")
        sp.add_argument("--output", help="output directory")

    sp = sub.add_parser("info", help="print version and environment")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("gendata", help="generate the procedural test material")
    _common(sp)
    sp.add_argument("--width", type=int, default=256)
    sp.add_argument("--height", type=int, default=192)
    sp.add_argument("--frames", type=int, default=6)
    sp.add_argument("--rows", type=int, default=12)
    sp.add_argument("--cols", type=int, default=16)
    sp.set_defaults(func=cmd_gendata)

    sp = sub.add_parser("demo", help="one frame through every configured method")
    _common(sp)
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("scramble", help="B1 permutation of a still image, and back")
    _common(sp)
    sp.add_argument("--image", help="input image file (default: a synthetic pattern)")
    sp.add_argument("--pattern", default="edges", help="synthetic pattern name")
    sp.set_defaults(func=cmd_scramble)

    sp = sub.add_parser("attack", help="attack the permutation baselines and check AEAD")
    _common(sp)
    sp.set_defaults(func=cmd_attack)

    sp = sub.add_parser("transmit", help="end-to-end comparison over the level-A channel")
    _common(sp)
    sp.set_defaults(func=cmd_transmit)

    sp = sub.add_parser("simulate", help="short end-to-end run over the level-B CVBS model")
    _common(sp)
    sp.add_argument("--preset", default="mild",
                    choices=["clean", "mild", "moderate", "harsh"])
    sp.add_argument("--method", default="B4")
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("tune", help="parameter search on the calibration split")
    _common(sp)
    sp.add_argument("--quick", action="store_true", help="small search space")
    sp.add_argument("--frames", type=int, default=1, help="frames per candidate")
    sp.set_defaults(func=cmd_tune)

    sp = sub.add_parser("ablate", help="ablations of the proposed configuration")
    _common(sp)
    sp.set_defaults(func=cmd_ablate)

    sp = sub.add_parser("sweep", help="quality versus impairment strength")
    _common(sp)
    sp.add_argument("--presets", nargs="+",
                    default=["clean", "mild", "moderate", "bursty", "harsh"])
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("budget", help="channel budget and latency accounting")
    _common(sp)
    sp.set_defaults(func=cmd_budget)

    sp = sub.add_parser("benchmark", help="run the whole profile and store all artifacts")
    _common(sp)
    sp.add_argument("--fast", action="store_true", help="skip sweeps and ablations")
    sp.add_argument("--presets", nargs="+",
                    default=["clean", "mild", "moderate", "bursty", "harsh"])
    sp.add_argument("--cvbs-preset", default="mild", dest="cvbs_preset")
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("report", help="build a Markdown report from run artifacts")
    sp.add_argument("--input", required=True, help="directory with run artifacts")
    sp.add_argument("--output", help="report directory")
    sp.add_argument("--title", default="Звіт avsec")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("ui", help="start the local web interface")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--runs", default="runs", help="directory for run outputs")
    sp.add_argument("--no-browser", action="store_true")
    sp.set_defaults(func=cmd_ui)

    sp = sub.add_parser("dataset", help="validate the dataset manifest")
    _common(sp)
    sp.add_argument("--manifest", help="existing dataset_manifest.csv to check")
    sp.add_argument("validate", nargs="?", default="validate",
                    help="the only subcommand; kept for `avsec dataset validate`")
    sp.set_defaults(func=cmd_dataset)

    sp = sub.add_parser("protocol-check", help="AEAD and framing protocol checks")
    _common(sp)
    sp.set_defaults(func=cmd_protocol_check)

    sp = sub.add_parser("matrix", help="run one experiment matrix (resumable)")
    _common(sp)
    sp.add_argument("--plan", help="matrix plan file (a config with a matrix: block)")
    sp.add_argument("--dry-run", action="store_true",
                    help="estimate time and output size, run nothing")
    sp.add_argument("--resume", action="store_true",
                    help="continue an existing run directory without duplicating jobs")
    sp.add_argument("--workers", type=int, default=0,
                    help="process workers (0 = the plan's value)")
    sp.add_argument("--limit", type=int, default=0,
                    help="run at most this many jobs, then stop")
    sp.add_argument("--no-measure", action="store_true",
                    help="skip the per-method cost probe in the estimate")
    sp.set_defaults(func=cmd_matrix)

    sp = sub.add_parser("analyze", help="statistics from a finished run")
    sp.add_argument("--input", required=True, help="run directory")
    sp.add_argument("--output", help="where to write the tables (default: --input)")
    sp.add_argument("--plan", help="statistics plan YAML")
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("lab",
                        help="ISITIA 2021: reproduce the scheme and repeat it "
                             "on real photographs")
    _common(sp)
    sp.set_defaults(func=cmd_lab)

    sp = sub.add_parser("subst-lab",
                        help="B2s: keyed substitution added to the block "
                             "permutation - tables, attacks, channel cost")
    _common(sp)
    sp.set_defaults(func=cmd_subst_lab)

    sp = sub.add_parser("gf-lab",
                        help="computed GF(2^8) substitution table: verification "
                             "against AES, properties, polynomial sweep")
    _common(sp)
    sp.set_defaults(func=cmd_gf_lab)

    sp = sub.add_parser("luma-lab",
                        help="luminance-balanced encryption: monochrome and "
                             "colour paths, capacity, attacks")
    _common(sp)
    sp.set_defaults(func=cmd_luma_lab)

    sp = sub.add_parser("verify",
                        help="recompute every published claim from the tables")
    sp.add_argument("--input", required=True, help="results directory")
    sp.add_argument("--output", help="where to write verification.json")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("plots", help="build the G01-G43 figure catalogue")
    sp.add_argument("--input", required=True, help="run directory")
    sp.add_argument("--output", help="figure directory (default: <input>/figures)")
    sp.add_argument("--lang", default="uk", choices=("uk", "en"))
    sp.add_argument("--formats", nargs="+", default=["png", "svg", "pdf"])
    sp.add_argument("--only", help="comma separated figure ids, e.g. G03,G07")
    sp.set_defaults(func=cmd_plots)

    sp = sub.add_parser("hardware", help="capture from a real video device")
    _common(sp)
    sp.add_argument("--device", type=int, default=0)
    sp.add_argument("--frames", type=int, default=16)
    sp.set_defaults(func=cmd_hardware)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
