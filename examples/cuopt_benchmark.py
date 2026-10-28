"""NVIDIA cuOpt vs HiGHS vs trnopt on a GPU box (meant for a g4dn: one T4, priced like a trn1.2xlarge).

    python examples/cuopt_adapter.py                                    # first: check the cuOpt API
    python examples/cuopt_benchmark.py --sections single --host g4dn.4xlarge
    python examples/cuopt_benchmark.py --sections mip --time-limit 300 --host g4dn.4xlarge

Strong branching (the batched workload) is in strong_branching_benchmark.py:
    python examples/strong_branching_benchmark.py --methods cpu-float32 cuda-float32 cuopt-pdlp --host g4dn.4xlarge

Sections
--------
single : one LP at a time (LP relaxations of the MIPLIB instances in data/miplib plus the Netlib ones in
         data/), tolerance 1e-4. cuOpt PDLP and cuOpt's default (concurrent) vs HiGHS (default, 1 thread)
         vs trnopt on the GPU (single LP, float32). Objective error is measured against HiGHS.
mip    : whole MIPs to proven optimality (or the gap at the time limit): cuOpt's MIP solver, HiGHS's MIP
         solver (all cores, own process), and trnopt's branch-and-bound with reliability branching whose
         strong-branching LPs run on the GPU. Instances: mip_benchmark.py's generated families plus the
         small MIPLIB instances.
Writes results/cuopt_<section>_<host>_<date>.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import multiprocessing
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from cuopt_adapter import solve_lp, solve_mip as cuopt_mip, to_data_model
from trnopt import Backend, PDLPOptions, load_mps, solve_pdlp
from trnopt.reference import solve_highs, solve_highs_mip

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TOL = 1e-4


def rel_err(a, b):
    return abs(a - b) / (1 + abs(b))


def write(rows, section, host):
    out = ROOT / "results" / f"cuopt_{section}_{host}_{dt.date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


def section_single(args, host):
    files = sorted(DATA.glob("*.mps")) + sorted((DATA / "miplib").glob("*.mps.gz"))
    gpu = Backend.preset("cuda", dtype=torch.float32) if torch.cuda.is_available() else None
    rows = []
    print(f"{'instance':28s} {'HiGHS':>9s} {'cuOpt PDLP':>12s} {'iters':>7s} {'err':>8s} {'cuOpt dflt':>11s} "
          f"{'trnopt GPU':>11s} {'iters':>7s} {'err':>8s}")
    for f in files:
        lp = load_mps(str(f))
        lp.integer = None                                       # LP relaxation
        ref = solve_highs(lp, time_limit=args.time_limit)
        dm = to_data_model(lp)
        solve_lp(dm, tol=TOL, time_limit=args.time_limit)        # warm-up (CUDA context, kernels)
        cp = solve_lp(dm, tol=TOL, method="pdlp", time_limit=args.time_limit)
        cc = solve_lp(dm, tol=TOL, method="concurrent", time_limit=args.time_limit)
        line = f"{lp.name[:28]:28s} {ref.seconds:9.3f} {cp.seconds:12.3f} {cp.iterations or 0:7d} " \
               f"{rel_err(cp.objective, ref.objective):8.1e} {cc.seconds:11.3f}"
        row = dict(instance=lp.name, m=lp.m, n=lp.n, nnz=lp.K.nnz, highs_status=ref.status, highs_obj=ref.objective,
                   highs_seconds=ref.seconds, cuopt_pdlp_status=cp.status, cuopt_pdlp_obj=cp.objective,
                   cuopt_pdlp_seconds=cp.seconds, cuopt_pdlp_solver_seconds=cp.solver_seconds,
                   cuopt_pdlp_iters=cp.iterations, cuopt_pdlp_err=rel_err(cp.objective, ref.objective),
                   cuopt_default_status=cc.status, cuopt_default_seconds=cc.seconds,
                   cuopt_default_err=rel_err(cc.objective, ref.objective))
        if gpu is not None:
            try:
                solve_pdlp(lp, PDLPOptions(tol=TOL, backend=gpu, max_iters=64))
                t = time.perf_counter()
                r = solve_pdlp(lp, PDLPOptions(tol=TOL, backend=gpu, time_limit=args.time_limit))
                sec = time.perf_counter() - t
                line += f" {sec:11.3f} {r.iterations:7d} {rel_err(r.objective, ref.objective):8.1e}"
                row.update(trnopt_gpu_status=r.status, trnopt_gpu_seconds=sec, trnopt_gpu_iters=r.iterations,
                           trnopt_gpu_err=rel_err(r.objective, ref.objective))
            except Exception as e:
                line += f"  trnopt GPU failed: {type(e).__name__}"
        print(line, flush=True)
        rows.append(row)
    write(rows, "single", host)


def section_mip(args, host):
    from mip_benchmark import DEFAULT, QUICK
    from trnopt.mip import solve_mip
    makers = list(QUICK if args.quick else DEFAULT)
    miplib = [p for p in sorted((DATA / "miplib").glob("*.mps.gz")) if p.stat().st_size < 200_000]
    makers += [(lambda p=p: load_mps(str(p))) for p in miplib]
    cores = os.cpu_count() or 1
    gpu = Backend.preset("cuda", dtype=torch.float32) if torch.cuda.is_available() else None
    rows = []
    for make in makers:
        lp = make()
        if lp.integer is None or not lp.integer.any():
            continue
        print(f"\n{lp.name}: {lp.m} x {lp.n}, {lp.K.nnz} nonzeros, {int(lp.integer.sum())} integer variables", flush=True)
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as ex:
            ref = ex.submit(solve_highs_mip, lp, args.time_limit, cores).result()
        print(f"   {'HiGHS MIP (all cores)':32s} {ref.status:12s} obj {ref.objective:.6g}  {ref.seconds:8.1f} s", flush=True)
        rows.append(dict(instance=lp.name, method="HiGHS MIP", status=ref.status, objective=ref.objective, seconds=ref.seconds))
        try:
            c = cuopt_mip(to_data_model(lp, integer=True), time_limit=args.time_limit)
            print(f"   {'cuOpt MIP':32s} {c.status:12s} obj {c.objective:.6g}  {c.seconds:8.1f} s "
                  f"(err vs HiGHS {rel_err(c.objective, ref.objective):.1e})", flush=True)
            rows.append(dict(instance=lp.name, method="cuOpt MIP", status=c.status, objective=c.objective, seconds=c.seconds,
                             solver_seconds=c.solver_seconds))
        except Exception as e:
            print(f"   cuOpt MIP failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        if gpu is not None and not args.skip_trnopt:
            r = solve_mip(lp, "reliability_trnopt", time_limit=args.time_limit, backend=gpu, slots=1024)
            print(f"   {'trnopt B&B, reliability on GPU':32s} {r.status:12s} obj {r.objective:.6g}  {r.seconds:8.1f} s "
                  f"nodes {r.nodes}", flush=True)
            rows.append(dict(instance=lp.name, method="trnopt B&B reliability (GPU)", status=r.status, objective=r.objective,
                             seconds=r.seconds, nodes=r.nodes, gap=r.gap))
    write(rows, "mip", host)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", default=["single", "mip"], choices=["single", "mip"])
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--quick", action="store_true", help="mip: the small generated instances only")
    ap.add_argument("--skip-trnopt", action="store_true", help="mip: only cuOpt and HiGHS")
    ap.add_argument("--host", default=platform.node() or "host")
    args = ap.parse_args()
    print(f"host {args.host}  GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}  "
          f"cores {os.cpu_count()}")
    if "single" in args.sections:
        section_single(args, args.host)
    if "mip" in args.sections:
        section_mip(args, args.host)


if __name__ == "__main__":
    main()
