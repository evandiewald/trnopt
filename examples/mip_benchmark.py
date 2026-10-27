"""End-to-end MIP: trnopt's branch-and-bound with different branching rules and LP engines.

    uv run python examples/mip_benchmark.py            # default instances, ~20-40 minutes
    uv run python examples/mip_benchmark.py --quick    # small instances only, a few minutes

Every run solves the whole MIP: find whole-number solutions and PROVE the best one optimal (or report
the gap at the time limit). The search core is the same in every row -- HiGHS dual simplex solves each
node's LP exactly -- and only the branching rule, and the engine for its strong-branching LPs, change:

  most fractional       branch on the variable closest to .5 (no extra LPs)
  pseudocost            branch on the variable that helped most in the past (no extra LPs)
  reliability, HiGHS    pseudocost, but strong-branch variables with too little history, solving
                        those LPs with HiGHS warm-started dual simplex (1 core, and all cores)
  reliability, trnopt   the same, solving those LPs with trnopt's batched PDLP (CPU, and Mac GPU)

HiGHS's own MIP solver (presolve, cuts, heuristics, all cores) is shown for context, not as a target:
this branch-and-bound has none of that machinery.
Writes results/mip_<host>_<date>.csv.
"""

import argparse
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import os
import platform
import subprocess

import torch

from trnopt.generators import combinatorial_auction, facility_location, independent_set, set_cover
from trnopt.mip import solve_mip
from trnopt.reference import solve_highs_mip
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

QUICK = [lambda: set_cover(200, 400, 0.05, seed=2), lambda: independent_set(300, 4, seed=2),
         lambda: combinatorial_auction(80, 400, seed=2), lambda: facility_location(40, 30, seed=2)]
DEFAULT = QUICK + [lambda: set_cover(400, 800, 0.05, seed=3), lambda: set_cover(500, 1000, 0.02, seed=3),
                   lambda: independent_set(500, 4, seed=3), lambda: independent_set(800, 4, seed=3),
                   lambda: combinatorial_auction(150, 700, seed=3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--time-limit", type=float, default=300.0)
    args = ap.parse_args()
    cores = os.cpu_count() or 1
    gpu = torch.backends.mps.is_available()
    host = (subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            if platform.system() == "Darwin" else platform.node()) or "host"
    print(f"host: {host}   cores: {cores}   Mac GPU: {gpu}   time limit {args.time_limit:.0f}s per run")
    runs = [("most fractional", "most_fractional", {}),
            ("pseudocost", "pseudocost", {}),
            ("reliability, HiGHS 1 core", "reliability_highs", {}),
            (f"reliability, HiGHS {cores} cores", "reliability_highs", dict(highs_workers=cores)),
            ("reliability, trnopt CPU", "reliability_trnopt", dict(backend="cpu-float32"))]
    if gpu:
        runs.append(("reliability, trnopt GPU", "reliability_trnopt", dict(backend="mps-metal")))
    rows = []
    for make in (QUICK if args.quick else DEFAULT):
        lp = make()
        # HiGHS's MIP solver gets all cores, in its own FRESH process ("spawn", not a fork that would
        # inherit this process's HiGHS thread pool): HiGHS allows one thread-pool size per process.
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as ex:
            ref = ex.submit(solve_highs_mip, lp, args.time_limit, cores).result()
        print(f"\n{lp.name}: {lp.m} x {lp.n}, {lp.K.nnz} nonzeros, {int(lp.integer.sum())} integer variables")
        print(f"   {'HiGHS MIP solver (context)':30s} {ref.status:10s} obj {ref.objective:.6g}  {ref.seconds:7.1f} s")
        rows.append(dict(instance=lp.name, method="HiGHS MIP (all features)", status=ref.status,
                         objective=ref.objective, seconds=ref.seconds))
        for label, br, kw in runs:
            r = solve_mip(lp, br, time_limit=args.time_limit, **kw)
            ok = abs(r.objective - ref.objective) <= 1e-6 * (1 + abs(ref.objective))
            tag = "" if r.status == "optimal" else f" gap {r.gap:.1e}"
            print(f"   {label:30s} {r.status:10s} obj {r.objective:.6g}{'' if ok or r.status != 'optimal' else ' (MISMATCH)'}  "
                  f"{r.seconds:7.1f} s  nodes {r.nodes:6d}  strong-branching LPs {r.counts['strong_lps']:6d} "
                  f"({r.times['strong']:.1f} s){tag}", flush=True)
            rows.append(dict(instance=lp.name, method=label, status=r.status, objective=r.objective, bound=r.bound,
                             gap=r.gap, seconds=r.seconds, nodes=r.nodes, strong_lps=r.counts["strong_lps"],
                             strong_seconds=r.times["strong"], node_lp_seconds=r.times["node_lp"],
                             heuristic_seconds=r.times["heuristic"], matches_highs=ok))
    out = ROOT / "results" / f"mip_{host.replace(' ', '_')}_{dt.date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
