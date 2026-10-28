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
  reliability, trnopt   the same, solving those LPs with trnopt's batched PDLP (CPU, Mac GPU, Trainium)
  strong, ...           full strong branching at every node (every fractional candidate, every node):
                        too expensive on CPUs, the question is whether a batched accelerator changes that

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


def _neuron_available() -> bool:
    try:
        import torch_neuronx  # noqa: F401
        return torch.neuron.device_count() > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--set", choices=["quick", "large", "default"], default=None,
                    help="instances: quick (= --quick), large (the default set minus the quick ones), default (both)")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--runs", nargs="+", default=None,
                    help="substrings of run labels to keep (e.g. 'pseudocost' 'trn1'); default: all available")
    ap.add_argument("--host", default=None, help="label for the machine in the CSV name, e.g. trn1.2xlarge")
    ap.add_argument("--warmup", action="store_true",
                    help="Trainium runs: first solve once untimed (compiles every batch size it hits), then time a "
                         "second solve; compile time is reported separately")
    args = ap.parse_args()
    cores = os.cpu_count() or 1
    gpu = torch.backends.mps.is_available()
    host = (subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            if platform.system() == "Darwin" else platform.node()) or "host"
    host = args.host or host
    neuron = _neuron_available()
    print(f"host: {host}   cores: {cores}   Mac GPU: {gpu}   Trainium: {neuron}   time limit {args.time_limit:.0f}s per run")
    runs = [("most fractional", "most_fractional", {}),
            ("pseudocost", "pseudocost", {}),
            ("reliability, HiGHS 1 core", "reliability_highs", {}),
            (f"reliability, HiGHS {cores} cores", "reliability_highs", dict(highs_workers=cores)),
            ("reliability, trnopt CPU", "reliability_trnopt", dict(backend="cpu-float32"))]
    if gpu:
        runs.append(("reliability, trnopt GPU", "reliability_trnopt", dict(backend="mps-metal")))
    if neuron:
        from trnopt import Backend
        trn = Backend.preset("neuron")        # dense K if it fits, else the NKI kernel
        runs += [("reliability, trnopt trn1", "reliability_trnopt", dict(backend=trn, slots=1024)),
                 (f"strong, HiGHS {cores} cores", "strong_highs", dict(highs_workers=cores)),
                 ("strong, trnopt trn1", "strong_trnopt", dict(backend=trn, slots=1024))]
    if torch.cuda.is_available():
        from trnopt import Backend
        cu = Backend.preset("cuda", dtype=torch.float32)
        runs += [("reliability, trnopt CUDA", "reliability_trnopt", dict(backend=cu, slots=1024)),
                 (f"strong, HiGHS {cores} cores", "strong_highs", dict(highs_workers=cores)),
                 ("strong, trnopt CUDA", "strong_trnopt", dict(backend=cu, slots=1024))]
    cuopt_runs = []
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import cuopt_adapter  # noqa: F401
        cuopt_runs = [("cuOpt MIP (defaults)", {}),
                      ("cuOpt MIP + batch-PDLP branching", {"mip_batch_pdlp_strong_branching": 1,
                                                            "mip_batch_pdlp_reliability_branching": 1})]
    except Exception:
        pass
    if args.runs:
        runs = [r for r in runs if any(k in r[0] for k in args.runs)]
        cuopt_runs = [r for r in cuopt_runs if any(k in r[0] for k in args.runs)]
    rows = []
    chosen = args.set or ("quick" if args.quick else "default")
    for make in {"quick": QUICK, "large": DEFAULT[len(QUICK):], "default": DEFAULT}[chosen]:
        lp = make()
        # HiGHS's MIP solver gets all cores, in its own FRESH process ("spawn", not a fork that would
        # inherit this process's HiGHS thread pool): HiGHS allows one thread-pool size per process.
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as ex:
            ref = ex.submit(solve_highs_mip, lp, args.time_limit, cores).result()
        print(f"\n{lp.name}: {lp.m} x {lp.n}, {lp.K.nnz} nonzeros, {int(lp.integer.sum())} integer variables")
        print(f"   {'HiGHS MIP solver (context)':30s} {ref.status:10s} obj {ref.objective:.6g}  {ref.seconds:7.1f} s")
        rows.append(dict(instance=lp.name, method="HiGHS MIP (all features)", status=ref.status,
                         objective=ref.objective, seconds=ref.seconds))
        for label, opts in cuopt_runs:
            from cuopt_adapter import solve_mip as cuopt_mip, to_data_model
            try:
                c = cuopt_mip(to_data_model(lp, integer=True), time_limit=args.time_limit, options=opts)
                ok = abs(c.objective - ref.objective) <= 1e-6 * (1 + abs(ref.objective))
                print(f"   {label:30s} {c.status:10s} obj {c.objective:.6g}{'' if ok else ' (differs)'}  {c.seconds:7.1f} s",
                      flush=True)
                rows.append(dict(instance=lp.name, method=label, status=c.status, objective=c.objective,
                                 seconds=c.seconds, matches_highs=ok))
            except Exception as e:
                print(f"   {label}: failed {type(e).__name__}: {str(e)[:150]}", flush=True)
        for label, br, kw in runs:
            compile_s = None
            if args.warmup and ("trn1" in label or "CUDA" in label):
                w = solve_mip(lp, br, time_limit=args.time_limit, **kw)      # compiles; Neuron caches per process
                compile_s = max(0.0, w.seconds)
            r = solve_mip(lp, br, time_limit=args.time_limit, **kw)
            ok = abs(r.objective - ref.objective) <= 1e-6 * (1 + abs(ref.objective))
            tag = "" if r.status == "optimal" else f" gap {r.gap:.1e}"
            print(f"   {label:30s} {r.status:10s} obj {r.objective:.6g}{'' if ok or r.status != 'optimal' else ' (MISMATCH)'}  "
                  f"{r.seconds:7.1f} s  nodes {r.nodes:6d}  strong-branching LPs {r.counts['strong_lps']:6d} "
                  f"({r.times['strong']:.1f} s){tag}"
                  + (f"   [first run incl. compile {compile_s:.1f} s]" if compile_s is not None else ""), flush=True)
            rows.append(dict(instance=lp.name, method=label, status=r.status, objective=r.objective, bound=r.bound,
                             gap=r.gap, seconds=r.seconds, nodes=r.nodes, strong_lps=r.counts["strong_lps"],
                             strong_seconds=r.times["strong"], node_lp_seconds=r.times["node_lp"],
                             heuristic_seconds=r.times["heuristic"], matches_highs=ok, first_run_seconds=compile_s))
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
