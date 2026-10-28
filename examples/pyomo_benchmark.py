"""End-to-end MIP benchmark through Pyomo: every solver gets the same Pyomo model, the same time limit and the
same relative gap, documented options only, and is timed from solve() call to result (Pyomo's own model
translation included, since a user pays it too).

    python examples/pyomo_benchmark.py --families quick large                       # generated families
    python examples/pyomo_benchmark.py --mps "data/miplib/*.mps.gz" --time-limit 300   # real MIPLIB MIPs
    python examples/pyomo_benchmark.py --solvers highs trnopt --backend neuron --repeat 2

Solvers (--solvers):
    highs      Pyomo's "appsi_highs" (HiGHS MIP: presolve, cuts, heuristics, parallel B&B)
    cuopt      Pyomo's "cuopt" (NVIDIA cuOpt MIP, defaults)
    cuopt-bp   the same with cuOpt's documented batch-PDLP branching options on
    trnopt     Pyomo's "trnopt" (this package; trnopt.pyomo_plugin), on --backend
--repeat 2 runs each solver twice in one process: on Trainium the first solve includes compilation.
Writes results/pyomo_<host>_<date>.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import torch

import trnopt.pyomo_plugin  # noqa: F401  (registers SolverFactory("trnopt"))
from trnopt import load_mps
from trnopt.pyomo_plugin import lp_to_pyomo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def default_backend() -> str:
    try:
        import torch_neuronx  # noqa: F401
        if torch.neuron.device_count() > 0:
            return "neuron"
    except Exception:
        pass
    return "cuda" if torch.cuda.is_available() else "cpu-float32"


def make_solver(name: str, time_limit: float, gap: float, backend: str):
    if name == "highs":
        opt = pyo.SolverFactory("appsi_highs")
        opt.config.time_limit = time_limit
        opt.config.mip_gap = gap
        return opt
    if name in ("cuopt", "cuopt-bp"):
        opt = pyo.SolverFactory("cuopt")
        opt.options["time_limit"] = time_limit
        opt.options["mip_relative_gap"] = gap
        if name == "cuopt-bp":
            opt.options["mip_batch_pdlp_strong_branching"] = 1
            opt.options["mip_batch_pdlp_reliability_branching"] = 1
        return opt
    if name == "trnopt":
        opt = pyo.SolverFactory("trnopt")
        opt.options.update(backend=backend, time_limit=time_limit, mip_gap=gap)
        return opt
    raise ValueError(name)


def instances(args):
    out = []
    if args.families:
        from mip_benchmark import DEFAULT, QUICK
        sets = {"quick": QUICK, "large": DEFAULT[len(QUICK):]}
        for f in args.families:
            out += [mk for mk in sets[f]]
    for pattern in args.mps or []:
        for path in sorted(glob.glob(str(ROOT / pattern)) or glob.glob(pattern)):
            out.append(lambda p=path: load_mps(p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", choices=["quick", "large"], default=[])
    ap.add_argument("--mps", nargs="*", help="MPS files or globs (relative to the repo root or absolute)")
    ap.add_argument("--solvers", nargs="+", default=None, choices=["highs", "cuopt", "cuopt-bp", "trnopt"])
    ap.add_argument("--backend", default=None, help="trnopt backend (default: neuron > cuda > cpu-float32)")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--gap", type=float, default=1e-4, help="relative MIP gap, the same for every solver")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--host", default=platform.node() or "host")
    args = ap.parse_args()
    backend = args.backend or default_backend()
    solvers = args.solvers
    if solvers is None:
        solvers = ["highs", "trnopt"]
        try:
            import cuopt  # noqa: F401
            solvers[1:1] = ["cuopt", "cuopt-bp"]
        except ImportError:
            pass
    print(f"host {args.host}   trnopt backend {backend}   solvers {solvers}   "
          f"time limit {args.time_limit:.0f}s   gap {args.gap:g}")
    rows = []
    for make in instances(args):
        lp = make()
        if lp.integer is None or not lp.integer.any():
            print(f"\n{lp.name}: no integer variables, skipped")
            continue
        print(f"\n{lp.name}: {lp.m} rows x {lp.n} cols, {lp.K.nnz} nonzeros, {int(lp.integer.sum())} integer", flush=True)
        for s in solvers:
            for rep in range(args.repeat):
                m = lp_to_pyomo(lp)                            # building the model is the user's job: not timed
                opt = make_solver(s, args.time_limit, args.gap, backend)
                t = time.perf_counter()
                try:
                    res = opt.solve(m)                         # every plugin loads its solution into m
                    wall = time.perf_counter() - t
                    term = str(res.solver.termination_condition)
                    try:
                        obj = pyo.value(m.obj)
                    except Exception:
                        obj = float("nan")
                    lb = res.problem.lower_bound
                    lb = float(lb) if lb is not None and np.isfinite(float(lb)) else float("nan")
                except Exception as e:
                    wall, term, obj, lb = time.perf_counter() - t, f"error: {type(e).__name__}: {str(e)[:100]}", float("nan"), float("nan")
                gap = abs(obj - lb) / max(1.0, abs(obj)) if np.isfinite(obj) and np.isfinite(lb) else float("nan")
                label = s if s != "trnopt" else f"trnopt ({backend})"
                tag = f" [run {rep + 1}]" if args.repeat > 1 else ""
                print(f"   {label + tag:28s} {term:16s} obj {obj:.6g}  bound {lb:.6g}  gap {gap:.1e}  {wall:8.1f} s", flush=True)
                rows.append(dict(instance=lp.name, rows=lp.m, cols=lp.n, nnz=lp.K.nnz, solver=label, run=rep + 1,
                                 termination=term, objective=obj, bound=lb, gap=gap, seconds=round(wall, 3),
                                 time_limit=args.time_limit, gap_tol=args.gap))
    out = ROOT / "results" / f"pyomo_{args.host}_{dt.date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
