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
--repeat 2 runs each solver twice in one process: on Trainium the first solve includes compilation
(--repeat-solvers trnopt repeats only that one).
Each solver runs in its own process, killed --grace seconds past the time limit (a safety net: a
misconfigured appsi_highs once ran 40+ minutes past a 120 s limit). Writes results/pyomo_<host>_<date>.csv as it goes.
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
        # HiGHS option names, passed straight to HiGHS. Not opt.config.time_limit: SolverFactory returns appsi's
        # legacy wrapper, whose solve() rebuilds the config from its own arguments, silently resetting
        # time_limit to None (no limit). That is why HiGHS "ignored" its time limit on the MIPLIB instances.
        opt.options["time_limit"] = float(time_limit)
        opt.options["mip_rel_gap"] = float(gap)
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
    """Instance specs (picklable, so a fresh worker process can rebuild each one)."""
    out = []
    for f in args.families:
        from mip_benchmark import DEFAULT, QUICK
        n = len(QUICK) if f == "quick" else len(DEFAULT) - len(QUICK)
        out += [("family", f, i) for i in range(n)]
    for pattern in args.mps or []:
        for path in sorted(glob.glob(str(ROOT / pattern)) or glob.glob(pattern)):
            out.append(("mps", path, 0))
    return out


def make_instance(spec):
    kind, a, i = spec
    if kind == "mps":
        return load_mps(a)
    from mip_benchmark import DEFAULT, QUICK
    return (QUICK if a == "quick" else DEFAULT[len(QUICK):])[i]()


def run_solver(spec, s, reps, time_limit, gap_tol, backend, out_q):
    """Worker process: solve one instance with one solver `reps` times, putting each result on out_q."""
    lp = make_instance(spec)
    for rep in range(reps):
        m = lp_to_pyomo(lp)                                    # building the model is the user's job: not timed
        opt = make_solver(s, time_limit, gap_tol, backend)
        t = time.perf_counter()
        try:
            res = opt.solve(m)                                 # every plugin loads its solution into m
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
        out_q.put(dict(run=rep + 1, termination=term, objective=obj, bound=lb, seconds=round(wall, 3)))


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", choices=["quick", "large"], default=[])
    ap.add_argument("--mps", nargs="*", help="MPS files or globs (relative to the repo root or absolute)")
    ap.add_argument("--solvers", nargs="+", default=None, choices=["highs", "cuopt", "cuopt-bp", "trnopt"])
    ap.add_argument("--backend", default=None, help="trnopt backend (default: neuron > cuda > cpu-float32)")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--gap", type=float, default=1e-4, help="relative MIP gap, the same for every solver")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--repeat-solvers", nargs="*", default=None,
                    help="only these solvers get --repeat runs (default: all), e.g. --repeat-solvers trnopt")
    ap.add_argument("--grace", type=float, default=120.0,
                    help="seconds past the time limit before a solve is killed (a solver that ignores its limit)")
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
    out = ROOT / "results" / f"pyomo_{args.host}_{dt.date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    fields = ["instance", "rows", "cols", "nnz", "solver", "run", "termination", "objective", "bound", "gap",
              "seconds", "time_limit", "gap_tol"]
    with open(out, "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()
    ctx = mp.get_context("spawn")                              # each solve in a fresh process: hard wall limit
    for spec in instances(args):
        lp = make_instance(spec)
        if lp.integer is None or not lp.integer.any():
            print(f"\n{lp.name}: no integer variables, skipped")
            continue
        print(f"\n{lp.name}: {lp.m} rows x {lp.n} cols, {lp.K.nnz} nonzeros, {int(lp.integer.sum())} integer", flush=True)
        for s in solvers:
            reps = args.repeat if args.repeat_solvers is None or s in args.repeat_solvers else 1
            q = ctx.Queue()
            proc = ctx.Process(target=run_solver, args=(spec, s, reps, args.time_limit, args.gap, backend, q))
            proc.start()
            deadline = time.time() + reps * (args.time_limit + args.grace) + 60     # + model building
            got = 0
            while got < reps:
                try:
                    r = q.get(timeout=max(1.0, deadline - time.time()))
                except Exception:                            # queue.Empty: past the hard limit
                    proc.kill()
                    r = dict(run=got + 1, termination="killed (ignored time limit)", objective=float("nan"),
                             bound=float("nan"), seconds=round(args.time_limit + args.grace, 3))
                    got = reps
                else:
                    got += 1
                obj, lb = r["objective"], r["bound"]
                gap = abs(obj - lb) / max(1.0, abs(obj)) if np.isfinite(obj) and np.isfinite(lb) else float("nan")
                label = s if s != "trnopt" else f"trnopt ({backend})"
                tag = f" [run {r['run']}]" if reps > 1 else ""
                print(f"   {label + tag:28s} {r['termination']:16s} obj {obj:.6g}  bound {lb:.6g}  gap {gap:.1e}  "
                      f"{r['seconds']:8.1f} s", flush=True)
                row = dict(instance=lp.name, rows=lp.m, cols=lp.n, nnz=lp.K.nnz, solver=label, gap=gap,
                           time_limit=args.time_limit, gap_tol=args.gap, **r)
                with open(out, "a", newline="") as f:          # written as we go: a crash keeps what's done
                    csv.DictWriter(f, fields).writerow(row)
            proc.join(timeout=30)
            if proc.is_alive():
                proc.kill()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
