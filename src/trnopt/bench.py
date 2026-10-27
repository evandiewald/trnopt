"""Benchmark harness: solve a set of LPs with HiGHS and one or more trnopt backends, write a CSV.

    uv run trnopt-bench                                         # Netlib + synthetic, CPU float64
    uv run trnopt-bench --backends cpu emulate-neuron --finish-float64
    trnopt-bench --backends neuron --hardware trn1.2xlarge --price-per-hour 1.344 --out trn1.csv
    uv run trnopt-bench --instances path/to/miplib/*.mps --time-limit 300

Every row records what was run, on what, with which software versions, so results from
different machines can be combined and compared later.

Two numbers matter on accelerators and are reported separately:
* compile_seconds -- a short warm-up solve that triggers all compiles (expensive on Neuron:
  every distinct program is compiled once, and cached after that);
* seconds         -- the real solve, after warm-up.

Summary uses the field's standard aggregate, the *shifted geometric mean* of solve time:
exp(mean(log(t + shift))) - shift. The geometric mean keeps one very slow instance from
dominating the way an ordinary average would; the shift (10 s by convention) stops tiny times
from dominating instead. Unsolved instances count at the time limit.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import math
import platform
import socket
import time
from dataclasses import replace
from importlib import metadata
from pathlib import Path

from .backend import Backend
from .lp import LP
from .pdlp import PDLPOptions, solve_pdlp
from .reference import load_mps, solve_highs

DATA = Path(__file__).resolve().parents[2] / "data"

FIELDS = [
    "timestamp", "hardware", "host", "instance", "rows", "cols", "nonzeros",
    "config", "backend", "dtype", "dense", "compile", "presolve", "finish_float64", "tol",
    "highs_status", "highs_objective", "highs_seconds",
    "status", "objective", "obj_rel_err", "primal_residual", "dual_residual", "gap",
    "iterations", "finish_iterations", "restarts", "compile_seconds", "seconds", "usd",
    "presolve_note", "versions",
]


def _versions() -> str:
    out = []
    for pkg in ("trnopt", "torch", "torch-neuronx", "nki", "highspy", "numpy", "scipy"):
        try:
            out.append(f"{pkg}={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            pass
    out.append(f"python={platform.python_version()}")
    return " ".join(out)


def _instances(specs: list[str]) -> list[LP]:
    """Turn command-line instance specs into LPs. 'netlib' and 'synthetic' are built-in sets."""
    from .problems import infeasible_lp, loan_selection, random_lp, unbounded_lp

    lps: list[LP] = []
    for spec in specs:
        if spec == "netlib":
            lps += [load_mps(str(p)) for p in sorted(DATA.glob("*.mps"))]
        elif spec == "synthetic":
            lps += [random_lp(), random_lp(n=2000, m_eq=300, m_ineq=600, density=0.005, seed=1),
                    loan_selection(), infeasible_lp(), unbounded_lp()]
        else:
            paths = sorted(glob.glob(spec)) or [spec]
            lps += [load_mps(p) for p in paths]
    return lps


def shifted_geomean(times: list[float], shift: float = 10.0) -> float:
    if not times:
        return float("nan")
    return math.exp(sum(math.log(t + shift) for t in times) / len(times)) - shift


def run(args: argparse.Namespace) -> list[dict]:
    lps = _instances(args.instances)
    configs = []
    for name in args.backends:
        be = Backend.preset(name, **({"compile": False} if args.no_compile else {}))
        tol = args.tol or (1e-6 if args.finish_float64 or str(be.dtype) == "torch.float64" else 1e-4)
        configs.append((name, PDLPOptions(
            tol=tol, backend=be, presolve=args.presolve, finish_float64=args.finish_float64,
            time_limit=args.time_limit, max_iters=args.max_iters,
        )))

    versions, host = _versions(), socket.gethostname()
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []
    out = open(args.out, "w", newline="") if args.out else None
    writer = csv.DictWriter(out, FIELDS) if out else None
    if writer:
        writer.writeheader()

    print(f"{'instance':<22}{'config':<36}{'status':<24}{'obj err':>9}{'iters':>9}{'compile s':>10}{'solve s':>9}{'HiGHS s':>9}")
    for lp in lps:
        ref = solve_highs(lp, time_limit=args.time_limit)
        for name, opts in configs:
            # Warm-up: a few hundred iterations to trigger every compile, timed separately.
            t0 = time.perf_counter()
            solve_pdlp(lp, replace(opts, max_iters=3 * opts.eval_every, presolve=False, finish_float64=False))
            compile_s = time.perf_counter() - t0 if (opts.backend.compile or opts.backend.lazy) else 0.0

            res = solve_pdlp(lp, opts)
            both = ref.status == "Optimal" and res.status == "optimal"
            err = abs(res.objective - ref.objective) / (1 + abs(ref.objective)) if both else float("nan")
            row = {
                "timestamp": stamp, "hardware": args.hardware, "host": host, "instance": lp.name,
                "rows": lp.m, "cols": lp.n, "nonzeros": lp.K.nnz,
                "config": name + ("+presolve" if opts.presolve else "") + ("+f64finish" if opts.finish_float64 else ""),
                "backend": name, "dtype": str(opts.backend.dtype).replace("torch.", ""), "dense": opts.backend.dense,
                "compile": opts.backend.compile, "presolve": opts.presolve, "finish_float64": opts.finish_float64,
                "tol": opts.tol,
                "highs_status": ref.status, "highs_objective": ref.objective, "highs_seconds": round(ref.seconds, 4),
                "status": res.status, "objective": res.objective, "obj_rel_err": err,
                "primal_residual": res.primal_residual, "dual_residual": res.dual_residual, "gap": res.gap,
                "iterations": res.iterations, "finish_iterations": res.finish_iterations, "restarts": res.restarts,
                "compile_seconds": round(compile_s, 3), "seconds": round(res.seconds, 4),
                "usd": round(res.seconds / 3600 * args.price_per_hour, 6) if args.price_per_hour else "",
                "presolve_note": res.presolve, "versions": versions,
            }
            rows.append(row)
            if writer:
                writer.writerow(row)
                out.flush()
            print(f"{lp.name[:22]:<22}{row['config'][:36]:<36}{res.status:<24}{err:>9.1e}{res.iterations:>9}"
                  f"{compile_s:>10.1f}{res.seconds:>9.2f}{ref.seconds:>9.2f}", flush=True)
    if out:
        out.close()
    _summary(rows, args.time_limit)
    return rows


def _summary(rows: list[dict], time_limit: float) -> None:
    """Per configuration: how many solved correctly, and the shifted geometric mean of solve time."""
    print("\nSummary (shifted geometric mean, shift 10 s; unsolved counted at the time limit)")
    print(f"{'config':<36}{'correct':>9}{'of':>4}{'SGM solve s':>13}{'SGM HiGHS s':>13}")
    for config in dict.fromkeys(r["config"] for r in rows):
        rs = [r for r in rows if r["config"] == config]
        expected = {"Optimal": "optimal", "Infeasible": "infeasible", "Unbounded": "unbounded"}
        correct = sum(1 for r in rs if expected.get(r["highs_status"]) == r["status"]
                      or (r["status"] == "unbounded_or_infeasible" and r["highs_status"] in ("Infeasible", "Unbounded")))
        times = [r["seconds"] if r["status"] in ("optimal", "infeasible", "unbounded", "unbounded_or_infeasible")
                 else time_limit for r in rs]
        print(f"{config:<36}{correct:>9}{len(rs):>4}{shifted_geomean(times):>13.3f}"
              f"{shifted_geomean([r['highs_seconds'] for r in rs]):>13.3f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", default=["netlib", "synthetic"],
                    help="MPS files or globs, or the built-in sets 'netlib' and 'synthetic'")
    ap.add_argument("--backends", nargs="+", default=["cpu"],
                    choices=["cpu", "cpu-float32", "cuda", "neuron", "neuron-xla", "emulate-neuron"])
    ap.add_argument("--tol", type=float, default=None,
                    help="default: 1e-6 for float64 or with --finish-float64, else 1e-4")
    ap.add_argument("--presolve", action="store_true")
    ap.add_argument("--finish-float64", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--max-iters", type=int, default=500_000)
    ap.add_argument("--hardware", default=platform.machine(), help="label for the machine, e.g. trn1.2xlarge")
    ap.add_argument("--price-per-hour", type=float, default=None, help="instance price, for a dollars-per-solve column")
    ap.add_argument("--out", default="results.csv")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
