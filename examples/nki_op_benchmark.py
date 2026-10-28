"""Does keeping the NKI kernel inside the compiled graph speed up batched PDHG? spmv="nki" vs "nki-op".

    NEURON_RT_VISIBLE_CORES=1 TRNOPT_CACHE_DIR=$(mktemp -d) python examples/nki_op_benchmark.py --backend nki
    NEURON_RT_VISIBLE_CORES=1 TRNOPT_CACHE_DIR=$(mktemp -d) python examples/nki_op_benchmark.py --backend nki-op

  nki      the kernel called from a torch.compiler.disable function: each multiply by K or K^T is a graph
           break, so the compiled step runs as several graphs with the kernel calls between them
  nki-op   the kernel registered as the custom op trnopt::sell_matmul (nki_op.py): one graph per step

Run one backend per process with an empty compile cache (TRNOPT_CACHE_DIR), so compile times are cold
and the two paths can't reuse each other's kernels. For each instance and batch size B it measures:

  first_solve_s     a 128-iteration solve from a clean torch.compile state: mostly compilation
  graphs, breaks    torch.compile's counts from that first solve
  us_per_iter       a fixed 640-iteration batch (tol 1e-12, so nothing finishes early), per iteration
  e2e_s             a strong-branching batch solved to tol 1e-4 (time limit 300 s), as trn_benchmark.py does

Appends to results/nki_op_<date>.csv (both processes write the same file).
"""

import argparse
import csv
import datetime as dt
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
from torch._dynamo.utils import counters

from trnopt import Backend, PDLPOptions, load_mps, solve_pdlp
from trnopt.batch import solve_pdlp_batch
from trnopt.problems import branching_variants

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KEYS = ["instance", "rows", "cols", "nnz", "backend", "spmv_used", "B", "first_solve_s", "graphs", "breaks",
        "us_per_iter", "e2e_s", "e2e_iters", "e2e_us_per_iter", "n_optimal", "distinct_lps"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["nki", "nki-op"])
    ap.add_argument("--instances", nargs="+", default=["chromaticindex512-7", "neos-4300652-rahue", "neos-933966"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 128])
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "results" / f"nki_op_{dt.date.today()}.csv"))
    args = ap.parse_args()
    torch._dynamo.config.recompile_limit = 1024

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, KEYS)
    if new:
        w.writeheader()

    be = Backend.preset("neuron", spmv=args.backend, dense=False)
    for name in args.instances:
        lp = load_mps(str(next((DATA / "miplib").glob(f"{name}*.mps*"))))
        root = solve_pdlp(lp, PDLPOptions(tol=1e-4, time_limit=120))           # CPU, float64: bounds to branch on
        L, U, labels = branching_variants(lp, root.x, args.candidates, tol=1e-3)
        print(f"{lp.name} ({lp.m} x {lp.n}, {lp.K.nnz} nnz), {len(labels)} branching LPs, backend {args.backend}",
              flush=True)
        for B in args.batch_sizes:
            LB, UB = np.resize(L, (B, lp.n)), np.resize(U, (B, lp.n))           # repeats rows if B > len(labels)
            row = dict(instance=lp.name, rows=lp.m, cols=lp.n, nnz=lp.K.nnz, backend=args.backend, B=B,
                       distinct_lps=min(B, len(labels)))
            solve = lambda **kw: solve_pdlp_batch(lp, B, lower=LB, upper=UB, opts=PDLPOptions(backend=be, **kw))
            torch._dynamo.reset()
            counters.clear()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    t = time.perf_counter()
                    solve(tol=1e-12, max_iters=128)
                    row["first_solve_s"] = round(time.perf_counter() - t, 2)
                    row["graphs"] = counters["stats"]["unique_graphs"]
                    row["breaks"] = sum(counters["graph_break"].values())
                    r = solve(tol=1e-12, max_iters=640)
                    row["us_per_iter"] = round(1e6 * r.seconds / max(r.total_iterations, 1), 1)
                    t = time.perf_counter()
                    r = solve(tol=1e-4, time_limit=300)
                    row["e2e_s"] = round(time.perf_counter() - t, 3)
                    row["e2e_iters"] = r.total_iterations
                    row["e2e_us_per_iter"] = round(1e6 * r.seconds / max(r.total_iterations, 1), 1)
                    row["n_optimal"] = r.status.count("optimal")
                except Exception:
                    traceback.print_exc(limit=4)
            # MatVec warns when "nki-op" falls back to "nki"
            row["spmv_used"] = "nki" if any("using 'nki'" in str(c.message) for c in caught) else args.backend
            print("  " + ", ".join(f"{k}={row.get(k)}" for k in KEYS[4:]), flush=True)
            w.writerow(row)
            f.flush()
    f.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
