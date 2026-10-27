"""trnopt on AWS Trainium: speed on a NeuronCore, in the same shape as examples/mac_benchmark.py.

    python examples/trn_benchmark.py                                   # all backends
    python examples/trn_benchmark.py --backends neuron neuron-nki      # the chip only

Backends:
    cpu-float32   the instance's own CPU (8 vCPUs on trn1.2xlarge), for reference
    neuron        TorchNeuron, dense K, torch.compile with the "neuron" backend
    neuron-nki    TorchNeuron, sparse K through our NKI sliced-ELL kernel (nki_sell.py), compiled
                  around the kernel calls (torch.compile can't trace into nki.jit, so it breaks there)

Writes results/trn_<instance-type>_<date>.csv after every measurement (same columns as the Mac
CSV), so a run cut short still leaves its data behind.
"""

import argparse
import csv
import datetime as dt
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

import torch
import torch._dynamo

from trnopt import Backend, PDLPOptions, load_mps, solve_pdlp
from trnopt.batch import solve_pdlp_batch
from trnopt.problems import branching_variants

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BACKENDS = {
    "cpu-float32": lambda: Backend.preset("cpu", dtype=torch.float32),
    "neuron": lambda: Backend.preset("neuron"),
    "neuron-nki": lambda: Backend.preset("neuron", spmv="nki", dense=False),
}
KEYS = ["B", "backend", "cols", "instance", "n_optimal", "rows", "sec_per_lp", "seconds", "section", "us_per_iter",
        "distinct_lps"]


def instance_type() -> str:
    try:
        tok = urllib.request.urlopen(urllib.request.Request(
            "http://169.254.169.254/latest/api/token", method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=2).read().decode()
        return urllib.request.urlopen(urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": tok}), timeout=2).read().decode()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=list(BACKENDS))
    ap.add_argument("--sections", nargs="+", default=["single", "batch"], choices=["single", "batch"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32, 128])
    ap.add_argument("--candidates", type=int, default=64,
                    help="branching candidates per instance (2 LPs each). If a batch is bigger than the "
                         "distinct LPs available, the set is repeated; distinct_lps in the CSV says so")
    ap.add_argument("--instances", nargs="+", default=None,
                    help="substrings of MIPLIB file names to keep (default: the small ones; naming lets in large ones)")
    ap.add_argument("--tag", default="", help="suffix for the CSV name")
    args = ap.parse_args()
    # Every problem (and batch size) is a new shape, so a new compile. That's expected here; the
    # default limit of 8 per function would silently drop later problems to uncompiled mode.
    torch._dynamo.config.recompile_limit = 1024
    # Each measurement also starts from a clean torch.compile cache (torch._dynamo.reset()), so its
    # time includes nothing left over from earlier runs. (Before the compile_fn cache fix, leftover
    # cache entries made every later solve slower; now it's just a precaution.)

    out = ROOT / "results" / f"trn_{instance_type()}_{dt.date.today()}{'_' + args.tag if args.tag else ''}.csv"
    out.parent.mkdir(exist_ok=True)
    f = open(out, "w", newline="")
    w = csv.DictWriter(f, KEYS)
    w.writeheader()

    def record(**row):
        w.writerow(row)
        f.flush()

    miplib = sorted((DATA / "miplib").glob("*.mps.gz"))
    if args.instances:
        miplib = [p for p in miplib if any(k in p.name for k in args.instances)]
    else:
        miplib = [p for p in miplib if p.stat().st_size < 200_000]

    # ---- one LP at a time: microseconds per PDHG iteration ------------------------------------
    if "single" in args.sections:
        print("One LP at a time: microseconds per PDHG iteration (640 fixed iterations)")
        singles = [load_mps(str(DATA / n)) for n in ("israel.mps", "shell.mps", "25fv47.mps")]
        singles += [load_mps(str(p)) for p in miplib]
        print(f"   {'problem':24s}" + "".join(f"{b:>14s}" for b in args.backends), flush=True)
        for lp in singles:
            line = f"   {lp.name[:24]:24s}"
            for b in args.backends:
                be = BACKENDS[b]()
                torch._dynamo.reset()
                try:
                    solve_pdlp(lp, PDLPOptions(tol=1e-12, max_iters=128, backend=be))      # warm-up / compile
                    t = time.perf_counter()
                    r = solve_pdlp(lp, PDLPOptions(tol=1e-12, max_iters=640, backend=be))
                    us = 1e6 * (time.perf_counter() - t) / max(r.iterations, 1)
                    record(section="single", instance=lp.name, rows=lp.m, cols=lp.n, backend=b, B=1,
                           seconds=round(us * r.iterations / 1e6, 4), us_per_iter=round(us, 1))
                    line += f"{us:>14.0f}"
                except Exception as e:
                    traceback.print_exc(limit=3)
                    line += f"{'ERR ' + type(e).__name__[:9]:>14s}"
            print(line, flush=True)

    # ---- batched strong branching: seconds per LP ---------------------------------------------
    if "batch" in args.sections:
        print("\nStrong-branching batches (tol 1e-4): seconds per LP")
        for p in miplib:
            lp = load_mps(str(p))
            root = solve_pdlp(lp, PDLPOptions(tol=1e-4))
            L, U, labels = branching_variants(lp, root.x, args.candidates, tol=1e-3)
            print(f"   {lp.name} ({lp.m} x {lp.n}), {len(labels)} branching LPs", flush=True)
            for b in args.backends:
                be = BACKENDS[b]()
                line = f"     {b:12s}"
                for B in args.batch_sizes:
                    distinct = min(B, len(labels))
                    LB, UB = np.resize(L, (B, lp.n)), np.resize(U, (B, lp.n))     # repeats rows if B > len(labels)
                    torch._dynamo.reset()
                    try:
                        solve_pdlp_batch(lp, B, lower=LB, upper=UB,
                                         opts=PDLPOptions(tol=1e-4, backend=be, max_iters=128))   # warm-up / compile
                        t = time.perf_counter()
                        r = solve_pdlp_batch(lp, B, lower=LB, upper=UB,
                                             opts=PDLPOptions(tol=1e-4, backend=be, time_limit=300))
                        sec, n_opt = time.perf_counter() - t, r.status.count("optimal")
                    except Exception:
                        traceback.print_exc(limit=3)
                        sec, n_opt = float("nan"), 0
                    line += f"   B={B:<3d} {sec / B:8.4f} s ({n_opt}/{B} opt)"
                    record(section="batch", instance=lp.name, rows=lp.m, cols=lp.n, backend=b, B=B,
                           seconds=round(sec, 3), sec_per_lp=round(sec / B, 5), n_optimal=n_opt,
                           distinct_lps=distinct)
                print(line, flush=True)
    f.close()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
