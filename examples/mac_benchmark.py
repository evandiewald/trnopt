"""trnopt on Apple silicon: correctness and speed on the Mac's GPU (PyTorch "mps" backend).

    uv run python examples/mac_benchmark.py            # full run, a few minutes
    uv run python examples/mac_benchmark.py --quick    # correctness checks only

Writes results/mac_<chip>_<date>.csv and prints a summary. Run it in macOS's own Terminal:
the GPU isn't visible from Linux VMs or containers.
"""

import argparse
import csv
import datetime as dt
import platform
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from trnopt import Backend, MatVec, PDLPOptions, load_mps, solve_highs, solve_pdlp
from trnopt.batch import solve_pdlp_batch
from trnopt.problems import branching_variants, loan_selection, textbook

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def rel_err(a, b):
    return abs(a - b) / (1 + abs(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="correctness checks only")
    args = ap.parse_args()

    chip = sysctl("machdep.cpu.brand_string") or platform.processor()
    mem_gb = int(sysctl("hw.memsize") or 0) / 2**30
    print(f"chip: {chip}   memory: {mem_gb:.0f} GB   torch {torch.__version__}   "
          f"MPS available: {torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_available():
        print("No MPS device: run this in macOS's own Terminal on an Apple silicon Mac.")
        return
    rows = []

    # ---- 1. correctness on the GPU, against HiGHS --------------------------------------------
    print("\n1. Correctness on the Mac GPU (float32, tol 1e-4) vs HiGHS")
    probs = [textbook(), load_mps(str(DATA / "afiro.mps")), load_mps(str(DATA / "israel.mps")),
             load_mps(str(DATA / "shell.mps")), loan_selection()]
    ok_all = True
    for lp in probs:
        ref = solve_highs(lp)
        res = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("mps")))
        err = rel_err(res.objective, ref.objective)
        ok = res.status == "optimal" and err < 1e-3
        ok_all &= ok
        print(f"   {lp.name[:22]:22s} {res.status:10s} obj err {err:.1e}  {'ok' if ok else 'FAIL'}")
    lp = load_mps(str(DATA / "shell.mps"))
    ref = solve_highs(lp)
    res = solve_pdlp(lp, PDLPOptions(tol=1e-6, backend=Backend.preset("mps"), finish_float64=True))
    ok = res.status == "optimal" and rel_err(res.objective, ref.objective) < 1e-5
    ok_all &= ok
    print(f"   shell, 1e-6 with float64 finish: {res.status}, {res.iterations - res.finish_iterations} GPU + "
          f"{res.finish_iterations} CPU iterations  {'ok' if ok else 'FAIL'}")

    # ---- 2. does PyTorch's sparse multiply work on MPS? -----------------------------------------
    print("\n2. PyTorch sparse (CSR) multiply on MPS")
    sparse_ok = False
    try:
        K = sp.random(300, 500, density=0.02, random_state=0, format="csr")
        x = np.random.default_rng(0).normal(size=(500, 4))
        mv = MatVec(K, Backend.preset("mps-sparse"))
        got = mv.K(torch.as_tensor(x, dtype=torch.float32).to("mps")).cpu().numpy()
        sparse_ok = np.abs(got - K @ x).max() < 1e-3
        print(f"   {'works' if sparse_ok else 'runs but gives wrong answers'}")
    except Exception as e:
        print(f"   not supported: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
    if args.quick:
        print("\nAll correctness checks passed." if ok_all else "\nSome correctness checks FAILED.")
        return

    backends = ["cpu", "cpu-float32", "mps"] + (["mps-sparse"] if sparse_ok else [])

    # ---- 3. single-LP speed: microseconds per iteration ---------------------------------------
    print("\n3. One LP at a time: microseconds per PDHG iteration (640 fixed iterations)")
    singles = [load_mps(str(DATA / f)) for f in ("israel.mps", "shell.mps", "25fv47.mps")]
    # The small instances only: the larger showcase ones are for examples/mac_kernel_benchmark.py.
    miplib = [p for p in sorted((DATA / "miplib").glob("*.mps.gz")) if p.stat().st_size < 200_000]
    singles += [load_mps(str(p)) for p in miplib]
    print(f"   {'problem':24s}" + "".join(f"{b:>14s}" for b in backends))
    for lp in singles:
        line = f"   {lp.name[:24]:24s}"
        for b in backends:
            opts = PDLPOptions(tol=1e-12, max_iters=640, backend=Backend.preset(b))
            solve_pdlp(lp, PDLPOptions(tol=1e-12, max_iters=128, backend=Backend.preset(b)))   # warm-up
            t = time.perf_counter()
            r = solve_pdlp(lp, opts)
            us = 1e6 * (time.perf_counter() - t) / max(r.iterations, 1)
            line += f"{us:>14.0f}"
            rows.append(dict(section="single", instance=lp.name, rows=lp.m, cols=lp.n, backend=b, B=1,
                             seconds=round(us * r.iterations / 1e6, 4), us_per_iter=round(us, 1)))
        print(line)

    # ---- 4. batched strong branching: seconds per LP -------------------------------------------
    print("\n4. Strong-branching batches (tol 1e-4): seconds per LP")
    for p in miplib:
        lp = load_mps(str(p))
        root = solve_pdlp(lp, PDLPOptions(tol=1e-4))
        L, U, labels = branching_variants(lp, root.x, 64, tol=1e-3)
        print(f"   {lp.name} ({lp.m} x {lp.n}), {len(labels)} branching LPs")
        for b in backends:
            line = f"     {b:12s}"
            for B in (1, 32, 128):
                if B > len(labels):
                    continue
                opts = PDLPOptions(tol=1e-4, backend=Backend.preset(b), time_limit=300)
                solve_pdlp_batch(lp, B, lower=L[:B], upper=U[:B],
                                 opts=PDLPOptions(tol=1e-4, backend=Backend.preset(b), max_iters=128))  # warm-up
                t = time.perf_counter()
                try:
                    r = solve_pdlp_batch(lp, B, lower=L[:B], upper=U[:B], opts=opts)
                    sec = time.perf_counter() - t
                    n_opt = r.status.count("optimal")
                except Exception:
                    traceback.print_exc()
                    sec, n_opt = float("nan"), 0
                line += f"   B={B:<3d} {sec / B:8.4f} s ({n_opt}/{B} opt)"
                rows.append(dict(section="batch", instance=lp.name, rows=lp.m, cols=lp.n, backend=b, B=B,
                                 seconds=round(sec, 3), sec_per_lp=round(sec / B, 5), n_optimal=n_opt))
            print(line, flush=True)

    out = ROOT / "results" / f"mac_{chip.replace(' ', '_')}_{dt.date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")
    print("All correctness checks passed." if ok_all else "Some correctness checks FAILED.")


if __name__ == "__main__":
    main()
