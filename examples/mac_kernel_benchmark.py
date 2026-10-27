"""Does the sparse Metal kernel make the Mac GPU worth it? Run in macOS's own Terminal:

    uv run python examples/mac_kernel_benchmark.py            # ~10-20 minutes
    uv run python examples/mac_kernel_benchmark.py --quick    # kernel correctness + multiply timings only

Three parts:
  1. The Metal kernel against SciPy (correctness).
  2. The multiply alone (K @ X and K^T @ Y, the work in every PDHG iteration): CPU sparse vs
     GPU full grid vs GPU with our kernel, at batch sizes 1, 32 and 128.
  3. Full solves of strong-branching batches: 128 LPs per instance, CPU vs GPU, against the best
     of HiGHS's three LP methods (simplex, interior point, and its own CPU PDLP), one LP at a time.

All PDLP runs use float32 (the GPU has no float64) and tolerance 1e-4.
Writes results/mac_kernel_<chip>_<date>.csv.
"""

import argparse
import csv
import dataclasses
import datetime as dt
import subprocess
import time
from pathlib import Path

import highspy
import numpy as np
import torch

from trnopt import Backend, MatVec, PDLPOptions, load_mps, solve_highs, solve_pdlp
from trnopt.batch import solve_pdlp_stream
from trnopt.metal_sell import MetalSELL
from trnopt.problems import branching_variants
from trnopt.reference import to_highs_lp

ROOT = Path(__file__).resolve().parent.parent
MIPLIB = ROOT / "data" / "miplib"


def sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def highs_seconds(lp, solver):
    """Seconds for HiGHS to solve lp from scratch with one of its LP methods, plus its status and objective."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", solver)
    h.setOptionValue("time_limit", 120.0)
    h.passModel(to_highs_lp(lp))
    t = time.perf_counter()
    h.run()
    return time.perf_counter() - t, h.modelStatusToString(h.getModelStatus()), h.getInfo().objective_function_value


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def time_multiply(mv: MatVec, n, m, B, device, reps):
    x = torch.randn(n, B, dtype=torch.float32).to(device)
    y = torch.randn(m, B, dtype=torch.float32).to(device)
    for _ in range(3):                     # warm-up (also compiles the Metal kernel once)
        mv.K(x), mv.KT(y)
    sync(device)
    t = time.perf_counter()
    for _ in range(reps):
        mv.K(x), mv.KT(y)
    sync(device)
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    chip = sysctl("machdep.cpu.brand_string")
    print(f"chip: {chip}   memory: {int(sysctl('hw.memsize') or 0) / 2**30:.0f} GB   torch {torch.__version__}")
    if not torch.backends.mps.is_available():
        print("No MPS device: run this in macOS's own Terminal on an Apple silicon Mac.")
        return
    instances = sorted(MIPLIB.glob("*.mps.gz"), key=lambda p: p.stat().st_size)
    rows = []

    # ---- 1. kernel correctness ---------------------------------------------------------------
    print("\n1. Metal kernel vs SciPy (max relative error)")
    for p in instances:
        K = load_mps(str(p)).K.tocsr()
        errs = []
        for M in (K, K.T.tocsr()):
            S = MetalSELL(M, device="mps")
            for B in (1, 32, 128):
                X = np.random.default_rng(B).normal(size=(M.shape[1], B))
                got = S.matmul(torch.as_tensor(X, dtype=torch.float32).to("mps")).cpu().numpy()
                ref = M @ X
                errs.append(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
        worst = max(errs)
        print(f"   {p.name[:30]:30s} {K.shape[0]:>7} x {K.shape[1]:<7} nnz {K.nnz:>8}  "
              f"stored/nnz {S.stored_entries / M.nnz:4.2f}  worst err {worst:.1e}  {'ok' if worst < 1e-4 else 'FAIL'}")

    # ---- 2. the multiply alone --------------------------------------------------------------
    print("\n2. One PDHG iteration's multiplies (K@X + K^T@Y), milliseconds")
    print(f"   {'instance':30s}{'B':>5}{'CPU sparse':>12}{'GPU grid':>11}{'GPU kernel':>12}{'kernel vs CPU':>15}")
    for p in instances:
        lp = load_mps(str(p))
        K = lp.K.tocsr()
        mvs = {"CPU sparse": (MatVec(K, Backend.preset("cpu-float32")), "cpu"),
               "GPU kernel": (MatVec(K, Backend.preset("mps-metal")), "mps")}
        try:
            mvs["GPU grid"] = (MatVec(K, Backend.preset("mps")), "mps")
        except ValueError:
            pass                                               # too big to store as a full grid
        for B in (1, 32, 128):
            t = {}
            for name, (mv, dev) in mvs.items():
                reps = 20 if B > 1 else 50
                t[name] = time_multiply(mv, lp.n, lp.m, B, dev, reps)
            grid = f"{1e3 * t['GPU grid']:11.3f}" if "GPU grid" in t else f"{'too big':>11}"
            print(f"   {p.name[:30]:30s}{B:>5}{1e3 * t['CPU sparse']:12.3f}{grid}{1e3 * t['GPU kernel']:12.3f}"
                  f"{t['CPU sparse'] / t['GPU kernel']:14.1f}x")
            for name, sec in t.items():
                rows.append(dict(section="multiply", instance=p.name, rows=lp.m, cols=lp.n, nnz=K.nnz,
                                 B=B, method=name, seconds=sec))
    if args.quick:
        _write(rows, chip)
        return

    # ---- 3. full solves: 128 strong-branching LPs per instance ------------------------------
    print("\n3. Strong branching, 128 LPs per instance (float32, tol 1e-4): seconds per LP")
    print(f"   {'instance':30s}{'best HiGHS*':>13}{'CPU B=1':>9}{'CPU B=128':>11}{'GPU kernel B=128':>18}"
          f"{'GPU vs CPU':>12}{'GPU vs HiGHS':>14}")
    for p in instances:
        lp = load_mps(str(p))
        root = solve_pdlp(lp, PDLPOptions(tol=1e-4, time_limit=120))
        L, U, labels = branching_variants(lp, root.x, 64, tol=1e-3)
        N = len(labels)
        if N < 8:
            print(f"   {p.name[:30]:30s} only {N} fractional variables; skipped")
            continue
        # HiGHS from scratch on 4 of the LPs with each of its methods; keep the best method's time.
        # (Inside a real MIP search HiGHS would warm-start dual simplex, which is faster still.)
        refs = [solve_highs(dataclasses.replace(lp, lower=L[i], upper=U[i])) for i in range(8)]
        per_method = {}
        for solver in ("simplex", "ipm", "pdlp"):
            per_method[solver] = np.mean([highs_seconds(dataclasses.replace(lp, lower=L[i], upper=U[i]), solver)[0]
                                          for i in range(4)])
        best_method = min(per_method, key=per_method.get)
        highs = per_method[best_method]
        res = {}
        for name, preset, slots, n_lps in (("CPU B=1", "cpu-float32", 1, 8),
                                           ("CPU B=128", "cpu-float32", 128, N),
                                           ("GPU kernel B=128", "mps-metal", 128, N)):
            opts = PDLPOptions(tol=1e-4, backend=Backend.preset(preset), time_limit=600)
            solve_pdlp_stream(lp, min(slots, n_lps), slots, lower=L[:slots], upper=U[:slots],
                              opts=dataclasses.replace(opts, max_iters=128))         # warm-up
            t = time.perf_counter()
            r = solve_pdlp_stream(lp, n_lps, slots, lower=L[:n_lps], upper=U[:n_lps], opts=opts)
            sec = (time.perf_counter() - t) / n_lps
            errs = [abs(r.objective[i] - refs[i].objective) / (1 + abs(refs[i].objective))
                    for i in range(min(8, n_lps)) if refs[i].status == "Optimal" and r.status[i] == "optimal"]
            res[name] = (sec, r.status.count("optimal"), n_lps, max(errs) if errs else float("nan"))
            rows.append(dict(section="branching", instance=p.name, rows=lp.m, cols=lp.n, nnz=lp.K.nnz, B=slots,
                             method=name, seconds=sec, n_optimal=res[name][1], n_lps=n_lps, obj_err=res[name][3]))
        for solver, sec in per_method.items():
            rows.append(dict(section="branching", instance=p.name, rows=lp.m, cols=lp.n, nnz=lp.K.nnz, B=1,
                             method=f"HiGHS {solver}", seconds=sec, n_lps=4))
        c, g = res["CPU B=128"], res["GPU kernel B=128"]
        print(f"   {p.name[:30]:30s}{highs:9.4f} {best_method:>3s}{res['CPU B=1'][0]:9.4f}{c[0]:11.4f}{g[0]:18.4f}"
              f"{c[0] / g[0]:11.1f}x{highs / g[0]:13.1f}x"
              f"   (optimal: CPU {c[1]}/{c[2]}, GPU {g[1]}/{g[2]}; worst obj err vs HiGHS {g[3]:.0e})", flush=True)
    print("\n   * Best of HiGHS simplex / interior point / PDLP, each LP solved from scratch. Inside a real MIP")
    print("     search HiGHS would warm-start dual simplex from the parent LP, which is typically faster still.")
    print("   Objective errors above ~1e-3 mean the 1e-4 stopping test was too loose for that instance:")
    print("   treat its timings as not comparable.")
    _write(rows, chip)


def _write(rows, chip):
    out = ROOT / "results" / f"mac_kernel_{chip.replace(' ', '_')}_{dt.date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
