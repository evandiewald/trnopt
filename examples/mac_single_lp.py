"""One big LP at a time: does the Metal kernel make the Mac GPU beat the CPU (and HiGHS)?

    uv run python examples/mac_single_lp.py        # ~5 minutes

For each of the larger MIPLIB LP relaxations:
  a) time per PDHG iteration (fixed 1,280 iterations): CPU float32 sparse vs GPU + Metal kernel
  b) a full solve to 1e-4 on each, checked against HiGHS, vs the best of HiGHS's three LP methods
Writes results/mac_single_<chip>_<date>.csv.
"""

import csv
import datetime as dt
import subprocess
import time
from pathlib import Path

import highspy
import numpy as np
import torch

from trnopt import Backend, PDLPOptions, load_mps, solve_highs, solve_pdlp
from trnopt.reference import to_highs_lp

ROOT = Path(__file__).resolve().parent.parent
MIPLIB = ROOT / "data" / "miplib"
BACKENDS = ("cpu-float32", "mps-metal")


def sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def highs_seconds(lp, solver):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", solver)
    h.setOptionValue("time_limit", 120.0)
    h.passModel(to_highs_lp(lp))
    t = time.perf_counter()
    h.run()
    return time.perf_counter() - t


def main():
    chip = sysctl("machdep.cpu.brand_string")
    print(f"chip: {chip}   torch {torch.__version__}")
    if not torch.backends.mps.is_available():
        print("No MPS device: run this in macOS's own Terminal.")
        return
    files = [p for p in sorted(MIPLIB.glob("*.mps.gz"), key=lambda p: p.stat().st_size)
             if p.stat().st_size >= 200_000]
    rows = []

    print("\na) Microseconds per PDHG iteration, one LP (1,280 fixed iterations)")
    print(f"   {'instance':28s}{'nnz':>9}{'CPU':>10}{'GPU kernel':>12}{'GPU vs CPU':>12}")
    lps = {}
    for p in files:
        lp = load_mps(str(p))
        lps[p.name] = lp
        us = {}
        for b in BACKENDS:
            solve_pdlp(lp, PDLPOptions(tol=1e-12, max_iters=128, backend=Backend.preset(b)))   # warm-up
            t = time.perf_counter()
            r = solve_pdlp(lp, PDLPOptions(tol=1e-12, max_iters=1280, backend=Backend.preset(b)))
            us[b] = 1e6 * (time.perf_counter() - t) / max(r.iterations, 1)
            rows.append(dict(section="per_iter", instance=p.name, nnz=lp.K.nnz, backend=b, us_per_iter=us[b]))
        print(f"   {p.name[:28]:28s}{lp.K.nnz:>9}{us['cpu-float32']:>10.0f}{us['mps-metal']:>12.0f}"
              f"{us['cpu-float32'] / us['mps-metal']:>11.1f}x", flush=True)

    print("\nb) Full solve to 1e-4, seconds (objective error vs HiGHS in brackets)")
    print(f"   {'instance':28s}{'HiGHS best':>16}{'CPU':>18}{'GPU kernel':>18}{'GPU vs HiGHS':>14}")
    for name, lp in lps.items():
        ref = solve_highs(lp)
        hs = {s: highs_seconds(lp, s) for s in ("simplex", "ipm", "pdlp")}
        best = min(hs, key=hs.get)
        line = f"   {name[:28]:28s}{hs[best]:>9.2f} {best:>6s}"
        sec = {}
        for b in BACKENDS:
            t = time.perf_counter()
            r = solve_pdlp(lp, PDLPOptions(tol=1e-4, time_limit=120, backend=Backend.preset(b)))
            sec[b] = time.perf_counter() - t
            err = abs(r.objective - ref.objective) / (1 + abs(ref.objective))
            tag = "" if r.status == "optimal" else f" {r.status}"
            line += f"{sec[b]:>9.2f} [{err:.0e}]{tag}"
            rows.append(dict(section="solve", instance=name, backend=b, seconds=sec[b], status=r.status,
                             iterations=r.iterations, obj_err=err))
        for s, v in hs.items():
            rows.append(dict(section="solve", instance=name, backend=f"HiGHS {s}", seconds=v))
        print(line + f"{hs[best] / sec['mps-metal']:>13.1f}x", flush=True)

    out = ROOT / "results" / f"mac_single_{chip.replace(' ', '_')}_{dt.date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
