"""Propagation rounding (FP 2.0's rounding step) on the CPU vs on an accelerator, plus the feasibility pump per
backend.

    python examples/propagation_chip_timing.py --backends cpu-float32 neuron neuron-eager

For each instance: one round of batched propagation rounding for 16 points (the pump's per-round work), with
propagate.py (sparse, numpy) and propagate_torch.py (dense tensor ops, on each --backend). First call timed
separately (compilation), then the median of 3. Reports whether the roundings agree with numpy's.
Then (--pump) a standalone 16-pump feasibility pump on each backend: rounds, solutions, best, seconds.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import time
from pathlib import Path

import numpy as np

from trnopt import Backend, load_mps
from trnopt.generators import combinatorial_auction, facility_location, independent_set, set_cover
from trnopt.propagate import Propagator
from trnopt.reference import solve_highs

ROOT = Path(__file__).resolve().parent.parent


def backend(name: str) -> Backend:
    if name == "neuron-eager":
        be = Backend.preset("neuron")
        return dataclasses.replace(be, compile=False) if dataclasses.is_dataclass(be) else be
    return Backend.preset(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="+", default=["cpu-float32"])
    ap.add_argument("--pump", action="store_true")
    args = ap.parse_args()
    lps = [set_cover(200, 400, 0.05, seed=2), independent_set(300, 4, seed=2), combinatorial_auction(80, 400, seed=2),
           facility_location(40, 30, seed=2)]
    lps += [load_mps(str(ROOT / "data" / "miplib" / f"{n}.mps.gz")) for n in ["neos-911970", "neos-4954672-berkel",
                                                                             "nu25-pr12"]]
    from trnopt.propagate_torch import TorchPropagator
    for lp in lps:
        x = solve_highs(dataclasses.replace(lp, integer=None)).x
        rng = np.random.default_rng(0)
        ints = np.flatnonzero(lp.integer)
        X = np.repeat(x[None], 16, 0)
        X[1:, ints] += rng.uniform(-0.3, 0.3, (15, len(ints)))
        pr = Propagator(lp)
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            ref = pr.round_batch(X, lp.lower, lp.upper)[0]
            ts.append(time.perf_counter() - t)
        print(f"\n{lp.name}: {lp.m} x {lp.n}, {lp.K.nnz} nnz, dense {(lp.m + lp.m_eq) * lp.n / 1e6:.1f}M entries", flush=True)
        print(f"   numpy sparse            {statistics.median(ts):8.3f} s / round", flush=True)
        if not TorchPropagator.fits(lp):
            print("   (too large for the dense version)")
            continue
        for name in args.backends:
            try:
                tp = TorchPropagator(lp, backend(name))
                t = time.perf_counter()
                tp.round_batch(X, lp.lower, lp.upper)
                first = time.perf_counter() - t
                ts = []
                for _ in range(3):
                    t = time.perf_counter()
                    out = tp.round_batch(X, lp.lower, lp.upper)[0]
                    ts.append(time.perf_counter() - t)
                same = np.mean([np.array_equal(ref[p, ints], out[p, ints]) for p in range(16)])
                print(f"   dense on {name:14s} {statistics.median(ts):8.3f} s / round  (first call {first:.1f} s)  "
                      f"same roundings {same:.0%}", flush=True)
            except Exception as e:
                print(f"   dense on {name:14s} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
    if args.pump:
        from trnopt.pump import feasibility_pump
        for lp in lps[4:]:
            x = solve_highs(dataclasses.replace(lp, integer=None)).x
            for name in args.backends:
                if name == "neuron-eager":
                    continue
                for rep in range(2):
                    r = feasibility_pump(lp, x, backend=backend(name), pumps=16, rounds=40, time_limit=60, stop_after=None)
                    best = f"{r.solutions[0][0]:.6g}" if r.solutions else "-"
                    print(f"pump {lp.name[:20]:20s} {name:12s} run {rep + 1}: {r.seconds:5.1f} s, {r.rounds} rounds, "
                          f"{len(r.solutions)} solutions, best {best}", flush=True)


if __name__ == "__main__":
    main()
