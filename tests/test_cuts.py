"""GMI cuts: never cut off an integer-feasible point (the MIP optimum), and raise the root bound."""

from pathlib import Path

import numpy as np
import pytest

from trnopt import load_mps
from trnopt.cuts import root_cuts
from trnopt.generators import combinatorial_auction, facility_location, independent_set, set_cover
from trnopt.reference import solve_highs_mip

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.parametrize("make", [lambda: set_cover(100, 200, 0.05, seed=1), lambda: independent_set(150, 4, seed=1),
                                  lambda: combinatorial_auction(50, 250, seed=1), lambda: facility_location(15, 10, seed=1),
                                  lambda: load_mps(str(DATA / "miplib" / "neos-911970.mps.gz")),
                                  lambda: load_mps(str(DATA / "miplib" / "nu25-pr12.mps.gz"))],
                         ids=["setcover", "indset", "cauction", "facility", "neos-911970", "nu25-pr12"])
def test_cuts_valid_and_tighten(make):
    lp = make()
    ref = solve_highs_mip(lp, 120)
    assert ref.status == "Optimal"
    out, info = root_cuts(lp, rounds=5, max_cuts=100)
    assert out.n == lp.n and out.m == lp.m + info.kept
    x = np.asarray(ref.x)
    G, D = out.K[lp.m:], out.q[lp.m:]
    if info.kept:
        slack = G @ x - D
        assert slack.min() >= -1e-6 * (1 + np.abs(D).max()), f"a cut removes the MIP optimum ({slack.min():.3g})"
    assert info.bound_after >= info.bound_before - 1e-9 * (1 + abs(info.bound_before))
    assert info.bound_after <= ref.objective + 1e-6 * (1 + abs(ref.objective))


@pytest.mark.parametrize("make", [lambda: independent_set(150, 4, seed=2), lambda: combinatorial_auction(50, 250, seed=2),
                                  lambda: facility_location(15, 10, seed=2)], ids=["indset", "cauction", "facility"])
def test_mip_with_cuts_same_optimum(make):
    from trnopt.mip import solve_mip
    lp = make()
    ref = solve_highs_mip(lp, 120)
    r = solve_mip(lp, "reliability_trnopt", time_limit=120, cuts=5, presolve=True, pump=8)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(ref.objective, rel=1e-6, abs=1e-6)
    assert r.x is not None and r.x.shape == (lp.n,)
