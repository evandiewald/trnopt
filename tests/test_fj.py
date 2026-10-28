"""Feasibility Jump returns only genuinely feasible points, and finds one on standard test families."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")

from trnopt import load_mps
from trnopt.fj import feasibility_jump
from trnopt.generators import combinatorial_auction, facility_location, independent_set, set_cover

DATA = Path(__file__).resolve().parent.parent / "data"


def is_feasible(lp, x, tol=1e-6):
    Kx = lp.K @ x
    eq = np.arange(lp.m) < lp.m_eq
    viol = np.where(eq, np.abs(Kx - lp.q), np.maximum(lp.q - Kx, 0)) - tol * (1 + np.abs(lp.q))
    ints = np.asarray(lp.integer, bool)
    return bool(viol.max(initial=-1) <= 0 and np.all(x >= lp.lower - 1e-9) and np.all(x <= lp.upper + 1e-9)
                and np.all(np.abs(x[ints] - np.round(x[ints])) < 1e-9))


@pytest.mark.parametrize("make", [lambda: set_cover(100, 200, 0.05, seed=1), lambda: independent_set(120, 4, seed=1),
                                  lambda: combinatorial_auction(40, 200, seed=1), lambda: facility_location(15, 10, seed=1),
                                  lambda: load_mps(str(DATA / "miplib" / "neos-911970.mps.gz")),
                                  lambda: load_mps(str(DATA / "miplib" / "nu25-pr12.mps.gz"))],
                         ids=["setcover", "indset", "cauction", "facility", "neos-911970", "nu25-pr12"])
def test_finds_feasible_points(make):
    lp = make()
    r = feasibility_jump(lp, time_limit=10)
    assert r.solutions, "no feasible point found"
    for obj, x in r.solutions:
        assert is_feasible(lp, x)
        assert obj == pytest.approx(lp.c @ x)
    objs = [o for o, _ in r.solutions]
    assert objs == sorted(objs)


def test_respects_time_limit():
    lp = load_mps(str(DATA / "miplib" / "neos-933966.mps.gz"))
    r = feasibility_jump(lp, time_limit=3)
    assert r.seconds < 4.5


@pytest.mark.parametrize("make", [lambda: set_cover(60, 120, 0.08, seed=4), lambda: independent_set(80, 4, seed=4),
                                  lambda: facility_location(8, 6, seed=4)], ids=["setcover", "indset", "facility"])
def test_solver_with_fj_and_rins_reaches_the_optimum(make):
    from trnopt.mip import solve_mip
    from trnopt.reference import solve_highs_mip
    lp = make()
    ref = solve_highs_mip(lp, 60)
    r = solve_mip(lp, "reliability_trnopt", time_limit=120, presolve=True, fj=True, rins=True, pump=8, cuts=2)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(ref.objective, rel=1e-6, abs=1e-6)
    assert is_feasible(lp, r.x)
