"""Batched feasibility pump: finds genuinely feasible points; switching it on doesn't change the MIP answer."""

import dataclasses

import numpy as np
import pytest

from trnopt import solve_highs
from trnopt.generators import combinatorial_auction, facility_location, set_cover
from trnopt.mip import _NodeLP, solve_mip
from trnopt.pump import feasibility_pump
from trnopt.reference import solve_highs_mip


def check_feasible(lp, x):
    ints = np.flatnonzero(lp.integer)
    assert np.all(np.abs(x[ints] - np.round(x[ints])) <= 1e-6)
    assert np.all(x >= lp.lower - 1e-6) and np.all(x <= lp.upper + 1e-6)
    Kx = lp.K @ x
    assert np.all(np.abs(Kx[:lp.m_eq] - lp.q[:lp.m_eq]) <= 1e-6 * (1 + np.abs(lp.q[:lp.m_eq])))
    assert np.all(Kx[lp.m_eq:] >= lp.q[lp.m_eq:] - 1e-6 * (1 + np.abs(lp.q[lp.m_eq:])))


def repair_for(lp):
    core, ints = _NodeLP(lp), np.flatnonzero(lp.integer)

    def repair(r):
        L, U = lp.lower.copy(), lp.upper.copy()
        L[ints] = U[ints] = r
        st, obj, x, _, _ = core.solve(L, U, None)
        return (st, obj, x) if st == "optimal" else None
    return repair


@pytest.mark.parametrize("make", [lambda: set_cover(150, 300, 0.05, seed=5),          # pure binary
                                  lambda: facility_location(20, 15, seed=5),          # binaries + continuous
                                  lambda: combinatorial_auction(60, 300, seed=5)],
                         ids=["setcover", "facility", "cauction"])
def test_pump_finds_feasible_points(make):
    lp = make()
    root = solve_highs(dataclasses.replace(lp, integer=None))
    pr = feasibility_pump(lp, root.x, pumps=8, rounds=30, time_limit=60, repair=repair_for(lp), stop_after=None)
    assert pr.solutions, "no feasible point found"
    for obj, x in pr.solutions:
        check_feasible(lp, x)
        assert obj == pytest.approx(lp.c @ x)
        assert obj >= root.objective - 1e-6 * (1 + abs(root.objective))     # can't beat the LP bound
    assert [o for o, _ in pr.solutions] == sorted(o for o, _ in pr.solutions)


def test_mip_with_pump_same_optimum():
    lp = facility_location(25, 20, seed=6)
    ref = solve_highs_mip(lp, 120)
    r = solve_mip(lp, "pseudocost", time_limit=120, pump=8)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(ref.objective, rel=1e-6)
    assert r.counts.get("pump_rounds", 0) > 0
