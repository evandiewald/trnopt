"""Bound propagation never cuts off a feasible solution, and catches simple contradictions."""

import numpy as np
import pytest

from trnopt import LP, load_mps
from trnopt.generators import combinatorial_auction, facility_location, independent_set, set_cover
from trnopt.propagate import Propagator
from trnopt.reference import solve_highs_mip
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def small():
    # x0 + x1 >= 1.5 (ints in [0,1])  ->  both must be 1;  x0 + x2 == 1  ->  x2 = 0;  x3 continuous, x3 - x2 >= 0.5
    import scipy.sparse as sp
    K = sp.csr_matrix(np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, -1, 1]], float))
    return LP(c=np.zeros(4), K=K, q=np.array([1, 1.5, 0.5]), m_eq=1, lower=np.zeros(4),
              upper=np.array([1, 1, 1, np.inf]), integer=np.array([True, True, True, False]))


def test_hand_example():
    lp = small()
    lo, up, bad = Propagator(lp).propagate(lp.lower, lp.upper)
    assert not bad
    np.testing.assert_allclose(lo[:3], [1, 1, 0]) and np.testing.assert_allclose(up[:3], [1, 1, 0])
    assert lo[3] == pytest.approx(0.5) and up[3] == np.inf


def test_contradiction():
    lp = small()
    up = lp.upper.copy()
    up[1] = 0                                          # x0 + x1 >= 1.5 with x1 = 0 is impossible
    assert Propagator(lp).propagate(lp.lower, up)[2]


@pytest.mark.parametrize("make", [lambda: set_cover(100, 200, 0.05, seed=1), lambda: independent_set(120, 4, seed=1),
                                  lambda: combinatorial_auction(40, 200, seed=1), lambda: facility_location(15, 10, seed=1),
                                  lambda: load_mps(str(DATA / "miplib" / "neos-911970.mps.gz"))],
                         ids=["setcover", "indset", "cauction", "facility", "neos-911970"])
def test_never_cuts_off_the_optimum(make):
    lp = make()
    ref = solve_highs_mip(lp, 60)
    x = np.asarray(ref.x)
    P = Propagator(lp)
    lo, up, bad = P.propagate(lp.lower, lp.upper)
    assert not bad
    assert np.all(x >= lo - 1e-6) and np.all(x <= up + 1e-6)
    # fix every integer to its optimal value: still no contradiction, and the continuous ones stay reachable
    ints = np.flatnonzero(lp.integer)
    L, U = lp.lower.copy(), lp.upper.copy()
    L[ints] = U[ints] = np.round(x[ints])
    lo, up, bad = P.propagate(L, U)
    assert not bad
    assert np.all(x >= lo - 1e-6) and np.all(x <= up + 1e-6)


@pytest.mark.parametrize("make", [lambda: set_cover(100, 200, 0.05, seed=1), lambda: independent_set(120, 4, seed=1),
                                  lambda: facility_location(15, 10, seed=1),
                                  lambda: load_mps(str(DATA / "miplib" / "neos-911970.mps.gz"))],
                         ids=["setcover", "indset", "facility", "neos-911970"])
@pytest.mark.parametrize("backend", ["cpu", "cpu-float32"])
def test_batched_and_dense_rounding_match(make, backend):
    """Propagator.round_batch == the per-point loop, and the dense tensor version (propagate_torch) agrees."""
    import dataclasses
    from trnopt.propagate_torch import TorchPropagator
    from trnopt.reference import solve_highs
    lp = make()
    x = solve_highs(dataclasses.replace(lp, integer=None)).x
    rng = np.random.default_rng(0)
    ints = np.flatnonzero(lp.integer)
    X = np.repeat(x[None], 6, 0)
    X[1:, ints] += rng.uniform(-0.3, 0.3, (5, len(ints)))
    P = Propagator(lp)
    loop = np.stack([P.round(X[p], lp.lower, lp.upper)[0] for p in range(6)])
    Xb = P.round_batch(X, lp.lower, lp.upper)[0]
    np.testing.assert_array_equal(loop, Xb)
    Xt = TorchPropagator(lp, backend).round_batch(X, lp.lower, lp.upper)[0]
    np.testing.assert_array_equal(Xb[:, ints], Xt[:, ints])
