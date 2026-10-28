"""The NKI sliced-ELL kernel's table copies (nki_sell.TABLE_COLS): one group, several groups, and slices too
wide for any group, which copy their tables chunk by chunk."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from trnopt import Backend
from trnopt.backend import MatVec
from trnopt.sell import build_sell
from test_pdlp import needs_neuron


def _matrix():
    # Mostly short rows plus a few long ones: with a small TABLE_COLS some slices group together and the
    # widest ones don't fit any group.
    rng = np.random.default_rng(3)
    K = sp.random(700, 900, density=0.004, random_state=4, format="lil")
    for r in rng.choice(700, 5, replace=False):
        K[r, rng.choice(900, 60, replace=False)] = rng.normal(size=60)
    return K.tocsr()


@pytest.mark.parametrize("table_cols", [4096, 16, 4])
def test_table_groups_in_simulator(monkeypatch, table_cols):
    pytest.importorskip("nki")
    from trnopt import nki_sell
    monkeypatch.setattr(nki_sell, "TABLE_COLS", table_cols)
    K = _matrix()
    X = np.random.default_rng(0).normal(size=(900, 4)).astype(np.float32)
    Y = nki_sell.sell_matmul_simulated(build_sell(K), X)
    assert np.abs(Y - K @ X).max() <= 1e-5 * np.abs(K @ X).max()


@needs_neuron
@pytest.mark.parametrize("table_cols", [4096, 16, 4])
@pytest.mark.parametrize("spmv", ["nki", "nki-op"])
def test_table_groups_on_device(monkeypatch, table_cols, spmv):
    from trnopt import nki_sell
    monkeypatch.setattr(nki_sell, "TABLE_COLS", table_cols)
    K = _matrix()
    mv = MatVec(K, Backend.preset("neuron", spmv=spmv))
    X = np.random.default_rng(0).normal(size=(900, 16))
    Y = mv.K(torch.as_tensor(X, dtype=torch.float32).to("neuron")).cpu().numpy()
    YT = mv.KT(torch.as_tensor(K @ X, dtype=torch.float32).to("neuron")).cpu().numpy()
    assert np.abs(Y - K @ X).max() <= 1e-5 * np.abs(K @ X).max()
    assert np.abs(YT - K.T @ (K @ X)).max() <= 1e-5 * np.abs(K.T @ (K @ X)).max()


# ---- ordered mode: block writes instead of scatters (nki_sell.output_order, MatVec permute=True) ----------

def _shell():
    from pathlib import Path
    from trnopt import load_mps
    return load_mps(str(Path(__file__).resolve().parent.parent / "data" / "shell.mps")).K.tocsr()  # split rows


@pytest.mark.parametrize("which", ["random", "shell", "shell-T", "mixed"])
@pytest.mark.parametrize("table_cols", [4096, 4])
def test_ordered_kernel_in_simulator(monkeypatch, which, table_cols):
    """The ordered kernel's output is K @ X with rows in output_order, X's rows renumbered by col_pos."""
    nki = pytest.importorskip("nki")
    from trnopt import nki_sell
    monkeypatch.setattr(nki_sell, "TABLE_COLS", table_cols)
    K = {"random": lambda: sp.random(300, 500, density=0.02, random_state=1, format="csr"),
         "shell": _shell, "shell-T": lambda: _shell().T.tocsr(), "mixed": _matrix}[which]()
    S = build_sell(K)
    ypos, _ = nki_sell.output_order(S)
    xpos = np.random.default_rng(1).permutation(K.shape[1])             # any renumbering of X's rows
    X = np.random.default_rng(0).normal(size=(K.shape[1], 4)).astype(np.float32)
    Xp = np.empty_like(X)
    Xp[xpos] = X
    kernel = nki_sell.make_sell_kernel(S, 4, ordered=True)
    Yp = np.asarray(nki.simulate(kernel)(*nki_sell.kernel_tables(S, col_pos=xpos, ordered=True), Xp))
    ref = K @ X
    assert np.abs(Yp[ypos] - ref).max() <= 1e-5 * np.abs(ref).max()


def test_output_order_is_a_permutation_with_blocks_first():
    from trnopt.nki_sell import output_order, scatter_passes
    for K in (_shell(), _shell().T.tocsr(), _matrix()):
        S = build_sell(K)
        pos, block = output_order(S)
        assert sorted(pos) == list(range(K.shape[0]))
        passes = scatter_passes(S, (pos, block))[1]
        assert all((p == 0) == b for p, b in zip(passes, block))
        assert sum(passes) < sum(scatter_passes(S)[1])                  # fewer scatters than before


def test_permuted_matvec_in_simulator():
    pytest.importorskip("nki")
    K = _shell()
    mv = MatVec(K, Backend.preset("cpu-float32", spmv="nki-sim"), permute=True)
    x = np.random.default_rng(0).normal(size=(K.shape[1], 3))
    y = np.random.default_rng(1).normal(size=(K.shape[0], 3))
    Kx = mv.K(torch.as_tensor(x[mv.col_perm], dtype=torch.float32)).numpy()
    KTy = mv.KT(torch.as_tensor(y[mv.row_perm], dtype=torch.float32)).numpy()
    assert np.abs(Kx - (K @ x)[mv.row_perm]).max() <= 1e-5 * np.abs(K @ x).max()
    assert np.abs(KTy - (K.T @ y)[mv.col_perm]).max() <= 1e-5 * np.abs(K.T @ y).max()


def _batch_check(backend, lp):
    """A batched solve through the permuted kernel: same answers as the plain CPU path, in the LP's own order."""
    from trnopt import PDLPOptions
    from trnopt.batch import solve_pdlp_batch
    U = np.maximum(lp.lower, np.minimum(lp.upper, np.random.default_rng(0).uniform(5, 50, size=(4, lp.n))))
    ref = solve_pdlp_batch(lp, 4, upper=U, opts=PDLPOptions(tol=1e-4, backend=Backend.preset("cpu-float32")))
    res = solve_pdlp_batch(lp, 4, upper=U, opts=PDLPOptions(tol=1e-4, backend=backend))
    assert res.status == ref.status
    assert np.allclose(res.objective, ref.objective, rtol=1e-3)
    # x and y came back in the LP's own order: the objective is c @ x, and x is as feasible as the reference's.
    # (Not x == ref.x: reductions now sum in another order, and float32 runs can end at different optima.)
    viol = lambda x: np.concatenate([np.abs(lp.K[:lp.m_eq] @ x - lp.q[:lp.m_eq]),
                                     np.maximum(lp.q[lp.m_eq:] - lp.K[lp.m_eq:] @ x, 0)]).max()
    for i in range(4):
        assert lp.c @ res.x[i] == pytest.approx(res.objective[i], rel=1e-4, abs=1e-6)
        assert viol(res.x[i]) <= 3 * viol(ref.x[i]) + 1e-6
    assert np.all(res.x >= lp.lower - 1e-6) and np.all(res.x <= U + 1e-6)


def test_batch_with_permuted_kernel_in_simulator():
    pytest.importorskip("nki")
    from trnopt.problems import random_lp
    _batch_check(Backend.preset("cpu-float32", spmv="nki-sim"), random_lp(seed=3))   # small: every multiply simulated


@needs_neuron
@pytest.mark.parametrize("spmv", ["nki", "nki-op"])
def test_batch_with_permuted_kernel_on_device(spmv):
    from pathlib import Path
    from trnopt import load_mps
    _batch_check(Backend.preset("neuron", spmv=spmv, dense=False),
                 load_mps(str(Path(__file__).resolve().parent.parent / "data" / "shell.mps")))     # split rows
