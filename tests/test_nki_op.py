"""spmv="nki-op": the NKI kernel as a TorchNeuron custom op (nki_op.py), so torch.compile keeps it in the graph."""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from trnopt import Backend, PDLPOptions, load_mps, solve_pdlp
from trnopt.backend import MatVec
from trnopt.sell import build_sell
from test_pdlp import needs_neuron

DATA = Path(__file__).resolve().parent.parent / "data"


def _matrices():
    yield "random", sp.random(300, 500, density=0.02, random_state=1, format="csr")
    yield "wide", sp.random(200, 3000, density=0.01, random_state=2, format="csr")
    yield "shell", load_mps(str(DATA / "shell.mps")).K.tocsr()       # long rows: split rows, scatter-add passes


def test_layout_ids_follow_the_layout():
    from trnopt.nki_op import layout_id
    K = sp.random(300, 500, density=0.02, random_state=1, format="csr")
    assert layout_id(build_sell(K)) == layout_id(build_sell(K.copy() * 2.0))   # values aren't baked in
    assert layout_id(build_sell(K)) != layout_id(build_sell(K.T.tocsr()))
    K2 = sp.random(300, 500, density=0.05, random_state=2, format="csr")         # wider slices
    assert layout_id(build_sell(K2)) != layout_id(build_sell(K))


@pytest.mark.skipif(importlib.util.find_spec("torch_neuronx") is not None, reason="torch_neuronx is installed")
def test_falls_back_to_nki_without_torch_neuronx():
    K = sp.random(30, 50, density=0.1, random_state=1, format="csr")
    with pytest.warns(UserWarning, match="using 'nki'"):
        mv = MatVec(K, Backend(spmv="nki-op", dtype=torch.float32))
    assert mv.spmv == "nki"


@needs_neuron
@pytest.mark.parametrize("name,K", list(_matrices()), ids=lambda v: v if isinstance(v, str) else "")
def test_op_matches_nki_path(name, K):
    """Same products as spmv="nki", eagerly, for one vector and for a batch, through K and K^T."""
    m, n = K.shape
    mv_op = MatVec(K, Backend.preset("neuron", spmv="nki-op"))
    mv_nki = MatVec(K, Backend.preset("neuron", spmv="nki"))
    rng = np.random.default_rng(0)
    for shape_x, shape_y in (((n,), (m,)), ((n, 16), (m, 16))):
        X = torch.as_tensor(rng.normal(size=shape_x), dtype=torch.float32).to("neuron")
        Y = torch.as_tensor(rng.normal(size=shape_y), dtype=torch.float32).to("neuron")
        for f in ("K", "KT"):
            v = X if f == "K" else Y
            a = getattr(mv_op, f)(v).cpu().numpy()
            b = getattr(mv_nki, f)(v).cpu().numpy()
            ref = (K if f == "K" else K.T) @ v.cpu().numpy().astype(np.float64)
            assert a.shape == ref.shape
            assert np.abs(a - b).max() <= 1e-6 * np.abs(ref).max()
            assert np.abs(a - ref).max() <= 1e-5 * np.abs(ref).max()
    assert mv_op.spmv == "nki-op"                                          # it didn't fall back


@needs_neuron
def test_op_compiles_without_graph_breaks():
    """The point of the op: a compiled function around K and K^T is one graph, and gives the right answer."""
    import torch._dynamo as dynamo
    K = sp.random(300, 500, density=0.02, random_state=1, format="csr")
    X = np.random.default_rng(1).normal(size=(500, 16)).astype(np.float32)

    def f(mv, x):
        return 2.0 * mv.KT(torch.relu(mv.K(x) - 1.0))

    mv = MatVec(K, Backend.preset("neuron", spmv="nki-op"))
    dynamo.reset()
    ex = dynamo.explain(f)(mv, torch.as_tensor(X).to("neuron"))
    assert ex.graph_count == 1 and ex.graph_break_count == 0, ex.break_reasons
    dynamo.reset()
    Z = torch.compile(f, backend="neuron", dynamic=False)(mv, torch.as_tensor(X).to("neuron")).cpu().numpy()
    ref = 2.0 * (K.T @ np.maximum(K @ X - 1.0, 0.0))
    assert np.abs(Z - ref).max() <= 1e-5 * np.abs(ref).max()


@needs_neuron
def test_solver_with_nki_op_on_device():
    """Every multiply through the op, inside the compiled step: same iterations and answer as spmv="nki"."""
    lp = load_mps(str(DATA / "afiro.mps"))
    ref = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("neuron", spmv="nki", dense=False)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("neuron", spmv="nki-op", dense=False)))
    assert not [c for c in caught if "using 'nki'" in str(c.message)]   # it didn't fall back
    assert res.status == ref.status == "optimal"
    assert res.objective == pytest.approx(ref.objective, rel=1e-5)
    assert abs(res.iterations - ref.iterations) <= 64                  # float32 rounding may shift a check


@needs_neuron
def test_auto_picks_the_op_above_the_dense_cutoff():
    K = sp.random(300, 500, density=0.02, random_state=1, format="csr")
    assert MatVec(K, Backend.preset("neuron")).spmv == "torch"                         # small: dense
    mv = MatVec(K, Backend.preset("neuron", auto_dense_entries=0))
    X = np.random.default_rng(0).normal(size=(500, 16))
    Y = mv.K(torch.as_tensor(X, dtype=torch.float32).to("neuron")).cpu().numpy()
    assert mv.spmv == "nki-op" and np.abs(Y - K @ X).max() <= 1e-5 * np.abs(K @ X).max()
