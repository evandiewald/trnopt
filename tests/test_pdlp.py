"""Every test runs against several backends.

"cpu"            -- the reference: float64, sparse K
"cpu-float32"    -- isolates the effect of float32 precision alone
"emulate-neuron" -- Trainium's restrictions (float32, dense K) on the CPU
"neuron"         -- the real chip (TorchNeuron); added automatically when a NeuronCore is present

Float32 backends get looser tolerances: about 7 significant digits is all float32 has.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from trnopt import LP, Backend, MatVec, PDLPOptions, Presolver, load_mps, solve_highs, solve_pdlp
from trnopt.problems import infeasible_lp, loan_selection, random_lp, textbook, unbounded_lp

DATA = Path(__file__).resolve().parent.parent / "data"

# name -> (backend, solver tolerance, allowed objective error vs HiGHS)
BACKENDS = {
    "cpu": (Backend.preset("cpu"), 1e-6, 1e-5),
    "cpu-float32": (Backend.preset("cpu", dtype=torch.float32), 1e-4, 1e-3),
    "emulate-neuron": (Backend.preset("emulate-neuron", compile=False), 1e-4, 1e-3),
}


def _neuron_device_present() -> bool:
    try:
        import torch_neuronx  # noqa: F401  (registers the "neuron" device)
        return torch.neuron.device_count() > 0
    except Exception:
        return False


HAVE_NEURON = _neuron_device_present()
needs_neuron = pytest.mark.skipif(not HAVE_NEURON, reason="no NeuronCore on this machine")
if HAVE_NEURON:
    BACKENDS["neuron"] = (Backend.preset("neuron"), 1e-4, 1e-3)


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / (1.0 + abs(b))


@pytest.fixture(params=list(BACKENDS), ids=list(BACKENDS))
def backend(request):
    return BACKENDS[request.param]


# ---- correctness against HiGHS ---------------------------------------------------------

def test_textbook(backend):
    be, tol, _ = backend
    res = solve_pdlp(textbook(), PDLPOptions(tol=tol, backend=be))
    assert res.status == "optimal"
    np.testing.assert_allclose(res.x, [2.0, 6.0], atol=1e-2 if be.dtype == torch.float32 else 1e-5)


@pytest.mark.parametrize(
    "lp",
    [random_lp(seed=s) for s in range(2)] + [random_lp(n=1000, m_eq=100, m_ineq=300, density=0.01, seed=9), loan_selection()],
    ids=lambda lp: lp.name,
)
def test_matches_highs(lp: LP, backend):
    be, tol, obj_tol = backend
    ref = solve_highs(lp)
    res = solve_pdlp(lp, PDLPOptions(tol=tol, backend=be))
    assert res.status == "optimal"
    assert rel_err(res.objective, ref.objective) < obj_tol
    assert res.primal_residual <= tol and res.dual_residual <= tol and res.gap <= tol


@pytest.mark.parametrize("name", ["afiro", "israel", "adlittle"])
def test_netlib(name: str, backend):
    be, tol, obj_tol = backend
    lp = load_mps(str(DATA / f"{name}.mps"))
    ref = solve_highs(lp)
    res = solve_pdlp(lp, PDLPOptions(tol=tol, backend=be))
    assert res.status == "optimal"
    assert rel_err(res.objective, ref.objective) < obj_tol


def test_solution_respects_bounds(backend):
    be, tol, _ = backend
    lp = random_lp(seed=4)
    res = solve_pdlp(lp, PDLPOptions(tol=tol, backend=be))
    assert np.all(res.x >= lp.lower - 1e-6) and np.all(res.x <= lp.upper + 1e-6)


# ---- infeasible and unbounded LPs ----------------------------------------------------------

@pytest.mark.parametrize("lp,expected", [
    (infeasible_lp(0), "infeasible"),
    (infeasible_lp(1), "infeasible"),
    (unbounded_lp(0), "unbounded"),
    (unbounded_lp(1), "unbounded"),
], ids=lambda v: v.name if isinstance(v, LP) else v)
def test_detects_no_solution(lp, expected, backend):
    be, tol, _ = backend
    assert solve_highs(lp).status.lower() == expected   # sanity-check the test problem itself
    res = solve_pdlp(lp, PDLPOptions(tol=tol, backend=be, max_iters=50_000))
    assert res.status == expected


# ---- presolve ----------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["afiro", "shell", "25fv47"])
def test_presolve_matches_highs(name, backend):
    be, tol, obj_tol = backend
    lp = load_mps(str(DATA / f"{name}.mps"))
    ref = solve_highs(lp)
    res = solve_pdlp(lp, PDLPOptions(tol=tol, backend=be, presolve=True))
    assert res.status == "optimal", res.presolve
    assert rel_err(res.objective, ref.objective) < obj_tol
    assert res.primal_residual <= tol            # measured on the ORIGINAL LP, after postsolve
    assert res.presolve.startswith("removed")


def test_presolve_can_solve_outright():
    res = solve_pdlp(textbook(), PDLPOptions(presolve=True))
    assert res.status == "optimal" and res.iterations == 0
    assert res.objective == pytest.approx(-36.0)


def test_presolve_reduces_shell():
    pre = Presolver(load_mps(str(DATA / "shell.mps")))
    assert pre.outcome.status == "reduced"
    assert pre.outcome.rows_removed > 100 and pre.outcome.cols_removed > 100


# ---- accelerator plumbing --------------------------------------------------------------------

def test_warm_start_is_fast():
    from trnopt.pdlp import _solve
    lp = load_mps(str(DATA / "israel.mps"))
    cold = _solve(lp, PDLPOptions(tol=1e-5))
    warm = _solve(lp, PDLPOptions(tol=1e-6), cold.x, cold.y)
    again = _solve(lp, PDLPOptions(tol=1e-6))
    assert warm.status == "optimal"
    assert warm.iterations < again.iterations


def test_compiled_step_matches_eager():
    """torch.compile must not change the math, only how it's executed."""
    lp = loan_selection(300)
    eager = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("emulate-neuron", compile=False)))
    compiled = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("emulate-neuron", compile=True)))
    assert compiled.status == eager.status == "optimal"
    assert rel_err(compiled.objective, eager.objective) < 1e-4


def test_matvec_dense_and_sparse_agree():
    import scipy.sparse as sp
    K = sp.random(30, 50, density=0.2, random_state=0, format="csr")
    dense = MatVec(K, Backend(dense=True))
    sparse = MatVec(K, Backend(dense=False))
    x = torch.randn(50, dtype=torch.float64)
    X = torch.randn(50, 4, dtype=torch.float64)        # a stack of vectors, as batching will use
    torch.testing.assert_close(dense.K(x), sparse.K(x))
    torch.testing.assert_close(dense.K(X), sparse.K(X))
    torch.testing.assert_close(dense.KT(dense.K(x)), sparse.KT(sparse.K(x)))


def test_dense_size_guard():
    import scipy.sparse as sp
    with pytest.raises(ValueError, match="dense limit"):
        MatVec(sp.eye(1000, format="csr"), Backend(dense=True, max_dense_entries=10_000))


def test_unknown_preset():
    with pytest.raises(ValueError, match="unknown backend preset"):
        Backend.preset("tpu")


# ---- float64 finish ------------------------------------------------------------------------------

def test_float64_finish_reaches_tight_tolerance():
    """float32 alone stalls on shell at 1e-6; float32 + a float64 CPU finish gets there."""
    lp = load_mps(str(DATA / "shell.mps"))
    ref = solve_highs(lp)
    res = solve_pdlp(lp, PDLPOptions(tol=1e-6, backend=Backend.preset("emulate-neuron", compile=False), finish_float64=True))
    assert res.status == "optimal"
    assert max(res.primal_residual, res.dual_residual, res.gap) <= 1e-6
    assert rel_err(res.objective, ref.objective) < 1e-5
    assert 0 < res.finish_iterations < res.iterations      # both phases did some of the work


def test_float64_finish_is_noop_for_float64():
    lp = loan_selection(300)
    res = solve_pdlp(lp, PDLPOptions(tol=1e-6, finish_float64=True))
    assert res.status == "optimal" and res.finish_iterations == 0


# ---- the legacy PyTorch/XLA path (torch-neuronx <= 2.9; runs only where torch_xla is installed) ---

def test_xla_backend():
    import os
    os.environ.setdefault("PJRT_DEVICE", "CPU")            # XLA's CPU mode when there's no Neuron device
    pytest.importorskip("torch_xla")
    lp = load_mps(str(DATA / "afiro.mps"))
    ref = solve_highs(lp)
    res = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("neuron-xla")))
    assert res.status == "optimal"
    assert rel_err(res.objective, ref.objective) < 1e-3


# ---- benchmark harness ---------------------------------------------------------------------------

def test_bench_writes_csv(tmp_path):
    import csv
    from trnopt.bench import main, shifted_geomean
    out = tmp_path / "r.csv"
    main(["--instances", str(DATA / "afiro.mps"), "--backends", "cpu", "emulate-neuron", "--no-compile",
          "--price-per-hour", "1.0", "--out", str(out)])
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2
    assert {r["status"] for r in rows} == {"optimal"}
    assert all(float(r["usd"]) >= 0 for r in rows)
    assert shifted_geomean([0.0, 0.0]) == pytest.approx(0.0)
    assert shifted_geomean([10.0]) == pytest.approx(10.0)


# ---- compile cache ---------------------------------------------------------------------------------

def _cache_env_after_import(**env):
    import os, subprocess, sys
    base = {k: v for k, v in os.environ.items()
            if k not in ("TRNOPT_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "NEURON_COMPILE_CACHE_URL", "NEURON_CC_FLAGS")}
    code = "import os, trnopt; print(os.environ.get('TORCHINDUCTOR_CACHE_DIR')); print(os.environ.get('NEURON_COMPILE_CACHE_URL'))"
    out = subprocess.run([sys.executable, "-c", code], env={**base, **env}, capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_compile_cache_env(tmp_path):
    assert _cache_env_after_import(TRNOPT_CACHE_DIR=str(tmp_path)) == [str(tmp_path / "inductor"), str(tmp_path / "neuron")]
    # never overrides what the user set; "off" leaves the compilers' own defaults
    assert _cache_env_after_import(TRNOPT_CACHE_DIR=str(tmp_path), NEURON_COMPILE_CACHE_URL="s3://b/c")[1] == "s3://b/c"
    assert _cache_env_after_import(TRNOPT_CACHE_DIR=str(tmp_path), NEURON_CC_FLAGS="--cache_dir=/x")[1] == "None"
    inductor, neuron = _cache_env_after_import(TRNOPT_CACHE_DIR="off")   # torch fills in its own inductor default
    assert "trnopt" not in inductor and neuron == "None"


def test_compile_cache_persists_across_processes(tmp_path):
    """A compiled solve in a fresh process leaves its compiled step in TRNOPT_CACHE_DIR."""
    import os, subprocess, sys
    code = ("from trnopt import Backend, PDLPOptions, solve_pdlp; from trnopt.problems import textbook; "
            "solve_pdlp(textbook(), PDLPOptions(backend=Backend.preset('emulate-neuron'), max_iters=64))")
    env = {**os.environ, "TRNOPT_CACHE_DIR": str(tmp_path)}
    env.pop("TORCHINDUCTOR_CACHE_DIR", None)
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    assert any((tmp_path / "inductor").rglob("*.py"))


# ---- sliced-ELL format and the NKI kernel ---------------------------------------------------------

@pytest.mark.parametrize("split", [None, 7, "auto"])
def test_sell_format_matches_scipy(split):
    import scipy.sparse as sp
    from trnopt.sell import build_sell, sell_matmul_reference
    rng = np.random.default_rng(0)
    # random sparse rows plus one dense row (the kind of row splitting exists for)
    K = sp.vstack([sp.random(200, 400, density=0.01, random_state=2), sp.csr_matrix(np.ones((1, 400)))]).tocsr()
    S = build_sell(K, split=split, dtype=np.float64)
    for X in (rng.normal(size=400), rng.normal(size=(400, 5))):
        np.testing.assert_allclose(sell_matmul_reference(S, X), K @ X, atol=1e-10)


def test_sell_auto_split_cuts_padding():
    from trnopt.sell import build_sell
    K = load_mps(str(DATA / "shell.mps")).K
    assert build_sell(K, split=256).slots / K.nnz > 5        # long rows force heavy padding
    assert build_sell(K).slots / K.nnz < 1.5                 # auto split fixes it


def test_nki_kernel_in_simulator():
    pytest.importorskip("nki")
    import scipy.sparse as sp
    from trnopt.nki_sell import sell_matmul_simulated
    from trnopt.sell import build_sell
    K = sp.random(300, 500, density=0.02, random_state=1, format="csr")
    X = np.random.default_rng(0).normal(size=(500, 4)).astype(np.float32)
    Y = sell_matmul_simulated(build_sell(K), X)
    ref = K @ X
    assert np.abs(Y - ref).max() / np.abs(ref).max() < 1e-5


def test_nki_kernels_cached_per_layout():
    """Same layout and B: the compiled kernel is reused. Anything baked into it differs: a new kernel."""
    pytest.importorskip("nki")
    import scipy.sparse as sp
    from trnopt.nki_sell import make_sell_kernel
    from trnopt.sell import build_sell
    K = sp.random(300, 500, density=0.02, random_state=1, format="csr")
    S = build_sell(K)
    assert make_sell_kernel(S, 4) is make_sell_kernel(build_sell(K.copy()), 4)
    assert make_sell_kernel(S, 4) is not make_sell_kernel(S, 8)
    # different values, same layout: the same kernel (values are inputs, not baked in)
    K2 = sp.random(300, 500, density=0.02, random_state=2, format="csr")
    if build_sell(K2).widths == S.widths:
        assert make_sell_kernel(build_sell(K2), 4) is make_sell_kernel(S, 4)
    K3 = sp.random(300, 500, density=0.05, random_state=2, format="csr")
    assert make_sell_kernel(build_sell(K3), 4) is not make_sell_kernel(S, 4)


def test_solver_with_simulated_nki_kernel():
    """Every multiply through the simulated NKI kernel: same answer as the PyTorch path."""
    pytest.importorskip("nki")
    lp = textbook()
    ref = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("cpu-float32")))
    res = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("cpu-float32", spmv="nki-sim")))
    assert res.status == "optimal" and res.iterations == ref.iterations
    assert res.objective == pytest.approx(ref.objective, rel=1e-5)


@needs_neuron
def test_nki_kernel_on_device():
    """The same kernel on a real NeuronCore, single vector and batched."""
    import scipy.sparse as sp
    from trnopt.sell import build_sell
    K = sp.random(300, 500, density=0.02, random_state=1, format="csr")
    mv = MatVec(K, Backend.preset("neuron", spmv="nki"))
    for X in (np.random.default_rng(0).normal(size=500), np.random.default_rng(1).normal(size=(500, 4))):
        Y = mv.K(torch.as_tensor(X, dtype=torch.float32, device="neuron")).cpu().numpy()
        ref = K @ X
        assert np.abs(Y - ref).max() / np.abs(ref).max() < 1e-5
        YT = mv.KT(torch.as_tensor(K @ X, dtype=torch.float32, device="neuron")).cpu().numpy()
        refT = K.T @ (K @ X)
        assert np.abs(YT - refT).max() / np.abs(refT).max() < 1e-5


@needs_neuron
def test_solver_with_nki_kernel_on_device():
    """Every multiply through the NKI kernel on the chip: same answer as HiGHS."""
    lp = load_mps(str(DATA / "afiro.mps"))
    res = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("neuron", spmv="nki", dense=False)))
    assert res.status == "optimal"
    assert rel_err(res.objective, solve_highs(lp).objective) < 1e-3


# ---- batched solving --------------------------------------------------------------------------------

def test_batch_matches_single_solves():
    """Each column of a batch must give exactly what solving that LP alone gives."""
    import dataclasses
    from trnopt.batch import solve_pdlp_batch
    from trnopt.problems import random_lp
    lp = random_lp(seed=3)
    rng = np.random.default_rng(0)
    U = np.minimum(lp.upper[None, :], rng.uniform(3, 12, size=(4, lp.n)))      # 4 different upper-bound sets
    res = solve_pdlp_batch(lp, 4, upper=U, opts=PDLPOptions(tol=1e-6))
    for i in range(4):
        single = solve_pdlp(dataclasses.replace(lp, upper=U[i]), PDLPOptions(tol=1e-6))
        assert res.status[i] == single.status == "optimal"
        assert res.iterations[i] == single.iterations
        assert res.objective[i] == pytest.approx(single.objective, rel=1e-9)


def test_batch_mixed_feasible_and_infeasible(backend):
    """Strong branching often creates impossible LPs; each column needs its own verdict."""
    import dataclasses
    from trnopt.batch import solve_pdlp_batch
    be, tol, obj_tol = backend
    lp = loan_selection(300)
    L = np.repeat(lp.lower[None, :], 3, axis=0)
    L[1, :] = 1.0                      # buy every loan: blows the budget -> infeasible
    L[2, :40] = 1.0                    # buy the first 40: feasible, different optimum
    res = solve_pdlp_batch(lp, 3, lower=L, opts=PDLPOptions(tol=tol, backend=be, max_iters=50_000))
    assert res.status == ["optimal", "infeasible", "optimal"]
    for i in (0, 2):
        ref = solve_highs(dataclasses.replace(lp, lower=L[i]))
        assert rel_err(res.objective[i], ref.objective) < obj_tol


def test_batch_scenarios_and_branching(backend):
    import dataclasses
    from trnopt.batch import solve_pdlp_batch
    from trnopt.problems import branching_variants, return_scenarios
    be, tol, obj_tol = backend
    lp = loan_selection(400)
    C = return_scenarios(lp, 6)
    res = solve_pdlp_batch(lp, 6, c=C, opts=PDLPOptions(tol=tol, backend=be))
    for i in range(6):
        assert res.status[i] == "optimal"
        assert rel_err(res.objective[i], solve_highs(dataclasses.replace(lp, c=C[i])).objective) < obj_tol
    lpm = load_mps(str(DATA / "afiro.mps"))
    lpm.integer = np.ones(lpm.n, dtype=bool)             # pretend afiro is a MIP, to exercise branching
    root = solve_pdlp(lpm, PDLPOptions(tol=1e-6))
    Lb, Ub, labels = branching_variants(lpm, root.x, 3)
    res = solve_pdlp_batch(lpm, len(labels), lower=Lb, upper=Ub, opts=PDLPOptions(tol=tol, backend=be))
    for i in range(len(labels)):
        ref = solve_highs(dataclasses.replace(lpm, lower=Lb[i], upper=Ub[i]))
        expect = {"Optimal": "optimal", "Infeasible": "infeasible"}[ref.status]
        assert res.status[i] == expect
        if expect == "optimal":
            assert rel_err(res.objective[i], ref.objective) < obj_tol


def test_batch_rejects_bad_shapes():
    from trnopt.batch import solve_pdlp_batch
    lp = textbook()
    with pytest.raises(ValueError, match="expected shape"):
        solve_pdlp_batch(lp, 3, upper=np.ones((2, lp.n)))


def test_batch_with_simulated_nki_kernel():
    """The batched solver with every multiply through the simulated NKI kernel (X is n x B)."""
    pytest.importorskip("nki")
    from trnopt.batch import solve_pdlp_batch
    from trnopt.problems import return_scenarios
    lp = loan_selection(150)
    C = return_scenarios(lp, 4)
    ref = solve_pdlp_batch(lp, 4, c=C, opts=PDLPOptions(tol=1e-4, backend=Backend.preset("cpu-float32")))
    res = solve_pdlp_batch(lp, 4, c=C, opts=PDLPOptions(tol=1e-4, backend=Backend.preset("cpu-float32", spmv="nki-sim")))
    assert res.status == ref.status == ["optimal"] * 4
    np.testing.assert_array_equal(res.iterations, ref.iterations)
    np.testing.assert_allclose(res.objective, ref.objective, rtol=1e-5)


# ---- continuous batching --------------------------------------------------------------------------

def test_stream_matches_single_solves():
    """24 LPs (some infeasible) through 6 slots: every LP must match solving it alone exactly.

    Refills happen only at evaluation points, so each LP is evaluated at the same ages it would
    be on its own, and gets the same trajectory, iteration count and answer.
    """
    import dataclasses
    from trnopt.batch import solve_pdlp_stream
    lp = loan_selection(200)
    rng = np.random.default_rng(1)
    N = 24
    L = np.repeat(lp.lower[None, :], N, axis=0)
    for i in range(N):
        L[i, rng.choice(lp.n, size=rng.integers(0, 60), replace=False)] = 1.0   # force some loans in
    L[[5, 17]] = 1.0                                                          # force all in: infeasible
    res = solve_pdlp_stream(lp, N, 6, lower=L, opts=PDLPOptions(tol=1e-6))
    for i in range(N):
        single = solve_pdlp(dataclasses.replace(lp, lower=L[i]), PDLPOptions(tol=1e-6))
        assert res.status[i] == single.status, i
        assert res.iterations[i] == single.iterations, i
        if single.status == "optimal":
            assert res.objective[i] == pytest.approx(single.objective, rel=1e-9)
    assert res.status[5] == res.status[17] == "infeasible"


def test_stream_keeps_slots_busy():
    """With a queue longer than the batch, slots stay busy: useful work beats a single fixed batch."""
    from trnopt.batch import solve_pdlp_batch, solve_pdlp_stream
    from trnopt.problems import return_scenarios
    lp = loan_selection(200)
    C = return_scenarios(lp, 32, spread=0.5)
    fixed = solve_pdlp_batch(lp, 32, c=C, opts=PDLPOptions(tol=1e-6))
    stream = solve_pdlp_stream(lp, 32, 4, c=C, opts=PDLPOptions(tol=1e-6))
    assert stream.status == fixed.status
    np.testing.assert_array_equal(stream.iterations, fixed.iterations)
    assert stream.useful_work > fixed.useful_work


def test_stream_float64_finish():
    """float32 alone stalls on shell at 1e-6; streamed with a float64 finish, every variant gets there."""
    import dataclasses
    from trnopt.batch import solve_pdlp_stream
    from trnopt.problems import return_scenarios
    lp = load_mps(str(DATA / "shell.mps"))
    C = return_scenarios(lp, 6, spread=0.1)
    res = solve_pdlp_stream(lp, 6, 3, c=C, opts=PDLPOptions(
        tol=1e-6, backend=Backend.preset("emulate-neuron", compile=False), finish_float64=True))
    assert res.status == ["optimal"] * 6
    assert np.all(np.maximum.reduce([res.primal_residual, res.dual_residual, res.gap]) <= 1e-6)
    assert np.all(res.finish_iterations < res.iterations)           # the device did part of every solve
    assert res.finish_iterations.sum() > 0                          # and the CPU finished at least one
    for i in range(6):
        ref = solve_highs(dataclasses.replace(lp, c=C[i]))
        assert rel_err(res.objective[i], ref.objective) < 1e-5


# ---- Apple silicon sparse kernel (metal_sell.py) ---------------------------------------------------

@pytest.mark.parametrize("which", ["K", "KT"])
def test_metal_layout_matches_scipy(which):
    """The Metal kernel's exact index arithmetic, run in PyTorch (works anywhere)."""
    import scipy.sparse as sp
    from trnopt.metal_sell import MetalSELL
    K = sp.vstack([sp.random(200, 400, density=0.01, random_state=2), sp.csr_matrix(np.ones((1, 400)))]).tocsr()
    M = K if which == "K" else K.T.tocsr()
    S = MetalSELL(M, device="cpu", reference=True)
    for B in (1, 7, 64):
        X = np.random.default_rng(B).normal(size=(M.shape[1], B))
        got = S.matmul(torch.as_tensor(X, dtype=torch.float32)).numpy()
        np.testing.assert_allclose(got, M @ X, rtol=1e-4, atol=1e-4)


def test_solver_with_metal_reference_path():
    """Solves through the Metal layout reach the same answer as the regular float32 path.

    Not the same iteration count: the layout adds each row's terms in a different order, float32
    rounding differs slightly, and PDHG's restart decisions amplify that (on an M4, shell took
    7104 vs 8960 iterations to the same objective)."""
    from trnopt.batch import solve_pdlp_stream
    from trnopt.problems import return_scenarios
    lp = load_mps(str(DATA / "shell.mps"))
    a = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("cpu-float32")))
    b = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("cpu-float32", spmv="metal-ref")))
    assert a.status == b.status == "optimal"
    assert abs(a.objective - b.objective) / (1 + abs(a.objective)) < 1e-4
    assert max(a.iterations, b.iterations) < 2 * min(a.iterations, b.iterations)
    lp = loan_selection(200)
    C = return_scenarios(lp, 6)
    r = solve_pdlp_stream(lp, 6, 3, c=C, opts=PDLPOptions(tol=1e-4, backend=Backend.preset("cpu-float32", spmv="metal-ref")))
    assert r.status == ["optimal"] * 6


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple silicon GPU")
def test_metal_kernel_on_gpu():
    """The real Metal kernel, on the Mac GPU, against SciPy."""
    import scipy.sparse as sp
    from trnopt.metal_sell import MetalSELL
    K = load_mps(str(DATA / "shell.mps")).K
    for M in (K.tocsr(), K.T.tocsr()):
        S = MetalSELL(M, device="mps")
        for B in (1, 32, 128):
            X = np.random.default_rng(B).normal(size=(M.shape[1], B))
            got = S.matmul(torch.as_tensor(X, dtype=torch.float32).to("mps")).cpu().numpy()
            ref = M @ X
            assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-5


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple silicon GPU")
def test_solver_on_gpu_with_metal_kernel():
    lp = load_mps(str(DATA / "israel.mps"))
    ref = solve_highs(lp)
    res = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("mps-metal")))
    assert res.status == "optimal" and rel_err(res.objective, ref.objective) < 1e-3


def test_float32_branch_lps_with_zero_objective_converge():
    """comp21-2idx branch LPs (objective 0, ~10k upper-bounded variables) used to stall in float32
    forever: rounding noise in the reduced costs, credited to bounds, biased the dual objective.
    With bounds credited only when x is near them (as Google's PDLP does), they solve quickly."""
    import dataclasses
    lp = load_mps(str(DATA / "miplib" / "comp21-2idx.mps.gz"))
    x = solve_highs(lp).x
    frac = np.flatnonzero(lp.integer & (np.abs(x - np.round(x)) > 1e-3))[:6]
    assert len(frac) >= 3
    for j in frac:
        for down in (True, False):
            lo, up = lp.lower.copy(), lp.upper.copy()
            if down:
                up[j] = np.floor(x[j])
            else:
                lo[j] = np.ceil(x[j])
            child = dataclasses.replace(lp, lower=lo, upper=up)
            ref = solve_highs(child)
            r = solve_pdlp(child, PDLPOptions(tol=1e-4, max_iters=20_000, backend=Backend.preset("cpu-float32")))
            assert r.status == "optimal", (j, down, r.status)
            assert rel_err(r.objective, ref.objective) < 1e-3


# ---- branch-and-bound MIP solver --------------------------------------------------------------

@pytest.mark.parametrize("branching", ["most_fractional", "pseudocost", "strong_highs", "reliability_highs",
                                       "reliability_trnopt", "strong_trnopt"])
def test_mip_matches_highs(branching):
    """Every branching rule (and both strong-branching engines) proves the same optimum as HiGHS's MIP solver."""
    from trnopt.generators import facility_location, independent_set, set_cover
    from trnopt.mip import solve_mip
    from trnopt.reference import solve_highs_mip
    for lp in (set_cover(60, 120, 0.06, seed=5), independent_set(80, 3, seed=5), facility_location(15, 10, seed=5)):
        ref = solve_highs_mip(lp, 60)
        r = solve_mip(lp, branching, time_limit=120)
        assert r.status == "optimal", (lp.name, r.status)
        assert abs(r.objective - ref.objective) <= 1e-6 * (1 + abs(ref.objective)), (lp.name, r.objective, ref.objective)
        assert r.gap <= 1e-4


def test_mip_after_multithreaded_highs():
    """HiGHS fixes its thread pool size with the first solver in a process and refuses ("Not Set") any
    later solver that asks for a different size. Our node solver must work after HiGHS's MIP solver ran
    multi-threaded (on a 10-core Mac the default is 5). Run in a fresh process so test order can't matter."""
    import subprocess, sys
    code = (
        "from trnopt.generators import set_cover\n"
        "from trnopt.mip import solve_mip\n"
        "from trnopt.reference import solve_highs_mip\n"
        "lp = set_cover(60, 120, 0.06, seed=5)\n"
        "ref = solve_highs_mip(lp, 60, threads=4)\n"
        "r = solve_mip(lp, 'most_fractional')\n"
        "assert ref.status == 'Optimal', ref.status\n"
        "assert r.status == 'optimal' and abs(r.objective - ref.objective) <= 1e-6 * (1 + abs(ref.objective))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]


def test_warm_up_runs_a_stream():
    """batch.warm_up runs the programs a stream of that width uses and reports how long it took."""
    from trnopt.batch import warm_up
    t = warm_up(random_lp(40, 10, 20, seed=2), 16, Backend.preset("cpu-float32"))
    assert 0 < t < 60


@needs_neuron
def test_mip_on_neuron_warms_up_and_matches_highs():
    """On Neuron the pump's and strong branching's programs are warmed up before those phases run (time
    reported as times["warmup"]), and the answer is still HiGHS's optimum."""
    from trnopt.generators import set_cover
    from trnopt.mip import solve_mip
    from trnopt.reference import solve_highs_mip
    lp = set_cover(60, 120, 0.06, seed=5)
    ref = solve_highs_mip(lp, 60)
    r = solve_mip(lp, "reliability_trnopt", time_limit=300, backend="neuron", pump=16)
    assert r.status == "optimal" and abs(r.objective - ref.objective) <= 1e-6 * (1 + abs(ref.objective))
    assert r.times.get("warmup", 0) > 0
