"""Pyomo interface: use trnopt like any other Pyomo solver.

    import pyomo.environ as pyo
    import trnopt.pyomo_plugin                      # registers SolverFactory("trnopt")

    opt = pyo.SolverFactory("trnopt")
    opt.options["backend"] = "neuron"               # or "cpu-float32", "cuda", a Backend object
    opt.options["time_limit"] = 300
    results = opt.solve(model)                      # values are loaded into the model
    print(results.solver.termination_condition, pyo.value(model.obj))

Models with integer (or binary) variables go to trnopt's branch-and-bound (`trnopt.mip.solve_mip`): node LPs
solved exactly by HiGHS dual simplex on the CPU, strong-branching LPs batched on the chosen backend. Pure LPs go
to PDLP (`trnopt.solve_pdlp`) on the backend, with a float64 finish on the CPU when the backend is float32.

Options (solver.options, or solve(..., options={...})), with defaults:
    backend       "cpu-float32"          where the batched PDLP runs: "cpu-float32", "cuda", "neuron", ...
    time_limit    600                    seconds (solve(timelimit=...) also works)
    mip_gap       1e-4                   relative gap at which the MIP search stops
    branching     "reliability_trnopt"   any rule solve_mip accepts (e.g. "strong_trnopt", "pseudocost")
    slots         None                   strong-branching batch size cap; None = 1024 on accelerators, 128 on CPU
    strong_iters  256                    PDHG iterations per strong-branching LP (a ranking, not a solve)
    lp_tol        1e-6                   relative tolerance for pure LPs
    presolve      True                   MIPs: HiGHS's MIP presolve before branch-and-bound (trnopt/highs_presolve.py)
    pump          16                     MIPs: parallel feasibility pumps at the root, batched on the backend (pump.py)
    cuts          5                      MIPs: rounds of Gomory mixed-integer cuts at the root (cuts.py); 0 = off
Linear models only (linear objective and constraints); one active objective.

The model is translated with Pyomo's own LinearStandardFormCompiler (mixed form, so each row keeps its
sense), then mapped to trnopt's LP form: equality rows first, then ">=" rows ("<=" rows negated).
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch
from pyomo.core.base.objective import minimize
from pyomo.environ import value
from pyomo.opt import SolverFactory, SolverResults, SolverStatus, TerminationCondition
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

from .backend import Backend
from .lp import LP

DEFAULTS = dict(backend="cpu-float32", time_limit=600.0, mip_gap=1e-4, branching="reliability_trnopt",
                slots=None, strong_iters=256, lp_tol=1e-6, presolve=True, pump=16, cuts=5)

# trnopt status -> (Pyomo solver status, termination condition)
_STATUS = {
    "optimal": (SolverStatus.ok, TerminationCondition.optimal),
    "infeasible": (SolverStatus.warning, TerminationCondition.infeasible),
    "unbounded": (SolverStatus.warning, TerminationCondition.unbounded),
    "unbounded_or_infeasible": (SolverStatus.warning, TerminationCondition.infeasibleOrUnbounded),
    "time_limit": (SolverStatus.aborted, TerminationCondition.maxTimeLimit),
    "node_limit": (SolverStatus.aborted, TerminationCondition.maxEvaluations),
    "iteration_limit": (SolverStatus.aborted, TerminationCondition.maxIterations),
    "inaccurate": (SolverStatus.warning, TerminationCondition.other),
    "stalled": (SolverStatus.warning, TerminationCondition.other),
    "numerical_error": (SolverStatus.error, TerminationCondition.error),
}


def model_to_lp(model):
    """Translate a linear Pyomo model into trnopt's LP. Returns (lp, info), info from Pyomo's compiler."""
    info = LinearStandardFormCompiler().write(model, mixed_form=True, set_sense=minimize)
    if len(info.objectives) != 1:
        raise ValueError(f"trnopt needs exactly one active objective, found {len(info.objectives)}")
    cols = info.columns
    n = len(cols)
    A = sp.csr_matrix(info.A, shape=(len(info.rows), n))
    rhs = np.asarray(info.rhs, dtype=np.float64).reshape(-1)
    # In mixed form each row is A_i x (<=, ==, >=) rhs_i; the row multiplier says which: +1 <=, 0 ==, -1 >=.
    mult = np.array([r[1] for r in info.rows], dtype=int)
    eq, ge, le = mult == 0, mult == -1, mult == 1
    K = sp.vstack([A[eq], A[ge], -A[le]]).tocsr()
    q = np.concatenate([rhs[eq], rhs[ge], -rhs[le]])
    lower = np.array([-np.inf if v.lb is None else v.lb for v in cols], dtype=np.float64)
    upper = np.array([np.inf if v.ub is None else v.ub for v in cols], dtype=np.float64)
    integer = np.array([v.is_integer() for v in cols], dtype=bool)
    c = np.asarray(info.c.toarray(), dtype=np.float64).reshape(-1) if n else np.zeros(0)
    lp = LP(c, K, q, int(eq.sum()), lower, upper, name=model.name or "pyomo",
            integer=integer if integer.any() else None)
    return lp, info


def lp_to_pyomo(lp: LP):
    """Build a Pyomo model from a trnopt LP (e.g. load_mps on a MIPLIB file, or a generated instance).

    Pyomo has no MPS reader, so this is how MPS benchmarks reach every solver through the same Pyomo model.
    Variables x[0..n-1] keep lp.lower/upper and lp.integer; rows are "eq[i]" (==) and "ge[i]" (>=).
    """
    import pyomo.environ as pyo
    from pyomo.core.expr.numeric_expr import LinearExpression

    m = pyo.ConcreteModel(name=lp.name)
    integer = lp.integer if lp.integer is not None else np.zeros(lp.n, dtype=bool)
    lb = [None if not np.isfinite(v) else float(v) for v in lp.lower]
    ub = [None if not np.isfinite(v) else float(v) for v in lp.upper]
    m.x = pyo.Var(range(lp.n), domain=lambda m, j: pyo.Integers if integer[j] else pyo.Reals,
                  bounds=lambda m, j: (lb[j], ub[j]))
    K = lp.K.tocsr()
    xs = [m.x[j] for j in range(lp.n)]

    def row(i):
        a, b = K.indptr[i], K.indptr[i + 1]
        return LinearExpression(constant=0.0, linear_coefs=K.data[a:b].tolist(),
                                linear_vars=[xs[j] for j in K.indices[a:b]])
    m.eq = pyo.Constraint(range(lp.m_eq), rule=lambda m, i: row(i) == float(lp.q[i]))
    m.ge = pyo.Constraint(range(lp.m - lp.m_eq), rule=lambda m, k: row(lp.m_eq + k) >= float(lp.q[lp.m_eq + k]))
    nz = np.flatnonzero(lp.c)
    m.obj = pyo.Objective(expr=LinearExpression(constant=0.0, linear_coefs=lp.c[nz].tolist(),
                                                linear_vars=[xs[j] for j in nz]))
    return m


class _Options(dict):
    """solver.options: a dict that also allows attribute access (opt.options.backend = "neuron")."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        self[k] = v


@SolverFactory.register("trnopt", doc="trnopt: batched-PDLP LP/MIP solver (CPU, CUDA or AWS Trainium)")
class TrnoptSolver:
    """A Pyomo solver object (the classic SolverFactory interface: solve, options, available, version)."""

    def __init__(self, **kwds):
        self.name = "trnopt"
        self.options = _Options(kwds.get("options") or {})
        self._last = None             # the raw trnopt result of the last solve, for inspection

    # -- the bits of the classic solver interface people call --------------------------------
    def available(self, exception_flag: bool = True) -> bool:
        return True

    def license_is_valid(self) -> bool:
        return True

    def version(self):
        from importlib.metadata import PackageNotFoundError, version
        try:
            return tuple(int(p) for p in version("trnopt").split(".")[:3])
        except (PackageNotFoundError, ValueError):
            return (0, 0, 0)

    def warm_start_capable(self) -> bool:
        return False

    def set_options(self, opts: dict) -> None:
        self.options.update(opts)

    # -- solving ------------------------------------------------------------------------------
    def solve(self, model, tee: bool = False, load_solutions: bool = True, timelimit: float | None = None,
              options: dict | None = None, **kwds) -> SolverResults:
        opts = {**DEFAULTS, **dict(self.options), **(options or {})}
        if timelimit is not None:
            opts["time_limit"] = float(timelimit)
        unknown = set(opts) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown trnopt option(s) {sorted(unknown)}; valid: {sorted(DEFAULTS)}")
        t0 = time.perf_counter()
        lp, info = model_to_lp(model)
        be = Backend.preset(opts["backend"]) if isinstance(opts["backend"], str) else opts["backend"]

        nodes = None
        bound = None
        if lp.integer is not None:
            from .mip import solve_mip
            slots = opts["slots"] or (128 if be.device == "cpu" else 1024)
            r = solve_mip(lp, opts["branching"], time_limit=opts["time_limit"], gap_tol=opts["mip_gap"],
                          backend=be, slots=slots, strong_iters=opts["strong_iters"], presolve=opts["presolve"],
                          pump=opts["pump"], cuts=opts["cuts"], verbose=tee)
            status, x, nodes, bound = r.status, r.x, r.nodes, r.bound
        else:
            from .pdlp import PDLPOptions, solve_pdlp
            r = solve_pdlp(lp, PDLPOptions(tol=opts["lp_tol"], backend=be, time_limit=opts["time_limit"],
                                           finish_float64=be.dtype == torch.float32))
            status, x = r.status, r.x
            if status == "optimal":
                bound = r.objective
        self._last = r
        wall = time.perf_counter() - t0

        solver_status, term = _STATUS.get(status, (SolverStatus.unknown, TerminationCondition.unknown))
        results = SolverResults()
        results.solver.name = "trnopt"
        results.solver.status = solver_status
        results.solver.termination_condition = term
        results.solver.termination_message = f"trnopt status: {status}"
        results.solver.wallclock_time = wall
        if nodes is not None:
            results.solver.statistics.branch_and_bound.number_of_bounded_subproblems = nodes
        results.problem.name = lp.name
        results.problem.number_of_variables = lp.n
        results.problem.number_of_constraints = lp.m
        results.problem.number_of_integer_variables = int(lp.integer.sum()) if lp.integer is not None else 0

        obj = info.objectives[0]
        sense_min = obj.sense == minimize
        have_x = x is not None and np.all(np.isfinite(x))
        if have_x and load_solutions:
            for v, val in zip(info.columns, x):
                v.set_value(round(float(val)) if v.is_integer() else float(val), skip_validation=True)
            objval = value(obj)                         # in the model's own sense, offset included
            # bound is in trnopt's (minimization) sense, without the constant offset Pyomo kept apart
            b = None if bound is None or not np.isfinite(bound) else float(bound) + float(info.c_offset[0])
            if sense_min:
                results.problem.upper_bound = objval
                if b is not None:
                    results.problem.lower_bound = b
            else:
                results.problem.lower_bound = objval
                if b is not None:
                    results.problem.upper_bound = -b
        return results
