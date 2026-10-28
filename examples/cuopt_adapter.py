"""Thin adapter between trnopt's LP and NVIDIA cuOpt's Python data-model API.

Written before having a GPU box, against cuOpt's documented data-model API (cuopt 26.x):
DataModel setters, SolverSettings.set_parameter / set_optimality_tolerance, solver.Solve, and the
Solution getters. Run `python examples/cuopt_adapter.py` on the GPU box first: it prints the
installed version, checks that every name used here exists, and solves a tiny LP and MIP.

trnopt's LP:  min c x  s.t.  K[:m_eq] x == q[:m_eq],  K[m_eq:] x >= q[m_eq:],  lower <= x <= upper.
cuOpt takes row bounds directly: [q, q] for equality rows, [q, +inf] for ">=" rows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

INF = np.inf


def _cuopt():
    from cuopt.linear_programming import data_model, solver, solver_settings
    return data_model, solver, solver_settings


def to_data_model(lp, integer: bool = False, lower=None, upper=None):
    """cuOpt DataModel for lp (integrality only if integer=True). lower/upper override the bounds."""
    data_model, _, _ = _cuopt()
    K = sp.csr_matrix(lp.K, dtype=np.float64)
    dm = data_model.DataModel()
    dm.set_csr_constraint_matrix(K.data, K.indices.astype(np.int32), K.indptr.astype(np.int32))
    row_lo = np.asarray(lp.q, dtype=np.float64)
    row_up = np.where(np.arange(lp.m) < lp.m_eq, row_lo, INF)
    dm.set_constraint_lower_bounds(row_lo)
    dm.set_constraint_upper_bounds(row_up)
    dm.set_objective_coefficients(np.asarray(lp.c, dtype=np.float64))
    dm.set_variable_lower_bounds(np.asarray(lp.lower if lower is None else lower, dtype=np.float64))
    dm.set_variable_upper_bounds(np.asarray(lp.upper if upper is None else upper, dtype=np.float64))
    if integer and lp.integer is not None:
        dm.set_variable_types(np.where(lp.integer, "I", "C"))
    return dm


@dataclass
class CuoptResult:
    status: str               # "optimal" / "infeasible" / "unbounded" / "time_limit" / other cuOpt text
    objective: float
    seconds: float            # wall, including cuOpt's own setup for this call
    solver_seconds: float     # cuOpt's reported solve time
    iterations: int | None
    warm: object = None       # PDLP warm-start data, if requested
    x: np.ndarray | None = None


def _raise_on_error(sol) -> None:
    """cuOpt reports some failures (e.g. invalid settings) as a solution with an error, not an exception."""
    get = getattr(sol, "get_error_status", None)
    err = get() if get else None
    if err is not None and "success" not in str(err).lower() and str(err) not in ("0", "None"):
        msg = getattr(sol, "get_error_message", lambda: "")()
        raise RuntimeError(f"cuOpt error {err}: {msg}")


def _status(sol) -> str:
    s = str(sol.get_termination_reason() if hasattr(sol, "get_termination_reason") else sol.get_termination_status())
    t = s.lower()
    for key, out in (("optimal", "optimal"), ("infeasib", "infeasible"), ("unbound", "unbounded"),
                     ("time", "time_limit"), ("iteration", "iteration_limit")):
        if key in t:
            return out
    return s


def solve_lp(dm, tol: float = 1e-4, method: str = "pdlp", time_limit: float = 600.0, warm=None,
             want_warm: bool = False, want_x: bool = False, pdlp_mode: str | None = None,
             precision: int | None = None) -> CuoptResult:
    """Solve the LP in dm. method: 'pdlp', 'dual_simplex', 'barrier' or 'concurrent'.

    precision (PDLP): None = cuOpt's default (-1), 0 = single, 1 = double, 2 = mixed (constants.h).
    """
    _, solver, solver_settings = _cuopt()
    from cuopt.linear_programming.solver.solver_parameters import CUOPT_METHOD, CUOPT_TIME_LIMIT
    from cuopt.linear_programming.solver_settings import SolverMethod
    st = solver_settings.SolverSettings()
    st.set_parameter("log_to_console", False)
    st.set_parameter(CUOPT_METHOD, {"pdlp": SolverMethod.PDLP, "dual_simplex": SolverMethod.DualSimplex,
                                    "barrier": SolverMethod.Barrier, "concurrent": SolverMethod.Concurrent}[method])
    st.set_parameter(CUOPT_TIME_LIMIT, float(time_limit))
    st.set_optimality_tolerance(tol)
    if pdlp_mode is not None:
        from cuopt.linear_programming.solver.solver_parameters import CUOPT_PDLP_SOLVER_MODE
        from cuopt.linear_programming.solver_settings import PDLPSolverMode
        st.set_parameter(CUOPT_PDLP_SOLVER_MODE, getattr(PDLPSolverMode, pdlp_mode))
    if precision is not None:
        from cuopt.linear_programming.solver.solver_parameters import CUOPT_PDLP_PRECISION
        st.set_parameter(CUOPT_PDLP_PRECISION, int(precision))
    if warm is not None or want_warm:
        # cuOpt 26.10 accepts PDLP warm starts only in Stable2 mode ("Only Stable2 mode supported for warm
        # start"), and only without presolve; use the same settings for the solve that produces the data.
        from cuopt.linear_programming.solver.solver_parameters import CUOPT_PDLP_SOLVER_MODE, CUOPT_PRESOLVE
        from cuopt.linear_programming.solver_settings import PDLPSolverMode
        st.set_parameter(CUOPT_PDLP_SOLVER_MODE, PDLPSolverMode.Stable2)
        st.set_parameter(CUOPT_PRESOLVE, 0)
    if warm is not None:
        st.set_pdlp_warm_start_data(warm)
    t = time.perf_counter()
    sol = solver.Solve(dm, st)
    wall = time.perf_counter() - t
    _raise_on_error(sol)
    stats = sol.get_lp_stats() if hasattr(sol, "get_lp_stats") else {}
    it = stats.get("nb_iterations") if isinstance(stats, dict) else None
    return CuoptResult(_status(sol), float(sol.get_primal_objective()), wall, float(sol.get_solve_time()), it,
                       sol.get_pdlp_warm_start_data() if want_warm else None,
                       np.asarray(sol.get_primal_solution()) if want_x else None)


def solve_mip(dm, time_limit: float = 300.0, gap: float = 1e-4, options: dict | None = None) -> CuoptResult:
    """cuOpt's MIP solver on dm (built with integer=True).

    options: extra parameters by name, e.g. {"mip_batch_pdlp_strong_branching": 1,
    "mip_batch_pdlp_reliability_branching": 1} for cuOpt's batched-PDLP branching (off by default).
    """
    _, solver, solver_settings = _cuopt()
    from cuopt.linear_programming.solver.solver_parameters import (CUOPT_MIP_RELATIVE_GAP, CUOPT_TIME_LIMIT)
    st = solver_settings.SolverSettings()
    st.set_parameter("log_to_console", False)
    st.set_parameter(CUOPT_TIME_LIMIT, float(time_limit))
    st.set_parameter(CUOPT_MIP_RELATIVE_GAP, float(gap))
    for k, v in (options or {}).items():
        st.set_parameter(k, v)
    t = time.perf_counter()
    sol = solver.Solve(dm, st)
    wall = time.perf_counter() - t
    _raise_on_error(sol)
    return CuoptResult(_status(sol), float(sol.get_primal_objective()), wall, float(sol.get_solve_time()), None)


if __name__ == "__main__":
    # Smoke test + API check on the GPU box.
    import importlib.metadata as md
    for pkg in ("cuopt-cu12", "cuopt-cu13", "cuopt"):
        try:
            print(pkg, md.version(pkg))
        except md.PackageNotFoundError:
            pass
    data_model, solver, solver_settings = _cuopt()
    from cuopt.linear_programming.solver import solver_parameters as P
    print("parameters:", [n for n in dir(P) if n.startswith("CUOPT_") and any(k in n for k in ("METHOD", "TIME", "GAP", "PDLP", "TOL", "PRESOLVE", "BRANCH", "BATCH", "STRONG"))])
    print("DataModel:", [n for n in dir(data_model.DataModel) if n.startswith("set_")])
    print("SolverSettings:", [n for n in dir(solver_settings.SolverSettings) if not n.startswith("_")])
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
    from trnopt.problems import textbook
    from trnopt.reference import solve_highs
    lp = textbook()
    r = solve_lp(to_data_model(lp), want_warm=True)
    print("textbook LP:", r, " HiGHS:", solve_highs(lp).objective)
    lp.integer = np.ones(lp.n, dtype=bool)
    print("textbook as MIP:", solve_mip(to_data_model(lp, integer=True), time_limit=30))
