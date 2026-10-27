"""HiGHS as the ground truth, plus MPS file loading.

HiGHS is a mature open-source LP/MIP solver (it is also what scipy.optimize.linprog
uses under the hood). Its answers are what we check our PDLP solver against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import highspy
import numpy as np
import scipy.sparse as sp

from .lp import LP

_INF = highspy.kHighsInf


@dataclass
class ReferenceResult:
    status: str
    objective: float
    x: np.ndarray
    seconds: float


def to_highs_lp(lp: LP) -> highspy.HighsLp:
    """Translate our LP into the HiGHS model format (rows have lower/upper bounds)."""
    h = highspy.HighsLp()
    h.num_col_ = lp.n
    h.num_row_ = lp.m
    h.col_cost_ = lp.c
    h.col_lower_ = np.where(np.isinf(lp.lower), -_INF, lp.lower)
    h.col_upper_ = np.where(np.isinf(lp.upper), _INF, lp.upper)
    row_lower = lp.q.copy()
    row_upper = np.full(lp.m, _INF)
    row_upper[: lp.m_eq] = lp.q[: lp.m_eq]          # equality rows: lower == upper == q
    h.row_lower_ = row_lower
    h.row_upper_ = row_upper
    K = lp.K.tocsc()
    h.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    h.a_matrix_.start_ = K.indptr
    h.a_matrix_.index_ = K.indices
    h.a_matrix_.value_ = K.data
    return h


def solve_highs(lp: LP, time_limit: float = 600.0) -> ReferenceResult:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.passModel(to_highs_lp(lp))
    t0 = time.perf_counter()
    h.run()
    seconds = time.perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus())
    x = np.asarray(h.getSolution().col_value)
    return ReferenceResult(status, h.getInfo().objective_function_value, x, seconds)


def load_mps(path: str) -> LP:
    """Read an MPS file (the standard format for Netlib / MIPLIB instances) into our LP form.

    Integer markers are dropped, so a MIPLIB file gives you its LP relaxation -- exactly
    the kind of LP a MIP solver solves at each node of its search tree.
    """
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    status = h.readModel(path)
    if status == highspy.HighsStatus.kError:
        raise ValueError(f"HiGHS could not read {path}")
    lp, _offset = highs_lp_to_lp(h.getLp(), name=path.rsplit("/", 1)[-1])
    # The objective offset (a constant some MPS files include) is dropped; compare against
    # HiGHS run on the LP returned here, not on the raw file.
    return lp


def highs_lp_to_lp(m: highspy.HighsLp, name: str = "lp") -> tuple[LP, float]:
    """Convert a HiGHS model into our LP form. Returns (lp, objective_offset)."""
    n, nrow = m.num_col_, m.num_row_
    c = np.asarray(m.col_cost_, dtype=np.float64)
    offset = float(m.offset_)
    if m.sense_ == highspy.ObjSense.kMaximize:
        c, offset = -c, -offset
    A = sp.csc_matrix(
        (np.asarray(m.a_matrix_.value_), np.asarray(m.a_matrix_.index_), np.asarray(m.a_matrix_.start_)),
        shape=(nrow, n),
    ).tocsr()
    rl = np.asarray(m.row_lower_, dtype=np.float64)
    ru = np.asarray(m.row_upper_, dtype=np.float64)
    rl[rl <= -_INF] = -np.inf
    ru[ru >= _INF] = np.inf

    # Each HiGHS row is  rl <= a @ x <= ru.  Split into our equality / ">=" rows.
    eq = rl == ru
    has_lo = ~eq & np.isfinite(rl)
    has_hi = ~eq & np.isfinite(ru)
    K = sp.vstack([A[eq], A[has_lo], -A[has_hi]]).tocsr()
    q = np.concatenate([rl[eq], rl[has_lo], -ru[has_hi]])

    lower = np.asarray(m.col_lower_, dtype=np.float64)
    upper = np.asarray(m.col_upper_, dtype=np.float64)
    lower[lower <= -_INF] = -np.inf
    upper[upper >= _INF] = np.inf
    integ = None
    if len(m.integrality_):
        integ = np.array([t == highspy.HighsVarType.kInteger for t in m.integrality_])
    return LP(c, K, q, int(eq.sum()), lower, upper, name=name, integer=integ), offset


def solve_highs_mip(lp: LP, time_limit: float = 600.0, threads: int | None = None) -> ReferenceResult:
    """HiGHS's full MIP solver (presolve, cuts, heuristics, branch-and-bound) on lp, honoring lp.integer.

    threads: leave None unless this runs in its own process. HiGHS sizes one thread pool per process
    with the first solver created, and any later solver asking for a different size fails ("Not Set").
    """
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    if threads:
        h.setOptionValue("threads", int(threads))
    model = highspy.HighsModel()
    model.lp_ = to_highs_lp(lp)
    if lp.integer is not None:
        model.lp_.integrality_ = [highspy.HighsVarType.kInteger if i else highspy.HighsVarType.kContinuous
                                  for i in lp.integer]
    h.passModel(model)
    t0 = time.perf_counter()
    h.run()
    seconds = time.perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus())
    x = np.asarray(h.getSolution().col_value)
    return ReferenceResult(status, h.getInfo().objective_function_value, x, seconds)
