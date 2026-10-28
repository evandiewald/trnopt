"""MIP presolve by HiGHS: shrink the problem before branch-and-bound, map the answer back afterwards.

Presolve is the first thing every serious MIP solver does: remove fixed and redundant variables and rows,
tighten bounds and coefficients, detect easy infeasibility. HiGHS's MIP presolve runs here (cuOpt uses PaPILO
for the same job); on MIPLIB it cuts some instances by 2-4x (neos-933966: 180k -> 48k nonzeros; berkel:
1848x1533 -> 609x885), while the generated families are left almost untouched.

    P = HighsPresolve(lp)
    if P.status == "reduced":   solve P.reduced, then x = P.postsolve(x_reduced)
    objective of the original = objective of the reduced + P.offset

HiGHS postsolve of a MIP solution recovers primal values only (it reports a warning, which is expected).
"""

from __future__ import annotations

import highspy
import numpy as np

from .lp import LP
from .reference import highs_lp_to_lp, to_highs_lp


class HighsPresolve:
    """status: "reduced", "not_reduced", "empty" (presolve solved it) or "infeasible"."""

    def __init__(self, lp: LP, time_limit: float = 300.0):
        self.lp = lp
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(time_limit))
        hl = to_highs_lp(lp)
        if lp.integer is not None:
            hl.integrality_ = [highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous
                               for b in lp.integer]
        h.passModel(hl)
        h.presolve()
        self._h = h
        ps = h.getModelPresolveStatus()
        P = h.getPresolvedLp()
        if ps == highspy.HighsPresolveStatus.kInfeasible or ps == highspy.HighsPresolveStatus.kUnboundedOrInfeasible:
            self.status, self.reduced, self.offset = "infeasible", None, 0.0
        elif ps == highspy.HighsPresolveStatus.kReducedToEmpty or P.num_col_ == 0:
            self.status, self.reduced, self.offset = "empty", None, float(P.offset_)
        elif ps == highspy.HighsPresolveStatus.kNotReduced:
            self.status, self.reduced, self.offset = "not_reduced", lp, 0.0
        else:
            red, offset = highs_lp_to_lp(P, name=f"{lp.name}_presolved")
            if red.integer is not None and not red.integer.any():
                red.integer = None
            self.status, self.reduced, self.offset = "reduced", red, float(offset)

    def summary(self) -> str:
        lp, r = self.lp, self.reduced
        if self.status != "reduced":
            return self.status
        return f"{lp.m}x{lp.n} ({lp.K.nnz} nnz) -> {r.m}x{r.n} ({r.K.nnz} nnz)"

    def postsolve(self, x_reduced: np.ndarray | None) -> np.ndarray:
        """Solution of the reduced problem -> solution of the original."""
        if self.status == "not_reduced":
            return np.asarray(x_reduced, dtype=np.float64)
        sol = highspy.HighsSolution()
        sol.col_value = [] if x_reduced is None else [float(v) for v in x_reduced]
        sol.value_valid = True
        self._h.postsolve(sol)
        return np.asarray(self._h.getSolution().col_value, dtype=np.float64)
