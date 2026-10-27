"""Borrowing HiGHS's presolve.

Presolve shrinks an LP before solving it: it drops redundant constraints, fixes variables whose
value is forced, merges duplicate columns, and so on. Then "postsolve" maps the answer on the
small problem back to the original variables.

Real presolve takes years of engineering (cuOpt, Gurobi and HiGHS each have their own), so we
use HiGHS's rather than writing one. Benchmarks must say so: "trnopt + HiGHS presolve".
"""

from __future__ import annotations

from dataclasses import dataclass

import highspy
import numpy as np

from .lp import LP
from .reference import highs_lp_to_lp, to_highs_lp

_S = highspy.HighsPresolveStatus


@dataclass
class PresolveOutcome:
    status: str                  # "reduced", "not_reduced", "solved", "infeasible", "unbounded_or_infeasible"
    reduced: LP | None           # the smaller LP to solve (None unless status is "reduced")
    offset: float                # constant to add to the reduced LP's objective
    rows_removed: int = 0
    cols_removed: int = 0


class Presolver:
    def __init__(self, lp: LP):
        self.lp = lp
        self._h = highspy.Highs()
        self._h.setOptionValue("output_flag", False)
        self._h.passModel(to_highs_lp(lp))
        self._h.presolve()
        status = self._h.getModelPresolveStatus()

        if status == _S.kReduced:
            reduced, offset = highs_lp_to_lp(self._h.getPresolvedLp(), name=lp.name + "_presolved")
            self.outcome = PresolveOutcome("reduced", reduced, offset, lp.m - reduced.m, lp.n - reduced.n)
        elif status == _S.kNotReduced:
            self.outcome = PresolveOutcome("not_reduced", None, 0.0)
        elif status == _S.kReducedToEmpty:
            self.outcome = PresolveOutcome("solved", None, 0.0, lp.m, lp.n)
        elif status == _S.kInfeasible:
            self.outcome = PresolveOutcome("infeasible", None, 0.0)
        elif status == _S.kUnboundedOrInfeasible:
            self.outcome = PresolveOutcome("unbounded_or_infeasible", None, 0.0)
        else:
            raise RuntimeError(f"HiGHS presolve returned {status}")

    def postsolve(self, x_reduced: np.ndarray | None) -> np.ndarray:
        """Map a solution of the reduced LP back to the original variables.

        We pass primal values only. Our dual values use a different sign convention from
        HiGHS's, and the primal solution is what users need.
        """
        sol = highspy.HighsSolution()
        sol.col_value = [] if x_reduced is None else np.asarray(x_reduced, dtype=np.float64).tolist()
        sol.value_valid = True
        sol.dual_valid = False
        self._h.postsolve(sol)   # returns a warning status for primal-only postsolve; the values are still exact
        return np.asarray(self._h.getSolution().col_value, dtype=np.float64)
