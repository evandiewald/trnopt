"""The LP problem format used throughout this project.

Every LP is stored in one shape:

    minimize    c @ x
    subject to  K[:m_eq] @ x == q[:m_eq]      (equality rows come first)
                K[m_eq:] @ x >= q[m_eq:]      (then ">=" rows)
                lower <= x <= upper           (variable bounds, may be +/- inf)

A "<=" constraint is stored as a ">=" row by flipping signs (a @ x <= b  ->  -a @ x >= -b).
Maximization is stored as minimizing -c. Keeping a single shape means the solver
only has to handle one case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class LP:
    c: np.ndarray            # (n,) objective coefficients
    K: sp.csr_matrix         # (m, n) constraint matrix, equality rows first
    q: np.ndarray            # (m,) right-hand side
    m_eq: int                # number of equality rows at the top of K
    lower: np.ndarray        # (n,) variable lower bounds (-inf allowed)
    upper: np.ndarray        # (n,) variable upper bounds (+inf allowed)
    name: str = "lp"
    integer: np.ndarray | None = None   # (n,) bool: which variables a MIP requires to be whole numbers.
                                        # The LP solver ignores it (it solves the relaxation); branching uses it.

    def __post_init__(self) -> None:
        self.c = np.asarray(self.c, dtype=np.float64)
        self.K = sp.csr_matrix(self.K, dtype=np.float64)
        self.q = np.asarray(self.q, dtype=np.float64)
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)
        m, n = self.K.shape
        if self.c.shape != (n,) or self.lower.shape != (n,) or self.upper.shape != (n,):
            raise ValueError("c, lower and upper must have length n = K.shape[1]")
        if self.q.shape != (m,):
            raise ValueError("q must have length m = K.shape[0]")
        if not 0 <= self.m_eq <= m:
            raise ValueError("m_eq must be between 0 and the number of rows")
        if self.integer is not None:
            self.integer = np.asarray(self.integer, dtype=bool)
            if self.integer.shape != (n,):
                raise ValueError("integer must have length n")
        if np.any(self.lower > self.upper):
            raise ValueError("some lower bound exceeds its upper bound")

    @property
    def n(self) -> int:
        return self.K.shape[1]

    @property
    def m(self) -> int:
        return self.K.shape[0]

    @classmethod
    def from_general(
        cls,
        c,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        lower=None,
        upper=None,
        maximize: bool = False,
        name: str = "lp",
    ) -> "LP":
        """Build from the familiar scipy.linprog-style form (A_ub @ x <= b_ub, A_eq @ x == b_eq).

        If maximize=True the stored objective is -c, so remember to negate the objective
        value when reporting it.
        """
        c = np.asarray(c, dtype=np.float64)
        n = c.size
        blocks, rhs = [], []
        m_eq = 0
        if A_eq is not None:
            A_eq = sp.csr_matrix(A_eq)
            blocks.append(A_eq)
            rhs.append(np.asarray(b_eq, dtype=np.float64))
            m_eq = A_eq.shape[0]
        if A_ub is not None:
            blocks.append(-sp.csr_matrix(A_ub))           # a @ x <= b  ->  -a @ x >= -b
            rhs.append(-np.asarray(b_ub, dtype=np.float64))
        K = sp.vstack(blocks).tocsr() if blocks else sp.csr_matrix((0, n))
        q = np.concatenate(rhs) if rhs else np.zeros(0)
        lower = np.zeros(n) if lower is None else np.broadcast_to(lower, (n,)).astype(np.float64)
        upper = np.full(n, np.inf) if upper is None else np.broadcast_to(upper, (n,)).astype(np.float64)
        return cls(-c if maximize else c, K, q, m_eq, lower, upper, name)
