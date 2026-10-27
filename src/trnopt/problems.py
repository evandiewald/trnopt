"""Test problems: a hand-checkable toy, random LPs with a known optimum, and a synthetic loan-selection relaxation."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .lp import LP


def textbook() -> LP:
    """Classic two-variable example you can solve with a pencil.

    maximize 3x + 5y  s.t.  x <= 4,  2y <= 12,  3x + 2y <= 18,  x, y >= 0
    Optimum: x = 2, y = 6, objective 36 (stored as minimizing -3x - 5y, i.e. -36).
    """
    return LP.from_general(
        c=[3, 5],
        A_ub=[[1, 0], [0, 2], [3, 2]],
        b_ub=[4, 12, 18],
        maximize=True,
        name="textbook",
    )


def random_lp(n: int = 200, m_eq: int = 40, m_ineq: int = 80, density: float = 0.05, seed: int = 0) -> LP:
    """A random sparse LP that is guaranteed to have an optimal solution.

    We pick a feasible point first and set the right-hand side from it (so the LP is
    feasible), then build the cost vector from nonnegative dual prices (so it is bounded).
    """
    rng = np.random.default_rng(seed)
    m = m_eq + m_ineq
    K = sp.random(m, n, density=density, random_state=rng, data_rvs=lambda k: rng.normal(size=k), format="csr")

    # Some variables boxed in [0, 10], the rest only x >= 0.
    lower = np.zeros(n)
    upper = np.where(rng.random(n) < 0.5, 10.0, np.inf)
    x_feas = rng.uniform(0, 5, n)

    Kx = K @ x_feas
    slack = np.where(rng.random(m_ineq) < 0.5, 0.0, rng.uniform(0, 1, m_ineq))  # half the '>=' rows tight
    q = np.concatenate([Kx[:m_eq], Kx[m_eq:] - slack])

    # Bounded objective: c = K^T y + r with y >= 0 on '>=' rows and r >= 0 wherever x has no upper bound.
    y = np.concatenate([rng.normal(size=m_eq), rng.uniform(0, 1, m_ineq)])
    r = np.where(np.isinf(upper), rng.uniform(0, 1, n), rng.normal(size=n))
    c = K.T @ y + r
    return LP(c, K, q, m_eq, lower, upper, name=f"random_n{n}_m{m}_s{seed}")


def loan_selection(n_loans: int = 2000, seed: int = 0) -> LP:
    """LP relaxation of a toy loan-selection problem (synthetic data, public-safe).

    Choose a fraction x_i in [0, 1] of each loan to buy, maximizing expected net return, subject to:
      * a total budget;
      * a cap on how much of the budget can go to each credit grade;
      * a cap on how much of the budget can go to any one state;
      * a minimum dollar-weighted average FICO score.

    In the real (MIP) version x_i would have to be 0 or 1. This relaxation is exactly
    the LP a MIP solver solves at the root of its search tree.
    """
    rng = np.random.default_rng(seed)
    grades = rng.integers(0, 5, n_loans)                       # A..E
    states = rng.integers(0, 10, n_loans)
    amount = rng.uniform(5_000, 50_000, n_loans)
    rate = np.array([0.07, 0.10, 0.14, 0.19, 0.25])[grades] + rng.normal(0, 0.01, n_loans)
    loss = np.array([0.01, 0.03, 0.06, 0.10, 0.16])[grades] + rng.normal(0, 0.005, n_loans)
    fico = np.array([780, 740, 700, 665, 630])[grades] + rng.normal(0, 15, n_loans)
    net_return = amount * (rate - loss)

    budget = 0.3 * amount.sum()
    grade_cap = np.array([1.0, 0.6, 0.4, 0.25, 0.15])         # share of budget per grade
    state_cap = 0.2
    min_fico = 710.0

    rows, rhs = [amount], [budget]
    for g in range(5):
        rows.append(amount * (grades == g))
        rhs.append(grade_cap[g] * budget)
    for s in range(10):
        rows.append(amount * (states == s))
        rhs.append(state_cap * budget)
    rows.append(-amount * (fico - min_fico))                   # sum a_i (fico_i - 710) x_i >= 0
    rhs.append(0.0)

    A_ub = sp.csr_matrix(np.vstack(rows))
    lp = LP.from_general(
        c=net_return, A_ub=A_ub, b_ub=np.array(rhs), lower=0.0, upper=1.0,
        maximize=True, name=f"loan_selection_{n_loans}",
    )
    lp.integer = np.ones(n_loans, dtype=bool)        # the real problem: buy each loan or don't
    return lp


def infeasible_lp(seed: int = 0) -> LP:
    """A random LP made impossible on purpose.

    We take a random feasible LP and add two rows that contradict each other:
    a @ x >= b + 1 and a @ x <= b, for a random dense row a.
    """
    base = random_lp(n=100, m_eq=20, m_ineq=40, density=0.1, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    a = rng.normal(size=base.n)
    b = float(rng.normal())
    K = sp.vstack([base.K, sp.csr_matrix(a), sp.csr_matrix(-a)]).tocsr()
    q = np.concatenate([base.q, [b + 1.0, -b]])
    return LP(base.c, K, q, base.m_eq, base.lower, base.upper, name=f"infeasible_s{seed}")


def unbounded_lp(seed: int = 0) -> LP:
    """A random feasible LP whose objective can improve forever.

    We add one new variable with no upper bound, a negative cost (so increasing it always
    lowers the objective), and nonnegative coefficients in the ">=" rows only (so increasing
    it never breaks a constraint).
    """
    base = random_lp(n=100, m_eq=20, m_ineq=40, density=0.1, seed=seed)
    rng = np.random.default_rng(seed + 2000)
    col = np.zeros(base.m)
    col[base.m_eq :] = rng.uniform(0, 1, base.m - base.m_eq) * (rng.random(base.m - base.m_eq) < 0.3)
    K = sp.hstack([base.K, sp.csr_matrix(col).T]).tocsr()
    return LP(
        np.append(base.c, -1.0), K, base.q, base.m_eq,
        np.append(base.lower, 0.0), np.append(base.upper, np.inf), name=f"unbounded_s{seed}",
    )


# ---- batch workloads: many LPs sharing one constraint matrix ----------------------------------

def branching_variants(lp: LP, x: np.ndarray, k: int, tol: float = 1e-6):
    """Strong-branching batch: for the k most fractional integer variables in x, a "down" LP
    (x_j <= floor) and an "up" LP (x_j >= ceil). Returns (lower, upper, labels), arrays of shape
    (2k, n), ready for solve_pdlp_batch.

    This is how a MIP solver picks which variable to branch on: try both directions for several
    candidates and see which moves the bound most. Every variant shares K, so they batch.
    """
    integer = lp.integer if lp.integer is not None else np.ones(lp.n, dtype=bool)
    frac = np.abs(x - np.round(x))
    frac[~integer] = 0.0
    cand = [j for j in np.argsort(-frac) if frac[j] > tol][:k]
    lowers, uppers, labels = [], [], []
    for j in cand:
        lo, up = lp.lower.copy(), lp.upper.copy()
        up[j] = np.floor(x[j])
        lowers.append(lp.lower.copy()); uppers.append(up); labels.append((int(j), "down"))
        lo[j] = np.ceil(x[j])
        lowers.append(lo); uppers.append(lp.upper.copy()); labels.append((int(j), "up"))
    return np.array(lowers), np.array(uppers), labels


def return_scenarios(lp: LP, n_scenarios: int, spread: float = 0.2, seed: int = 0) -> np.ndarray:
    """Scenario batch: the objective (e.g. loan returns) perturbed by up to +/- spread, one row per scenario."""
    rng = np.random.default_rng(seed)
    return lp.c[None, :] * (1.0 + spread * rng.uniform(-1, 1, size=(n_scenarios, lp.n)))
