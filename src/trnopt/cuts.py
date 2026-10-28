"""Gomory mixed-integer (GMI) cuts at the root: tighten the LP relaxation before branch-and-bound.

A cut is a new constraint that every whole-number solution satisfies but the current fractional LP solution
does not. Adding cuts raises the LP bound, so branch-and-bound proves optimality with far fewer nodes. GMI cuts
are the classic general-purpose family (Gomory 1960; the mixed-integer form is what every modern solver uses):

1. Solve the LP relaxation (HiGHS dual simplex, exact, float64). Take an integer variable x_b that is basic and
   fractional, say x_b* = 3.4. Its simplex tableau row expresses it through the nonbasic variables, each sitting
   at one of its bounds:   x_b + sum_j alpha_j v_j = 0   (over structural variables and row activities).
2. Measure every nonbasic variable as a distance from its bound, t_j >= 0 (t_j = v_j - lower_j or upper_j - v_j).
   Then x_b + sum_j alpha'_j t_j = x_b*, and at the LP solution all t_j are 0.
3. With f0 = frac(x_b*) and f_j = frac(alpha'_j), the GMI inequality
       sum_{integer j, f_j <= f0} f_j/f0 t_j + sum_{integer j, f_j > f0} (1-f_j)/(1-f0) t_j
     + sum_{continuous j, alpha'_j > 0} alpha'_j/f0 t_j + sum_{continuous j, alpha'_j < 0} -alpha'_j/(1-f0) t_j >= 1
   holds for every point where x_b is a whole number, and is violated (0 >= 1) by the LP solution.
4. Substitute back to the original variables (row activities r_i = a_i x) to get a row  gamma x >= delta.

HiGHS conventions (checked numerically): basis columns come from [A I] with logical s = -r, so the tableau row is
abar x - binv r = 0 with abar = getReducedRow(k), binv = getBasisInverseRow(k).

Cuts are numerically delicate, so each one is cleaned (tiny coefficients removed with a bound-based rhs
correction), rejected if its coefficient range is extreme or it barely cuts off the LP point, and checked
against near-duplicates. Rounds repeat (solve -> cut -> resolve) until the bound stops moving. Cuts that are slack
at the final LP are dropped. Every cut is globally valid, so branch-and-bound runs on the strengthened model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import highspy
import numpy as np
import scipy.sparse as sp

from .lp import LP
from .reference import to_highs_lp

BS = highspy.HighsBasisStatus


@dataclass
class CutInfo:
    rounds: int = 0
    generated: int = 0
    kept: int = 0
    bound_before: float = float("nan")
    bound_after: float = float("nan")
    seconds: float = 0.0


def _gmi_cuts(h, lp: LP, K_all: sp.csr_matrix, row_lo, row_up, integer, max_cuts: int,
              min_frac: float = 0.01, max_dyn: float = 1e6, min_eff: float = 1e-5, max_par: float = 0.99):
    """GMI cuts from the current optimal basis of h. Returns a list of (gamma (n,), delta, efficacy)."""
    n = lp.n
    sol = h.getSolution()
    x = np.asarray(sol.col_value, dtype=np.float64)
    basis = h.getBasis()
    cs = np.array([int(s) for s in basis.col_status])
    rs = np.array([int(s) for s in basis.row_status])
    lower, upper = lp.lower, lp.upper
    _, basic = h.getBasicVariables()
    basic = np.asarray(basic)

    col_lo, col_up, col_zero = cs == int(BS.kLower), cs == int(BS.kUpper), cs == int(BS.kZero)
    col_nb = (cs != int(BS.kBasic)) & (lower < upper)                       # fixed variables: t_j == 0
    row_lo_s, row_up_s = rs == int(BS.kLower), rs == int(BS.kUpper)
    row_nb = (rs != int(BS.kBasic)) & (row_lo < row_up)                      # equality rows: t_i == 0
    int_bound = integer & np.where(col_up, np.isclose(upper, np.round(upper)), np.isclose(lower, np.round(lower)))
    KT = K_all.T.tocsr()

    cand = [(k, j) for k, j in enumerate(basic) if j >= 0 and integer[j]]
    cand = [(k, j) for k, j in cand if min_frac < x[j] - np.floor(x[j]) < 1 - min_frac]
    # most fractional first: those tend to give the deepest cuts
    cand.sort(key=lambda kj: -min(x[kj[1]] - np.floor(x[kj[1]]), np.ceil(x[kj[1]]) - x[kj[1]]))
    out = []
    for k, j in cand[: 4 * max_cuts]:
        f0 = x[j] - np.floor(x[j])
        _, abar = h.getReducedRow(int(k))
        _, binv = h.getBasisInverseRow(int(k))
        alpha = np.asarray(abar, dtype=np.float64)
        alpha[j] = 0.0
        alpha[np.abs(alpha) < 1e-12] = 0.0
        beta = -np.asarray(binv, dtype=np.float64)                         # coefficient of row activity r_i
        beta[np.abs(beta) < 1e-12] = 0.0
        a_c = np.where(col_nb, alpha, 0.0)
        a_r = np.where(row_nb, beta, 0.0)
        if np.any(col_zero & (a_c != 0)):                                    # a free nonbasic variable: no cut
            continue
        if np.any((a_c != 0) & ~(col_lo | col_up)) or np.any((a_r != 0) & ~(row_lo_s | row_up_s)):
            continue
        ap_c = np.where(col_up, -a_c, a_c)                                   # alpha' (t measured from the bound)
        ap_r = np.where(row_up_s, -a_r, a_r)
        # pi: integer structural t_j (integer variable at an integral bound), else continuous
        fj = ap_c - np.floor(ap_c)
        pi_int = np.where(fj <= f0, fj / f0, (1 - fj) / (1 - f0))
        pi_cont_c = np.where(ap_c > 0, ap_c / f0, -ap_c / (1 - f0))
        pi_c = np.where(int_bound, pi_int, pi_cont_c)
        pi_c = np.where(a_c != 0, pi_c, 0.0)
        pi_r = np.where(ap_r > 0, ap_r / f0, -ap_r / (1 - f0))
        pi_r = np.where(a_r != 0, pi_r, 0.0)
        # back to original variables: sum pi t >= 1
        lb_c = np.where(np.isfinite(lower), lower, 0.0)
        ub_c = np.where(np.isfinite(upper), upper, 0.0)
        gamma = np.where(col_up, -pi_c, pi_c)
        const = float(np.sum(np.where(col_up, pi_c * ub_c, -pi_c * lb_c)))
        rlo = np.where(np.isfinite(row_lo), row_lo, 0.0)
        rup = np.where(np.isfinite(row_up), row_up, 0.0)
        g_r = np.where(row_up_s, -pi_r, pi_r)
        gamma = gamma + KT @ g_r
        const += float(np.sum(np.where(row_up_s, pi_r * rup, -pi_r * rlo)))
        delta = 1.0 - const
        # clean: drop tiny coefficients, moving them to the rhs with a bound (keeps the cut valid)
        mx = np.abs(gamma).max(initial=0.0)
        if mx == 0 or not np.isfinite(mx) or not np.isfinite(delta):
            continue
        tiny = (gamma != 0) & (np.abs(gamma) < 1e-9 * mx)
        if np.any(tiny):
            worst = np.maximum(gamma[tiny] * lower[tiny], gamma[tiny] * upper[tiny])
            if not np.all(np.isfinite(worst)):
                continue
            delta -= float(worst.sum())
            gamma[tiny] = 0.0
        nzv = np.abs(gamma[gamma != 0])
        if nzv.size == 0 or nzv.max() / nzv.min() > max_dyn:
            continue
        viol = delta - float(gamma @ x)
        norm = float(np.linalg.norm(gamma))
        eff = viol / norm
        if viol <= 1e-6 * (1 + abs(delta)) or eff < min_eff:
            continue
        out.append((gamma, delta, eff))
    # pick the most effective, skipping near-parallel duplicates
    out.sort(key=lambda c: -c[2])
    chosen = []
    for g, d, e in out:
        gn = g / np.linalg.norm(g)
        if all(abs(gn @ (c[0] / np.linalg.norm(c[0]))) < max_par for c in chosen):
            chosen.append((g, d, e))
        if len(chosen) >= max_cuts:
            break
    return chosen


def root_cuts(lp: LP, rounds: int = 5, max_cuts: int = 100, time_limit: float = 60.0, min_gain: float = 1e-4,
              verbose: bool = False) -> tuple[LP, CutInfo]:
    """Add rounds of GMI cuts to lp's root relaxation. Returns (lp with cut rows appended as ">=" rows, info)."""
    t0 = time.perf_counter()
    info = CutInfo()
    if lp.integer is None or not lp.integer.any():
        return lp, info
    import dataclasses
    integer = np.asarray(lp.integer, dtype=bool)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(to_highs_lp(dataclasses.replace(lp, integer=None)))
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return lp, info
    info.bound_before = info.bound_after = h.getInfo().objective_function_value
    rows_K = [lp.K.tocsr()]
    row_lo = np.where(np.arange(lp.m) < lp.m_eq, lp.q, lp.q).astype(np.float64)
    row_up = np.where(np.arange(lp.m) < lp.m_eq, lp.q, np.inf).astype(np.float64)
    cut_g, cut_d = [], []
    stalls = 0
    for rd in range(rounds):
        if time.perf_counter() - t0 > time_limit:
            break
        K_all = sp.vstack(rows_K).tocsr()
        cuts = _gmi_cuts(h, lp, K_all, row_lo, row_up, integer, max_cuts)
        info.rounds = rd + 1
        if not cuts:
            break
        info.generated += len(cuts)
        G = sp.csr_matrix(np.array([g for g, _, _ in cuts]))
        D = np.array([d for _, d, _ in cuts])
        h.addRows(G.shape[0], D, np.full(G.shape[0], highspy.kHighsInf), G.nnz,
                  G.indptr[:-1].astype(np.int32), G.indices.astype(np.int32), G.data)
        rows_K.append(G)
        row_lo = np.concatenate([row_lo, D])
        row_up = np.concatenate([row_up, np.full(len(D), np.inf)])
        cut_g.append(G)
        cut_d.append(D)
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            break
        new = h.getInfo().objective_function_value
        gain = new - info.bound_after
        if verbose:
            print(f"  cuts round {rd + 1}: {len(cuts)} cuts, root bound {info.bound_after:.6g} -> {new:.6g}")
        info.bound_after = new
        stalls = stalls + 1 if gain <= min_gain * max(1.0, abs(new)) else 0
        if stalls >= 2:
            break
    if not cut_g:
        info.seconds = time.perf_counter() - t0
        return lp, info
    # keep only cuts that are tight at the final LP solution (the slack ones just make every node LP bigger)
    G = sp.vstack(cut_g).tocsr()
    D = np.concatenate(cut_d)
    x = np.asarray(h.getSolution().col_value)
    act = G @ x
    keep = act <= D + 1e-6 * (1 + np.abs(D))
    G, D = G[keep], D[keep]
    info.kept = int(keep.sum())
    out = dataclasses.replace(lp, K=sp.vstack([lp.K, G]).tocsr(), q=np.concatenate([lp.q, D]),
                              name=lp.name)
    info.seconds = time.perf_counter() - t0
    return out, info
