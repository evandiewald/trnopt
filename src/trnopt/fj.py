"""Feasibility Jump: an LP-free local search for a first feasible MIP solution (CPU, numba).

Luteberget & Sartor, "Feasibility Jump: an LP-free Lagrangian MIP heuristic" (Math. Prog. Comp. 2023). HiGHS and
cuOpt both run it. Why it's here: on the big MIPLIB instances every other trnopt heuristic waits for the root LP,
and there the root LP is the slow part, so nothing was found in 120 s. This needs no LP at all.

How it works, in plain terms:
  - Every constraint is written as  a x <= b  (">=" rows negated, equalities as two rows), with a weight w_i = 1.
  - The "badness" of a point is the weighted total violation  sum_i w_i * max(0, a_i x - b_i).
  - Start from a simple point (every variable at the bound nearest zero). Repeatedly:
      pick a violated constraint at random; for each variable in it, find the single value that variable could
      "jump" to that lowers the badness most (holding everything else fixed - a 1-D convex problem, solved exactly
      by sorting breakpoints); make the best such jump.
  - When no jump helps (a local minimum), raise the weights of all violated constraints. That changes the
    landscape: constraints that keep being violated matter more, until some jump helps again.
  - Once the point is feasible it is a solution. To look for a better one, an objective row  c x <= best - delta
    is switched on and the search continues.

Everything runs in numba-compiled loops: millions of tiny moves, no vectorisable structure - CPU work by nature.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .lp import LP

try:
    import numba
    from numba import njit
except ImportError:                                                 # numba is optional
    numba = None

    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not (len(a) == 1 and callable(a[0])) else a[0]


VIOL_TOL = 1e-6


@njit(cache=True, nogil=True)
def _viol(act, b, i):
    return act[i] - b[i] > VIOL_TOL * (1.0 + abs(b[i]))


@njit(cache=True, nogil=True)
def _jump_value(j, x, act, b, w, lo, up, is_int, cptr, crow, cval, keys, slopes):
    """Value of x_j minimising sum_i w_i max(0, act_i + a_ij (v - x_j) - b_i) over [lo_j, up_j]; returns (v, delta)."""
    k = 0
    slope = 0.0
    for p in range(cptr[j], cptr[j + 1]):
        i = crow[p]
        a = cval[p]
        if a == 0.0 or w[i] == 0.0 or not np.isfinite(b[i]):
            continue
        t = x[j] + (b[i] - act[i]) / a                              # row i is tight at v = t
        keys[k] = t
        slopes[k] = w[i] * abs(a)
        k += 1
        if a < 0:
            slope -= w[i] * abs(a)                                  # as v -> -inf, rows with a < 0 are violated
    # sweep breakpoints in increasing order; the slope rises by w|a| at each; minimum where it turns positive
    order = np.argsort(keys[:k])
    v = lo[j]
    if slope >= 0.0:
        v = lo[j]
    else:
        v = up[j]
        for q in range(k):
            slope += slopes[order[q]]
            if slope >= 0.0:
                v = keys[order[q]]
                break
    if v < lo[j]:
        v = lo[j]
    if v > up[j]:
        v = up[j]
    best_v = v
    best_d = np.inf
    # integers: try floor and ceil of the continuous minimiser (convex, so one of them is optimal)
    n_try = 2 if is_int[j] else 1
    for tr in range(n_try):
        if is_int[j]:
            cand = np.floor(v + 1e-9) if tr == 0 else np.ceil(v - 1e-9)
            if cand < lo[j] or cand > up[j]:
                continue
        else:
            cand = v
        if not np.isfinite(cand) or abs(cand - x[j]) < 1e-12:
            continue
        d = 0.0
        for p in range(cptr[j], cptr[j + 1]):
            i = crow[p]
            a = cval[p]
            old = act[i] - b[i]
            new = old + a * (cand - x[j])
            if np.isfinite(b[i]):
                d += w[i] * (max(0.0, new) - max(0.0, old))
        if d < best_d:
            best_d = d
            best_v = cand
    return best_v, best_d


@njit(cache=True, nogil=True)
def _apply(j, v, x, act, b, cptr, crow, cval, viol_list, viol_pos, nviol):
    dx = v - x[j]
    x[j] = v
    for p in range(cptr[j], cptr[j + 1]):
        i = crow[p]
        was = viol_pos[i] >= 0
        act[i] += cval[p] * dx
        now = _viol(act, b, i)
        if now and not was:
            viol_pos[i] = nviol
            viol_list[nviol] = i
            nviol += 1
        elif was and not now:
            pos = viol_pos[i]
            last = viol_list[nviol - 1]
            viol_list[pos] = last
            viol_pos[last] = pos
            viol_pos[i] = -1
            nviol -= 1
    return nviol


@njit(cache=True, nogil=True)
def _run(steps, seed, x, act, b, w, lo, up, is_int, rptr, rcol, cptr, crow, cval, viol_list, viol_pos, nviol,
         sample, keys, slopes):
    """Up to `steps` iterations. Returns (steps used, nviol). Stops early when nothing is violated."""
    np.random.seed(seed)
    for s in range(steps):
        if nviol == 0:
            return s, nviol
        i = viol_list[np.random.randint(nviol)]
        start, end = rptr[i], rptr[i + 1]
        length = end - start
        best_j = -1
        best_v = 0.0
        best_d = 0.0
        tries = length if length <= sample else sample
        for t in range(tries):
            p = start + t if length <= sample else start + np.random.randint(length)
            j = rcol[p]
            if lo[j] == up[j]:
                continue
            v, d = _jump_value(j, x, act, b, w, lo, up, is_int, cptr, crow, cval, keys, slopes)
            if d < best_d - 1e-12:
                best_d = d
                best_j = j
                best_v = v
        if best_j >= 0:
            nviol = _apply(best_j, best_v, x, act, b, cptr, crow, cval, viol_list, viol_pos, nviol)
        else:
            for q in range(nviol):                                  # local minimum: reweight what's violated
                w[viol_list[q]] += 1.0
    return steps, nviol


@dataclass
class FJResult:
    solutions: list = field(default_factory=list)   # (objective, x), best first
    steps: int = 0
    seconds: float = 0.0
    best_violation: float = np.inf                   # unweighted total violation of the best infeasible point seen


def feasibility_jump(lp: LP, time_limit: float = 10.0, lower=None, upper=None, seed: int = 0,
                     improve: bool = True, max_solutions: int = 20, chunk: int = 1000, sample: int = 64,
                     verbose: bool = False) -> FJResult:
    """Feasibility Jump on lp (lp.integer marks integer variables). Returns feasible points found, best first.

    improve: after a solution, keep searching with the objective row c x <= best - delta switched on.
    """
    if numba is None:
        raise ImportError("feasibility_jump needs numba (pip install numba)")
    t0 = time.perf_counter()
    res = FJResult()
    n = lp.n
    lo = np.asarray(lp.lower if lower is None else lower, dtype=np.float64).copy()
    up = np.asarray(lp.upper if upper is None else upper, dtype=np.float64).copy()
    is_int = np.zeros(n, dtype=np.bool_) if lp.integer is None else np.asarray(lp.integer, dtype=np.bool_)
    lo[is_int] = np.ceil(lo[is_int] - 1e-9)
    up[is_int] = np.floor(up[is_int] + 1e-9)
    # rows as "<=": >= rows negated, equalities both ways, plus the objective row (inactive until a solution)
    K = sp.csr_matrix(lp.K)
    Keq = K[: lp.m_eq]
    A = sp.vstack([-K, Keq, sp.csr_matrix(lp.c.reshape(1, -1))]).tocsr()
    A.eliminate_zeros()
    b = np.concatenate([-lp.q, lp.q[: lp.m_eq], [np.inf]]).astype(np.float64)
    obj_row = A.shape[0] - 1
    Ac = A.tocsc()
    rptr, rcol = A.indptr.astype(np.int64), A.indices.astype(np.int64)
    cptr, crow, cval = Ac.indptr.astype(np.int64), Ac.indices.astype(np.int64), Ac.data.astype(np.float64)
    # start at the bound closest to zero (0 when allowed)
    x = np.clip(np.zeros(n), lo, up)
    x = np.where(np.isfinite(x), x, 0.0)
    act = A @ x
    w = np.ones(A.shape[0])
    R = A.shape[0]
    viol_list = np.zeros(R, dtype=np.int64)
    viol_pos = np.full(R, -1, dtype=np.int64)
    nviol = 0
    for i in range(R):
        if act[i] - b[i] > VIOL_TOL * (1 + abs(b[i])) and np.isfinite(b[i]):
            viol_pos[i] = nviol
            viol_list[nviol] = i
            nviol += 1
    maxk = int(np.diff(cptr).max(initial=1)) + 1
    keys, slopes = np.zeros(maxk), np.zeros(maxk)
    rng = np.random.default_rng(seed)
    cn = np.abs(lp.c).max(initial=0.0)
    while time.perf_counter() - t0 < time_limit:
        tc = time.perf_counter()
        used, nviol = _run(chunk, int(rng.integers(2**31)), x, act, b, w, lo, up, is_int, rptr, rcol, cptr, crow,
                           cval, viol_list, viol_pos, nviol, sample, keys, slopes)
        res.steps += used
        dt = time.perf_counter() - tc
        if used == chunk:                                           # aim for ~20 ms per call: responsive time checks
            chunk = int(min(1_000_000, max(100, chunk * 0.02 / max(dt, 1e-6))))
        if nviol == 0:
            obj = float(lp.c @ x)
            res.solutions.append((obj, x.copy()))
            if verbose:
                print(f"    FJ: solution {obj:.6g} after {res.steps} steps, {time.perf_counter() - t0:.2f}s")
            if not improve or cn == 0 or len(res.solutions) >= max_solutions:
                break
            # demand a strictly better objective: switch the objective row on (integer objectives step by 1)
            int_obj = bool(np.all(is_int[lp.c != 0]) and np.allclose(lp.c, np.round(lp.c)))
            delta = 1.0 if int_obj else 1e-4 * max(1.0, abs(obj))
            b[obj_row] = obj - delta
            act[obj_row] = obj
            if act[obj_row] - b[obj_row] > VIOL_TOL * (1 + abs(b[obj_row])):
                viol_pos[obj_row] = nviol
                viol_list[nviol] = obj_row
                nviol = 1
    if nviol and res.solutions == []:
        r = act - b
        res.best_violation = float(np.maximum(r[:obj_row], 0).sum())
    res.solutions.sort(key=lambda s: s[0])
    res.seconds = time.perf_counter() - t0
    return res
