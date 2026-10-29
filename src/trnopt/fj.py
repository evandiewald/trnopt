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


@njit(cache=True, nogil=True)
def _lift(x, act, b, lo, up, is_int, c, order, cptr, crow, cval, obj_row):
    """Lift moves: push each variable as far as it can go in the direction that lowers the objective without
    breaking any row (every row a x <= b keeps its slack >= 0) or a bound. Sweeps until nothing moves.
    Returns the total objective decrease. Keeps a feasible x feasible."""
    total = 0.0
    moved = 1
    while moved:
        moved = 0
        for q in range(len(order)):
            j = order[q]
            d = -1.0 if c[j] > 0 else 1.0
            room = (up[j] - x[j]) if d > 0 else (x[j] - lo[j])
            if room <= 1e-9:
                continue
            step = room
            for p in range(cptr[j], cptr[j + 1]):
                i = crow[p]
                if i == obj_row or not np.isfinite(b[i]):
                    continue
                ad = cval[p] * d
                if ad > 0.0:
                    slack = b[i] - act[i]
                    if slack < 0.0:
                        slack = 0.0
                    s_ = slack / ad
                    if s_ < step:
                        step = s_
            if is_int[j]:
                step = np.floor(step + 1e-9)
            if not np.isfinite(step) or step <= 1e-9:
                continue
            x[j] += d * step
            for p in range(cptr[j], cptr[j + 1]):
                act[crow[p]] += cval[p] * d * step
            total += abs(c[j]) * step
            moved += 1
    return total


@dataclass
class FJResult:
    solutions: list = field(default_factory=list)   # (objective, x), best first
    steps: int = 0
    seconds: float = 0.0
    best_violation: float = np.inf                   # unweighted total violation where the search ended (if infeasible)
    x_last: np.ndarray | None = None                 # where the search ended (feasible or not)


class FeasibilityJump:
    """Feasibility Jump for one model: the row/column structure is built once, then run() as often as needed,
    from any start point and within any bounds (the feasibility pump calls it after every rounding)."""

    def __init__(self, lp: LP):
        if numba is None:
            raise ImportError("feasibility_jump needs numba (pip install numba)")
        self.lp = lp
        n = lp.n
        self.is_int = np.zeros(n, dtype=np.bool_) if lp.integer is None else np.asarray(lp.integer, dtype=np.bool_)
        # rows as "<=": >= rows negated, equalities both ways, plus the objective row (inactive until a cutoff)
        K = sp.csr_matrix(lp.K)
        A = sp.vstack([-K, K[: lp.m_eq], sp.csr_matrix(lp.c.reshape(1, -1))]).tocsr()
        A.eliminate_zeros()
        self.A = A
        self.b0 = np.concatenate([-lp.q, lp.q[: lp.m_eq], [np.inf]]).astype(np.float64)
        self.obj_row = A.shape[0] - 1
        Ac = A.tocsc()
        self.rptr, self.rcol = A.indptr.astype(np.int64), A.indices.astype(np.int64)
        self.cptr, self.crow, self.cval = Ac.indptr.astype(np.int64), Ac.indices.astype(np.int64), Ac.data.astype(np.float64)
        maxk = int(np.diff(self.cptr).max(initial=1)) + 1
        self.keys, self.slopes = np.zeros(maxk), np.zeros(maxk)
        self.cn = np.abs(lp.c).max(initial=0.0)
        self.int_obj = bool(np.all(self.is_int[lp.c != 0]) and np.allclose(lp.c, np.round(lp.c)))

    def run(self, time_limit: float = 10.0, x0=None, lower=None, upper=None, cutoff: float | None = None,
            seed: int = 0, improve: bool = True, max_solutions: int = 20, chunk: int = 50, sample: int = 64,
            verbose: bool = False) -> FJResult:
        """Search from x0 (default: every variable at the bound nearest zero) within [lower, upper].

        cutoff: only accept solutions with objective <= cutoff (the objective row is switched on from the start).
        improve: after a solution, keep searching for strictly better ones.
        """
        lp = self.lp
        t0 = time.perf_counter()
        res = FJResult()
        is_int = self.is_int
        lo = np.asarray(lp.lower if lower is None else lower, dtype=np.float64).copy()
        up = np.asarray(lp.upper if upper is None else upper, dtype=np.float64).copy()
        lo[is_int] = np.ceil(lo[is_int] - 1e-9)
        up[is_int] = np.floor(up[is_int] + 1e-9)
        b = self.b0.copy()
        if cutoff is not None and np.isfinite(cutoff):
            b[self.obj_row] = cutoff
        x = np.zeros(lp.n) if x0 is None else np.array(x0, dtype=np.float64)
        x[is_int] = np.round(x[is_int])
        x = np.clip(x, lo, up)
        x = np.where(np.isfinite(x), x, np.clip(0.0, lo, up))
        x = np.where(np.isfinite(x), x, 0.0)
        act = self.A @ x
        w = np.ones(self.A.shape[0])
        R = self.A.shape[0]
        viol = np.isfinite(b) & (act - b > VIOL_TOL * (1 + np.abs(np.where(np.isfinite(b), b, 0.0))))
        viol_list = np.zeros(R, dtype=np.int64)
        viol_pos = np.full(R, -1, dtype=np.int64)
        vi = np.flatnonzero(viol)
        nviol = len(vi)
        viol_list[:nviol] = vi
        viol_pos[vi] = np.arange(nviol)
        rng = np.random.default_rng(seed)
        while time.perf_counter() - t0 < time_limit:
            tc = time.perf_counter()
            used, nviol = _run(chunk, int(rng.integers(2**31)), x, act, b, w, lo, up, is_int, self.rptr, self.rcol,
                               self.cptr, self.crow, self.cval, viol_list, viol_pos, nviol, sample, self.keys,
                               self.slopes)
            res.steps += used
            dt = time.perf_counter() - tc
            if used == chunk:                                       # aim for ~20 ms per call: responsive time checks
                # (starts at 50 moves: a move can cost ~1 ms on models with long rows, so 1000 overran short budgets)
                chunk = int(min(1_000_000, max(100, chunk * 0.02 / max(dt, 1e-6))))
            if nviol == 0:
                obj = float(lp.c @ x)
                res.solutions.append((obj, x.copy()))
                if verbose:
                    print(f"    FJ: solution {obj:.6g} after {res.steps} steps, {time.perf_counter() - t0:.2f}s")
                if not improve or self.cn == 0 or len(res.solutions) >= max_solutions:
                    break
                # demand a strictly better objective: switch the objective row on (integer objectives step by 1)
                delta = 1.0 if self.int_obj else 1e-4 * max(1.0, abs(obj))
                b[self.obj_row] = obj - delta
                if act[self.obj_row] - b[self.obj_row] > VIOL_TOL * (1 + abs(b[self.obj_row])):
                    viol_pos[self.obj_row] = nviol
                    viol_list[nviol] = self.obj_row
                    nviol = 1
        r = act - b
        res.best_violation = float(np.maximum(np.where(np.isfinite(r[: self.obj_row]), r[: self.obj_row], 0.0), 0).sum())
        res.x_last = x
        res.solutions.sort(key=lambda s: s[0])
        res.seconds = time.perf_counter() - t0
        return res

    def improve(self, x_feas, time_limit: float = 5.0, lower=None, upper=None, seed: int = 0, kick: int = 8,
                stall: float | None = None, verbose: bool = False) -> FJResult:
        """Improve a FEASIBLE point by local search (cuOpt's Local-MIP move types, on the CPU):

        * lift moves: each variable moves as far as it can in the direction that lowers the objective while
          every row stays satisfied (_lift); repeated until nothing moves. Always feasible.
        * breakthrough moves: when no lift move is left, push `kick` random objective-improving variables one
          step (integers by 1) regardless of the rows, then let Feasibility Jump repair the damage with the
          objective cutoff c x <= best - delta switched on. A successful repair is a strictly better solution;
          lift moves resume from it. A failed one is discarded and other variables are kicked next time.

        stall: stop after this many seconds without an improvement (default: max(2 s, 30% of time_limit)).
        Returns the improved solutions found, best first (empty if none beat x_feas).
        """
        lp = self.lp
        t0 = time.perf_counter()
        res = FJResult()
        rng = np.random.default_rng(seed)
        is_int = self.is_int
        lo = np.asarray(lp.lower if lower is None else lower, dtype=np.float64).copy()
        up = np.asarray(lp.upper if upper is None else upper, dtype=np.float64).copy()
        lo[is_int] = np.ceil(lo[is_int] - 1e-9)
        up[is_int] = np.floor(up[is_int] + 1e-9)
        c = np.asarray(lp.c, dtype=np.float64)
        movable = np.flatnonzero(c != 0)
        order = movable[np.argsort(-np.abs(c[movable]), kind="stable")].astype(np.int64)
        b = self.b0.copy()
        x = np.array(x_feas, dtype=np.float64)
        best = float(c @ x)
        stall = max(2.0, 0.3 * time_limit) if stall is None else stall
        last_gain = time.perf_counter()
        while time.perf_counter() - t0 < time_limit and time.perf_counter() - last_gain < stall:
            act = self.A @ x
            gain = _lift(x, act, b, lo, up, is_int, c, order, self.cptr, self.crow, self.cval, self.obj_row)
            obj = float(c @ x)
            if gain > 1e-9 * (1 + abs(best)) and obj < best - 1e-9 * (1 + abs(best)):
                best = obj
                res.solutions.append((obj, x.copy()))
                last_gain = time.perf_counter()
                if verbose:
                    print(f"    local search: {obj:.6g} (lift, {time.perf_counter() - t0:.2f}s)")
            # breakthrough: kick a few objective-improving variables one step, repair with the cutoff on
            if len(order) == 0:
                break
            y = x.copy()
            kicked = rng.choice(order, size=min(kick, len(order)), replace=False)
            d = np.where(c[kicked] > 0, -1.0, 1.0)
            step = np.where(is_int[kicked], 1.0, np.maximum(1e-3, 0.1 * np.abs(y[kicked]) + 1e-3))
            y[kicked] = np.clip(y[kicked] + d * step, lo[kicked], up[kicked])
            delta = 1.0 if self.int_obj else 1e-4 * max(1.0, abs(best))
            left = time_limit - (time.perf_counter() - t0)
            if left <= 0:
                break
            fr = self.run(time_limit=min(left, max(0.05, 0.1 * time_limit)), x0=y, lower=lo, upper=up,
                          cutoff=best - delta, seed=int(rng.integers(2**31)), improve=False)
            res.steps += fr.steps
            if fr.solutions:
                obj, xs = fr.solutions[0]
                if obj < best - 1e-9 * (1 + abs(best)):
                    best, x = obj, xs.copy()
                    res.solutions.append((obj, x.copy()))
                    last_gain = time.perf_counter()
                    if verbose:
                        print(f"    local search: {obj:.6g} (breakthrough + repair, {time.perf_counter() - t0:.2f}s)")
        res.x_last = x
        res.solutions.sort(key=lambda s: s[0])
        res.seconds = time.perf_counter() - t0
        return res


def feasibility_jump(lp: LP, time_limit: float = 10.0, lower=None, upper=None, seed: int = 0,
                     improve: bool = True, max_solutions: int = 20, chunk: int = 50, sample: int = 64,
                     x0=None, verbose: bool = False) -> FJResult:
    """Feasibility Jump on lp (lp.integer marks integer variables). Returns feasible points found, best first.

    improve: after a solution, keep searching with the objective row c x <= best - delta switched on.
    """
    return FeasibilityJump(lp).run(time_limit, x0=x0, lower=lower, upper=upper, seed=seed, improve=improve,
                                   max_solutions=max_solutions, chunk=chunk, sample=sample, verbose=verbose)
