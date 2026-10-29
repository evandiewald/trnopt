"""Batched flip search: improve a solution by flipping integer variables, with the candidates screened as one
batched LP solve on the accelerator and only the most promising checked exactly (HiGHS, warm-started).

Why: on some models the LP relaxation says nothing about the integer variables (bts4-cta: a disclosure-control
model where each binary picks the direction of a table-cell adjustment; at the root every binary is exactly 0.5),
so LP-guided heuristics (rounding, the pump, diving, RINS) have nothing to work with, and the incumbent comes
from a trivial all-0 / all-1 assignment. The way forward is a neighbourhood search on the integers: change a few,
re-solve the continuous variables, keep it if it's better. Each candidate costs an LP -- ~1 s with HiGHS warm-
started from the incumbent's basis on bts4-cta -- and there are thousands of single flips.

All candidate LPs share one constraint matrix and differ only in bounds (the flipped integers' fixed values), the
same shape as strong branching. So a batch of them is solved approximately with PDLP (warm-started from the
incumbent), and the batch is used to SCREEN: drop candidates whose LP stays clearly infeasible (large primal
residual), rank the rest by estimated objective, verify the best few exactly. On bts4-cta, 32 single flips: HiGHS
warm 35 s; PDLP 1000 iterations on the CPU 27 s, picking the same best candidate (rank correlation 0.89) and
separating feasible from infeasible flips by ~4x in primal residual.

Moves: single flips of random integer variables (binaries flip 0 <-> 1, general integers move +-1). When several
verified flips improve, their combination is tried too (one more exact check). Tabu: a variable just flipped isn't
flipped back for a while.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend import Backend
from .lp import LP


@dataclass
class FlipResult:
    solutions: list = field(default_factory=list)   # (objective, x), best first
    rounds: int = 0
    screened: int = 0                                 # candidate LPs solved approximately (batched)
    verified: int = 0                                 # exact LP checks
    accepted: int = 0
    seconds: float = 0.0
    screen_seconds: float = 0.0                       # in the batched solves (the accelerator part)


def flip_search(lp: LP, x_inc: np.ndarray, exact, backend: Backend | str = "cpu-float32", time_limit: float = 60.0,
                batch: int = 32, pdlp_iters: int = 1000, verify: int = 4, resid_tol: float = 1.2e-3,
                lower=None, upper=None, seed: int = 0, tabu: int = 50, verbose: bool = False) -> FlipResult:
    """Improve the feasible point x_inc by integer flips.

    exact(int_values) -> (status, objective, x) : exact LP in the continuous variables with the integers fixed
        (the caller warm-starts it, e.g. from the incumbent's basis).
    batch: candidates per screening solve; pdlp_iters: PDHG iterations for it; verify: exact checks per round;
    resid_tol: relative primal residual above which a screened candidate counts as infeasible.
    """
    from .batch import solve_pdlp_batch
    from .pdlp import PDLPOptions

    t0 = time.perf_counter()
    be = Backend.preset(backend) if isinstance(backend, str) else backend
    rng = np.random.default_rng(seed)
    res = FlipResult()
    ints = np.flatnonzero(lp.integer)
    lo = np.asarray(lp.lower if lower is None else lower, dtype=np.float64)
    up = np.asarray(lp.upper if upper is None else upper, dtype=np.float64)
    x = np.array(x_inc, dtype=np.float64)
    best = float(lp.c @ x)
    recent: dict = {}                                     # variable -> round it was last flipped
    free = ints[up[ints] > lo[ints]]
    if len(free) == 0:
        res.seconds = time.perf_counter() - t0
        return res
    Y = None

    def flipped_value(j, cur):
        if up[j] - lo[j] == 1:
            return lo[j] + up[j] - cur
        step = rng.choice([-1.0, 1.0])
        return float(np.clip(cur + step, lo[j], up[j])) if lo[j] < cur + step < up[j] or cur + step in (lo[j], up[j]) \
            else float(np.clip(cur - step, lo[j], up[j]))

    while time.perf_counter() - t0 < time_limit:
        res.rounds += 1
        pool = np.array([j for j in free if res.rounds - recent.get(int(j), -10**9) > tabu])
        if len(pool) == 0:
            recent.clear()
            pool = free
        cand = rng.choice(pool, size=min(batch, len(pool)), replace=False)
        newv = np.array([flipped_value(int(j), x[j]) for j in cand])
        B = len(cand)
        L = np.repeat(lo[None], B, 0)
        U = np.repeat(up[None], B, 0)
        L[:, ints] = U[:, ints] = x[ints][None, :]           # integers fixed at the incumbent...
        L[np.arange(B), cand] = U[np.arange(B), cand] = newv  # ...except the one flipped
        X0 = np.repeat(x[None], B, 0)
        X0[np.arange(B), cand] = newv
        # 1. screen: all candidates' LPs in one batched solve
        ts = time.perf_counter()
        r = solve_pdlp_batch(lp, B, lower=L, upper=U, x0=X0, y0=Y,
                             opts=PDLPOptions(tol=1e-4, backend=be, max_iters=pdlp_iters,
                                              time_limit=max(1.0, time_limit - (time.perf_counter() - t0))))
        res.screen_seconds += time.perf_counter() - ts
        res.screened += B
        est = np.asarray(r.objective, dtype=np.float64)
        pres = np.asarray(r.primal_residual, dtype=np.float64)
        ok = np.array([s != "infeasible" for s in r.status]) & (pres <= resid_tol) & (est < best)
        order = np.flatnonzero(ok)[np.argsort(est[ok])][:verify]
        if verbose:
            print(f"    flip round {res.rounds}: {ok.sum()}/{B} pass the screen, verifying {len(order)}")
        # 2. verify the most promising exactly
        good = []
        for k in order:
            if time.perf_counter() - t0 > time_limit:
                break
            vals = x[ints].copy()
            vals[np.searchsorted(ints, cand[k])] = newv[k]
            st, obj, xs = exact(vals)
            res.verified += 1
            if st == "optimal" and obj < best - 1e-9 * (1 + abs(best)):
                good.append((obj, k, xs))
        for k in range(B):
            recent[int(cand[k])] = res.rounds
        if not good:
            continue
        good.sort(key=lambda g: g[0])
        obj, k, xs = good[0]
        # 3. several improving flips: try them all at once (one more exact check)
        if len(good) > 1 and time.perf_counter() - t0 < time_limit:
            vals = x[ints].copy()
            for _, kk, _ in good:
                vals[np.searchsorted(ints, cand[kk])] = newv[kk]
            st, o2, x2 = exact(vals)
            res.verified += 1
            if st == "optimal" and o2 < obj:
                obj, xs = o2, x2
        x, best = np.array(xs, dtype=np.float64), float(obj)
        res.accepted += 1
        res.solutions.append((best, x.copy()))
        Y = np.repeat(np.asarray(r.y)[k][None], B, 0) if r.y is not None else None   # warm duals for the next round
        if verbose:
            print(f"    flip search: {best:.6g} ({time.perf_counter() - t0:.1f}s)")
    res.solutions.sort(key=lambda s: s[0])
    res.seconds = time.perf_counter() - t0
    return res
