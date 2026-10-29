"""Batched LP-guided diving: many dives at once, each dive's LP one column of a batched PDLP solve.

A dive is a quick, greedy path from the LP relaxation to a whole-number solution:

    repeat: fix some integer variables (the ones the LP already has whole, plus the least fractional few,
            rounded), propagate the bounds that implies, and re-solve the LP with those variables fixed --
    until every integer variable is fixed. If the last LP is feasible, that's a solution.

One dive is cheap but usually fails or lands on a poor solution. Many dives that fix variables in different
orders and directions explore many different paths, which is the point of running them together: D dives
differ only in their variable bounds, so every round's D LPs share one constraint matrix -- the same "many LPs,
one matrix" shape as strong branching and the feasibility pump, and the shape the accelerator is good at.
The LPs only need to be good enough to round from, so they run at a loose tolerance and a small iteration
budget, warm-started from the dive's previous LP.

Dives differ by rule:
  - "fractional": round the least fractional variables to the nearest integer (the classic fractional dive);
  - "random": round each chosen variable up with probability equal to its fractional part;
  - "guided": round toward the incumbent's value where one exists (a cheap, many-path version of guided dives);
plus a per-dive random tie-break, so dives with the same rule still take different paths. Each round's fixings are applied
a chunk at a time, each value clipped into the range bound propagation still allows, propagating after every
chunk (FP 2.0's propagation rounding, per dive step). A dive whose round still ends in a contradiction backs off:
the round is undone and the dive fixes half as many variables next time; only when it is down to one variable
does it restart from the start bounds with a new seed. A dive whose LP comes back infeasible restarts too, and a
finished dive restarts, so slots stay busy until time runs out.

Propagation (propagate.py) runs on the CPU between LP rounds, batched across dives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend import Backend
from .lp import LP

RULES = ("fractional", "random", "guided", "fractional")


@dataclass
class DiveResult:
    solutions: list = field(default_factory=list)     # (objective, x), best first
    rounds: int = 0                                     # batched LP rounds
    dives_finished: int = 0                             # dives that fixed every integer variable
    dives_failed: int = 0                               # dives that hit a contradiction
    lp_iterations: int = 0
    seconds: float = 0.0
    lp_seconds: float = 0.0                             # time in the batched LP solves (the accelerator part)


def batched_dives(lp: LP, x_lp: np.ndarray, backend: Backend | str = "cpu-float32", dives: int = 16,
                  time_limit: float = 30.0, lower=None, upper=None, incumbent=None, repair=None,
                  fix_frac: float = 0.1, chunks: int = 4, prop_passes: int = 5, lp_iters: int = 400, lp_tol: float = 1e-3, seed: int = 0,
                  verbose: bool = False) -> DiveResult:
    """Run `dives` LP-guided dives in parallel from the LP solution x_lp, for up to time_limit seconds.

    lower, upper: start bounds (default lp's; e.g. a B&B node's).
    incumbent: optional best known solution, used by the "guided" dives.
    repair(int_values) -> (status, objective, x) | None: exact solve of the continuous variables with the
        integers fixed (needed when the MIP has continuous variables). Without it, a dive's last LP point,
        with its integers rounded, must itself be feasible.
    """
    from .batch import solve_pdlp_batch
    from .pdlp import PDLPOptions
    from .propagate import Propagator

    t0 = time.perf_counter()
    be = Backend.preset(backend) if isinstance(backend, str) else backend
    rng = np.random.default_rng(seed)
    res = DiveResult()
    D = dives
    ints = np.flatnonzero(lp.integer)
    has_cont = len(ints) < lp.n
    lo0 = np.asarray(lp.lower if lower is None else lower, dtype=np.float64)
    up0 = np.asarray(lp.upper if upper is None else upper, dtype=np.float64)
    prop = Propagator(lp)
    eq_rows = np.arange(lp.m) < lp.m_eq
    row_tol = 1e-6 * (1 + np.abs(lp.q))
    found = set()

    def record(x, how):
        key = tuple(np.round(x[ints]).astype(np.int64))
        if key in found:
            return
        found.add(key)
        obj = float(lp.c @ x)
        res.solutions.append((obj, x.copy()))
        if verbose:
            print(f"    dives: solution {obj:.6g} ({how}, round {res.rounds}, {time.perf_counter() - t0:.1f}s)")

    def rows_ok(x):
        Kx = lp.K @ x
        v = np.where(eq_rows, np.abs(Kx - lp.q), np.maximum(lp.q - Kx, 0.0)) - row_tol
        return bool(np.all(v <= 0))

    rule = np.array([RULES[d % len(RULES)] for d in range(D)])
    L = np.repeat(lo0[None], D, 0)
    U = np.repeat(up0[None], D, 0)
    X = np.repeat(np.clip(x_lp, lo0, up0)[None], D, 0)
    Y = None
    noise = rng.uniform(0, 0.02, (D, len(ints)))            # per-dive tie-break
    share = np.full(D, fix_frac)                              # fraction of free integers each dive fixes per round

    def restart(d):
        L[d], U[d] = lo0, up0
        X[d] = np.clip(x_lp, lo0, up0)
        noise[d] = rng.uniform(0, 0.02, len(ints))
        share[d] = fix_frac

    while time.perf_counter() - t0 < time_limit:
        res.rounds += 1
        # 1. fixings: choose this round's variables and values per dive...
        xi = X[:, ints]
        free = L[:, ints] < U[:, ints]
        frac = np.abs(xi - np.round(xi))
        L_prev, U_prev = L.copy(), U.copy()
        chosen = []                                           # per dive: (variable ids in fixing order, values)
        for d in range(D):
            f = np.flatnonzero(free[d])
            if len(f) == 0:
                chosen.append((np.zeros(0, dtype=np.int64), np.zeros(0)))
                continue
            v = np.round(xi[d, f])
            if rule[d] == "random":
                v = np.floor(xi[d, f]) + (rng.random(len(f)) < (xi[d, f] - np.floor(xi[d, f])))
            elif rule[d] == "guided" and incumbent is not None:
                v = np.where(frac[d, f] < 0.5, np.round(incumbent[ints][f]), v)
            whole = frac[d, f] < 1e-6                             # already integral in the LP: fix as is
            k = min(len(f), max(1, int(np.ceil(share[d] * len(ints)))))   # a share of ALL integers: ~1/share rounds
            order = np.argsort(frac[d, f] + noise[d, f], kind="stable")
            pick = np.zeros(len(f), dtype=bool)
            pick[order[:k]] = True
            # whole-valued ones too, but only while the dive isn't backing off
            if share[d] >= fix_frac:
                pick |= whole
            sel = order[pick[order]]                              # most integral first
            chosen.append((ints[f[sel]], v[sel]))
        # ...then fix them a chunk at a time, every value clipped into the range propagation still allows, and
        # propagate (all dives at once) after each chunk -- FP 2.0's propagation rounding, applied to a dive step
        bad = np.zeros(D, dtype=bool)
        L, U, bad0 = prop.propagate_batch(L, U, max_passes=prop_passes)
        bad |= bad0
        for c in range(chunks):
            for d in np.flatnonzero(~bad):
                j, v = chosen[d]
                if len(j) == 0:
                    continue
                part = np.array_split(np.arange(len(j)), chunks)[c]
                if len(part) == 0:
                    continue
                jj = j[part]
                lo_, up_ = np.minimum(L[d, jj], U[d, jj]), np.maximum(L[d, jj], U[d, jj])
                val = np.clip(v[part], lo_, up_)
                L[d, jj] = U[d, jj] = val
            act = np.flatnonzero(~bad)
            if len(act) == 0:
                break
            l2, u2, b2 = prop.propagate_batch(L[act], U[act], max_passes=prop_passes)
            L[act], U[act] = l2, u2
            bad[act] |= b2
        # 2. a contradiction: back off (undo the round, fix half as many next time); restart only at one variable
        for d in np.flatnonzero(bad):
            L[d], U[d] = L_prev[d], U_prev[d]
            if share[d] * len(ints) <= 1:
                res.dives_failed += 1
                restart(d)
            else:
                share[d] = share[d] / 2
        for d in np.flatnonzero(~bad):
            share[d] = min(fix_frac, share[d] * 1.5)              # recover slowly after a success
        bad[:] = False
        # 3. dives with every integer fixed: candidate solutions
        done = np.all(L[:, ints] == U[:, ints], axis=1) & ~bad
        for d in np.flatnonzero(done):
            r_ints = L[d, ints]
            if repair is not None and has_cont:
                out = repair(r_ints)
                if out is not None and out[0] == "optimal":
                    record(out[2], "dive + repair")
            else:
                x = np.clip(X[d], L[d], U[d])
                x[ints] = r_ints
                if rows_ok(x):
                    record(x, "dive")
            res.dives_finished += 1
            restart(d)
        if time.perf_counter() - t0 >= time_limit:
            break
        # 4. the next LP of every dive, in one batched solve (the accelerator part)
        tl = time.perf_counter()
        r = solve_pdlp_batch(lp, D, lower=L, upper=U, x0=X, y0=Y,
                             opts=PDLPOptions(tol=lp_tol, backend=be, max_iters=lp_iters,
                                              time_limit=max(1.0, time_limit - (time.perf_counter() - t0))))
        res.lp_seconds += time.perf_counter() - tl
        res.lp_iterations += int(r.total_iterations)
        X = np.clip(np.asarray(r.x), L, U)
        Y = np.asarray(r.y)
        for d in range(D):
            if r.status[d] == "infeasible":
                res.dives_failed += 1
                restart(d)
    res.solutions.sort(key=lambda s: s[0])
    res.seconds = time.perf_counter() - t0
    return res
