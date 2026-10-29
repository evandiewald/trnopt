"""Batched feasibility pump: many pumps at once, each one column of a batched PDLP solve.

The feasibility pump (Fischetti, Glover & Lodi 2005; objective variant: Achterberg & Berthold 2007) looks for a
whole-number solution by alternating between two kinds of points:

    x  : satisfies the constraints (an LP solution) but may be fractional
    x~ : round(x) on the integer variables -- whole numbers, but may violate constraints

Each round solves an LP for the feasible point x closest to the current x~, then rounds again, until the rounded
point happens to satisfy every constraint. Closeness is measured on integer variables sitting at a bound:
distance = sum over x~_j = lower_j of (x_j - lower_j) + sum over x~_j = upper_j of (upper_j - x_j), which is a
linear objective with +1 / -1 coefficients and no new variables, so every projection LP has the SAME constraint
matrix as the original problem. A fading multiple of the real objective is added (alpha_t = 0.9^t), so the pump
prefers good solutions early and pure feasibility later.

Why this suits an accelerator: P pumps differ only in their objective vectors, so one batched PDLP solve does a
round for all of them -- the same "many LPs, one matrix" pattern as strong branching. Projection LPs only need
to be good enough to round from, so they run at a loose tolerance and a small iteration budget, warm-started
from the previous round. Pumps start from differently perturbed copies of the LP solution and are perturbed
again when they cycle, so they explore different roundings.

Rounding (rounding="propagate", the default) is Feasibility Pump 2.0's (Fischetti & Salvagnin 2009): integer
variables are fixed a chunk at a time, each clipped into the range bound propagation still allows, propagating
after every chunk (propagate.py). On MIPLIB nu25-pr12 and neos-911970 the plain pump found nothing in 30 rounds
even with exact projections, while propagation rounding found feasible points in 2 and 28 rounds.
rounding="simple" is the original all-at-once rounding.

Integer variables strictly inside their bounds after rounding (general integers) get no distance term: a proper
|x_j - x~_j| term needs extra variables, which would change the matrix. Binary-heavy MIPs are the main target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend import Backend
from .lp import LP


@dataclass
class PumpResult:
    solutions: list = field(default_factory=list)     # (objective, x) feasible points found, best first
    rounds: int = 0
    lp_iterations: int = 0
    seconds: float = 0.0
    repairs: int = 0                                   # exact fix-and-solve LPs tried
    fj_calls: int = 0                                  # local-search runs (repairs and cycle breaks)


def feasibility_pump(lp: LP, x_lp: np.ndarray, backend: Backend | str = "cpu-float32", pumps: int = 16,
                     rounds: int = 40, time_limit: float = 30.0, lp_iters: int = 400, lp_tol: float = 1e-3,
                     repair=None, repairs_per_round: int | None = 4, lower=None, upper=None, seed: int = 0,
                     stop_after: int | None = 1, rounding: str = "propagate", perturb: float = 0.0,
                     lp_solver: str = "pdlp", patience: int | None = 8, fj_repair: float = 0.25,
                     fj_candidates: int = 4, cutoff: bool = True, fj_min: float = 0.02, round_chunks: int = 32,
                     verbose: bool = False) -> PumpResult:
    """Run `pumps` feasibility pumps in parallel from the LP solution x_lp.

    lower, upper: bounds to pump within (default lp's; e.g. a B&B node's bounds).
    repair(xi) -> (status, objective, x) | None: optional exact repair of a candidate whose integer part is
        given (fix the integers, solve the LP in the continuous variables). Needed when the MIP has continuous
        variables; for pure-integer MIPs a rounded point is either feasible or not.
    stop_after: stop once this many distinct feasible solutions are found (None = no such limit).
    patience: once a solution exists, stop after this many rounds without a better one (None = run all rounds).
    fj_repair: local-search repair, as in cuOpt's extended pump. After each rounding, Feasibility Jump (fj.py,
        CPU) starts from the fj_candidates least-violated rounded points, with fj_repair x the round's LP time
        between them; and a pump that cycles is moved to where Feasibility Jump from its point ends, instead of
        random flips. 0 = off (or numba missing).
    cutoff: once a solution exists, keep pumping for better ones: rounded and repaired points must beat it
        (c x <= best - delta), Feasibility Jump searches with that cutoff, and the objective term's weight stops
        fading (it holds at its value from the round the solution appeared, at least 0.1). There is no cutoff
        ROW in the projection LPs: its right-hand side (an objective value) would dominate the norm PDLP's
        relative tolerance is measured against, and an inactive placeholder would be worse.
    """
    from .batch import solve_pdlp_batch
    from .pdlp import PDLPOptions

    t0 = time.perf_counter()
    be = Backend.preset(backend) if isinstance(backend, str) else backend
    rng = np.random.default_rng(seed)
    res = PumpResult()
    ints = np.flatnonzero(lp.integer)
    lo = np.asarray(lp.lower if lower is None else lower, dtype=np.float64)
    up = np.asarray(lp.upper if upper is None else upper, dtype=np.float64)
    has_cont = len(ints) < lp.n
    P = pumps
    eq_rows = np.arange(lp.m) < lp.m_eq
    row_tol = 1e-6 * (1 + np.abs(lp.q))

    # objective term, scaled so it's comparable to the distance term (which is up to #ints)
    cn = np.linalg.norm(lp.c)
    c_scaled = lp.c * (np.sqrt(len(ints)) / cn) if cn > 0 else np.zeros(lp.n)

    prop = None
    if rounding == "propagate":
        from .propagate import Propagator
        prop = Propagator(lp)

    def round_ints(X):
        """Rounded integer parts (P, #ints), and the continuous parts to check them with."""
        if prop is None:
            return np.clip(np.round(X[:, ints]), lo[ints], up[ints]), X
        Xr, _, _, _ = prop.round_batch(X, lo, up, chunks=round_chunks, rng=rng)   # all pumps, one batched propagation
        return Xr[:, ints], Xr

    # start points: the LP solution, then copies whose fractional integer values are randomly jittered
    X = np.repeat(np.clip(x_lp, lo, up)[None, :], P, axis=0)
    if P > 1:
        frac = X[1:, ints] - np.floor(X[1:, ints])
        X[1:, ints] = np.floor(X[1:, ints]) + np.clip(frac + rng.uniform(-0.3, 0.3, frac.shape), 0, 1)
    Y = None
    prev = [None] * P
    seen = [set() for _ in range(P)]
    found = set()
    tried = set()                                              # integer assignments already repaired
    best_obj, since_best = None, 0

    def record(x, obj, how):
        key = tuple(np.round(x[ints]).astype(np.int64))
        if key in found:
            return
        found.add(key)
        res.solutions.append((float(obj), x.copy()))
        if verbose:
            print(f"    pump: feasible solution {obj:.6g} ({how}, round {res.rounds}, {time.perf_counter() - t0:.1f}s)")

    # local search for repairs and cycles (CPU)
    fjs = None
    if fj_repair > 0:
        try:
            from .fj import FeasibilityJump
            fjs = FeasibilityJump(lp)
        except ImportError:
            fjs = None
    alpha_hold = None                                              # objective weight once a solution exists
    gen = (up[ints] - lo[ints]) > 1                                # general integers (not 0/1)
    lp_seconds = 0.5                                               # last round's LP time (sets the FJ budget)

    def cutoff_value():
        if best_obj is None or not cutoff:
            return None
        delta = 1.0 if int_obj else 1e-4 * max(1.0, abs(best_obj))
        return best_obj - delta
    int_obj = bool(np.all(lp.integer[lp.c != 0]) and np.allclose(lp.c, np.round(lp.c)))

    def fj_from(x0, budget, seed_):
        return fjs.run(time_limit=budget, x0=x0, lower=lo, upper=up, cutoff=cutoff_value(), seed=seed_,
                       improve=False)

    t = -1
    while True:
        t += 1
        if t >= rounds or time.perf_counter() - t0 > time_limit:
            break
        res.rounds = t + 1
        R, Xbase = round_ints(X)
        cut = cutoff_value()

        # 1. feasibility of the rounded points (continuous variables taken from the LP point, clipped into the
        #    propagated bounds when rounding="propagate")
        Xc = Xbase.copy()
        Xc[:, ints] = R
        Kx = (lp.K @ Xc.T).T                                   # (P, m)
        viol_rows = np.where(eq_rows, np.abs(Kx - lp.q), np.maximum(lp.q - Kx, 0.0)) - row_tol
        viol = np.maximum(viol_rows, 0.0).sum(axis=1)          # (P,)
        objs = Xc @ lp.c
        better = np.ones(P, dtype=bool) if cut is None else objs <= cut + 1e-9 * (1 + abs(cut))
        for p in np.flatnonzero((viol <= 0) & better):
            record(Xc[p], objs[p], "rounded")
        # 2. local-search repair of the closest rounded points (cuOpt's extended pump: local search after
        #    every rounding), then exact LP repair of the continuous variables
        if fjs is not None:
            cand = [p for p in np.argsort(viol) if viol[p] > 0][:fj_candidates]
            each = min(2.0, max(fj_min, fj_repair * lp_seconds / max(1, len(cand))))
            for p in cand:
                fr = fj_from(Xc[p], each, int(rng.integers(2**31)))
                res.fj_calls += 1
                for o, xs in fr.solutions:
                    record(xs, o, "rounded + local search")
        if repair is not None and has_cont:
            k = P if repairs_per_round is None else repairs_per_round
            for p in np.argsort(viol)[:k]:
                key = R[p].tobytes()
                if viol[p] <= 0 or key in tried:
                    continue
                tried.add(key)
                out = repair(R[p])
                res.repairs += 1
                if out is not None and out[0] == "optimal" and (cut is None or out[1] <= cut + 1e-9 * (1 + abs(cut))):
                    record(out[2], out[1], "repaired")
        if stop_after is not None and len(res.solutions) >= stop_after:
            break
        if res.solutions:
            cur = min(o for o, _ in res.solutions)
            if best_obj is None or cur < best_obj - 1e-9 * (1 + abs(cur)):
                best_obj, since_best = cur, 0
            else:
                since_best += 1
            if patience is not None and since_best >= patience:
                break

        # 3. cycle handling: a pump whose rounding didn't change is moved to where local search from it ends
        #    (cuOpt); without local search, its least-settled integers are flipped
        for p in range(P):
            key = R[p].tobytes()
            if prev[p] is not None and np.array_equal(R[p], prev[p]):
                if fjs is not None:
                    fr = fj_from(Xc[p], min(1.0, max(0.02, fj_repair * lp_seconds / P)), int(rng.integers(2**31)))
                    res.fj_calls += 1
                    for o, xs in fr.solutions:
                        record(xs, o, "cycle + local search")
                    R[p] = np.round(fr.x_last[ints])
                else:
                    dist = np.abs(X[p, ints] - R[p])
                    k = int(rng.integers(10, 31))
                    flip = np.argsort(-dist)[:k]
                    for i in flip:
                        j = ints[i]
                        R[p, i] = (up[j] if R[p, i] == lo[j] else lo[j]) if up[j] - lo[j] == 1 else \
                            np.clip(R[p, i] + (1 if X[p, j] > R[p, i] else -1), lo[j], up[j])
            elif key in seen[p]:                               # a longer cycle: shake harder
                shake = rng.random(len(ints)) < 0.1
                width = up[ints] - lo[ints]
                R[p, shake] = np.clip(np.where(width[shake] == 1, lo[ints][shake] + up[ints][shake] - R[p, shake],
                                               R[p, shake] + rng.choice([-1, 1], shake.sum())), lo[ints][shake], up[ints][shake])
            seen[p].add(key)
            prev[p] = R[p].copy()

        # 4. projection LPs: distance to each pump's rounded point (+ fading objective), all in one batch.
        #    Variables at a bound: +1 / -1 toward it. General integers strictly inside their range: a push from
        #    the side the LP point is on (sign(x - x~)), which keeps the matrix fixed (no |x - x~| variables).
        C = np.zeros((P, lp.n))
        at_lo = R == lo[ints][None, :]
        at_up = R == up[ints][None, :]
        inner = gen[None, :] & ~at_lo & ~at_up
        push = np.sign(X[:, ints] - R) * inner
        C[:, ints] = at_lo.astype(np.float64) - at_up.astype(np.float64) + push
        alpha = 0.9 ** t
        if cutoff and best_obj is not None:
            alpha_hold = max(0.1, alpha) if alpha_hold is None else alpha_hold
            alpha = alpha_hold
        C += alpha * c_scaled[None, :]
        if perturb > 0:
            # tiny random tilt per pump: makes each projection LP's optimum (nearly) unique, so a first-order
            # method converges to a corner-like point instead of the middle of a flat optimal face
            C += perturb * rng.uniform(-1, 1, C.shape)
        if lp_solver == "highs":                                   # exact vertex solutions (reference / diagnosis)
            import dataclasses
            from .reference import solve_highs
            base = dataclasses.replace(lp, lower=lo, upper=up, integer=None)
            X = np.stack([solve_highs(dataclasses.replace(base, c=C[p])).x for p in range(P)])
            continue
        tl = time.perf_counter()
        r = solve_pdlp_batch(lp, P, lower=lo, upper=up, c=C, x0=X, y0=Y,
                             opts=PDLPOptions(tol=lp_tol, backend=be, max_iters=lp_iters,
                                              time_limit=max(1.0, time_limit - (time.perf_counter() - t0))))
        lp_seconds = time.perf_counter() - tl
        X = np.clip(np.asarray(r.x), lo, up)
        Y = np.asarray(r.y)
        res.lp_iterations += int(r.total_iterations)

    res.solutions.sort(key=lambda s: s[0])
    res.seconds = time.perf_counter() - t0
    return res
