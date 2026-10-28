"""Bound propagation and propagation rounding (the rounding step of Feasibility Pump 2.0).

Bound propagation, in one sentence: for a constraint  sum_j a_j x_j >= q,  if every OTHER variable is at the
bound that helps the row most, variable j must still make up the rest -- which may raise its lower bound (a_j > 0)
or lower its upper bound (a_j < 0). Equality rows act in both directions. Integer bounds round inward. Repeat
until nothing changes: fixing one variable can ripple through the whole model.

Vectorised over the matrix's nonzeros: per-row sums of best/worst contributions (two sparse products), then per
nonzero the implied bound, then a per-column max (new lower bounds) / min (new upper bounds). That last step is
the "reduce over each column" operation a Trainium kernel would do with min/max instead of add.

Infinite bounds: a row's activity bound is infinite if any variable contributes an infinite term; it still
implies a finite bound for a variable when that variable is the ONLY infinite contributor.

Propagation rounding (Fischetti & Salvagnin 2009): instead of rounding all integer variables at once, fix them a
chunk at a time -- each clipped into its currently propagated range -- and propagate after each chunk, so later
roundings respect what earlier ones forced. If propagation proves a contradiction, the rest are rounded plainly.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .lp import LP

FEAS_TOL = 1e-6
INT_TOL = 1e-6


class Propagator:
    """Bound propagation for one LP (its matrix structure is preprocessed once; bounds change per call)."""

    def __init__(self, lp: LP):
        self.lp = lp
        K = sp.coo_matrix(lp.K)
        nz = K.data != 0                                              # explicit zeros would give 0 * inf = nan
        K = sp.coo_matrix((K.data[nz], (K.row[nz], K.col[nz])), shape=K.shape)
        # every row as "a x >= rhs": ">=" rows as they are, equality rows also negated (the "<=" side)
        eq = K.row < lp.m_eq
        self.row = np.concatenate([K.row, K.row[eq] + lp.m])        # equality rows' "<=" copies get ids m..
        self.col = np.concatenate([K.col, K.col[eq]])
        self.val = np.concatenate([K.data, -K.data[eq]])
        self.rhs = np.concatenate([lp.q, -lp.q])                    # rows m.. only used for equality copies
        self.nrows = 2 * lp.m
        self.integer = np.zeros(lp.n, dtype=bool) if lp.integer is None else np.asarray(lp.integer, bool)
        self.pos = self.val > 0

    def propagate(self, lower, upper, max_passes: int = 20, tol: float = 1e-9):
        """Tighten (lower, upper) as far as the rows allow. Returns (lower, upper, infeasible)."""
        lo, up, bad = self.propagate_batch(np.asarray(lower, dtype=np.float64)[None, :],
                                           np.asarray(upper, dtype=np.float64)[None, :], max_passes, tol)
        return lo[0], up[0], bool(bad[0])

    def _prep_batch(self):
        """Index structures for the batched passes (built on first use)."""
        if hasattr(self, "_S"):
            return
        nnz = len(self.val)
        # row sums as one sparse product: S (nrows x nnz) has a 1 for each nonzero in its row
        self._S = sp.csr_matrix((np.ones(nnz), (self.row, np.arange(nnz))), shape=(self.nrows, nnz))
        self._used = np.zeros(self.nrows, dtype=bool)
        self._used[self.row] = True
        # per-column max / min as a segmented reduction over nonzeros sorted by column
        for name, mask in (("lo", self.pos), ("up", ~self.pos)):
            idx = np.flatnonzero(mask)
            idx = idx[np.argsort(self.col[idx], kind="stable")]
            cols = self.col[idx]
            starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]]) if len(cols) else np.zeros(0, int)
            setattr(self, f"_{name}_idx", idx)
            setattr(self, f"_{name}_starts", starts)
            setattr(self, f"_{name}_cols", cols[starts] if len(cols) else np.zeros(0, int))

    def propagate_batch(self, lower, upper, max_passes: int = 20, tol: float = 1e-9):
        """propagate() for P bound vectors at once: lower, upper (P, n). Returns (lo, up, infeasible (P,)).

        A vector whose bounds become contradictory is frozen at that point (its bounds are returned as they were
        when the contradiction showed up), the others keep propagating.
        """
        self._prep_batch()
        lo, up = np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)
        P = lo.shape[0]
        bad = np.zeros(P, dtype=bool)
        r, c, a = self.row, self.col, self.val
        n = self.lp.n
        ints = self.integer
        scale = lambda b: tol * (1 + np.abs(np.where(np.isfinite(b), b, 0.0)))      # no inf - inf
        act = np.arange(P)                                            # vectors still changing
        for _ in range(max_passes):
            L, U = lo[act], up[act]
            # contribution of each nonzero to its row's MAXIMUM activity: a*up if a>0, a*lo if a<0
            best = np.where(self.pos, a * U[:, c], a * L[:, c])       # (p, nnz)
            inf_term = ~np.isfinite(best)
            fin = np.where(inf_term, 0.0, best)
            row_sum = np.asarray((self._S @ fin.T).T)                 # (p, nrows)
            row_inf = np.asarray((self._S @ inf_term.T.astype(np.float64)).T)
            infeas = np.any(self._used & (row_inf == 0) & (row_sum < self.rhs - FEAS_TOL * (1 + np.abs(self.rhs))),
                            axis=1)
            others_inf = row_inf[:, r] - inf_term
            resid = row_sum[:, r] - fin
            ok = others_inf == 0
            with np.errstate(invalid="ignore"):
                bound = np.where(ok, (self.rhs[r] - resid) / a, np.nan)
            new_lo = np.full(L.shape, -np.inf)
            new_up = np.full(U.shape, np.inf)
            if len(self._lo_idx):
                v = np.where(ok[:, self._lo_idx], bound[:, self._lo_idx], -np.inf)
                new_lo[:, self._lo_cols] = np.maximum.reduceat(v, self._lo_starts, axis=1)
            if len(self._up_idx):
                v = np.where(ok[:, self._up_idx], bound[:, self._up_idx], np.inf)
                new_up[:, self._up_cols] = np.minimum.reduceat(v, self._up_starts, axis=1)
            new_lo[:, ints] = np.ceil(new_lo[:, ints] - INT_TOL)
            new_up[:, ints] = np.floor(new_up[:, ints] + INT_TOL)
            lo2, up2 = np.maximum(L, new_lo), np.minimum(U, new_up)
            crossed = np.any(lo2 > up2 + FEAS_TOL * (1 + np.abs(lo2)), axis=1)
            up2 = np.where(crossed[:, None], up2, np.maximum(up2, lo2))  # tiny crossings from rounding noise
            changed = np.any(lo2 > L + scale(L), axis=1) | np.any(up2 < U - scale(U), axis=1)
            # rows proven infeasible keep their bounds from before this pass; crossed ones the crossed bounds
            keep_old = infeas
            lo[act] = np.where(keep_old[:, None], L, lo2)
            up[act] = np.where(keep_old[:, None], U, up2)
            stop = infeas | crossed
            bad[act[stop]] = True
            act = act[~stop & changed]
            if len(act) == 0:
                break
        return lo, up, bad

    def round(self, x, lower=None, upper=None, chunks: int = 32, rng=None):
        """Propagation rounding of the integer variables of x. Returns (x_rounded, lower, upper, contradiction).

        Variables are fixed most-integral first, in `chunks` groups, each value clipped into the current
        propagated range, propagating after each group. Continuous variables are returned clipped into the
        final propagated bounds (a caller with continuous variables should still repair them exactly).
        """
        xr, lo, up, bad = self.round_batch(np.asarray(x, dtype=np.float64)[None, :], lower, upper, chunks, rng)
        return xr[0], lo[0], up[0], bool(bad[0])

    def round_batch(self, X, lower=None, upper=None, chunks: int = 32, rng=None):
        """round() for P points at once (X: (P, n)), sharing one batched propagation per chunk."""
        lp = self.lp
        lo0 = np.array(lp.lower if lower is None else lower, dtype=np.float64)
        up0 = np.array(lp.upper if upper is None else upper, dtype=np.float64)
        X = np.array(X, dtype=np.float64)
        P = X.shape[0]
        ints = np.flatnonzero(self.integer)
        l1, u1, b1 = self.propagate(lo0, up0)                         # same start bounds for every point
        lo, up = np.repeat(l1[None], P, 0), np.repeat(u1[None], P, 0)
        bad = np.full(P, b1)
        frac = np.abs(X[:, ints] - np.round(X[:, ints]))
        if rng is not None:
            frac = frac + rng.uniform(0, 1e-3, frac.shape)            # break ties differently per point
        order = ints[np.argsort(frac, axis=1, kind="stable")]        # (P, #ints)
        for block in np.array_split(np.arange(len(ints)), max(1, min(chunks, len(ints)))):
            act = np.flatnonzero(~bad)
            if len(act) == 0:
                break
            g = order[act][:, block]                                  # (a, chunk) variable ids per point
            ra = act[:, None]
            v = np.clip(np.round(X[ra, g]), lo[ra, g], up[ra, g])
            v = np.where(np.isfinite(v), v, np.round(X[ra, g]))
            lo[ra, g] = v
            up[ra, g] = v
            l2, u2, b2 = self.propagate_batch(lo[act], up[act])
            lo[act], up[act] = l2, u2
            bad[act] = b2
        # whatever wasn't fixed by propagation: plain rounding into the (possibly crossed) range
        Xi = X[:, ints]
        X[:, ints] = np.round(np.clip(Xi, np.minimum(lo[:, ints], up[:, ints]), np.maximum(lo[:, ints], up[:, ints])))
        cont = ~self.integer
        X[:, cont] = np.clip(X[:, cont], lo[:, cont], up[:, cont])
        return X, lo, up, bad
