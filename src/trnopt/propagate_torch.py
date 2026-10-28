"""Bound propagation for many bound vectors at once, as dense tensor ops (for an accelerator).

propagate.py works per nonzero (gathers and segmented reductions), which suits a CPU. Here the same pass is
rewritten as dense products, the operations accelerators are built for:

    row max-activity    rowsum = U+ @ K+^T + L- @ K-^T          two matmuls, (P, n) x (n, m)
    row slack           d      = rhs - rowsum                   (<= 0 when the row can still be satisfied)
    implied bounds      for a_ij > 0:  x_j >= U_j + d_i / a_ij  (every other variable at its best value)
                        for a_ij < 0:  x_j <= L_j + d_i / a_ij
    per column          new_lo_j = max_i (...),  new_up_j = min_i (...)

The last step is a "max-times" product of the slacks (P, m) with the reciprocal matrix 1/K (m, n): a matrix
product with (max, x) in place of (+, x). The tensor engine only does (+, x), so here it is materialised in row
blocks of (P, rows, n) and reduced on the vector engine; a custom kernel would keep the blocks on-chip.

Infinite bounds: a row whose max activity is infinite only implies a bound for the variable that is its single
infinite contributor (same rule as propagate.py). Results match propagate.py up to float32 rounding.
Only for matrices that fit densely (P * m * n within `max_block` elements per row block).
"""

from __future__ import annotations

import numpy as np
import torch

from .backend import Backend
from .lp import LP

FEAS_TOL = 1e-6
INT_TOL = 1e-6


def _pass(L, U, bad, Kp, Kn, nzp, nzn, invKp, invKn, rhs, used, integer, rows_per_block: int, tol: float):
    """One propagation pass over every row. L, U: (P, n); bad: (P,) bool. Returns (L, U, bad, changed)."""
    infU = torch.isinf(U)
    infL = torch.isinf(L)
    Uf = torch.where(infU, torch.zeros_like(U), U)
    Lf = torch.where(infL, torch.zeros_like(L), L)
    rowsum = Uf @ Kp.T + Lf @ Kn.T                                   # (P, m)
    rowinf = infU.to(U.dtype) @ nzp.T + infL.to(U.dtype) @ nzn.T     # number of infinite contributors
    infeas = (used & (rowinf == 0) & (rowsum < rhs - FEAS_TOL * (1 + rhs.abs()))).any(dim=1)
    d = rhs - rowsum
    neg_inf = torch.tensor(-float("inf"), dtype=U.dtype, device=U.device)
    pos_inf = torch.tensor(float("inf"), dtype=U.dtype, device=U.device)
    new_lo = torch.full_like(L, -float("inf"))
    new_up = torch.full_like(U, float("inf"))
    m = Kp.shape[0]
    for r0 in range(0, m, rows_per_block):
        r1 = min(m, r0 + rows_per_block)
        dB = d[:, r0:r1, None]                                       # (P, b, 1)
        ri = rowinf[:, r0:r1, None]
        # a_ij > 0: x_j >= U_j + d_i / a_ij, valid if no other infinite contributor in row i
        okp = (nzp[None, r0:r1] > 0) & ((ri == 0) | ((ri == 1) & infU[:, None, :]))
        bl = torch.where(okp, Uf[:, None, :] + dB * invKp[None, r0:r1], neg_inf)
        new_lo = torch.maximum(new_lo, bl.amax(dim=1))
        okn = (nzn[None, r0:r1] > 0) & ((ri == 0) | ((ri == 1) & infL[:, None, :]))
        bu = torch.where(okn, Lf[:, None, :] + dB * invKn[None, r0:r1], pos_inf)
        new_up = torch.minimum(new_up, bu.amin(dim=1))
    new_lo = torch.where(integer, torch.ceil(new_lo - INT_TOL * (1 + new_lo.abs())), new_lo)
    new_up = torch.where(integer, torch.floor(new_up + INT_TOL * (1 + new_up.abs())), new_up)
    L2 = torch.maximum(L, new_lo)
    U2 = torch.minimum(U, new_up)
    crossed = (L2 > U2 + FEAS_TOL * (1 + L2.abs())).any(dim=1)
    U2 = torch.maximum(U2, L2)
    scaleL = tol * (1 + torch.where(torch.isinf(L), torch.zeros_like(L), L).abs())
    scaleU = tol * (1 + Uf.abs())
    changed = ((L2 > L + scaleL).any(dim=1) | (U2 < U - scaleU).any(dim=1)) & ~bad
    nb = bad | infeas | crossed
    keep = bad[:, None] | infeas[:, None]                            # infeasible: keep the bounds before this pass
    L = torch.where(keep, L, L2)
    U = torch.where(keep, U, U2)
    return L, U, nb, changed & ~nb


class TorchPropagator:
    """Batched propagation (and propagation rounding) on a torch backend; same interface as Propagator's batch API."""

    def __init__(self, lp: LP, backend: Backend | str = "cpu-float32", rows_per_block: int | None = None,
                 max_block: int = 16_000_000, P_hint: int = 16, passes_per_check: int = 4):
        self.lp = lp
        self.be = Backend.preset(backend) if isinstance(backend, str) else backend
        dev, dt = self.be.device, self.be.dtype
        K = lp.K.toarray()
        K2 = np.vstack([K, -K[: lp.m_eq]])                           # every row as ">=" (equalities both ways)
        rhs = np.concatenate([lp.q, -lp.q[: lp.m_eq]])
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dt).to(dev)
        self.Kp, self.Kn = t(np.maximum(K2, 0)), t(np.minimum(K2, 0))
        self.nzp, self.nzn = t(K2 > 0), t(K2 < 0)
        with np.errstate(divide="ignore"):
            self.invKp = t(np.where(K2 > 0, 1.0 / np.where(K2 > 0, K2, 1), 0.0))
            self.invKn = t(np.where(K2 < 0, 1.0 / np.where(K2 < 0, K2, 1), 0.0))
        self.rhs = t(rhs)
        self.used = torch.as_tensor((K2 != 0).any(axis=1)).to(dev)
        self.integer = torch.as_tensor(np.zeros(lp.n, bool) if lp.integer is None else np.asarray(lp.integer, bool)).to(dev)
        self.integer_np = np.asarray(self.integer.cpu())
        m2, n = K2.shape
        self.rows_per_block = rows_per_block or max(1, min(m2, max_block // max(1, P_hint * n)))
        self.passes_per_check = passes_per_check
        self.tol = 1e-6 if dt == torch.float32 else 1e-9
        self._pass = self.be.compile_fn(_pass)

    @staticmethod
    def fits(lp: LP, P: int = 16, max_dense: int = 20_000_000) -> bool:
        return (lp.m + lp.m_eq) * lp.n <= max_dense

    def _t(self, a):
        return torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=self.be.dtype).to(self.be.device)

    def propagate_batch_t(self, L, U, bad, max_passes: int = 20):
        """On-device propagation. L, U: (P, n) tensors; bad: (P,) bool tensor."""
        for k in range(max_passes):
            L, U, bad, changed = self._pass(L, U, bad, self.Kp, self.Kn, self.nzp, self.nzn, self.invKp,
                                            self.invKn, self.rhs, self.used, self.integer, self.rows_per_block,
                                            self.tol)
            if (k + 1) % self.passes_per_check == 0 and not bool(changed.any()):
                break
        return L, U, bad

    def propagate_batch(self, lower, upper, max_passes: int = 20):
        lower, upper = np.atleast_2d(lower), np.atleast_2d(upper)
        bad = torch.zeros(lower.shape[0], dtype=torch.bool, device=self.be.device)
        L, U, bad = self.propagate_batch_t(self._t(lower), self._t(upper), bad, max_passes)
        return (np.asarray(L.cpu(), dtype=np.float64), np.asarray(U.cpu(), dtype=np.float64),
                np.asarray(bad.cpu()))

    def round_batch(self, X, lower=None, upper=None, chunks: int = 32, rng=None):
        """Propagation rounding of P points (same algorithm as Propagator.round_batch), bounds kept on device."""
        lp = self.lp
        X = np.array(X, dtype=np.float64)
        P = X.shape[0]
        lo0 = np.asarray(lp.lower if lower is None else lower, dtype=np.float64)
        up0 = np.asarray(lp.upper if upper is None else upper, dtype=np.float64)
        ints = np.flatnonzero(self.integer_np)
        L = self._t(np.repeat(lo0[None], P, 0))
        U = self._t(np.repeat(up0[None], P, 0))
        bad = torch.zeros(P, dtype=torch.bool, device=self.be.device)
        L, U, bad = self.propagate_batch_t(L, U, bad)
        frac = np.abs(X[:, ints] - np.round(X[:, ints]))
        if rng is not None:
            frac = frac + rng.uniform(0, 1e-3, frac.shape)
        order = ints[np.argsort(frac, axis=1, kind="stable")]
        Xr = self._t(np.round(X))
        Xt = self._t(X)
        for block in np.array_split(np.arange(len(ints)), max(1, min(chunks, len(ints)))):
            g = torch.as_tensor(order[:, block], device=self.be.device)              # (P, chunk)
            v = torch.clamp(Xr.gather(1, g), L.gather(1, g), U.gather(1, g))
            v = torch.where(torch.isfinite(v), v, Xr.gather(1, g))
            fix = ~bad[:, None].expand_as(v)                                       # bad points stop fixing
            L = L.scatter(1, g, torch.where(fix, v, L.gather(1, g)))
            U = U.scatter(1, g, torch.where(fix, v, U.gather(1, g)))
            L, U, bad = self.propagate_batch_t(L, U, bad)
        lo = np.asarray(L.cpu(), dtype=np.float64)
        up = np.asarray(U.cpu(), dtype=np.float64)
        b = np.asarray(bad.cpu())
        X[:, ints] = np.round(np.clip(X[:, ints], np.minimum(lo[:, ints], up[:, ints]), np.maximum(lo[:, ints], up[:, ints])))
        cont = ~self.integer_np
        X[:, cont] = np.clip(X[:, cont], lo[:, cont], up[:, cont])
        del Xt
        return X, lo, up, b
