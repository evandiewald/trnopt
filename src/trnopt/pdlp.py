"""A readable PDLP (restarted PDHG) LP solver in PyTorch.

The big idea, in plain terms
----------------------------
An LP has two "sides":

* the primal: the variables x you actually care about (how much of each loan to take);
* the dual: one price y_i per constraint, saying how much the objective would improve
  if that constraint were loosened a little (the "shadow price" of the budget, etc.).

At the optimum the two sides agree: the best primal objective equals the best dual
objective. PDHG ("primal-dual hybrid gradient") hunts for that agreement by alternating
two cheap moves, over and over:

1. primal step: nudge x downhill on   cost - (prices * constraints),  then clip x back
   into its bounds;
2. dual step:   nudge the prices y up where constraints are violated and down where they
   are slack, then clip y so ">=" constraints never get negative prices.

Each move costs one multiply by the constraint matrix K (or its transpose). There is no
factorization and no sequential pivoting, which is why it maps well onto GPUs -- and,
with the right problem shapes, possibly onto Trainium.

Plain PDHG converges slowly. PDLP is PDHG plus four practical tricks, all implemented here:

* scaling:      rescale rows/columns of K so no constraint or variable dominates;
* primal weight: balance step sizes between the x side and the y side;
* restarts:     periodically jump to the best point seen recently (often the running
                average) and start fresh from there -- this is the single biggest speedup;
* stopping test: stop when primal violation, dual violation and the primal/dual gap are all
                tiny relative to the problem's size.

It also detects LPs with no solution (infeasible) or with an objective that improves forever
(unbounded), and can borrow HiGHS's presolve to shrink the problem first.

One deliberate simplification versus Google's PDLP / NVIDIA's cuPDLP: we use a *fixed* step
size (computed once from the matrix) instead of their adaptive step-size search. Fixed steps
mean every iteration does exactly the same work with no data-dependent retry loops, which is
the shape that hardware accelerators -- Trainium included -- like best.

How the code is laid out for accelerators
-----------------------------------------
* `_Problem.step` is one full PDHG iteration written as a pure function of tensors. With
  `Backend.compile=True` it is fused by torch.compile into a single step on the device.
* The loop only reads numbers back to the host (`.item()`) every `eval_every` iterations,
  when it checks for convergence. Each read forces the host to wait for the chip.
* Step sizes are tensors, not Python floats. A compiler bakes Python numbers in as constants,
  so changing one (at every restart) would trigger a full recompile.

References: Applegate et al., "Practical Large-Scale Linear Programming using Primal-Dual
Hybrid Gradient" (NeurIPS 2021); Applegate et al., "Infeasibility detection with primal-dual
hybrid gradient for large-scale linear programming" (2021); Lu & Yang, "cuPDLP.jl" (2023).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp
import torch

from .backend import Backend, MatVec
from .lp import LP
from .presolve import Presolver


INFEAS_WARMUP = 4      # evaluations before infeasibility checks start
INFEAS_STREAK = 3      # consecutive evaluations that must agree before declaring infeasible/unbounded
RAY_COS = 0.99         # ...and the drift direction must be this steady between checks (cosine similarity)
STALL_GATE = 1e-4      # precision stalls are only declared below this relative error
STALL_FRACTION = 2     # stall window = at least 1/STALL_FRACTION of the evaluations so far


@dataclass
class PDLPOptions:
    tol: float = 1e-6                 # relative tolerance for the stopping test
    max_iters: int = 200_000
    time_limit: float = 600.0         # seconds
    eval_every: int = 64              # how often to check stopping / restart conditions
    ruiz_iters: int = 10              # scaling passes
    step_safety: float = 0.9          # fraction of the largest theoretically safe step
    infeasibility_tol: float | None = None  # how clean a "no solution" proof must be before we trust it.
                                      # None = 1e-8 in float64, 1e-4 in float32: float32's rounding
                                      # noise (~1e-7 per operation) stops proofs getting cleaner than ~1e-5.
    presolve: bool = False            # shrink the LP with HiGHS's presolve first
    finish_float64: bool = False      # float32 backends: solve to device_tol on the device, then
                                      # finish to `tol` in float64 on the CPU from that answer
    device_tol: float = 1e-6          # float32 phase target when finish_float64; it usually hands over
                                      # earlier, as soon as progress stalls (see stall_window)
    stall_window: int = 30            # evaluations (x eval_every iterations) over which the error
                                      # must at least halve, or the float32 phase hands over
    backend: Backend = field(default_factory=Backend)
    verbose: bool = False


@dataclass
class PDLPResult:
    status: str                       # "optimal", "inaccurate", "infeasible", "unbounded", "stalled",
                                      # "unbounded_or_infeasible", "iteration_limit", "time_limit"
                                      # or "numerical_error". "inaccurate" = converged on the presolved
                                      # LP, but postsolve pushed violations above the tolerance.
    objective: float                  # c @ x for the LP as given (see LP docstring)
    x: np.ndarray                     # primal solution (best point found if not optimal)
    y: np.ndarray | None              # dual solution, one price per row (None when presolve was used)
    iterations: int
    restarts: int
    seconds: float
    primal_residual: float            # relative constraint violation
    dual_residual: float              # relative dual violation
    gap: float                        # relative primal/dual objective gap
    presolve: str = "off"             # what presolve did, e.g. "removed 148 rows, 132 cols"
    finish_iterations: int = 0        # of `iterations`, how many ran in the float64 CPU finish
    omega: float = 1.0                # final primal weight (lets a warm start keep the learned step balance)


# --------------------------------------------------------------------------------------
# Scaling and step size (computed once, on the CPU, in float64)
# --------------------------------------------------------------------------------------

def _scale(K: sp.csr_matrix, ruiz_iters: int):
    """Return row scales d_r and column scales d_c so that diag(d_r) @ K @ diag(d_c) is well balanced.

    Ruiz equilibration repeatedly divides every row and column by the square root of its
    largest entry; a final Pock-Chambolle pass uses row/column sums instead. Badly scaled
    data (one constraint in dollars, another in basis points) is the most common reason
    first-order methods like PDHG stall, so this matters a lot in practice.
    """
    A = abs(K).tocsr()
    m, n = A.shape
    d_r, d_c = np.ones(m), np.ones(n)
    if A.nnz == 0:
        return d_r, d_c

    def inv_sqrt(v):
        v = np.asarray(v).ravel()
        out = np.ones_like(v)
        nz = v > 0
        out[nz] = 1.0 / np.sqrt(v[nz])
        return out

    for _ in range(ruiz_iters):
        r = inv_sqrt(A.max(axis=1).toarray())
        s = inv_sqrt(A.max(axis=0).toarray())
        A = sp.diags(r) @ A @ sp.diags(s)
        d_r *= r
        d_c *= s
    r = inv_sqrt(A.sum(axis=1))
    s = inv_sqrt(A.sum(axis=0))
    return d_r * r, d_c * s


def _operator_norm(K: sp.csr_matrix, iters: int = 200) -> float:
    """Largest singular value of K by power iteration, done on the CPU in float64.

    It runs once per solve. Doing it on the device would mean ~100 round trips to read the
    estimate back, which is exactly the kind of host-device chatter accelerators punish.
    """
    if K.nnz == 0:
        return 0.0
    v = np.random.default_rng(0).normal(size=K.shape[1])
    v /= np.linalg.norm(v)
    sigma2 = 0.0
    for _ in range(iters):
        w = K.T @ (K @ v)
        nrm = np.linalg.norm(w)
        if nrm == 0.0:
            return 0.0
        v = w / nrm
        if abs(nrm - sigma2) <= 1e-8 * nrm:
            break
        sigma2 = nrm
    return math.sqrt(nrm)


# --------------------------------------------------------------------------------------
# The scaled problem on the device
# --------------------------------------------------------------------------------------

class _Problem:
    """The scaled problem held as tensors on the chosen device, plus helpers to measure solution quality."""

    def __init__(self, lp: LP, opts: PDLPOptions):
        be = opts.backend
        self.be = be
        self.d_r, self.d_c = _scale(lp.K, opts.ruiz_iters)
        Ks = (sp.diags(self.d_r) @ lp.K @ sp.diags(self.d_c)).tocsr()
        self.knorm = _operator_norm(Ks)
        self.mv = MatVec(Ks, be)

        self.c = be.tensor(lp.c * self.d_c)
        self.q = be.tensor(lp.q * self.d_r)
        self.lower = be.tensor(lp.lower / self.d_c)          # +/-inf stays +/-inf
        self.upper = be.tensor(lp.upper / self.d_c)
        self.dr = be.tensor(self.d_r)
        self.dc = be.tensor(self.d_c)
        self.n, self.m = lp.n, lp.m
        ineq = np.zeros(lp.m, dtype=bool)
        ineq[lp.m_eq :] = True
        self.ineq = torch.as_tensor(ineq, device=be.device)  # True on ">=" rows

        self.lower_finite = torch.isfinite(self.lower)
        self.upper_finite = torch.isfinite(self.upper)
        zero = torch.zeros_like(self.lower)
        self.lower_or_0 = torch.where(self.lower_finite, self.lower, zero)
        self.upper_or_0 = torch.where(self.upper_finite, self.upper, zero)
        self.c_norm = float(np.linalg.norm(lp.c))   # norms of the *original* data, for relative tolerances
        self.q_norm = float(np.linalg.norm(lp.q))
        self.c_scaled_max = float(np.abs(lp.c * self.d_c).max(initial=0.0))
        self.q_scaled_max = float(np.abs(lp.q * self.d_r).max(initial=0.0))

        compile_ = (lambda f: torch.compile(f, backend=be.compile_backend or "inductor")) if be.compile else (lambda f: f)
        self.step = compile_(self._step)
        self.eval_stats = compile_(self._eval_stats)

    # ---- the core PDHG iteration -----------------------------------------------------
    def _step(self, x, y, Kx, KTy, x_sum, y_sum, tau, sigma):
        """One PDHG iteration: two matrix-vector products and a few elementwise operations.

        Written as a pure function of tensors (no Python numbers, no reads back to the host)
        so torch.compile can turn it into a single fused step.
        """
        x_new = torch.clamp(x - tau * (self.c - KTy), self.lower, self.upper)
        Kx_new = self.mv.K(x_new)
        # The dual step uses the "extrapolated" point 2*x_new - x (that's the "hybrid" in PDHG);
        # K @ (2 x_new - x) = 2 Kx_new - Kx, so no extra multiply is needed.
        y_new = y + sigma * (self.q - (2.0 * Kx_new - Kx))
        y_new = self.project_dual(y_new)
        KTy_new = self.mv.KT(y_new)
        return x_new, y_new, Kx_new, KTy_new, x_sum + x_new, y_sum + y_new

    def project_dual(self, y: torch.Tensor) -> torch.Tensor:
        """Prices on '>=' rows can't be negative; equality-row prices are free.

        Done with a mask rather than slicing and gluing the vector back together: same math,
        fixed shapes, and much friendlier to compilers.
        """
        return torch.where(self.ineq, torch.clamp(y, min=0.0), y)

    # ---- measuring progress (only at evaluation points) --------------------------------
    #
    # Everything the convergence / restart / infeasibility checks need is computed on the device
    # and returned as ONE small tensor, which the host reads in a single transfer. Reading values
    # one at a time would mean ~25 separate round trips per check -- and on the XLA backend, each
    # read also cuts the recorded recipe at a different place, producing dozens of distinct
    # programs to compile. (We measured 36 compiles before this change on afiro.)

    def point_stats(self, x, y, Kx, KTy) -> torch.Tensor:
        """[primal violation, dual violation (original units), the same two in scaled units,
        primal objective, dual objective] for one (x, y) point."""
        # Primal violation: equality rows should be exactly q, '>=' rows at least q.
        resid = Kx - self.q
        viol = torch.where(self.ineq, torch.clamp(-resid, min=0.0), resid)

        # Reduced cost r = c - K^T y. Where x has a finite bound, a nonzero r is fine (the
        # bound "absorbs" it and it enters the dual objective). Where x has no bound in that
        # direction, any r is a dual violation.
        r = self.c - KTy
        r_pos, r_neg = torch.clamp(r, min=0.0), torch.clamp(-r, min=0.0)
        # A reduced cost is credited to a bound (in the dual objective) only if x is actually near
        # that bound; otherwise it counts as a dual violation. Google's PDLP does the same
        # ("handle_some_primal_gradients_on_finite_bounds_as_residuals"). Why it matters in float32:
        # r carries rounding noise on every variable, and the clamps turn that noise one-sided, so
        # crediting it to bounds biases the dual objective by ~(number of variables) x noise x bound.
        # On comp21-2idx branch LPs (objective 0, ~10k variables with upper bounds up to 3) that bias
        # alone kept the relative gap above ~5e-3 forever; float64 solved the same LPs in ~900 iterations.
        near_lo = self.lower_finite & (torch.abs(x - self.lower_or_0) <= torch.abs(x))
        near_up = self.upper_finite & (torch.abs(x - self.upper_or_0) <= torch.abs(x))
        dual_viol = torch.where(near_lo, 0.0, r_pos) + torch.where(near_up, 0.0, r_neg)

        pobj = torch.dot(self.c, x)
        dobj = (torch.dot(self.q, y) + torch.dot(torch.where(near_lo, self.lower_or_0, 0.0), r_pos)
                - torch.dot(torch.where(near_up, self.upper_or_0, 0.0), r_neg))
        return torch.stack([
            torch.linalg.norm(viol / self.dr), torch.linalg.norm(dual_viol / self.dc),
            torch.linalg.norm(viol), torch.linalg.norm(dual_viol),
            pobj, dobj,
        ])

    def ray_stats(self, dx, dy, pdx, pdy, x, y) -> torch.Tensor:
        """Numbers for the infeasibility / unboundedness test on the recent drift (dx, dy).

        When an LP has an optimal solution, PDHG's iterates settle down. When it doesn't, they
        drift off in a steady direction, and that direction is a *proof*:

        * Infeasible (no x satisfies the constraints): the prices y drift along a direction that
          makes the constraints contradict each other -- e.g. "x1 + x2 >= 5" and "x1 + x2 <= 3"
          get prices that add up to "0 >= 2". This is called a Farkas certificate.
        * Unbounded (the objective improves forever): x drifts along a direction that keeps
          every constraint satisfied while the cost keeps falling.
        """
        # Infeasibility proof from the dual drift dy.
        dray = self.project_dual(dy)
        r = -self.mv.KT(dray)
        r_pos, r_neg = torch.clamp(r, min=0.0), torch.clamp(-r, min=0.0)
        bad = torch.where(self.lower_finite, 0.0, r_pos) + torch.where(self.upper_finite, 0.0, r_neg)
        dobj = torch.dot(self.q, dray) + torch.dot(self.lower_or_0, r_pos) - torch.dot(self.upper_or_0, r_neg)

        # Unboundedness proof from the primal drift dx, kept inside the bounds' directions.
        pray = torch.where(self.lower_finite, torch.clamp(dx, min=0.0), dx)
        pray = torch.where(self.upper_finite, torch.clamp(pray, max=0.0), pray)
        Kr = self.mv.K(pray)
        viol = torch.where(self.ineq, torch.clamp(-Kr, min=0.0), Kr)

        # Is the drift heading in a steady direction? A real proof is a fixed direction; on
        # MIPLIB's neos-3627168-kasai (feasible) float32 saw a "proof" whose direction flipped
        # back and forth between checks (cosine -0.99, 0.97, -0.96, ...).
        cos = lambda a, b: torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b) + 1e-30)
        return torch.stack([
            torch.linalg.norm(dray), dobj, torch.linalg.norm(bad),
            torch.linalg.norm(pray), torch.dot(self.c, pray), torch.linalg.norm(viol),
            cos(dy, pdy), cos(dx, pdx),
            # Size of the variables a "leftover" error multiplies: x on sides with no bound, and y.
            # A 2e-6 leftover is harmless when those are ~1 and decisive when they're ~1e6 (kasai).
            torch.where(self.lower_finite & self.upper_finite, 0.0, torch.abs(x)).max(),
            torch.abs(y).max() if y.numel() else torch.zeros((), dtype=x.dtype, device=x.device),
        ])

    def _eval_stats(self, x, y, Kx, KTy, x_avg, y_avg, Kx_avg, KTy_avg, dx, dy, pdx, pdy) -> torch.Tensor:
        finite = (torch.isfinite(x).all() & torch.isfinite(y).all()).to(x.dtype).reshape(1)
        return torch.cat([
            finite,
            self.point_stats(x, y, Kx, KTy),
            self.point_stats(x_avg, y_avg, Kx_avg, KTy_avg),
            self.ray_stats(dx, dy, pdx, pdy, x, y),
        ])

    # ---- turning those numbers into decisions (on the host, plain Python) ------------------
    def relative(self, st) -> tuple[float, float, float, float]:
        """(relative primal violation, relative dual violation, relative gap, primal objective)."""
        p, d, _, _, pobj, dobj = st
        return (p / (1.0 + self.q_norm), d / (1.0 + self.c_norm),
                abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj)), pobj)

    @staticmethod
    def kkt(st, omega: float) -> float:
        """One number summarizing 'how far from optimal' -- used to decide when to restart."""
        _, _, p, d, pobj, dobj = st
        return math.sqrt(omega * p * p + d * d / omega + (pobj - dobj) ** 2)

    def no_solution(self, rs, tol: float) -> str | None:
        """Is the recent drift a proof that the LP is infeasible or unbounded (up to tol)?"""
        dnorm, dobj, dbad, pnorm, pobj, pviol, dcos, pcos, xfree, ymax = rs
        # dobj > 0 is impossible for a feasible LP; demand it clearly above rounding noise, that
        # the leftover error stays small even multiplied by the size of the variables it acts on,
        # and that the drift direction is steady (see ray_stats).
        if (dnorm > 0 and dobj > tol * dnorm * (1.0 + self.q_scaled_max)
                and dbad * max(1.0, xfree) <= tol * dobj and dcos >= RAY_COS):
            return "infeasible"
        if (pnorm > 0 and -pobj > tol * pnorm * (1.0 + self.c_scaled_max)
                and pviol * max(1.0, ymax) <= tol * -pobj and pcos >= RAY_COS):
            return "unbounded"
        return None


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------

def solve_pdlp(lp: LP, opts: PDLPOptions | None = None) -> PDLPResult:
    opts = opts or PDLPOptions()
    if not opts.presolve:
        return _solve_full(lp, opts)

    t0 = time.perf_counter()
    pre = Presolver(lp)
    out = pre.outcome
    n = lp.n

    def finished(status, x, res=None, note=""):
        x = np.zeros(n) if x is None else x
        return PDLPResult(
            status=status,
            objective=float(lp.c @ x),
            x=x,
            y=None,
            iterations=res.iterations if res else 0,
            restarts=res.restarts if res else 0,
            seconds=time.perf_counter() - t0,
            primal_residual=_relative_primal_residual(lp, x),
            dual_residual=res.dual_residual if res else 0.0,
            gap=res.gap if res else 0.0,
            presolve=note,
        )

    if out.status in ("infeasible", "unbounded_or_infeasible"):
        return finished(out.status, None, note=f"presolve proved {out.status}")
    if out.status == "solved":
        return finished("optimal", pre.postsolve(None), note="presolve solved it outright")
    if out.status == "not_reduced":
        res = _solve_full(lp, opts)
        res.presolve = "no reduction"
        res.seconds = time.perf_counter() - t0
        return res

    # Postsolve rebuilds eliminated variables from the kept ones, so small errors in the kept
    # variables can grow (about 30x on Netlib's `shell`). Check the answer on the ORIGINAL LP;
    # if it misses the tolerance, re-solve the reduced LP more tightly, warm-starting from the
    # last answer so the retry is cheap.
    note = f"removed {out.rows_removed} rows, {out.cols_removed} cols"
    # float32 often can't get much below ~1e-5 (see finish_float64), so retries stop at 1e-6; a retry
    # that can't get there just runs out of iterations and the answer is reported as "inaccurate".
    floor = 1e-9 if opts.backend.dtype == torch.float64 or opts.finish_float64 else 1e-6
    reduced_tol, x0, y0, iters, restarts = opts.tol, None, None, 0, 0
    for attempt in range(3):
        res = _solve_full(out.reduced, replace(opts, tol=reduced_tol), x0, y0)
        iters, restarts = iters + res.iterations, restarts + res.restarts
        res.iterations, res.restarts = iters, restarts
        if res.status in ("infeasible", "unbounded"):
            return finished(res.status, None, res, note)
        x = pre.postsolve(res.x)
        if res.status != "optimal" or _relative_primal_residual(lp, x) <= opts.tol:
            return finished(res.status, x, res, note)
        if reduced_tol <= floor:
            break
        reduced_tol = max(reduced_tol / 10.0, floor)
        x0, y0 = res.x, res.y
        note += f"; re-solved at tol {reduced_tol:.0e}"
    return finished("inaccurate", x, res, note)


def _solve_full(lp: LP, opts: PDLPOptions, x0=None, y0=None) -> PDLPResult:
    """_solve, plus the optional float64 finish.

    float32 (all Trainium offers) carries ~7 significant digits, so PDHG's progress stalls
    somewhere around 1e-5 relative error on harder problems. We let the device do the bulk of
    the iterations to `device_tol`, then hand the answer to a float64 CPU solve that only has to
    close the last gap. The CPU phase starts from a nearly-converged point, so it's short.
    """
    if not opts.finish_float64 or opts.backend.dtype == torch.float64:
        return _solve(lp, opts, x0, y0)
    first = _solve(lp, replace(opts, tol=max(opts.tol, opts.device_tol)), x0, y0, stop_on_stall=True)
    if first.status in ("infeasible", "unbounded", "numerical_error", "time_limit"):
        return first
    if max(first.primal_residual, first.dual_residual, first.gap) <= opts.tol:
        return first
    finish = _solve(
        lp,
        replace(opts, backend=Backend.preset("cpu"), max_iters=max(opts.max_iters - first.iterations, 1),
                time_limit=max(opts.time_limit - first.seconds, 1.0)),
        first.x, first.y, omega0=first.omega,
    )
    finish.iterations += first.iterations
    finish.restarts += first.restarts
    finish.seconds += first.seconds
    finish.finish_iterations = finish.iterations - first.iterations
    return finish


def _relative_primal_residual(lp: LP, x: np.ndarray) -> float:
    Kx = lp.K @ x
    viol = np.concatenate([Kx[: lp.m_eq] - lp.q[: lp.m_eq], np.minimum(Kx[lp.m_eq :] - lp.q[lp.m_eq :], 0.0)])
    bound_viol = np.maximum(lp.lower - x, 0.0) + np.maximum(x - lp.upper, 0.0)
    return float(np.sqrt(viol @ viol + bound_viol @ bound_viol) / (1.0 + np.linalg.norm(lp.q)))


def _solve(lp: LP, opts: PDLPOptions, x0: np.ndarray | None = None, y0: np.ndarray | None = None,
           stop_on_stall: bool = False, omega0: float | None = None) -> PDLPResult:
    """Run restarted PDHG on `lp`. x0 / y0 (optional) warm-start from a previous answer.

    stop_on_stall=True returns status "stalled" once the error stops halving within
    opts.stall_window evaluations (used to decide when float32 has done all it usefully can).
    """
    t0 = time.perf_counter()
    P = _Problem(lp, opts)
    be = opts.backend

    # Step size: PDHG is guaranteed stable when tau * sigma * ||K||^2 < 1.
    # We write tau = eta / omega, sigma = eta * omega, so the condition is eta < 1 / ||K||.
    eta = opts.step_safety / P.knorm if P.knorm > 0 else 1.0
    infeas_tol = opts.infeasibility_tol
    if infeas_tol is None:
        infeas_tol = 1e-8 if be.dtype == torch.float64 else 1e-4

    # Primal weight omega balances how fast x moves vs how fast y moves.
    cn, qn = torch.linalg.norm(P.c).item(), torch.linalg.norm(P.q).item()
    omega = cn / qn if cn > 1e-10 and qn > 1e-10 else 1.0
    if omega0 is not None:
        omega = omega0
    tau, sigma = be.tensor(eta / omega), be.tensor(eta * omega)

    # Start at x = 0 (clipped into bounds), y = 0 -- or from a previous answer, moved into scaled units.
    x = be.tensor(np.zeros(P.n) if x0 is None else x0 / P.d_c)
    y = be.tensor(np.zeros(P.m) if y0 is None else y0 / P.d_r)
    x = torch.clamp(x, P.lower, P.upper)
    y = P.project_dual(y)
    Kx, KTy = P.mv.K(x), P.mv.KT(y)

    # Restart bookkeeping
    x_sum, y_sum, n_avg = torch.zeros_like(x), torch.zeros_like(y), 0
    x_restart, y_restart = x.clone(), y.clone()
    x_prev_eval, y_prev_eval = x.clone(), y.clone()
    dx_prev, dy_prev = torch.zeros_like(x), torch.zeros_like(y)    # previous drift, for the steadiness test
    best_stats = P.point_stats(x, y, Kx, KTy).tolist()    # one read: stats of the starting point
    kkt_last_restart = P.kkt(best_stats, omega)
    kkt_prev_candidate = math.inf
    iters_since_restart = 0
    restarts = 0

    status = "iteration_limit"
    best = (x, y, Kx, KTy)
    err_history: list[float] = []     # best error so far, at each evaluation (for stall detection)
    verdict_streak, last_verdict = 0, None
    it = 0
    while it < opts.max_iters:
        # ---- the hot loop: no reads back to the host in here --------------------------
        x, y, Kx, KTy, x_sum, y_sum = P.step(x, y, Kx, KTy, x_sum, y_sum, tau, sigma)
        be.sync()   # XLA: run this iteration's recipe now (compiled once, then reused)
        n_avg += 1
        it += 1
        iters_since_restart += 1
        if it % opts.eval_every != 0:
            continue

        # ---- every eval_every iterations: ONE read of everything the checks need --------
        x_avg, y_avg = x_sum / n_avg, y_sum / n_avg
        Kx_avg, KTy_avg = P.mv.K(x_avg), P.mv.KT(y_avg)
        dx_now, dy_now = x - x_prev_eval, y - y_prev_eval
        stats = P.eval_stats(x, y, Kx, KTy, x_avg, y_avg, Kx_avg, KTy_avg, dx_now, dy_now, dx_prev, dy_prev).tolist()
        finite, cur_st, avg_st, ray_st = stats[0], stats[1:7], stats[7:13], stats[13:23]
        dx_prev, dy_prev = dx_now, dy_now
        if finite < 1.0 or not all(map(math.isfinite, stats)):
            status = "numerical_error"
            break

        # Stop if either the current point or the running average is good enough.
        for cand, st in (((x, y, Kx, KTy), cur_st), ((x_avg, y_avg, Kx_avg, KTy_avg), avg_st)):
            rp, rd, gap, _ = P.relative(st)
            if rp <= opts.tol and rd <= opts.tol and gap <= opts.tol:
                status, best, best_stats = "optimal", cand, st
                break
        if status == "optimal":
            break

        # Does the recent drift prove there's no solution? Early on, iterates move fast and can
        # look like a proof by accident; in float32, with its looser threshold, that produced a
        # false "infeasible" on MIPLIB's cost266-UUE at iteration 128. So we skip the first few
        # checks and require the same verdict at INFEAS_STREAK checks in a row. A genuine proof
        # persists indefinitely, so this only delays real detections slightly.
        verdict = P.no_solution(ray_st, infeas_tol) if it > INFEAS_WARMUP * opts.eval_every else None
        verdict_streak = verdict_streak + 1 if verdict and verdict == last_verdict else (1 if verdict else 0)
        last_verdict = verdict
        if verdict_streak >= INFEAS_STREAK:
            status = verdict
            break
        if time.perf_counter() - t0 > opts.time_limit:
            status = "time_limit"
            break

        # Restart candidate: whichever of current / average is closer to optimal.
        kkt_cur, kkt_avg = P.kkt(cur_st, omega), P.kkt(avg_st, omega)
        if kkt_avg < kkt_cur:
            cand, best_stats, kkt_cand = (x_avg, y_avg, Kx_avg, KTy_avg), avg_st, kkt_avg
        else:
            cand, best_stats, kkt_cand = (x, y, Kx, KTy), cur_st, kkt_cur
        best = cand

        if stop_on_stall:
            err = max(P.relative(best_stats)[:3])
            err_history.append(min(err, err_history[-1]) if err_history else err)
            # Only call it a precision stall once the error is small enough for float32's ~7 digits
            # to plausibly be the limit, and the window grows with the run so slow-but-steady
            # problems (which plateau even in float64) aren't cut off early.
            w = max(opts.stall_window, len(err_history) // STALL_FRACTION)
            if err_history[-1] < STALL_GATE and len(err_history) > w and err_history[-1] > 0.5 * err_history[-1 - w]:
                status = "stalled"
                break

        if opts.verbose:
            rp, rd, gap, pobj = P.relative(best_stats)
            print(f"iter {it:7d}  obj {pobj: .8e}  primal {rp:.1e}  dual {rd:.1e}  gap {gap:.1e}  omega {omega:.2e}")

        sufficient = kkt_cand <= 0.2 * kkt_last_restart                        # big progress
        stalled = kkt_cand <= 0.8 * kkt_last_restart and kkt_cand > kkt_prev_candidate  # some progress, now stuck
        too_long = iters_since_restart >= 0.36 * it                            # safety net
        kkt_prev_candidate = kkt_cand

        if sufficient or stalled or too_long:
            x, y, Kx, KTy = (t.clone() for t in cand)
            # Re-balance the primal weight using how far each side moved since the last restart.
            dx, dy = torch.stack([torch.linalg.norm(x - x_restart), torch.linalg.norm(y - y_restart)]).tolist()
            if dx > 1e-10 and dy > 1e-10:
                omega = math.exp(0.5 * math.log(dy / dx) + 0.5 * math.log(omega))
                tau.fill_(eta / omega)        # update in place: same tensors, so no recompile
                sigma.fill_(eta * omega)
            x_restart, y_restart = x.clone(), y.clone()
            x_sum, y_sum = torch.zeros_like(x), torch.zeros_like(y)
            n_avg = 0
            iters_since_restart = 0
            kkt_last_restart = P.kkt(best_stats, omega)   # the new start IS the candidate: no re-measure
            kkt_prev_candidate = math.inf
            restarts += 1
        x_prev_eval, y_prev_eval = x.clone(), y.clone()

    xb, yb, Kxb, KTyb = best
    rp, rd, gap, pobj = P.relative(best_stats)
    return PDLPResult(
        status=status,
        objective=pobj,
        # Undo the scaling, and clip to the bounds: the best point can be an average of iterates, and
        # in float32 a sum of values sitting exactly on a bound drifts a few ulps past it.
        x=np.clip((xb * P.dc).cpu().double().numpy(), lp.lower, lp.upper),
        y=(yb * P.dr).cpu().double().numpy(),
        iterations=it,
        restarts=restarts,
        seconds=time.perf_counter() - t0,
        primal_residual=rp,
        dual_residual=rd,
        gap=gap,
        omega=omega,
    )
