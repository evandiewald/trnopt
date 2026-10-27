"""Experiment: restarted reflected Halpern PDHG (r2HPDHG, the method in cuPDLPx and NVIDIA's batched
paper) vs trnopt's restarted averaged PDHG (PDLP). A prototype, not wired into the solver.

    uv run python examples/halpern_experiment.py [miplib instance names...]

What's different from PDLP, in plain terms:
  * PDLP restarts from the AVERAGE of recent iterates. Halpern instead pulls every step a little back
    toward a fixed "anchor" point (weight 1/(k+2) at step k since restart), which provably converges
    faster in the worst case.
  * Reflection overshoots each step: (1+gamma) * PDHG(z) - gamma * z. gamma = 1 is the theory's choice.
  * Restarts are decided by the "fixed-point residual" (how far one PDHG step moves the point), with
    PDLP's thresholds (0.2 / 0.8 / 0.36).

Result (float32, tol 1e-4, 30 s limit, 2-core VM; see the README):
  * gamma = 1: 2.7x fewer iterations on facility location, median 1.5x on MIPLIB, but solved 61/79
    MIPLIB relaxations vs PDLP's 68 (unstable on harder instances in float32).
  * gamma = 0.5: solved 66/79, median 1.28x fewer iterations, but MORE total iterations overall
    (592k vs 502k) because some hard instances got slower. Facility location 11.3k vs 17.5k.
  So: worthwhile on some families, not a default. cuPDLPx reports 2.5-3.6x on MIPLIB with pieces this
  prototype lacks (PID-controlled primal weight, tuned constants); worth revisiting.
"""
import math, sys, time
import numpy as np
import torch
from trnopt import Backend, PDLPOptions, load_mps, solve_pdlp
from trnopt.pdlp import _Problem


def r2h(lp, tol=1e-4, backend="cpu-float32", gamma=1.0, safety=0.998, eval_every=64, max_iters=100_000,
        weight_smooth=0.5, restart_at="T", time_limit=30.0):
    opts = PDLPOptions(tol=tol, backend=Backend.preset(backend))
    P = _Problem(lp, opts)
    be = opts.backend
    eta = safety / P.knorm
    cn, qn = torch.linalg.norm(P.c).item(), torch.linalg.norm(P.q).item()
    omega = cn / qn if cn > 1e-10 and qn > 1e-10 else 1.0
    tau, sigma = be.tensor(eta / omega), be.tensor(eta * omega)
    x = torch.clamp(be.tensor(np.zeros(P.n)), P.lower, P.upper)
    y = P.project_dual(be.tensor(np.zeros(P.m)))
    Kx, KTy = P.mv.K(x), P.mv.KT(y)
    anchor = (x, y, Kx, KTy)
    zx, zy = torch.zeros_like(x), torch.zeros_like(y)
    k, it, restarts = 0, 0, 0
    r0, r_prev = None, math.inf
    status = "iteration_limit"
    t0 = time.time()
    pobj = float("nan")
    while it < max_iters:
        xh, yh, Kxh, KTyh, _, _ = P._step(x, y, Kx, KTy, zx, zy, tau, sigma)    # T(z)
        a, b = (k + 1) / (k + 2), 1 / (k + 2)
        xb, yb, Kxb, KTyb = x, y, Kx, KTy                                        # z before the step
        x = a * ((1 + gamma) * xh - gamma * x) + b * anchor[0]
        y = a * ((1 + gamma) * yh - gamma * y) + b * anchor[1]
        Kx = a * ((1 + gamma) * Kxh - gamma * Kx) + b * anchor[2]
        KTy = a * ((1 + gamma) * KTyh - gamma * KTy) + b * anchor[3]
        k += 1
        it += 1
        if it % eval_every:
            continue
        st = P.point_stats(xh, yh, Kxh, KTyh).tolist()
        rp, rd, gap, pobj = P.relative(st)
        if rp <= tol and rd <= tol and gap <= tol:
            status = "optimal"
            break
        if time.time() - t0 > time_limit:
            status = "time_limit"
            break
        dx, dy = xb - xh, yb - yh
        r2 = (dx @ dx) / tau + (dy @ dy) / sigma - 2 * (dy @ (Kxb - Kxh))
        r = math.sqrt(max(r2.item(), 0.0))
        if r0 is None:
            r0 = r
        if r <= 0.2 * r0 or (r <= 0.8 * r0 and r > r_prev) or k >= 0.36 * it:
            nx, ny = (xh, yh) if restart_at == "T" else (x, y)
            mx = torch.linalg.norm(nx - anchor[0]).item()
            my = torch.linalg.norm(ny - anchor[1]).item()
            if mx > 1e-10 and my > 1e-10:
                omega = math.exp(weight_smooth * math.log(my / mx) + (1 - weight_smooth) * math.log(omega))
                tau.fill_(eta / omega)
                sigma.fill_(eta * omega)
            if restart_at == "T":
                x, y, Kx, KTy = xh.clone(), yh.clone(), Kxh.clone(), KTyh.clone()
            anchor = (x.clone(), y.clone(), Kx.clone(), KTy.clone())
            k, r0, r_prev = 0, r, math.inf
            restarts += 1
        else:
            r_prev = r
    return status, it, restarts, pobj


if __name__ == "__main__":
    from trnopt.generators import facility_location, set_cover, independent_set, combinatorial_auction
    probs = [facility_location(400, 200), set_cover(1000, 1000, 0.01), independent_set(1000, 10),
             combinatorial_auction(360, 1400)]
    for nm in sys.argv[1:]:
        probs.append(load_mps(f"data/miplib/{nm}.mps.gz"))
    for lp in probs:
        t = time.time(); a = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("cpu-float32"), time_limit=300)); ta = time.time() - t
        t = time.time(); s, i, rs, obj = r2h(lp); tb = time.time() - t
        print(f"{lp.name:28s} current: {a.status[:4]} {a.iterations:>7} it {ta:6.1f}s obj {a.objective:.6e} | "
              f"r2HPDHG: {s[:4]} {i:>7} it {rs:>3} restarts {tb:6.1f}s obj {obj:.6e}", flush=True)
