"""Full strong branching, the workload in NVIDIA's batched-PDLP paper (arXiv 2601.21990).

    uv run python examples/strong_branching_benchmark.py                 # smaller size of each family + chromaticindex512-7
    uv run python examples/strong_branching_benchmark.py --quick         # smaller size of each family only
    uv run python examples/strong_branching_benchmark.py --all           # also the larger sizes (can take hours)
    uv run python examples/strong_branching_benchmark.py --instances facility_400x200

What strong branching is: at a node of a MIP search, for each integer variable whose LP value is
fractional (say x_j = 2.4), solve two LPs -- one with x_j <= 2, one with x_j >= 3 -- and see how
much each one worsens the objective. The variable whose branches worsen it most is the best one to
branch on. It's the most informative branching rule and the most expensive, which is why GPUs are
interesting for it: all those LPs share the same constraint matrix.

Methods, all solving the same LPs:
  * HiGHS cold     : each LP from scratch, one core, best of simplex / IPM / PDLP (sampled).
  * HiGHS warm     : each LP by dual simplex starting from the root LP's optimal basis, one LP per
                     CPU core in parallel. This is how a real CPU MIP solver does it, and the fair
                     baseline (the paper makes the same point about earlier GPU work).
  * trnopt CPU     : batched PDLP, float32, 128 LPs at a time, warm-started from the root solution.
  * trnopt GPU     : the same on the Mac GPU with the Metal kernel (if available).
  * trnopt CUDA    : the same on an NVIDIA GPU (--methods cuda-float32; e.g. the T4 in a g4dn).
  * cuOpt PDLP     : NVIDIA cuOpt's PDLP solving each LP in turn (it has no public batched-LP call),
                     warm-started from cuOpt's own root solve where its API allows (--methods cuopt-pdlp).
  * trnopt trn1    : the same on one Trainium NeuronCore (--methods neuron: dense K if it fits, else the
                     NKI kernel; neuron-dense / neuron-nki force one), or on both (--methods neuron-2core).

Reported: wall seconds for ALL the LPs, objective error vs HiGHS, and how good the variable chosen
from our scores is when judged by exact scores (1.0 = as good as the best; symmetric instances
often have many tied candidates, so "same variable" would be the wrong test).
Writes results/strong_branching_<host>_<date>.csv.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import datetime as dt
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import highspy
import numpy as np
import torch

from trnopt import Backend, PDLPOptions, load_mps, solve_pdlp
from trnopt.batch import solve_pdlp_stream
from trnopt.generators import GENERATORS, PAPER_SIZES
from trnopt.reference import to_highs_lp

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-4
# --methods name -> (label, backend factory)
METHODS = {
    "cpu-float32": ("trnopt CPU", lambda: Backend.preset("cpu-float32")),
    "mps-metal": ("trnopt GPU", lambda: Backend.preset("mps-metal")),
    "neuron-nki": ("trnopt trn1 NKI", lambda: Backend.preset("neuron", spmv="nki", dense=False)),
    "neuron": ("trnopt trn1", lambda: Backend.preset("neuron")),                 # dense if K fits, else NKI
    "neuron-dense": ("trnopt trn1 dense", lambda: Backend.preset("neuron", spmv="torch")),
    "neuron-2core": ("trnopt trn1 x2", None),                                    # both NeuronCores, see trn_two_cores.py
    "cuda-float32": ("trnopt CUDA", lambda: Backend.preset("cuda", dtype=torch.float32)),
    "cuda-float32-compiled": ("trnopt CUDA compiled", lambda: Backend.preset("cuda", dtype=torch.float32, compile=True)),
    "cuopt-pdlp": ("cuOpt PDLP per LP", None),                                   # see cuopt_adapter.py
    "cuopt-pdlp-f32": ("cuOpt PDLP f32 per LP", None),                           # pdlp_precision = single
    "cuopt-pdlp-warm": ("cuOpt PDLP warm per LP", None),                         # Stable2 mode + warm start
}

# ---- HiGHS, warm-started, in worker processes ------------------------------------------------
_H = None
_BASIS = None


def _worker_init(lp, root_basis_cols, root_basis_rows):
    global _H, _BASIS
    _H = highspy.Highs()
    _H.setOptionValue("output_flag", False)
    _H.setOptionValue("solver", "simplex")
    _H.setOptionValue("simplex_strategy", 1)            # dual simplex
    _H.setOptionValue("threads", 1)
    _H.passModel(to_highs_lp(lp))
    _BASIS = (root_basis_cols, root_basis_rows)


def _set_basis():
    b = highspy.HighsBasis()
    b.col_status = [highspy.HighsBasisStatus(int(s)) for s in _BASIS[0]]
    b.row_status = [highspy.HighsBasisStatus(int(s)) for s in _BASIS[1]]
    b.valid = True
    _H.setBasis(b)


def _warm_solve(task):
    j, lo, up, old_lo, old_up = task
    _H.changeColBounds(j, lo, up)
    _set_basis()
    t = time.perf_counter()
    _H.run()
    sec = time.perf_counter() - t
    st = _H.modelStatusToString(_H.getModelStatus())
    obj = _H.getInfo().objective_function_value
    _H.changeColBounds(j, old_lo, old_up)
    return sec, st, obj


def highs_root(lp):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", "simplex")
    h.passModel(to_highs_lp(lp))
    t = time.perf_counter()
    h.run()
    sec = time.perf_counter() - t
    sol, basis = h.getSolution(), h.getBasis()
    return (np.array(sol.col_value), h.getInfo().objective_function_value, sec,
            [int(s) for s in basis.col_status], [int(s) for s in basis.row_status])


def highs_cold(lp, solver):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", solver)
    h.setOptionValue("time_limit", 120.0)      # cold solves are only for reference
    h.passModel(to_highs_lp(lp))
    t = time.perf_counter()
    h.run()
    return time.perf_counter() - t


# ---- branching -----------------------------------------------------------------------------
def candidates(lp, x, max_candidates):
    frac = np.abs(x - np.round(x))
    idx = np.flatnonzero(lp.integer & (frac > 1e-6)) if lp.integer is not None else np.array([], int)
    idx = idx[np.argsort(-frac[idx], kind="stable")][:max_candidates]      # most fractional first
    return idx


def child_bounds(lp, x, idx):
    """2 LPs per candidate, ordered [down_0, up_0, down_1, up_1, ...].

    Returns functions ids -> bound rows (built on demand: 2,000 full copies of a 240k-variable
    bound vector would be several GB) plus the same LPs as (column, new bounds) tasks for HiGHS.
    """
    var = np.repeat(idx, 2)
    new_lo = np.where(np.arange(len(var)) % 2 == 1, np.ceil(x[var]), lp.lower[var])
    new_up = np.where(np.arange(len(var)) % 2 == 0, np.floor(x[var]), lp.upper[var])

    def rows(base, new):
        def get(ids):
            ids = np.asarray(ids)
            out = np.repeat(base[None, :], len(ids), 0)
            out[np.arange(len(ids)), var[ids]] = new[ids]
            return out
        return get

    tasks = [(int(j), float(a), float(b), float(lp.lower[j]), float(lp.upper[j]))
             for j, a, b in zip(var, new_lo, new_up)]
    return rows(lp.lower, new_lo), rows(lp.upper, new_up), tasks


def best_variable(idx, root_obj, objs, infeasible):
    """Product score (as in SCIP): max(down gain, eps) * max(up gain, eps). Infeasible = big gain."""
    big = 1e6 * (1 + abs(root_obj))
    gains = np.where(infeasible, big, np.maximum(np.asarray(objs) - root_obj, 0.0))
    down, up = gains[0::2], gains[1::2]
    score = np.maximum(down, 1e-6) * np.maximum(up, 1e-6)
    return int(idx[int(np.argmax(score))]), score


def load_instance(name):
    if name.endswith(".mps.gz"):
        return load_mps(str(ROOT / "data" / "miplib" / name))
    fam, _, _ = name.partition("_")
    for kw in PAPER_SIZES[fam]:
        lp = GENERATORS[fam](**kw)
        if lp.name.startswith(name):
            return lp
    raise ValueError(name)


def default_instances(quick, everything):
    names = []
    for fam, sizes in PAPER_SIZES.items():
        for kw in (sizes if everything else sizes[:1]):
            names.append(GENERATORS[fam](**kw).name.rsplit("_s", 1)[0])
    if not quick:
        names.append("chromaticindex512-7.mps.gz")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--all", action="store_true", help="include the larger instance of each family")
    ap.add_argument("--instances", nargs="*")
    ap.add_argument("--max-candidates", type=int, default=1000,
                    help="fractional variables to branch on (2 LPs each); the paper used all of them")
    ap.add_argument("--slots", type=int, default=128)
    ap.add_argument("--chip-slots", type=int, default=None,
                    help="batch size for the Trainium methods (default: --slots); the chip wants bigger batches than CPUs")
    ap.add_argument("--methods", nargs="+", default=None, choices=list(METHODS),
                    help="trnopt variants to run (default: cpu-float32, plus mps-metal on a Mac)")
    ap.add_argument("--host", default=None, help="label for the machine in the CSV name, e.g. trn1.2xlarge")
    ap.add_argument("--highs-cache", action="store_true",
                    help="reuse HiGHS warm results from an earlier run on this host (results/cache/), or save them")
    ap.add_argument("--cuopt-budget", type=float, default=600.0,
                    help="seconds per instance for cuopt-pdlp; past it, the total is extrapolated from the LPs done")
    args = ap.parse_args()
    workers = os.cpu_count() or 1
    gpu = torch.backends.mps.is_available()
    host = (subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            if platform.system() == "Darwin" else platform.node()) or "host"
    host = args.host or host
    print(f"host: {host}   CPU workers for HiGHS: {workers}   Mac GPU: {gpu}   torch {torch.__version__}")
    rows = []
    methods = [(METHODS[m][0], m) for m in (args.methods or (["cpu-float32"] + (["mps-metal"] if gpu else [])))]
    picked = [m for _, m in methods]
    if "neuron-2core" in picked and any(m.startswith("neuron") and m != "neuron-2core" for m in picked):
        # A process that has used the chip holds its NeuronCores until it exits, so the two worker
        # processes couldn't get theirs. Run neuron-2core in its own invocation (--highs-cache helps).
        ap.error("run --methods neuron-2core in a separate invocation from the other neuron methods")

    for name in args.instances or default_instances(args.quick, args.all):
        lp = load_instance(name)
        x_root, root_obj, root_sec, bcols, brows = highs_root(lp)
        idx = candidates(lp, x_root, args.max_candidates)
        if len(idx) == 0:
            print(f"\n{lp.name}: root LP is already integral; skipped")
            continue
        L, U, tasks = child_bounds(lp, x_root, idx)
        N = len(tasks)
        print(f"\n{lp.name}: {lp.m} x {lp.n}, {lp.K.nnz} nonzeros; root LP {root_sec:.2f}s (HiGHS simplex); "
              f"{len(idx)} candidates -> {N} LPs", flush=True)

        # HiGHS warm, all LPs, all cores.
        # keyed on the machine's real hostname, not the --host label, so differently labelled runs share it
        cache = ROOT / "results" / "cache" / f"highs_warm_{platform.node()}_{workers}w_{lp.name}_{len(idx)}.pkl"
        if args.highs_cache and cache.exists():
            import pickle
            with open(cache, "rb") as f:
                out, warm_wall = pickle.load(f)
            print(f"   (HiGHS warm results reused from {cache.name})")
        else:
            t = time.perf_counter()
            # spawn, not fork: a forked worker would inherit this process's accelerator runtime and HiGHS state.
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_worker_init, initargs=(lp, bcols, brows)) as ex:
                out = list(ex.map(_warm_solve, tasks, chunksize=max(1, N // (4 * workers))))
            warm_wall = time.perf_counter() - t
            if args.highs_cache:
                import pickle
                cache.parent.mkdir(parents=True, exist_ok=True)
                with open(cache, "wb") as f:
                    pickle.dump((out, warm_wall), f)
        ref_obj = np.array([o for _, _, o in out])
        ref_inf = np.array(["nfeasible" in s for _, s, _ in out])
        ref_best, ref_score = best_variable(idx, root_obj, ref_obj, ref_inf)
        n_ties = int(np.sum(ref_score >= ref_score.max() * (1 - 1e-6)))
        warm_one_core = sum(s for s, _, _ in out)
        print(f"   HiGHS warm dual simplex: {warm_wall:8.2f} s on {workers} cores "
              f"({1e3 * warm_one_core / N:.1f} ms/LP on one core); {ref_inf.sum()} infeasible children; "
              f"{n_ties} candidate(s) tied for the best exact score")
        rows.append(dict(instance=lp.name, m=lp.m, n=lp.n, nnz=lp.K.nnz, lps=N, method="HiGHS warm (all cores)",
                         seconds=warm_wall, sec_per_lp=warm_wall / N, pick_quality=1.0, ties=n_ties))

        # HiGHS cold, sampled, best method.
        sample = list(range(0, N, max(1, N // 2)))[:2]
        import dataclasses
        cold = {s: np.mean([highs_cold(dataclasses.replace(lp, lower=L([i])[0], upper=U([i])[0]), s) for i in sample])
                for s in ("simplex", "ipm", "pdlp")}
        best_cold = min(cold, key=cold.get)
        print(f"   HiGHS cold (one core, best = {best_cold}): {cold[best_cold] * N:8.2f} s est. "
              f"({1e3 * cold[best_cold]:.1f} ms/LP)")
        rows.append(dict(instance=lp.name, m=lp.m, n=lp.n, nnz=lp.K.nnz, lps=N, method=f"HiGHS cold {best_cold} (1 core, est.)",
                         seconds=cold[best_cold] * N, sec_per_lp=cold[best_cold]))

        # trnopt, warm-started from its own root solution.
        root = solve_pdlp(lp, PDLPOptions(tol=TOL, backend=Backend.preset("cpu-float32"), time_limit=300))
        for label, preset in methods:
            B = min(args.chip_slots if (args.chip_slots and preset.startswith(("neuron", "cuda"))) else args.slots, N)
            if preset.startswith("cuopt"):
                from cuopt_adapter import solve_lp as _solve_lp, to_data_model
                prec = 0 if preset.endswith("f32") else None
                use_warm = preset.endswith("warm")
                solve_lp = lambda dm, **kw: _solve_lp(dm, precision=prec, **kw)
                dm = to_data_model(lp)
                rr = solve_lp(dm, tol=TOL, want_warm=use_warm)             # cuOpt's own root (also a warm-up)
                warm = rr.warm if use_warm else None
                order = np.random.default_rng(0).permutation(N)            # random order, so a partial run is a fair sample
                done, status, objective = [], [], []
                t = time.perf_counter()
                for i in order:
                    dm.set_variable_lower_bounds(np.asarray(L([i])[0], dtype=np.float64))
                    dm.set_variable_upper_bounds(np.asarray(U([i])[0], dtype=np.float64))
                    try:
                        ri = solve_lp(dm, tol=TOL, warm=warm)
                    except Exception as e:                                   # warm start not accepted: go cold
                        if warm is None:
                            raise
                        print(f"   cuOpt warm start failed ({type(e).__name__}: {str(e)[:120]}); solving cold")
                        warm = None
                        ri = solve_lp(dm, tol=TOL)
                    done.append(int(i)); status.append(ri.status); objective.append(ri.objective)
                    if time.perf_counter() - t > args.cuopt_budget:
                        break
                k = len(done)
                wall = (time.perf_counter() - t) * N / k
                done = np.array(done)
                st = np.array(status); ob = np.array(objective)
                ok = np.isin(st, ["optimal", "infeasible"])
                feas = ~ref_inf[done] & (st == "optimal")
                err = np.abs(ob[feas] - ref_obj[done][feas]) / (1 + np.abs(ref_obj[done][feas]))
                tag = f"{'warm' if use_warm else 'cold'}" + ("" if k == N else f", est. from {k}/{N} LPs")
                print(f"   {label} ({tag}): {wall:8.2f} s  ({ok.sum()}/{k} finished, "
                      f"worst obj err {err.max() if err.size else float('nan'):.0e}, "
                      f"infeasible agree {np.sum((st == 'infeasible') == ref_inf[done])}/{k})   vs HiGHS warm: {warm_wall / wall:.2f}x",
                      flush=True)
                rows.append(dict(instance=lp.name, m=lp.m, n=lp.n, nnz=lp.K.nnz, lps=N, method=f"{label} ({tag})",
                                 seconds=wall, sec_per_lp=wall / N, finished=int(ok.sum()), sampled=k,
                                 worst_obj_err=float(err.max()) if err.size else None,
                                 median_obj_err=float(np.median(err)) if err.size else None,
                                 infeasible_agree=int(np.sum((st == "infeasible") == ref_inf[done]))))
                continue
            if preset == "neuron-2core":
                from trn_two_cores import split_solve
                ids = np.arange(N)
                wall, status, objective, _ = split_solve(lp, L(ids).astype(np.float32), U(ids).astype(np.float32),
                                                         x0=root.x, y0=root.y, cores=2, slots=min(B, -(-N // 2)), tol=TOL)
                r = argparse.Namespace(status=status, objective=objective)
            else:
                opts = PDLPOptions(tol=TOL, backend=METHODS[preset][1](), time_limit=3600)
                solve_pdlp_stream(lp, B, B, lower=L, upper=U, x0=root.x, y0=root.y,
                                  opts=PDLPOptions(tol=TOL, backend=opts.backend, max_iters=64))   # warm-up
                t = time.perf_counter()
                r = solve_pdlp_stream(lp, N, B, lower=L, upper=U, x0=root.x, y0=root.y, opts=opts)
                wall = time.perf_counter() - t
            ok = np.array([s in ("optimal", "infeasible") for s in r.status])
            ours_inf = np.array([s == "infeasible" for s in r.status])
            feas = ~ref_inf & (np.array(r.status) == "optimal")
            err = np.abs(np.asarray(r.objective)[feas] - ref_obj[feas]) / (1 + np.abs(ref_obj[feas]))
            our_best, _ = best_variable(idx, root_obj, np.where(ours_inf, 0, r.objective), ours_inf)
            # How good is our pick, judged by EXACT scores? 1.0 = as good as the best (ties count).
            pick_quality = float(ref_score[list(idx).index(our_best)] / ref_score.max())
            print(f"   {label:12s} B={B}: {wall:8.2f} s  ({ok.sum()}/{N} finished, "
                  f"worst obj err {err.max() if err.size else float('nan'):.0e}, median {np.median(err) if err.size else float('nan'):.0e}; "
                  f"infeasible agree {np.sum(ours_inf == ref_inf)}/{N}; our pick's exact score = {pick_quality:.3f} of best)"
                  f"   vs HiGHS warm: {warm_wall / wall:.2f}x", flush=True)
            rows.append(dict(instance=lp.name, m=lp.m, n=lp.n, nnz=lp.K.nnz, lps=N, method=f"{label} B={B}",
                             seconds=wall, sec_per_lp=wall / N, finished=int(ok.sum()),
                             worst_obj_err=float(err.max()) if err.size else None,
                             median_obj_err=float(np.median(err)) if err.size else None,
                             infeasible_agree=int(np.sum(ours_inf == ref_inf)), pick_quality=pick_quality, ties=n_ties))

    out = ROOT / "results" / f"strong_branching_{host.replace(' ', '_')}_{dt.date.today()}.csv"
    out.parent.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
