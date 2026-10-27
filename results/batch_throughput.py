"""Batch throughput: seconds per LP vs batch size, for strong-branching batches on MIPLIB instances.

    python results/batch_throughput.py path/to/miplib    # folder holding the .mps.gz files
"""
import csv, sys, time
import numpy as np
MIPLIB_DIR = sys.argv[1] if len(sys.argv) > 1 else "miplib"
from trnopt import Backend, PDLPOptions, load_mps, solve_pdlp, solve_highs
from trnopt.batch import solve_pdlp_batch
from trnopt.problems import branching_variants
import dataclasses

INST = ["neos-911970", "neos-4954672-berkel", "neos-3627168-kasai", "nu25-pr12"]
BACKENDS = ["cpu", "cpu-float32", "emulate-neuron"]
SIZES = [1, 8, 32, 128]
out = csv.writer(open("batch_throughput.csv", "w", newline=""))
out.writerow(["instance", "backend", "B", "seconds", "sec_per_lp", "total_iterations", "mean_lp_iterations",
              "useful_work", "n_optimal", "n_infeasible", "other", "max_obj_err_vs_highs_checked"])
for name in INST:
    lp = load_mps(f"{MIPLIB_DIR}/{name}.mps.gz")
    root = solve_pdlp(lp, PDLPOptions(tol=1e-4))
    L, U, labels = branching_variants(lp, root.x, 64, tol=1e-3)
    # HiGHS reference for the first 8 variants (spot check of correctness)
    ref = [solve_highs(dataclasses.replace(lp, lower=L[i], upper=U[i])) for i in range(8)]
    for bname in BACKENDS:
        be = Backend.preset(bname, compile=False)
        for B in SIZES:
            if B > len(labels):
                continue
            opts = PDLPOptions(tol=1e-4, backend=be, time_limit=300)
            solve_pdlp_batch(lp, B, lower=L[:B], upper=U[:B], opts=dataclasses.replace(opts, max_iters=128))  # warm-up
            t = time.perf_counter()
            r = solve_pdlp_batch(lp, B, lower=L[:B], upper=U[:B], opts=opts)
            dt = time.perf_counter() - t
            errs = [abs(r.objective[i] - ref[i].objective) / (1 + abs(ref[i].objective))
                    for i in range(min(B, 8)) if ref[i].status == "Optimal" and r.status[i] == "optimal"]
            st = r.status
            row = [name, bname, B, round(dt, 3), round(dt / B, 4), r.total_iterations, int(np.mean(r.iterations)),
                   round(r.useful_work, 3), st.count("optimal"), st.count("infeasible"),
                   len(st) - st.count("optimal") - st.count("infeasible"), f"{max(errs) if errs else float('nan'):.1e}"]
            out.writerow(row)
            print(*row, flush=True)
