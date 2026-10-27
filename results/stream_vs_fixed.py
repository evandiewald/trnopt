"""Continuous batching vs fixed batches: 128 strong-branching LPs through B slots.

    python results/stream_vs_fixed.py path/to/miplib    # folder holding the .mps.gz files

"fixed": split the 128 LPs into chunks of B and solve each chunk as one batch, one after another
         (each chunk waits for its slowest LP).
"stream": one queue of 128 LPs through B slots, refilling each slot as its LP finishes.
"""
import csv, sys, time
import numpy as np
from trnopt import Backend, PDLPOptions, load_mps, solve_pdlp
from trnopt.batch import solve_pdlp_batch, solve_pdlp_stream
from trnopt.problems import branching_variants

MIPLIB_DIR = sys.argv[1] if len(sys.argv) > 1 else "miplib"
INST = ["neos-911970", "neos-4954672-berkel", "nu25-pr12"]
BACKENDS = ["cpu-float32", "emulate-neuron"]
SLOTS = [8, 32]
out = csv.writer(open("stream_vs_fixed.csv", "w", newline=""))
out.writerow(["instance", "backend", "slots", "mode", "n_lps", "seconds", "sec_per_lp", "batch_iterations",
              "useful_work", "n_optimal", "n_infeasible", "other"])
for name in INST:
    lp = load_mps(f"{MIPLIB_DIR}/{name}.mps.gz")
    root = solve_pdlp(lp, PDLPOptions(tol=1e-4))
    L, U, labels = branching_variants(lp, root.x, 64, tol=1e-3)
    N = len(labels)
    for bname in BACKENDS:
        opts = PDLPOptions(tol=1e-4, backend=Backend.preset(bname, compile=False), time_limit=600)
        for B in SLOTS:
            # warm-up (one short batch) so neither mode pays first-call costs
            solve_pdlp_batch(lp, B, lower=L[:B], upper=U[:B], opts=PDLPOptions(**{**opts.__dict__, "max_iters": 128}))
            t = time.perf_counter()
            statuses, iters, total_it, work = [], 0, 0, 0
            for s in range(0, N, B):
                r = solve_pdlp_batch(lp, min(B, N - s), lower=L[s:s + B], upper=U[s:s + B], opts=opts)
                statuses += r.status
                total_it += r.total_iterations
                work += int(r.iterations.sum())
            dt = time.perf_counter() - t
            rows = [("fixed", dt, total_it, work / (B * total_it), statuses)]
            t = time.perf_counter()
            r = solve_pdlp_stream(lp, N, B, lower=L, upper=U, opts=opts)
            rows.append(("stream", time.perf_counter() - t, r.total_iterations, r.useful_work, r.status))
            for mode, sec, bit, uw, st in rows:
                row = [name, bname, B, mode, N, round(sec, 3), round(sec / N, 4), bit, round(uw, 3),
                       st.count("optimal"), st.count("infeasible"), N - st.count("optimal") - st.count("infeasible")]
                out.writerow(row)
                print(*row, flush=True)
