"""Split one batch of LPs across both NeuronCores of a trn1.2xlarge.

    python examples/trn_two_cores.py --instance nu25 --batch 512 --cores 2            # strong-branching LPs
    python examples/trn_two_cores.py --instance rahue --batch 1024 --cores 2 --spmv nki

Each core gets its own process (NEURON_RT_VISIBLE_CORES pins it), compiles and warms up, then waits
for a shared start signal, so the timed part excludes compilation. Reports wall time from the start
signal until the last core finishes. `split_solve` is also used by strong_branching_benchmark.py.
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _backend(spmv: str):
    from trnopt import Backend
    if spmv == "nki":
        return Backend.preset("neuron", spmv="nki", dense=False)
    if spmv == "torch":
        return Backend.preset("neuron", spmv="torch")
    return Backend.preset("neuron")                       # "auto"


def _worker(job: str, lo: int, hi: int, ready: str, go: str, out: str) -> None:
    import torch
    from trnopt import PDLPOptions
    from trnopt.batch import solve_pdlp_stream
    with open(job, "rb") as f:
        J = pickle.load(f)
    lp, L, U, x0, y0 = J["lp"], J["L"][lo:hi], J["U"][lo:hi], J["x0"], J["y0"]
    be = _backend(J["spmv"])
    torch._dynamo.config.recompile_limit = 1024
    N, B = len(L), min(J["slots"], len(L))
    solve_pdlp_stream(lp, B, B, lower=L[:B], upper=U[:B], x0=x0, y0=y0,
                      opts=PDLPOptions(tol=J["tol"], backend=be, max_iters=64))             # warm-up / compile
    Path(ready).touch()
    while not Path(go).exists():
        time.sleep(0.005)
    r = solve_pdlp_stream(lp, N, B, lower=L, upper=U, x0=x0, y0=y0,
                          opts=PDLPOptions(tol=J["tol"], backend=be, time_limit=J["time_limit"]))
    end = time.time()
    np.savez(out, end=end, status=np.array(r.status, dtype=object), objective=np.asarray(r.objective),
             iterations=np.asarray(r.iterations), allow_pickle=True)


def split_solve(lp, L, U, x0=None, y0=None, cores: int = 2, slots: int = 1024, tol: float = 1e-4,
                spmv: str = "auto", time_limit: float = 3600.0):
    """Solve the LPs (rows of L, U) split evenly across `cores` NeuronCores.

    Returns (wall seconds from the start signal to the last finish, status list, objectives, iterations).
    """
    L, U = np.asarray(L), np.asarray(U)
    N = len(L)
    tmp = Path(tempfile.mkdtemp())
    job = tmp / "job.pkl"
    with open(job, "wb") as f:
        pickle.dump(dict(lp=lp, L=L, U=U, x0=x0, y0=y0, slots=slots, tol=tol, spmv=spmv, time_limit=time_limit), f)
    go = tmp / "go"
    bounds = np.linspace(0, N, cores + 1).astype(int)
    procs = []
    for c in range(cores):
        env = dict(os.environ, NEURON_RT_VISIBLE_CORES=str(c))
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--worker", str(job), str(bounds[c]), str(bounds[c + 1]),
             str(tmp / f"ready{c}"), str(go), str(tmp / f"out{c}.npz")], env=env))
    while not all((tmp / f"ready{c}").exists() for c in range(cores)):
        if any(p.poll() not in (None, 0) for p in procs):
            for p in procs:
                p.kill()
            raise RuntimeError("a NeuronCore worker failed during warm-up")
        time.sleep(0.05)
    start = time.time()
    go.touch()
    for p in procs:
        if p.wait() != 0:
            raise RuntimeError("a NeuronCore worker failed")
    outs = [np.load(tmp / f"out{c}.npz", allow_pickle=True) for c in range(cores)]
    wall = max(float(o["end"]) for o in outs) - start
    status = [s for o in outs for s in o["status"].tolist()]
    return (wall, status, np.concatenate([o["objective"] for o in outs]),
            np.concatenate([o["iterations"] for o in outs]))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=512)
    ap.add_argument("--spmv", default="auto", choices=["auto", "torch", "nki"],
                    help="auto = dense if K is small enough, else NKI; torch = dense; nki = sparse kernel")
    args = ap.parse_args()

    from trnopt import PDLPOptions, load_mps, solve_pdlp
    from trnopt.problems import branching_variants
    mps = next((ROOT / "data" / "miplib").glob(f"*{args.instance}*.mps.gz"))
    lp = load_mps(str(mps))
    root = solve_pdlp(lp, PDLPOptions(tol=1e-4))
    L, U, labels = branching_variants(lp, root.x, args.candidates, tol=1e-3)
    distinct = min(args.batch, len(labels))
    L, U = np.resize(L, (args.batch, lp.n)), np.resize(U, (args.batch, lp.n))
    wall, status, _, _ = split_solve(lp, L, U, cores=args.cores, slots=args.batch, spmv=args.spmv)
    n_opt = status.count("optimal")
    print(f"{lp.name} spmv={args.spmv} B={args.batch} cores={args.cores} distinct={distinct}: {wall:.2f} s wall, "
          f"{wall / args.batch:.5f} s/LP, {n_opt}/{args.batch} optimal", flush=True)


if __name__ == "__main__":
    main()
