"""Split one strong-branching batch across both NeuronCores of a trn1.2xlarge.

    python examples/trn_two_cores.py --instance nu25 --batch 512 --cores 2

Each core gets its own process (NEURON_RT_VISIBLE_CORES pins it), compiles and warms up, then
waits for a shared start signal, so the timed part excludes compilation. Reports wall time from
the start signal until the last core finishes, as seconds per LP over the whole batch.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def worker(args) -> None:
    import torch
    from trnopt import Backend, PDLPOptions, load_mps
    from trnopt.batch import solve_pdlp_batch
    d = np.load(args.data)
    lp = load_mps(str(args.mps))
    L, U = d["L"][args.lo:args.hi], d["U"][args.lo:args.hi]
    be = Backend.preset("neuron", spmv=args.spmv, dense=args.spmv != "nki")
    torch._dynamo.config.recompile_limit = 1024
    solve_pdlp_batch(lp, len(L), lower=L, upper=U, opts=PDLPOptions(tol=1e-4, backend=be, max_iters=128))
    Path(args.ready).touch()
    while not Path(args.go).exists():
        time.sleep(0.01)
    r = solve_pdlp_batch(lp, len(L), lower=L, upper=U, opts=PDLPOptions(tol=1e-4, backend=be, time_limit=600))
    end = time.time()
    print(json.dumps(dict(end=end, n=len(L), n_opt=r.status.count("optimal"))), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=512)
    ap.add_argument("--spmv", default="torch", choices=["torch", "nki"], help="torch = dense K, nki = sparse NKI kernel")
    # internal (worker mode)
    for a in ("--worker-data", "--worker-ready", "--worker-go", "--worker-mps"):
        ap.add_argument(a)
    ap.add_argument("--worker-lo", type=int)
    ap.add_argument("--worker-hi", type=int)
    args = ap.parse_args()
    if args.worker_data:
        worker(argparse.Namespace(data=args.worker_data, ready=args.worker_ready, go=args.worker_go,
                                  mps=args.worker_mps, lo=args.worker_lo, hi=args.worker_hi, spmv=args.spmv))
        return

    from trnopt import PDLPOptions, load_mps, solve_pdlp
    from trnopt.problems import branching_variants
    mps = next((ROOT / "data" / "miplib").glob(f"*{args.instance}*.mps.gz"))
    lp = load_mps(str(mps))
    root = solve_pdlp(lp, PDLPOptions(tol=1e-4))
    L, U, labels = branching_variants(lp, root.x, args.candidates, tol=1e-3)
    distinct = min(args.batch, len(labels))
    L, U = np.resize(L, (args.batch, lp.n)), np.resize(U, (args.batch, lp.n))
    tmp = Path(tempfile.mkdtemp())
    np.savez(tmp / "data.npz", L=L, U=U)
    go = tmp / "go"
    bounds = np.linspace(0, args.batch, args.cores + 1).astype(int)
    procs = []
    for c in range(args.cores):
        env = dict(os.environ, NEURON_RT_VISIBLE_CORES=str(c))
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--instance", args.instance, "--worker-data", str(tmp / "data.npz"),
             "--worker-ready", str(tmp / f"ready{c}"), "--worker-go", str(go), "--worker-mps", str(mps),
             "--worker-lo", str(bounds[c]), "--worker-hi", str(bounds[c + 1]), "--spmv", args.spmv],
            env=env, stdout=subprocess.PIPE, text=True))
    while not all((tmp / f"ready{c}").exists() for c in range(args.cores)):
        if any(p.poll() not in (None, 0) for p in procs):
            sys.exit("a worker failed during warm-up")
        time.sleep(0.1)
    start = time.time()
    go.touch()
    outs = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
    wall = max(o["end"] for o in outs) - start
    n_opt = sum(o["n_opt"] for o in outs)
    print(f"{lp.name} spmv={args.spmv} B={args.batch} cores={args.cores} distinct={distinct}: {wall:.2f} s wall, "
          f"{wall / args.batch:.5f} s/LP, {n_opt}/{args.batch} optimal", flush=True)


if __name__ == "__main__":
    main()
