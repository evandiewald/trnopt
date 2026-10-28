"""Profile one call of the NKI sliced-ELL kernel (nki_sell.py) on a NeuronCore with neuron-explorer.

    NEURON_RT_VISIBLE_CORES=1 python examples/profile_nki_kernel.py chromaticindex512-7 --B 32 --tag v2

Builds the kernel for K (or K^T, --transpose) of a MIPLIB instance at batch width B, compiles it standalone
(NumPy inputs, so the NEFF holds just this kernel), checks its product against SciPy, then:

  neuron-explorer capture   -> <tag>.ntff (with DGE notifications, so the software-generated gathers show up)
  neuron-explorer view      -> <tag>_summary.txt and <tag>.pftrace (open in https://ui.perfetto.dev)

Writes to results/profiles/<instance>_<K|KT>_B<B>_<tag>.*. To profile another version of the kernel, run this
script with PYTHONPATH pointing at that version's src/.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# nki reads this when it compiles; it keeps the compile's artifacts (the NEFF among them) instead of a temp dir.
WORK = os.environ.setdefault("NKI_ARTIFACTS_DIR", tempfile.mkdtemp(prefix="nki_profile_"))

from trnopt import load_mps                                     # noqa: E402
from trnopt.nki_sell import kernel_tables, make_sell_kernel, output_order     # noqa: E402
from trnopt.sell import build_sell                              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXPLORER = shutil.which("neuron-explorer") or "/opt/aws/neuron/bin/neuron-explorer"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", help="MIPLIB file name prefix in data/miplib/")
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--transpose", action="store_true", help="profile K^T instead of K")
    ap.add_argument("--ordered", action="store_true", help="the ordered kernel (block writes, MatVec permute=True)")
    ap.add_argument("--tag", default="current")
    ap.add_argument("--out", default=str(ROOT / "results" / "profiles"))
    ap.add_argument("--check", action="store_true", help=argparse.SUPPRESS)   # internal: see below
    args = ap.parse_args()

    lp = load_mps(str(next((ROOT / "data" / "miplib").glob(f"{args.instance}*.mps*"))))
    M = (lp.K.T if args.transpose else lp.K).tocsr()
    S = build_sell(M)
    tables = kernel_tables(S, ordered=args.ordered)
    pos = output_order(S)[0] if args.ordered else np.arange(M.shape[0])     # output row of each row of M
    X = np.random.default_rng(0).normal(size=(M.shape[1], args.B)).astype(np.float32)

    name = f"{args.instance}_{'KT' if args.transpose else 'K'}_B{args.B}_{args.tag}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kernel = make_sell_kernel(S, args.B, args.ordered)
    if args.check:
        # Child process: the product through TorchNeuron. A process that touches the chip holds its
        # NeuronCore until it exits, so this can't run in the parent, which still needs to capture.
        import torch
        import torch_neuronx  # noqa: F401  (the "neuron" device)
        Y = kernel(*[torch.as_tensor(a).to("neuron") for a in (*tables, X)]).cpu().numpy()
        err = np.abs(Y[pos] - M @ X).max() / np.abs(M @ X).max()
        print(f"{name}: {M.shape[0]} x {M.shape[1]}, {M.nnz} nnz, {S.num_slices} slices, "
              f"total width {sum(S.widths)}; relative error {err:.1e}", flush=True)
        sys.exit(0 if err < 1e-5 else "kernel output is wrong")
    try:
        kernel(*tables, X)                       # NumPy inputs: compile standalone (the NEFF we want), then run
    except ImportError:
        pass                                     # no standalone NKI runtime (nrtpy) here; the NEFF is written
    subprocess.run([sys.executable, *sys.argv, "--check"], check=True)
    neff = out / f"{name}.neff"
    shutil.copy(next(Path(WORK).rglob("kernel.neff")), neff)

    ifmaps = []
    for arg, a in zip(("cols", "vals", "out_rows", "X"), (*tables, X)):
        np.save(Path(WORK) / f"{arg}.npy", np.ascontiguousarray(a))
        ifmaps += [arg, str(Path(WORK) / f"{arg}.npy")]
    ntff = out / f"{name}.ntff"
    run = lambda *a: subprocess.run([EXPLORER, *a], check=True)
    run("capture", "-n", str(neff), "-s", str(ntff), "--enable-dge-notifs", *ifmaps)
    view = ["view", "-n", str(neff), "-s", str(ntff)]
    with open(out / f"{name}_summary.txt", "w") as f:
        subprocess.run([EXPLORER, *view, "--output-format", "summary-text"], check=True, stdout=f)
    run(*view, "--output-format", "perfetto", "--output-file", str(out / f"{name}.pftrace"))
    print(f"Wrote {out / name}.*", flush=True)


if __name__ == "__main__":
    main()
