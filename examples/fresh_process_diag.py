"""Where does a MIP solve's time go, first solve in a process vs. later ones? (diagnosis for Trainium)

    python examples/fresh_process_diag.py data/miplib/neos-933966.mps.gz --repeat 2 --time-limit 120

Solves the instance --repeat times in this process with the Pyomo plugin's defaults and prints, per solve:
MIPResult.times / counts / incumbents; every batched PDLP call (who called it, LPs, slots, iterations,
seconds); the first call of each compiled function per input shape (Dynamo trace + Neuron compile or cache
load + NEFF load); NKI kernel builds; TorchNeuron's compilation-cache counters; and which on-disk caches
gained files. Monkeypatches only, nothing in trnopt changes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import trnopt  # noqa: F401  first: compile cache locations (compile_cache.py)

import numpy as np
import torch

from trnopt import backend as tb
from trnopt import batch as tbatch
from trnopt import compile_cache, load_mps
from trnopt.mip import solve_mip

T_START = time.perf_counter()
EV: list = []                      # (t, kind, details) for the current solve


def now():
    return time.perf_counter() - T_START


def cache_dirs():
    root = compile_cache.cache_dir()
    d = {k: Path(os.environ.get(v, "")) for k, v in (("neff", "TORCH_NEURONX_NEFF_CACHE_DIR"),
                                                     ("hlo", "TORCH_NEURONX_HLO_CACHE_DIR"),
                                                     ("inductor", "TORCHINDUCTOR_CACHE_DIR"))}
    d["nki-trace"] = Path(f"/var/tmp/nki-trace-cache-uid{os.getuid()}")
    d["/tmp/neff_cache"] = Path("/tmp/neff_cache")
    d["/tmp/hlo_cache"] = Path("/tmp/hlo_cache")
    return root, d


def snapshot():
    out = {}
    for k, p in cache_dirs()[1].items():
        files = {}
        if p.is_dir():
            for f in p.rglob("*"):
                try:
                    if f.is_file():
                        files[str(f)] = f.stat().st_mtime
                except OSError:
                    pass
        out[k] = files
    return out


def diff(a, b):
    return {k: (len(b[k]), len(set(b[k]) - set(a[k])), sum(1 for f in b[k] if f in a[k] and b[k][f] != a[k][f]))
            for k in b}


def caller():
    for fr in reversed(traceback.extract_stack(limit=30)):
        if fr.name in ("warm_up", "strong_scores", "feasibility_pump", "solve_pdlp"):
            return {"strong_scores": "strong", "feasibility_pump": "pump", "warm_up": "warmup"}.get(fr.name, fr.name)
    return "?"


# ---- instrumentation --------------------------------------------------------------------------
_orig_compile_fn = tb.Backend.compile_fn
_seen_shapes: set = set()


def compile_fn(self, f):
    g = _orig_compile_fn(self, f)
    if g is f:
        return f

    def timed(*args, **kw):
        key = (f.__name__,) + tuple(tuple(a.shape) for a in args if isinstance(a, torch.Tensor))
        if key in _seen_shapes:
            return g(*args, **kw)
        t = time.perf_counter()
        out = g(*args, **kw)
        if isinstance(out, torch.Tensor):
            out.cpu()
        elif isinstance(out, tuple):
            out[0].cpu()
        dt = time.perf_counter() - t
        _seen_shapes.add(key)
        EV.append((now(), "first-call", f"{f.__name__} x{key[1] if len(key) > 1 else ''}", dt))
        return out
    return timed


tb.Backend.compile_fn = compile_fn

_orig_stream = tbatch._stream


def stream(lp, N, B, g, opts, stop_on_stall=False, omega0=None):
    t = time.perf_counter()
    r = _orig_stream(lp, N, B, g, opts, stop_on_stall, omega0)
    dt = time.perf_counter() - t
    tl = sum(s == "time_limit" for s in r.status)
    EV.append((now(), "stream", f"{caller():6s} {opts.backend.device:6s} N={N:5d} B={B:5d} n={lp.n} m={lp.m} "
               f"it={r.total_iterations:5d} tl={opts.time_limit:6.1f} timed_out={tl}", dt))
    return r


tbatch._stream = stream

try:
    from trnopt import nki_op
    _orig_msk = nki_op.make_sell_kernel

    def msk(S, B, ordered=False):
        t = time.perf_counter()
        k = _orig_msk(S, B, ordered)
        EV.append((now(), "nki-kernel", f"make_sell_kernel {S.shape} B={B}", time.perf_counter() - t))
        return k
    nki_op.make_sell_kernel = msk
except Exception:
    pass

_orig_setup = tbatch._shared_setup


def shared_setup(K, ruiz, be):
    t = time.perf_counter()
    hit = tbatch._SETUP_CACHE.get((id(K), ruiz, be))
    out = _orig_setup(K, ruiz, be)
    if hit is None:
        EV.append((now(), "setup", f"scaling+MatVec {K.shape} spmv={out[4].spmv}", time.perf_counter() - t))
    return out


tbatch._shared_setup = shared_setup


def neuron_metrics():
    try:
        import torch_neuronx.metrics as met
        return {k: v for k, v in met.get_metrics_dict().items()
                if any(s in k for s in ("Cache", "Compil", "Neff", "NEFF", "Load"))}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mps")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--backend", default="neuron")
    args = ap.parse_args()
    root, dirs = cache_dirs()
    print(f"cache root {root}")
    for k in ("TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_HLO_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
              "NEURON_COMPILE_CACHE_URL", "NEURON_RT_VISIBLE_CORES", "TORCH_NEURONX_METRICS_ENABLED"):
        print(f"  {k}={os.environ.get(k)}")
    lp = load_mps(args.mps)
    print(f"{lp.name}: {lp.m} x {lp.n}, nnz {lp.K.nnz}, {int(lp.integer.sum())} integer   (loaded at {now():.1f}s)")
    from trnopt.pyomo_plugin import DEFAULTS as D
    for rep in range(args.repeat):
        EV.clear()
        snap0 = snapshot()
        m0 = neuron_metrics()
        dyn0 = dict(torch._dynamo.utils.counters["stats"])
        t = time.perf_counter()
        t_solve0 = now()
        r = solve_mip(lp, D["branching"], time_limit=args.time_limit, gap_tol=D["mip_gap"], backend=args.backend,
                      slots=1024, strong_iters=D["strong_iters"], presolve=D["presolve"], pump=D["pump"],
                      cuts=D["cuts"], fj=D["fj"], rins=D["rins"])
        wall = time.perf_counter() - t
        print(f"\n===== solve {rep + 1}: {r.status} obj {r.objective:.6g} bound {r.bound:.6g} gap {r.gap:.3g} "
              f"nodes {r.nodes}  wall {wall:.1f}s (result.seconds {r.seconds:.1f})")
        print("  times  :", {k: round(v, 2) for k, v in r.times.items()})
        print("  counts :", {k: v for k, v in r.counts.items() if not isinstance(v, str)})
        print("  incumb :", [(round(a, 1), round(b, 6)) for a, b in r.incumbents])
        print("  events (t from solve start, duration):")
        agg = defaultdict(lambda: [0, 0.0])
        for te, kind, what, dt in EV:
            print(f"    {te - t_solve0:7.2f}s  {dt:7.2f}s  {kind:10s} {what}")
            agg[kind][0] += 1
            agg[kind][1] += dt
        print("  totals :", {k: (n, round(s, 2)) for k, (n, s) in agg.items()})
        snap1 = snapshot()
        print("  cache files (total, new, rewritten):", diff(snap0, snap1))
        m1 = neuron_metrics()
        print("  neuron metrics delta:", {k: (m1[k] - m0.get(k, 0)) if isinstance(m1[k], int) else m1[k]["formatted"]
                                          for k in m1 if k != "error"} or m1)
        dyn1 = dict(torch._dynamo.utils.counters["stats"])
        print("  dynamo stats delta:", {k: dyn1[k] - dyn0.get(k, 0) for k in dyn1})
        sys.stdout.flush()


if __name__ == "__main__":
    main()
