"""Repro: torch.compile(backend="neuron") computes torch.linalg.norm(v / d, dim=0) wrong for some shapes.

Found 2026-10-27 on trn1.2xlarge (torch-neuronx 2.14.3, neuronx-cc 2.28.7631, Neuron DLAMI Ubuntu 24.04).
v: (m, B) float32, d: (m, 1) float32 (a per-row scaling, broadcast across B columns).
At m=1000, B=624: 240 of 624 columns wrong, by up to 40%. Correct at (1000, 512), (1000, 513), (1000, 1024)
and (7430, 624). Mathematically identical forms are correct at every shape: norm(v * (1/d)) and
sqrt(sum((v/d)**2)). Eager execution on the same device is correct.

How it showed up: in batched PDLP on setcover strong-branching LPs, the original-units violation norms in the
convergence check were wrong, so 33 of 624 LPs never met the tolerance and ran to the iteration limit (the
CPU, and the chip without torch.compile, solved all 624). trnopt now multiplies by precomputed reciprocals.

Run on a Neuron instance:  python examples/neuron_norm_bug_repro.py
(The "sum(v / d)" column only flags near-zero sums whose relative error is cancellation noise, not the bug.)
"""
import numpy as np, torch, torch_neuronx
torch._dynamo.config.recompile_limit = 1024
rng = np.random.default_rng(0)
variants = {
    "norm(v / d)":            lambda v, d, di: torch.linalg.norm(v / d, dim=0),
    "norm(v * dinv)":         lambda v, d, di: torch.linalg.norm(v * di, dim=0),
    "sqrt(sum((v/d)**2))":    lambda v, d, di: torch.sqrt(((v / d) ** 2).sum(0)),
    "norm(v)":                lambda v, d, di: torch.linalg.norm(v, dim=0),
    "sum(v / d)":             lambda v, d, di: (v / d).sum(0),
}
for m, B in ((1000, 128), (1000, 512), (1000, 513), (1000, 624), (7430, 624), (1000, 1024)):
    V = rng.normal(size=(m, B)).astype(np.float32); D = rng.uniform(0.1, 2, size=(m, 1)).astype(np.float32)
    v, d = torch.as_tensor(V).to("neuron"), torch.as_tensor(D).to("neuron"); di = 1.0 / d
    line = f"m={m:5d} B={B:5d} "
    for name, f in variants.items():
        torch._dynamo.reset()
        got = torch.compile(f, backend="neuron", dynamic=False)(v, d, di).cpu().double().numpy()
        ref = f(torch.as_tensor(V).double(), torch.as_tensor(D).double(), 1.0 / torch.as_tensor(D).double()).numpy()
        rel = np.abs(got - ref) / (np.abs(ref) + 1e-12)
        bad = np.where(rel > 1e-4)[0]
        line += f"| {name}: max rel {rel.max():.0e}" + (f" bad cols {bad.min()}..{bad.max()} ({len(bad)})" if len(bad) else "") + " "
    print(line, flush=True)
