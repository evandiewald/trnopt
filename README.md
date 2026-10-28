# trnopt — PDLP in PyTorch, on the way to Trainium

A small, readable LP solver using the same family of algorithm as NVIDIA's cuOpt (PDLP),
written in plain PyTorch and checked against HiGHS. It's a learning project heading toward
AWS Trainium; see the roadmap doc for the full plan.

**Status:** Phase 0 (CPU reference) and Phase 1 (portable backend) are done. Phase 2 prep is
done on CPU: the trn1 (PyTorch/XLA) code path, a float64 finishing step, a benchmark harness,
a first MIPLIB 2017 run, and a custom sparse-multiply NKI kernel verified in AWS's simulator.
Phase 3 (batching, including continuous batching) is built and measured on CPU. A sparse Metal
kernel runs it on Apple silicon GPUs, and a full-strong-branching benchmark replicates the setup of
NVIDIA's batched-PDLP paper. Next: the first run on a real trn1 instance.

## Quick start

```bash
uv sync
uv run pytest                                                # 89 tests (NKI / Mac GPU ones skip without the hardware)
uv run python examples/compare.py                            # PDLP vs HiGHS table, CPU float64
uv run python examples/compare.py --backend emulate-neuron   # Trainium's restrictions, on your CPU
uv run python examples/compare.py --presolve                 # shrink each LP with HiGHS presolve first
uv run trnopt-bench --backends cpu emulate-neuron --finish-float64 --presolve   # CSV + summary
```

Using it directly:

```python
from trnopt import LP, Backend, PDLPOptions, solve_pdlp, solve_highs

lp = LP.from_general(c=[3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], maximize=True)
print(solve_pdlp(lp).objective, solve_highs(lp).objective)   # -36.0 -36.0 (negated, since maximize)

# Same solve under Trainium's restrictions, with presolve:
res = solve_pdlp(lp, PDLPOptions(tol=1e-4, backend=Backend.preset("emulate-neuron"), presolve=True))
```

## The concepts, briefly

**Primal and dual.** Every LP has two sides. The *primal* is your decision variables (how
much of each loan to buy). The *dual* is one "price" per constraint: how much the objective
would improve if that constraint loosened slightly. At the optimum the two sides' objective
values are equal. That equality is how we know we're done.

**PDHG** (primal-dual hybrid gradient) is the core loop. Each iteration does two cheap moves:

1. Nudge `x` in the direction that lowers `cost − prices·constraints`, then clip `x` back into its bounds.
2. Nudge the prices `y` up for violated constraints and down for slack ones, then clip so `>=` constraints never get a negative price.

Each move is one multiplication by the constraint matrix `K` (or its transpose). No
factorizations and no sequential pivoting like simplex, which is why it suits parallel hardware.

**PDLP** = PDHG + practical tricks, all implemented here:

| Trick | What it does | Where |
|---|---|---|
| Scaling | Rebalances rows/columns of `K` so no constraint dominates (dollars vs basis points) | `_scale` |
| Primal weight `omega` | Balances how fast `x` moves vs how fast `y` moves | `omega` updates at restarts |
| Restarts | Periodically jump to the best recent point (often the running average) and start fresh. The biggest speedup | restart block in `_solve` |
| Stopping test | Stop when constraint violation, dual violation and the primal/dual gap are all small, relative to problem size | `relative_errors` |
| No-solution detection | Recognizes infeasible and unbounded LPs from how the iterates drift | `check_infeasible` |

**One deliberate difference from Google's PDLP and cuPDLP:** they search for an adaptive step
size every iteration. Here the step size is fixed, computed once from the matrix, so every
iteration does identical work. That's what accelerators handle best.

## Backends: how device differences are handled

All device differences live in one settings object, `Backend` (`src/trnopt/backend.py`). The
solver reads its settings and never asks which device it's on.

| Preset | Device | Numbers | Matrix storage | Compiled | What it's for |
|---|---|---|---|---|---|
| `cpu` | CPU | float64 | sparse | no | The reference |
| `cpu-float32` | CPU | float32 | sparse | no | Isolates the effect of float32 alone |
| `cuda` | NVIDIA GPU | float64 | sparse | no | Untested so far; for GPU comparisons later |
| `mps` | Apple silicon GPU | float32 | dense | no | Your Mac; see "Apple silicon" below |
| `neuron` | Trainium2/3 (TorchNeuron) | float32 | dense | yes | Benchmarks on trn2 |
| `neuron-xla` | Trainium1 / Inferentia2 (PyTorch/XLA) | float32 | dense | XLA compiles | Development on trn1 |
| `emulate-neuron` | CPU | float32 | dense | yes | Trainium's restrictions, on your laptop, for free |

Any field can be overridden: `Backend.preset("cpu", dtype=torch.float32)` isolates the effect of
float32 alone.

Why each restriction exists, and how we deal with it:

- **float32 only.** Trainium has no float64, and float32 carries about 7 significant digits
  instead of 16. Tight tolerances get slow or impossible (`shell`, `stair` and `25fv47` don't reach
  1e-6 within 100k iterations). `finish_float64=True` solves on the chip until progress stalls, then
  finishes in float64 on the CPU from that point (see below). NVIDIA GPUs have float64, so cuOpt
  doesn't face this.
- **No sparse matrices.** TorchNeuron (the new way to run PyTorch on Trainium) doesn't support
  them, and the chip has no sparse hardware. We store `K` as a full grid instead, and every
  multiply then does work on the zeros too. `MatVec` is the one place this is handled; a custom
  NKI kernel will plug in there later (Phase 4). cuOpt builds on cuSPARSE, NVIDIA's mature
  sparse library.
- **Round trips to the chip are expensive.** The hot loop is one pure function
  (`_Problem._step`) that `torch.compile` can fuse into a single step. Numbers are read back to
  the host only every 64 iterations, and step sizes are tensors, not Python numbers, so restarts
  don't trigger recompiles.
- **The trn1 software runs lazily (PyTorch/XLA).** It records operations into a recipe and only
  compiles and runs it when a value is read back. Two consequences, both handled: we `sync()` after
  every iteration (otherwise 64 iterations get unrolled into one giant program; on CPU this ran
  XLA's compiler out of memory), and every convergence check reads all 19 numbers it needs in ONE
  transfer. That cut the distinct programs to compile from 36 to 13, a fixed count for any problem
  size or iteration count. Compiles on real Neuron hardware can take tens of seconds each, so this
  matters.
- **Every new problem size means a recompile.** TorchNeuron doesn't support shapes that change, so
  each distinct problem size compiles once. On CPU that takes 1-20 s. It's fine for batches of
  same-size LPs (Phase 3), but expensive for one-off solves.
- **Compiles are cached on disk, so a new session doesn't redo them.** TorchNeuron keeps compiled
  programs (NEFFs) in `/tmp` by default, which is wiped every time the instance stops, so each
  session on a stop/start instance compiled everything again. Importing trnopt points TorchNeuron's
  caches (`TORCH_NEURONX_NEFF_CACHE_DIR`, `TORCH_NEURONX_HLO_CACHE_DIR`), Inductor's
  (`TORCHINDUCTOR_CACHE_DIR`) and the older XLA stack's (`NEURON_COMPILE_CACHE_URL`) at one
  directory, `TRNOPT_CACHE_DIR` (default `~/.cache/trnopt`). They're keyed on the program, not the
  LP, so re-solving an instance, or solving another of the same size, skips the compiler. You can
  copy the directory to a new instance or share it on EFS. Variables you set yourself win;
  `TRNOPT_CACHE_DIR=off` disables it and `trnopt.compile_cache.clear()` empties it. `import torch`
  loads torch-neuronx, which reads these locations as it loads, so import trnopt first.
  - Measured on trn1 (israel, first solve in a new process after wiping `/tmp` as a stop does):
    dense K 37–46 s before, 10–12 s now; NKI kernel 20 s before, 10 s now. Of the remaining time,
    ~7.5 s is starting the Neuron runtime (the first copy to the device), not compiling.
  - NKI kernels: NKI's own cross-process cache (in `/var/tmp`) keys on the kernel's name and input
    shapes, not on constants captured from the surrounding Python. Our kernel bakes in the matrix's
    slice layout that way, so two matrices with equally sized tables could have shared a kernel. Each
    kernel is now named after its layout. Within a process, kernels are also shared across solves of
    the same layout.

### What Phase 1 measured (CPU, 640 fixed iterations)

| problem | size | nonzeros in K | sparse float64 | dense float32 | dense float32, compiled |
|---|---|---|---|---|---|
| israel | 174 × 142 | 9% | 116 µs/iter | 84 µs/iter | 136 µs/iter |
| 25fv47 | 821 × 1571 | 0.8% | 211 µs/iter | 380 µs/iter | 429 µs/iter |
| random | 2000 × 4000 | 0.3% | 295 µs/iter | 2947 µs/iter | 4133 µs/iter |

Two lessons:

1. **Dense storage wastes work in proportion to how empty K is.** At 0.3% nonzeros, the dense
   multiply does ~300× more arithmetic, and comes out 10× slower even though dense math is more
   efficient per operation. Trainium's matrix engine is far faster than a CPU's, but this waste
   factor is why sparse support (Phase 4) matters for big problems, and why batching (Phase 3)
   is the better route: it turns the waste into useful work.
2. **Compiling doesn't help on a CPU, and here it hurts slightly.** On a CPU the operations
   already run in-process with no device round trip, so there's nothing for fusing to save. The
   benefit only appears on an accelerator. We'll measure that in Phase 2.

### The float64 finishing step

`finish_float64=True` (float32 backends only) runs the chip until progress stalls, then hands the
answer, plus the learned step-size balance, to a float64 solve on the CPU. "Stalls" means the
error is below 1e-4 *and* hasn't halved over the last half of the run. Problems like `stair`
plateau even in float64, so a short fixed window cut them off too early.

Iterations to 1e-6, emulated Trainium + float64 finish vs CPU float64 alone:

| problem | CPU float64 alone | on chip (float32) | CPU finish (float64) | CPU work vs alone |
|---|---|---|---|---|
| loan_selection_2000 | 2,240 | 2,624 | 0 | 0% |
| israel | 9,088 | 10,816 | 0 | 0% |
| shell | 29,056 | 14,272 | 2,560 | 9% |
| 25fv47 | 129,856 | 50,240 | 38,400 | 30% |
| random (seed 0) | 4,736 | 6,336 | 2,624 | 55% |
| stair | 79,680 | 129,984 | 43,392 | 54% |
| adlittle | 12,032 | 4,032 | 10,496 | 87% |

The honest reading: the last digits of accuracy are the slow part for PDLP, and they're exactly
what float32 can't do. On easy problems the chip does everything; on hard ones the CPU still does
a third to most of the work. Two implications for later phases:

- **Trainium fits best where ~1e-4 accuracy is enough.** MIP heuristics (Phase 5), branching
  estimates and scenario screening all qualify, and that's also how cuOpt uses PDLP inside its
  MIP heuristics.
- **Tuning on 7 problems would just overfit them.** Handoff settings get revisited with the
  benchmark harness on a larger instance set.

### Presolve

`presolve=True` borrows HiGHS's presolve, which removes redundant rows, fixes forced variables
and so on. It shrinks `shell` from 536 × 1775 to 229 × 1142, and proves the infeasible and
unbounded test LPs have no solution before PDLP even starts. Benchmarks must label it:
"trnopt + HiGHS presolve".

One trap, now handled: postsolve rebuilds the removed variables from the kept ones, so PDLP's
small errors get amplified (about 30× on `shell`). Simplex answers are exact, so HiGHS never sees
this. trnopt checks the answer on the *original* LP after postsolve. If it misses the tolerance,
it re-solves the reduced LP 10× tighter, warm-starting from the last answer. If that still fails,
it reports `inaccurate` instead of `optimal`.

## MIPLIB 2017: first results

The 240-instance MIPLIB 2017 benchmark set, run as LP relaxations (the LP a MIP solver solves
at its root node).

**How the matrices look.** Median density is 0.14%: 99.86% zeros. As a float32 full grid,
79 instances fit in 40 MB, 129 in 400 MB, and 216 in 32 GB (one trn1 chip).

**Float32 vs float64 at 1e-4** (the 79 instances that fit in 40 MB, 30 s limit, presolve on,
sparse storage for both so only precision differs):

| | solved correctly | SGM solve time |
|---|---|---|
| HiGHS | 79 / 79 | 0.42 s |
| PDLP, CPU float64 | 69 / 79 | 2.6 s |
| PDLP, CPU float32 | 69 / 79 | 3.6 s |

- At 1e-4, float32 costs almost nothing. On instances both solve, the median iteration count is
  identical, and 90% need at most 1.44x the iterations.
- About 5% of instances are float32-hard: `rococoC10-001000` takes 1,472 iterations in float64
  and over 259,000 in float32. `finish_float64` is for these.
- "1e-4" doesn't mean a 1e-4 objective error. Four instances (`swath1`, `swath3`,
  `supportcase26`, `neos-3754480-nidda`) meet the tolerance in both precisions but are 0.1-0.5%
  off HiGHS's objective. PDLP's test measures violations relative to the size of the whole
  right-hand side, and a big right-hand side lets small relative violations move the objective.
  Report objective error alongside tolerance.
- The run caught a bug: float32's looser infeasibility threshold falsely declared
  `cost266-UUE` infeasible 128 iterations in. Fix: skip the first few checks, and require
  three agreeing checks in a row.
- **Update (2026-10-26): a fix for float32 stalls, after this table was made.** Part of the
  optimality test adds up "bound x reduced cost" for every bounded variable. Reduced costs carry
  float32 rounding noise, and the way the sum is written lets that noise only push one way, so on
  LPs with many bounded variables it adds up to a permanent gap floor (about 5e-3 on comp21-2idx's
  branch LPs). Now a bound only counts when x is actually at it, which is what Google's PDLP does by
  default. `rococoC10-001000` (above: over 259,000 float32 iterations) now solves in 0.5 s, `n5-3`
  in 0.4 s, and all 128 comp21 branch LPs solve where 29 used to stall. The table hasn't been rerun.

## Batching (Phase 3)

`solve_pdlp_batch(lp, B, lower=..., upper=..., c=..., q=...)` solves B LPs that share the
constraint matrix K but differ in bounds, objective or right-hand side. That's exactly what MIP
branching and scenario sweeps produce. Every iteration multiplies K by an n x B matrix instead
of a vector, so each byte of K read does B times the useful work.

- **Shared:** scaling, step size, K. **Per LP:** primal weight, restarts, convergence and
  infeasibility verdicts, all done with column masks.
- **Fixed shapes:** finished LPs are frozen in place rather than removed, because on Trainium a
  new shape means a recompile. `useful_work` reports the cost: 64–98% at B = 128 with one fixed
  batch.
- **Continuous batching** (`solve_pdlp_stream(lp, n_lps, slots, ...)`): a queue of LPs feeds the
  slots, and each slot gets the next LP as soon as its current one finishes, the same idea LLM
  servers use. Refills are full-width masked updates, so they never create new programs to
  compile. On the trn1-style XLA path, 48 LPs through 6 slots compiled nothing beyond the first
  run's programs. Each LP counts its own iterations from when it was loaded, and refills only
  happen at check points, so every LP's result matches solving it alone exactly.
- **Workloads** (`problems.py`): `branching_variants` (strong branching: down/up LPs for the most
  fractional integer variables) and `return_scenarios` (objective perturbations).
- **Correctness:** every column gives exactly what solving that LP alone gives (same iterations,
  same objective). Mixed batches of feasible and infeasible LPs get the right verdict per column
  in float64, float32 and emulated Trainium. The batched solver with the simulated NKI kernel
  matches the PyTorch path iteration for iteration.

**Throughput, strong-branching batches** (2-core cloud CPU, tol 1e-4; full data in `results/`):

| instance (rows x cols) | setting | per LP, alone | best batched | speedup |
|---|---|---|---|---|
| neos-911970 (107 x 888) | CPU float64 | 0.089 s | 0.0057 s (B=128) | 16x |
| | emulated Trainium | 0.063 s | 0.0063 s (B=128) | 10x |
| neos-4954672-berkel (1848 x 1533) | CPU float64 | 0.43 s | 0.081 s (B=32) | 5.4x |
| | emulated Trainium | 1.15 s | 0.24 s (B=32) | 4.7x |
| nu25-pr12 (2313 x 5868) | CPU float64 | 1.0 s | 0.29 s (B=32) | 3.5x |
| | emulated Trainium | 14.9 s | 1.31 s (B=128) | 11x |
| neos-3627168-kasai (1655 x 1462) | CPU float64 | 4.4 s | 0.87 s (B=32) | 5x |

What it shows:
- **Batching pays off everywhere: 3.5–16x per LP.** The full-grid (Trainium-style) path gains
  most, because a matrix-times-matrix product puts the wasted zeros to use better than
  matrix-times-vector does.
- **But batching doesn't remove the full grid's waste.** On nu25-pr12 (0.13% nonzeros), batched
  emulated Trainium at 1.31 s per LP is still slower than batched sparse CPU at 0.29 s. That's the
  case for the sparse NKI kernel below.
- **On a CPU the sweet spot is often B = 32,** with B = 128 slightly worse (cache limits plus
  frozen columns). Trainium's sweet spot will differ and needs measuring on hardware.
- **neos-3627168-kasai is float32-hard.** Its branches need 19k–32k iterations in float64, and many
  don't finish within 200k in float32. Batched float32 times for it are omitted above; see the
  CSV.

**float64 finish in batch mode** (`finish_float64=True`): each slot watches its LP for a precision
stall. A stalled LP leaves its slot, which refills at once. Stalled LPs, plus any that use up the
device budget (`max_iters`), are then finished in float64 on the CPU in a second batched stream,
warm-started from the device's answer. On kasai's 32 branches (target 1e-4), float32 alone solves
12/32; with the finish, 32/32, with the CPU doing 12% of the iterations
(`results/2026-10-25_kasai_float64_finish.md`). The stall rule is deliberately conservative:
kasai plateaus for long stretches in float64 too, and an eager rule would hand LPs off mid-progress.

**Continuous vs fixed batches** (128 strong-branching LPs; full data in `results/`):

| instance | setting | slots | fixed batches (useful work) | streamed (useful work) | faster |
|---|---|---|---|---|---|
| neos-911970 | emulated Trainium | 8 | 3.38 s (85%) | 2.58 s (97%) | 24% |
| berkel | emulated Trainium | 8 | 88.8 s (82%) | 72.8 s (96%) | 18% |
| nu25-pr12 | emulated Trainium | 8 | 553 s (85%) | 468 s (99%) | 15% |
| nu25-pr12 | emulated Trainium | 32 | 183 s (78%) | 156 s (89%) | 15% |
| berkel | CPU float32 | 32 | 7.44 s (81%) | 7.30 s (85%) | 2% |

Streaming lifts useful work to 85–99% and cuts time 2–24%. With 32 slots and only 128 LPs, the
final drain (the last LPs finishing while slots empty out) limits the gain. A MIP search
supplies thousands of LPs, so it looks like the 8-slot rows: only 1–4% waste.

**A float32 trap batching uncovered:** on kasai, float32 first declared every branch infeasible.
The prices were oscillating along a line, not diverging, and the leftover error (~2e-6)
multiplied variables of size ~920,000. The infeasibility test now also requires a steady drift
direction (cosine >= 0.99 between checks) and weights the leftover error by the size of the
variables it acts on.

## Apple silicon (Mac GPU)

The `mps` backend setting runs trnopt on a Mac's GPU through PyTorch's Metal backend:
float32 (Apple GPUs have no float64), full-grid K (PyTorch's sparse tensors don't run on MPS:
`NotImplementedError: Could not run 'new_compressed_tensor' from the 'mps:0' device`). CPU and
GPU share memory, so the float64 finish needs no copies.

Run it in macOS's own Terminal (the GPU isn't visible from Linux VMs or containers):

```bash
uv run python examples/mac_benchmark.py            # correctness + speed, a few minutes
uv run python examples/mac_benchmark.py --quick    # correctness only
```

**First results, base M4 MacBook Pro (16 GB, 10-core GPU)**, `results/mac_Apple_M4_2026-10-26.csv`:

- **Correct:** all 5 problems match HiGHS on the GPU, and shell reaches 1e-6 with the float64 finish
  (14,272 GPU + 2,816 CPU iterations).
- **One LP at a time, the GPU loses:** 3–18x slower per iteration than the M4's own CPU (e.g.
  nu25-pr12: 1,378 µs on the GPU vs 74 µs on the CPU). A single full-grid matrix-times-vector does
  hundreds of times the necessary work and leaves the GPU mostly idle.
- **Batched, the GPU catches up:** batching speeds the GPU up 29–43x per LP, against 7–14x on the
  CPU. At B = 128:

| instance (nonzeros) | CPU float64, sparse | CPU float32, sparse | Mac GPU float32, full grid |
|---|---|---|---|
| neos-911970 (3.6%) | 0.0025 s/LP | 0.0026 s/LP | **0.0022 s/LP** |
| neos-4954672-berkel (0.28%) | 0.0285 s/LP | **0.0195 s/LP** | 0.0266 s/LP |
| nu25-pr12 (0.13%) | 0.106 s/LP | **0.064 s/LP** | 0.123 s/LP |

The GPU roughly ties the CPU's sparse path while doing 28–770x more arithmetic (the full grid's
zeros).

### The sparse Metal kernel (`mps-metal`)

`metal_sell.py` ports the sliced-ELL layout (see the NKI section below) to a ~25-line Metal
kernel, compiled at runtime with `torch.mps.compile_shader` and run directly on PyTorch's GPU
tensors. One GPU thread per (row, batch column); Apple GPUs handle scattered reads natively, so it
was far easier than on Trainium.

- **Correct:** matches SciPy to ~1e-7 on 7 MIPLIB instances up to 77k x 33k, with 0–5% padding
  (the full grid wastes 28–770x, and the four biggest instances don't fit as a full grid at all).
- **One LP at a time, it still loses to the M4's CPU, ~3x per iteration**
  (`examples/mac_single_lp.py`): comp21-2idx 401 vs 123 µs, neos-933966 634 vs 202,
  chromaticindex512-7 873 vs 320, rahue 1,395 vs 430.

Why, in plain terms:
1. **Fixed cost per GPU operation.** An iteration is the two multiplies plus ~20 small vector
   operations, and each is a separate GPU launch with a fixed cost. On comp21 the multiplies are
   ~190 µs of the GPU's 401 µs; the rest is overhead. Fusing the iteration into one kernel would
   cut that, but the GPU multiply alone is already slower than the CPU's whole iteration.
2. **Shared memory removes the GPU's usual edge.** Sparse multiplies are limited by memory reads,
   not arithmetic. NVIDIA GPUs (and Trainium) read their own HBM/GDDR at many times the CPU's memory
   speed; that's what cuOpt rides. A Mac's CPU and GPU read the same memory, and these matrices
   (≤2 MB) fit in the CPU's caches anyway.

Local LLMs run well on Macs for the opposite reasons: gigabytes of weights per token (no cache
helps), a few hundred large fused operations per step, and dense matrices. The win there is mostly
capacity (a 70B model fits) and fast memory compared with a PC's RAM, not a GPU beating the Mac's CPU.

Batched, the kernel does win modestly: see the strong-branching results below, where the GPU beats
the CPU by 1.1–1.4x on 4 of 5 instances.

## Full strong branching vs NVIDIA's batched-PDLP paper

The closest published work to this project is *Batched First-Order Methods for Parallel LP Solving
in MIP* (Blin, Maes, Gualandi, Lodi, Stellato; arXiv 2601.21990, Jan 2026), from NVIDIA's cuOpt team:
batched PDHG for full strong branching, reporting 12–489x over 8-thread dual simplex on an H100/B200.
(cuOpt's MIP announcement is about something else: GPU primal heuristics, measured by solutions
found within a time limit.)

- `generators.py` builds the paper's four problem families (from Gasse et al. 2019) at its sizes:
  set cover, combinatorial auction (simplified generator), maximum independent set (clique
  formulation) and capacitated facility location (matches the paper's sizes exactly).
- `examples/strong_branching_benchmark.py` branches on every fractional variable (capped at 1,000,
  so up to 2,000 LPs) and compares against **HiGHS dual simplex warm-started from the root LP's
  basis, one LP per core on all cores**. That's how a CPU MIP solver does strong branching, and the
  fair baseline (the paper makes the same point about earlier GPU work). trnopt is batched (128 LPs
  at a time), float32, tolerance 1e-4, warm-started from its root solution.

**Base M4 MacBook Pro, HiGHS on all 10 cores** (`results/strong_branching_Apple_M4_2026-10-26.csv`):

| instance (nonzeros) | LPs | HiGHS warm, 10 cores | trnopt CPU | trnopt Mac GPU | GPU vs HiGHS |
|---|---|---|---|---|---|
| setcover 1000x1000 (10k) | 624 | 4.3 s | 8.0 s | 5.6 s | 0.77x |
| cauction 360x1400 (8k) | 706 | 6.1 s | 7.6 s | 6.3 s | 0.96x |
| indset 1000 (16k) | 1,698 | 12.0 s | **7.2 s** | 7.7 s | **1.55x** (CPU 1.66x) |
| facility 400x200 (320k) | 20 | 3.5 s | 92.6 s | 79.4 s | 0.04x |
| chromaticindex512-7 (135k) | 2,000 | 352 s | 123 s | **112 s** | **3.15x** |

- **Every answer was usable:** all LPs finished, objectives within 1e-4 of HiGHS, every
  infeasible child agreed, and the branching variable picked from our scores was always as good as
  the best by exact scores.
- **Wins on 2 of 5, against a fair multi-core baseline, on a laptop:** chromaticindex512-7 (the
  paper's 176x instance) and independent set. On chromaticindex warm-started dual simplex is
  unusually slow (1.6 s per LP on one core, slower than cold PDLP), the case PDLP is built for.
  Caveat: chromaticindex is so symmetric that all 1,000 candidates tie on score, so it measures LP
  throughput, not branching quality.
- **Facility location is PDLP's weak spot here:** ~17,500 iterations per LP. The paper's 489x
  there comes from H100 memory bandwidth (~28x the M4's) and a newer PDHG variant (reflected Halpern
  iterations) that needs fewer iterations.
- **The Mac GPU beats the Mac CPU on 4 of 5, by 1.1–1.4x.** Modest, as the unified-memory argument
  predicts. The paper's magnitude needs hardware with its own fast memory: Trainium is next.

## End-to-end MIP: a branch-and-bound split the way cuOpt splits it

`trnopt.mip.solve_mip(lp, branching=...)` solves whole MIPs: it finds whole-number solutions and
proves the best one optimal. The design follows cuOpt's division of labor:

- **Exact core on the CPU.** Every node's LP is solved by HiGHS dual simplex, warm-started from its
  parent's basis (a child differs from its parent by one bound, so this takes a handful of pivots).
  Bounds, pruning and the optimality proof only ever use these exact results, so float32
  approximations can never make the answer wrong.
- **Batched, approximate work on the accelerator.** Strong branching (try both halves of a candidate
  variable to see which split raises the bound most) runs on trnopt's batched PDLP. Approximate answers
  are fine there: they only decide *which* variable to split on.

`examples/mip_benchmark.py` compares branching rules on the same search core:

| rule | extra LPs | |
|---|---|---|
| `most_fractional` | none | split the variable closest to .5 |
| `pseudocost` | none | split the variable that helped most in the past |
| `reliability_highs` | strong branching on variables with too little history, solved by HiGHS | what HiGHS, Gurobi, cuOpt do |
| `reliability_trnopt` | the same LPs, solved by batched trnopt | |

**What I learned building it** (all on a 2-core cloud CPU; Mac results to come):

1. **Full strong branching loses on time no matter what solves its LPs.** On a 300-node independent
   set it shrinks the tree 12x (577 -> 49 nodes) but adds ~450 LPs per node, each about as expensive as
   the node's own LP. That's why real solvers use reliability branching, and so does this benchmark.
2. **Strong branching needs a ranking, not an answer.** Stopping PDLP after 256 iterations picked the
   same variable as exact scores on every family, at ~10x less cost than full convergence (dual simplex
   strong branching is usually pivot-limited for the same reason).
3. **Measure gains approximate-vs-approximate.** Each batch also re-solves the parent LP, and gains are
   taken relative to that, so trnopt's systematic error cancels.
4. **Keep the search identical across engines.** HiGHS strong branching produces the chosen children's
   exact LPs for free; to compare fairly, the trnopt path solves those two children exactly right away
   too. Before that fix, trnopt's trees were up to 2x larger for reasons unrelated to its accuracy.

**Results so far (2-core cloud CPU, every run proven optimal, objectives match HiGHS's MIP solver):**

| MIP | HiGHS MIP solver (context) | reliability, HiGHS 1 core | 2 cores | reliability, trnopt CPU |
|---|---|---|---|---|
| independent set, 800 nodes | 8.6 s | 59.3 s | 44.0 s | **42.5 s** |
| combinatorial auction 150x700 | 41.6 s | 41.2 s | 34.1 s | 36.3 s |
| set cover 200x400 | 2.0 s | 0.7 s | | 1.1 s |
| facility location 40x30 | 0.7 s | 0.4 s | | 2.6 s |

- With the engine as the only difference, trnopt on a CPU is roughly even with HiGHS on the two MIPs
  where strong branching dominates (99% of the time on independent set), and behind where batches are
  tiny (facility location has 2–4 candidates per node, so there is nothing to batch).
- Our simple solver is competitive with HiGHS's full MIP solver on the auction and set cover instances
  and far behind on independent set, where HiGHS's presolve and clique cuts do the work. That gap is
  the machinery this solver deliberately leaves out.

### Experiment: Halpern PDHG (the method behind cuPDLPx)

The paper's solver uses *restarted reflected Halpern PDHG* rather than PDLP's restarted averaging.
`examples/halpern_experiment.py` prototypes it on trnopt's building blocks. Instead of restarting
from an average, each step is pulled slightly back toward a fixed anchor point, and "reflection"
overshoots each step by a factor gamma.

| (float32, 1e-4, 30 s, 79 MIPLIB relaxations) | solved | median iterations vs PDLP | total iterations | facility 400x200 |
|---|---|---|---|---|
| PDLP (current) | 68 | 1x | 502k | 17,536 |
| Halpern, gamma = 1 | 61 | 1.5x fewer | n/a | 6,400 |
| Halpern, gamma = 0.5 | 66 | 1.28x fewer | 592k | 11,328 |

Full reflection is fastest when it works but unstable on harder instances in float32; half reflection
is robust but no faster overall. So it stays an experiment, not the default. cuPDLPx reports
2.5–3.6x on MIPLIB with pieces this prototype lacks (a PID controller for the primal weight, tuned
constants). On facility location the new method was also more accurate (objective error 1.2e-5 vs 3.2e-4).
Data: `results/2026-10-26_halpern_vs_pdlp_miplib79.csv`.

## The sparse NKI kernel (Phase 4, started early)

Trainium has no sparse hardware, and TorchNeuron has no sparse tensors. So we wrote our own
kernel, `nki_sell.py`, using a layout called sliced ELL (`sell.py`):

1. Split rows longer than a chosen size into pieces. The size is picked per matrix (`choose_split`).
2. Sort rows by length and group them in slices of 128, one row per lane of Trainium's
   128-partition on-chip memory.
3. Pad rows within a slice to equal length, and store a column-index table and a value table.

Each lane then repeatedly fetches `X[col, :]` (one "indirect DMA" moves 128 independently
addressed rows), multiplies and accumulates.

How much work each layout does, relative to the actual nonzeros (median over a 29-instance
MIPLIB sample):

| layout | work vs nonzeros |
|---|---|
| full grid (current Trainium path) | 723x |
| only non-empty 128x128 tiles (the original Phase 4 plan) | 175x |
| sliced ELL with row splitting | 1.08x |

Status: correct in AWS's NKI CPU simulator, on random matrices and real MIPLIB matrices, for
single vectors and batches. Full solves with every multiply going through the simulated kernel
(`Backend(spmv="nki-sim")`) match the PyTorch path iteration for iteration. Not yet compiled or
timed on hardware: compiling needs the Neuron runtime library, which only exists on Neuron
machines.

To use it locally: `pip install neuronx-cc --extra-index-url https://pip.repos.neuron.amazonaws.com`
(Python 3.11 matches the trn1 stack).

## Results (CPU, float64, tolerance 1e-6, no presolve)

| problem | rows × cols | HiGHS obj | PDLP obj | PDLP iters | obj rel. error |
|---|---|---|---|---|---|
| textbook | 3 × 2 | -36 | -36 | 192 | 1e-9 |
| random (seed 0) | 120 × 200 | -212.7063 | -212.7062 | 4,736 | 5e-7 |
| loan_selection_2000 | 17 × 2000 | -1,480,271 | -1,480,272 | 2,240 | 9e-7 |
| afiro (Netlib) | 27 × 32 | -464.753 | -464.753 | 640 | 1e-7 |
| adlittle (Netlib) | 56 × 97 | 225,495.0 | 225,494.6 | 12,032 | 2e-6 |
| israel (Netlib) | 174 × 142 | -896,644.8 | -896,644.8 | 9,088 | 1e-7 |
| shell (Netlib) | 536 × 1775 | 1.20883e9 | 1.20883e9 | 29,056 | 2e-7 |
| stair (Netlib) | 356 × 467 | -251.267 | -251.267 | 79,680 | 1e-7 |

HiGHS is much faster on all of these. That's expected: on small problems simplex on a CPU is
hard to beat, and PDLP's advantage only shows on very large problems or batches on parallel
hardware. Infeasible and unbounded test LPs are detected correctly in both float64 and float32.

## Running on trn1 (Phase 2 checklist)

trn1 needs the older PyTorch/XLA-based Neuron stack (PyTorch 2.9). TorchNeuron, the newer path,
supports only Trn2/Trn3.

1. Launch a `trn1.2xlarge` (spot is fine) from AWS's Neuron Deep Learning AMI, and activate its
   PyTorch 2.9 / torch-neuronx environment. Check the AMI's notes for the exact name.
2. `pip install -e . && pip install pytest`
3. `pytest -q` (the `neuron-xla` test runs on the chip when torch_xla finds a Neuron device)
4. `trnopt-bench --backends neuron-xla --finish-float64 --hardware trn1.2xlarge --price-per-hour 1.344 --out trn1.csv`

To rehearse locally, use XLA's CPU mode with the same versions trn1 uses:

```bash
uv venv -p 3.11 .venv-xla
VIRTUAL_ENV=.venv-xla uv pip install "torch==2.9.*" "torch_xla==2.9.*" numpy scipy highspy pytest setuptools -e .
PJRT_DEVICE=CPU .venv-xla/bin/python -m pytest -q          # 59 tests, including the XLA path
```

## Layout

```
src/trnopt/
  backend.py     Backend presets + MatVec ("multiply by K", sparse or dense; NKI plugs in here)
  compile_cache.py  one persistent directory for compiled programs (Inductor + Neuron caches)
  bench.py       benchmark harness: CSV per solve, compile time separate, $/solve, summary
  batch.py       batched + continuous-batching solver: LPs sharing K through fixed slots, per-slot restarts / verdicts
  sell.py        sliced-ELL sparse layout + NumPy reference multiply
  nki_sell.py    the NKI kernel for it (runs in AWS's simulator; hardware untested)
  metal_sell.py  the same layout as a Metal kernel for Apple silicon GPUs (backend "mps-metal")
  generators.py  set cover / combinatorial auction / independent set / facility location MIPs at published sizes
  mip.py         branch-and-bound: exact HiGHS core per node, strong/reliability branching on HiGHS or trnopt
  pdlp.py        the solver (heavily commented, start here)
  presolve.py    HiGHS presolve / postsolve wrapper
  lp.py          one LP format: min c@x, equality rows then ">=" rows, variable bounds
  reference.py   HiGHS wrapper + MPS file loader (for Netlib / MIPLIB files)
  problems.py    textbook toy, random LPs, loan selection, deliberately infeasible / unbounded LPs
tests/           every test runs on cpu, cpu-float32 and emulate-neuron (+ neuron-xla if torch_xla is installed)
examples/        compare.py side-by-side table; mac_benchmark.py, mac_kernel_benchmark.py and
                 mac_single_lp.py for Apple silicon; strong_branching_benchmark.py (vs warm-started HiGHS);
                 halpern_experiment.py (reflected Halpern PDHG prototype); mip_benchmark.py (whole MIPs)
data/            six small Netlib instances (from the HiGHS repo); data/miplib/ has seven MIPLIB 2017
                 instances (https://miplib.zib.de) for the batch and kernel benchmarks
```

## Known gaps

- Never run on real Trainium yet.
- `cuda` preset is untested.
- Batch mode has no presolve yet.
- Dual values aren't returned when presolve is on (postsolve maps only the primal solution).
