# Why the first solve in a fresh process was worse on Trainium (trn1.2xlarge, 2026-10-28)

Benchmark: `examples/pyomo_benchmark.py --mps "data/miplib/*.mps.gz" --solvers trnopt --backend neuron
--time-limit 120`. Diagnosis: `examples/fresh_process_diag.py` (per-solve phase times, every batched PDLP call,
first call of each compiled function per shape, TorchNeuron cache counters with `TORCH_NEURONX_METRICS_ENABLED=1`).

## Root cause

The persistent cache worked: the keys are stable across processes. In a fresh process with a warm cache, all
92 of neos-933966's and rahue's NEFF lookups hit. Three other things were going on:

1. **Which batch sizes a solve needs depended on timings.** Strong branching's candidate cap divides a time
   share by the seconds per strong LP measured so far. The first call's measurement included compile and
   load time, so a fresh process picked small caps and a later solve picked large ones. Each new
   power-of-two bucket is a new set of programs: 35-38 s to compile on neos-933966 and 47-101 s on rahue,
   about 11 s of that in the eager ops that load slots and record results. Every process wandered into
   buckets no earlier run had compiled, e.g. a fresh process on a warm cache compiled B=256 and B=64
   (74 misses, 72 s of its 80 s of strong branching). This is also why run 2 looked better: it reused
   buckets run 1 had paid for.
2. **Compile and load time was charged to phases with their own budgets.** The pump's first round includes
   the first call of its programs. On a cache hit that is 9.5 s (neos-933966) or 12.8 s (rahue) of loading
   in a fresh process. On a miss it is 14-50 s. The pump's whole budget is 24 s, so run 1 got 1 pump round
   instead of 7, and that alone is neos-933966's 54,297 vs 1,366.
3. **Compiles can't be interrupted.** A strong-branching call that hits a new bucket near the end compiles
   for 35-100 s whatever the time limit says, which is how chromaticindex and rahue reached 225 s.

Smaller, per-process items (unavoidable or cheap): first device touch (Neuron runtime init) 5-10 s; Dynamo
tracing and NEFF loading per cached shape, about 1-6 s each; NKI kernel builds ~0 s (traced lazily, cached).

Also found: a torch.compile'd graph's HLO carries source locations. The two compiled graphs per bucket miss
the cache when trnopt runs from another path, or after an edit to batch.py above them; the eager ops still
hit. This explained a 2-NEFF miss when the same code ran from a second directory.

## Per-phase breakdown, before the fix (base code, `examples/fresh_process_diag.py`)

neos-933966 (12047 x 31762):

| | fresh process, warm cache | same process, 2nd solve |
|---|---|---|
| pump, first round | 10.95 s (warm: 1.5 s): load | 1.5 s |
| root strong, B=512 | 4.5 s (warm: 2.7 s): load | 2.7 s |
| new buckets | B=256 38.6 s, B=64 34.0 s: compile (74 misses) | B=1024 38.0 s, B=128 34.2 s: compile |
| pump rounds / incumbent | 4 / 2,365 | 7 / 1,366 |

neos-4300652-rahue (76992 x 33003):

| | fresh process, warm cache (0 misses) | same process, 2nd solve |
|---|---|---|
| first device touch | 8.1 s | 3.6 s |
| pump, first round | 16.8 s (warm: 4.0 s): load | 4.0 s |
| root strong, B=512 | 22.4 s (warm: 15.9 s): load | 15.9 s |
| new buckets | none (B=32 cached) | B=64 47.2 s, B=32 67.9 s: compile, wall 176 s |

## Fix (src/trnopt/mip.py, src/trnopt/batch.py)

* `batch.warm_up(lp, slots, backend)` runs every program a stream of that width uses, once (two
  evaluation intervals: load, step, measurements, restart, record).
* On Neuron, `solve_mip` runs warm-ups in one background thread. The pump's warm-up starts at the
  beginning, overlapping FJ, cuts and the root LP. The pump waits for it; that wait is reported as
  `times["warmup"]` and is not part of the pump's budget. Strong branching waits at most its root share
  (10% of the limit) at the root and doesn't wait at other nodes: until its programs are ready, nodes
  branch on pseudocosts. A compile can no longer eat a phase's budget, distort the speed the candidate cap
  is computed from, or overrun the limit. The main thread never uses the device while a warm-up runs.
* Strong branching uses at most two slot counts per solve, both fixed by the problem rather than timings,
  so the next process finds them cached: the first call's bucket, plus the bucket for the smallest node
  batch (8 candidates -> 32) if the warm-ups so far suggest it fits in half the remaining time. Each call
  takes the cheaper of the two (passes x measured seconds per pass); larger batches stream through.
* `counts["warmup_seconds"]` is the total background warm-up time.

## After the fix: fresh process, `--repeat 1`, 120 s

`pyomo_trn1.2xlarge-miplib-fresh_2026-10-28.csv`: warm cache. `...-fresh-coldgraphs_...csv`: the pass before
it, where the compiled graphs had to be recompiled (new path and edited batch.py).

| instance | before: run 1 (fresh) | before: run 2 (warm) | after: fresh, warm cache | after: fresh, cold graphs |
|---|---|---|---|---|
| chromaticindex512-7 | 4, **225.4 s** | 4, 163.6 s | 4, 122.5 s | 4, 122.9 s |
| comp21-2idx | 1315, 151.6 s | 1307, 135.7 s | 1401, 121.4 s | 1493, 120.6 s |
| neos-4300652-rahue | 7.35, **226.7 s** | 7.87, 128.7 s | 7.81, 122.0 s | 10.08, 122.0 s |
| neos-4954672-berkel | 7.09e6, 129.7 s | 3.84e6, 120.1 s | **3.84e6**, 120.1 s | 3.84e6, 120.1 s |
| neos-911970 | 85.68, 129.9 s | 75.79, 120.0 s | 91.01, 120.0 s | 85.20, 120.0 s |
| neos-933966 | 54,297, 122.2 s | 1,366, 121.9 s | **1,366**, 121.7 s | 1,366, 122.2 s |
| nu25-pr12 | 54,165, 137.0 s | 53,905, 120.1 s | 54,160, 120.1 s | 54,155, 120.2 s |

Longest solve: 226.7 s -> 122.9 s. neos-933966 and berkel now get the warm answer in a fresh process.
comp21-2idx and neos-911970 look worse, but checking old vs new code side by side (two solves each)
shows their only incumbent comes from Feasibility Jump or the root rounding in the first 6-16 s and never
improves, in either version (911970: old 94.0 / 94.0, new 93.4 / 85.0; comp21: old 1358 / 1299, new
1438 / 1317). FJ is time-limited, so these objectives are run-to-run noise, not the fix. The fix did let
those solves search more: 911970 fresh 1486 nodes vs 444, comp21 fresh 3164 strong LPs vs 964, on time
(base: 131.5 s).

## Left

* The background warm-up shares the GIL with the CPU phases it overlaps: comp21's cuts took 15.3 s instead
  of 9.0 s (still within their own 24 s cap). neuronx-cc itself runs as a subprocess.
* A never-seen instance still compiles its programs once (2 matrices x 1-3 buckets, 15-100 s each on these
  models). Now that's spent in the background, with nodes going on without strong branching meanwhile.
  The pump uses the uncut model, so it has its own matrix and compiles separately from strong branching.
* Compiled-graph keys depend on source path and line numbers (see above): a reinstall elsewhere, or an edit
  to batch.py, recompiles those graphs.
* A warm-up still running when the time limit hits finishes in the background (the thread is joined at
  interpreter exit). In a long-lived process, it could overlap the next solve's device use.
* Strong-branching calls started just before the limit still run their 1 s minimum stream: up to ~3 s over.
