# Benchmark results

Mostly from `trnopt-bench`, LP relaxations of MIPLIB 2017 benchmark instances, on a 2-core cloud
container CPU (not a benchmark-grade machine: use for float32-vs-float64 comparisons, not for
absolute speed).

| file | what |
|---|---|
| `miplib2017_benchmark_sizes.csv` | rows, cols, nonzeros, density for all 240 benchmark instances |
| `2026-10-25_miplib79_cpu-float64.csv` | 79 smallest instances, CPU float64 sparse, tol 1e-4, presolve, 30 s limit |
| `2026-10-25_miplib79_cpu-float32.csv` | same, float32 sparse (isolates precision) |
| `2026-10-25_miplib79_emulate-neuron_pre-fix.csv` | same, float32 dense, run BEFORE the infeasibility-streak and float32 retry-floor fixes (shows the cost266-UUE false "infeasible") |

| `2026-10-25_batch_throughput_strong_branching.csv` | seconds per LP vs batch size (1/8/32/128), strong-branching batches on 4 MIPLIB instances, cpu / cpu-float32 / emulate-neuron |
| `batch_throughput.py` | the script that produced it |
| `2026-10-25_stream_vs_fixed.csv` | continuous batching vs back-to-back fixed batches, 128 strong-branching LPs, 8 and 32 slots |
| `stream_vs_fixed.py` | the script that produced it |
| `2026-10-25_kasai_float64_finish.md` | batch float64 finish on neos-3627168-kasai's 32 branches |

| `mac_Apple_M4_2026-10-26.csv` | `examples/mac_benchmark.py` on a base M4 MacBook Pro (16 GB): single-LP µs/iteration and strong-branching batches, CPU vs Mac GPU |
| `mac_single_Apple_M4_2026-10-26.csv` | `examples/mac_single_lp.py`: one big LP at a time, CPU vs GPU + Metal kernel (µs/iteration, full solves vs HiGHS) |
| `strong_branching_Apple_M4_2026-10-26.csv` | `examples/strong_branching_benchmark.py`: full strong branching, HiGHS warm dual simplex on 10 cores vs trnopt CPU / Mac GPU, objective errors and pick quality |
| `2026-10-26_halpern_vs_pdlp_miplib79.csv` | PDLP vs reflected Halpern PDHG (gamma 1 and 0.5) on 79 MIPLIB relaxations, float32, tol 1e-4, 30 s, no presolve (2-core VM) |

MIPLIB instances themselves are not included; download from https://miplib.zib.de.
