"""Run PDLP and HiGHS side by side and print a comparison table.

    uv run python examples/compare.py                              # default suite, CPU float64
    uv run python examples/compare.py --backend emulate-neuron     # Trainium's restrictions, on your CPU
    uv run python examples/compare.py --presolve                   # shrink each LP with HiGHS presolve first
    uv run python examples/compare.py --tol 1e-4 --verbose
"""

import argparse
from pathlib import Path

from trnopt import Backend, PDLPOptions, load_mps, solve_highs, solve_pdlp
from trnopt.problems import infeasible_lp, loan_selection, random_lp, textbook, unbounded_lp

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "neuron", "emulate-neuron"])
    ap.add_argument("--no-compile", action="store_true", help="turn off torch.compile even if the preset uses it")
    ap.add_argument("--tol", type=float, default=None, help="default: 1e-6 for float64 backends, 1e-4 for float32")
    ap.add_argument("--presolve", action="store_true")
    ap.add_argument("--max-iters", type=int, default=200_000)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--mps", nargs="*", help="extra MPS files to include")
    args = ap.parse_args()

    be = Backend.preset(args.backend, **({"compile": False} if args.no_compile else {}))
    tol = args.tol or (1e-6 if str(be.dtype) == "torch.float64" else 1e-4)

    problems = [textbook(), random_lp(), random_lp(n=2000, m_eq=300, m_ineq=600, density=0.005, seed=1), loan_selection()]
    problems += [infeasible_lp(), unbounded_lp()]
    problems += [load_mps(str(p)) for p in sorted(DATA.glob("*.mps"))]
    problems += [load_mps(p) for p in args.mps or []]

    opts = PDLPOptions(tol=tol, backend=be, presolve=args.presolve, max_iters=args.max_iters, verbose=args.verbose)
    print(f"backend={args.backend} dtype={be.dtype} dense={be.dense} compile={be.compile} tol={tol:g} presolve={args.presolve}\n")
    header = (f"{'problem':<22}{'rows':>6}{'cols':>6} | {'HiGHS':<11}{'objective':>15}{'sec':>6} | "
              f"{'PDLP':<16}{'objective':>15}{'iters':>8}{'sec':>7}{'obj err':>9}")
    print(header)
    print("-" * len(header))
    for lp in problems:
        ref = solve_highs(lp)
        res = solve_pdlp(lp, opts)
        both_optimal = ref.status == "Optimal" and res.status == "optimal"
        err = f"{abs(res.objective - ref.objective) / (1 + abs(ref.objective)):9.1e}" if both_optimal else f"{'':>9}"
        ref_obj = f"{ref.objective:15.8g}" if ref.status == "Optimal" else f"{'':>15}"
        res_obj = f"{res.objective:15.8g}" if res.status in ("optimal", "inaccurate") else f"{'':>15}"
        print(
            f"{lp.name[:22]:<22}{lp.m:>6}{lp.n:>6} | {ref.status:<11}{ref_obj}{ref.seconds:>6.2f} | "
            f"{res.status:<16}{res_obj}{res.iterations:>8}{res.seconds:>7.2f}{err}",
            flush=True,
        )


if __name__ == "__main__":
    main()
