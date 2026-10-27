"""A small branch-and-bound MIP solver, split the way cuOpt splits it.

* The exact core runs on the CPU. Every node's LP is solved by HiGHS dual simplex, warm-started
  from its parent's basis (a child differs from its parent by one bound, so this takes a handful of
  pivots). Node bounds, pruning and the optimality proof only ever use these exact LP results.
* The accelerator-friendly work is batched and approximate: strong branching (score every
  fractional candidate by solving both of its child LPs) with trnopt's batched PDLP. Approximate
  answers are fine here, because they only decide WHICH variable to branch on, never what is proven.

How branch-and-bound works, in brief
------------------------------------
Solve the LP relaxation (drop the whole-number rules). If its answer happens to be whole, done.
Otherwise pick a variable with a fractional value, say x_j = 2.4, and split the problem in two:
x_j <= 2 and x_j >= 3. Each half's LP value is a lower bound for every solution in that half
(minimizing), so a half whose bound is already worse than the best whole-number solution found
(the incumbent) can be discarded. Keep splitting the most promising open half until the best open
bound meets the incumbent: then the incumbent is proven optimal.

Which variable to split on decides how big the tree gets:
* "most_fractional": the variable closest to .5. Free to compute, often a poor choice.
* "strong_highs":    try both halves of every candidate with HiGHS (exact, warm-started, all cores),
                     pick the one whose halves raise the bound the most. Small trees, costly nodes.
* "strong_trnopt":   the same scores from trnopt's batched PDLP (float32 by default, on whatever
                     backend you choose). Same small trees, if the scores agree; cheaper nodes, if
                     the batched solves are faster.

Not here (on purpose): presolve, cutting planes, conflict analysis, symmetry handling -- the
machinery that makes HiGHS, Gurobi and cuOpt fast at MIP overall. This solver exists to measure
what accelerated branching does to an otherwise fixed search, not to compete with them.
"""

from __future__ import annotations

import heapq
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import highspy
import numpy as np

from .backend import Backend
from .lp import LP
from .pdlp import PDLPOptions
from .reference import to_highs_lp

INT_TOL = 1e-6


@dataclass
class MIPResult:
    status: str                  # "optimal", "infeasible", "time_limit", "node_limit"
    objective: float             # incumbent objective (inf if none found)
    bound: float                 # proven lower bound
    gap: float                   # (objective - bound) / max(1, |objective|)
    x: np.ndarray | None
    nodes: int
    seconds: float
    times: dict = field(default_factory=dict)        # seconds per component
    counts: dict = field(default_factory=dict)       # LPs solved per component
    incumbents: list = field(default_factory=list)   # (seconds, objective) each time it improved


@dataclass(order=True)
class _Node:
    bound: float
    neg_depth: int
    seq: int
    lower: np.ndarray = field(compare=False)
    upper: np.ndarray = field(compare=False)
    basis: object = field(compare=False, default=None)
    lp_result: tuple | None = field(compare=False, default=None)   # pre-solved by strong branching
    branched: tuple | None = field(compare=False, default=None)    # (var, 0 down / 1 up, distance moved, parent obj)


class _NodeLP:
    """HiGHS holding the model; solve(bounds, warm-start basis) -> (status, obj, x, y, basis)."""

    def __init__(self, lp: LP):
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("solver", "simplex")
        self.h.setOptionValue("simplex_strategy", 1)      # dual simplex
        self.h.setOptionValue("threads", 1)
        self.h.passModel(to_highs_lp(lp))
        self.idx = np.arange(lp.n, dtype=np.int32)
        self.n = lp.n

    def solve(self, lower, upper, basis=None):
        h = self.h
        h.changeColsBounds(self.n, self.idx, _hinf(lower), _hinf(upper))
        if basis is not None:
            h.setBasis(basis)
        h.run()
        st = h.getModelStatus()
        if st == highspy.HighsModelStatus.kInfeasible:
            return "infeasible", np.inf, None, None, None
        if st != highspy.HighsModelStatus.kOptimal:
            return h.modelStatusToString(st), np.inf, None, None, None
        sol = h.getSolution()
        return ("optimal", h.getInfo().objective_function_value, np.array(sol.col_value),
                np.array(sol.row_dual), h.getBasis())


def _hinf(v):
    return np.where(np.isinf(v), np.sign(v) * highspy.kHighsInf, v)


# ---- strong branching with HiGHS, one child LP per task, all cores --------------------------
_W = None


def _w_init(lp):
    global _W
    _W = _NodeLP(lp)


def _w_children(task):
    lower, upper, basis_cols, basis_rows, children = task
    b = highspy.HighsBasis()
    b.col_status = [highspy.HighsBasisStatus(int(s)) for s in basis_cols]
    b.row_status = [highspy.HighsBasisStatus(int(s)) for s in basis_rows]
    b.valid = True
    out = []
    for j, lo, up in children:
        L, U = lower.copy(), upper.copy()
        L[j], U[j] = lo, up
        st, obj, x, y, basis = _W.solve(L, U, b)
        out.append((st, obj, x, None if basis is None else
                    ([int(s) for s in basis.col_status], [int(s) for s in basis.row_status])))
    return out


def _basis_lists(basis):
    return [int(s) for s in basis.col_status], [int(s) for s in basis.row_status]


def _basis_from_lists(cols, rows):
    b = highspy.HighsBasis()
    b.col_status = [highspy.HighsBasisStatus(s) for s in cols]
    b.row_status = [highspy.HighsBasisStatus(s) for s in rows]
    b.valid = True
    return b


def solve_mip(lp: LP, branching: str = "strong_trnopt", time_limit: float = 600.0, gap_tol: float = 1e-4,
              node_limit: int = 1_000_000, max_candidates: int | None = None, strong_depth: int | None = None,
              backend: str | Backend = "cpu-float32", slots: int = 128, strong_tol: float = 1e-4,
              highs_workers: int = 1, verify_top: int = 0, reliability: int = 4, strong_iters: int = 256,
              learn_from_estimates: bool = True,
              verbose: bool = False) -> MIPResult:
    """Branch-and-bound on lp (lp.integer marks the integer variables). See the module docstring.

    max_candidates: strong-branch on at most this many fractional variables (most fractional first);
                    None = all of them ("full strong branching").
    strong_depth:   strong-branch only at nodes up to this depth, most-fractional below it; None = everywhere.
    highs_workers:  processes for strong_highs (give it the machine's cores for a fair comparison).
    strong_iters:   trnopt engine: PDHG iterations per strong-branching LP (warm-started from the parent).
                    Strong branching only needs a RANKING, so like dual simplex with a pivot limit (the
                    usual practice) we stop early and score from the estimate. 256 picked the same variable
                    as exact scores on every test family; full convergence cost ~10x more for no gain.
    reliability:    reliability branching: a variable's pseudocost counts as reliable after this many
                    observations in each direction; until then it is strong-branched.
    verify_top:     trnopt engine only: re-solve the children of trnopt's top-k candidates exactly with HiGHS
                    and pick the best by exact score ("screening"). trnopt's float32 1e-4 answers rank
                    candidates well at the root, but deeper in the tree the gains between candidates shrink
                    to the size of the approximation error; confirming a short list fixes that cheaply.
    """
    if lp.integer is None or not lp.integer.any():
        raise ValueError("lp.integer marks no integer variables")
    rules = {"most_fractional": ("most_fractional", None), "pseudocost": ("pseudocost", None),
             "strong_highs": ("strong", "highs"), "strong_trnopt": ("strong", "trnopt"),
             "reliability_highs": ("reliability", "highs"), "reliability_trnopt": ("reliability", "trnopt")}
    if branching not in rules:
        raise ValueError(f"branching must be one of {sorted(rules)}")
    rule, engine = rules[branching]
    t0 = time.perf_counter()
    be = Backend.preset(backend) if isinstance(backend, str) else backend
    ints = np.flatnonzero(lp.integer)
    times = dict(node_lp=0.0, strong=0.0, heuristic=0.0)
    counts = dict(node_lp=0, strong_lps=0, strong_nodes=0, verify_lps=0, top1_confirmed=0)
    incumbent, x_inc, inc_log = np.inf, None, []
    core = _NodeLP(lp)
    pool = None
    if engine == "highs" and highs_workers > 1:
        pool = ProcessPoolExecutor(highs_workers, initializer=_w_init, initargs=(lp,))
    seq = itertools.count()
    # Pseudocosts: average bound gain per unit of distance moved, per variable and direction.
    pc_sum, pc_n = np.zeros((2, lp.n)), np.zeros((2, lp.n))

    def elapsed():
        return time.perf_counter() - t0

    def try_incumbent(x, obj, how):
        nonlocal incumbent, x_inc
        if obj < incumbent - 1e-9:
            incumbent, x_inc = obj, x.copy()
            inc_log.append((elapsed(), obj))
            if verbose:
                print(f"  {elapsed():7.2f}s  new incumbent {obj:.6g} ({how})")

    def feasible(x):
        """Whole numbers where required, within bounds, and every row satisfied (to 1e-6, relative)."""
        if np.any(np.abs(x[ints] - np.round(x[ints])) > INT_TOL):
            return False
        Kx = lp.K @ x
        viol = np.where(np.arange(lp.m) < lp.m_eq, np.abs(Kx - lp.q), np.maximum(lp.q - Kx, 0))
        return bool(np.all(viol <= 1e-6 * (1 + np.abs(lp.q))))

    def heuristic(node, x):
        """Round the integer variables and repair the continuous ones with one LP (fix-and-solve)."""
        t = time.perf_counter()
        L, U = node.lower.copy(), node.upper.copy()
        r = np.clip(np.round(x[ints]), L[ints], U[ints])
        L[ints] = U[ints] = r
        st, obj, xr, _, _ = core.solve(L, U, None)
        counts["node_lp"] += 1
        if st == "optimal" and feasible(xr):
            try_incumbent(xr, obj, "rounding")
        times["heuristic"] += time.perf_counter() - t

    def solve_node(node):
        if node.lp_result is not None:
            return node.lp_result
        t = time.perf_counter()
        res = core.solve(node.lower, node.upper, node.basis)
        times["node_lp"] += time.perf_counter() - t
        counts["node_lp"] += 1
        return res

    def strong_scores(node, x, y, basis, cand, depth):
        """Product scores for each candidate; also returns exact child results when HiGHS computed them."""
        down_up = []
        for j in cand:
            down_up.append((int(j), node.lower[j], np.floor(x[j])))
            down_up.append((int(j), np.ceil(x[j]), node.upper[j]))
        t = time.perf_counter()
        counts["strong_lps"] += len(down_up)
        counts["strong_nodes"] += 1
        exact = None
        if engine == "highs":
            if pool is None:
                objs, exact = [], []
                for j, lo, up in down_up:
                    L, U = node.lower.copy(), node.upper.copy()
                    L[j], U[j] = lo, up
                    r = core.solve(L, U, basis)
                    objs.append(r[1])
                    exact.append(r)
            else:
                bc, br = _basis_lists(basis)
                chunks = np.array_split(np.arange(len(down_up)), highs_workers * 2)
                tasks = [(node.lower, node.upper, bc, br, [down_up[i] for i in ch]) for ch in chunks if len(ch)]
                exact = []
                for part in pool.map(_w_children, tasks):
                    for st, obj, xc, bl in part:
                        exact.append((st, obj, xc, None, None if bl is None else _basis_from_lists(*bl)))
                objs = [r[1] for r in exact]
            infeasible = np.array([r[0] == "infeasible" for r in exact])
        else:
            from .batch import solve_pdlp_stream
            # One extra LP: the parent itself (its own bounds), so gains are measured approximate-vs-
            # approximate and the solver's systematic error largely cancels.
            j0 = int(down_up[0][0])
            down_up_p = down_up + [(j0, node.lower[j0], node.upper[j0])]
            var = np.array([d[0] for d in down_up_p])
            nlo = np.array([d[1] for d in down_up_p])
            nup = np.array([d[2] for d in down_up_p])

            def rows(base, new):
                def get(ids):
                    ids = np.asarray(ids)
                    out = np.repeat(base[None, :], len(ids), 0)
                    out[np.arange(len(ids)), var[ids]] = new[ids]
                    return out
                return get

            r = solve_pdlp_stream(lp, len(down_up_p), min(slots, len(down_up_p)), lower=rows(node.lower, nlo),
                                  upper=rows(node.upper, nup), x0=x, y0=y,
                                  opts=PDLPOptions(tol=strong_tol, backend=be, max_iters=strong_iters, time_limit=300))
            infeasible = np.array([s == "infeasible" for s in r.status[:-1]])
            objs = list(np.where(infeasible, np.inf, np.asarray(r.objective[:-1]) - r.objective[-1] + node.bound))
        times["strong"] += time.perf_counter() - t
        parent = node.bound
        big = 1e6 * (1 + abs(parent))
        gains = np.where(infeasible, big, np.maximum(np.array(objs, dtype=float) - parent, 0.0))
        score = np.maximum(gains[0::2], 1e-6) * np.maximum(gains[1::2], 1e-6)
        # Every strong-branching result also teaches the pseudocosts (approximate gains are fine here:
        # pseudocosts are estimates anyway).
        for k, j in enumerate(cand):
            if engine == "trnopt" and not learn_from_estimates:
                break
            for side, dist in ((0, x[j] - np.floor(x[j])), (1, np.ceil(x[j]) - x[j])):
                if not infeasible[2 * k + side] and dist > INT_TOL:
                    pc_sum[side, j] += gains[2 * k + side] / dist
                    pc_n[side, j] += 1
        return score, exact

    # ---- root ---------------------------------------------------------------------------------
    root = _Node(-np.inf, 0, next(seq), lp.lower.copy(), lp.upper.copy())
    heap = [root]
    nodes = 0
    status = "optimal"
    best_bound = -np.inf
    while heap:
        if elapsed() > time_limit:
            status = "time_limit"
            break
        if nodes >= node_limit:
            status = "node_limit"
            break
        node = heapq.heappop(heap)
        if node.bound >= incumbent - 1e-9 * max(1.0, abs(incumbent)):
            continue                                   # pruned: can't beat the incumbent
        st, obj, x, y, basis = solve_node(node)
        nodes += 1
        if node.branched is not None and st == "optimal":
            j_, side_, dist_, pobj_ = node.branched
            if dist_ > INT_TOL:
                pc_sum[side_, j_] += max(obj - pobj_, 0.0) / dist_
                pc_n[side_, j_] += 1
        if st != "optimal" or obj >= incumbent - 1e-9 * max(1.0, abs(incumbent)):
            continue                                   # infeasible, or bound too weak
        node.bound = max(node.bound, obj)
        frac = np.abs(x[ints] - np.round(x[ints]))
        fractional = ints[frac > INT_TOL]
        if len(fractional) == 0:
            try_incumbent(x, obj, "integral LP")
            continue
        if nodes == 1 or nodes % 20 == 0:
            heuristic(node, x)
        # gap check against the best open bound
        lb = min([node.bound] + [n.bound for n in heap])
        if incumbent < np.inf and (incumbent - lb) <= gap_tol * max(1.0, abs(incumbent)):
            best_bound = lb
            break

        depth = -node.neg_depth
        f = np.abs(x[fractional] - np.round(x[fractional]))
        order = fractional[np.argsort(-f, kind="stable")]
        exact = None
        if rule == "most_fractional" or (strong_depth is not None and depth > strong_depth and rule == "strong"):
            j, k = int(order[0]), None
        else:
            if rule == "strong":
                cand = order if max_candidates is None else order[:max_candidates]
            else:
                # Pseudocost estimate for every fractional variable; unknown pseudocosts use the average.
                known = pc_n > 0
                avg = np.array([pc_sum[d][known[d]].sum() / max(known[d].sum(), 1) if known[d].any() else 1.0
                                for d in (0, 1)])
                pc = np.where(known, pc_sum / np.maximum(pc_n, 1), avg[:, None])
                fd, fu = x[order] - np.floor(x[order]), np.ceil(x[order]) - x[order]
                est = np.maximum(pc[0, order] * fd, 1e-6) * np.maximum(pc[1, order] * fu, 1e-6)
                if rule == "pseudocost":
                    cand = np.array([], dtype=int)
                else:
                    unreliable = np.minimum(pc_n[0, order], pc_n[1, order]) < reliability
                    rank = np.argsort(-est, kind="stable")
                    cand = order[rank][unreliable[rank]]
                    if max_candidates is not None:
                        cand = cand[:max_candidates]
            if len(cand):
                score, exact = strong_scores(node, x, y, basis, cand, depth)
            if len(cand) and engine == "trnopt" and verify_top > 0 and len(cand) > 1:
                # Screening: confirm trnopt's short list with exact HiGHS solves.
                t = time.perf_counter()
                top = np.argsort(-score, kind="stable")[:verify_top]
                ex, ex_score = {}, []
                for kk in top:
                    jj = int(cand[kk])
                    pair = []
                    for lo, up in ((node.lower[jj], np.floor(x[jj])), (np.ceil(x[jj]), node.upper[jj])):
                        L, U = node.lower.copy(), node.upper.copy()
                        L[jj], U[jj] = lo, up
                        pair.append(core.solve(L, U, basis))
                    counts["verify_lps"] += 2
                    ex[int(kk)] = pair
                    g = [1e6 * (1 + abs(node.bound)) if r[0] == "infeasible" else max(r[1] - node.bound, 0.0) for r in pair]
                    ex_score.append(max(g[0], 1e-6) * max(g[1], 1e-6))
                    score[kk] = ex_score[-1]           # exact scores replace approximate ones
                counts["top1_confirmed"] += int(int(top[int(np.argmax(ex_score))]) == int(top[0]))
                exact = [None] * (2 * len(cand))
                for kk, pair in ex.items():
                    exact[2 * kk], exact[2 * kk + 1] = pair
                times["strong"] += time.perf_counter() - t
            if rule == "strong":
                k = int(np.argmax(score))
                j = int(cand[k])
            else:
                # Strong-branched candidates use their measured scores, the rest their estimates.
                total = est.copy()
                pos = {int(v): i for i, v in enumerate(order)}
                if len(cand):
                    for kk, v in enumerate(cand):
                        total[pos[int(v)]] = score[kk]
                i_best = int(np.argmax(total))
                j = int(order[i_best])
                k = None
                if len(cand) and j in set(int(v) for v in cand):
                    k = int(np.flatnonzero(cand == j)[0])
            if k is None or exact is None or exact[2 * k] is None:
                exact = None
        children = ((node.lower[j], np.floor(x[j])), (np.ceil(x[j]), node.upper[j]))
        for side, (lo, up) in enumerate(children):
            L, U = node.lower.copy(), node.upper.copy()
            L[j], U[j] = lo, up
            child = _Node(obj, node.neg_depth - 1, next(seq), L, U, basis,
                          branched=(j, side, abs(x[j] - (np.floor(x[j]) if side == 0 else np.ceil(x[j]))), obj))
            if exact is not None:                    # HiGHS already solved this child exactly: reuse it
                child.lp_result = exact[2 * k + side]
                child.branched = None                # strong branching already recorded this observation
            elif rule != "most_fractional" and rule != "pseudocost":
                # Solve the chosen children now (exactly, warm-started), so they enter the tree with their
                # true bounds -- as they do when HiGHS strong branching already solved them. Each child is
                # solved exactly once either way; this keeps the search identical across engines.
                t = time.perf_counter()
                child.lp_result = core.solve(L, U, basis)
                times["node_lp"] += time.perf_counter() - t
                counts["node_lp"] += 1
                if child.lp_result[0] == "optimal":
                    pc_sum[side, j] += max(child.lp_result[1] - obj, 0.0) / max(child.branched[2], INT_TOL)
                    pc_n[side, j] += 1
                child.branched = None
            if child.lp_result is not None:
                child.bound = max(obj, child.lp_result[1])
                if child.lp_result[0] != "optimal" or child.bound >= incumbent - 1e-9 * max(1.0, abs(incumbent)):
                    continue
            heapq.heappush(heap, child)
        if verbose and nodes % 50 == 0:
            lb = min([n.bound for n in heap], default=incumbent)
            print(f"  {elapsed():7.2f}s  nodes {nodes:6d}  open {len(heap):6d}  bound {lb:.6g}  incumbent {incumbent:.6g}")

    if pool is not None:
        pool.shutdown()
    if status == "optimal":
        best_bound = incumbent if not heap or best_bound == -np.inf else best_bound
        if incumbent == np.inf:
            status = "infeasible"
    else:
        best_bound = min([n.bound for n in heap], default=incumbent)
    gap = (incumbent - best_bound) / max(1.0, abs(incumbent)) if np.isfinite(incumbent) else np.inf
    return MIPResult(status, incumbent, best_bound, gap, x_inc, nodes, elapsed(), times, counts, inc_log)
