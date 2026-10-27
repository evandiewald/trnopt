"""Random MIP families used in machine-learning-for-MIP research, at published benchmark sizes.

These four families come from Gasse et al. (2019), "Exact Combinatorial Optimization with Graph
Convolutional Neural Networks", and are the strong-branching benchmark in NVIDIA's batched-PDLP
paper (Blin, Maes, Gualandi, Lodi, Stellato, arXiv 2601.21990, Table 1). Generating them here
lets us compare against that paper at the same sizes.

Each function returns an LP whose `integer` mask marks the variables a MIP needs to be whole
numbers; the LP solver works on the relaxation.

What each family looks like, in plain terms:

* set_cover:          pick the cheapest set of "columns" so that every "row" is covered at least once.
* combinatorial_auction: accept bids on bundles of items to maximize revenue, each item sold at most once.
* independent_set:    pick as many nodes of a graph as possible with no two picked nodes adjacent.
* facility_location:  decide which warehouses to open (fixed cost) and how to serve customers'
                      demand from open ones (shipping cost), within warehouse capacities.

Fidelity: set cover, independent set and facility location follow the published generators'
recipes (sizes, distributions and formulations). combinatorial_auction is a simplified version of
Leyton-Brown's "arbitrary relationships" generator (same structure -- bundles of related items,
prices from item values plus synergy -- without every refinement).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .lp import LP


def set_cover(n_rows: int = 1000, n_cols: int = 1000, density: float = 0.05, max_cost: int = 100,
              seed: int = 0) -> LP:
    """Balas & Ho (1980) set cover: min c @ x  s.t.  A @ x >= 1, x binary. A is a random 0/1 matrix."""
    rng = np.random.default_rng(seed)
    nnz = int(n_rows * n_cols * density)
    # Every column covers at least one row and every row is coverable, as in the reference generator.
    cols = np.concatenate([np.arange(n_cols), rng.integers(0, n_cols, nnz - n_cols)])
    rows = np.concatenate([rng.permutation(np.resize(np.arange(n_rows), n_cols)),
                           rng.integers(0, n_rows, nnz - n_cols)])
    A = sp.csr_matrix((np.ones(nnz), (rows, cols)), shape=(n_rows, n_cols))
    A.data[:] = 1.0                                    # duplicates summed -> back to 0/1
    c = rng.integers(1, max_cost + 1, n_cols).astype(float)
    return LP(c=c, K=A, q=np.ones(n_rows), m_eq=0, lower=np.zeros(n_cols), upper=np.ones(n_cols),
              name=f"setcover_{n_rows}x{n_cols}_s{seed}", integer=np.ones(n_cols, dtype=bool))


def combinatorial_auction(n_items: int = 100, n_bids: int = 500, seed: int = 0) -> LP:
    """Winner determination: max sum(price_b x_b)  s.t.  each item in at most one accepted bid, x binary.

    Simplified Leyton-Brown "arbitrary relationships": items have private values and pairwise
    compatibilities; each bid grows a bundle by repeatedly adding an item compatible with what it
    already holds, and is priced at the bundle's value plus a synergy bonus, with noise.
    """
    rng = np.random.default_rng(seed)
    values = rng.uniform(1, 100, n_items)
    compat = rng.uniform(0, 1, (n_items, n_items))
    compat = (compat + compat.T) / 2
    np.fill_diagonal(compat, 0)
    rows, cols, prices = [], [], []
    for b in range(n_bids):
        bundle = [int(rng.integers(n_items))]
        while rng.uniform() < 0.82 and len(bundle) < n_items:      # mean bundle size ~5.5
            w = compat[bundle].mean(0)
            w[bundle] = 0
            bundle.append(int(rng.choice(n_items, p=w / w.sum())))
        synergy = 1 + 0.2 * (len(bundle) - 1) ** 1.2 / len(bundle)
        prices.append(values[bundle].sum() * synergy * rng.uniform(0.9, 1.1))
        rows += bundle
        cols += [b] * len(bundle)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_items, n_bids))
    lp = LP.from_general(c=np.array(prices), A_ub=A, b_ub=np.ones(n_items), lower=0.0, upper=1.0,
                         maximize=True, name=f"cauction_{n_items}x{n_bids}_s{seed}")
    lp.integer = np.ones(n_bids, dtype=bool)
    return lp


def _barabasi_albert(n: int, affinity: int, rng) -> set[tuple[int, int]]:
    edges = {(i, j) for i in range(affinity + 1) for j in range(i)}
    degree = np.zeros(n)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    for v in range(affinity + 1, n):
        targets = rng.choice(v, size=affinity, replace=False, p=degree[:v] / degree[:v].sum())
        for t in targets:
            edges.add((v, int(t)))
            degree[v] += 1
            degree[t] += 1
    return edges


def independent_set(n_nodes: int = 1000, affinity: int = 4, seed: int = 0) -> LP:
    """Maximum independent set on a Barabasi-Albert graph, clique formulation.

    "Adjacent nodes can't both be picked" could be one constraint per edge, but that LP relaxation
    is useless (every x = 1/2 is optimal). As in the reference generator, we instead cover the
    edges with cliques found greedily: each clique gets one constraint "at most one of these".
    """
    rng = np.random.default_rng(seed)
    remaining = _barabasi_albert(n_nodes, affinity, rng)
    adj = [set() for _ in range(n_nodes)]
    for i, j in remaining:
        adj[i].add(j)
        adj[j].add(i)
    cliques = []
    while any(adj):
        v = max(range(n_nodes), key=lambda u: len(adj[u]))
        clique = [v]
        for u in sorted(adj[v], key=lambda u: -len(adj[u])):
            if all(u in adj[w] for w in clique):
                clique.append(u)
        for a in clique:                          # those edges are now covered
            for b in clique:
                adj[a].discard(b)
        cliques.append(clique)
    rows = [r for r, cl in enumerate(cliques) for _ in cl]
    cols = [v for cl in cliques for v in cl]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cliques), n_nodes))
    lp = LP.from_general(c=np.ones(n_nodes), A_ub=A, b_ub=np.ones(len(cliques)), lower=0.0, upper=1.0,
                         maximize=True, name=f"indset_{n_nodes}_a{affinity}_s{seed}")
    lp.integer = np.ones(n_nodes, dtype=bool)
    return lp


def facility_location(n_customers: int = 400, n_facilities: int = 200, ratio: float = 5.0,
                      seed: int = 0) -> LP:
    """Capacitated facility location (Cornuejols, Sridharan & Thizy 1991), as in Gasse et al.

    Variables: x_ij in [0, 1] = share of customer i's demand served by facility j (continuous),
               y_j in {0, 1}  = open facility j.
    minimize   sum_ij ship_ij x_ij + sum_j fixed_j y_j
    subject to sum_j x_ij >= 1                       every customer fully served
               sum_i demand_i x_ij <= cap_j y_j      capacity, only if open
               sum_j cap_j y_j >= sum_i demand_i     enough capacity overall (redundant, helps the LP)
               x_ij <= y_j                           can't use a closed facility (tightening)
    Sizes: n = C*F + F variables, m = C + F + 1 + C*F rows, 4*C*F + 2*F nonzeros.
    """
    rng = np.random.default_rng(seed)
    C, F = n_customers, n_facilities
    cx, cy = rng.uniform(size=C), rng.uniform(size=C)
    fx, fy = rng.uniform(size=F), rng.uniform(size=F)
    demand = rng.integers(5, 36, C).astype(float)
    cap = rng.integers(10, 161, F).astype(float)
    fixed = (rng.integers(100, 111, F) * np.sqrt(cap) + rng.integers(0, 91, F)).astype(float)
    cap = cap * ratio * demand.sum() / cap.sum()
    dist = np.sqrt((cx[:, None] - fx[None, :]) ** 2 + (cy[:, None] - fy[None, :]) ** 2)
    ship = dist * 10 * demand[:, None]                  # (C, F)

    xi = lambda i, j: i * F + j                         # x_ij column index; y_j is C*F + j
    yj = lambda j: C * F + j
    n = C * F + F
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    ii, jj = np.meshgrid(np.arange(C), np.arange(F), indexing="ij")
    # Every row stored as ">=" (the LP class's format).
    # demand: sum_j x_ij >= 1
    rows.append(ii.ravel()); cols.append(xi(ii, jj).ravel()); vals.append(np.ones(C * F)); rhs.append(np.ones(C))
    r += C
    # capacity: cap_j y_j - sum_i d_i x_ij >= 0
    rows.append(r + jj.ravel()); cols.append(xi(ii, jj).ravel()); vals.append(-np.repeat(demand, F))
    rows.append(r + np.arange(F)); cols.append(yj(np.arange(F))); vals.append(cap); rhs.append(np.zeros(F))
    r += F
    # total capacity: sum_j cap_j y_j >= sum_i d_i
    rows.append(np.full(F, r)); cols.append(yj(np.arange(F))); vals.append(cap); rhs.append([demand.sum()])
    r += 1
    # tightening: y_j - x_ij >= 0
    k = np.arange(C * F)
    rows.append(r + k); cols.append(xi(ii, jj).ravel()); vals.append(-np.ones(C * F))
    rows.append(r + k); cols.append(yj(jj.ravel())); vals.append(np.ones(C * F)); rhs.append(np.zeros(C * F))
    r += C * F
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, n))
    c = np.concatenate([ship.ravel(), fixed])
    integer = np.zeros(n, dtype=bool)
    integer[C * F:] = True
    return LP(c=c, K=K, q=np.concatenate([np.asarray(v, float) for v in rhs]), m_eq=0,
              lower=np.zeros(n), upper=np.ones(n), name=f"facility_{C}x{F}_s{seed}", integer=integer)


# Sizes spanning arXiv 2601.21990 Table 1 (smallest and largest instance of each family there).
PAPER_SIZES = {
    "setcover": [dict(n_rows=1000, n_cols=1000, density=0.01), dict(n_rows=3000, n_cols=3000)],
    "cauction": [dict(n_items=360, n_bids=1400), dict(n_items=1025, n_bids=2867)],
    "indset": [dict(n_nodes=1000, affinity=10), dict(n_nodes=3000, affinity=12)],
    "facility": [dict(n_customers=400, n_facilities=200), dict(n_customers=1200, n_facilities=200)],
}
GENERATORS = {"setcover": set_cover, "cauction": combinatorial_auction, "indset": independent_set,
              "facility": facility_location}
