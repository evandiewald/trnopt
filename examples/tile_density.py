#!/usr/bin/env python3
"""Usage: python examples/tile_density.py [instance ...]   (default: the 11 MIPLIB instances in data/miplib)

How 'blocky' are the MIPLIB matrices? Tile the presolved constraint matrix into T x T blocks and count the
non-empty ones, under the original order and under a bandwidth-reducing reordering (reverse Cuthill-McKee on the
row/column bipartite graph)."""
import glob, numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from trnopt import load_mps
from trnopt.highs_presolve import HighsPresolve

def tiles(K, T):
    K = K.tocoo()
    key = (K.row // T).astype(np.int64) * ((K.shape[1] + T - 1) // T) + (K.col // T)
    return len(np.unique(key))

def rcm_perm(K):
    m, n = K.shape
    B = sp.bmat([[None, K], [K.T, None]]).tocsr()          # bipartite graph rows + cols
    p = reverse_cuthill_mckee(B, symmetric_mode=True)
    rp = p[p < m]; cp = p[p >= m] - m
    return rp, cp

def sell_dmas(K, S=128):
    # current kernel: one indexed gather per (128-row slice, ELL column) -> sum over slices of the longest row
    rl = np.diff(K.tocsr().indptr)
    pad = (-len(rl)) % S
    rl = np.concatenate([rl, np.zeros(pad, int)]).reshape(-1, S)
    return int(rl.max(axis=1).sum())

import sys
names = sys.argv[1:] or ["liu", "nu25-pr12", "neos-911970", "neos-4954672-berkel", "comp21-2idx", "neos-933966", "chromaticindex512-7",
         "neos-4300652-rahue", "polygonpack4-7", "neos-3355120-tarago", "bts4-cta"]
print(f"{'instance':20s} {'m x n':>15s} {'nnz':>8s} | {'T':>4s} {'order':6s} {'tiles':>8s} {'fill':>6s} {'stored/nnz':>10s} {'MB fp32':>8s} | {'gathers now (K)':>15s}")
for name in names:
    lp0 = load_mps(f"data/miplib/{name}.mps.gz")
    P = HighsPresolve(lp0, 60)
    lp = P.reduced if P.status == "reduced" else lp0
    K = sp.csr_matrix(lp.K); K.eliminate_zeros()
    m, n = K.shape
    rp, cp = rcm_perm(K)
    Kr = K[rp][:, cp]
    g = sell_dmas(K)
    for T in (128,):
        for label, M in (("orig", K), ("rcm", Kr)):
            t = tiles(M, T)
            stored = t * T * T
            print(f"{name[:20]:20s} {f'{m}x{n}':>15s} {K.nnz:8d} | {T:4d} {label:6s} {t:8d} {K.nnz / stored:6.3f} {stored / K.nnz:10.1f} {stored * 4 / 1e6:8.1f} | {g:15d}", flush=True)
