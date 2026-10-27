"""EXPERIMENT (kept for the record, not used by the solver): batch-on-partitions CSR SpMM on Trainium.

Idea: put the batch (up to 128 LPs) on SBUF's 128 partitions and X^T in SBUF, so every partition
gathers with the SAME column indices. That replaces per-row indirect DMAs (~2 us each on trn1) with
on-chip gathers (nisa.nc_n_gather on the GpSimd engine), and a resetting scan sums each row's
segment exactly, with no padding, sorting or row splitting.

Result on trn1.2xlarge (2026-10-27): correct (rel err <= 3e-7) but 4-10x SLOWER than the sliced-ELL
kernel in src/trnopt/nki_sell.py (nu25-pr12 K, B=128: 4,395 us vs 404 us). Profile
(results/profiles/csr_t_nu25_K_B127_summary.txt): GpSimd busy 4.25 ms of 6.4 ms; the Vector engine
did all multiplies and the scan in 66 us. The math was cheap; the gather engine was the bottleneck.

Usage on a Neuron instance:  python examples/nki_batch_on_partitions_experiment.py hw   (or: sim)
"""
import sys, time
import numpy as np, scipy.sparse as sp
import nki, nki.isa as nisa, nki.language as nl

EC = 4096          # nonzeros per chunk (chunks end on row boundaries)

def prepare(K):
    K = sp.coo_matrix(K, dtype=np.float32); m, n = K.shape
    rows, cols, vals = K.row, K.col, K.data
    empty = np.setdiff1d(np.arange(m), rows)          # empty rows get one explicit zero, so every row has an end
    rows = np.concatenate([rows, empty]); cols = np.concatenate([cols, np.zeros(len(empty), cols.dtype)])
    vals = np.concatenate([vals, np.zeros(len(empty), np.float32)])
    order = np.lexsort((cols, rows)); rows, cols, vals = rows[order], cols[order], vals[order]
    ip = np.searchsorted(rows, np.arange(m + 1))       # CSR row pointers, explicit zeros kept
    E = len(rows)
    mask = np.ones(E, np.float32); mask[ip[:-1]] = 0.0  # 0 where a row starts: reset the running sum
    chunks, e0, r0 = [], 0, 0
    while r0 < m:
        r1 = r0
        while r1 < m and ip[r1 + 1] - e0 <= EC: r1 += 1
        if r1 == r0: raise ValueError(f"row {r0} has more than {EC} nonzeros")
        chunks.append((int(e0), int(ip[r1]), int(r0), int(r1))); e0, r0 = ip[r1], r1
    ends = ip[1:] - 1
    ends_local = np.empty(m, np.uint32)
    for e0, e1, r0, r1 in chunks: ends_local[r0:r1] = ends[r0:r1] - e0
    return dict(shape=(m, n), cols=cols.astype(np.uint32)[None], vals=vals.astype(np.float32)[None],
                mask=mask[None], ends=ends_local[None], chunks=chunks)

def make_kernel(Pr, P):
    m, n = Pr["shape"]; chunks = Pr["chunks"]
    @nki.jit
    def csr_t(XT, cols, vals, mask, ends):
        YT = nl.ndarray((P, m), dtype=nl.float32, buffer=nl.shared_hbm)
        xt = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=xt, src=XT)
        yt = nl.ndarray((P, m), dtype=nl.float32, buffer=nl.sbuf)
        for e0, e1, r0, r1 in chunks:
            ec, rc = e1 - e0, r1 - r0
            ci = nl.ndarray((P, ec), dtype=nl.uint32, buffer=nl.sbuf)
            cv = nl.ndarray((P, ec), dtype=nl.float32, buffer=nl.sbuf)
            cm = nl.ndarray((P, ec), dtype=nl.float32, buffer=nl.sbuf)
            ce = nl.ndarray((P, rc), dtype=nl.uint32, buffer=nl.sbuf)
            # the same row of indices/values on every partition: partition stride 0 (a broadcast DMA)
            nisa.dma_copy(dst=ci, src=cols.ap(pattern=[[0, P], [1, ec]], offset=e0))
            nisa.dma_copy(dst=cv, src=vals.ap(pattern=[[0, P], [1, ec]], offset=e0))
            nisa.dma_copy(dst=cm, src=mask.ap(pattern=[[0, P], [1, ec]], offset=e0))
            nisa.dma_copy(dst=ce, src=ends.ap(pattern=[[0, P], [1, rc]], offset=r0))
            g = nl.ndarray((P, ec), dtype=nl.float32, buffer=nl.sbuf)
            nisa.nc_n_gather(dst=g, data=xt, indices=ci)                      # g[b, e] = X^T[b, col[e]]
            nisa.tensor_tensor(dst=g, data1=g, data2=cv, op=nl.multiply)      # times the matrix value
            s = nl.ndarray((P, ec), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(dst=s, data0=cm, data1=g, initial=0.0, op0=nl.multiply, op1=nl.add)
            nisa.nc_n_gather(dst=yt[:, r0:r1], data=s, indices=ce)            # each row total sits at its last entry
        nisa.dma_copy(dst=YT, src=yt)
        return YT
    return csr_t

if __name__ == "__main__":
    mode = sys.argv[1]
    rng = np.random.default_rng(0)
    K = sp.random(300, 500, density=0.02, random_state=1, format="csr")
    K = sp.vstack([K, sp.csr_matrix((2, 500)), sp.csr_matrix(np.ones((1, 500)))]).tocsr()
    Pr = prepare(K)
    for P in (1, 7, 128):
        X = rng.normal(size=(K.shape[1], P)).astype(np.float32)
        args = (np.ascontiguousarray(X.T), Pr["cols"], Pr["vals"], Pr["mask"], Pr["ends"])
        k = make_kernel(Pr, P)
        if mode == "sim":
            YT = np.asarray(nki.simulate(k)(*args))
        else:
            import torch, torch_neuronx
            YT = k(*[torch.as_tensor(a).to("neuron") for a in args]).cpu().numpy()
        ref = (K @ X).T
        print(mode, "P", P, "chunks", len(Pr["chunks"]), "rel err %.2e" % (np.abs(YT - ref).max() / np.abs(ref).max()), flush=True)
