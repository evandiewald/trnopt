"""NKI kernel: Y = K @ X with K in sliced-ELL layout (see sell.py), X a batch of B vectors.

What the kernel does, in Trainium terms
---------------------------------------
Trainium's on-chip memory (SBUF) is organized as 128 partitions: parallel lanes, each with its
own slice of fast memory. Data lives in the chip's main memory (HBM) until a DMA engine copies it
into SBUF, where the compute engines can work on it.

For each slice of 128 rows:
  1. Copy the slice's column-index table and value table into SBUF (128 lanes x width).
  2. For each position k across the slice's width:
       - every lane fetches row  X[col[lane, k], :]  from HBM. That's ONE "indirect" DMA that
         moves 128 independently addressed rows at once (the gather);
       - every lane multiplies that row by its value val[lane, k] and adds it to its accumulator
         (the Vector engine, all 128 lanes in parallel).
  3. Copy the 128 accumulated rows back to HBM.

The Tensor (matrix-multiply) engine sits idle: a sparse multiply has no dense blocks for it.
What matters here is memory traffic, and batching is what makes each gather worthwhile: with
B = 1 a lane fetches 4 bytes per DMA descriptor; with B = 128 it fetches 512 contiguous bytes.

This version is written for clarity and correctness, checked in the NKI CPU simulator. Speed
tuning (overlapping the next gather with the current multiply, splitting B into chunks that
fit SBUF, spreading slices over both NeuronCores) needs real hardware to measure.

Requires neuronx-cc (AWS's pip repo). Targets trn1 by default when no device is present.
"""

from __future__ import annotations

import numpy as np

from .sell import LANES, SELL, combine

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    HAVE_NKI = True
except ImportError:  # the rest of trnopt works without it
    HAVE_NKI = False


def make_sell_kernel(widths: tuple[int, ...], offsets: tuple[int, ...]):
    """Build a kernel specialized to one matrix's slice layout.

    Slice widths differ, so they're baked in as compile-time constants. That's the normal
    Trainium trade: one compile per matrix structure, then every multiply reuses it (PDLP does
    hundreds of thousands of multiplies with the same K).
    """
    if not HAVE_NKI:
        raise ImportError("neuronx-cc is not installed; see the README for how to get it")
    num_slices = len(widths)

    @nki.jit
    def sell_matmul(cols, vals, X):
        # cols, vals: (128, total_width) in HBM.  X: (n + 1, B) in HBM, last row all zeros.
        B = X.shape[1]
        Y = nl.ndarray((num_slices * LANES, B), dtype=X.dtype, buffer=nl.shared_hbm)
        i_p = nl.arange(LANES)[:, None]          # lane (partition) index
        i_b = nl.arange(B)[None, :]              # position along the batch
        for s in nl.static_range(num_slices):
            w, o = widths[s], offsets[s]
            i_w = nl.arange(w)[None, :]
            col_tile = nl.load(cols[i_p, o + i_w])               # (128, w) indices into SBUF
            val_tile = nl.load(vals[i_p, o + i_w])               # (128, w) values into SBUF
            acc = nl.zeros((LANES, B), dtype=nl.float32, buffer=nl.sbuf)
            for k in nl.sequential_range(w):
                rows = nl.load(X[col_tile[i_p, k], i_b])         # gather: lane p gets X[col[p, k], :]
                acc[i_p, i_b] = nl.add(acc, nl.multiply(rows, val_tile[i_p, k]))
            nl.store(Y[s * LANES + i_p, i_b], value=acc)
        return Y

    return sell_matmul


def sell_matmul_simulated(S: SELL, X: np.ndarray) -> np.ndarray:
    """Run the kernel in NKI's CPU simulator and return K @ X (for testing without hardware)."""
    X = np.asarray(X, dtype=np.float32)
    squeeze = X.ndim == 1
    X2 = X.reshape(X.shape[0], -1)
    Xz = np.vstack([X2, np.zeros((1, X2.shape[1]), np.float32)])
    kernel = make_sell_kernel(S.widths, S.offsets)
    slot_out = nki.simulate_kernel(kernel, S.cols, S.vals.astype(np.float32), Xz)
    Y = combine(S, np.asarray(slot_out))
    return Y[:, 0] if squeeze else Y
