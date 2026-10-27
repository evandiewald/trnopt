"""Sliced ELL: a sparse-matrix layout shaped for Trainium's 128 lanes.

The problem it solves
---------------------
Trainium has no sparse-matrix hardware, and the PyTorch path on it doesn't support sparse tensors.
Storing K as a full grid wastes ~700x the necessary work on a typical MIPLIB instance (99.86% of
entries are zero). Storing only the non-empty 128x128 tiles still wastes ~175x, because MIP
nonzeros are scattered too thinly for any tiling to catch.

The layout
----------
1. Split any row with more than `split` nonzeros into pieces ("pseudo-rows") of at most `split`.
   A budget constraint touching every variable would otherwise force huge padding.
2. Sort pseudo-rows by length, longest first, and group them into slices of 128. 128 is the
   number of partitions (parallel lanes) in Trainium's on-chip memory: one lane per row.

3. Within a slice, pad each row to the slice's longest row. Store two tables of shape
   (128, total width): column indices (padding points at an all-zero row of x) and values.

Multiplying then means: for each slice, for each position k across its width, every lane fetches
x[col[lane, k]] (a "gather"), multiplies by val[lane, k] and adds. Finally, pseudo-rows are added
back into their original rows.

On MIPLIB 2017 (29-instance sample) this stores a median of 1.08x the true nonzeros (worst 1.4x
on large instances), versus 723x for a full grid.

With a batch of B vectors (X is n x B), each gather fetches a whole contiguous row of X: B
numbers in one go. That's the access pattern Trainium's DMA engines handle best, and the reason
batching (Phase 3) and this layout work well together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

LANES = 128


@dataclass
class SELL:
    shape: tuple[int, int]        # (m, n) of the original matrix
    cols: np.ndarray              # (128, total_width) int32; padding = n (the zero row appended to x)
    vals: np.ndarray              # (128, total_width) float
    widths: tuple[int, ...]       # width of each slice
    offsets: tuple[int, ...]      # starting column of each slice in cols / vals
    out_rows: np.ndarray          # (num_slices * 128,) original row of each lane slot, -1 for padding

    @property
    def num_slices(self) -> int:
        return len(self.widths)

    @property
    def slots(self) -> int:
        return LANES * sum(self.widths)


def choose_split(K: sp.spmatrix, candidates=(8, 16, 32, 64, 128, 256, None)) -> int | None:
    """Pick the row-split size for this matrix.

    Smaller splits cut padding (fewer wasted slots) but create more pseudo-rows, so more slices
    to loop over and more partial sums to add back. Score = slots + a fixed cost per slice.
    On Netlib's `shell` this takes padding from 9.8x the nonzeros (split 256) to about 1.1x.
    """
    K = sp.csr_matrix(K)
    lens = np.diff(K.indptr)
    best, best_cost = None, None
    for split in candidates:
        pl = lens if split is None else np.concatenate(
            [np.diff(np.minimum(np.arange(0, max(l, 1) + split, split), l))[: max(-(-l // split), 1)] for l in lens])
        pl = np.sort(pl)[::-1]
        pl = np.concatenate([pl, np.zeros(-len(pl) % LANES, int)]).reshape(-1, LANES)
        slices = pl.shape[0]
        cost = LANES * np.maximum(pl.max(axis=1), 1).sum() + LANES * 8 * slices
        if best_cost is None or cost < best_cost:
            best, best_cost = split, cost
    return best


def build_sell(K: sp.spmatrix, split: int | None | str = "auto", dtype=np.float32) -> SELL:
    K = sp.csr_matrix(K)
    K.sum_duplicates()
    if split == "auto":
        split = choose_split(K)
    m, n = K.shape
    indptr, indices, data = K.indptr, K.indices, K.data

    # 1. Pseudo-rows: (original row, start, end) into the CSR arrays.
    starts, ends, origin = [], [], []
    for i in range(m):
        a, b = indptr[i], indptr[i + 1]
        step = split if split else max(b - a, 1)
        for s in range(a, max(b, a + 1), step):          # empty rows still get one (empty) pseudo-row
            starts.append(s)
            ends.append(min(s + step, b))
            origin.append(i)
    starts, ends, origin = map(np.asarray, (starts, ends, origin))
    lengths = ends - starts

    # 2. Longest first, in slices of 128.
    order = np.argsort(-lengths, kind="stable")
    num_slices = -(-len(order) // LANES)
    padded = np.full(num_slices * LANES, -1)
    padded[: len(order)] = order
    slot_len = np.where(padded >= 0, lengths[np.maximum(padded, 0)], 0).reshape(num_slices, LANES)
    widths = np.maximum(slot_len.max(axis=1), 1)
    offsets = np.concatenate([[0], np.cumsum(widths)[:-1]])

    # 3. Fill the two tables.
    total = int(widths.sum())
    cols = np.full((LANES, total), n, dtype=np.int32)
    vals = np.zeros((LANES, total), dtype=dtype)
    out_rows = np.full(num_slices * LANES, -1, dtype=np.int64)
    for slot, pr in enumerate(padded):
        if pr < 0:
            continue
        s, lane = divmod(slot, LANES)
        a, b = starts[pr], ends[pr]
        o = offsets[s]
        cols[lane, o : o + (b - a)] = indices[a:b]
        vals[lane, o : o + (b - a)] = data[a:b]
        out_rows[slot] = origin[pr]
    return SELL((m, n), cols, vals, tuple(int(w) for w in widths), tuple(int(o) for o in offsets), out_rows)


def sell_matmul_reference(S: SELL, X: np.ndarray) -> np.ndarray:
    """Y = K @ X computed exactly the way the kernel does it, in plain NumPy (for checking)."""
    X = np.asarray(X)
    squeeze = X.ndim == 1
    X = X.reshape(X.shape[0], -1)
    Xz = np.vstack([X, np.zeros((1, X.shape[1]), X.dtype)])     # row n = zeros, where padding points
    slot_out = np.zeros((S.num_slices * LANES, X.shape[1]), dtype=np.result_type(S.vals, X))
    for s, (w, o) in enumerate(zip(S.widths, S.offsets)):
        acc = np.zeros((LANES, X.shape[1]), dtype=slot_out.dtype)
        for k in range(w):
            acc += S.vals[:, o + k, None] * Xz[S.cols[:, o + k]]   # every lane gathers its own row of X
        slot_out[s * LANES : (s + 1) * LANES] = acc
    return combine(S, slot_out[:, 0] if squeeze else slot_out)


def combine(S: SELL, slot_out: np.ndarray) -> np.ndarray:
    """Add pseudo-rows back into their original rows (and drop padding lanes)."""
    keep = S.out_rows >= 0
    Y = np.zeros((S.shape[0],) + slot_out.shape[1:], dtype=slot_out.dtype)
    np.add.at(Y, S.out_rows[keep], slot_out[keep])
    return Y
