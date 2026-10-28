"""NKI kernel: Y = K @ X with K in sliced-ELL layout (see sell.py), X a batch of B vectors.

What the kernel does, in Trainium terms
---------------------------------------
Trainium's on-chip memory (SBUF) is organized as 128 partitions: parallel lanes, each with its
own slice of fast memory. Data lives in the chip's main memory (HBM) until a DMA engine copies it
into SBUF, where the compute engines can work on it.

First copy the column-index and value tables (128 lanes x total width) into SBUF, one DMA each
(a few for very wide matrices, see TABLE_COLS). Then for each slice of 128 rows:
  1. Look up the slice's part of the tables (already on chip).
  2. For each position k across the slice's width, every lane fetches row X[col[lane, k], :] from
     HBM: ONE "indirect" DMA moves 128 independently addressed rows (the gather). These gathers
     write to separate SBUF regions, so they don't wait on each other or on the arithmetic.
  3. Multiply-accumulate: acc += val[:, k] * gathered[k] (Vector engine, all 128 lanes at once).
  4. Write the 128 results to Y. In ordered mode (make_sell_kernel(ordered=True), used by the batched
     solver) Y's rows come in the layout's own order, so a slice of whole rows writes one contiguous
     block, a plain DMA. Otherwise, or for slices holding pieces of split rows, scatter the 128 results
     straight to their rows of Y with one more indirect DMA. Split rows
     (see sell.py) arrive as several pieces, so for those matrices Y starts at zero and the
     scatter adds in place (dma_compute). A scatter-add needs distinct target rows, so a slice
     holding k pieces of one row scatters in k passes (pass j: each row's j-th piece; other
     lanes point out of bounds and are skipped). Only a few slices ever need more than one.

So one call does the whole multiply. An earlier version left the scatter-back and a zero-padding
row to PyTorch; on the chip each of those eager ops costs a round trip, and together they took
3-5x as long as the kernel itself.

What limits it (measured on trn1, see results/profiles/)
-------------------------------------------------------
Each indirect DMA (gather or scatter) costs ~2-3.5 us almost regardless of size, and the chip sits
idle ~80-93% of the kernel's run. On trn1 the descriptors for these DMAs are generated in software,
one at a time. Plain DMAs are far cheaper: copying the index tables slice by slice took ~0.4 us per
copy (results/profiles/chromaticindex512-7_*_B32_v1 vs _v2), so copying them up front saved only 4-5%;
replacing scatters with block writes saved ~1.7 us each (_v2 vs _v3, 10-14%). See
results/nki_dma_2026-10-28.md. So time tracks the number of gathers (total slice width), not the bytes: B=1 and B=128 cost
nearly the same. Batching is therefore what makes the kernel pay: at B=1024 it beats a dense
K @ X by ~2.4x on nu25-pr12. Two things that would cut the gather count are unavailable on trn1:
a single DMA gathering a whole slice (2D vector_offset) fails to compile for NeuronCore-v2, and
gathering on-chip with GpSimd (batch on partitions, nc_n_gather) was correct but 4-10x slower.

Requires the standalone `nki` package (Neuron SDK 2.29+; preinstalled in the Neuron DLAMI venvs).
"""

from __future__ import annotations

import hashlib

import numpy as np

from .sell import LANES, SELL

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    HAVE_NKI = True
except ImportError:  # the rest of trnopt works without it
    HAVE_NKI = False

SBUF_BUDGET = 48 * 1024        # bytes per partition for one chunk of gathered rows
# Index-table columns per partition copied to SBUF in one go (x 4 bytes x 2 tables = 32 KB). Copying
# each slice's tables separately cost two DMAs per slice: on short-row matrices (2-15 nonzeros a row)
# a third of the kernel's DMAs, but plain DMAs are cheap next to gathers, so the saving was 4-5%.
TABLE_COLS = 4096
KERNEL_VERSION = 3             # bump when the generated code changes: it's part of the kernel's name (see below)

_KERNELS: dict = {}            # make_sell_kernel's cache: (layout, B) -> kernel, for the life of the process


def output_order(S: SELL) -> tuple[np.ndarray, tuple[bool, ...]]:
    """Ordered mode's output order: (position of each of S's rows in the kernel's output, block flag per slice).

    A slice holding only whole rows (no pieces of split rows) writes its 128 results as one contiguous
    block, a plain DMA, instead of scattering them. Those slices come first, in slice order, lane by lane.
    The rows of the other slices (pieces of split rows, and whole rows sharing a slice with them) take the
    positions after, and are scattered there as before. Padding lanes only ever end the last slice.
    """
    m = S.shape[0]
    lanes = S.out_rows.reshape(-1, LANES)
    real, counts = np.unique(S.out_rows[S.out_rows >= 0], return_counts=True)
    split = np.zeros(m, dtype=bool)
    split[real[counts > 1]] = True
    block = tuple(bool(np.all((r < 0) | ~split[np.maximum(r, 0)])) for r in lanes)
    pos, p = np.full(m, -1, dtype=np.int64), 0
    for r, b in zip(lanes, block):
        if b:
            r = r[r >= 0]
            pos[r] = np.arange(p, p + len(r))
            p += len(r)
    rest = np.flatnonzero(pos < 0)
    pos[rest] = np.arange(p, p + len(rest))
    return pos, block


def scatter_passes(S: SELL, order=None) -> tuple[np.ndarray, tuple[int, ...]]:
    """Scatter targets, split into passes with distinct rows: (out (128, total_passes), passes per slice).

    Lane p of slice s, in its pass j, writes to row out[p, first_pass[s] + j]; every other pass of
    that slice has row m there (out of bounds, skipped). Padding lanes are m in every pass.
    With order = output_order(S): targets are output positions, and block slices have no passes.
    """
    m = S.shape[0]
    cols_out, passes = [], []
    for s in range(S.num_slices):
        rows = S.out_rows[s * LANES:(s + 1) * LANES]
        if order is not None:
            if order[1][s]:
                passes.append(0)
                continue
            rows = np.where(rows >= 0, order[0][np.maximum(rows, 0)], -1)
        rank = np.zeros(LANES, int)                 # how many earlier lanes in this slice share my row
        seen: dict[int, int] = {}
        for p, r in enumerate(rows):
            if r >= 0:
                rank[p] = seen.get(int(r), 0)
                seen[int(r)] = rank[p] + 1
        k = max(seen.values(), default=1)
        for j in range(k):
            cols_out.append(np.where((rows >= 0) & (rank == j), rows, m))
        passes.append(k)
    if not cols_out:                                # every slice writes a block: a placeholder column, unused
        cols_out.append(np.full(LANES, m))
    return np.ascontiguousarray(np.stack(cols_out, axis=1).astype(np.int32)), tuple(passes)


def kernel_tables(S: SELL, col_pos=None, ordered: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The kernel's inputs that describe K: (cols, vals, out_rows).

    Padding slots gather row 0 times value 0 (instead of an appended zero row), and padding
    lanes scatter to row m, which is out of bounds and skipped. col_pos renumbers the columns (X's rows
    are in that order); ordered: scatter targets are output_order positions (see make_sell_kernel).
    """
    n = S.shape[1]
    cols = np.where(S.cols >= n, 0, S.cols)
    if col_pos is not None:
        cols = np.asarray(col_pos)[cols]
    out = scatter_passes(S, output_order(S) if ordered else None)[0]
    return cols.astype(np.int32), S.vals.astype(np.float32), out


def layout_key(S: SELL, ordered: bool = False) -> tuple:
    """Everything about S that make_sell_kernel bakes into a kernel, apart from B (offsets follow from widths;
    in ordered mode, which slices write blocks follows from the passes, and their positions from the lane count,
    which only that mode bakes in)."""
    m, n = S.shape
    lanes = int((S.out_rows >= 0).sum())
    has_splits = len(np.unique(S.out_rows[S.out_rows >= 0])) < lanes
    passes = scatter_passes(S, output_order(S) if ordered else None)[1]
    return (m, n, S.widths, has_splits, passes, lanes if ordered else None, ordered)


def make_sell_kernel(S: SELL, B: int, ordered: bool = False):
    """Build a kernel specialized to one matrix's slice layout and one batch size B.

    Slice widths differ, so they're baked in as compile-time constants. That's the normal
    Trainium trade: one compile per (matrix, B), then every multiply reuses it (PDLP does
    hundreds of thousands of multiplies with the same K).

    Kernels are cached per process on everything baked in (shape, slice widths, scatter passes, B),
    so solving the same K again reuses the compiled kernel. The matrix's values and column
    indices are kernel inputs, so they aren't part of the key.

    ordered=True: Y comes out in output_order(S)'s row order instead of K's, so most slices write one
    contiguous block instead of scattering (a scatter is an indirect DMA, ~3.4 us on trn1; a plain write
    is far cheaper). The caller keeps its vectors in that order (MatVec(permute=True)).
    """
    if not HAVE_NKI:
        raise ImportError("the nki package is not installed; see the README for how to get it")
    widths, offsets, num_slices = S.widths, S.offsets, S.num_slices
    lk = layout_key(S, ordered)
    m, n, _, has_splits, passes, _, _ = lk
    lanes = int((S.out_rows >= 0).sum())
    key = lk + (B, TABLE_COLS)
    if key in _KERNELS:
        return _KERNELS[key]
    first_pass = tuple(int(x) for x in np.concatenate([[0], np.cumsum(passes)[:-1]]))
    total_passes = int(sum(passes))
    # Block slices (no passes; ordered mode only) write lanes 0..k-1 to Y[base:base + k]. All 128 lanes are
    # real except in the last slice. Scattered rows come after the blocks, so only they need zeroing.
    block = tuple(ordered and p == 0 for p in passes)
    real = tuple(LANES if s < num_slices - 1 else lanes - LANES * (num_slices - 1) for s in range(num_slices))
    base = tuple(int(x) for x in np.concatenate([[0], np.cumsum([r if b else 0 for r, b in zip(real, block)])[:-1]]))
    zero_from = sum(r for r, b in zip(real, block) if b)
    kc_max = max(1, SBUF_BUDGET // (B * 4))                  # gathered rows per chunk that fit SBUF
    # Runs of consecutive slices whose tables fit TABLE_COLS share one copy: (first, last + 1, copied).
    # A slice wider than that on its own copies its tables chunk by chunk instead (copied=False).
    groups, s = [], 0
    while s < num_slices:
        e, gw = s, 0
        while e < num_slices and gw + widths[e] <= TABLE_COLS:
            gw += widths[e]
            e += 1
        groups.append((s, e, True) if e > s else (s, s + 1, False))
        s = max(e, s + 1)

    def sell_matmul(cols, vals, out_rows, X):
        # cols (int32), vals (float32): (128, total_width). out_rows (int32): (128, total_passes). X: (n, B).
        Y = nl.ndarray((m, B), dtype=nl.float32, buffer=nl.shared_hbm)
        if has_splits and zero_from < m:                     # pieces get added in place: start from zero
            zr = max(1, min(-(-(m - zero_from) // LANES), SBUF_BUDGET // (B * 4)))
            zero = nl.ndarray((LANES, zr * B), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=zero, value=0.0)
            for r0 in range(zero_from, m, LANES * zr):
                nr = min(LANES * zr, m - r0)
                full = nr // zr
                if full:
                    nisa.dma_copy(dst=Y[r0:r0 + full * zr, :].ap(pattern=[[zr * B, full], [1, zr * B]]),
                                  src=zero[0:full, :])
                rem = nr - full * zr                         # fewer than zr rows left: one partition holds them
                if rem:
                    nisa.dma_copy(dst=Y[r0 + full * zr:r0 + nr, :].ap(pattern=[[rem * B, 1], [1, rem * B]]),
                                  src=zero[0:1, 0:rem * B])
        if total_passes:
            orow = nl.ndarray((LANES, total_passes), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=orow, src=out_rows)
        for s0, s1, copied in groups:
            g0, g1 = offsets[s0], offsets[s1 - 1] + widths[s1 - 1]
            if copied:
                col_g = nl.ndarray((LANES, g1 - g0), dtype=nl.int32, buffer=nl.sbuf)
                val_g = nl.ndarray((LANES, g1 - g0), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=col_g, src=cols[:, g0:g1])
                nisa.dma_copy(dst=val_g, src=vals[:, g0:g1])
            for s in range(s0, s1):                          # unrolled: each slice has its own width
                w, o = widths[s], offsets[s]
                acc = nl.ndarray((LANES, B), dtype=nl.float32, buffer=nl.sbuf)
                nisa.memset(dst=acc, value=0.0)
                for k0 in range(0, w, kc_max):
                    kc = min(kc_max, w - k0)
                    if copied:
                        col_tile = col_g[:, o - g0 + k0:o - g0 + k0 + kc]
                        val_tile = val_g[:, o - g0 + k0:o - g0 + k0 + kc]
                    else:
                        col_tile = nl.ndarray((LANES, kc), dtype=nl.int32, buffer=nl.sbuf)
                        val_tile = nl.ndarray((LANES, kc), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.dma_copy(dst=col_tile, src=cols[:, o + k0:o + k0 + kc])
                        nisa.dma_copy(dst=val_tile, src=vals[:, o + k0:o + k0 + kc])
                    rows = nl.ndarray((LANES, kc * B), dtype=nl.float32, buffer=nl.sbuf)
                    for k in range(kc):
                        # The gather: lane p reads row X[col[p, k], :]. `vector_offset` gives every lane
                        # its own row index; software descriptor generation (swdge) is what allows that.
                        nisa.dma_copy(dst=rows[:, k * B:(k + 1) * B],
                                      src=X.ap(pattern=[[B, LANES], [1, B]], offset=0,
                                               vector_offset=col_tile[:, k:k + 1], indirect_dim=0),
                                      dge_mode=nisa.dge_mode.swdge)
                    for k in range(kc):
                        # acc = rows_k * val[:, k] + acc, one fused Vector-engine instruction
                        nisa.scalar_tensor_tensor(dst=acc, data=rows[:, k * B:(k + 1) * B], op0=nl.multiply,
                                                  operand0=val_tile[:, k:k + 1], op1=nl.add, operand1=acc)
                if block[s]:                                 # lane p -> Y[base + p]: one plain DMA
                    k, b0 = real[s], base[s]
                    nisa.dma_copy(dst=Y[b0:b0 + k, :].ap(pattern=[[B, k], [1, B]]), src=acc[0:k, :])
                # The scatter: lane p's result goes to row out_rows[p, pass] (padding lanes: row m, skipped).
                for j in range(passes[s]):
                    c = first_pass[s] + j
                    dst = Y.ap(pattern=[[B, LANES], [1, B]], offset=0, vector_offset=orow[:, c:c + 1], indirect_dim=0)
                    if has_splits:
                        nisa.dma_compute(dst=dst, srcs=[dst, acc], reduce_op=nl.add, oob_mode=nisa.oob_mode.skip)
                    else:
                        nisa.dma_copy(dst=dst, src=acc, oob_mode=nisa.oob_mode.skip, dge_mode=nisa.dge_mode.swdge)
        return Y

    # NKI's cross-process trace cache (NKI_ENABLE_TRACE_CACHE) may not see constants captured from this
    # closure, so two layouts with equal table shapes could collide. A per-layout name keeps them apart
    # if the cache keys on it.
    tag = hashlib.sha1(repr((KERNEL_VERSION,) + key).encode()).hexdigest()[:16]
    sell_matmul.__name__ = sell_matmul.__qualname__ = f"sell_matmul_{tag}"
    _KERNELS[key] = kernel = nki.jit(sell_matmul)
    return kernel


def sell_matmul_simulated(S: SELL, X: np.ndarray) -> np.ndarray:
    """Run the kernel in NKI's CPU simulator and return K @ X (for testing without hardware)."""
    X = np.asarray(X, dtype=np.float32)
    squeeze = X.ndim == 1
    X2 = np.ascontiguousarray(X.reshape(X.shape[0], -1))
    kernel = make_sell_kernel(S, X2.shape[1])
    Y = np.asarray(nki.simulate(kernel)(*kernel_tables(S), X2))
    return Y[:, 0] if squeeze else Y
