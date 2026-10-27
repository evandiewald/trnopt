"""Sparse multiply on Apple silicon GPUs: a Metal kernel for Y = K @ X in sliced-ELL layout.

Why a custom kernel
-------------------
PyTorch's sparse tensors don't run on Apple's GPU ("Could not run 'new_compressed_tensor' from
the 'mps:0' device"), so the `mps` backend stores K as a full grid. On MIPLIB that wastes
28-770x the necessary arithmetic. This kernel uses the same sliced-ELL layout as the Trainium
NKI kernel (see sell.py), which does ~1.08x the necessary work on MIPLIB.

Unlike Trainium, Apple GPUs (like NVIDIA's) handle scattered reads natively, so the kernel is
short: one GPU thread per (row slot, batch column).

How the threads are laid out
----------------------------
Grid: x = batch column b (0..B-1), y = row slot (0..num_slices*128-1).
Thread (b, slot): slot belongs to slice s = slot // 128, lane = slot % 128. It walks across the
slice's width, and for each entry reads column index c and value v, then adds v * X[c, b].

* X is stored row-major (n x B), so the 32 threads of a SIMD group (consecutive b) read 32
  consecutive floats of the same row of X: one coalesced read. That's why batching helps so
  much: with B >= 32, every gather is a full, contiguous chunk.
* The tables are stored entry-major ((total_width, 128): entry k of every lane, then entry
  k+1, ...), so with B = 1, consecutive threads (consecutive lanes) read consecutive table
  entries.
* Padding entries hold column -1 and are skipped, so X needs no appended zero row.

The kernel is compiled at runtime with torch.mps.compile_shader and called directly on PyTorch
tensors on the "mps" device: no copies, no conversion.

`MetalSELL(..., reference=True)` runs the same index arithmetic in plain PyTorch instead. It
works on any device, and is how the layout and bookkeeping are tested off-Mac.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .sell import LANES, build_sell

METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void sell_spmm(
    device float*       out     [[buffer(0)]],   // (num_slices*128, B) row-major
    device const int*   cols    [[buffer(1)]],   // (total_width, 128) entry-major, -1 = padding
    device const float* vals    [[buffer(2)]],   // (total_width, 128) entry-major
    device const int*   widths  [[buffer(3)]],   // (num_slices,)
    device const int*   offsets [[buffer(4)]],   // (num_slices,)
    device const float* X       [[buffer(5)]],   // (n, B) row-major
    constant long&      B       [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]])
{
    const uint b = pos.x;
    const uint slot = pos.y;
    if ((long)b >= B) return;
    const uint s = slot / 128;
    const uint lane = slot % 128;
    const int w = widths[s];
    const int off = offsets[s];
    float acc = 0.0f;
    for (int k = 0; k < w; ++k) {
        const int j = (off + k) * 128 + (int)lane;
        const int c = cols[j];
        if (c >= 0) {
            acc = fma(vals[j], X[(long)c * B + b], acc);
        }
    }
    out[(long)slot * B + b] = acc;
}
"""

_LIB = None


def _metal_lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(METAL_SOURCE)
    return _LIB


class MetalSELL:
    """One sparse matrix in sliced-ELL layout on the GPU, with a multiply method."""

    def __init__(self, K: sp.spmatrix, device: str = "mps", reference: bool = False, split="auto"):
        K = sp.csr_matrix(K)
        S = build_sell(K, split=split, dtype=np.float32)
        self.shape = S.shape
        self.reference = reference
        self.slots = S.num_slices * LANES
        n = S.shape[1]
        cols = np.where(S.cols == n, -1, S.cols).astype(np.int32)        # padding -> -1
        self.cols = torch.as_tensor(np.ascontiguousarray(cols.T)).to(device)     # (total_width, 128)
        self.vals = torch.as_tensor(np.ascontiguousarray(S.vals.T.astype(np.float32))).to(device)
        self.widths = torch.as_tensor(np.asarray(S.widths, dtype=np.int32)).to(device)
        self.offsets = torch.as_tensor(np.asarray(S.offsets, dtype=np.int32)).to(device)
        self._widths_np, self._offsets_np = np.asarray(S.widths), np.asarray(S.offsets)

        # Putting pseudo-rows back into original rows. If no row was split, every original row
        # owns exactly one slot and it's a single gather; otherwise a scatter-add.
        keep = S.out_rows >= 0
        counts = np.bincount(S.out_rows[keep], minlength=S.shape[0])
        self.permutation_only = bool(np.all(counts == 1))
        if self.permutation_only:
            slot_of_row = np.empty(S.shape[0], dtype=np.int64)
            slot_of_row[S.out_rows[keep]] = np.flatnonzero(keep)
            self.slot_of_row = torch.as_tensor(slot_of_row).to(device)
        else:
            self.valid_slots = torch.as_tensor(np.flatnonzero(keep)).to(device)
            self.valid_rows = torch.as_tensor(S.out_rows[keep]).to(device)
        self.stored_entries = S.slots
        self.nnz = K.nnz

    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """K @ x for x of shape (n,) or (n, B), float32 on this object's device."""
        squeeze = x.dim() == 1
        X = x.reshape(x.shape[0], -1)
        if X.dtype != torch.float32:
            X = X.to(torch.float32)
        X = X.contiguous()
        B = X.shape[1]
        if self.reference:
            slot_out = self._reference(X)
        else:
            slot_out = torch.empty(self.slots, B, dtype=torch.float32, device=X.device)
            gx = min(B, 32)
            _metal_lib().sell_spmm(slot_out, self.cols, self.vals, self.widths, self.offsets, X, B,
                                   threads=[B, self.slots], group_size=[gx, max(1, 256 // gx)])
        if self.permutation_only:
            Y = slot_out.index_select(0, self.slot_of_row)
        else:
            Y = torch.zeros(self.shape[0], B, dtype=slot_out.dtype, device=slot_out.device)
            Y.index_add_(0, self.valid_rows, slot_out.index_select(0, self.valid_slots))
        return Y[:, 0] if squeeze else Y

    def _reference(self, X: torch.Tensor) -> torch.Tensor:
        """Exactly the kernel's index arithmetic, slice by slice, in plain PyTorch."""
        B = X.shape[1]
        out = torch.zeros(self.slots, B, dtype=torch.float32, device=X.device)
        Xz = torch.cat([X, torch.zeros(1, B, dtype=X.dtype, device=X.device)])   # -1 -> zero row
        for s, (w, off) in enumerate(zip(self._widths_np, self._offsets_np)):
            c = self.cols[off: off + w].long()                  # (w, 128)
            c = torch.where(c < 0, X.shape[0], c)
            v = self.vals[off: off + w]                         # (w, 128)
            out[s * LANES:(s + 1) * LANES] = (v[:, :, None] * Xz[c]).sum(0)
        return out
