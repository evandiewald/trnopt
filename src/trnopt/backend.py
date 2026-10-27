"""Where device differences live: one settings object, chosen once, plus the "multiply by K" wrapper.

The rest of the solver never asks "am I on Trainium?". It reads these settings instead.

Why each setting exists
-----------------------
* dtype   -- Trainium has no float64. Its widest number type is float32 (about 7 significant
             digits instead of 16), so tight tolerances get harder.
* dense   -- PyTorch's new Trainium support (TorchNeuron) doesn't handle sparse matrices, so on
             Trainium we store K as a full grid, zeros included. NVIDIA GPUs don't have this
             problem: cuSPARSE is mature, which is one reason cuOpt can work with huge sparse LPs.
* compile -- On an accelerator, every separate tensor operation is a trip from the host to the
             chip. torch.compile fuses a whole iteration into one step, cutting those trips.
* lazy    -- The older Trainium software (PyTorch/XLA, the only option on trn1) doesn't run
             operations when you call them. It records them into a recipe, then compiles and runs
             the whole recipe when you read a value back or call sync(). We sync after every
             iteration so the recipe is always "one iteration": compiled once, then reused. Without
             that, the 64 iterations between convergence checks get unrolled into one huge program.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import torch


@dataclass(frozen=True)
class Backend:
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    dense: bool = False
    compile: bool = False
    compile_backend: str | None = None       # None = PyTorch's default compiler for the device
    lazy: bool = False                       # PyTorch/XLA-style lazy execution: sync() after each iteration
    spmv: str = "torch"                      # how to multiply by K: "torch" (dense or sparse per `dense`),
                                             # "nki" (custom Trainium kernel, sliced-ELL layout),
                                             # "nki-sim" (the same kernel in NKI's CPU simulator, for testing),
                                             # "metal" (custom Apple GPU kernel, sliced-ELL layout) or
                                             # "metal-ref" (the Metal kernel's arithmetic in plain PyTorch, for testing)
    max_dense_entries: int = 100_000_000     # refuse to build a dense K bigger than this (~400 MB in float32)

    @classmethod
    def preset(cls, name: str, **overrides) -> "Backend":
        """Named starting points. Any field can be overridden, e.g. Backend.preset("cpu", dtype=torch.float32)."""
        presets = {
            "cpu": cls(),
            # float32 alone, with fast sparse storage: isolates the effect of Trainium's precision.
            "cpu-float32": cls(dtype=torch.float32),
            "cuda": cls(device="cuda"),
            # Trainium2/3 with TorchNeuron (PyTorch 2.10+): operations run as they're called.
            "neuron": cls(device="neuron", dtype=torch.float32, dense=True, compile=True),
            # Trainium1 / Inferentia2 with the older PyTorch/XLA stack (torch-neuronx <= 2.9).
            # Also runs on a plain CPU with torch_xla installed and PJRT_DEVICE=CPU, for testing.
            "neuron-xla": cls(device="xla", dtype=torch.float32, dense=True, lazy=True),
            # Same restrictions as Trainium, but runs on your laptop: catches problems before you pay for hardware.
            "emulate-neuron": cls(device="cpu", dtype=torch.float32, dense=True, compile=True),
            # Apple silicon GPU via PyTorch's Metal backend. No float64 on Apple GPUs; full-grid K
            # because PyTorch's sparse support on MPS is limited. CPU and GPU share memory, so the
            # float64 finish on the CPU costs no copies.
            "mps": cls(device="mps", dtype=torch.float32, dense=True),
            # Experimental: try PyTorch's sparse CSR multiply on MPS (not supported as of torch 2.14).
            "mps-sparse": cls(device="mps", dtype=torch.float32, dense=False),
            # Apple silicon GPU with our own sparse kernel (sliced ELL, see metal_sell.py).
            "mps-metal": cls(device="mps", dtype=torch.float32, spmv="metal"),
        }
        if name not in presets:
            raise ValueError(f"unknown backend preset {name!r}; choose from {sorted(presets)}")
        return replace(presets[name], **overrides)

    def sync(self) -> None:
        """Lazy (XLA) backends: compile-and-run everything recorded so far. A no-op otherwise."""
        if self.lazy:
            import torch_xla
            torch_xla.sync()

    def tensor(self, a) -> torch.Tensor:
        # Convert on the CPU first: some devices (Apple's MPS) can't hold float64 even briefly.
        return torch.as_tensor(np.asarray(a), dtype=self.dtype).to(self.device)


class MatVec:
    """Multiply by K and by K-transpose, in whichever storage format the backend wants.

    Works on a single vector (shape (n,)) or a stack of vectors (shape (n, k)) -- the second is
    what batching will use in Phase 3, where many LPs share one K.

    This is also where a custom Trainium (NKI) kernel will plug in later: the solver only ever
    calls `matvec.K(x)` and `matvec.KT(y)`.
    """

    def __init__(self, K: sp.spmatrix, backend: Backend):
        K = sp.csr_matrix(K, dtype=np.float64)
        m, n = K.shape
        self.dense = backend.dense
        self.be = backend
        if backend.spmv in ("nki", "nki-sim"):
            self._init_nki(K)
            return
        if backend.spmv in ("metal", "metal-ref"):
            from .metal_sell import MetalSELL
            ref = backend.spmv == "metal-ref"
            self._mK = MetalSELL(K, device=backend.device, reference=ref)
            self._mKT = MetalSELL(K.T.tocsr(), device=backend.device, reference=ref)
            return
        if backend.dense:
            if m * n > backend.max_dense_entries:
                raise ValueError(
                    f"K is {m} x {n} = {m * n:,} entries, over the dense limit of {backend.max_dense_entries:,}. "
                    "Dense storage is a stopgap until sparse support (Phase 4); use a sparse backend for this problem."
                )
            self._K = backend.tensor(K.toarray())
            self._KT = self._K.T.contiguous()
        else:
            self._K = _to_torch_csr(K, backend)
            self._KT = _to_torch_csr(K.T.tocsr(), backend)

    def K(self, x: torch.Tensor) -> torch.Tensor:
        if self.be.spmv in ("metal", "metal-ref"):
            return self._mK.matmul(x).to(x.dtype)
        if self.be.spmv != "torch":
            return self._nki_apply(self._sK, self._kK, self._K_tabs, x)
        return self._K @ x

    def KT(self, y: torch.Tensor) -> torch.Tensor:
        if self.be.spmv in ("metal", "metal-ref"):
            return self._mKT.matmul(y).to(y.dtype)
        if self.be.spmv != "torch":
            return self._nki_apply(self._sKT, self._kKT, self._KT_tabs, y)
        return self._KT @ y

    # ---- the custom Trainium kernel (sliced ELL, see sell.py / nki_sell.py) -------------------
    def _init_nki(self, K: sp.csr_matrix) -> None:
        from .nki_sell import make_sell_kernel
        from .sell import build_sell
        self._sK, self._sKT = build_sell(K), build_sell(K.T.tocsr())
        self._kK = make_sell_kernel(self._sK.widths, self._sK.offsets)
        self._kKT = make_sell_kernel(self._sKT.widths, self._sKT.offsets)
        dev = self.be.device if self.be.spmv == "nki" else "cpu"
        tabs = lambda S: (torch.as_tensor(S.cols, device=dev), torch.as_tensor(S.vals, dtype=torch.float32, device=dev),
                          torch.as_tensor(np.maximum(S.out_rows, 0), device=dev),
                          torch.as_tensor(S.out_rows >= 0, device=dev))
        self._K_tabs, self._KT_tabs = tabs(self._sK), tabs(self._sKT)

    def _nki_apply(self, S, kernel, tabs, x: torch.Tensor) -> torch.Tensor:
        cols, vals, out_rows, keep = tabs
        squeeze = x.dim() == 1
        X = x.reshape(x.shape[0], -1).to(torch.float32)
        Xz = torch.cat([X, torch.zeros(1, X.shape[1], dtype=X.dtype, device=X.device)])   # row n = 0 for padding
        if self.be.spmv == "nki-sim":
            import neuronxcc.nki as nki
            slot_out = torch.as_tensor(np.asarray(nki.simulate_kernel(kernel, cols.numpy(), vals.numpy(), Xz.cpu().numpy())))
        else:
            slot_out = kernel(cols, vals, Xz)          # runs on the NeuronCore (untested until hardware)
        # Add pseudo-rows back into their original rows; padding lanes are masked out.
        slot_out = slot_out * keep.to(slot_out.dtype)[:, None]
        Y = torch.zeros(S.shape[0], X.shape[1], dtype=slot_out.dtype, device=slot_out.device)
        Y = Y.index_add(0, out_rows, slot_out)
        Y = Y.to(dtype=x.dtype, device=x.device)
        return Y[:, 0] if squeeze else Y


def _to_torch_csr(M: sp.csr_matrix, backend: Backend) -> torch.Tensor:
    with warnings.catch_warnings():  # silence torch's "sparse CSR is beta" notices
        warnings.simplefilter("ignore", UserWarning)
        t = torch.sparse_csr_tensor(
            torch.from_numpy(M.indptr.astype(np.int64)),
            torch.from_numpy(M.indices.astype(np.int64)),
            torch.from_numpy(M.data),
            size=M.shape,
            dtype=backend.dtype,
        )
        return t.to(backend.device)
