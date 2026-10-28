"""Where device differences live: one settings object, chosen once, plus the "multiply by K" wrapper.

The rest of the solver never asks "am I on Trainium?". It reads these settings instead.

Why each setting exists
-----------------------
* dtype   -- Trainium has no float64. Its widest number type is float32 (about 7 significant
             digits instead of 16), so tight tolerances get harder.
* dense   -- PyTorch's native Trainium support (TorchNeuron) doesn't handle sparse matrices, so on
             Trainium we store K as a full grid, zeros included. NVIDIA GPUs don't have this
             problem: cuSPARSE is mature, which is one reason cuOpt can work with huge sparse LPs.
* compile -- On an accelerator, every separate tensor operation is a trip from the host to the
             chip. torch.compile fuses a whole iteration into one step, cutting those trips.
             On Neuron this needs torch-neuronx's own "neuron" compiler backend (PyTorch's default,
             Inductor, has no Neuron support) and fixed shapes: the Neuron compiler can't handle
             the symbolic shapes torch.compile switches to after it sees a second problem size.
* lazy    -- The older Trainium software (PyTorch/XLA, torch-neuronx <= 2.9) doesn't run
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


_COMPILED: dict = {}          # compile_fn's cache: one compiled function per (function, compiler, shape mode)


@dataclass(frozen=True)
class Backend:
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    dense: bool = False
    compile: bool = False
    compile_backend: str | None = None       # None = PyTorch's default compiler for the device
    lazy: bool = False                       # PyTorch/XLA-style lazy execution: sync() after each iteration
    spmv: str = "torch"                      # how to multiply by K: "torch" (dense or sparse per `dense`),
                                             # "auto" (Trainium: dense if K is small enough, else "nki-op"),
                                             # "nki" (custom Trainium kernel, sliced-ELL layout),
                                             # "nki-op" (the same kernel as a custom op inside the compiled
                                             #   graph, see nki_op.py; falls back to "nki" if that fails),
                                             # "nki-sim" (the same kernel in NKI's CPU simulator, for testing),
                                             # "metal" (custom Apple GPU kernel, sliced-ELL layout) or
                                             # "metal-ref" (the Metal kernel's arithmetic in plain PyTorch, for testing)
    max_dense_entries: int = 100_000_000     # refuse to build a dense K bigger than this (~400 MB in float32)
    # spmv="auto": dense up to this many entries (~80 MB in float32), the NKI kernel above it. Measured on trn1
    # (2026-10-27): dense won outright on small K (setcover 1000x1000: 2.4 s vs 4.5 s for strong branching),
    # tied at 2.8M entries (berkel) and lost slightly at 13.6M (nu25, B=1024); NKI is the only option
    # for the large MIPLIB instances. Above the cutoff, "nki-op": 3-7% faster per iteration than "nki" and
    # quicker to compile (results/nki_op_2026-10-28.md).
    auto_dense_entries: int = 20_000_000

    @classmethod
    def preset(cls, name: str, **overrides) -> "Backend":
        """Named starting points. Any field can be overridden, e.g. Backend.preset("cpu", dtype=torch.float32)."""
        presets = {
            "cpu": cls(),
            # float32 alone, with fast sparse storage: isolates the effect of Trainium's precision.
            "cpu-float32": cls(dtype=torch.float32),
            "cuda": cls(device="cuda"),
            # Trainium (trn1, trn2, trn3) with TorchNeuron, PyTorch's native Neuron device
            # (torch-neuronx 2.10+): operations run as they're called. Verified on trn1. Dense K when it's
            # small enough, the NKI sparse kernel otherwise (spmv="auto").
            "neuron": cls(device="neuron", dtype=torch.float32, dense=True, compile=True, compile_backend="neuron",
                          spmv="auto"),
            # Legacy: the older PyTorch/XLA stack (torch-neuronx <= 2.9). Current Neuron DLAMIs don't ship it.
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

    def compile_fn(self, f):
        """torch.compile `f` if this backend compiles, else return it unchanged.

        Cached per (function, compiler, shape mode) for the life of the process, so every solve
        reuses the same compiled function; pass per-problem data in as arguments, never by
        closure or bound method (see pdlp._Problem.__init__).
        """
        if not self.compile:
            return f
        # dynamic=False: always specialize to concrete shapes (see "compile" above). Elsewhere,
        # None keeps PyTorch's default of switching to symbolic shapes after a recompile.
        dynamic = False if self.device.startswith("neuron") else None
        if dynamic is False:
            # Fixed shapes mean one compiled version per shape (problem size, batch-size bucket).
            # Dynamo's default of 8 versions per function would silently fall back to uncompiled
            # code in a long run (a benchmark over many problems, a MIP with several buckets).
            import torch._dynamo as dynamo
            dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 256)
        key = (f, self.compile_backend or "inductor", dynamic)
        if key not in _COMPILED:
            _COMPILED[key] = torch.compile(f, backend=key[1], dynamic=dynamic)
        return _COMPILED[key]

    def batch_slots(self, n: int, cap: int) -> int:
        """How many slots to give a batch of n LPs (at most cap).

        On Neuron every batch size is a separate compile (seconds), so round up to a power of two
        (at least 16): a branch-and-bound then compiles a handful of sizes instead of one per
        distinct candidate count. Unused slots stay frozen. Elsewhere, exactly min(n, cap).
        """
        if not self.device.startswith("neuron"):
            return max(1, min(n, cap))
        b = 16
        while b < n:
            b *= 2
        return max(1, min(b, cap))

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
        spmv = backend.spmv
        if spmv == "auto":
            spmv = "torch" if m * n <= backend.auto_dense_entries else "nki-op"
            if spmv == "torch":
                backend = replace(backend, dense=True)
        self.spmv = spmv                         # the method actually used for this matrix
        self.dense = backend.dense and spmv == "torch"
        self.be = backend
        if spmv in ("nki", "nki-op", "nki-sim"):
            self._init_nki(K)
            return
        if spmv in ("metal", "metal-ref"):
            from .metal_sell import MetalSELL
            ref = spmv == "metal-ref"
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
        if self.spmv in ("metal", "metal-ref"):
            return self._mK.matmul(x).to(x.dtype)
        if self.spmv == "nki-op":
            return self._nki_op_apply("K", x)
        if self.spmv != "torch":
            return self._nki_apply("K", x)
        return self._K @ x

    def KT(self, y: torch.Tensor) -> torch.Tensor:
        if self.spmv in ("metal", "metal-ref"):
            return self._mKT.matmul(y).to(y.dtype)
        if self.spmv == "nki-op":
            return self._nki_op_apply("KT", y)
        if self.spmv != "torch":
            return self._nki_apply("KT", y)
        return self._KT @ y

    # ---- the custom Trainium kernel (sliced ELL, see sell.py / nki_sell.py) -------------------
    def _init_nki(self, K: sp.csr_matrix) -> None:
        from .nki_sell import kernel_tables
        from .sell import build_sell
        self._sK, self._sKT = build_sell(K), build_sell(K.T.tocsr())
        dev = "cpu" if self.spmv == "nki-sim" else self.be.device
        tabs = lambda S: tuple(torch.as_tensor(t, device=dev) for t in kernel_tables(S))
        self._K_tabs, self._KT_tabs = tabs(self._sK), tabs(self._sKT)
        self._kernels = {}                       # (which, B) -> kernel; one compile per batch size
        if self.spmv == "nki-op":
            try:
                from . import nki_op
                nki_op.register()
                self._layout = {"K": nki_op.layout_id(self._sK), "KT": nki_op.layout_id(self._sKT)}
                self._op_checked = set()         # (which, B) the op has already run eagerly without error
            except Exception as e:               # no torch_neuronx, or an API it doesn't have
                warnings.warn(f"spmv='nki-op' unavailable ({type(e).__name__}: {e}); using 'nki'")
                self.spmv = "nki"

    # The kernel as a custom op (nki_op.py): torch.compile keeps it in the graph, no graph break.
    def _nki_op_apply(self, which: str, x: torch.Tensor) -> torch.Tensor:
        S, tabs = (self._sK, self._K_tabs) if which == "K" else (self._sKT, self._KT_tabs)
        X = x.reshape(x.shape[0], -1)
        if X.dtype != torch.float32:
            X = X.to(torch.float32)
        args = (*tabs, X.contiguous(), S.shape[0], self._layout[which])
        key = (which, X.shape[1])
        if torch.compiler.is_compiling() or key in self._op_checked:
            Y = torch.ops.trnopt.sell_matmul(*args)
        else:
            # The first eager call per (which, B) compiles the kernel through the op. If that fails, fall back
            # for good. (A failure inside a torch.compile'd graph can't be caught here; it surfaces as is.)
            try:
                Y = torch.ops.trnopt.sell_matmul(*args)
                self._op_checked.add(key)
            except Exception as e:
                warnings.warn(f"spmv='nki-op' failed ({type(e).__name__}: {e}); using 'nki'")
                self.spmv = "nki"
                return self._nki_apply(which, x)
        Y = Y.to(x.dtype)
        return Y[:, 0] if x.dim() == 1 else Y

    # torch.compile can't trace into an nki.jit call, so it stops here: the code around each multiply
    # still gets fused, and the kernel runs as its own call between the compiled pieces.
    @torch.compiler.disable
    def _nki_apply(self, which: str, x: torch.Tensor) -> torch.Tensor:
        from .nki_sell import make_sell_kernel
        S, tabs = (self._sK, self._K_tabs) if which == "K" else (self._sKT, self._KT_tabs)
        squeeze = x.dim() == 1
        X = x.reshape(x.shape[0], -1)
        if X.dtype != torch.float32:
            X = X.to(torch.float32)
        B = X.shape[1]
        kernel = self._kernels.get((which, B))
        if kernel is None:
            kernel = self._kernels[(which, B)] = make_sell_kernel(S, B)
        if self.spmv == "nki-sim":
            import nki
            args = [t.numpy() for t in tabs] + [X.detach().cpu().contiguous().numpy()]
            Y = torch.as_tensor(np.asarray(nki.simulate(kernel)(*args)))
        else:
            Y = kernel(*tabs, X.contiguous())     # the whole multiply, one call on the NeuronCore
        if Y.dtype != x.dtype or Y.device != x.device:
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
