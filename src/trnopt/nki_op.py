"""The NKI sliced-ELL kernel (nki_sell.py) as a PyTorch custom op, so torch.compile can keep it in the graph.

Backend(spmv="nki") calls the `nki.jit` kernel from a `torch.compiler.disable` function: every multiply
by K or K^T is a graph break, and the compiled PDHG step runs as several pieces with the kernel between
them. spmv="nki-op" calls `torch.ops.trnopt.sell_matmul` instead, registered with TorchNeuron's
`nki_op`, which Dynamo can trace (through the fake implementation below) and the Neuron backend lowers
to an NKI call inside the compiled graph.

The kernel is specialized to one slice layout and batch width B (see make_sell_kernel). The op can't
take a layout object, so layouts get a small integer id here, and the op's arguments are just the
kernel's input tensors plus (m, layout id). Tensors are reached through the solver's arguments, so
torch.compile sees them as graph inputs, and the ints are compile-time constants: one graph per
(layout, B), the same granularity as the kernels themselves.

No `from __future__ import annotations` here: torch.library reads the op's schema from the
annotations, and string annotations confuse it.
"""

import torch

from .nki_sell import layout_key, make_sell_kernel
from .sell import SELL

_LAYOUTS: list = []          # layout id -> (SELL with that layout, ordered): what the kernel bakes in
_LAYOUT_IDS: dict = {}       # layout_key(S, ordered) -> layout id
_KERNELS: dict = {}          # (layout id, B) -> wrapped kernel; make_sell_kernel's own lookup costs a pass over S
_OP = None                   # the registered op, once register() has succeeded


def layout_id(S: SELL, ordered: bool = False) -> int:
    """A stable integer for S's layout (and output order, see make_sell_kernel), for the life of the process.
    Equal layouts share an id."""
    key = layout_key(S, ordered)
    if key not in _LAYOUT_IDS:
        _LAYOUT_IDS[key] = len(_LAYOUTS)
        _LAYOUTS.append((S, ordered))
    return _LAYOUT_IDS[key]


def register():
    """Register trnopt::sell_matmul (once per process) and return it. Raises if TorchNeuron can't."""
    global _OP
    if _OP is not None:
        return _OP
    from torch_neuronx import nki_op, wrap_nki

    def fake(cols: torch.Tensor, vals: torch.Tensor, out_rows: torch.Tensor, X: torch.Tensor,
             m: int, layout: int) -> torch.Tensor:
        return X.new_empty((m, X.shape[1]), dtype=torch.float32)

    # fake_fn is required: running the real body on fake tensors would try to compile the kernel.
    @nki_op("trnopt::sell_matmul", mutates_args={}, fake_fn=fake)
    def sell_matmul(cols: torch.Tensor, vals: torch.Tensor, out_rows: torch.Tensor, X: torch.Tensor,
                    m: int, layout: int) -> torch.Tensor:
        key = (layout, X.shape[1])
        kernel = _KERNELS.get(key)
        if kernel is None:
            # wrap_nki: the traceable form of the kernel (torch_neuronx's own nki_op kernels call it this way)
            S, ordered = _LAYOUTS[layout]
            kernel = _KERNELS[key] = wrap_nki(make_sell_kernel(S, X.shape[1], ordered))
        return kernel(cols, vals, out_rows, X)

    _OP = sell_matmul
    return _OP
