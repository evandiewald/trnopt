"""One persistent home for everything trnopt compiles, so a new process doesn't compile it again.

What gets compiled, and where it's kept
---------------------------------------
* torch.compile on a CPU (emulate-neuron) or GPU: Inductor writes the generated code and its
  compiled binaries to TORCHINDUCTOR_CACHE_DIR. PyTorch's default is under /tmp, which is wiped
  on reboot (and on every fresh spot instance).
* torch.compile with the "neuron" backend (TorchNeuron, torch-neuronx 2.10+): each graph is
  lowered to HLO and compiled by neuronx-cc into a NEFF. TorchNeuron keeps the NEFFs in
  TORCH_NEURONX_NEFF_CACHE_DIR and eager ops' HLO in TORCH_NEURONX_HLO_CACHE_DIR. Both default to
  /tmp, which is wiped whenever the instance stops, so every session on a stop/start instance
  compiled from scratch. (Its lock files, TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR, must stay on a local
  disk, so they stay in /tmp.)
* The older PyTorch/XLA stack (preset "neuron-xla") uses NEURON_COMPILE_CACHE_URL instead.
* NKI kernels: NKI keeps its own trace cache in /var/tmp, which survives reboots; left alone.

These caches are keyed on the program itself (graph, input shapes and dtypes, compiler version),
not on which LP it came from: re-solving an instance, or any instance of the same size, in a new
process skips the compiler. What still runs every process is tracing the Python step function,
which is cheap next to a compile.

configure() points all of them at one directory, TRNOPT_CACHE_DIR (default ~/.cache/trnopt), so
they survive reboots and can be shared: put it on EFS, or copy it to the next instance. It never
overrides a variable you set yourself. TRNOPT_CACHE_DIR=off leaves PyTorch's and Neuron's own
defaults alone.

It runs when trnopt is imported. `import torch` loads torch-neuronx, which reads its cache
locations as it loads, so import trnopt before torch (or set the variables in your shell).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ENV = "TRNOPT_CACHE_DIR"
_OFF = ("", "0", "off", "none", "false")


def cache_dir() -> Path | None:
    """The cache's root directory, or None if TRNOPT_CACHE_DIR turns caching off."""
    v = os.environ.get(ENV)
    if v is not None:
        return None if v.strip().lower() in _OFF else Path(v).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "trnopt"


def configure() -> Path | None:
    """Point the Inductor and Neuron compile caches at cache_dir(). Returns the root, or None if off."""
    root = cache_dir()
    if root is None:
        return None
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor"))
    os.environ.setdefault("TORCH_NEURONX_NEFF_CACHE_DIR", str(root / "neff"))
    os.environ.setdefault("TORCH_NEURONX_HLO_CACHE_DIR", str(root / "hlo"))
    # neuronx-cc also takes the location as --cache_dir in NEURON_CC_FLAGS, which wins; leave that alone.
    if "--cache_dir" not in os.environ.get("NEURON_CC_FLAGS", ""):
        os.environ.setdefault("NEURON_COMPILE_CACHE_URL", str(root / "neuron"))
    return root


def clear() -> None:
    """Delete trnopt's cache directory (the next run compiles everything again)."""
    root = cache_dir()
    if root is not None and root.exists():
        shutil.rmtree(root)
