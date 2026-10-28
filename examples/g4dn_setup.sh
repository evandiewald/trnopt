#!/bin/bash
# One-time setup on a g4dn (Deep Learning Base GPU AMI, Ubuntu 24.04: NVIDIA driver + CUDA preinstalled).
#   bash examples/g4dn_setup.sh        (from the repo root)
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
CUDA_MAJOR=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+")
echo "driver supports CUDA $CUDA_MAJOR"
# The AMI default python3 can be newer than PyTorch / cuOpt wheels support; prefer 3.12.
PY=$(command -v python3.12 || command -v python3.13 || command -v python3)
echo "using $PY"
$PY -m venv ~/venv && source ~/venv/bin/activate
pip install -q --upgrade pip
# PyTorch with CUDA (cu12x wheels run on any driver that supports CUDA 12)
pip install -q torch --index-url https://download.pytorch.org/whl/cu126
pip install -q numpy scipy highspy pytest
if [ "$CUDA_MAJOR" -ge 13 ]; then CUOPT=cuopt-cu13; else CUOPT=cuopt-cu12; fi
pip install -q --extra-index-url=https://pypi.nvidia.com "$CUOPT"
pip install -q --no-deps -e .
python -c "import torch; print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python examples/cuopt_adapter.py
