"""Small cross-cutting utilities: device selection and seeding.

Unlike the other biotp modules (stubs), these are implemented, since they are
trivial infrastructure the whole pipeline relies on to run unchanged on a SLURM
GPU node (CUDA), the MacBook (Apple MPS), or CPU.
"""

from __future__ import annotations

import os
import random


def get_device(prefer_gpu: bool = True) -> str:
    """Return the best available torch device string: 'cuda', 'mps', or 'cpu'.

    Preference order is CUDA (SLURM GPU nodes), then Apple MPS (the MacBook),
    then CPU. Pass prefer_gpu=False to force CPU (useful for debugging or exact
    determinism). On MPS, set PYTORCH_ENABLE_MPS_FALLBACK=1 in the environment
    so ops without an MPS kernel fall back to CPU rather than erroring.
    """
    if not prefer_gpu:
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clamp_unit_interval(value: float) -> float:
    """Clamp a value to the closed unit interval [0.0, 1.0].

    Metric helpers occasionally produce values a hair outside [0, 1] through
    floating-point error, and downstream plotting treats that as a hard error.

    Args:
        value: The value to clamp.

    Returns:
        The value constrained to [0.0, 1.0].

    Raises:
        ValueError: If the value is NaN, which cannot be meaningfully clamped.
    """
    if value != value:
        raise ValueError("cannot clamp NaN to the unit interval")
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
