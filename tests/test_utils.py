"""Tests for biotp.utils, the one module that is implemented rather than stubbed.

These are real assertions, not scaffolding: device selection and seeding are what
let the same code run unchanged on a SLURM GPU node, the MacBook, or CPU.
"""

from __future__ import annotations

import builtins
import os
import random

import numpy as np
import pytest

from biotp.utils import get_device, set_seed

VALID_DEVICES = {"cuda", "mps", "cpu"}


def test_get_device_returns_a_valid_device() -> None:
    assert get_device() in VALID_DEVICES


def test_get_device_prefer_gpu_false_forces_cpu() -> None:
    assert get_device(prefer_gpu=False) == "cpu"


def test_get_device_follows_documented_preference_order() -> None:
    """CUDA first, then Apple MPS, then CPU."""
    import torch

    device = get_device()
    if torch.cuda.is_available():
        assert device == "cuda"
    elif torch.backends.mps.is_available():
        assert device == "mps"
    else:
        assert device == "cpu"


def test_get_device_prefer_gpu_false_does_not_need_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing CPU must short-circuit before torch is imported.

    The torch import lives inside the function body precisely so the CPU path
    stays usable where torch is absent or broken, so a broken torch should not
    affect prefer_gpu=False while it does still surface on the default path.
    """
    real_import = builtins.__import__

    def fail_on_torch(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("torch is unavailable in this test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fail_on_torch)

    assert get_device(prefer_gpu=False) == "cpu"
    with pytest.raises(ImportError):
        get_device(prefer_gpu=True)


def test_set_seed_makes_python_random_reproducible() -> None:
    set_seed(0)
    first = [random.random() for _ in range(5)]
    set_seed(0)
    assert [random.random() for _ in range(5)] == first


def test_set_seed_makes_numpy_reproducible() -> None:
    set_seed(0)
    first = np.random.rand(5)
    set_seed(0)
    np.testing.assert_array_equal(np.random.rand(5), first)


def test_set_seed_makes_torch_reproducible() -> None:
    import torch

    set_seed(0)
    first = torch.rand(5)
    set_seed(0)
    assert torch.equal(torch.rand(5), first)


def test_set_seed_distinct_seeds_give_distinct_draws() -> None:
    """A seeded run should still depend on the seed, not be constant."""
    set_seed(0)
    first = np.random.rand(5)
    set_seed(1)
    assert not np.array_equal(np.random.rand(5), first)


def test_set_seed_sets_pythonhashseed() -> None:
    set_seed(1234)
    assert os.environ["PYTHONHASHSEED"] == "1234"
