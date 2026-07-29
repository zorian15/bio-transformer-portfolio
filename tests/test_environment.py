"""Environment invariant: the env must hold exactly one OpenMP runtime.

PyTorch has to come from conda-forge, never the pip wheel. The wheel bundles its
own libomp.dylib alongside the one conda-forge installs for the env, and importing
torch into a process that has not already initialized an OpenMP runtime then
aborts with "OMP: Error #15" and SIGABRT. That kills the entire pytest process
rather than failing a single test, so it cannot be caught where it happens. This
structural check is the substitute: it fails with an actionable message instead of
letting a future `pip install torch` turn the suite into a mystery crash.

The conda-forge build symlinks torch/lib/libomp.dylib to the env's copy, so a pass
here means the symlink is intact and only one runtime can ever load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="checks macOS dylib packaging; the Linux/CUDA cluster env differs",
)

REMEDY = (
    "Reinstall torch from conda-forge rather than pip: "
    "`pip uninstall -y torch && mamba env update -n biollm -f environment.yml`."
)


def test_torch_does_not_bundle_a_second_openmp_runtime() -> None:
    import torch

    bundled = Path(torch.__file__).parent / "lib" / "libomp.dylib"
    if not bundled.exists():
        pytest.skip("this torch build bundles no libomp at all, so nothing can clash")

    assert bundled.is_symlink(), (
        f"{bundled} is a real file, so a second OpenMP runtime is present and "
        f"importing torch before numpy will abort the process. {REMEDY}"
    )

    env_libomp = Path(sys.prefix) / "lib" / "libomp.dylib"
    assert bundled.resolve() == env_libomp.resolve(), (
        f"{bundled} resolves to {bundled.resolve()}, not the env's copy at "
        f"{env_libomp}. {REMEDY}"
    )
