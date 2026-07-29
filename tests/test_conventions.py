"""Repo-wide conventions checked mechanically, so they survive refactors.

Two things are pinned here: the shared API surface promised in PLANNING.md
still exists under the expected names, and the stub APIs stay free of default
argument values so implementing them forces every call site to be explicit.

biotp.utils is deliberately outside the no-defaults check: `get_device(prefer_gpu=True)`
is an existing, intentional default. The convention is about not adding defaults
to these APIs as they grow, not about banning defaults everywhere.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import pytest

import biotp
from biotp import embeddings, evaluation, release, training

STUB_MODULES = [embeddings, evaluation, release, training]

EXPECTED_FUNCTIONS = {
    "biotp.embeddings": {"load_esm2", "embed_sequences", "cached_embeddings"},
    "biotp.evaluation": {"grouped_split", "spearman", "classification_metrics"},
    "biotp.release": {"build_model_card", "push_to_hub"},
    "biotp.training": {"build_head", "train"},
}


def public_functions(module: ModuleType) -> list[object]:
    """Return functions defined in this module, skipping imports and privates."""
    return [
        obj
        for name, obj in vars(module).items()
        if inspect.isfunction(obj)
        and not name.startswith("_")
        and obj.__module__ == module.__name__
    ]


def module_id(module: ModuleType) -> str:
    return module.__name__


def test_version_is_exported() -> None:
    assert biotp.__version__


@pytest.mark.parametrize("module", STUB_MODULES, ids=module_id)
def test_module_exposes_its_planned_api(module: ModuleType) -> None:
    names = {func.__name__ for func in public_functions(module)}  # type: ignore[attr-defined]
    assert names == EXPECTED_FUNCTIONS[module.__name__]


@pytest.mark.parametrize("module", STUB_MODULES, ids=module_id)
def test_stub_apis_declare_no_default_arguments(module: ModuleType) -> None:
    offenders = sorted(
        f"{func.__name__}({name}={param.default!r})"  # type: ignore[attr-defined]
        for func in public_functions(module)
        for name, param in inspect.signature(func).parameters.items()  # type: ignore[arg-type]
        if param.default is not inspect.Parameter.empty
    )
    assert not offenders, f"defaults added to {module.__name__}: {offenders}"
