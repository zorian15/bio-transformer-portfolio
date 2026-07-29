"""Contract tests for biotp.training (currently a scaffold stub).

See test_embeddings.py for how the `stub` marker and xfail_strict interact.
"""

from __future__ import annotations

import inspect
import typing

import pytest

from biotp import training

stub = pytest.mark.xfail(
    raises=NotImplementedError, reason="training is a scaffold stub"
)

FINETUNE_MODES: list[training.FinetuneMode] = ["linear_probe", "lora", "full"]


def test_finetune_mode_covers_the_three_planned_regimes() -> None:
    assert set(typing.get_args(training.FinetuneMode)) == set(FINETUNE_MODES)


def test_train_mode_has_no_default() -> None:
    """Every call site must state its regime explicitly rather than inherit one."""
    mode = inspect.signature(training.train).parameters["mode"]
    assert mode.default is inspect.Parameter.empty


@stub
def test_build_head_supports_regression() -> None:
    assert (
        training.build_head(input_dim=480, output_dim=1, task="regression") is not None
    )


@stub
def test_build_head_supports_classification() -> None:
    head = training.build_head(input_dim=640, output_dim=10, task="classification")
    assert head is not None


@stub
@pytest.mark.parametrize("mode", FINETUNE_MODES)
def test_train_supports_every_finetune_mode(mode: training.FinetuneMode) -> None:
    """linear_probe is the MVP regime; lora and full are the planned ramp."""
    model = training.build_head(input_dim=480, output_dim=1, task="regression")
    fitted, history = training.train(
        model, train_data=[], val_data=[], mode=mode, max_epochs=1, lr=1e-3
    )
    assert fitted is not None
    assert history is not None
