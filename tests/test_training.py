"""Tests for biotp.training.

Only `linear_probe` is implemented, so the LoRA and full modes are tested for the
error they raise rather than marked xfail: "this mode is not available yet" is
behavior worth pinning, not a hole in coverage.

Training runs here use tiny synthetic data, so they need no downloads and finish
in well under a second each.
"""

from __future__ import annotations

import inspect
import typing

import numpy as np
import pytest

from biotp import training
from biotp.utils import set_seed

FINETUNE_MODES: list[training.FinetuneMode] = ["linear_probe", "lora", "full"]
UNIMPLEMENTED_MODES: list[training.FinetuneMode] = ["lora", "full"]

N_FEATURES = 12
N_CLASSES = 3


# Class centres are drawn once and shared by every split. Drawing them per split
# would put train, val, and test blobs in unrelated places, making the task
# unlearnable by construction rather than testing anything about the head.
CENTRE_SEED = 1234


def linearly_separable(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Well-separated Gaussian blobs, one per class, so a working head must learn."""
    centres = (
        np.random.default_rng(CENTRE_SEED).normal(size=(N_CLASSES, N_FEATURES)) * 5.0
    )
    rng = np.random.default_rng(seed)
    features = np.concatenate(
        [
            centres[label] + rng.normal(scale=0.3, size=(n_per_class, N_FEATURES))
            for label in range(N_CLASSES)
        ]
    ).astype(np.float32)
    labels = np.repeat(np.arange(N_CLASSES), n_per_class)
    return features, labels


def classification_split() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    train = linearly_separable(60, seed=0)
    val = linearly_separable(20, seed=1)
    test = linearly_separable(20, seed=2)
    return train, val, test


def test_finetune_mode_covers_the_three_planned_regimes() -> None:
    assert set(typing.get_args(training.FinetuneMode)) == set(FINETUNE_MODES)


def test_train_mode_has_no_default() -> None:
    """Every call site must state its regime explicitly rather than inherit one."""
    mode = inspect.signature(training.train).parameters["mode"]
    assert mode.default is inspect.Parameter.empty


def test_build_head_carries_its_task() -> None:
    """The head knows its own task, so train cannot pick a mismatched loss."""
    head = training.build_head(input_dim=8, output_dim=1, task="regression")
    assert head.task == "regression"


def test_build_head_maps_input_width_to_output_width() -> None:
    import torch

    head = training.build_head(
        input_dim=N_FEATURES, output_dim=N_CLASSES, task="classification"
    )
    output = head(torch.zeros(4, N_FEATURES))
    assert output.shape == (4, N_CLASSES)


@pytest.mark.parametrize(
    ("input_dim", "output_dim", "task"),
    [
        (0, 3, "classification"),
        (-1, 3, "classification"),
        (8, 0, "classification"),
    ],
)
def test_build_head_rejects_nonpositive_dimensions(
    input_dim: int, output_dim: int, task: str
) -> None:
    with pytest.raises(AssertionError, match="must be positive"):
        training.build_head(input_dim, output_dim, task)  # type: ignore[arg-type]


def test_build_head_rejects_an_unknown_task() -> None:
    with pytest.raises(AssertionError, match="unknown task"):
        training.build_head(8, 3, "ranking")  # type: ignore[arg-type]


def test_train_learns_a_separable_classification_task() -> None:
    """A head that cannot fit well-separated blobs is broken, not unlucky."""
    set_seed(0)
    train_data, val_data, test_data = classification_split()

    head = training.build_head(N_FEATURES, N_CLASSES, "classification")
    head, history = training.train(
        head, train_data, val_data, mode="linear_probe", max_epochs=60, lr=1e-2
    )

    predictions = training.predict(head, test_data[0])
    accuracy = float((predictions == test_data[1]).mean())
    assert accuracy > 0.9, f"expected a separable task to be learned, got {accuracy}"
    assert history["val_loss"][-1] < history["val_loss"][0]


def test_train_returns_the_documented_history_fields() -> None:
    set_seed(0)
    train_data, val_data, _ = classification_split()
    head = training.build_head(N_FEATURES, N_CLASSES, "classification")
    _, history = training.train(
        head, train_data, val_data, mode="linear_probe", max_epochs=15, lr=1e-2
    )

    assert set(history) >= {
        "train_loss",
        "val_loss",
        "n_train",
        "n_val",
        "mode",
        "best_epoch",
        "best_val_loss",
        "epochs_run",
    }
    assert history["n_train"] == len(train_data[0])
    assert history["epochs_run"] == len(history["val_loss"])


def test_train_restores_the_best_epoch_not_the_last() -> None:
    """Reported weights come from the best validation epoch, so a late overfit
    does not get reported at its worst point."""
    set_seed(0)
    train_data, val_data, _ = classification_split()
    head = training.build_head(N_FEATURES, N_CLASSES, "classification")
    head, history = training.train(
        head, train_data, val_data, mode="linear_probe", max_epochs=40, lr=1e-2
    )

    assert history["best_val_loss"] == pytest.approx(min(history["val_loss"]))
    assert history["best_epoch"] == history["val_loss"].index(min(history["val_loss"]))


def test_train_stops_early_rather_than_running_every_epoch() -> None:
    """Patience should cut a long run short once validation stops improving."""
    set_seed(0)
    train_data, val_data, _ = classification_split()
    head = training.build_head(N_FEATURES, N_CLASSES, "classification")
    _, history = training.train(
        head, train_data, val_data, mode="linear_probe", max_epochs=500, lr=1e-2
    )
    assert history["epochs_run"] < 500


# --- The scaffolding both rungs share ------------------------------------------
#
# `train` and `train_lora` used to carry their own copy of the best-epoch
# bookkeeping and the stopping rule. They are rungs 2 and 3 of a ladder whose
# whole purpose is that they differ in exactly one respect, so a stopping rule
# changed in one and not the other would move the measured delta while every
# test still passed. These tests pin the shared implementation directly, since
# both loops now read the same one.


def tracker_over(
    losses: list[float],
) -> tuple[training._BestEpochTracker[int], list[int]]:
    """Feed a fixed sequence of validation losses.

    Returns the tracker and the epochs its restore callback was handed, which is
    empty until `finish` runs. The snapshot is the index of the epoch that took
    it, so a restore can be checked by value rather than by identity.
    """
    restored: list[int] = []
    epoch = 0

    def snapshot() -> int:
        return epoch

    tracker = training._BestEpochTracker(snapshot, restored.append)
    for epoch, loss in enumerate(losses):
        if tracker.update(loss):
            break
    return tracker, restored


def test_tracker_indexes_epochs_from_zero() -> None:
    """best_epoch is an index into val_loss, which both call sites' tests assume."""
    tracker, _ = tracker_over([0.5])
    assert tracker.best_epoch == 0


def test_tracker_keeps_the_first_of_tied_minima() -> None:
    """A tie must not advance the best epoch.

    Both call sites' tests locate the best epoch with a first-minimum rule
    (`list.index(min(...))` and `np.argmin`), so the improvement test has to stay
    strictly less-than. A `<=` would pass here only by moving those.
    """
    tracker, _ = tracker_over([1.0, 0.5, 0.5, 0.5])
    assert tracker.best_epoch == 1
    assert tracker.best_val == 0.5


def test_tracker_stops_exactly_at_the_patience_boundary() -> None:
    """One epoch earlier or later is a silent change to every reported result."""
    patience = training.EARLY_STOPPING_PATIENCE
    tracker, _ = tracker_over([1.0] + [2.0] * (patience + 5))

    assert tracker.best_epoch == 0
    assert tracker.epochs_seen == patience + 1


def test_tracker_restores_the_snapshot_from_the_best_epoch() -> None:
    """Not the last snapshot taken, and not one taken at a worse epoch."""
    tracker, restored = tracker_over([1.0, 0.5, 0.9])
    history: dict[str, object] = {"val_loss": [1.0, 0.5, 0.9]}

    tracker.finish(history)

    assert restored == [1]
    assert history["best_epoch"] == 1
    assert history["best_val_loss"] == 0.5
    assert history["epochs_run"] == 3


def test_tracker_finish_rejects_a_run_with_no_best_epoch() -> None:
    """A NaN validation loss never improves, so there is nothing to restore.

    `nan < inf` is False, so the patience branch is reachable from the initial
    sentinel and the run ends without a snapshot. Failing here is the point: the
    alternative is reporting whatever weights the last epoch happened to leave.
    """
    tracker, _ = tracker_over([float("nan")] * 40)
    with pytest.raises(AssertionError, match="without a best epoch"):
        tracker.finish({"val_loss": [float("nan")] * tracker.epochs_seen})


def test_tracker_finish_rejects_a_history_that_lost_an_epoch() -> None:
    """The tracker and the loop are two owners of the same epoch count.

    They cannot be merged without giving the tracker the loss lists too, so the
    coupling is made loud instead of left implicit.
    """
    tracker, _ = tracker_over([1.0, 0.5])
    with pytest.raises(AssertionError, match="recorded 1 validation losses"):
        tracker.finish({"val_loss": [1.0]})


def test_train_is_deterministic_for_a_fixed_seed() -> None:
    train_data, val_data, test_data = classification_split()

    def run() -> np.ndarray:
        set_seed(11)
        head = training.build_head(N_FEATURES, N_CLASSES, "classification")
        head, _ = training.train(
            head, train_data, val_data, mode="linear_probe", max_epochs=20, lr=1e-2
        )
        return training.predict(head, test_data[0])

    np.testing.assert_array_equal(run(), run())


def test_train_handles_a_regression_head() -> None:
    set_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 4)).astype(np.float32)
    weights = rng.normal(size=4)
    y = (x @ weights).astype(np.float32)

    head = training.build_head(4, 1, "regression")
    head, history = training.train(
        head,
        (x[:80], y[:80]),
        (x[80:], y[80:]),
        mode="linear_probe",
        max_epochs=80,
        lr=1e-2,
    )

    predictions = training.predict(head, x[80:])
    assert predictions.shape == (40,)
    assert history["val_loss"][-1] < history["val_loss"][0]


@pytest.mark.parametrize("mode", UNIMPLEMENTED_MODES)
def test_train_refuses_unimplemented_modes(mode: training.FinetuneMode) -> None:
    """Asking for LoRA must not silently deliver a linear probe."""
    train_data, val_data, _ = classification_split()
    head = training.build_head(N_FEATURES, N_CLASSES, "classification")

    with pytest.raises(NotImplementedError, match=mode):
        training.train(head, train_data, val_data, mode=mode, max_epochs=1, lr=1e-3)


def test_train_rejects_a_head_without_a_task() -> None:
    """A bare module would otherwise be trained with an arbitrary loss."""
    import torch
    from torch import nn

    bare = nn.Linear(N_FEATURES, N_CLASSES)
    assert not hasattr(bare, "task")
    train_data, val_data, _ = classification_split()

    with pytest.raises(AssertionError, match="task"):
        training.train(
            bare, train_data, val_data, mode="linear_probe", max_epochs=1, lr=1e-3
        )
    assert isinstance(bare, torch.nn.Module)


@pytest.mark.parametrize(
    ("max_epochs", "lr"), [(0, 1e-3), (-1, 1e-3), (5, 0.0), (5, -1.0)]
)
def test_train_rejects_invalid_hyperparameters(max_epochs: int, lr: float) -> None:
    train_data, val_data, _ = classification_split()
    head = training.build_head(N_FEATURES, N_CLASSES, "classification")

    with pytest.raises(AssertionError, match="must be positive"):
        training.train(
            head,
            train_data,
            val_data,
            mode="linear_probe",
            max_epochs=max_epochs,
            lr=lr,
        )


def test_train_rejects_mismatched_features_and_labels() -> None:
    train_data, val_data, _ = classification_split()
    head = training.build_head(N_FEATURES, N_CLASSES, "classification")

    with pytest.raises(AssertionError, match="different lengths"):
        training.train(
            head,
            (train_data[0], train_data[1][:-5]),
            val_data,
            mode="linear_probe",
            max_epochs=1,
            lr=1e-3,
        )
