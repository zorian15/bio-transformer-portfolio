"""Tests for the dms-benchmark ladder runner.

Offline, against small hand-built frames. What is worth testing here is the split
and subsample logic: both would keep producing plausible Spearman values while
measuring something other than what the pre-registration says.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "dms-benchmark"
    / "scripts"
    / "run_arms.py"
)


def load_run_arms() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dms_run_arms", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_arms = load_run_arms()


def assay_frame(position_of_fold: dict[int, list[int]]) -> pd.DataFrame:
    """Build an assay whose fold assignment and positions are chosen by hand."""
    rows = []
    for fold, positions in position_of_fold.items():
        for position in positions:
            rows.append(
                {
                    "position": position,
                    "DMS_score": float(position),
                    "fold_random_5": fold,
                    "fold_modulo_5": fold,
                    "fold_contiguous_5": fold,
                }
            )
    return pd.DataFrame(rows)


def spread(folds: int = 5, per_fold: int = 6) -> pd.DataFrame:
    """Position-disjoint folds, as modulo and contiguous both guarantee."""
    return assay_frame(
        {
            fold: list(range(fold * per_fold, (fold + 1) * per_fold))
            for fold in range(folds)
        }
    )


def test_splits_assign_every_fold_to_a_role() -> None:
    splits = run_arms.make_splits(spread(), "TEST", "fold_modulo_5")
    assert len(splits.test) == 6
    assert len(splits.val) == 6
    assert len(splits.train_pool) == 18
    total = len(splits.test) + len(splits.val) + len(splits.train_pool)
    assert total == 30, "every row belongs to exactly one role"


def test_splits_reject_position_leakage_under_modulo() -> None:
    """The property that makes the scheme worth running.

    If a position appeared in both the training pool and the test fold, the arm
    would measure memorization of site effects and report it as generalization,
    with nothing anomalous in the number.
    """
    leaky = assay_frame({0: [1, 2, 3], 1: [4, 5, 6], 2: [1, 7, 8], 3: [9], 4: [10]})
    with pytest.raises(AssertionError, match="hold out residue positions"):
        run_arms.make_splits(leaky, "TEST", "fold_modulo_5")


def test_splits_allow_shared_positions_under_random() -> None:
    """`random` is supposed to share positions, so the guard must not fire."""
    shared = assay_frame({0: [1, 2, 3], 1: [1, 2, 3], 2: [1, 2, 3], 3: [1], 4: [2]})
    splits = run_arms.make_splits(shared, "TEST", "fold_random_5")
    assert len(splits.train_pool) == 5


def test_splits_reject_an_unknown_scheme() -> None:
    with pytest.raises(AssertionError, match="unknown scheme"):
        run_arms.make_splits(spread(), "TEST", "fold_made_up_5")


# --- Subsampling --------------------------------------------------------------


POOL = np.arange(100)


def test_subsample_is_reproducible_across_processes() -> None:
    """Frozen values, which is what makes this catch the bug it was written for.

    An earlier version seeded the draw with `hash(assay)`. Python randomizes
    string hashing per process, so every invocation would have trained on a
    different subset while every downstream number stayed plausible. Asserting
    only that two calls agree would not have caught it, because within one
    process they do. A golden value fails the moment the seed stops being a pure
    function of the configuration.
    """
    drawn = run_arms.subsample(POOL, 5, "ASSAY_A", "fold_modulo_5", 0)
    assert sorted(drawn.tolist()) == sorted(
        run_arms.subsample(POOL, 5, "ASSAY_A", "fold_modulo_5", 0).tolist()
    )
    assert drawn.tolist() == [14, 78, 9, 73, 62]


def test_subsample_varies_with_every_part_of_the_configuration() -> None:
    base = run_arms.subsample(POOL, 8, "ASSAY_A", "fold_modulo_5", 0).tolist()
    assert base != run_arms.subsample(POOL, 8, "ASSAY_B", "fold_modulo_5", 0).tolist()
    assert base != run_arms.subsample(POOL, 8, "ASSAY_A", "fold_random_5", 0).tolist()
    assert base != run_arms.subsample(POOL, 8, "ASSAY_A", "fold_modulo_5", 1).tolist()


def test_subsample_draws_without_replacement() -> None:
    drawn = run_arms.subsample(POOL, 40, "ASSAY_A", "fold_modulo_5", 0)
    assert len(set(drawn.tolist())) == 40


def test_subsample_refuses_to_exceed_the_pool() -> None:
    """A quietly smaller training set would flatten the data-efficiency curve."""
    with pytest.raises(AssertionError, match="asked for"):
        run_arms.subsample(np.arange(10), 32, "ASSAY_A", "fold_modulo_5", 0)


def test_the_ladder_shares_one_checkpoint() -> None:
    """The invariant the rung-2-to-rung-3 delta depends on.

    Rung 1 additionally reports a second size, which is a separate arm rather
    than a substitution inside the ladder.
    """
    assert run_arms.LADDER_CHECKPOINT in run_arms.ZERO_SHOT_CHECKPOINTS
    assert len(run_arms.ZERO_SHOT_CHECKPOINTS) == 2


def test_the_grid_covers_the_pre_registered_axes() -> None:
    configs = list(run_arms.grid("lora", ("A", "B", "C")))
    assert len(configs) == 3 * 3 * 3 * 4 * 3, "assays x schemes x readouts x N x seeds"
    assert {c.checkpoint for c in configs} == {run_arms.LADDER_CHECKPOINT}
    assert {c.n for c in configs} == set(run_arms.TRAINING_SIZES)
    assert {c.seed for c in configs} == set(run_arms.SEEDS)


def test_validation_is_capped_and_stable() -> None:
    """Validation exists to pick a stopping epoch, not to be a second test set.

    Uncapped, the fine-tuned rung re-encodes the whole ~1000-variant fold every
    epoch, which at N=32 is thirty times more work than training. The cap has to
    be fixed per (assay, scheme) and independent of N and seed, so every arm
    selects its stopping epoch against the same variants.
    """
    big = assay_frame(
        {fold: list(range(fold * 400, (fold + 1) * 400)) for fold in range(5)}
    )
    first = run_arms.make_splits(big, "ASSAY_A", "fold_modulo_5")
    again = run_arms.make_splits(big, "ASSAY_A", "fold_modulo_5")

    assert len(first.val) == run_arms.VAL_SUBSAMPLE
    assert first.val.tolist() == again.val.tolist(), "must not vary between calls"
    assert (
        run_arms.make_splits(big, "ASSAY_B", "fold_modulo_5").val.tolist()
        != first.val.tolist()
    ), "different assays should not share a draw"


def test_a_small_validation_fold_is_left_alone() -> None:
    splits = run_arms.make_splits(spread(), "ASSAY_A", "fold_modulo_5")
    assert len(splits.val) == 6, "smaller than the cap, so untouched"
