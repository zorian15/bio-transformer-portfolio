"""Tests for biotp.evaluation.

This is the module where tests matter most: leakage-aware splitting is the
methodological backbone of all three projects, so group disjointness, complete
partitioning, and seed determinism are the properties held down hardest here.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal

import pytest

from biotp import evaluation

# Ten records over five families, so a group-aware split has something to do.
RECORDS: list[dict[str, Any]] = [
    {"id": i, "family": f"family_{i % 5}"} for i in range(10)
]
FRACTIONS = (0.6, 0.2, 0.2)
AVERAGING_MODES: list[Literal["macro", "micro"]] = ["macro", "micro"]


def family_of(record: dict[str, Any]) -> str:
    value = record["family"]
    assert isinstance(value, str)
    return value


def test_grouped_split_seed_has_no_default() -> None:
    """Reproducibility should be stated at the call site, not inherited silently."""
    seed = inspect.signature(evaluation.grouped_split).parameters["seed"]
    assert seed.default is inspect.Parameter.empty


def test_grouped_split_keeps_groups_disjoint() -> None:
    """The whole point: no family may appear in more than one split."""
    splits = evaluation.grouped_split(RECORDS, family_of, FRACTIONS, seed=0)
    train, val, test = ({family_of(r) for r in split} for split in splits)

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_grouped_split_partitions_every_record_exactly_once() -> None:
    splits = evaluation.grouped_split(RECORDS, family_of, FRACTIONS, seed=0)
    ids = [record["id"] for split in splits for record in split]

    assert sorted(ids) == sorted(record["id"] for record in RECORDS)


def test_grouped_split_is_deterministic_for_a_fixed_seed() -> None:
    first = evaluation.grouped_split(RECORDS, family_of, FRACTIONS, seed=7)
    second = evaluation.grouped_split(RECORDS, family_of, FRACTIONS, seed=7)
    assert first == second


def test_grouped_split_depends_on_the_seed() -> None:
    """Different seeds should generally give different assignments."""
    splits = [
        evaluation.grouped_split(RECORDS, family_of, FRACTIONS, seed=seed)
        for seed in range(8)
    ]
    assert (
        len({tuple(tuple(r["id"] for r in s) for s in split) for split in splits}) > 1
    )


def test_grouped_split_ignores_input_order() -> None:
    """A shuffled input list must not change a seeded split."""
    forward = evaluation.grouped_split(RECORDS, family_of, FRACTIONS, seed=3)
    backward = evaluation.grouped_split(
        list(reversed(RECORDS)), family_of, FRACTIONS, seed=3
    )
    as_families = [{family_of(r) for r in split} for split in forward]
    assert as_families == [{family_of(r) for r in split} for split in backward]


def test_grouped_split_approximates_the_requested_fractions() -> None:
    """Whole groups cannot hit exact proportions, but should land close."""
    many = [{"id": i, "family": f"family_{i % 50}"} for i in range(500)]
    train, val, test = evaluation.grouped_split(many, family_of, FRACTIONS, seed=0)

    assert len(train) / len(many) == pytest.approx(0.6, abs=0.08)
    assert len(val) / len(many) == pytest.approx(0.2, abs=0.08)
    assert len(test) / len(many) == pytest.approx(0.2, abs=0.08)


def test_grouped_split_supports_a_zero_fraction() -> None:
    """Asking for train/val only is expressed as a 0.0 third fraction."""
    train, val, test = evaluation.grouped_split(
        RECORDS, family_of, (0.8, 0.2, 0.0), seed=0
    )
    assert test == []
    assert len(train) + len(val) == len(RECORDS)


@pytest.mark.parametrize(
    "fractions", [(0.5, 0.2, 0.2), (0.6, 0.3, 0.2), (1.0, 0.1, 0.0)]
)
def test_grouped_split_rejects_fractions_that_do_not_sum_to_one(
    fractions: tuple[float, float, float],
) -> None:
    """Bad fractions fail loudly; they are never quietly renormalized."""
    with pytest.raises(AssertionError, match="sum to 1.0"):
        evaluation.grouped_split(RECORDS, family_of, fractions, seed=0)


def test_grouped_split_rejects_negative_fractions() -> None:
    with pytest.raises(AssertionError, match=">= 0"):
        evaluation.grouped_split(RECORDS, family_of, (1.2, -0.2, 0.0), seed=0)


def test_grouped_split_rejects_empty_input() -> None:
    with pytest.raises(AssertionError, match="no items"):
        evaluation.grouped_split([], family_of, FRACTIONS, seed=0)


def test_grouped_split_fails_loudly_when_groups_cannot_fill_a_split() -> None:
    """One group cannot be spread across three requested splits."""
    single_group = [{"id": i, "family": "only"} for i in range(5)]
    with pytest.raises(AssertionError, match="received no items"):
        evaluation.grouped_split(single_group, family_of, FRACTIONS, seed=0)


def test_spearman_is_one_for_a_monotonic_relationship() -> None:
    assert evaluation.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_is_minus_one_for_a_reversed_relationship() -> None:
    assert evaluation.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_ignores_monotonic_rescaling() -> None:
    """A rank correlation must not care about the scale of the predictions."""
    y_true = [1, 2, 3, 4]
    assert evaluation.spearman(y_true, [1, 4, 9, 16]) == pytest.approx(
        evaluation.spearman(y_true, [2, 8, 18, 32])
    )


def test_spearman_rejects_constant_input() -> None:
    """Correlation with a constant is undefined, so it must not return nan."""
    with pytest.raises(AssertionError, match="undefined"):
        evaluation.spearman([1, 2, 3, 4], [5, 5, 5, 5])


def test_spearman_rejects_mismatched_lengths() -> None:
    with pytest.raises(AssertionError, match="mismatched lengths"):
        evaluation.spearman([1, 2, 3], [1, 2])


@pytest.mark.parametrize("average", AVERAGING_MODES)
def test_classification_metrics_returns_the_expected_keys(
    average: Literal["macro", "micro"],
) -> None:
    metrics = evaluation.classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], average)
    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "n",
    }
    assert metrics["n"] == 4


def test_classification_metrics_scores_a_perfect_prediction_as_one() -> None:
    labels = ["a", "b", "c", "a"]
    metrics = evaluation.classification_metrics(labels, labels, "macro")
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_classification_metrics_computes_accuracy_correctly() -> None:
    metrics = evaluation.classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], "macro")
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_classification_metrics_macro_f1_punishes_ignoring_a_rare_class() -> None:
    """The reason macro-F1 is the headline metric on imbalanced data."""
    y_true = ["common"] * 9 + ["rare"]
    always_common = ["common"] * 10

    metrics = evaluation.classification_metrics(y_true, always_common, "macro")
    assert metrics["accuracy"] == pytest.approx(0.9)
    # Perfect on one class, zero on the other, so macro-F1 lands near 0.47.
    assert metrics["f1"] < 0.5


def test_classification_metrics_rejects_an_unknown_average() -> None:
    with pytest.raises(AssertionError, match="unsupported average"):
        evaluation.classification_metrics([0, 1], [0, 1], "weighted")  # type: ignore[arg-type]


def test_classification_metrics_rejects_empty_input() -> None:
    with pytest.raises(AssertionError, match="no observations"):
        evaluation.classification_metrics([], [], "macro")


def test_per_class_f1_reports_every_class_including_the_missed_ones() -> None:
    scores = evaluation.per_class_f1(["a", "b", "c"], ["a", "a", "c"])
    assert set(scores) == {"a", "b", "c"}
    assert scores["c"] == pytest.approx(1.0)
    assert scores["b"] == pytest.approx(0.0)


def test_per_class_f1_includes_classes_only_ever_predicted() -> None:
    """A hallucinated class must be visible, not silently dropped."""
    scores = evaluation.per_class_f1(["a", "a"], ["a", "z"])
    assert "z" in scores


def test_majority_class_accuracy_matches_the_largest_share() -> None:
    labels = ["a"] * 7 + ["b"] * 3
    assert evaluation.majority_class_accuracy(labels) == pytest.approx(0.7)


def test_majority_class_accuracy_rejects_empty_input() -> None:
    with pytest.raises(AssertionError, match="no observations"):
        evaluation.majority_class_accuracy([])
