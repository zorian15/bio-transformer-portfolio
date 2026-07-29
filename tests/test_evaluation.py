"""Contract tests for biotp.evaluation (currently a scaffold stub).

This is the module where the tests matter most: leakage-aware splitting is the
methodological backbone of all three projects, so the group-disjointness and
determinism assertions below are the ones to keep once grouped_split is real.

See test_embeddings.py for how the `stub` marker and xfail_strict interact.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal

import pytest

from biotp import evaluation

stub = pytest.mark.xfail(
    raises=NotImplementedError, reason="evaluation is a scaffold stub"
)

# Ten records spread over five families, so a group-aware split has something to do.
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


@stub
def test_grouped_split_keeps_groups_disjoint() -> None:
    """The whole point: no family may appear in more than one split."""
    splits = evaluation.grouped_split(RECORDS, family_of, FRACTIONS, seed=0)
    train, val, test = ({family_of(r) for r in split} for split in splits)

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


@stub
def test_grouped_split_partitions_every_record_exactly_once() -> None:
    """Splitting must lose no records and duplicate none."""
    splits = evaluation.grouped_split(RECORDS, family_of, FRACTIONS, seed=0)
    ids = [record["id"] for split in splits for record in split]

    assert sorted(ids) == sorted(record["id"] for record in RECORDS)


@stub
def test_grouped_split_is_deterministic_for_a_fixed_seed() -> None:
    first = evaluation.grouped_split(RECORDS, family_of, FRACTIONS, seed=7)
    second = evaluation.grouped_split(RECORDS, family_of, FRACTIONS, seed=7)
    assert first == second


@stub
def test_grouped_split_rejects_fractions_that_do_not_sum_to_one() -> None:
    """Bad fractions fail loudly; they are never quietly renormalized."""
    with pytest.raises(AssertionError):
        evaluation.grouped_split(RECORDS, family_of, (0.5, 0.2, 0.2), seed=0)


@stub
def test_spearman_is_one_for_a_monotonic_relationship() -> None:
    assert evaluation.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


@stub
def test_spearman_is_minus_one_for_a_reversed_relationship() -> None:
    assert evaluation.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


@stub
def test_spearman_ignores_monotonic_rescaling() -> None:
    """A rank correlation must not care about the scale of the predictions."""
    y_true = [1, 2, 3, 4]
    assert evaluation.spearman(y_true, [1, 4, 9, 16]) == pytest.approx(
        evaluation.spearman(y_true, [2, 8, 18, 32])
    )


@stub
@pytest.mark.parametrize("average", AVERAGING_MODES)
def test_classification_metrics_returns_a_metric_mapping(
    average: Literal["macro", "micro"],
) -> None:
    metrics = evaluation.classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], average)
    assert isinstance(metrics, dict)
    assert metrics
