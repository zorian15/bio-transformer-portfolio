"""Leakage-aware evaluation: splits by entity, plus metrics.

The central design choice: splits hold out entities (proteins, families,
epitopes, donors), never random rows, so reported performance reflects
generalization rather than memorization. This is the project's methodological
backbone; see PLANNING.md.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np


def grouped_split(
    items: list,
    group_key: Callable[[Any], Any],
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[list, list, list]:
    """Split into train/val/test so no group spans two splits.

    Args:
        items: the records to split.
        group_key: callable mapping a record to its group id (e.g. the epitope,
            the protein family). Membership, not row identity, defines the split.
        fractions: (train, val, test), summing to 1.0. Asserted, not silently
            renormalized. A fraction of exactly 0.0 yields an empty split, which
            is how a caller requests train/val only.
        seed: RNG seed for reproducibility.

    Returns:
        Three lists of records with disjoint groups.

    Groups are shuffled in a canonical order before assignment, so the result
    depends on the seed but not on the order the items happened to arrive in.
    Each group then goes to whichever split is furthest below its target size, so
    realized sizes approximate `fractions` as closely as whole groups allow. Exact
    proportions are impossible with grouped data: that is the price of not
    leaking, so read the realized sizes rather than assuming them.
    """
    assert items, "grouped_split received no items"
    assert len(fractions) == 3, f"expected 3 fractions, got {len(fractions)}"
    assert all(fraction >= 0.0 for fraction in fractions), "fractions must be >= 0"
    total = sum(fractions)
    assert abs(total - 1.0) < 1e-9, f"fractions must sum to 1.0, got {total}"

    groups: dict[Any, list] = {}
    for item in items:
        groups.setdefault(group_key(item), []).append(item)

    # Sort by string form for a canonical starting order across mixed key types,
    # so the shuffle depends only on the seed.
    group_ids = sorted(groups, key=str)
    random.Random(seed).shuffle(group_ids)

    targets = [fraction * len(items) for fraction in fractions]
    splits: list[list] = [[], [], []]

    for group_id in group_ids:
        eligible = [index for index in range(3) if targets[index] > 0]
        assert eligible, "no split has a positive fraction"
        chosen = max(eligible, key=lambda index: targets[index] - len(splits[index]))
        splits[chosen].extend(groups[group_id])

    for index, fraction in enumerate(fractions):
        assert not (fraction > 0 and not splits[index]), (
            f"split {index} was given fraction {fraction} but received no items; "
            f"there are only {len(group_ids)} groups to distribute"
        )

    assigned = sum(len(split) for split in splits)
    assert assigned == len(items), f"split {assigned} items from {len(items)}"
    return splits[0], splits[1], splits[2]


def spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Spearman rank correlation, the primary metric for the DMS benchmark."""
    from scipy import stats

    assert len(y_true) == len(y_pred), "spearman got mismatched lengths"
    assert len(y_true) >= 2, "spearman needs at least two observations"

    rho = float(stats.spearmanr(y_true, y_pred).statistic)
    assert np.isfinite(rho), "spearman is undefined, likely a constant input"
    return rho


def classification_metrics(
    y_true: Sequence[Any], y_pred: Sequence[Any], average: Literal["macro", "micro"]
) -> dict:
    """Return accuracy and averaged precision/recall/F1 for predicted labels.

    `average` selects how per-class scores are pooled: "macro" weights every class
    equally, which is what the imbalanced localization task is judged on, while
    "micro" weights every protein equally and so tracks accuracy.

    AUROC is deliberately absent. It needs predicted scores rather than the hard
    labels this function takes, so computing it here would mean inventing them.
    Balanced accuracy stands in as a rare-class-sensitive companion to accuracy
    that hard labels do support.

    Undefined per-class scores (a class never predicted) count as 0 rather than
    raising, since with rare classes that is an expected outcome, not a bug.
    """
    from sklearn import metrics

    assert average in ("macro", "micro"), f"unsupported average {average!r}"
    assert len(y_true) == len(y_pred), "metrics got mismatched lengths"
    assert len(y_true) > 0, "metrics received no observations"

    return {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "precision": float(
            metrics.precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            metrics.recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(metrics.f1_score(y_true, y_pred, average=average, zero_division=0)),
        "n": len(y_true),
    }


def per_class_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, float]:
    """Return F1 per class, so a gain confined to common classes stays visible.

    Classes come from the union of true and predicted labels, so a class that
    appears only in predictions still shows up rather than being hidden.
    """
    from sklearn import metrics

    assert len(y_true) == len(y_pred), "per_class_f1 got mismatched lengths"
    assert len(y_true) > 0, "per_class_f1 received no observations"

    labels = sorted({*y_true, *y_pred}, key=str)
    scores = metrics.f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {str(label): float(score) for label, score in zip(labels, scores)}


def majority_class_accuracy(y_true: Sequence[Any]) -> float:
    """Accuracy of always predicting the most common class.

    The floor any real method has to clear. On this localization task it is 0.292,
    high enough that raw accuracy can look respectable while the model ignores
    every rare compartment.
    """
    assert len(y_true) > 0, "majority_class_accuracy received no observations"

    counts: dict[Any, int] = {}
    for label in y_true:
        counts[label] = counts.get(label, 0) + 1
    return max(counts.values()) / len(y_true)
