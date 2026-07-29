"""Leakage-aware evaluation: splits by entity, plus metrics.

The central design choice: splits hold out entities (proteins, families,
epitopes, donors), never random rows, so reported performance reflects
generalization rather than memorization. This is the project's methodological
backbone; see PLANNING.md.
"""

from __future__ import annotations

from typing import Literal


def grouped_split(
    items: list,
    group_key,
    fractions: tuple[float, float, float],
    seed: int,
):
    """Split into train/val/test so no group spans two splits.

    Args:
        items: the records to split.
        group_key: callable mapping a record to its group id (e.g. the epitope,
            the protein family). Membership, not row identity, defines the split.
        fractions: (train, val, test), summing to 1.0. Asserted, not silently
            renormalized.
        seed: RNG seed for reproducibility.

    Returns:
        Three lists of records with disjoint groups.
    """
    raise NotImplementedError


def spearman(y_true, y_pred) -> float:
    """Spearman rank correlation, the primary metric for the DMS benchmark."""
    raise NotImplementedError


def classification_metrics(y_true, y_pred, average: Literal["macro", "micro"]) -> dict:
    """Return accuracy/F1/AUROC as appropriate for a classification task."""
    raise NotImplementedError
