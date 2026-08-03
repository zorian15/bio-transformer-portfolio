"""Paired contrasts between the ladder's rungs, as a committed artifact.

The numbers a writeup actually quotes are not the per-arm Spearman values in
`frozen.csv` and `lora.csv`: they are the *contrasts* between rungs, the
Wilcoxon p-values on those contrasts, and the same broken out by split scheme.
Until now those were computed ad hoc while drafting, so nothing in the repo
regenerated them and a reader could not trace a cited number to anything. This
script closes that, which matters more with two model sizes in play: the same
table has to be produced twice and the two must not be conflated.

Three points of method, each of which changes the answer:

**Seeds are averaged before pairing, not treated as independent runs.** Three
seeds of one configuration are three draws of the same experiment, so counting
them as three observations would inflate n threefold and narrow every interval
for free. Pairing is on (checkpoint, assay, scheme, readout, N).

**The pairing is exact, and asserted.** A contrast is only meaningful between
two arms that saw identical data, so this refuses to compare anything it cannot
match on every key rather than falling back to an outer join and quietly
averaging over whatever happens to be present.

**Wilcoxon signed-rank rather than a t-test.** Spearman correlations are bounded
and their differences are not close to normal at n=108; the signed-rank test
assumes only symmetry.

Run from the repo root once both supervised rungs have results:

    python projects/dms-benchmark/scripts/analyse_contrasts.py

Writes `results/contrasts.csv`, one row per (checkpoint, contrast, scheme), plus
an `all` scheme row carrying the headline for that size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biotp.runlog import DEFAULT_LOG_DIR, get_logger, run_context

log = get_logger("dms-analyse-contrasts")

# What makes two arms the same experiment. Seed is deliberately absent: it is
# averaged over before pairing, so a pair is one configuration's mean against
# another's, not one run against another.
PAIR_KEYS = ("checkpoint", "assay", "scheme", "readout", "n")

# The contrasts worth reporting, as (name, minuend rung, subtrahend rung). Ridge
# is not here: issue #35 demoted it from a benchmark to a stated caveat, so it
# stays in `frozen_ridge.csv` and in the log rather than in the headline table.
CONTRASTS = (
    ("lora_minus_frozen", "lora", "frozen"),
    ("frozen_minus_zero_shot", "frozen", "zero_shot"),
    ("lora_minus_zero_shot", "lora", "zero_shot"),
)

# Written into the scheme column for the row pooling every scheme. Not a scheme
# name, and chosen so it cannot collide with one.
ALL_SCHEMES = "all"


def seed_averaged(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse seeds, so one row is one configuration rather than one run.

    Three seeds are three draws of the same experiment. Pairing on them
    individually would treat n=324 where there are 108 configurations, which
    narrows every confidence interval and lowers every p-value for nothing.

    Args:
        frame: supervised or zero-shot rows carrying a `spearman` column.

    Returns:
        One row per PAIR_KEYS combination, with `spearman` the mean over seeds.
    """
    return (
        frame.groupby(list(PAIR_KEYS), as_index=False)["spearman"]
        .mean()
        .sort_values(list(PAIR_KEYS))
        .reset_index(drop=True)
    )


def paired(results: pd.DataFrame, left_rung: str, right_rung: str) -> pd.DataFrame:
    """Join two rungs on the keys that make them the same experiment.

    An inner join with an assertion rather than a merge with `how="outer"`: an
    arm present on one rung and missing on the other means an incomplete grid,
    and averaging over whatever survived would report a contrast computed on a
    different set of configurations than the one it names.

    Zero-shot has no readout or N, so when it is one side of the contrast those
    two keys are dropped from the pairing and its single value per (checkpoint,
    assay, scheme) is broadcast across the supervised arms it is compared to.
    That is deliberate: rung 1 is a floor, and the question is how far each
    supervised arm sits above the floor at its own size.

    Args:
        results: all rungs concatenated.
        left_rung: the rung the contrast is measured from.
        right_rung: the rung subtracted.

    Returns:
        One row per pairing key with a `delta` column, empty if either rung is
        absent from `results`.
    """
    left = results[results["rung"] == left_rung]
    right = results[results["rung"] == right_rung]
    if left.empty or right.empty:
        return pd.DataFrame()

    if right_rung == "zero_shot":
        keys = [key for key in PAIR_KEYS if key not in ("readout", "n")]
        right = right.groupby(keys, as_index=False)["spearman"].mean()
        left = seed_averaged(left)
    else:
        keys = list(PAIR_KEYS)
        left, right = seed_averaged(left), seed_averaged(right)

    merged = left.merge(right, on=keys, how="inner", suffixes=("_left", "_right"))
    assert len(merged) == len(left), (
        f"{len(left)} {left_rung} configurations but {len(merged)} matched a "
        f"{right_rung} one on {keys}; the grid is incomplete, and pairing over "
        f"the survivors would report a contrast on a different set of "
        f"configurations than the one it names"
    )

    merged["delta"] = merged["spearman_left"] - merged["spearman_right"]
    return merged


def summarise(deltas: pd.Series) -> dict[str, Any]:
    """Mean, Wilcoxon p, win rate and n for one set of paired differences.

    Wilcoxon signed-rank rather than a t-test: these are differences of bounded
    rank correlations, not normal at this sample size, and the signed-rank test
    assumes only symmetry.

    Args:
        deltas: paired differences, one per configuration.

    Returns:
        The summary fields. `p_value` is None when the test cannot run, which
        happens when every difference is zero or fewer than two pairs exist;
        reporting a p there would be inventing one.
    """
    from scipy import stats

    values = np.asarray(deltas, dtype=float)
    p_value: float | None = None
    if len(values) >= 2 and np.any(values != 0.0):
        p_value = float(stats.wilcoxon(values).pvalue)

    return {
        "n_pairs": len(values),
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "win_rate": float(np.mean(values > 0.0)),
        "p_value": p_value,
    }


def contrast_table(results: pd.DataFrame) -> pd.DataFrame:
    """Every contrast, per checkpoint, pooled and broken out by scheme.

    The per-scheme rows are not decoration. The 35M headline survives pooling
    but holds only under the split that shares residue positions between folds,
    and a table that reported the pooled number alone would state the opposite
    of what the experiment found.

    Args:
        results: all rungs concatenated, as `load_results` returns.

    Returns:
        One row per (checkpoint, contrast, scheme), with an ALL_SCHEMES row
        per contrast and checkpoint carrying the pooled figure.
    """
    rows = []
    for name, left_rung, right_rung in CONTRASTS:
        pairs = paired(results, left_rung, right_rung)
        if pairs.empty:
            log.info(f"skipping {name}: one of its rungs has no results yet")
            continue

        for checkpoint, at_size in pairs.groupby("checkpoint"):
            scheme_groups: list[tuple[str, pd.DataFrame]] = [(ALL_SCHEMES, at_size)] + [
                (str(scheme), rows_) for scheme, rows_ in at_size.groupby("scheme")
            ]
            for scheme, subset in scheme_groups:
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "contrast": name,
                        "scheme": scheme,
                        **summarise(subset["delta"]),
                    }
                )

    assert rows, (
        "no contrast could be computed; at least two rungs must have results "
        "before this table means anything"
    )
    return pd.DataFrame(rows)


def load_results(results_dir: Path) -> pd.DataFrame:
    """Concatenate whichever rung tables exist, so partial runs still report."""
    frames = []
    for rung in ("zero_shot", "frozen", "lora"):
        path = results_dir / f"{rung}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
            log.info(f"loaded {path.name} ({len(frames[-1])} rows)")
        else:
            log.info(f"skipping {path.name}, not present yet")
    assert frames, f"no result tables found under {results_dir}"
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("projects/dms-benchmark/results")
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    with run_context("dms-analyse-contrasts", log_dir=args.log_dir, params=vars(args)):
        results = load_results(args.results_dir)
        table = contrast_table(results)

        destination = args.results_dir / "contrasts.csv"
        table.to_csv(destination, index=False)
        log.info(f"wrote {destination} ({len(table)} rows)")

        for _, row in table[table["scheme"] == ALL_SCHEMES].iterrows():
            p_value = "n/a" if pd.isna(row["p_value"]) else f"{row['p_value']:.2g}"
            log.info(
                f"  {row['checkpoint']} {row['contrast']}: "
                f"{row['mean_delta']:+.4f} (p={p_value}, n={row['n_pairs']})"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
