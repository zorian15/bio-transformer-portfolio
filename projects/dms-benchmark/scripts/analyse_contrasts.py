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

# Coverage of the assay-clustered interval.
#
# **Not a bootstrap, deliberately, and this was got wrong once.** The first
# version drew assays with replacement and took the 2.5th and 97.5th percentiles
# of the resampled means. With three clusters that is degenerate: the chance of
# every draw landing on one assay is 1/27 = 3.7%, which is larger than 2.5%, so
# both percentiles collapse onto the smallest and largest per-assay mean. The
# interval was therefore exactly the range of three numbers, the resample count
# and seed did nothing, and it was *narrower* than any honest small-K method
# while being documented as the conservative counterweight to Wilcoxon. Review
# caught it; verified against the committed table, where the reported bounds
# equalled min and max of the assay means on every row.
CONFIDENCE = 0.95


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
    # Both directions. Checking only `len(merged) == len(left)` catches rows
    # missing from the right rung and silently accepts rows missing from the
    # left one, because a shorter left side simply matches itself. That is the
    # wrong way round: the left rung of the headline contrast is LoRA, which is
    # the expensive one and therefore the one whose SLURM tasks actually die.
    # Verified by dropping LoRA rows, which produced a clean table with a
    # quietly reduced n before this was symmetric.
    expected = len(left) if right_rung == "zero_shot" else max(len(left), len(right))
    assert len(merged) == expected, (
        f"{len(left)} {left_rung} and {len(right)} {right_rung} configurations "
        f"produced {len(merged)} pairs on {keys}; the grid is incomplete, and "
        f"pairing over the survivors would report a contrast on a different set "
        f"of configurations than the one it names"
    )

    merged["delta"] = merged["spearman_left"] - merged["spearman_right"]
    return merged


def clustered_interval(
    subset: pd.DataFrame, confidence: float
) -> tuple[float | None, float | None]:
    """Cluster-level t interval, treating each assay as one observation.

    This is the number that qualifies the headline, and it exists because the
    Wilcoxon p beside it is anti-conservative. That test assumes the pairs are
    independent; ours are not. The seed-averaged configurations rest on three
    assays and the per-assay deltas change sign, so the effective sample size
    for a claim about proteins in general is the number of assays, not the
    number of configurations.

    So the estimator is the mean of the \\(K\\) per-assay means, its standard
    error is their sample standard deviation over \\(\\sqrt{K}\\), and the
    interval uses \\(t\\) with \\(K-1\\) degrees of freedom. At \\(K=3\\) that
    multiplier is 4.303, which is why the intervals are wide: three clusters
    genuinely cannot support a narrow one. **That width is the finding**, not a
    defect of the method. An interval that looked tight here would be
    describing precision this cohort does not have.

    Args:
        subset: paired rows carrying `delta` and the `assay` they came from.
        confidence: coverage, e.g. 0.95.

    Returns:
        The interval bounds, or (None, None) with one cluster, where a spread
        cannot be estimated at all and any number printed would be invented.
    """
    from scipy import stats

    sizes = subset.groupby("assay")["delta"].size().to_numpy()
    means = subset.groupby("assay")["delta"].mean().to_numpy()
    if len(means) < 2:
        return (None, None)

    # Equal cluster sizes are what make the mean of the assay means equal the
    # mean over rows that the table reports beside this interval. The grid gives
    # every assay the same configurations, so an imbalance means the results are
    # incomplete rather than that a weighted estimator is needed.
    assert len(set(sizes.tolist())) == 1, (
        f"assays contribute unequal numbers of configurations ({sorted(sizes)}); "
        f"the interval is centred on the mean of assay means, which no longer "
        f"matches the mean over rows reported beside it"
    )

    half_width = stats.t.ppf(0.5 + confidence / 2.0, len(means) - 1) * (
        means.std(ddof=1) / np.sqrt(len(means))
    )
    centre = float(means.mean())
    return (centre - float(half_width), centre + float(half_width))


def summarise(subset: pd.DataFrame, confidence: float) -> dict[str, Any]:
    """Every reported figure for one set of paired differences.

    Wilcoxon signed-rank rather than a paired t-test: these are differences of
    bounded rank correlations, they pile up near zero, and there is no reason to
    expect normality at this sample size. Signed-rank assumes only that the
    difference distribution is symmetric about its centre.

    The win rate is reported beside the mean deliberately. It is nearly
    assumption-free, so a mean that disagrees with it is being carried by a few
    arms rather than by a consistent effect.

    Args:
        subset: paired rows carrying `delta` and `assay`.
        confidence: coverage for the assay-clustered interval.

    Returns:
        The summary fields. `p_value` is None when the test cannot run, which
        happens when every difference is zero or fewer than two pairs exist;
        reporting a p there would be inventing one.
    """
    from scipy import stats

    values = subset["delta"].to_numpy(dtype=float)
    p_value: float | None = None
    if len(values) >= 2 and np.any(values != 0.0):
        p_value = float(stats.wilcoxon(values).pvalue)

    low, high = clustered_interval(subset, confidence)
    assay_means = subset.groupby("assay")["delta"].mean()
    return {
        "n_pairs": len(values),
        "n_assays": int(subset["assay"].nunique()),
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "win_rate": float(np.mean(values > 0.0)),
        "p_value": p_value,
        "ci_low": low,
        "ci_high": high,
        # The raw spread the interval is estimated from. Reported because with
        # three assays these three numbers are more informative than any
        # interval derived from them, and because seeing them makes it obvious
        # when the per-assay effects disagree in sign.
        "assay_min": float(assay_means.min()),
        "assay_max": float(assay_means.max()),
        # The honest headline. A Wilcoxon p below 0.05 whose clustered interval
        # straddles zero is a claim the design cannot support, and that
        # combination is exactly what happened on two of three schemes at 35M.
        # Both bounds tested rather than just `low`: they are always None
        # together, but stating that is what lets a type checker see it.
        "excludes_zero": (
            None if low is None or high is None else bool(low > 0.0 or high < 0.0)
        ),
    }


def contrast_table(results: pd.DataFrame, confidence: float) -> pd.DataFrame:
    """Every contrast, per checkpoint, pooled and broken out by scheme.

    The per-scheme rows are not decoration. The 35M headline survives pooling
    but holds only under the split that shares residue positions between folds,
    and a table that reported the pooled number alone would state the opposite
    of what the experiment found.

    Args:
        results: all rungs concatenated, as `load_results` returns.
        confidence: coverage for the assay-clustered intervals.

    Returns:
        One row per (checkpoint, contrast, scheme), with an ALL_SCHEMES row
        per contrast and checkpoint carrying the pooled figure.
    """
    assert not results["spearman"].isna().any(), (
        f"{int(results['spearman'].isna().sum())} rows carry a NaN spearman; "
        f"every contrast touching them would be NaN and this would still write "
        f"a full-looking table and exit 0"
    )

    # One completeness check covering every contrast, because `paired` cannot do
    # it alone: when the floor is on the right it has far fewer rows by design,
    # so a count comparison there cannot tell a missing LoRA arm from the
    # broadcast. Comparing the two supervised rungs' key sets does, and it is
    # the same grid every contrast draws from.
    supervised = {
        rung: set(
            map(
                tuple,
                seed_averaged(results[results["rung"] == rung])[list(PAIR_KEYS)].values,
            )
        )
        for rung in ("frozen", "lora")
        if not results[results["rung"] == rung].empty
    }
    if len(supervised) == 2:
        missing = supervised["frozen"] ^ supervised["lora"]
        assert not missing, (
            f"{len(missing)} configurations appear on one supervised rung and "
            f"not the other, e.g. {sorted(missing)[:3]}; the grid is incomplete "
            f"and every contrast below would silently describe a subset"
        )

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
                        **summarise(subset, confidence),
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

    with run_context(
        "dms-analyse-contrasts", log_dir=args.log_dir, params=vars(args)
    ) as run:
        results = load_results(args.results_dir)
        table = contrast_table(results, CONFIDENCE)

        destination = args.results_dir / "contrasts.csv"
        table.to_csv(destination, index=False)
        log.info(f"wrote {destination} ({len(table)} rows)")

        run.record("interval_confidence", CONFIDENCE)
        run.record("contrast_rows", len(table))

        for _, row in table[table["scheme"] == ALL_SCHEMES].iterrows():
            p_value = "n/a" if pd.isna(row["p_value"]) else f"{row['p_value']:.2g}"
            interval = (
                "n/a"
                if pd.isna(row["ci_low"])
                else f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
            )
            log.info(
                f"  {row['checkpoint']} {row['contrast']}: "
                f"{row['mean_delta']:+.4f} (p={p_value}, n={row['n_pairs']}) "
                f"assay-clustered CI {interval}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
