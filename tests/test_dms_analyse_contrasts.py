"""Tests for the paired-contrast table.

These numbers are the ones a writeup quotes, so the failure mode is not a crash
but a plausible table computed the wrong way: seeds counted as independent
observations, two model sizes pooled into one contrast, or a contrast averaged
over whichever arms happened to finish. Each of those produces a table that
looks entirely normal, so each gets a test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "dms-benchmark"
    / "scripts"
    / "analyse_contrasts.py"
)

SMALL = "esm2_t12_35M_UR50D"
LARGE = "esm2_t33_650M_UR50D"


def load_module(path: Path, name: str) -> ModuleType:
    assert path.exists(), f"expected a script at {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyse = load_module(SCRIPT_PATH, "dms_analyse_contrasts")


def ladder_frame(
    checkpoints: tuple[str, ...] = (SMALL,),
    lora_bonus: float = 0.10,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    """Both supervised rungs over a small grid, with LoRA ahead by a fixed amount.

    A constant advantage makes the expected contrast exactly `lora_bonus`, so a
    summary that averaged the wrong things lands somewhere else rather than
    somewhere merely different.
    """
    rows = []
    for checkpoint in checkpoints:
        for scheme in ("fold_random_5", "fold_modulo_5"):
            for readout in ("mean", "at_position"):
                for n in (32, 512):
                    for seed in seeds:
                        base = 0.30 + 0.001 * seed
                        for rung, value in (
                            ("frozen", base),
                            ("lora", base + lora_bonus),
                        ):
                            rows.append(
                                {
                                    "rung": rung,
                                    "assay": "ASSAY_A",
                                    "scheme": scheme,
                                    "readout": readout,
                                    "n": n,
                                    "seed": seed,
                                    "checkpoint": checkpoint,
                                    "spearman": value,
                                }
                            )
    return pd.DataFrame(rows)


def table_for(frame: pd.DataFrame) -> pd.DataFrame:
    """`contrast_table` with the script's own bootstrap settings."""
    return analyse.contrast_table(
        frame, analyse.BOOTSTRAP_RESAMPLES, analyse.BOOTSTRAP_SEED
    )


def disagreeing_assays() -> pd.DataFrame:
    """Three assays whose per-assay deltas disagree in sign.

    This is the real 35M situation, and the reason the clustered interval
    exists. Two assays favour LoRA and one favours frozen, so most individual
    rows are positive and Wilcoxon sees a confident effect, while the unit the
    study generalises over, the assay, is split two to one.

    The dissenting assay is deliberately the *smallest* in magnitude. Wilcoxon
    ranks by absolute value, so making it the largest would have pulled the
    signed-rank statistic negative and the test would have passed for the wrong
    reason: it must be a case where the row-level test is genuinely confident
    and positive, and only the clustering disagrees.
    """
    rows = []
    for assay, delta in (("ASSAY_A", 0.10), ("ASSAY_B", 0.08), ("ASSAY_C", -0.03)):
        for scheme in ("fold_random_5", "fold_modulo_5"):
            for readout in ("mean", "at_position"):
                for n in (32, 512):
                    for seed in (0, 1, 2):
                        base = 0.30 + 0.001 * seed
                        for rung, value in (("frozen", base), ("lora", base + delta)):
                            rows.append(
                                {
                                    "rung": rung,
                                    "assay": assay,
                                    "scheme": scheme,
                                    "readout": readout,
                                    "n": n,
                                    "seed": seed,
                                    "checkpoint": SMALL,
                                    "spearman": value,
                                }
                            )
    return pd.DataFrame(rows)


def lora_minus_frozen(table: pd.DataFrame, checkpoint: str) -> pd.Series:
    """The pooled row for one size, which is what a headline quotes."""
    match = table[
        (table["contrast"] == "lora_minus_frozen")
        & (table["checkpoint"] == checkpoint)
        & (table["scheme"] == analyse.ALL_SCHEMES)
    ]
    assert len(match) == 1, f"expected one pooled row, got {len(match)}"
    return match.iloc[0]


def test_seeds_are_averaged_before_pairing_not_counted_as_observations() -> None:
    """Three seeds of one configuration are one experiment, not three.

    Pairing on seed as well would report n=24 where there are 8 configurations,
    which narrows every interval and lowers every p-value for free. Nothing
    about the resulting table looks wrong.
    """
    table = table_for(ladder_frame())
    row = lora_minus_frozen(table, SMALL)

    # 2 schemes x 2 readouts x 2 N, seeds collapsed.
    assert row["n_pairs"] == 8, "seeds were counted as separate observations"
    assert row["mean_delta"] == pytest.approx(0.10)


def test_the_two_sizes_are_never_pooled_into_one_contrast() -> None:
    """The same failure the figures had: averaging across model scale.

    A contrast pooled over sizes answers no question anyone asked, and with a
    real gap between the sizes it would land between the two and look like a
    modest effect at both.
    """
    frame = pd.concat(
        [
            ladder_frame(checkpoints=(SMALL,), lora_bonus=0.02),
            ladder_frame(checkpoints=(LARGE,), lora_bonus=0.20),
        ],
        ignore_index=True,
    )
    table = table_for(frame)

    assert lora_minus_frozen(table, SMALL)["mean_delta"] == pytest.approx(0.02)
    assert lora_minus_frozen(table, LARGE)["mean_delta"] == pytest.approx(0.20)
    # 0.11 is what pooling would give, and it describes neither size.
    for checkpoint in (SMALL, LARGE):
        assert lora_minus_frozen(table, checkpoint)["mean_delta"] != pytest.approx(0.11)


def test_each_size_gets_its_own_per_scheme_breakdown() -> None:
    """The pooled figure alone would have misreported the 35M result.

    That delta survives pooling but holds only under the split sharing residue
    positions between folds, so the per-scheme rows are the finding rather than
    supporting detail.
    """
    table = table_for(ladder_frame(checkpoints=(SMALL, LARGE)))
    schemes = set(
        table[
            (table["contrast"] == "lora_minus_frozen") & (table["checkpoint"] == LARGE)
        ]["scheme"]
    )

    assert schemes == {analyse.ALL_SCHEMES, "fold_random_5", "fold_modulo_5"}


def test_an_incomplete_grid_is_refused_rather_than_averaged() -> None:
    """A missing arm means the contrast describes a different set than it names.

    This is the realistic failure: a SLURM array where some tasks died, then an
    analysis run over whatever shards exist. An outer join would report a mean
    over the survivors under the same column heading.
    """
    frame = ladder_frame()
    incomplete = frame.drop(
        frame[(frame["rung"] == "frozen") & (frame["n"] == 512)].index
    )

    with pytest.raises(AssertionError, match="grid is incomplete"):
        table_for(incomplete)


def test_a_degenerate_contrast_reports_no_p_value_rather_than_inventing_one() -> None:
    """Wilcoxon is undefined when every difference is zero.

    Reporting 1.0, or letting scipy's warning path pick something, would put a
    number in the table that no test produced.
    """
    table = table_for(ladder_frame(lora_bonus=0.0))
    row = lora_minus_frozen(table, SMALL)

    assert row["mean_delta"] == pytest.approx(0.0)
    assert pd.isna(row["p_value"])


def test_the_zero_shot_floor_pairs_without_a_readout_or_n() -> None:
    """Rung 1 has neither, so pairing on them would match nothing at all.

    The question rung 1 answers is how far a supervised arm sits above the
    no-labels floor, so its single value per (checkpoint, assay, scheme) is
    compared against every supervised arm at that size.
    """
    frame = ladder_frame()
    floor = pd.DataFrame(
        [
            {
                "rung": "zero_shot",
                "assay": "ASSAY_A",
                "scheme": scheme,
                "readout": "none",
                "n": 0,
                "seed": 0,
                "checkpoint": SMALL,
                "spearman": 0.20,
            }
            for scheme in ("fold_random_5", "fold_modulo_5")
        ]
    )
    table = table_for(pd.concat([frame, floor], ignore_index=True))

    row = table[
        (table["contrast"] == "frozen_minus_zero_shot")
        & (table["scheme"] == analyse.ALL_SCHEMES)
    ].iloc[0]

    assert row["n_pairs"] == 8, "the floor did not broadcast across the arms"
    # frozen averages 0.301 over seeds, against a 0.20 floor.
    assert row["mean_delta"] == pytest.approx(0.101)


def test_the_interval_resamples_assays_not_rows() -> None:
    """The whole point: Wilcoxon is confident where the design is not.

    With two assays favouring LoRA and one favouring frozen, most individual
    rows are positive, so the signed-rank test on 72 pairs reports a small p.
    But the quantity that would have to generalise is the assay, and those are
    split two to one, so an interval built by resampling assays has to straddle
    zero. A bootstrap that resampled rows instead would inherit the same false
    confidence and agree with Wilcoxon, which is exactly the mistake.
    """
    row = lora_minus_frozen(table_for(disagreeing_assays()), SMALL)

    assert row["n_assays"] == 3
    assert row["p_value"] < 0.05, "the row-level test should look confident here"
    assert row["ci_low"] < 0.0 < row["ci_high"], (
        "the assay-clustered interval excludes zero, so resampling is not "
        "happening at the assay level"
    )
    assert not row["excludes_zero"]


def test_a_consistent_effect_still_excludes_zero() -> None:
    """The interval must not simply be wide for everything.

    A guard that always straddled zero would be useless in the other
    direction, so this pins the case where every assay agrees.
    """
    frame = disagreeing_assays()
    # LoRA rows only. Shifting both rungs moves the level and leaves the
    # delta exactly where it was, which is how the first draft of this test
    # passed nothing at all.
    dissenting = (frame["assay"] == "ASSAY_C") & (frame["rung"] == "lora")
    frame.loc[dissenting, "spearman"] += 0.25

    row = lora_minus_frozen(table_for(frame), SMALL)

    assert row["ci_low"] > 0.0
    assert row["excludes_zero"]


def test_one_assay_gets_no_interval_rather_than_a_fake_one() -> None:
    """Resampling a single cluster returns its own mean every time.

    That would print a zero-width interval, which reads as perfect precision
    rather than as "this cannot be estimated from one assay".
    """
    row = lora_minus_frozen(table_for(ladder_frame()), SMALL)

    assert row["n_assays"] == 1
    assert pd.isna(row["ci_low"]) and pd.isna(row["ci_high"])
    assert pd.isna(row["excludes_zero"])


def test_the_interval_is_reproducible_for_a_fixed_seed() -> None:
    """This writes a committed artifact, so the numbers cannot drift per run."""
    first = lora_minus_frozen(table_for(disagreeing_assays()), SMALL)
    second = lora_minus_frozen(table_for(disagreeing_assays()), SMALL)

    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]
