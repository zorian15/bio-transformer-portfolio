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
    table = analyse.contrast_table(ladder_frame())
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
    table = analyse.contrast_table(frame)

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
    table = analyse.contrast_table(ladder_frame(checkpoints=(SMALL, LARGE)))
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
        analyse.contrast_table(incomplete)


def test_a_degenerate_contrast_reports_no_p_value_rather_than_inventing_one() -> None:
    """Wilcoxon is undefined when every difference is zero.

    Reporting 1.0, or letting scipy's warning path pick something, would put a
    number in the table that no test produced.
    """
    table = analyse.contrast_table(ladder_frame(lora_bonus=0.0))
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
    table = analyse.contrast_table(pd.concat([frame, floor], ignore_index=True))

    row = table[
        (table["contrast"] == "frozen_minus_zero_shot")
        & (table["scheme"] == analyse.ALL_SCHEMES)
    ].iloc[0]

    assert row["n_pairs"] == 8, "the floor did not broadcast across the arms"
    # frozen averages 0.301 over seeds, against a 0.20 floor.
    assert row["mean_delta"] == pytest.approx(0.101)
