"""Tests for the dms-benchmark figure generation.

The figures had no tests at all before issue #34, which is how they came to
average two model sizes into one curve without anyone noticing. Plotting code
is easy to leave untested because a wrong figure still renders, and the repo's
own convention is that a figure disagreeing with the table beside it means the
script was not re-run. A figure that silently merges an axis disagrees with no
table: it is wrong on its own terms and looks fine.

Only the summarisation is tested here, not the rendering. That is deliberate:
the summarisation is where a result can be misstated, and asserting on SVG
output would pin styling choices that are meant to change freely.
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
    / "make_figures.py"
)

RUN_ARMS_PATH = SCRIPT_PATH.parent / "run_arms.py"

SMALL = "esm2_t12_35M_UR50D"
LARGE = "esm2_t33_650M_UR50D"


def load_module(path: Path, name: str) -> ModuleType:
    """Import a project script by path, since neither is an installed module."""
    assert path.exists(), f"expected a script at {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


make_figures = load_module(SCRIPT_PATH, "dms_make_figures")
run_arms = load_module(RUN_ARMS_PATH, "dms_run_arms_for_figures")


def two_size_cell() -> pd.DataFrame:
    """One assay and scheme, both rungs, both sizes, two seeds each.

    The two sizes are given deliberately separated Spearman values, so a
    summary that averaged across them would land halfway between and be
    obviously wrong rather than plausibly wrong.
    """
    rows = []
    for checkpoint, base in ((SMALL, 0.20), (LARGE, 0.60)):
        for rung, bump in (("frozen", 0.0), ("lora", 0.05)):
            for n in (32, 2048):
                for seed, jitter in enumerate((-0.01, 0.01)):
                    rows.append(
                        {
                            "assay": "ASSAY_A",
                            "scheme": "fold_modulo_5",
                            "rung": rung,
                            "readout": "mean",
                            "n": n,
                            "seed": seed,
                            "checkpoint": checkpoint,
                            "spearman": base + bump + jitter,
                        }
                    )
    return pd.DataFrame(rows)


def test_curve_summary_keeps_the_two_sizes_apart() -> None:
    """The bug this file exists for: `groupby("n")` merged 35M with 650M.

    The old summary grouped by training size alone, so each point averaged a
    35M and a 650M run and the gap between them landed in the error bar. The
    figure still rendered, and a reader would have read a scale difference as
    seed noise.
    """
    summary = make_figures.curve_summary(two_size_cell())

    small = summary[
        (summary["checkpoint"] == SMALL)
        & (summary["rung"] == "frozen")
        & (summary["n"] == 32)
    ]
    large = summary[
        (summary["checkpoint"] == LARGE)
        & (summary["rung"] == "frozen")
        & (summary["n"] == 32)
    ]

    assert len(small) == 1 and len(large) == 1
    assert small["mean"].item() == pytest.approx(0.20)
    assert large["mean"].item() == pytest.approx(0.60)
    # The midpoint is what the old code produced. Naming it makes the
    # regression unmistakable if this ever collapses again.
    assert small["mean"].item() != pytest.approx(0.40)


def test_curve_summary_still_averages_seeds() -> None:
    """Separating the checkpoints must not stop seeds from being averaged.

    Seeds are the thing the error bar is supposed to describe. A summary keyed
    on seed as well would draw one point per run with no spread at all.
    """
    summary = make_figures.curve_summary(two_size_cell())
    row = summary[
        (summary["checkpoint"] == LARGE)
        & (summary["rung"] == "lora")
        & (summary["n"] == 2048)
    ]

    assert len(row) == 1, "one point per (checkpoint, rung, n), not one per seed"
    assert row["mean"].item() == pytest.approx(0.65)
    assert row["std"].item() > 0, "the spread across seeds is what the bar shows"


def test_readout_summary_keeps_the_two_sizes_apart() -> None:
    """The other half of the same bug, and the half review caught me missing.

    `curve_summary` was extracted and tested; the bar figure's aggregation was
    left inline, so nothing executed it and a regression to the flat
    `cell[...]["spearman"].mean()` it replaced would have kept the suite green
    while merging the two sizes into one bar per readout.
    """
    summary = make_figures.readout_summary(two_size_cell())

    small = summary[
        (summary["checkpoint"] == SMALL)
        & (summary["rung"] == "lora")
        & (summary["readout"] == "mean")
    ]
    large = summary[
        (summary["checkpoint"] == LARGE)
        & (summary["rung"] == "lora")
        & (summary["readout"] == "mean")
    ]

    assert len(small) == 1 and len(large) == 1
    assert small["mean"].item() == pytest.approx(0.25)
    assert large["mean"].item() == pytest.approx(0.65)
    # 0.45 is the midpoint a flat mean over the cell would have produced.
    assert small["mean"].item() != pytest.approx(0.45)


def test_readout_summary_gives_one_bar_per_rung_and_size() -> None:
    """Four bars per readout with two sizes, not two.

    The bar count is the visible symptom of the merge: two bars where there
    should be four is a figure that silently answers a different question.
    """
    summary = make_figures.readout_summary(two_size_cell())
    per_readout = summary[summary["readout"] == "mean"]

    assert len(per_readout) == 4, "expected frozen and lora at each of two sizes"
    assert set(zip(per_readout["rung"], per_readout["checkpoint"])) == {
        ("frozen", SMALL),
        ("frozen", LARGE),
        ("lora", SMALL),
        ("lora", LARGE),
    }


def test_every_size_the_grid_runs_has_a_distinct_encoding() -> None:
    """Cross-module guard: adding a size to the ladder must reach the figures.

    `SUPERVISED_CHECKPOINTS` is what the grid runs; these maps are what the
    figures can draw. Adding a third size to the grid without touching this
    file would otherwise plot it with whatever the lookup fell back to, and two
    sizes sharing an encoding are indistinguishable rather than absent, which
    is the harder failure to spot.

    All three maps, not just the linestyle. The hatch used to be assigned by
    position ("" for the first size, "//" for every other), so a third size
    shared the second one's bars: the same merge-two-sizes failure this file
    exists to prevent, one figure over. Review caught it.
    """
    # Both tuples, not only the supervised one. Rung 1 keeps its own list and is
    # the one most likely to grow alone, since a zero-shot arm at a new size is
    # nearly free, and the figure draws its floors from that list.
    every_size = tuple(
        dict.fromkeys(run_arms.SUPERVISED_CHECKPOINTS + run_arms.ZERO_SHOT_CHECKPOINTS)
    )
    for checkpoint in every_size:
        assert checkpoint in make_figures.CHECKPOINT_LABELS
        assert checkpoint in make_figures.CHECKPOINT_STYLES
        assert checkpoint in make_figures.CHECKPOINT_HATCHES

    for name in ("CHECKPOINT_LABELS", "CHECKPOINT_STYLES", "CHECKPOINT_HATCHES"):
        encoding = getattr(make_figures, name)
        drawn = [encoding[c] for c in every_size]
        assert len(set(drawn)) == len(drawn), f"two sizes share a value in {name}"


def test_the_two_figures_order_the_sizes_the_same_way() -> None:
    """One helper, because the two figures had reached for different expressions.

    The curves sorted the checkpoint values alphabetically and the bars filtered
    the style registry. Those agree only because "esm2_t12_35M" happens to sort
    before "esm2_t33_650M", which is a coincidence of naming rather than
    anything guaranteed, so the two figures would have disagreed about line and
    bar order the first time a size was added that sorted differently.
    """
    frame = two_size_cell()
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)

    order = make_figures.present_checkpoints(frame)

    assert order == list(make_figures.CHECKPOINT_STYLES)
    assert make_figures.present_checkpoints(reversed_frame) == order, (
        "the order depends on how the rows happen to be arranged, so two "
        "figures reading the same results can disagree"
    )


def test_present_checkpoints_refuses_a_size_it_cannot_draw() -> None:
    """The guard has to fire wherever an unknown size first reaches a figure.

    `checkpoint_label` asserts, but the bar figure indexes the hatch map
    directly, so a results file carrying an unrecognised checkpoint would have
    raised a bare KeyError there rather than the explanation.
    """
    frame = two_size_cell()
    frame.loc[0, "checkpoint"] = "esm2_t36_3B_UR50D"

    with pytest.raises(AssertionError, match="no figure label"):
        make_figures.present_checkpoints(frame)


def test_an_unknown_checkpoint_is_refused_rather_than_drawn() -> None:
    """Fail loudly, per the repo convention, instead of picking a default.

    A silent fallback here would draw the new size as though it were one of the
    known ones, which is a mislabelled figure rather than a missing one.
    """
    with pytest.raises(AssertionError, match="no figure label"):
        make_figures.checkpoint_label("esm2_t36_3B_UR50D")
