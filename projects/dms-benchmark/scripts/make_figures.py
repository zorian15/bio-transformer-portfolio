"""Generate the dms-benchmark figures from the committed result tables.

Figures are generated, never hand-drawn, so a plot cannot drift from the table
beside it. Re-run whenever `results/*.csv` changes and commit the SVGs; CI builds
the docs without matplotlib and serves whatever is in git.

Run from the repo root:

    python projects/dms-benchmark/scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from biotp.runlog import DEFAULT_LOG_DIR, get_logger, run_context

# One colour per rung, held constant across every figure so a reader learns the
# mapping once. Chosen to stay distinguishable in greyscale and to common forms
# of colour blindness.
RUNG_COLOURS = {
    "zero_shot": "#767676",
    "frozen": "#1b6ca8",
    "lora": "#c2492d",
}
RUNG_LABELS = {
    "zero_shot": "zero-shot",
    "frozen": "frozen + head",
    "lora": "LoRA + head",
}

SCHEME_LABELS = {
    "fold_random_5": "random",
    "fold_modulo_5": "modulo (position-disjoint)",
    "fold_contiguous_5": "contiguous (position-disjoint)",
}

# Model size is an axis of the supervised ladder since issue #34, so it needs an
# encoding of its own. Colour is already spent on the rung, which is the
# comparison the figures exist to make, so size gets the linestyle. That also
# matches how the zero-shot reference lines already distinguish the two sizes.
CHECKPOINT_LABELS = {
    "esm2_t12_35M_UR50D": "35M",
    "esm2_t33_650M_UR50D": "650M",
}
CHECKPOINT_STYLES = {
    "esm2_t12_35M_UR50D": "-",
    "esm2_t33_650M_UR50D": "--",
}
# Bars cannot carry a linestyle, so the same distinction is a hatch there. A
# registry rather than "" for the first size and "//" for the rest: assigning by
# position means a third size silently shares the second one's hatch, which is
# the merge-two-sizes-into-one failure this file was just fixed for, one figure
# over. The tests cross-check all three maps against the grid's own sizes.
CHECKPOINT_HATCHES = {
    "esm2_t12_35M_UR50D": "",
    "esm2_t33_650M_UR50D": "//",
}

log = get_logger("dms-make-figures")


def checkpoint_label(checkpoint: str) -> str:
    """Short display name for a checkpoint, e.g. "650M".

    Asserts rather than falling back, because a size drawn under a borrowed
    label is a mislabelled figure, which is harder to notice than a missing
    one. `tests/test_dms_make_figures.py` cross-checks this map against the
    grid's own list of sizes, so adding one to the ladder fails here first.

    Args:
        checkpoint: an ESM-2 checkpoint name as it appears in the results.

    Returns:
        The short label for that checkpoint.
    """
    assert checkpoint in CHECKPOINT_LABELS, (
        f"no figure label for checkpoint {checkpoint!r}; add it to "
        f"CHECKPOINT_LABELS and CHECKPOINT_STYLES so it is drawn distinctly "
        f"rather than sharing another size's encoding"
    )
    return CHECKPOINT_LABELS[checkpoint]


def present_checkpoints(frame: pd.DataFrame) -> list[str]:
    """Which sizes appear in `frame`, in the canonical figure order.

    One helper rather than each figure working it out, because they had reached
    for different expressions: one sorted the values alphabetically and the
    other filtered the style registry. Those agree only because "esm2_t12_35M"
    happens to sort before "esm2_t33_650M", so the two figures would have
    disagreed about bar and line order the first time a size was added whose
    name did not sort that way.

    Args:
        frame: any results frame carrying a `checkpoint` column.

    Returns:
        The checkpoints present, ordered as `CHECKPOINT_STYLES` lists them.
    """
    present = set(frame["checkpoint"])
    for checkpoint in sorted(present):
        # Reuse the label lookup's assertion rather than restating it, so an
        # unrecognised size fails identically wherever it first reaches a figure.
        checkpoint_label(checkpoint)
    return [checkpoint for checkpoint in CHECKPOINT_STYLES if checkpoint in present]


def curve_summary(cell: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of Spearman per (checkpoint, rung, N).

    Checkpoint is in the key, not implicit. It used to be absent, so a frame
    holding both model sizes averaged them into a single curve and the gap
    between them was reported as seed spread. The figure rendered cleanly and
    disagreed with no table, which is why nothing caught it.

    Seeds stay aggregated: they are what the error bar describes.

    Args:
        cell: result rows for one assay and scheme, supervised rungs only.

    Returns:
        One row per (checkpoint, rung, n) with `mean` and `std` columns.
    """
    return (
        cell.groupby(["checkpoint", "rung", "n"])["spearman"]
        .agg(["mean", "std"])
        .reset_index()
    )


def readout_summary(cell: pd.DataFrame) -> pd.DataFrame:
    """Mean Spearman per (checkpoint, rung, readout), for the bar figure.

    Extracted rather than left inline for the same reason as `curve_summary`,
    and after review pointed out it had been missed: this is the *other* half of
    the size-merging bug. The bar figure took a flat mean over each cell, so two
    model sizes averaged into one bar exactly as the curves did. Inline, nothing
    executed it and a regression to that flat mean would keep the suite green.

    Args:
        cell: supervised rows for one split scheme.

    Returns:
        One row per (checkpoint, rung, readout) with a `mean` column.
    """
    return (
        cell.groupby(["checkpoint", "rung", "readout"], as_index=False)["spearman"]
        .mean()
        .rename(columns={"spearman": "mean"})
    )


def load_results(results_dir: Path) -> pd.DataFrame:
    """Concatenate whichever rung tables exist, so partial runs still plot."""
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


def data_efficiency_figure(results: pd.DataFrame, destination: Path) -> None:
    """The money plot: Spearman against label count, one panel per scheme.

    Zero-shot has no training labels, so it is drawn as a horizontal reference
    line rather than a point at some arbitrary x. That is the floor the
    supervised rungs have to clear, and drawing it any other way would invite a
    reader to interpolate toward it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assays = sorted(results["assay"].unique())
    schemes = [s for s in SCHEME_LABELS if s in set(results["scheme"])]
    figure, axes = plt.subplots(
        len(assays),
        len(schemes),
        figsize=(4.2 * len(schemes), 3.4 * len(assays)),
        squeeze=False,
        sharey="row",
    )

    for row, assay in enumerate(assays):
        for column, scheme in enumerate(schemes):
            axis = axes[row][column]
            cell = results[(results["assay"] == assay) & (results["scheme"] == scheme)]

            supervised = cell[cell["rung"].isin(("frozen", "lora"))]

            # Colour carries the rung and linestyle carries the size, so the
            # rung-2-to-rung-3 comparison the ladder is about stays the visually
            # obvious one and scale reads as a secondary axis.
            summary = curve_summary(supervised)
            for rung in ("frozen", "lora"):
                for checkpoint in present_checkpoints(supervised):
                    curve = summary[
                        (summary["rung"] == rung)
                        & (summary["checkpoint"] == checkpoint)
                    ].sort_values("n")
                    if curve.empty:
                        continue
                    axis.errorbar(
                        curve["n"],
                        curve["mean"],
                        yerr=curve["std"].fillna(0.0),
                        marker="o",
                        markersize=4,
                        capsize=3,
                        linewidth=1.6,
                        color=RUNG_COLOURS[rung],
                        linestyle=CHECKPOINT_STYLES[checkpoint],
                        label=f"{RUNG_LABELS[rung]} ({checkpoint_label(checkpoint)})",
                    )

            # Validated through the same helper as the supervised rows rather
            # than indexing the style map directly. Rung 1 has its own list of
            # sizes, which is the one most likely to grow on its own since a
            # zero-shot arm is nearly free, and a direct lookup would fail with
            # a bare KeyError from a figure whose docstring promises otherwise.
            floors = cell[cell["rung"] == "zero_shot"]
            present_checkpoints(floors)
            for _, line in floors.iterrows():
                checkpoint = str(line["checkpoint"])
                axis.axhline(
                    line["spearman"],
                    color=RUNG_COLOURS["zero_shot"],
                    linestyle=CHECKPOINT_STYLES[checkpoint],
                    linewidth=1.3,
                    label=f"{RUNG_LABELS['zero_shot']} "
                    f"({checkpoint_label(checkpoint)})",
                )

            axis.set_xscale("log", base=2)
            axis.axhline(0.0, color="#cccccc", linewidth=0.8, zorder=0)
            if row == 0:
                axis.set_title(SCHEME_LABELS[scheme], fontsize=10)
            if row == len(assays) - 1:
                axis.set_xlabel("labelled training variants")
            if column == 0:
                axis.set_ylabel(f"{assay}\nSpearman", fontsize=8)
            axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    figure.legend(
        list(seen.values()),
        list(seen),
        loc="lower center",
        ncol=len(seen),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.tight_layout()
    figure.savefig(destination, format="svg", bbox_inches="tight")
    plt.close(figure)
    log.info(f"wrote {destination}")


def readout_figure(results: pd.DataFrame, destination: Path) -> None:
    """Readout as an axis: does the choice interact with the split scheme?

    Pre-registered as a full axis rather than tuned, so this reports all of them
    side by side instead of showing a winner.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    supervised = results[results["rung"].isin(["frozen", "lora"])]
    if supervised.empty:
        log.info("no supervised rows yet; skipping the readout figure")
        return

    schemes = [s for s in SCHEME_LABELS if s in set(supervised["scheme"])]
    readouts = sorted(supervised["readout"].unique())
    figure, axes = plt.subplots(
        1, len(schemes), figsize=(4.2 * len(schemes), 3.6), squeeze=False, sharey=True
    )

    # Four bars per readout rather than two: rung by colour, size by hatch. A
    # flat mean over the cell used to average the two model sizes together,
    # which understated whichever size was better at every readout.
    checkpoints = present_checkpoints(supervised)
    width = 0.8 / max(len(checkpoints) * 2, 1)

    for column, scheme in enumerate(schemes):
        axis = axes[0][column]
        summary = readout_summary(supervised[supervised["scheme"] == scheme])
        bars = [(rung, c) for rung in ("frozen", "lora") for c in checkpoints]
        for position, (rung, checkpoint) in enumerate(bars):
            offset = (position - (len(bars) - 1) / 2) * width
            matched = summary[
                (summary["rung"] == rung) & (summary["checkpoint"] == checkpoint)
            ].set_index("readout")["mean"]
            values = [matched.get(readout, float("nan")) for readout in readouts]
            axis.bar(
                [index + offset for index in range(len(readouts))],
                values,
                width,
                color=RUNG_COLOURS[rung],
                hatch=CHECKPOINT_HATCHES[checkpoint],
                edgecolor="white",
                label=f"{RUNG_LABELS[rung]} ({checkpoint_label(checkpoint)})",
            )
        axis.set_xticks(range(len(readouts)))
        axis.set_xticklabels([r.replace("_", "\n") for r in readouts], fontsize=8)
        axis.set_title(SCHEME_LABELS[scheme], fontsize=10)
        axis.axhline(0.0, color="#cccccc", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        if column == 0:
            axis.set_ylabel("Spearman, mean over assays, N and seeds")

    axes[0][0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, format="svg", bbox_inches="tight")
    plt.close(figure)
    log.info(f"wrote {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("projects/dms-benchmark/results")
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("docs/dms-benchmark/figures")
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    with run_context(
        "dms-make-figures", log_dir=args.log_dir, params=vars(args)
    ) as run:
        with run.step("load results"):
            results = load_results(args.results_dir)
            run.record("result_rows", len(results))

        args.figures_dir.mkdir(parents=True, exist_ok=True)
        with run.step("draw figures"):
            data_efficiency_figure(results, args.figures_dir / "data-efficiency.svg")
            readout_figure(results, args.figures_dir / "readout-by-scheme.svg")

    return 0


if __name__ == "__main__":
    sys.exit(main())
