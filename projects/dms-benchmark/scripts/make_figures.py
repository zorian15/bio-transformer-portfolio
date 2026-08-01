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

log = get_logger("dms-make-figures")


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

            for rung in ("frozen", "lora"):
                curve = cell[cell["rung"] == rung]
                if curve.empty:
                    continue
                summary = curve.groupby("n")["spearman"].agg(["mean", "std"])
                axis.errorbar(
                    summary.index,
                    summary["mean"],
                    yerr=summary["std"].fillna(0.0),
                    marker="o",
                    markersize=4,
                    capsize=3,
                    linewidth=1.6,
                    color=RUNG_COLOURS[rung],
                    label=RUNG_LABELS[rung],
                )

            for _, line in cell[cell["rung"] == "zero_shot"].iterrows():
                style = "-" if "35M" in str(line["checkpoint"]) else "--"
                axis.axhline(
                    line["spearman"],
                    color=RUNG_COLOURS["zero_shot"],
                    linestyle=style,
                    linewidth=1.3,
                    label=f"{RUNG_LABELS['zero_shot']} "
                    f"({'35M' if '35M' in str(line['checkpoint']) else '650M'})",
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

    width = 0.35
    for column, scheme in enumerate(schemes):
        axis = axes[0][column]
        cell = supervised[supervised["scheme"] == scheme]
        offsets = {"frozen": -width / 2, "lora": width / 2}
        for rung, offset in offsets.items():
            values = [
                cell[(cell["rung"] == rung) & (cell["readout"] == readout)][
                    "spearman"
                ].mean()
                for readout in readouts
            ]
            axis.bar(
                [index + offset for index in range(len(readouts))],
                values,
                width,
                color=RUNG_COLOURS[rung],
                label=RUNG_LABELS[rung],
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
