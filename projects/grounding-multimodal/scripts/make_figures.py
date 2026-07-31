"""Render the figures used in the grounding-multimodal writeup.

Reads the committed result files rather than re-running anything, so a figure
cannot drift from the table it sits beside: both are generated from
`results/arms_all.csv` and `results/per_class_f1_all.json`, which `run_arms.py`
writes.

Outputs SVG into `docs/grounding-multimodal/figures/`, which is tracked by git
(small text files) and served by mkdocs.

    python projects/grounding-multimodal/scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

# Non-interactive backend: this script only ever writes files, and the default
# backend would try to reach a display on a cluster node.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from biotp.runlog import DEFAULT_LOG_DIR, get_logger, run_context

log = get_logger("make-figures")

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "projects" / "grounding-multimodal" / "results"
FIGURES_DIR = REPO_ROOT / "docs" / "grounding-multimodal" / "figures"

# Palette: slots 1-3 of the validated categorical set, which clear the
# colourblind-separation and normal-vision floors on all pairs. Every bar is
# also direct-labelled, which is what licenses the aqua slot despite its
# sub-3:1 contrast against a white surface.
BLUE = "#2a78d6"  # grounded arms (free text)
ORANGE = "#eb6834"  # structured arms: the leakage bound
AQUA = "#1baf7a"  # the shuffled-text control
GREY = "#8a8986"  # reference arms carrying no claim
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedddb"

# Role assignment for the six arms, and the display names used on the axis.
ARM_STYLE: dict[str, tuple[str, str, str]] = {
    # csv arm name: (display name, colour, legend role)
    "sequence-only": ("sequence-only", GREY, "baseline"),
    "text-only-free": ("text-only, free text", GREY, "baseline"),
    "text-only-structured": ("text-only, structured", ORANGE, "leakage bound"),
    "sequence+free-text": ("sequence + free text", BLUE, "grounded (headline)"),
    "sequence+structured": ("sequence + structured", ORANGE, "leakage bound"),
    "shuffled-text-control": ("shuffled-text control", AQUA, "control"),
}

# Descending by macro-F1, drawn top to bottom: the leakage ceiling first, then
# the headline arm, then the two baselines, then the control below them all.
ARM_ORDER = [
    "text-only-structured",
    "sequence+structured",
    "sequence+free-text",
    "text-only-free",
    "sequence-only",
    "shuffled-text-control",
]

# The per-class JSON uses dots where the compartment name has spaces.
CLASS_DISPLAY = {
    "Cell.membrane": "Cell membrane",
    "Endoplasmic.reticulum": "Endoplasmic reticulum",
    "Golgi.apparatus": "Golgi apparatus",
}


def style_axes(ax: plt.Axes) -> None:
    """Strip the frame down to a recessive grid, per the house plot style."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK_MUTED, labelsize=9)
    ax.set_axisbelow(True)


def read_arms(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"no rows in {path}"
    return rows


def plot_arms(arms_csv: Path, manifest_json: Path, out_path: Path) -> Path:
    """Horizontal bars of macro-F1 per arm, with seed spread and the floor."""
    rows = {row["arm"]: row for row in read_arms(arms_csv)}
    missing = set(ARM_ORDER) - set(rows)
    assert not missing, f"{arms_csv} is missing arms: {sorted(missing)}"

    # Read the floor from the run that produced these numbers rather than
    # hardcoding it, so the annotation cannot drift away from the table.
    manifest = json.loads(manifest_json.read_text())
    records = manifest.get("records", manifest)
    floor = float(records["majority_class_accuracy"])

    names, values, errors, colors = [], [], [], []
    for arm in ARM_ORDER:
        display, color, _role = ARM_STYLE[arm]
        names.append(display)
        values.append(float(rows[arm]["macro_f1"]))
        errors.append(float(rows[arm]["macro_f1_sd"]))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    # barh counts up from the bottom, so reverse to put ARM_ORDER[0] on top.
    positions = list(range(len(names)))[::-1]

    ax.barh(
        positions,
        values,
        xerr=errors,
        color=colors,
        height=0.52,
        error_kw={"ecolor": INK_MUTED, "elinewidth": 1.2, "capsize": 3},
    )

    ax.axvline(floor, color=INK_MUTED, linestyle=":", linewidth=1.4)
    ax.text(
        floor + 0.01,
        len(names) - 0.4,
        f"majority-class floor {floor:.3f}",
        fontsize=8,
        color=INK_MUTED,
        va="center",
    )

    for pos, value, error in zip(positions, values, errors):
        ax.text(
            value + error + 0.012,
            pos,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            color=INK,
        )

    ax.set_yticks(positions, names, fontsize=9.5)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Macro-F1 on the held-out test split", fontsize=9.5, color=INK_MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    style_axes(ax)

    ax.set_title(
        "Adding free text to sequence gains +0.124 macro-F1;\n"
        "pairing the same protein with someone else's text loses it",
        fontsize=11,
        color=INK,
        loc="left",
        pad=12,
    )

    seen: dict[str, str] = {}
    for arm in ARM_ORDER:
        _display, color, role = ARM_STYLE[arm]
        seen.setdefault(role, color)
    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=8, color=color, label=role)
        for role, color in seen.items()
    ]
    # Below the axes: every horizontal band inside the plot holds a bar, so an
    # in-plot legend would sit on top of the two longest ones.
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=len(handles),
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_MUTED,
        handletextpad=0.5,
        columnspacing=1.6,
    )

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out_path


def plot_per_class(per_class_json: Path, out_path: Path) -> Path:
    """Dumbbell chart: per-class F1, sequence-only against sequence + free text."""
    per_class = json.loads(per_class_json.read_text())
    for arm in ("sequence-only", "sequence+free-text"):
        assert arm in per_class, f"{per_class_json} has no arm {arm!r}"

    baseline = per_class["sequence-only"]
    grounded = per_class["sequence+free-text"]

    # Worst-performing sequence-only class at the bottom, so the eye travels down
    # into exactly the region where the gains are largest.
    classes = sorted(baseline, key=lambda name: baseline[name], reverse=True)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    positions = list(range(len(classes)))[::-1]

    for pos, name in zip(positions, classes):
        low, high = baseline[name], grounded[name]
        ax.plot([low, high], [pos, pos], color=GRID, linewidth=3, zorder=1)
        ax.scatter([low], [pos], s=64, color=GREY, zorder=2)
        ax.scatter([high], [pos], s=64, color=BLUE, zorder=2)
        ax.text(
            max(low, high) + 0.02,
            pos,
            f"+{high - low:.3f}",
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )

    labels = [CLASS_DISPLAY.get(name, name) for name in classes]
    ax.set_yticks(positions, labels, fontsize=9.5)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(
        "Per-class F1 on the held-out test split", fontsize=9.5, color=INK_MUTED
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    style_axes(ax)

    ax.set_title(
        "Text helps most where sequence does worst:\n"
        "the rare compartments at the bottom gain the most",
        fontsize=11,
        color=INK,
        loc="left",
        pad=12,
    )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            color=GREY,
            label="sequence-only",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=8,
            color=BLUE,
            label="sequence + free text",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_MUTED,
        handletextpad=0.5,
    )

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    with run_context("make-figures", log_dir=args.log_dir, params=vars(args)) as run:
        with run.step("arm comparison"):
            written = plot_arms(
                args.results_dir / "arms_all.csv",
                args.results_dir / "run_manifest_all.json",
                args.figures_dir / "arms-macro-f1.svg",
            )
            run.record("arms_figure", str(written.relative_to(REPO_ROOT)))
            run.record("arms_figure_bytes", written.stat().st_size)

        with run.step("per-class comparison"):
            written = plot_per_class(
                args.results_dir / "per_class_f1_all.json",
                args.figures_dir / "per-class-f1.svg",
            )
            run.record("per_class_figure", str(written.relative_to(REPO_ROOT)))
            run.record("per_class_figure_bytes", written.stat().st_size)


if __name__ == "__main__":
    main()
