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
    # The grounding-versus-leakage ablation (issue #5). No new hues: the free-text
    # arms stay in the blue family at different information levels, and the
    # length-matched random arms take the control colour, which is their role.
    "text-only-free-cleaned": ("text-only, cleaned", GREY, "baseline"),
    "text-only-free-ablated": ("text-only, ablated", GREY, "baseline"),
    "text-only-free-random-ablated": ("text-only, random-ablated", AQUA, "control"),
    "sequence+free-text-cleaned": ("sequence + cleaned text", BLUE, "grounded"),
    "sequence+free-text-ablated": ("sequence + ablated text", BLUE, "grounded"),
    "sequence+free-text-random-ablated": (
        "sequence + random-ablated text",
        AQUA,
        "control",
    ),
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

# The ablation ladder, in the order the argument is made: the sequence baseline,
# then free text at four information levels, then the same four with sequence
# removed. Each pair of adjacent bars isolates one thing.
ABLATION_ORDER = [
    "sequence+free-text",
    "sequence+free-text-cleaned",
    "sequence+free-text-ablated",
    "sequence+free-text-random-ablated",
    "sequence-only",
    "text-only-free",
    "text-only-free-cleaned",
    "text-only-free-ablated",
    "text-only-free-random-ablated",
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


def assert_every_arm_is_drawn(arms_csv: Path) -> None:
    """Fail if an arm exists in the results but appears in no figure, or vice versa.

    The check runs in both directions on purpose. A missing arm was already loud,
    but an arm the runner produces and no figure draws used to vanish silently,
    which is how a new condition ends up in the table with no picture beside it.
    """
    produced = {row["arm"] for row in read_arms(arms_csv)}
    drawn = set(ARM_ORDER) | set(ABLATION_ORDER)
    assert produced == drawn, (
        f"{arms_csv} arms drawn in no figure: {sorted(produced - drawn)}; "
        f"figure arms absent from the results: {sorted(drawn - produced)}"
    )


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

    # Computed, not hardcoded: this number sits in the title next to the bars it
    # describes, so a stale literal would contradict the figure it labels.
    gain = float(rows["sequence+free-text"]["macro_f1"]) - float(
        rows["sequence-only"]["macro_f1"]
    )

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
        f"Adding free text to sequence gains +{gain:.3f} macro-F1;\n"
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


def plot_ablation(arms_csv: Path, out_path: Path) -> Path:
    """The ablation ladder: free text at four information levels, with and without sequence."""
    rows = {row["arm"]: row for row in read_arms(arms_csv)}
    missing = set(ABLATION_ORDER) - set(rows)
    assert not missing, f"{arms_csv} is missing arms: {sorted(missing)}"

    grounded = float(rows["sequence+free-text"]["macro_f1"])
    baseline = float(rows["sequence-only"]["macro_f1"])
    ablated = float(rows["sequence+free-text-ablated"]["macro_f1"])
    control = float(rows["sequence+free-text-random-ablated"]["macro_f1"])
    # Both shares are of the *unfiltered* gain, which is what the title compares
    # them as: "the ablation keeps X%, removing as much at random keeps Y%". The
    # gap between the two is the part attributable to the compartment vocabulary,
    # and putting them on one denominator is what makes that gap readable.
    survives = (ablated - baseline) / (grounded - baseline)

    names, values, errors, colors = [], [], [], []
    for arm in ABLATION_ORDER:
        display, color, _role = ARM_STYLE[arm]
        names.append(display)
        values.append(float(rows[arm]["macro_f1"]))
        errors.append(float(rows[arm]["macro_f1_sd"]))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    positions = list(range(len(names)))[::-1]
    ax.barh(
        positions,
        values,
        xerr=errors,
        color=colors,
        height=0.56,
        error_kw={"ecolor": INK_MUTED, "elinewidth": 1.2, "capsize": 3},
    )

    ax.axvline(baseline, color=INK_MUTED, linestyle=":", linewidth=1.4)
    ax.text(
        baseline + 0.01,
        len(names) - 0.45,
        f"sequence-only {baseline:.3f}",
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
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Macro-F1 on the held-out test split", fontsize=9.5, color=INK_MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    style_axes(ax)

    ax.set_title(
        f"Removing localization-stating sentences keeps {survives:.0%} of the text gain;\n"
        f"removing as much text at random keeps {(control - baseline) / (grounded - baseline):.0%}",
        fontsize=11,
        color=INK,
        loc="left",
        pad=12,
    )

    # Keyed on colour, not role: "grounded (headline)" and "grounded" are the same
    # blue here, and two swatches of one colour reads as a distinction that is not
    # being drawn. The headline qualifier belongs on the six-arm figure, not this one.
    seen: dict[str, str] = {}
    for arm in ABLATION_ORDER:
        _display, color, role = ARM_STYLE[arm]
        seen.setdefault(color, role.replace(" (headline)", ""))
    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=8, color=color, label=role)
        for color, role in seen.items()
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
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


def plot_ablation_removal(ablation_json: Path, out_path: Path) -> Path:
    """How much text the filter takes per compartment, and where it takes all of it."""
    ablation = json.loads(ablation_json.read_text())
    per_class = ablation["per_class"]
    assert per_class, f"{ablation_json} has no per-class breakdown"

    classes = sorted(per_class, key=lambda name: per_class[name]["trimmed_share"])
    trimmed = [per_class[name]["trimmed_share"] for name in classes]
    emptied = [per_class[name]["emptied_share"] for name in classes]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    positions = list(range(len(classes)))

    # Emptied is a subset of trimmed, so it is drawn over the same bar rather than
    # beside it: the darker segment is the part of the class that lost everything.
    ax.barh(positions, trimmed, color=GRID, height=0.58, label="lost a sentence")
    ax.barh(positions, emptied, color=ORANGE, height=0.58, label="lost all text")

    for pos, share, empty in zip(positions, trimmed, emptied):
        ax.text(
            share + 0.02,
            pos,
            f"{share:.0%}   ({empty:.0%} emptied)",
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )

    labels = [CLASS_DISPLAY.get(name, name) for name in classes]
    ax.set_yticks(positions, labels, fontsize=9.5)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    ax.set_xlabel(
        "Share of annotated proteins in the class", fontsize=9.5, color=INK_MUTED
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    style_axes(ax)

    ax.set_title(
        "The filter is not class-neutral, and that is the confound:\n"
        "organelle proteins lose the most text, and most often lose all of it",
        fontsize=11,
        color=INK,
        loc="left",
        pad=12,
    )
    # Below the axes, like the other bar figures: every band inside the plot holds
    # a bar and a label, so an in-plot legend collides with one of them.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
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
    for arm in ("sequence-only", "sequence+free-text", "sequence+free-text-ablated"):
        assert arm in per_class, f"{per_class_json} has no arm {arm!r}"

    baseline = per_class["sequence-only"]
    grounded = per_class["sequence+free-text"]
    ablated = per_class["sequence+free-text-ablated"]

    # Worst-performing sequence-only class at the bottom, so the eye travels down
    # into exactly the region where the gains are largest.
    classes = sorted(baseline, key=lambda name: baseline[name], reverse=True)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    positions = list(range(len(classes)))[::-1]

    for pos, name in zip(positions, classes):
        low, high = baseline[name], grounded[name]
        # The ablated point sits on the same row: how much of each class's gain is
        # left once the prose can no longer name the compartment.
        ax.plot([low, high], [pos, pos], color=GRID, linewidth=3, zorder=1)
        ax.scatter([low], [pos], s=64, color=GREY, zorder=2)
        ax.scatter([ablated[name]], [pos], s=52, color=AQUA, zorder=3)
        ax.scatter([high], [pos], s=64, color=BLUE, zorder=2)
        ax.text(
            max(low, high, ablated[name]) + 0.02,
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
            markersize=7,
            color=AQUA,
            label="sequence + ablated text",
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
        with run.step("check every arm is drawn"):
            # Both cohorts, because they carry the same arms and only the `_all`
            # one is plotted: a new arm missing from the annotated table would
            # otherwise go unnoticed until someone read that file.
            for cohort in ("all", "annotated"):
                assert_every_arm_is_drawn(args.results_dir / f"arms_{cohort}.csv")

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

        with run.step("ablation ladder"):
            written = plot_ablation(
                args.results_dir / "arms_all.csv",
                args.figures_dir / "ablation-macro-f1.svg",
            )
            run.record("ablation_figure", str(written.relative_to(REPO_ROOT)))
            run.record("ablation_figure_bytes", written.stat().st_size)

        with run.step("ablation removal by class"):
            written = plot_ablation_removal(
                args.results_dir / "ablation_all.json",
                args.figures_dir / "ablation-removal-by-class.svg",
            )
            run.record("removal_figure", str(written.relative_to(REPO_ROOT)))
            run.record("removal_figure_bytes", written.stat().st_size)


if __name__ == "__main__":
    main()
