"""Run the six arms of the grounding experiment and write a metrics table.

Each arm trains the same head architecture, on the same splits, with the same
frozen encoders. Only the input features differ, so a difference between arms is
attributable to what the model was allowed to see. See
docs/grounding-multimodal/introduction.md for what each arm buys.

Run from the repo root, after prepare_data.py:

    python projects/grounding-multimodal/scripts/run_arms.py

Embeddings are computed once per encoder and cached under data/processed/, so
re-running to change the head or add an arm costs seconds rather than minutes.
Results go to projects/grounding-multimodal/results/ and are small enough to commit.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biotp.embeddings import (
    DEFAULT_SENTENCE_ENCODER,
    cached_embeddings,
    cached_text_embeddings,
)
from biotp.evaluation import (
    classification_metrics,
    grouped_split,
    majority_class_accuracy,
    per_class_f1,
)
from biotp.training import build_head, predict, train
from biotp.utils import set_seed

DUAL_LOCALIZATION = "Cytoplasm-Nucleus"
SEQUENCE_ENCODER = "esm2_t12_35M_UR50D"

# The structured-text condition: controlled-vocabulary annotation, which is where
# the label tends to appear verbatim.
STRUCTURED_FIELDS = (
    "go_cellular_component",
    "go_biological_process",
    "go_molecular_function",
    "keywords",
)

# Train/validation fractions carved out of the non-test pool. The third entry is
# zero because the test split is DeepLoc's, not ours to re-derive.
TRAIN_VAL_TEST_FRACTIONS = (0.85, 0.15, 0.0)

MAX_EPOCHS = 200
LEARNING_RATE = 1e-3
EMBED_BATCH_SIZE = 16
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Arm:
    """One experimental condition: which feature blocks the head may see."""

    name: str
    blocks: tuple[str, ...]
    shuffle_text: bool
    purpose: str


ARMS = (
    Arm("sequence-only", ("sequence",), False, "baseline to beat"),
    Arm("text-only-free", ("text_free",), False, "how much does prose alone explain"),
    Arm("text-only-structured", ("text_structured",), False, "leakage upper bound"),
    Arm("sequence+free-text", ("sequence", "text_free"), False, "headline comparison"),
    Arm(
        "sequence+structured",
        ("sequence", "text_structured"),
        False,
        "headline, with leaky text",
    ),
    Arm(
        "shuffled-text-control",
        ("sequence", "text_free"),
        True,
        "detects gains not tied to this protein's text",
    ),
)


def load_table(path: Path) -> pd.DataFrame:
    """Load the prepared table and keep the single-label proteins."""
    assert path.exists(), f"missing {path}; run prepare_data.py first"
    table = pd.read_parquet(path)

    single = table[table["localization"] != DUAL_LOCALIZATION].reset_index(drop=True)
    assert len(single) > 0, "no single-label proteins found"
    assert single["localization"].nunique() == 10, "expected exactly 10 classes"
    return single


def structured_text(table: pd.DataFrame) -> list[str]:
    """Join the controlled-vocabulary fields into one string per protein."""
    parts = [table[field].fillna("").astype(str) for field in STRUCTURED_FIELDS]
    joined = parts[0]
    for part in parts[1:]:
        joined = joined.str.cat(part, sep="; ")
    return [text.strip("; ").strip() for text in joined]


def build_feature_blocks(table: pd.DataFrame, cache_dir: Path) -> dict[str, np.ndarray]:
    """Compute (or load) one embedding matrix per modality, aligned to table rows."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"embedding {len(table)} sequences with {SEQUENCE_ENCODER}")
    sequence = cached_embeddings(
        table["sequence"].tolist(),
        SEQUENCE_ENCODER,
        cache_dir / "sequence_esm2_35m.npz",
        EMBED_BATCH_SIZE,
    )

    print(f"embedding free text with {DEFAULT_SENTENCE_ENCODER}")
    text_free = cached_text_embeddings(
        table["function_text"].fillna("").tolist(),
        DEFAULT_SENTENCE_ENCODER,
        cache_dir / "text_free_minilm.npz",
        EMBED_BATCH_SIZE * 4,
    )

    print("embedding structured annotation text")
    text_structured = cached_text_embeddings(
        structured_text(table),
        DEFAULT_SENTENCE_ENCODER,
        cache_dir / "text_structured_minilm.npz",
        EMBED_BATCH_SIZE * 4,
    )

    blocks = {
        "sequence": sequence,
        "text_free": text_free,
        "text_structured": text_structured,
    }
    for name, block in blocks.items():
        assert len(block) == len(table), f"{name} embeddings misaligned with table"
        print(f"  {name:16} {block.shape}")
    return blocks


def assemble_features(
    arm: Arm, blocks: dict[str, np.ndarray], row_order: np.ndarray, seed: int
) -> np.ndarray:
    """Concatenate this arm's feature blocks for the given rows.

    For the shuffled control, the text block is permuted across proteins before
    slicing, so each protein keeps its own sequence but receives some other
    protein's annotation. The permutation is seeded, and it is applied to the whole
    dataset rather than within a split, mirroring the real pairing being broken.
    """
    columns = []
    for block_name in arm.blocks:
        block = blocks[block_name]
        if arm.shuffle_text and block_name.startswith("text"):
            permutation = np.random.default_rng(seed).permutation(len(block))
            block = block[permutation]
        columns.append(block[row_order])

    features = np.concatenate(columns, axis=1) if len(columns) > 1 else columns[0]
    return np.ascontiguousarray(features, dtype=np.float32)


def run_arm(
    arm: Arm,
    blocks: dict[str, np.ndarray],
    indices: dict[str, np.ndarray],
    labels: np.ndarray,
    class_names: list[str],
    seed: int,
) -> dict[str, Any]:
    """Train and evaluate one arm at one seed."""
    set_seed(seed)

    features = {
        split: assemble_features(arm, blocks, rows, seed)
        for split, rows in indices.items()
    }
    targets = {split: labels[rows] for split, rows in indices.items()}

    head = build_head(
        input_dim=features["train"].shape[1],
        output_dim=len(class_names),
        task="classification",
    )
    head, history = train(
        head,
        (features["train"], targets["train"]),
        (features["val"], targets["val"]),
        mode="linear_probe",
        max_epochs=MAX_EPOCHS,
        lr=LEARNING_RATE,
    )

    predictions = predict(head, features["test"])
    true_names = [class_names[index] for index in targets["test"]]
    predicted_names = [class_names[index] for index in predictions]

    metrics = classification_metrics(true_names, predicted_names, "macro")
    return {
        "arm": arm.name,
        "seed": seed,
        "input_dim": int(features["train"].shape[1]),
        "epochs_run": history["epochs_run"],
        "best_epoch": history["best_epoch"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "per_class_f1": per_class_f1(true_names, predicted_names),
        "predictions": predicted_names,
    }


def _spread(runs: list[dict[str, Any]], key: str) -> tuple[float, float]:
    """Mean and standard deviation of one metric across seeds."""
    values = [run[key] for run in runs]
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), deviation


def summarize(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-seed runs into one row per arm, with spread across seeds."""
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_runs = [run for run in runs if run["arm"] == arm.name]
        if not arm_runs:
            continue

        accuracy, accuracy_sd = _spread(arm_runs, "accuracy")
        macro_f1, macro_f1_sd = _spread(arm_runs, "macro_f1")
        balanced, balanced_sd = _spread(arm_runs, "balanced_accuracy")
        rows.append(
            {
                "arm": arm.name,
                "purpose": arm.purpose,
                "input_dim": arm_runs[0]["input_dim"],
                "seeds": len(arm_runs),
                "accuracy": accuracy,
                "accuracy_sd": accuracy_sd,
                "macro_f1": macro_f1,
                "macro_f1_sd": macro_f1_sd,
                "balanced_accuracy": balanced,
                "balanced_accuracy_sd": balanced_sd,
            }
        )
    return pd.DataFrame(rows)


def render_markdown(
    summary: pd.DataFrame, majority: float, cohort: str, n_test: int
) -> str:
    """Render the summary as a Markdown table, with the floor stated alongside."""
    lines = [
        f"## Test-set results ({cohort})",
        "",
        (
            f"{n_test} held-out proteins from DeepLoc's homology-partitioned test "
            f"split. Mean over {len(SEEDS)} seeds, with standard deviation. "
            f"Majority-class accuracy floor: {majority:.3f}."
        ),
        "",
        "| Arm | Purpose | Dim | Accuracy | Macro-F1 | Balanced acc. |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['arm']} | {row['purpose']} | {row['input_dim']} | "
            f"{row['accuracy']:.3f} ± {row['accuracy_sd']:.3f} | "
            f"{row['macro_f1']:.3f} ± {row['macro_f1_sd']:.3f} | "
            f"{row['balanced_accuracy']:.3f} ± {row['balanced_accuracy_sd']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def split_indices(table: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Split rows into train/val/test.

    Test is DeepLoc's official partition, inherited rather than re-derived. Train
    and validation are carved from the remainder with grouped_split keyed on
    accession, which makes this a random split: it is the documented limitation
    that the train/validation boundary is not family-grouped yet. Reported numbers
    come from the test split, which is unaffected.
    """
    pool = table.index[~table["is_test"]].tolist()
    test = table.index[table["is_test"]].to_numpy()
    assert len(pool) > 0 and len(test) > 0, "expected a non-empty pool and test split"

    accession = table["accession"]
    train, val, empty = grouped_split(
        pool,
        lambda row: accession.iloc[row],
        TRAIN_VAL_TEST_FRACTIONS,
        seed,
    )
    assert not empty, "third fraction is 0.0, so the third split must be empty"

    return {
        "train": np.asarray(train, dtype=int),
        "val": np.asarray(val, dtype=int),
        "test": test,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("projects/grounding-multimodal/results"),
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="restrict to proteins that have free-text function annotation",
    )
    args = parser.parse_args()

    table = load_table(args.data_root / "processed" / "deeploc_annotated.parquet")
    blocks = build_feature_blocks(table, args.data_root / "processed" / "embeddings")

    cohort = "all proteins"
    if args.annotated_only:
        # Subset after embedding, so the cached vectors are shared with the full run.
        keep = table.index[table["has_function_text"]].to_numpy()
        table = table.loc[keep].reset_index(drop=True)
        blocks = {name: block[keep] for name, block in blocks.items()}
        cohort = "annotated subset"
    print(f"\ncohort: {cohort}, {len(table)} proteins")

    class_names = sorted(table["localization"].unique())
    label_index = {name: index for index, name in enumerate(class_names)}
    labels = table["localization"].map(label_index).to_numpy()

    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        indices = split_indices(table, seed)
        sizes = {split: len(rows) for split, rows in indices.items()}
        print(f"\nseed {seed}: {sizes}")
        for arm in ARMS:
            result = run_arm(arm, blocks, indices, labels, class_names, seed)
            runs.append(result)
            print(
                f"  {arm.name:24} acc={result['accuracy']:.3f} "
                f"macroF1={result['macro_f1']:.3f} "
                f"(best epoch {result['best_epoch']}/{result['epochs_run']})"
            )

    test_labels = [
        class_names[index] for index in labels[split_indices(table, 0)["test"]]
    ]
    majority = majority_class_accuracy(test_labels)
    summary = summarize(runs)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "annotated" if args.annotated_only else "all"
    summary.to_csv(args.results_dir / f"arms_{suffix}.csv", index=False)
    markdown = render_markdown(summary, majority, cohort, len(test_labels))
    (args.results_dir / f"arms_{suffix}.md").write_text(markdown)
    (args.results_dir / f"per_class_f1_{suffix}.json").write_text(
        json.dumps(
            {
                run["arm"]: run["per_class_f1"]
                for run in runs
                if run["seed"] == SEEDS[0]
            },
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(markdown)
    print(f"wrote results to {args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
