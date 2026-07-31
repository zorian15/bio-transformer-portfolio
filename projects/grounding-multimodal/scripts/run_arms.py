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
from biotp.runlog import DEFAULT_LOG_DIR, get_logger, run_context
from biotp.text_ablation import (
    AblationResult,
    ablate_sentences,
    ablation_summary,
    clean_annotation_text,
    compile_term_pattern,
    random_ablate_sentences,
    sentence_seed,
    split_sentences,
    term_mention_counts,
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

# The localization vocabulary for the grounding-versus-leakage ablation (issue #5).
# A sentence mentioning any of these is dropped from the free-text variant, so the
# ablated arm cannot read the compartment out of the prose.
#
# The lexicon prefers recall over precision, deliberately. A false negative, a
# synonym missed here, leaves the answer in the text and biases the result toward
# "grounding" invisibly. A false positive removes real functional content, which
# looks like leakage but is visible, because the length-matched random control
# removes just as much from the same proteins. Terms were counted against the
# corpus before being included; see docs/grounding-multimodal/ablation.md.
COMPARTMENT_TERMS: dict[str, tuple[str, ...]] = {
    "Cell.membrane": (
        "cell membrane",
        "plasma membrane",
        "cytoplasmic membrane",
        "plasmalemma",
        "sarcolemma",
        "cell surface",
        "apical membrane",
        "basolateral membrane",
    ),
    "Cytoplasm": ("cytoplasm", "cytoplasmic", "cytosol", "cytosolic"),
    "Endoplasmic.reticulum": (
        "endoplasmic reticulum",
        "sarcoplasmic reticulum",
        "microsome",
        "microsomal",
        # Bare "ER" is deliberately absent: case-insensitive matching would make it
        # noise. The phrases it actually appears in are enumerated instead.
        "ER lumen",
        "ER membrane",
        "ER stress",
        "ER-associated",
        "ER exit",
    ),
    "Extracellular": (
        "extracellular",
        "extracellular space",
        "extracellular matrix",
        "secreted",
        "apoplast",
        "periplasm",
        "periplasmic",
        "blood plasma",
    ),
    "Golgi.apparatus": ("golgi", "trans-golgi network", "TGN"),
    "Lysosome/Vacuole": (
        "lysosome",
        "lysosomal",
        "vacuole",
        "vacuolar",
        "tonoplast",
        "endolysosome",
        "autolysosome",
        "lytic vacuole",
    ),
    "Mitochondrion": (
        "mitochondrion",
        "mitochondria",
        "mitochondrial",
        "mitochondrially",
        "mitochondrial matrix",
        "intermembrane space",
        "cristae",
        "mitoribosome",
    ),
    "Nucleus": (
        "nucleus",
        "nuclei",
        "nuclear",
        "nucleoplasm",
        "nucleolus",
        "nucleoli",
        "nucleolar",
        "perinuclear",
        "nuclear pore",
        "nuclear envelope",
        "nuclear lamina",
    ),
    "Peroxisome": (
        "peroxisome",
        "peroxisomal",
        "glyoxysome",
        "glyoxysomal",
        "peroxisome targeting signal",
        "PTS1",
        "PTS2",
    ),
    "Plastid": (
        "plastid",
        "chloroplast",
        "thylakoid",
        "amyloplast",
        "chromoplast",
        "etioplast",
        "plastoglobule",
        # Bare "stroma" is deliberately absent: all 21 corpus mentions are animal
        # connective tissue ("stromal cells"), not the plastid compartment.
        "chloroplast stroma",
        "stromal thylakoid",
    ),
    # Not a compartment, but the verbs and signals that state a location. Kept as
    # its own group so its marginal contribution is measured rather than assumed:
    # these almost always co-occur with a compartment noun that already fires.
    "localization_language": (
        "localizes to",
        "localized to",
        "localization signal",
        "targeted to",
        "targeting signal",
        "translocates to",
        "translocated to",
        "signal peptide",
        "transit peptide",
        "retention signal",
    ),
}

# Phrases that contain a lexicon term without stating a location. A sentence is
# kept only when an exclusion accounts for all of its matches, so "the nuclear
# receptor is retained in the nucleus" is still removed. Every entry here is a
# deliberate false negative, the dangerous direction, so the list stays short and
# each entry is tested by name.
COMPARTMENT_EXCLUSIONS: tuple[str, ...] = (
    "nuclear receptor",
    "nuclear factor",
    "nuclear hormone receptor",
    "cytoplasmic tail",
    "cytoplasmic domain",
    "cytoplasmic side",
    "cytoplasmic face",
    "cytoplasmic dynein",
)

# Location-adjacent words deliberately left out of the filter, measured on the
# surviving text as the false-negative probe. Each is either cross-class (one word
# for two compartments), or names something that is not a DeepLoc class, or is
# functional prose the experiment is about. `chromatin` is the sharpest of these:
# it is a near-perfect Nucleus indicator, but "chromatin remodeling" is exactly the
# functional content the grounding hypothesis concerns, so it is measured, not cut.
SENTINEL_TERMS: tuple[str, ...] = (
    "membrane",
    "organelle",
    "lumen",
    "vesicle",
    "envelope",
    "endosome",
    "granule",
    "matrix",
    "chromatin",
    "nucleosome",
    "secretion",
    "secretory",
    "secretory pathway",
    "stroma",
    "stromal",
    "cell wall",
    "ER",
    "compartment",
    "intracellular",
    "microtubule",
    "cytoskeleton",
    "nucleoid",
)

# The embedding cache invalidates automatically when the lexicon changes, because
# the filtered strings change and the cache key hashes its inputs. The reported
# statistics would change silently, though, so the lexicon carries a version the
# way EMBEDDING_IMPL_VERSION does, making a DECISION_LOG entry citable.
ABLATION_LEXICON_VERSION = 1

# Train/validation fractions carved out of the non-test pool. The third entry is
# zero because the test split is DeepLoc's, not ours to re-derive.
TRAIN_VAL_TEST_FRACTIONS = (0.85, 0.15, 0.0)

MAX_EPOCHS = 200
LEARNING_RATE = 1e-3

# Re-tuned after length-bucketed batching landed (issue #3). Bigger is not better
# here: on MPS the binding constraint is the attention matrix, not utilisation, and
# batch 64 drove the machine into swap and had to be abandoned. Between 8 and 16 the
# medians are close (37.4s vs 39.2s per 300 proteins), but 8 repeated within 0.3s
# while 16 spread over 14s, and 8 halves peak memory. Predictable wins.
# Batching does not change the vectors, so this is not part of the cache key.
EMBED_BATCH_SIZE = 8

# The text encoder is small and its inputs are short, so it is not under the same
# memory pressure and keeps the batch size it was already using.
TEXT_BATCH_SIZE = 64

SEEDS = (0, 1, 2)

log = get_logger("run-arms")


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
    # The grounding-versus-leakage ablation (issue #5). Appended rather than
    # interleaved so the first six rows of the results table diff cleanly against
    # the committed MVP numbers. Arm order does not affect any result: run_arm
    # reseeds before every fit.
    Arm(
        "text-only-free-cleaned",
        ("text_free_cleaned",),
        False,
        "prose alone, database bookkeeping removed",
    ),
    Arm(
        "text-only-free-ablated",
        ("text_free_ablated",),
        False,
        "prose alone, compartment sentences removed",
    ),
    Arm(
        "text-only-free-random-ablated",
        ("text_free_random_ablated",),
        False,
        "prose alone, as much text removed at random",
    ),
    Arm(
        "sequence+free-text-cleaned",
        ("sequence", "text_free_cleaned"),
        False,
        "isolates the evidence-code confound",
    ),
    Arm(
        "sequence+free-text-ablated",
        ("sequence", "text_free_ablated"),
        False,
        "the ablation: grounding or leakage",
    ),
    Arm(
        "sequence+free-text-random-ablated",
        ("sequence", "text_free_random_ablated"),
        False,
        "length-matched control for the ablation",
    ),
)

# Which blocks are text, and so may be permuted for the shuffled control. This is
# an explicit set rather than a name prefix test: a block named `free_text_...`
# would silently be left unshuffled, quietly turning the control into a second
# grounded arm.
TEXT_BLOCKS = frozenset(
    {
        "text_free",
        "text_structured",
        "text_free_cleaned",
        "text_free_ablated",
        "text_free_random_ablated",
    }
)

# Blocks whose contents depend on the run seed, so each seed gets a fresh draw and
# the reported spread includes draw variance rather than treating one draw as the
# truth. These are stored per seed and looked up as `{name}_seed{seed}`.
SEED_DEPENDENT_BLOCKS = frozenset({"text_free_random_ablated"})

_REFERENCED_BLOCKS = {name for arm in ARMS for name in arm.blocks}
assert _REFERENCED_BLOCKS <= TEXT_BLOCKS | {"sequence"}, (
    f"unclassified feature blocks: {sorted(_REFERENCED_BLOCKS - TEXT_BLOCKS - {'sequence'})}"
)
assert not any(
    arm.shuffle_text and set(arm.blocks) & SEED_DEPENDENT_BLOCKS for arm in ARMS
), "an arm cannot both permute a block and redraw it per seed"
assert len({arm.name for arm in ARMS}) == len(ARMS), (
    "arm names must be unique; summarize groups by name and would merge duplicates"
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


@dataclass(frozen=True)
class TextVariants:
    """The four free-text conditions, each aligned to the table's rows.

    Deriving these separately from embedding them keeps the derivation testable
    without a sentence encoder, and keeps the expensive step ignorant of the
    scientific judgement in the lexicon.
    """

    cleaned: list[str]
    ablated: list[str]
    random_ablated: dict[int, list[str]]
    results: list[AblationResult]


def build_text_variants(table: pd.DataFrame, seeds: tuple[int, ...]) -> TextVariants:
    """Derive the cleaned, ablated and length-matched-random free-text variants.

    The random variant removes the same *number* of sentences as the ablation did
    for that protein, drawn per protein from a seed that encodes the accession, so
    the draw does not depend on row order or on which cohort is being run. That
    matching is what makes the ablation interpretable: it empties the same proteins,
    and so hands out the same population of zero vectors, leaving the compartment
    vocabulary as the only difference between the two arms.
    """
    pattern = compile_term_pattern(
        tuple(term for group in COMPARTMENT_TERMS.values() for term in group)
    )
    exclusions = compile_term_pattern(COMPARTMENT_EXCLUSIONS)

    raw = table["function_text"].fillna("").tolist()
    accessions = table["accession"].tolist()
    results = [ablate_sentences(text, pattern, exclusions) for text in raw]

    variants = TextVariants(
        cleaned=[clean_annotation_text(text) for text in raw],
        ablated=[" ".join(result.kept) for result in results],
        random_ablated={
            seed: [
                random_ablate_sentences(
                    text, len(result.removed), sentence_seed(accession, seed)
                )
                for text, result, accession in zip(raw, results, accessions)
            ]
            for seed in seeds
        },
        results=results,
    )

    for name, texts in [("cleaned", variants.cleaned), ("ablated", variants.ablated)]:
        assert len(texts) == len(table), f"{name} text misaligned with the table"
    for seed, texts in variants.random_ablated.items():
        assert len(texts) == len(table), f"random-ablated seed {seed} misaligned"
    return variants


def build_feature_blocks(
    table: pd.DataFrame, variants: TextVariants, cache_dir: Path
) -> dict[str, np.ndarray]:
    """Compute (or load) one embedding matrix per modality, aligned to table rows."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"embedding {len(table)} sequences with {SEQUENCE_ENCODER}")
    sequence = cached_embeddings(
        table["sequence"].tolist(),
        SEQUENCE_ENCODER,
        cache_dir / "sequence_esm2_35m.npz",
        EMBED_BATCH_SIZE,
    )

    log.info(f"embedding free text with {DEFAULT_SENTENCE_ENCODER}")
    text_free = cached_text_embeddings(
        table["function_text"].fillna("").tolist(),
        DEFAULT_SENTENCE_ENCODER,
        cache_dir / "text_free_minilm.npz",
        TEXT_BATCH_SIZE,
    )

    log.info("embedding structured annotation text")
    text_structured = cached_text_embeddings(
        structured_text(table),
        DEFAULT_SENTENCE_ENCODER,
        cache_dir / "text_structured_minilm.npz",
        TEXT_BATCH_SIZE,
    )

    # Each variant gets its own cache file. Sharing one path between two inputs
    # would make every run miss and recompute, permanently alternating between the
    # two, so the separate files are what keep all of them warm.
    log.info("embedding the free-text ablation variants")
    blocks = {
        "sequence": sequence,
        "text_free": text_free,
        "text_structured": text_structured,
        "text_free_cleaned": cached_text_embeddings(
            variants.cleaned,
            DEFAULT_SENTENCE_ENCODER,
            cache_dir / "text_free_cleaned_minilm.npz",
            TEXT_BATCH_SIZE,
        ),
        "text_free_ablated": cached_text_embeddings(
            variants.ablated,
            DEFAULT_SENTENCE_ENCODER,
            cache_dir / "text_free_ablated_minilm.npz",
            TEXT_BATCH_SIZE,
        ),
    }
    for seed, texts in variants.random_ablated.items():
        blocks[f"text_free_random_ablated_seed{seed}"] = cached_text_embeddings(
            texts,
            DEFAULT_SENTENCE_ENCODER,
            cache_dir / f"text_free_random_ablated_seed{seed}_minilm.npz",
            TEXT_BATCH_SIZE,
        )

    for name, block in blocks.items():
        assert len(block) == len(table), f"{name} embeddings misaligned with table"
        log.info(f"  {name:36} {block.shape}")
    return blocks


def assemble_features(
    arm: Arm, blocks: dict[str, np.ndarray], row_order: np.ndarray, seed: int
) -> np.ndarray:
    """Concatenate this arm's feature blocks for the given rows.

    For the shuffled control, the text block is permuted across proteins before
    slicing, so each protein keeps its own sequence but receives some other
    protein's annotation. The permutation is seeded, and it is applied to the whole
    dataset rather than within a split, mirroring the real pairing being broken.

    Seed-dependent blocks resolve to that seed's copy, so the random-ablation
    control draws afresh per seed rather than reusing one draw three times.
    """
    columns = []
    for block_name in arm.blocks:
        key = (
            f"{block_name}_seed{seed}"
            if block_name in SEED_DEPENDENT_BLOCKS
            else block_name
        )
        assert key in blocks, f"{arm.name} wants block {key!r}, which was not built"
        block = blocks[key]
        if arm.shuffle_text and block_name in TEXT_BLOCKS:
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


def _random_control_retention(
    results: list[AblationResult], random_texts: list[str]
) -> dict[str, float]:
    """Character retention of the length-matched control, for comparison.

    The control matches the number of sentences removed, not the number of
    characters, and localization sentences need not be average length. Reporting
    both retentions makes any gap visible rather than assumed away.
    """
    assert len(results) == len(random_texts), "control texts misaligned with results"

    before = sum(result.characters_before for result in results)
    assert before > 0, "no annotated characters to compare against"

    # Count the control the same way AblationResult does, as a sum over sentence
    # lengths, so the two retentions are commensurate.
    after = sum(
        len(sentence) for text in random_texts for sentence in split_sentences(text)
    )
    return {
        "ablated_retention": sum(r.characters_after for r in results) / before,
        "random_retention": after / before,
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
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"where the run log and manifest go (default: {DEFAULT_LOG_DIR})",
    )
    args = parser.parse_args()

    with run_context("run-arms", log_dir=args.log_dir, params=vars(args)) as run:
        with run.step("load prepared table"):
            table = load_table(
                args.data_root / "processed" / "deeploc_annotated.parquet"
            )

        # Batch sizes do not change the vectors, so they are deliberately absent
        # from the cache key, but they do change how long the run takes. Recording
        # them keeps a wall-clock comparison between two manifests honest: without
        # this, a run that changed both the code and the batch size looks from the
        # artifacts like a run that changed only the code.
        run.record("embed_batch_size", EMBED_BATCH_SIZE)
        run.record("text_batch_size", TEXT_BATCH_SIZE)

        run.record("ablation_lexicon_version", ABLATION_LEXICON_VERSION)
        run.record(
            "ablation_terms",
            len({term for group in COMPARTMENT_TERMS.values() for term in group}),
        )

        with run.step("derive text variants"):
            variants = build_text_variants(table, SEEDS)

        with run.step("build feature blocks"):
            blocks = build_feature_blocks(
                table, variants, args.data_root / "processed" / "embeddings"
            )

        cohort = "all proteins"
        results = variants.results
        if args.annotated_only:
            # Subset after embedding, so cached vectors are shared with the full run.
            keep = table.index[table["has_function_text"]].to_numpy()
            table = table.loc[keep].reset_index(drop=True)
            blocks = {name: block[keep] for name, block in blocks.items()}
            results = [variants.results[index] for index in keep]
            cohort = "annotated subset"
        run.record("cohort", cohort)
        run.record("proteins", len(table))

        # The ablation statistics describe the annotated proteins within whichever
        # cohort is running, since a protein with no text has nothing to ablate and
        # would otherwise dilute every retention figure toward 100%.
        with run.step("summarize the ablation"):
            annotated = table.index[table["has_function_text"]].to_numpy()
            ablation = ablation_summary(
                [results[index] for index in annotated],
                table["localization"].iloc[annotated].tolist(),
            )
            ablation["lexicon_version"] = ABLATION_LEXICON_VERSION
            ablation["residual_sentinel_mentions"] = term_mention_counts(
                [" ".join(results[index].kept) for index in annotated],
                SENTINEL_TERMS,
            )
            # The random control matches sentence count, not character count, so
            # its retention is recorded too: if the two diverge materially, the
            # comparison is not as clean as the design intends and should say so.
            ablation["random_control_retention"] = _random_control_retention(
                [results[index] for index in annotated],
                [variants.random_ablated[SEEDS[0]][index] for index in annotated],
            )
            run.record("ablation", ablation)
            log.info(
                "ablation: %d/%d proteins trimmed, %d emptied, %.1f%% of characters kept",
                ablation["proteins_trimmed"]["count"],
                ablation["proteins"],
                ablation["proteins_emptied"]["count"],
                100 * ablation["characters"]["corpus_retention"],
            )

        class_names = sorted(table["localization"].unique())
        label_index = {name: index for index, name in enumerate(class_names)}
        labels = table["localization"].map(label_index).to_numpy()
        run.record("classes", class_names)

        runs: list[dict[str, Any]] = []
        for seed in SEEDS:
            indices = split_indices(table, seed)
            sizes = {split: len(rows) for split, rows in indices.items()}
            run.record(f"split_sizes_seed_{seed}", sizes)

            for arm in ARMS:
                with run.step(f"seed {seed}: {arm.name}"):
                    result = run_arm(arm, blocks, indices, labels, class_names, seed)
                    runs.append(result)
                    log.info(
                        "%-24s acc=%.3f macroF1=%.3f (best epoch %d/%d)",
                        arm.name,
                        result["accuracy"],
                        result["macro_f1"],
                        result["best_epoch"],
                        result["epochs_run"],
                    )

        test_labels = [
            class_names[index] for index in labels[split_indices(table, 0)["test"]]
        ]
        majority = majority_class_accuracy(test_labels)
        run.record("majority_class_accuracy", round(majority, 4))
        summary = summarize(runs)

        with run.step("write results"):
            args.results_dir.mkdir(parents=True, exist_ok=True)
            suffix = "annotated" if args.annotated_only else "all"
            summary.to_csv(args.results_dir / f"arms_{suffix}.csv", index=False)
            markdown = render_markdown(summary, majority, cohort, len(test_labels))
            (args.results_dir / f"arms_{suffix}.md").write_text(markdown)
            (args.results_dir / f"per_class_f1_{suffix}.json").write_text(
                json.dumps(
                    {
                        result["arm"]: result["per_class_f1"]
                        for result in runs
                        if result["seed"] == SEEDS[0]
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            (args.results_dir / f"ablation_{suffix}.json").write_text(
                json.dumps(ablation, indent=2, sort_keys=True)
            )
            for _, row in summary.iterrows():
                run.record(
                    f"result_{row['arm']}",
                    {
                        "accuracy": round(row["accuracy"], 4),
                        "accuracy_sd": round(row["accuracy_sd"], 4),
                        "macro_f1": round(row["macro_f1"], 4),
                        "macro_f1_sd": round(row["macro_f1_sd"], 4),
                    },
                )

        # A copy of the manifest beside the committed metrics, so a number in the
        # writeup can be traced to the commit and device that produced it. Deferred
        # to run exit so the copy records the final status rather than "running".
        run.also_write_manifest_to(args.results_dir / f"run_manifest_{suffix}.json")
        log.info("results written to %s", args.results_dir)
        log.info("\n%s", markdown)

    return 0


if __name__ == "__main__":
    sys.exit(main())
