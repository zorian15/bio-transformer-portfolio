"""Run the three-rung DMS ladder from issue #11.

    rung 1  zero-shot masked-marginals      the pretrained prior, no labels
    rung 2  frozen embeddings + MLP head    what supervision buys at a fixed
                                            representation
    rung 3  LoRA + the same head            what adapting the representation buys
                                            on top

Each step changes exactly one thing, and all three run the same checkpoint. The
headline is rung 2 to rung 3; rung 1 is what makes that number legible, because
without it "supervision reached Spearman 0.6" has no floor to be measured against.

**Splits.** ProteinGym ships five folds per scheme. Fold 0 is the test set, fold 1
is validation for early stopping, and folds 2-4 are the pool the training subsets
are drawn from. This is a single held-out fold rather than full 5-fold
cross-validation, which is what the pre-registration says and what the compute
budget assumes; rotating the test fold would multiply every arm by five.

**Why the schemes differ.** Under `random`, folds share almost every residue
position, so a model can learn site-specific effects and score well without
transferring anything. Under `modulo` and `contiguous` the folds are
position-disjoint by construction, which this script asserts rather than trusts.

**Rung 3 is one configuration per invocation**, and one SLURM array task is one
configuration: 324 of them, being 3 assays x 3 schemes x 3 readouts x 4 training
sizes x 3 seeds. `--task-id` maps an array index onto a configuration through
`grid`, so the mapping is Python that a test can reach rather than arithmetic in
a batch script. `--all` loops the same entry point locally, which is what the
smoke test and the cheap rungs use.

**One configuration writes one file.** A task writes its own shard under
`<results-dir>/<rung>_shards/`, and `--aggregate` combines them into
`<rung>.csv` afterwards. The earlier behavior, read-modify-write against a single
CSV, silently lost rows when two tasks finished close together: both read the
pre-existing file and both wrote it, and the loser disappeared into a well-formed
CSV with fewer rows than jobs that reported success.

Run from the repo root, after prepare_data.py:

    python projects/dms-benchmark/scripts/run_arms.py --rung zero_shot --all
    python projects/dms-benchmark/scripts/run_arms.py --rung frozen --all
    python projects/dms-benchmark/scripts/run_arms.py --rung lora \
        --assay R1AB_SARS2_Flynn_2022 --scheme fold_modulo_5 \
        --readout at_position --n 128 --seed 0

    # As the array runs it, then once afterwards to combine the shards:
    python projects/dms-benchmark/scripts/run_arms.py --rung lora --task-id 0
    python projects/dms-benchmark/scripts/run_arms.py --rung lora --aggregate
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biotp.embeddings import cached_embeddings, load_esm2
from biotp.evaluation import spearman
from biotp.runlog import DEFAULT_LOG_DIR, get_logger, run_context
from biotp.training import (
    LORA_READOUTS,
    LoraSpec,
    VariantSplit,
    build_head,
    predict,
    predict_lora,
    train,
    train_lora,
)
from biotp.utils import set_seed
from biotp.zero_shot import masked_marginal_scores

# The checkpoint every rung shares. If rung 3 ran at a different size because a
# GPU made it affordable, the rung 2 to rung 3 delta would conflate adaptation
# with model scale and the headline would be uninterpretable.
LADDER_CHECKPOINT = "esm2_t12_35M_UR50D"

# Rung 1 additionally runs at 650M. It costs one forward pass per distinct
# mutated position, so it is nearly free at any size, and reporting both answers
# the objection that a weak prior flattered the supervised rungs.
ZERO_SHOT_CHECKPOINTS = ("esm2_t12_35M_UR50D", "esm2_t33_650M_UR50D")

SCHEMES = ("fold_random_5", "fold_modulo_5", "fold_contiguous_5")

# The three rungs of the ladder, named once so argparse and the manifest agree
# on the set and a fourth rung cannot be added to one without the other.
RUNGS = ("zero_shot", "frozen", "lora")

# Fold 0 tests, fold 1 selects, folds 2-4 supply training data. Fixed rather than
# rotated: see the module docstring.
TEST_FOLD = 0
VAL_FOLD = 1

TRAINING_SIZES = (32, 128, 512, 2048)

# Validation exists only to pick the early-stopping epoch, and ProteinGym's folds
# are ~1000-1270 variants each. Left uncapped, the fine-tuned rung re-encodes all
# of them every epoch: at N=32 that is validating on thirty times more data than
# it trains on, and validation becomes ~90% of the run.
#
# Capped identically for both supervised rungs, not just the expensive one. The
# ladder's whole claim is that its rungs differ in exactly one thing, and model
# selection on different data would be a second difference.
VAL_SUBSAMPLE = 256
SEEDS = (0, 1, 2)

# Batch sizes. On MPS the binding constraint is the attention matrix rather than
# device utilisation, so these are memory knobs; see the 2026-07-30 log entry.
EMBED_BATCH_SIZE = 8
ZERO_SHOT_BATCH_SIZE = 8
# Scoring only. predict_lora's batching cannot change a result, so it is free to
# differ from the training batch size and is kept separate to say so.
LORA_SCORING_BATCH_SIZE = 8

MAX_EPOCHS = 200

# One optimiser for both supervised rungs, read by run_frozen and run_lora alike.
# They used to differ: rung 2 batched at 256 with lr 1e-3, rung 3 at 8 with 1e-4.
# Early-stopping patience is counted in epochs, so an epoch was 5x (N=32) to 25x
# (N=2048) more gradient updates on rung 3, and the rung-2-to-rung-3 delta mixed
# "adapting the encoder helped" with "rung 3 optimised for longer". The ladder's
# whole claim is that consecutive rungs differ in exactly one respect, and with
# two optimisers it was false. See issue #33.
#
# Rung 3's values win because rung 3 is the constrained one: its batch size is a
# memory limit set by the attention matrix, while rung 2 runs on cached vectors
# and can use any batch size. tests/test_dms_run_arms.py asserts both rungs read
# these, so the agreement is a property rather than a coincidence.
SUPERVISED_BATCH_SIZE = 8
SUPERVISED_LEARNING_RATE = 1e-4

# The adapter configuration every rung-3 run uses. One object rather than three
# constants, so the SLURM array can carry it per job and the run manifest records
# it as one block.
LORA_SPEC = LoraSpec(rank=8, alpha=16, target_modules=("q_proj", "v_proj"))

log = get_logger("dms-run-arms")


@dataclass(frozen=True)
class Config:
    """One point of the grid, and the unit a SLURM array task maps onto."""

    rung: str
    assay: str
    scheme: str
    readout: str
    n: int
    seed: int
    checkpoint: str


@dataclass(frozen=True)
class Splits:
    """Row positions into one assay's table, by role."""

    test: np.ndarray
    val: np.ndarray
    train_pool: np.ndarray


def load_inputs(data_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the prepared variant table and its per-assay metadata."""
    table_path = data_root / "processed" / "proteingym_variants.parquet"
    metadata_path = data_root / "processed" / "proteingym_assays.json"
    assert table_path.exists(), f"missing {table_path}; run prepare_data.py first"
    assert metadata_path.exists(), f"missing {metadata_path}; run prepare_data.py first"
    return pd.read_parquet(table_path), json.loads(metadata_path.read_text())


def make_splits(assay: pd.DataFrame, assay_id: str, scheme: str) -> Splits:
    """Partition one assay by ProteinGym's fold assignment for `scheme`.

    Asserts the property the scheme claims: under `modulo` and `contiguous` no
    residue position may appear in both the training pool and the test fold. That
    is the whole reason those schemes exist, and a silent violation would turn a
    generalization result into a memorization one.
    """
    assert scheme in SCHEMES, f"unknown scheme {scheme!r}"
    folds = assay[scheme].to_numpy()

    full_val = np.flatnonzero(folds == VAL_FOLD)
    # Fixed per (assay, scheme) and independent of N and seed, so every arm picks
    # its stopping epoch against the same held-out variants.
    val_rng = np.random.default_rng(
        [zlib.crc32(assay_id.encode()), zlib.crc32(scheme.encode()), 7919]
    )
    val = (
        full_val
        if len(full_val) <= VAL_SUBSAMPLE
        else np.sort(val_rng.choice(full_val, size=VAL_SUBSAMPLE, replace=False))
    )

    splits = Splits(
        test=np.flatnonzero(folds == TEST_FOLD),
        val=val,
        train_pool=np.flatnonzero((folds != TEST_FOLD) & (folds != VAL_FOLD)),
    )
    for name, rows in asdict(splits).items():
        assert len(rows) > 0, f"{scheme} produced an empty {name} split"

    if scheme in ("fold_modulo_5", "fold_contiguous_5"):
        positions = assay["position"].to_numpy()
        shared = set(positions[splits.train_pool]) & set(positions[splits.test])
        assert not shared, (
            f"{scheme} is supposed to hold out residue positions, but "
            f"{len(shared)} appear in both the training pool and the test fold"
        )
    return splits


def subsample(train_pool: np.ndarray, n: int, assay: str, scheme: str, seed: int):
    """Draw n training rows, independently per n as the pre-registration specifies.

    The draw deliberately does not depend on the readout, so the three readouts
    are compared on identical training sets and their difference is attributable
    to the representation rather than to which variants each one happened to see.
    """
    assert n <= len(train_pool), (
        f"asked for {n} training variants but the pool for {assay}/{scheme} holds "
        f"{len(train_pool)}"
    )
    # Seeded by the whole configuration so a rerun of one array task reproduces,
    # and so two configurations do not silently share a draw.
    #
    # crc32 rather than hash(): Python randomizes string hashing per process, so
    # `hash(assay)` would give a different training subset on every invocation
    # while every number downstream stayed plausible. `set_seed` sets
    # PYTHONHASHSEED, but only for processes started afterwards, not this one.
    rng = np.random.default_rng(
        [zlib.crc32(assay.encode()), zlib.crc32(scheme.encode()), n, seed]
    )
    return rng.choice(train_pool, size=n, replace=False)


def variant_split(
    assay: pd.DataFrame, rows: np.ndarray, readout: str, wildtype: str | None
):
    """Build the VariantSplit train_lora and predict_lora consume.

    Both `positions` and `wildtype` are readout-dependent, and `_check_split`
    treats either one arriving where it does not apply as an error rather than
    something to ignore: the mean readout pools every residue and cannot honour
    a position, so carrying one would mean the caller thinks it asked for
    something the run will not do.

    `wildtype` was threaded that way from the start and `positions` was not,
    which failed every mean configuration of rung 3 at the guard.
    """
    subset = assay.iloc[rows]
    return VariantSplit(
        sequences=subset["mutated_sequence"].tolist(),
        positions=None if readout == "mean" else subset["position"].tolist(),
        targets=subset["DMS_score"].to_numpy(dtype=np.float32),
        wildtype=wildtype,
    )


@lru_cache(maxsize=8)
def _assay_features_cached(
    assay_id: str,
    readout: str,
    wildtype: str,
    checkpoint: str,
    cache_dir: str,
    sequences: tuple[str, ...],
    positions: tuple[int, ...],
) -> np.ndarray:
    """Embed one assay's whole variant set once, for one readout.

    The cache key hashes the exact list of items, so embedding only the rows of
    the current arm would give every arm a different key and therefore a miss.
    Embedding the assay once and indexing into it is what makes rung 2 cost one
    pass rather than one per grid point: 324 arms share three matrices.

    `lru_cache` needs hashable arguments, hence the tuples; the on-disk cache
    survives across processes and this only avoids repeating work within one.
    """
    directory = Path(cache_dir)
    if readout == "mean":
        return cached_embeddings(
            list(sequences),
            checkpoint,
            directory / f"{assay_id}_{checkpoint}_mean.npz",
            EMBED_BATCH_SIZE,
            readout="mean",
            positions=None,
        )

    mutant = cached_embeddings(
        list(sequences),
        checkpoint,
        directory / f"{assay_id}_{checkpoint}_at_position.npz",
        EMBED_BATCH_SIZE,
        readout="at_position",
        positions=list(positions),
    )
    if readout == "at_position":
        return mutant

    # The wild type is one sequence, so the reference term needs only the
    # distinct positions rather than one row per variant.
    distinct = sorted(set(positions))
    reference = cached_embeddings(
        [wildtype] * len(distinct),
        checkpoint,
        directory / f"{assay_id}_{checkpoint}_wildtype.npz",
        EMBED_BATCH_SIZE,
        readout="at_position",
        positions=distinct,
    )
    index = {position: row for row, position in enumerate(distinct)}
    return mutant - reference[[index[position] for position in positions]]


def assay_features(
    assay: pd.DataFrame,
    assay_id: str,
    readout: str,
    wildtype: str,
    checkpoint: str,
    cache_dir: Path,
) -> np.ndarray:
    """Frozen features for every variant of one assay, aligned to its row order."""
    features = _assay_features_cached(
        assay_id,
        readout,
        wildtype,
        checkpoint,
        str(cache_dir),
        tuple(assay["mutated_sequence"].tolist()),
        tuple(int(position) for position in assay["position"]),
    )
    assert len(features) == len(assay), (
        f"got {len(features)} feature rows for {len(assay)} variants; the cached "
        "matrix does not describe this assay"
    )
    return features


@lru_cache(maxsize=2)
def cached_bundle(checkpoint: str) -> Any:
    """Load a checkpoint once per process.

    `--all` walks many configurations against the same encoder, and 650M is a
    2.5 GB load. Without this the zero-shot sweep would pay that cost once per
    grid point rather than once per checkpoint.
    """
    return load_esm2(checkpoint)


def run_zero_shot(
    assay: pd.DataFrame, splits: Splits, wildtype: str, checkpoint: str
) -> dict[str, Any]:
    """Rung 1: score the test fold from the pretrained likelihood alone."""
    subset = assay.iloc[splits.test]
    variants = [
        [(int(position), str(wt), str(mut))]
        for position, wt, mut in zip(
            subset["position"], subset["wildtype_aa"], subset["mutant_aa"]
        )
    ]
    bundle = cached_bundle(checkpoint)
    scores = masked_marginal_scores(bundle, wildtype, variants, ZERO_SHOT_BATCH_SIZE)
    return {
        "spearman": spearman(subset["DMS_score"].tolist(), scores.tolist()),
        "n_test": len(subset),
    }


def run_frozen(
    assay: pd.DataFrame,
    assay_id: str,
    splits: Splits,
    rows: np.ndarray,
    readout: str,
    wildtype: str,
    checkpoint: str,
    cache_dir: Path,
) -> dict[str, Any]:
    """Rung 2: supervision at a fixed representation."""
    everything = assay_features(
        assay, assay_id, readout, wildtype, checkpoint, cache_dir
    )
    features = {
        name: everything[part]
        for name, part in (("train", rows), ("val", splits.val), ("test", splits.test))
    }
    labels = {
        name: assay.iloc[part]["DMS_score"].to_numpy(dtype=np.float32)
        for name, part in (("train", rows), ("val", splits.val), ("test", splits.test))
    }

    head = build_head(features["train"].shape[1], 1, "regression")
    head, history = train(
        head,
        (features["train"], labels["train"]),
        (features["val"], labels["val"]),
        mode="linear_probe",
        max_epochs=MAX_EPOCHS,
        lr=SUPERVISED_LEARNING_RATE,
        batch_size=SUPERVISED_BATCH_SIZE,
    )
    predictions = predict(head, features["test"])
    return {
        "spearman": spearman(labels["test"].tolist(), predictions.tolist()),
        "n_test": len(labels["test"]),
        "best_epoch": history["best_epoch"],
        "epochs_run": history["epochs_run"],
    }


def run_lora(
    assay: pd.DataFrame,
    splits: Splits,
    rows: np.ndarray,
    readout: str,
    wildtype: str,
    checkpoint: str,
    seed: int,
) -> dict[str, Any]:
    """Rung 3: the same head, with the encoder allowed to adapt."""
    reference = wildtype if readout == "difference_at_position" else None
    encoder = load_esm2(checkpoint)
    head = build_head(encoder.embedding_dim, 1, "regression")

    encoder, head, history = train_lora(
        encoder=encoder,
        head=head,
        train_data=variant_split(assay, rows, readout, reference),
        val_data=variant_split(assay, splits.val, readout, reference),
        readout=readout,  # type: ignore[arg-type]
        max_epochs=MAX_EPOCHS,
        lr=SUPERVISED_LEARNING_RATE,
        batch_size=SUPERVISED_BATCH_SIZE,
        lora=LORA_SPEC,
        seed=seed,
    )
    test = variant_split(assay, splits.test, readout, reference)
    predictions = predict_lora(
        encoder, head, test, readout, LORA_SCORING_BATCH_SIZE  # type: ignore[arg-type]
    )
    return {
        "spearman": spearman(test.targets.tolist(), predictions.tolist()),
        "n_test": len(test.targets),
        "best_epoch": history["best_epoch"],
        "epochs_run": history["epochs_run"],
        "trainable_encoder_parameters": history["trainable_encoder_parameters"],
    }


def evaluate(config: Config, table: pd.DataFrame, metadata: dict, cache_dir: Path):
    """Run one grid point and return the row it contributes to the results table."""
    assay = table[table["dms_id"] == config.assay].reset_index(drop=True)
    assert not assay.empty, f"no rows for assay {config.assay!r}"
    wildtype = metadata["assays"][config.assay]["target_seq"]

    splits = make_splits(assay, config.assay, config.scheme)
    set_seed(config.seed)

    row: dict[str, Any] = {**asdict(config)}
    if config.rung == "zero_shot":
        rows = np.array([], dtype=int)
        row.update(run_zero_shot(assay, splits, wildtype, config.checkpoint))
    else:
        rows = subsample(
            splits.train_pool, config.n, config.assay, config.scheme, config.seed
        )
        if config.rung == "frozen":
            row.update(
                run_frozen(
                    assay,
                    config.assay,
                    splits,
                    rows,
                    config.readout,
                    wildtype,
                    config.checkpoint,
                    cache_dir,
                )
            )
        else:
            row.update(
                run_lora(
                    assay,
                    splits,
                    rows,
                    config.readout,
                    wildtype,
                    config.checkpoint,
                    config.seed,
                )
            )

    # Recorded rather than inferred: under `contiguous` a small draw covers few
    # distinct sites, so a flat point on the curve may be a site-coverage limit
    # rather than a label-count limit, and only this number tells them apart.
    row["train_positions"] = (
        int(assay.iloc[rows]["position"].nunique()) if len(rows) else 0
    )
    row["n_train_pool"] = len(splits.train_pool)
    row["n_val"] = len(splits.val)
    return row


def grid(rung: str, assays: tuple[str, ...]) -> Iterator[Config]:
    """Every configuration of one rung, in a deterministic order."""
    if rung == "zero_shot":
        for assay_id in assays:
            for scheme in SCHEMES:
                for checkpoint in ZERO_SHOT_CHECKPOINTS:
                    yield Config(
                        "zero_shot", assay_id, scheme, "none", 0, 0, checkpoint
                    )
        return

    for assay_id in assays:
        for scheme in SCHEMES:
            for readout in LORA_READOUTS:
                for n in TRAINING_SIZES:
                    for seed in SEEDS:
                        yield Config(
                            rung, assay_id, scheme, readout, n, seed, LADDER_CHECKPOINT
                        )


def grid_size(rung: str, assays: tuple[str, ...]) -> int:
    """How many configurations this rung has, which is the SLURM array bound.

    Exists so the bound is derived from the same `grid` the tasks are indexed
    into, rather than retyped into the sbatch and left to drift when an axis
    changes. An array bound set too low finishes cleanly having skipped
    configurations, which is a silent shortfall rather than an error.
    """
    return len(list(grid(rung, assays)))


def config_for_task(rung: str, assays: tuple[str, ...], task_id: int) -> Config:
    """The configuration a SLURM array task index maps onto.

    Zero-based, matching `#SBATCH --array=0-<size-1>`. The mapping is `grid`'s
    own order, which is deterministic, so this adds no ordering of its own.

    Deliberately Python rather than argv arithmetic in the batch script: an
    off-by-one in a shell expression is invisible until the results are short by
    one configuration, and nothing in a batch file is reachable by a test.
    """
    configs = list(grid(rung, assays))
    assert 0 <= task_id < len(configs), (
        f"task id {task_id} is outside the {rung} grid, which has "
        f"{len(configs)} configurations (valid ids 0 to {len(configs) - 1}); "
        "check the --array bound in the sbatch against --grid-size"
    )
    return configs[task_id]


def shard_name(config: Config) -> str:
    """Filename holding one configuration's result.

    Keyed by the configuration rather than by the array task id, for two
    reasons. A rerun of the same configuration overwrites its own shard, so a
    requeued or manually repeated task is idempotent rather than duplicating a
    row. And the name stays meaningful if the grid is ever reordered or extended,
    where `task-7.csv` would quietly come to describe something else.

    Every field of Config appears, so two configurations cannot collide.
    """
    name = (
        f"{config.rung}-{config.assay}-{config.scheme}-{config.readout}"
        f"-n{config.n}-seed{config.seed}-{config.checkpoint}.csv"
    )
    # A Hugging Face style checkpoint like "facebook/esm2_t12_35M_UR50D" would
    # turn this into a nested path, so a task would write into a directory that
    # --aggregate never looks in and the shard would read as missing. Fail at the
    # name rather than three steps later at the aggregation.
    assert "/" not in name and "\\" not in name, (
        f"configuration produces a shard name containing a path separator: "
        f"{name!r}. A checkpoint holding a slash needs escaping first."
    )
    return name


def shard_dir(results_dir: Path, rung: str) -> Path:
    """Where one rung's per-task shards live, beside the aggregated CSV."""
    return results_dir / f"{rung}_shards"


def aggregate_shards(
    results_dir: Path, rung: str, expected: list[Config]
) -> pd.DataFrame:
    """Combine per-task shards into one frame, refusing anything incomplete.

    Every assertion here exists because the alternative is a well-formed CSV that
    under-reports. A preempted or OOM-killed array task leaves no shard, and
    aggregating whatever happens to be present would produce a file that looks
    finished, with no downstream check able to tell.

    Args:
        results_dir: the directory holding `<rung>_shards/`.
        rung: which rung to aggregate.
        expected: every configuration that should have produced a shard.

    Returns:
        One row per configuration, in `grid` order.
    """
    directory = shard_dir(results_dir, rung)
    assert directory.is_dir(), (
        f"no shard directory at {directory}; nothing to aggregate. Run the "
        f"array first, or point --results-dir at where its tasks wrote."
    )

    wanted = {shard_name(config): config for config in expected}
    found = {path.name for path in directory.glob("*.csv")}

    missing = sorted(set(wanted) - found)
    assert not missing, (
        f"{len(missing)} of {len(wanted)} configuration(s) produced no shard, so "
        f"aggregating now would silently under-report. Missing: "
        f"{missing[:10]}{f' and {len(missing) - 10} more' if len(missing) > 10 else ''}"
    )

    # A shard the grid does not contain is a leftover from an earlier grid, and
    # including it would report a configuration this run never asked for.
    unexpected = sorted(found - set(wanted))
    assert not unexpected, (
        f"{len(unexpected)} shard(s) in {directory} are not in this grid, so the "
        f"directory mixes two runs. Remove them or use a fresh --results-dir: "
        f"{unexpected[:10]}"
    )

    # Iterating `expected` rather than sorting filenames: sorting sorts strings,
    # which puts n2048 before n32 before n512 and would leave the aggregated CSV
    # in an order no other rung's CSV shares. `--all` writes in grid order, so
    # this is what keeps lora.csv row-comparable to frozen.csv and zero_shot.csv.
    frames = []
    for config in expected:
        name = shard_name(config)
        frame = pd.read_csv(directory / name)
        assert len(frame) == 1, (
            f"shard {name} holds {len(frame)} rows; one configuration is one row, "
            "so this file was written by something other than a single task"
        )
        # The filename says which configuration this is; the row has to agree.
        # A task that wrote the wrong row would otherwise be indistinguishable
        # from one that wrote the right one, and the aggregate would attribute a
        # number to an arm that never produced it.
        row = frame.iloc[0]
        for field, value in asdict(config).items():
            assert row[field] == value, (
                f"shard {name} claims {field}={value} in its name but holds "
                f"{row[field]!r}; the file and its contents describe different runs"
            )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def configuration_records(rung: str) -> dict[str, Any]:
    """The manifest entries describing how this rung was configured.

    Only rung 3 has configuration that argv does not already carry: the adapter
    hyperparameters are a module constant, so `vars(args)` records nothing about
    them and a manifest would otherwise describe a fine-tuning run without
    saying what was adapted. The commit hash would be the only trace, which is
    archaeology at best and useless from a dirty tree.

    A separate function rather than an inline `run.record`, so the rung-to-
    records mapping is testable without a pipeline run. Nothing else in this
    script is, which is how the gap this closes went unnoticed.
    """
    assert rung in RUNGS, f"unknown rung {rung!r}; expected one of {sorted(RUNGS)}"

    records: dict[str, Any] = {}
    if rung in ("frozen", "lora"):
        # Both supervised rungs, because the point of recording it is that the
        # two are comparable only while these agree. A manifest that named the
        # optimiser for one rung and not the other would be worse than silent.
        records["supervised"] = {
            "batch_size": SUPERVISED_BATCH_SIZE,
            "learning_rate": SUPERVISED_LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
        }
    if rung != "lora":
        return records
    # One nested block rather than three loose keys, matching the shape
    # train_lora reports in its history. Read it back with
    # LoraSpec.from_history_block.
    records["lora"] = LORA_SPEC.as_history_block()
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", required=True, choices=list(RUNGS))
    parser.add_argument(
        "--all", action="store_true", help="loop the whole rung locally"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        help="zero-based index into this rung's grid; one SLURM array task",
    )
    parser.add_argument(
        "--grid-size",
        action="store_true",
        help="print this rung's configuration count and exit, to size --array",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="combine per-task shards into <rung>.csv and exit",
    )
    parser.add_argument("--assay")
    parser.add_argument("--scheme", choices=SCHEMES)
    parser.add_argument("--readout", choices=list(LORA_READOUTS))
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", default=LADDER_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--results-dir", type=Path, default=Path("projects/dms-benchmark/results")
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    if args.grid_size:
        # Handled before run_context on purpose. This is a query rather than a
        # run: it changes nothing, so it should not leave a manifest in logs/,
        # and more importantly run_context logs to stdout, which would put a
        # dozen log lines in front of the number. The sbatch reads this with
        # command substitution to size --array, so stdout has to hold the number
        # and nothing else.
        _, metadata = load_inputs(args.data_root)
        print(grid_size(args.rung, tuple(sorted(metadata["assays"]))))
        return 0

    # `args.task_id is not None`, not `args.task_id`: task 0 is falsy, and it is
    # the id slurm/README.md tells you to run by hand when debugging. Under a
    # truthiness test this guard passed over exactly the case most likely to be
    # typed, and `--all --task-id 0` overwrote the combined CSV with one row.
    selected = {
        "--all": args.all,
        "--task-id": args.task_id is not None,
        "--aggregate": args.aggregate,
    }
    modes = [flag for flag, given in selected.items() if given]
    assert len(modes) <= 1, (
        f"{' and '.join(modes)} were given together, and they select different "
        "things to do. Whichever lost would be silently ignored: --all with "
        "--task-id runs one configuration and exits 0, and --aggregate with "
        "--task-id trains nothing."
    )

    if args.task_id is not None:
        # The task id names the configuration by itself, so an axis given
        # alongside it would be quietly discarded rather than honoured.
        ignored = [
            f"--{name}"
            for name in ("assay", "scheme", "readout", "n", "seed")
            if getattr(args, name) is not None
        ]
        assert not ignored, (
            f"--task-id selects the whole configuration, so {', '.join(ignored)} "
            "would be ignored. Give one or the other."
        )

    # Each array task needs its own manifest and log. run_context builds both
    # filenames from the run name and a timestamp with one-second resolution, so
    # 16 tasks starting in the same second under `%16` would share a name: the
    # manifest is written with write_text and would be overwritten, and the log
    # handler appends, so their lines would interleave into one file. That is the
    # same silent shortfall this script now avoids for the CSVs, one layer down,
    # and it would defeat issue #20's own requirement that a task's manifest
    # record what it ran.
    run_name = "dms-run-arms"
    if args.task_id is not None:
        run_name = f"{run_name}-task{args.task_id}"

    with run_context(run_name, log_dir=args.log_dir, params=vars(args)) as run:
        with run.step("load prepared inputs"):
            table, metadata = load_inputs(args.data_root)
            assays = tuple(sorted(metadata["assays"]))
            run.record("assays", list(assays))
            run.record("variants", len(table))

        for key, value in configuration_records(args.rung).items():
            run.record(key, value)

        if args.aggregate:
            with run.step("aggregate shards"):
                expected = list(grid(args.rung, assays))
                frame = aggregate_shards(args.results_dir, args.rung, expected)
                destination = args.results_dir / f"{args.rung}.csv"
                frame.to_csv(destination, index=False)
                log.info(f"wrote {destination} ({len(frame)} rows)")
            run.record("configurations_aggregated", len(frame))
            run.record("median_spearman", float(np.median(frame["spearman"])))
            return 0

        if args.task_id is not None:
            configs = [config_for_task(args.rung, assays, args.task_id)]
            run.record("task_id", args.task_id)
        elif args.all:
            configs = list(grid(args.rung, assays))
        else:
            for name in ("assay", "scheme", "seed"):
                assert getattr(args, name) is not None, (
                    f"--{name} is required without --all, so an array task maps "
                    "onto exactly one configuration"
                )
            if args.rung != "zero_shot":
                # `is not None` rather than truthiness, so `--n 0` fails on its
                # own terms below instead of being reported as a missing flag.
                assert (
                    args.readout is not None and args.n is not None
                ), "--readout and --n are required on a supervised rung"
                assert (
                    args.n > 0
                ), f"--n must be positive on a supervised rung, got {args.n}"
            configs = [
                Config(
                    args.rung,
                    args.assay,
                    args.scheme,
                    args.readout or "none",
                    args.n or 0,
                    args.seed,
                    args.checkpoint,
                )
            ]

        cache_dir = args.data_root / "processed" / "dms_embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        with run.step(f"run {len(configs)} {args.rung} configuration(s)"):
            for index, config in enumerate(configs, start=1):
                log.info(
                    "[%d/%d] %s %s %s readout=%s n=%d seed=%d %s",
                    index,
                    len(configs),
                    config.rung,
                    config.assay,
                    config.scheme,
                    config.readout,
                    config.n,
                    config.seed,
                    config.checkpoint,
                )
                rows.append(evaluate(config, table, metadata, cache_dir))
                log.info("    spearman=%.4f", rows[-1]["spearman"])

        with run.step("write results"):
            if args.all:
                # One process owns the file, so there is nothing to race with.
                args.results_dir.mkdir(parents=True, exist_ok=True)
                destination = args.results_dir / f"{args.rung}.csv"
                pd.DataFrame(rows).to_csv(destination, index=False)
                log.info(f"wrote {destination} ({len(rows)} rows)")
            else:
                # One configuration, one file. Never read-modify-write a shared
                # CSV here: this is the path a SLURM array takes, and two tasks
                # finishing close together would both read the old file and both
                # write it, dropping a row into a file that still looks complete.
                # `--aggregate` combines the shards once every task has finished.
                directory = shard_dir(args.results_dir, args.rung)
                directory.mkdir(parents=True, exist_ok=True)
                for config, row in zip(configs, rows, strict=True):
                    destination = directory / shard_name(config)
                    pd.DataFrame([row]).to_csv(destination, index=False)
                    log.info(f"wrote {destination}")

        run.record("configurations_run", len(configs))
        run.record("median_spearman", float(np.median([r["spearman"] for r in rows])))

    return 0


if __name__ == "__main__":
    sys.exit(main())
