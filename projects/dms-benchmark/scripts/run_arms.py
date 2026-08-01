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

**Rung 3 is one configuration per invocation** so the SLURM array in the third PR
can map a task id onto it directly. `--all` loops the same entry point locally,
which is what the smoke test and the cheap rungs use.

Run from the repo root, after prepare_data.py:

    python projects/dms-benchmark/scripts/run_arms.py --rung zero_shot --all
    python projects/dms-benchmark/scripts/run_arms.py --rung frozen --all
    python projects/dms-benchmark/scripts/run_arms.py --rung lora \
        --assay R1AB_SARS2_Flynn_2022 --scheme fold_modulo_5 \
        --readout at_position --n 128 --seed 0
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
LORA_BATCH_SIZE = 8

MAX_EPOCHS = 200
LEARNING_RATE = 1e-3
LORA_LEARNING_RATE = 1e-4
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ("q_proj", "v_proj")

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


def variant_split(assay: pd.DataFrame, rows: np.ndarray, wildtype: str | None):
    """Build the VariantSplit train_lora and predict_lora consume."""
    subset = assay.iloc[rows]
    return VariantSplit(
        sequences=subset["mutated_sequence"].tolist(),
        positions=subset["position"].tolist(),
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
        lr=LEARNING_RATE,
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
        train_data=variant_split(assay, rows, reference),
        val_data=variant_split(assay, splits.val, reference),
        readout=readout,  # type: ignore[arg-type]
        max_epochs=MAX_EPOCHS,
        lr=LORA_LEARNING_RATE,
        batch_size=LORA_BATCH_SIZE,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        seed=seed,
    )
    test = variant_split(assay, splits.test, reference)
    predictions = predict_lora(
        encoder, head, test, readout, LORA_BATCH_SIZE  # type: ignore[arg-type]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rung", required=True, choices=["zero_shot", "frozen", "lora"]
    )
    parser.add_argument(
        "--all", action="store_true", help="loop the whole rung locally"
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

    with run_context("dms-run-arms", log_dir=args.log_dir, params=vars(args)) as run:
        with run.step("load prepared inputs"):
            table, metadata = load_inputs(args.data_root)
            assays = tuple(sorted(metadata["assays"]))
            run.record("assays", list(assays))
            run.record("variants", len(table))

        if args.all:
            configs = list(grid(args.rung, assays))
        else:
            for name in ("assay", "scheme", "seed"):
                assert getattr(args, name) is not None, (
                    f"--{name} is required without --all, so an array task maps "
                    "onto exactly one configuration"
                )
            if args.rung != "zero_shot":
                assert args.readout and args.n, "--readout and --n are required"
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
            args.results_dir.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(rows)
            destination = args.results_dir / f"{args.rung}.csv"
            if not args.all and destination.exists():
                frame = pd.concat([pd.read_csv(destination), frame], ignore_index=True)
                frame = frame.drop_duplicates(
                    subset=[
                        "rung",
                        "assay",
                        "scheme",
                        "readout",
                        "n",
                        "seed",
                        "checkpoint",
                    ],
                    keep="last",
                )
            frame.to_csv(destination, index=False)
            log.info(f"wrote {destination} ({len(frame)} rows)")

        run.record("configurations_run", len(configs))
        run.record("median_spearman", float(np.median([r["spearman"] for r in rows])))

    return 0


if __name__ == "__main__":
    sys.exit(main())
