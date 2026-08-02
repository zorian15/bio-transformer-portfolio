"""A converged frozen-feature reference for rung 2, as a sanity floor.

Rung 2 trains an MLP head with Adam under early stopping, and how well that head
is fit is a property of the optimizer settings rather than of the representation.
Rung 3 uses a different batch size and learning rate, so an epoch is a different
number of gradient updates on the two rungs: 5x more for rung 3 at N=32 and 25x
at N=2048. Any rung-2-to-rung-3 delta therefore mixes "adapting the encoder
helped" with "rung 3 optimized longer".

This script closes the ambiguity from one side. Ridge regression on the *same*
cached features, the *same* splits and the *same* subsample draws has a
closed-form solution, so it is converged by construction and cannot be
under-fit. It is not a fourth rung and not a competitor: it is the answer to
"how much of the delta is the frozen representation, fit properly?"

The penalty is chosen on the same validation fold rung 2 selects its epoch on,
so the two see identical data. Test folds are never touched during selection.
Features are standardized using training statistics only.

Run from the repo root, after run_arms.py has cached the embeddings:

    python projects/dms-benchmark/scripts/frozen_reference.py

Writes `results/frozen_ridge.csv`, one row per configuration, with the same key
columns as the other rungs so it joins to them directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from biotp.evaluation import spearman
from biotp.runlog import DEFAULT_LOG_DIR, get_logger, run_context

# Spans seven orders of magnitude because the readouts differ enormously in
# conditioning: a mean-pooled vector over a 300-residue protein barely moves
# between variants, so its useful penalty is far larger than a single residue's.
# Selected on validation, so a wide grid costs time rather than validity. The
# top end was extended to 1e8 after 38 of 324 configurations pinned at 1e5,
# which would have understated the baseline this exists to establish.
ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)

log = get_logger("dms-frozen-reference")


def load_run_arms() -> Any:
    """Import the ladder runner, whose splits and subsampling this reuses.

    Imported rather than reimplemented: a reference baseline that drew its own
    splits would answer a different question than the rung it is a floor for,
    and the difference would be invisible in the output.
    """
    path = Path(__file__).resolve().parent / "run_arms.py"
    spec = importlib.util.spec_from_file_location("dms_run_arms", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ridge_spearman(
    features: np.ndarray,
    targets: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
) -> tuple[float, float]:
    """Fit ridge at every penalty, pick on validation MSE, score the test fold.

    Returns (spearman, alpha). The alpha is reported so a penalty pinned at an
    end of the grid is visible rather than silently accepted.
    """
    mean = features[train_rows].mean(axis=0)
    scale = features[train_rows].std(axis=0) + 1e-8
    train = (features[train_rows] - mean) / scale
    val = (features[val_rows] - mean) / scale
    test = (features[test_rows] - mean) / scale

    best_model, best_alpha, best_mse = None, float("nan"), np.inf
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(train, targets[train_rows])
        mse = float(np.mean((model.predict(val) - targets[val_rows]) ** 2))
        if mse < best_mse:
            best_mse, best_alpha, best_model = mse, alpha, model

    assert best_model is not None, "no penalty was selected, which cannot happen"
    predictions = best_model.predict(test)
    return spearman(targets[test_rows].tolist(), predictions.tolist()), best_alpha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--results-dir", type=Path, default=Path("projects/dms-benchmark/results")
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    run_arms = load_run_arms()

    with run_context(
        "dms-frozen-reference", log_dir=args.log_dir, params=vars(args)
    ) as run:
        with run.step("load prepared inputs"):
            table, metadata = run_arms.load_inputs(args.data_root)

        run.record("alphas", list(ALPHAS))
        cache_dir = args.data_root / "processed" / "dms_embeddings"
        rows: list[dict[str, Any]] = []

        with run.step("fit ridge over the rung-2 grid"):
            for assay_id in sorted(metadata["assays"]):
                info = metadata["assays"][assay_id]
                assay = table[table.dms_id == assay_id].reset_index(drop=True)
                targets = assay["DMS_score"].to_numpy(dtype=np.float64)

                for scheme in run_arms.SCHEMES:
                    splits = run_arms.make_splits(assay, assay_id, scheme)
                    for readout in run_arms.LORA_READOUTS:
                        features = run_arms.assay_features(
                            assay,
                            assay_id,
                            readout,
                            info["target_seq"],
                            run_arms.LADDER_CHECKPOINT,
                            cache_dir,
                        )
                        for n in run_arms.TRAINING_SIZES:
                            for seed in run_arms.SEEDS:
                                train_rows = run_arms.subsample(
                                    splits.train_pool, n, assay_id, scheme, seed
                                )
                                score, alpha = ridge_spearman(
                                    features,
                                    targets,
                                    train_rows,
                                    splits.val,
                                    splits.test,
                                )
                                rows.append(
                                    {
                                        "rung": "frozen_ridge",
                                        "assay": assay_id,
                                        "scheme": scheme,
                                        "readout": readout,
                                        "n": n,
                                        "seed": seed,
                                        "checkpoint": run_arms.LADDER_CHECKPOINT,
                                        "spearman": score,
                                        "alpha": alpha,
                                        "n_test": len(splits.test),
                                        "n_val": len(splits.val),
                                    }
                                )
                    log.info(f"{assay_id} {scheme}: {len(rows)} configurations so far")

        with run.step("write results"):
            args.results_dir.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(rows)
            destination = args.results_dir / "frozen_ridge.csv"
            frame.to_csv(destination, index=False)
            log.info(f"wrote {destination} ({len(frame)} rows)")

        run.record("configurations_run", len(rows))
        run.record("median_spearman", float(np.median(frame["spearman"])))
        # A penalty pinned at either end means the grid was too narrow, which
        # would understate the baseline and flatter the rung it is a floor for.
        run.record(
            "alpha_at_grid_edge",
            int(frame["alpha"].isin((ALPHAS[0], ALPHAS[-1])).sum()),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
