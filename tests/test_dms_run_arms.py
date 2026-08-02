"""Tests for the dms-benchmark ladder runner.

Offline, against small hand-built frames. What is worth testing here is the split
and subsample logic: both would keep producing plausible Spearman values while
measuring something other than what the pre-registration says.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from biotp import training
from biotp.runlog import run_context
from biotp.training import LoraSpec

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "dms-benchmark"
    / "scripts"
    / "run_arms.py"
)


def load_run_arms() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dms_run_arms", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_arms = load_run_arms()


def assay_frame(position_of_fold: dict[int, list[int]]) -> pd.DataFrame:
    """Build an assay whose fold assignment and positions are chosen by hand."""
    rows = []
    for fold, positions in position_of_fold.items():
        for position in positions:
            rows.append(
                {
                    "position": position,
                    "DMS_score": float(position),
                    "fold_random_5": fold,
                    "fold_modulo_5": fold,
                    "fold_contiguous_5": fold,
                }
            )
    return pd.DataFrame(rows)


def spread(folds: int = 5, per_fold: int = 6) -> pd.DataFrame:
    """Position-disjoint folds, as modulo and contiguous both guarantee."""
    return assay_frame(
        {
            fold: list(range(fold * per_fold, (fold + 1) * per_fold))
            for fold in range(folds)
        }
    )


def test_splits_assign_every_fold_to_a_role() -> None:
    splits = run_arms.make_splits(spread(), "TEST", "fold_modulo_5")
    assert len(splits.test) == 6
    assert len(splits.val) == 6
    assert len(splits.train_pool) == 18
    total = len(splits.test) + len(splits.val) + len(splits.train_pool)
    assert total == 30, "every row belongs to exactly one role"


def test_splits_reject_position_leakage_under_modulo() -> None:
    """The property that makes the scheme worth running.

    If a position appeared in both the training pool and the test fold, the arm
    would measure memorization of site effects and report it as generalization,
    with nothing anomalous in the number.
    """
    leaky = assay_frame({0: [1, 2, 3], 1: [4, 5, 6], 2: [1, 7, 8], 3: [9], 4: [10]})
    with pytest.raises(AssertionError, match="hold out residue positions"):
        run_arms.make_splits(leaky, "TEST", "fold_modulo_5")


def test_splits_allow_shared_positions_under_random() -> None:
    """`random` is supposed to share positions, so the guard must not fire."""
    shared = assay_frame({0: [1, 2, 3], 1: [1, 2, 3], 2: [1, 2, 3], 3: [1], 4: [2]})
    splits = run_arms.make_splits(shared, "TEST", "fold_random_5")
    assert len(splits.train_pool) == 5


def test_splits_reject_an_unknown_scheme() -> None:
    with pytest.raises(AssertionError, match="unknown scheme"):
        run_arms.make_splits(spread(), "TEST", "fold_made_up_5")


# --- Subsampling --------------------------------------------------------------


POOL = np.arange(100)


def test_subsample_is_reproducible_across_processes() -> None:
    """Frozen values, which is what makes this catch the bug it was written for.

    An earlier version seeded the draw with `hash(assay)`. Python randomizes
    string hashing per process, so every invocation would have trained on a
    different subset while every downstream number stayed plausible. Asserting
    only that two calls agree would not have caught it, because within one
    process they do. A golden value fails the moment the seed stops being a pure
    function of the configuration.
    """
    drawn = run_arms.subsample(POOL, 5, "ASSAY_A", "fold_modulo_5", 0)
    assert sorted(drawn.tolist()) == sorted(
        run_arms.subsample(POOL, 5, "ASSAY_A", "fold_modulo_5", 0).tolist()
    )
    assert drawn.tolist() == [14, 78, 9, 73, 62]


def test_subsample_varies_with_every_part_of_the_configuration() -> None:
    base = run_arms.subsample(POOL, 8, "ASSAY_A", "fold_modulo_5", 0).tolist()
    assert base != run_arms.subsample(POOL, 8, "ASSAY_B", "fold_modulo_5", 0).tolist()
    assert base != run_arms.subsample(POOL, 8, "ASSAY_A", "fold_random_5", 0).tolist()
    assert base != run_arms.subsample(POOL, 8, "ASSAY_A", "fold_modulo_5", 1).tolist()


def test_subsample_draws_without_replacement() -> None:
    drawn = run_arms.subsample(POOL, 40, "ASSAY_A", "fold_modulo_5", 0)
    assert len(set(drawn.tolist())) == 40


def test_subsample_refuses_to_exceed_the_pool() -> None:
    """A quietly smaller training set would flatten the data-efficiency curve."""
    with pytest.raises(AssertionError, match="asked for"):
        run_arms.subsample(np.arange(10), 32, "ASSAY_A", "fold_modulo_5", 0)


def test_the_ladder_shares_one_checkpoint() -> None:
    """The invariant the rung-2-to-rung-3 delta depends on.

    Rung 1 additionally reports a second size, which is a separate arm rather
    than a substitution inside the ladder.
    """
    assert run_arms.LADDER_CHECKPOINT in run_arms.ZERO_SHOT_CHECKPOINTS
    assert len(run_arms.ZERO_SHOT_CHECKPOINTS) == 2


def test_the_grid_covers_the_pre_registered_axes() -> None:
    configs = list(run_arms.grid("lora", ("A", "B", "C")))
    assert len(configs) == 3 * 3 * 3 * 4 * 3, "assays x schemes x readouts x N x seeds"
    assert {c.checkpoint for c in configs} == {run_arms.LADDER_CHECKPOINT}
    assert {c.n for c in configs} == set(run_arms.TRAINING_SIZES)
    assert {c.seed for c in configs} == set(run_arms.SEEDS)


def test_validation_is_capped_and_stable() -> None:
    """Validation exists to pick a stopping epoch, not to be a second test set.

    Uncapped, the fine-tuned rung re-encodes the whole ~1000-variant fold every
    epoch, which at N=32 is thirty times more work than training. The cap has to
    be fixed per (assay, scheme) and independent of N and seed, so every arm
    selects its stopping epoch against the same variants.
    """
    big = assay_frame(
        {fold: list(range(fold * 400, (fold + 1) * 400)) for fold in range(5)}
    )
    first = run_arms.make_splits(big, "ASSAY_A", "fold_modulo_5")
    again = run_arms.make_splits(big, "ASSAY_A", "fold_modulo_5")

    assert len(first.val) == run_arms.VAL_SUBSAMPLE
    assert first.val.tolist() == again.val.tolist(), "must not vary between calls"
    assert (
        run_arms.make_splits(big, "ASSAY_B", "fold_modulo_5").val.tolist()
        != first.val.tolist()
    ), "different assays should not share a draw"


def test_a_small_validation_fold_is_left_alone() -> None:
    splits = run_arms.make_splits(spread(), "ASSAY_A", "fold_modulo_5")
    assert len(splits.val) == 6, "smaller than the cap, so untouched"


# --- What the manifest records about the configuration -------------------------
#
# The adapter hyperparameters are a module constant, not an argv flag, so they
# reach the manifest only if something records them deliberately. Nothing did:
# `train_lora` built the nested block in its history and this script dropped it,
# so a rung-3 manifest named the assay, scheme and seed but never said what was
# adapted. That went unnoticed because the unit tests assert on the history and
# nothing asserted on the script. These close that.


def test_rung_three_records_the_adapter_configuration() -> None:
    assert run_arms.configuration_records("lora") == {
        "lora": {"rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}
    }


@pytest.mark.parametrize("rung", ["zero_shot", "frozen"])
def test_the_other_rungs_record_no_adapter_configuration(rung: str) -> None:
    """A block describing adapters would be a lie on a rung that attaches none."""
    assert run_arms.configuration_records(rung) == {}


def test_every_rung_has_a_decision_recorded_for_it() -> None:
    """A fourth rung must not silently inherit rung 1's empty block.

    Asserting rather than merely calling: as first written this passed against a
    `configuration_records` whose entire body was `return {}`, which is exactly
    the regression its name claims to guard against.
    """
    records = {rung: run_arms.configuration_records(rung) for rung in run_arms.RUNGS}

    assert set(records) == set(run_arms.RUNGS)
    assert records["lora"], "rung 3 must record the adapter configuration it ran"


def test_an_unknown_rung_is_rejected() -> None:
    with pytest.raises(AssertionError, match="unknown rung"):
        run_arms.configuration_records("full_finetune")


def test_the_recorded_block_rebuilds_the_spec_that_actually_ran() -> None:
    """The manifest has to carry enough to reconstruct the run, not just describe it.

    This is what a SLURM array task does in reverse: the block is written to a
    manifest, read back, and turned into the object the next run is configured
    from. Asserting equality against LORA_SPEC rather than against literals ties
    the manifest to the constant the pipeline actually uses, so changing the rank
    without the manifest following it fails here.
    """
    block = run_arms.configuration_records("lora")["lora"]

    assert LoraSpec.from_history_block(block) == run_arms.LORA_SPEC


def test_the_adapter_block_survives_a_real_manifest(tmp_path: Path) -> None:
    """End to end through runlog, so JSON serialization is not assumed.

    `configuration_records` returning the right dict proves nothing if the
    manifest cannot hold a nested block, so this writes and reads a real one.
    """
    with run_context("records-test", log_dir=tmp_path, params={}) as run:
        for key, value in run_arms.configuration_records("lora").items():
            run.record(key, value)

    manifest = json.loads(next(tmp_path.glob("records-test-*.json")).read_text())

    assert manifest["records"]["lora"] == run_arms.LORA_SPEC.as_history_block()
    assert LoraSpec.from_history_block(manifest["records"]["lora"]) == (
        run_arms.LORA_SPEC
    )


# --- SLURM array: task mapping and per-task result shards -----------------------
#
# The array writes one file per configuration instead of read-modify-writing a
# single CSV. That pattern lost rows: `not args.all` is the array path, so two
# tasks finishing close together both read the pre-existing file and both wrote
# it, and the loser vanished into a well-formed CSV with fewer rows than jobs
# that reported success. Nothing cross-checked the two, which is the same shape
# as the rest of this project's hazards.


def lora_grid(assays: tuple[str, ...] = ("ASSAY_A", "ASSAY_B")) -> list:
    return list(run_arms.grid("lora", assays))


def write_shard(directory: Path, config, spearman: float = 0.5) -> None:
    """One configuration's result, as a task would leave it behind."""
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "rung": config.rung,
                "assay": config.assay,
                "scheme": config.scheme,
                "readout": config.readout,
                "n": config.n,
                "seed": config.seed,
                "checkpoint": config.checkpoint,
                "spearman": spearman,
            }
        ]
    ).to_csv(directory / run_arms.shard_name(config), index=False)


def test_a_task_id_maps_onto_the_grid_configuration() -> None:
    """The mapping lives in Python so it can be tested, not in the sbatch."""
    assays = ("ASSAY_A", "ASSAY_B")
    configs = lora_grid(assays)

    for index in (0, 1, len(configs) // 2, len(configs) - 1):
        assert run_arms.config_for_task("lora", assays, index) == configs[index]


@pytest.mark.parametrize("offset", [-1, 0])
def test_a_task_id_outside_the_grid_is_rejected(offset: int) -> None:
    """An off-by-one in the sbatch --array bound must not silently rerun task 0."""
    assays = ("ASSAY_A", "ASSAY_B")
    out_of_range = len(lora_grid(assays)) if offset == 0 else offset

    with pytest.raises(AssertionError, match="outside the .* grid"):
        run_arms.config_for_task("lora", assays, out_of_range)


def test_every_configuration_gets_its_own_shard_name() -> None:
    """The property the whole scheme rests on: no two tasks write the same file."""
    configs = lora_grid()
    names = [run_arms.shard_name(config) for config in configs]

    assert len(set(names)) == len(configs)


def test_aggregate_reconstructs_every_configuration(tmp_path: Path) -> None:
    configs = lora_grid()
    directory = run_arms.shard_dir(tmp_path, "lora")
    for config in configs:
        write_shard(directory, config)

    frame = run_arms.aggregate_shards(tmp_path, "lora", configs)

    assert len(frame) == len(configs)
    assert set(frame["assay"]) == {"ASSAY_A", "ASSAY_B"}


def test_aggregate_names_the_configurations_that_produced_nothing(
    tmp_path: Path,
) -> None:
    """A silently short CSV is the failure this whole change exists to prevent.

    A preempted or OOM-killed array task leaves no shard. Aggregating what
    happens to be present would produce a well-formed file that under-reports,
    and nothing downstream would notice.
    """
    configs = lora_grid()
    directory = run_arms.shard_dir(tmp_path, "lora")
    for config in configs[1:]:
        write_shard(directory, config)

    with pytest.raises(AssertionError, match="produced no shard"):
        run_arms.aggregate_shards(tmp_path, "lora", configs)


def test_aggregate_rejects_a_shard_from_another_grid(tmp_path: Path) -> None:
    """A stale shard would report a configuration this grid does not contain."""
    configs = lora_grid()
    directory = run_arms.shard_dir(tmp_path, "lora")
    for config in configs:
        write_shard(directory, config)
    stale = run_arms.Config(
        "lora", "ASSAY_GONE", "fold_modulo_5", "mean", 32, 0, "ckpt"
    )
    write_shard(directory, stale)

    with pytest.raises(AssertionError, match="not in this grid"):
        run_arms.aggregate_shards(tmp_path, "lora", configs)


def test_aggregate_rejects_a_shard_holding_more_than_one_row(tmp_path: Path) -> None:
    """One configuration is one row; anything else means a task wrote the wrong file."""
    configs = lora_grid()
    directory = run_arms.shard_dir(tmp_path, "lora")
    for config in configs:
        write_shard(directory, config)
    doubled = pd.concat(
        [pd.read_csv(directory / run_arms.shard_name(configs[0]))] * 2,
        ignore_index=True,
    )
    doubled.to_csv(directory / run_arms.shard_name(configs[0]), index=False)

    with pytest.raises(AssertionError, match="holds 2 rows"):
        run_arms.aggregate_shards(tmp_path, "lora", configs)


def test_aggregate_without_a_shard_directory_fails_loudly(tmp_path: Path) -> None:
    """Rather than writing an empty results file that looks like a finished run."""
    with pytest.raises(AssertionError, match="no shard directory"):
        run_arms.aggregate_shards(tmp_path, "lora", lora_grid())


def test_the_documented_lora_array_bound_is_the_real_one() -> None:
    """slurm/submit-finetune.sh hardcodes --array=0-323; this is what pins it.

    Three real assays, so the number in the batch script and the number the code
    produces cannot drift apart without a test noticing.
    """
    real_assays = (
        "A4GRB6_PSEAI_Chen_2020",
        "CCR5_HUMAN_Gill_2023",
        "R1AB_SARS2_Flynn_2022",
    )
    size = run_arms.grid_size("lora", real_assays)

    assert size == 324
    sbatch = (
        Path(__file__).resolve().parents[1] / "slurm" / "submit-finetune.sh"
    ).read_text()
    assert f"--array=0-{size - 1}%" in sbatch, (
        f"slurm/submit-finetune.sh does not declare --array=0-{size - 1}; the grid "
        "and the batch script disagree about how many tasks there are"
    )


# --- main(), the wiring the pure helpers hang off ------------------------------
#
# The helpers above were fully covered while every new line in main() was not,
# including the shard-writing branch that is the entire point of issue #20.
# Reverting it to the old read-modify-write would have failed no test. That is
# the same gap as the manifest one in #14: the library half tested, the script
# half assumed. These execute main() end to end with the expensive parts stubbed.


def run_main(monkeypatch, tmp_path: Path, argv: list[str], assays=("ASSAY_A",)):
    """Drive main() with load_inputs and evaluate stubbed, so no model is loaded.

    Everything between argument parsing and writing results is real: grid
    construction, task-id mapping, the shard-versus-CSV branch and the file
    layout. Only the two functions that need data and a GPU are replaced.
    """
    monkeypatch.setattr(
        run_arms, "load_inputs", lambda _root: (pd.DataFrame(), {"assays": assays})
    )
    monkeypatch.setattr(
        run_arms,
        "evaluate",
        lambda config, table, metadata, cache_dir: {
            "rung": config.rung,
            "assay": config.assay,
            "scheme": config.scheme,
            "readout": config.readout,
            "n": config.n,
            "seed": config.seed,
            "checkpoint": config.checkpoint,
            # Deliberately skewed across the N axis rather than constant, so a
            # manifest recording the mean instead of the median is detectable.
            "spearman": float(config.n),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_arms.py",
            *argv,
            "--data-root",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )
    return run_arms.main()


def test_a_task_writes_its_shard_and_never_the_combined_csv(
    monkeypatch, tmp_path: Path
) -> None:
    """The fix issue #20 asked for, executed rather than inferred from filenames.

    If this branch were reverted to read-modify-writing lora.csv, this fails:
    the shard would be absent and lora.csv present.
    """
    assert run_main(monkeypatch, tmp_path, ["--rung", "lora", "--task-id", "0"]) == 0

    results = tmp_path / "results"
    expected = run_arms.shard_name(run_arms.config_for_task("lora", ("ASSAY_A",), 0))

    assert (run_arms.shard_dir(results, "lora") / expected).is_file()
    assert not (results / "lora.csv").exists(), (
        "a task wrote the combined CSV; that is the read-modify-write path whose "
        "concurrent use silently drops rows"
    )


def test_two_tasks_leave_two_shards_and_no_shared_file(
    monkeypatch, tmp_path: Path
) -> None:
    """The race, exercised: the second task must not be able to clobber the first."""
    for task_id in ("0", "1"):
        run_main(monkeypatch, tmp_path, ["--rung", "lora", "--task-id", task_id])

    results = tmp_path / "results"
    shards = sorted(p.name for p in run_arms.shard_dir(results, "lora").glob("*.csv"))

    assert len(shards) == 2, f"expected one shard per task, got {shards}"
    assert not (results / "lora.csv").exists()


def test_rerunning_a_task_overwrites_its_own_shard(monkeypatch, tmp_path: Path) -> None:
    """A requeued array task must be idempotent, not append a second row."""
    run_main(monkeypatch, tmp_path, ["--rung", "lora", "--task-id", "0"])
    run_main(monkeypatch, tmp_path, ["--rung", "lora", "--task-id", "0"])

    shards = list(run_arms.shard_dir(tmp_path / "results", "lora").glob("*.csv"))

    assert len(shards) == 1
    assert len(pd.read_csv(shards[0])) == 1


def test_all_writes_the_combined_csv_rather_than_shards(
    monkeypatch, tmp_path: Path
) -> None:
    """One process owns the file, so --all keeps writing it directly."""
    assert run_main(monkeypatch, tmp_path, ["--rung", "lora", "--all"]) == 0

    results = tmp_path / "results"
    frame = pd.read_csv(results / "lora.csv")

    assert len(frame) == run_arms.grid_size("lora", ("ASSAY_A",))
    assert not run_arms.shard_dir(results, "lora").exists()


def test_aggregate_writes_the_combined_csv_from_shards(
    monkeypatch, tmp_path: Path
) -> None:
    """The whole array round trip: every task, then one aggregation."""
    size = run_arms.grid_size("lora", ("ASSAY_A",))
    for task_id in range(size):
        run_main(monkeypatch, tmp_path, ["--rung", "lora", "--task-id", str(task_id)])

    assert run_main(monkeypatch, tmp_path, ["--rung", "lora", "--aggregate"]) == 0

    frame = pd.read_csv(tmp_path / "results" / "lora.csv")
    assert len(frame) == size


def test_aggregate_refuses_a_short_run_rather_than_writing_it(
    monkeypatch, tmp_path: Path
) -> None:
    """A preempted task must block aggregation, not silently shrink the results."""
    run_main(monkeypatch, tmp_path, ["--rung", "lora", "--task-id", "0"])

    with pytest.raises(AssertionError, match="produced no shard"):
        run_main(monkeypatch, tmp_path, ["--rung", "lora", "--aggregate"])

    assert not (tmp_path / "results" / "lora.csv").exists()


def test_aggregated_rows_come_back_in_grid_order(monkeypatch, tmp_path: Path) -> None:
    """So lora.csv is row-comparable to the CSVs --all writes for the other rungs.

    Sorting shard filenames instead would order them as strings, putting n2048
    before n32 before n512.
    """
    size = run_arms.grid_size("lora", ("ASSAY_A",))
    for task_id in range(size):
        run_main(monkeypatch, tmp_path, ["--rung", "lora", "--task-id", str(task_id)])
    run_main(monkeypatch, tmp_path, ["--rung", "lora", "--aggregate"])

    frame = pd.read_csv(tmp_path / "results" / "lora.csv")
    expected = list(run_arms.grid("lora", ("ASSAY_A",)))

    assert list(zip(frame["readout"], frame["n"], frame["seed"])) == [
        (config.readout, config.n, config.seed) for config in expected
    ]


def test_grid_size_is_what_the_task_ids_actually_span(
    monkeypatch, tmp_path: Path
) -> None:
    """The bound and the mapping must agree, checked by using both rather than
    by restating one in terms of the other.

    Every id below grid_size resolves, and the next one does not.
    """
    assays = ("ASSAY_A",)
    size = run_arms.grid_size("lora", assays)

    resolved = {run_arms.config_for_task("lora", assays, i) for i in range(size)}

    assert len(resolved) == size
    with pytest.raises(AssertionError, match="outside the .* grid"):
        run_arms.config_for_task("lora", assays, size)


def test_each_task_writes_its_own_manifest(monkeypatch, tmp_path: Path) -> None:
    """Concurrent tasks must not share a manifest filename.

    run_context derives the manifest and log paths from the run name plus a
    one-second timestamp. Under `--array=...%16`, sixteen tasks start in the same
    second, so a shared name means write_text overwrites all but one manifest and
    the log handler interleaves their lines. Three tasks in one second used to
    leave two manifests. This is the CSV failure one layer down, and it defeats
    the requirement that a task's manifest record what it ran.
    """
    for task_id in ("0", "1", "2"):
        run_main(monkeypatch, tmp_path, ["--rung", "lora", "--task-id", task_id])

    manifests = sorted((tmp_path / "logs").glob("*.json"))
    recorded = [
        json.loads(path.read_text())["records"]["task_id"] for path in manifests
    ]

    assert sorted(recorded) == [0, 1, 2], (
        f"three tasks left {len(manifests)} manifests recording {recorded}; "
        "one overwrote another"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--all", "--task-id", "3"],
        ["--aggregate", "--task-id", "3"],
        ["--all", "--aggregate"],
        # Task 0 specifically: it is falsy, so a truthiness test let it through,
        # and it is the id the README tells you to run by hand when debugging.
        ["--all", "--task-id", "0"],
        ["--aggregate", "--task-id", "0"],
    ],
)
def test_modes_that_contradict_each_other_are_rejected(
    monkeypatch, tmp_path: Path, argv: list[str]
) -> None:
    """Whichever flag lost would be silently ignored, and exit 0 having done the
    wrong thing: --all --task-id ran one configuration, --aggregate --task-id
    trained nothing."""
    with pytest.raises(AssertionError, match="were given together"):
        run_main(monkeypatch, tmp_path, ["--rung", "lora", *argv])


def test_aggregate_rejects_a_shard_whose_contents_contradict_its_name(
    tmp_path: Path,
) -> None:
    """A task that wrote the wrong row is otherwise indistinguishable from one
    that wrote the right one, and the aggregate would attribute a number to an
    arm that never produced it."""
    configs = lora_grid()
    directory = run_arms.shard_dir(tmp_path, "lora")
    for config in configs:
        write_shard(directory, config)
    path = directory / run_arms.shard_name(configs[0])
    frame = pd.read_csv(path)
    frame.loc[0, "seed"] = 99
    frame.to_csv(path, index=False)

    with pytest.raises(AssertionError, match="describe different runs"):
        run_arms.aggregate_shards(tmp_path, "lora", configs)


def test_grid_size_prints_the_number_and_nothing_else(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """The sbatch reads this with command substitution to size --array.

    run_context logs to stdout, so handling --grid-size inside it would put a
    dozen log lines in front of the number and the batch script would set a
    nonsense bound. This pins the contract that comment argues for, which was
    previously only ever checked by hand.
    """
    assert run_main(monkeypatch, tmp_path, ["--rung", "lora", "--grid-size"]) == 0

    captured = capsys.readouterr()

    assert captured.out == f"{run_arms.grid_size('lora', ('ASSAY_A',))}\n"
    assert not (tmp_path / "logs").exists(), "a query should not leave a manifest"


def test_a_single_configuration_run_needs_the_axes_that_identify_it(
    monkeypatch, tmp_path: Path
) -> None:
    """Without --all there is no default configuration to fall back on."""
    with pytest.raises(AssertionError, match="is required without --all"):
        run_main(monkeypatch, tmp_path, ["--rung", "lora"])


def test_a_supervised_rung_without_a_readout_or_n_is_rejected(
    monkeypatch, tmp_path: Path
) -> None:
    """Otherwise the Config defaults fill in, and rung 3 trains on nothing.

    `readout or "none"` and `n or 0` exist for zero-shot, which has neither. On a
    supervised rung they would produce a run with no training data and an
    unusable readout, which still writes a shard and still reports a number.
    """
    with pytest.raises(AssertionError, match="--readout and --n are required"):
        run_main(
            monkeypatch,
            tmp_path,
            [
                "--rung",
                "lora",
                "--assay",
                "ASSAY_A",
                "--scheme",
                "fold_modulo_5",
                "--seed",
                "0",
            ],
        )


def test_explicit_axes_select_the_same_configuration_as_the_task_id(
    monkeypatch, tmp_path: Path
) -> None:
    """The two ways of naming one configuration must not disagree."""
    config = run_arms.config_for_task("lora", ("ASSAY_A",), 0)
    run_main(
        monkeypatch,
        tmp_path,
        [
            "--rung",
            "lora",
            "--assay",
            config.assay,
            "--scheme",
            config.scheme,
            "--readout",
            config.readout,
            "--n",
            str(config.n),
            "--seed",
            str(config.seed),
        ],
    )

    shards = list(run_arms.shard_dir(tmp_path / "results", "lora").glob("*.csv"))

    assert [path.name for path in shards] == [run_arms.shard_name(config)]


def test_task_zero_does_not_slip_past_the_mode_guard(
    monkeypatch, tmp_path: Path
) -> None:
    """Regression: `--all --task-id 0` used to overwrite the combined CSV.

    `args.task_id` was tested for truthiness, and 0 is falsy, so the mode guard
    passed over the one id most likely to be typed by hand. The run then took the
    --all branch's write path with a single configuration's row, replacing a
    complete lora.csv with a one-row file and exiting 0.
    """
    run_main(monkeypatch, tmp_path, ["--rung", "lora", "--all"])
    complete = len(pd.read_csv(tmp_path / "results" / "lora.csv"))

    with pytest.raises(AssertionError, match="were given together"):
        run_main(monkeypatch, tmp_path, ["--rung", "lora", "--all", "--task-id", "0"])

    assert len(pd.read_csv(tmp_path / "results" / "lora.csv")) == complete


def test_task_id_alongside_an_explicit_axis_is_rejected(
    monkeypatch, tmp_path: Path
) -> None:
    """The axis would be discarded, so the run would not be the one asked for."""
    with pytest.raises(AssertionError, match="would be ignored"):
        run_main(
            monkeypatch,
            tmp_path,
            ["--rung", "lora", "--task-id", "0", "--seed", "2"],
        )


def test_a_supervised_rung_rejects_a_zero_training_size(
    monkeypatch, tmp_path: Path
) -> None:
    """`--n 0` is falsy, so it used to be reported as a missing flag."""
    with pytest.raises(AssertionError, match="--n must be positive"):
        run_main(
            monkeypatch,
            tmp_path,
            [
                "--rung",
                "lora",
                "--assay",
                "ASSAY_A",
                "--scheme",
                "fold_modulo_5",
                "--readout",
                "mean",
                "--n",
                "0",
                "--seed",
                "0",
            ],
        )


def test_the_aggregate_run_records_what_it_combined(
    monkeypatch, tmp_path: Path
) -> None:
    """The aggregate manifest is the one a DECISION_LOG entry would cite.

    The task side was pinned and this side was not, so the numbers a writeup
    quotes were the ones nothing checked. Swapping the median for a mean, or
    dropping both records, used to keep the suite green.
    """
    size = run_arms.grid_size("lora", ("ASSAY_A",))
    for task_id in range(size):
        run_main(monkeypatch, tmp_path, ["--rung", "lora", "--task-id", str(task_id)])
    run_main(monkeypatch, tmp_path, ["--rung", "lora", "--aggregate"])

    manifests = sorted((tmp_path / "logs").glob("dms-run-arms-2*.json"))
    records = json.loads(manifests[-1].read_text())["records"]

    assert records["configurations_aggregated"] == size
    # The stub's spearman is the training size, so the four sizes appear equally
    # often: the median is 320 while the mean would be 680.
    assert records["median_spearman"] == float(
        np.median([float(config.n) for config in run_arms.grid("lora", ("ASSAY_A",))])
    )


def test_the_shard_directory_is_the_one_the_docs_spell_out() -> None:
    """`shard_dir` is otherwise only ever compared against itself.

    Renaming it would pass every test while contradicting the batch script and
    the README, which write the path out for a human to look in.
    """
    directory = run_arms.shard_dir(Path("results"), "lora")

    assert directory.name == "lora_shards"
    slurm = Path(__file__).resolve().parents[1] / "slurm"
    for path in (slurm / "README.md", slurm / "submit-finetune.sh"):
        assert directory.name in path.read_text(), (
            f"{path.name} does not mention {directory.name}; the docs and the "
            "code disagree about where a task writes its shard"
        )


def test_shard_directories_are_not_committed() -> None:
    """324 intermediates per rung must not land in git; <rung>.csv is the artifact.

    `results/` is a tracked directory, so without a rule the array leaves 324
    untracked files one `git add -A` away from being committed. Paired with
    `test_the_shard_directory_is_the_one_the_docs_spell_out`, which pins the name
    this pattern has to cover.
    """
    repo_root = Path(__file__).resolve().parents[1]
    ignored = (repo_root / ".gitignore").read_text()
    directory = run_arms.shard_dir(Path("results"), "lora").name

    assert any(
        line.strip() == "*_shards/" or line.strip() == f"{directory}/"
        for line in ignored.splitlines()
    ), f".gitignore has no rule covering {directory}/"


# --- variant_split against the guard it has to satisfy --------------------------
#
# The split builder and the guard that rejects a malformed split live in
# different modules, and nothing checked them against each other. `wildtype` was
# threaded through by readout from the start; `positions` was not, so every mean
# configuration of rung 3 died at `_check_split` after loading a checkpoint. The
# unit tests covered train_lora with a hand-built split, and the script tests
# stub `evaluate`, so the one path that builds a real split never met the guard.


def variant_frame(count: int, length: int) -> pd.DataFrame:
    """An assay carrying the columns variant_split reads."""
    wildtype = ("MKTFFVLLLACDEFGHIKLM" * 2)[:length]
    rows = []
    for index in range(count):
        position = index % length
        mutant = "ACDEFG"[index % 6]
        rows.append(
            {
                "mutated_sequence": wildtype[:position]
                + mutant
                + wildtype[position + 1 :],
                "position": position,
                "DMS_score": float(index),
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("readout", list(run_arms.LORA_READOUTS))
def test_variant_split_satisfies_the_guard_for_its_own_readout(readout: str) -> None:
    """The check that was missing: build a split, then run the real guard on it.

    `_check_split` is what train_lora calls first, so this reproduces the failure
    without a model or a GPU. Against the previous code the mean case raises
    "carries positions, but the mean readout pools every residue".
    """
    length = 20
    frame = variant_frame(count=6, length=length)
    wildtype = (
        frame["mutated_sequence"].iloc[0]
        if readout == "difference_at_position"
        else None
    )

    split = run_arms.variant_split(frame, np.arange(len(frame)), readout, wildtype)

    training._check_split(split, readout, length, "train")  # type: ignore[arg-type]


def test_the_mean_readout_split_carries_no_positions() -> None:
    """Named directly, because the guard test above would also pass if
    variant_split returned an empty split for every readout."""
    frame = variant_frame(count=6, length=20)

    mean = run_arms.variant_split(frame, np.arange(len(frame)), "mean", None)
    at_position = run_arms.variant_split(
        frame, np.arange(len(frame)), "at_position", None
    )

    assert mean.positions is None
    assert at_position.positions == frame["position"].tolist()
    assert len(mean.sequences) == len(frame)


def test_slurm_output_goes_to_a_tracked_but_ignored_directory() -> None:
    """SLURM resolves --output before the job runs and will not create the path.

    Three things have to agree or a job silently produces no output at all: the
    batch scripts write into the directory, the directory survives a fresh clone,
    and its contents stay out of git. A mkdir inside the script would be too late
    to help.
    """
    repo_root = Path(__file__).resolve().parents[1]
    directory = "slurm-logs"

    assert (repo_root / directory / ".gitkeep").is_file(), (
        f"{directory}/.gitkeep is what makes the directory exist in a clone; "
        "without it SLURM has nowhere to write and the job produces no output"
    )
    for script in ("submit-finetune.sh", "submit-embed.sh"):
        text = (repo_root / "slurm" / script).read_text()
        assert f"--output={directory}/" in text, f"{script} writes outside {directory}/"

    ignored = (repo_root / ".gitignore").read_text().splitlines()
    assert f"{directory}/*" in [line.strip() for line in ignored]
    assert f"!{directory}/.gitkeep" in [line.strip() for line in ignored]


# --- Both supervised rungs must optimise the same way ---------------------------
#
# The ladder's claim is that consecutive rungs differ in exactly one respect.
# Rung 2 used to batch at 256 with lr 1e-3 and rung 3 at 8 with 1e-4, while both
# stopped after ten epochs without improvement. Patience in epochs plus different
# batch sizes meant rung 3 got 5x to 25x more gradient updates, so the delta the
# whole project exists to measure mixed adaptation with optimisation budget.
# Sharing the constants fixes it; this is what stops them drifting apart again.


def supervised_call_source(function_name: str) -> str:
    """The source of one rung's runner, for checking what it passes to training."""
    import inspect

    return inspect.getsource(getattr(run_arms, function_name))


@pytest.mark.parametrize("rung", ["run_frozen", "run_lora"])
def test_both_supervised_rungs_read_the_shared_optimiser(rung: str) -> None:
    """Neither may name its own learning rate or batch size.

    Checked in the source rather than by running the rungs, which would need a
    GPU and a checkpoint. The failure this guards against is a constant quietly
    reintroduced beside the shared one, which no numerical test would catch until
    the delta had already moved.
    """
    source = supervised_call_source(rung)

    assert "lr=SUPERVISED_LEARNING_RATE" in source, (
        f"{rung} does not pass SUPERVISED_LEARNING_RATE; the two supervised "
        "rungs must optimise identically or their delta is not attributable"
    )
    assert "batch_size=SUPERVISED_BATCH_SIZE" in source, (
        f"{rung} does not pass SUPERVISED_BATCH_SIZE; patience is counted in "
        "epochs, so a different batch size is a different optimisation budget"
    )


def test_no_rung_specific_optimiser_constants_survive() -> None:
    """The old names must be gone, not merely unused.

    A leftover LEARNING_RATE or LORA_BATCH_SIZE is an invitation to wire one rung
    back to it, which is exactly how the rungs diverged the first time.
    """
    for name in ("LEARNING_RATE", "LORA_LEARNING_RATE", "LORA_BATCH_SIZE"):
        assert not hasattr(run_arms, name), (
            f"{name} still exists; it is a per-rung optimiser setting and the "
            "supervised rungs now share SUPERVISED_LEARNING_RATE and "
            "SUPERVISED_BATCH_SIZE"
        )


def test_an_epoch_is_the_same_number_of_updates_on_both_rungs() -> None:
    """The property that makes shared epoch-patience a fair stopping rule.

    Equal batch sizes and equal training-set sizes mean equal updates per epoch,
    so "ten epochs without improvement" means the same thing on both rungs. This
    is the statement the fix actually rests on.
    """
    import math

    for n in run_arms.TRAINING_SIZES:
        frozen_updates = math.ceil(n / run_arms.SUPERVISED_BATCH_SIZE)
        lora_updates = math.ceil(n / run_arms.SUPERVISED_BATCH_SIZE)
        assert frozen_updates == lora_updates, f"N={n} differs across rungs"
