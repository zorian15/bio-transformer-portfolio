"""Tests for biotp.runlog.

The property that matters most is that a manifest exists and is honest after a
*failed* run. A logging layer that only records successes would let a crashed
pipeline look like one that never started.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from biotp import runlog


@pytest.fixture(autouse=True)
def clean_root_handlers() -> Iterator[None]:
    """Remove handlers this module installs, so tests cannot leak into each other."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, runlog._HANDLER_TAG, False):
            root.removeHandler(handler)
            handler.close()


def read_manifest(log_dir: Path) -> dict[str, Any]:
    manifests = list(log_dir.glob("*.json"))
    assert len(manifests) == 1, f"expected exactly one manifest, found {manifests}"
    parsed: dict[str, Any] = json.loads(manifests[0].read_text())
    return parsed


def test_run_context_writes_a_log_file_and_a_manifest(tmp_path: Path) -> None:
    with runlog.run_context("unit-test", log_dir=tmp_path) as run:
        run.log.info("hello from the pipeline")

    logs = list(tmp_path.glob("*.log"))
    assert len(logs) == 1
    assert "hello from the pipeline" in logs[0].read_text()
    assert read_manifest(tmp_path)["status"] == "completed"


def test_log_file_is_readable_before_the_run_finishes(tmp_path: Path) -> None:
    """The original motivation: a buffered file shows nothing until exit."""
    with runlog.run_context("unit-test", log_dir=tmp_path) as run:
        run.log.info("first line")
        written = next(tmp_path.glob("*.log")).read_text()
        assert "first line" in written, "log should be flushed while still running"


def test_manifest_records_params_and_records(tmp_path: Path) -> None:
    with runlog.run_context(
        "unit-test", log_dir=tmp_path, params={"seed": 3, "path": Path("data")}
    ) as run:
        run.record("proteins", 13858)

    manifest = read_manifest(tmp_path)
    assert manifest["params"] == {"seed": 3, "path": "data"}
    assert manifest["records"]["proteins"] == 13858


def test_manifest_captures_environment_and_git(tmp_path: Path) -> None:
    with runlog.run_context("unit-test", log_dir=tmp_path):
        pass

    manifest = read_manifest(tmp_path)
    assert manifest["environment"]["device"] in {"cuda", "mps", "cpu"}
    assert manifest["environment"]["packages"]["numpy"] != "missing"
    assert set(manifest["git"]) == {"commit", "branch", "dirty"}


def test_steps_are_timed_and_named(tmp_path: Path) -> None:
    with runlog.run_context("unit-test", log_dir=tmp_path) as run:
        with run.step("first stage"):
            pass
        with run.step("second stage"):
            pass

    steps = read_manifest(tmp_path)["steps"]
    assert [step["name"] for step in steps] == ["first stage", "second stage"]
    assert all(step["duration_seconds"] >= 0 for step in steps)
    assert not any(step.get("failed") for step in steps)


def test_failed_run_still_writes_a_manifest_and_reraises(tmp_path: Path) -> None:
    """A crashed pipeline must leave evidence, and must not swallow the error."""
    with (
        pytest.raises(ValueError, match="deliberate"),
        runlog.run_context("unit-test", log_dir=tmp_path) as run,
    ):
        run.record("got_this_far", True)
        raise ValueError("deliberate failure")

    manifest = read_manifest(tmp_path)
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "ValueError"
    assert "deliberate failure" in manifest["error"]["message"]
    assert manifest["records"]["got_this_far"] is True


def test_failing_step_is_marked_in_the_manifest(tmp_path: Path) -> None:
    """The manifest should say which stage died, not merely that the run did."""
    with (
        pytest.raises(RuntimeError),
        runlog.run_context("unit-test", log_dir=tmp_path) as run,
    ):
        with run.step("healthy stage"):
            pass
        with run.step("doomed stage"):
            raise RuntimeError("boom")

    steps = read_manifest(tmp_path)["steps"]
    assert steps[0]["name"] == "healthy stage"
    assert not steps[0].get("failed")
    assert steps[1]["name"] == "doomed stage"
    assert steps[1]["failed"] is True


def test_configure_logging_does_not_duplicate_handlers(tmp_path: Path) -> None:
    """Calling it twice must not double every line."""
    log_path = tmp_path / "run.log"
    runlog.configure_logging(log_path)
    runlog.configure_logging(log_path)

    runlog.get_logger("dup-check").info("only once please")
    occurrences = log_path.read_text().count("only once please")
    assert occurrences == 1, f"line appeared {occurrences} times"


def test_configure_logging_without_a_path_only_uses_stdout(tmp_path: Path) -> None:
    runlog.configure_logging(None)
    runlog.get_logger("stdout-only").info("no file wanted")
    assert list(tmp_path.glob("*.log")) == []


def test_git_state_tolerates_git_being_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside a repo the manifest reports unknown provenance instead of crashing."""
    monkeypatch.setattr(runlog, "_git", lambda *args: None)
    state = runlog.git_state()
    assert state == {"commit": None, "branch": None, "dirty": None}


def test_package_versions_marks_absent_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing package is reported, so its absence cannot be mistaken for a gap."""
    monkeypatch.setattr(runlog, "RECORDED_PACKAGES", ("numpy", "definitely_not_real"))
    versions = runlog.package_versions()
    assert versions["numpy"] != "missing"
    assert versions["definitely_not_real"] == "missing"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Path("a/b"), "a/b"),
        (np.float32(1.5), 1.5),
        (np.int64(7), 7),
        ({"nested": [Path("x"), np.int32(2)]}, {"nested": ["x", 2]}),
        (None, None),
        (True, True),
    ],
)
def test_plain_coerces_values_json_cannot_hold(value: Any, expected: Any) -> None:
    assert runlog._plain(value) == expected


def test_manifest_is_json_serializable_with_numpy_records(tmp_path: Path) -> None:
    """Numpy scalars are pervasive in this codebase and must not break the write."""
    with runlog.run_context("unit-test", log_dir=tmp_path) as run:
        run.record("macro_f1", np.float32(0.412))
        run.record("counts", {"Nucleus": np.int64(4043)})

    manifest = read_manifest(tmp_path)
    assert manifest["records"]["macro_f1"] == pytest.approx(0.412, abs=1e-6)
    assert manifest["records"]["counts"]["Nucleus"] == 4043


def test_extra_manifest_copy_records_the_final_status(tmp_path: Path) -> None:
    """A copy beside committed results must say completed, not running.

    Writing it inside the run body stamped `status: running`, which read as an
    unfinished run sitting next to metrics that had in fact completed.
    """
    target = tmp_path / "results" / "run_manifest.json"
    with runlog.run_context("unit-test", log_dir=tmp_path / "logs") as run:
        run.also_write_manifest_to(target)
        run.record("macro_f1", 0.41)

    copied = json.loads(target.read_text())
    assert copied["status"] == "completed"
    assert copied["records"]["macro_f1"] == 0.41


def test_extra_manifest_copy_is_written_on_failure_too(tmp_path: Path) -> None:
    target = tmp_path / "results" / "run_manifest.json"
    with (
        pytest.raises(RuntimeError),
        runlog.run_context("unit-test", log_dir=tmp_path / "logs") as run,
    ):
        run.also_write_manifest_to(target)
        raise RuntimeError("late failure")

    assert json.loads(target.read_text())["status"] == "failed"


def test_write_manifest_to_an_explicit_path(tmp_path: Path) -> None:
    """Used to drop provenance beside committed metrics."""
    target = tmp_path / "results" / "run_manifest.json"
    with runlog.run_context("unit-test", log_dir=tmp_path / "logs") as run:
        run.record("k", 1)
        run.write_manifest(target)

    assert json.loads(target.read_text())["records"]["k"] == 1
