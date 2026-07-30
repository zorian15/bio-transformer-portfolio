"""Run logging: a timestamped log file plus a machine-readable run manifest.

Two problems this solves, both hit in practice on this project.

First, visibility. Bare `print` into a redirected file is block-buffered, so a
long step (embedding 13,858 sequences) shows nothing at all until the process
exits or the buffer fills. `logging` handlers flush on every record, so the log
is readable while the pipeline is still running.

Second, provenance. `DECISION_LOG.md` entries are supposed to record the setup
that produced a number: data, model, config, device, code version. Reconstructing
that from memory after the fact is how logs become wrong. Every run here writes a
JSON manifest next to its log with the git commit, the device, package versions,
the parameters it was given, per-step timings, and whatever counts the script
chose to record. The log is for reading; the manifest is for citing.

Typical use from a pipeline script:

    with run_context("prepare-data", params=vars(args)) as run:
        run.record("proteins_parsed", len(records))
        with run.step("fetch annotations"):
            annotations = fetch_uniprot_annotations(...)

On exit the manifest records `status: completed`, or `status: failed` with the
exception type and message, before the exception propagates. A run that died
partway is therefore distinguishable from one that finished, which is the whole
point of logging completion rather than just progress.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Recorded in every manifest, since a result can hinge on any of them. Absent
# packages are reported as missing rather than omitted, so a manifest that lacks
# torch is distinguishable from one where torch failed to import.
RECORDED_PACKAGES = (
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "transformers",
    "sentence_transformers",
    "esm",
)

_HANDLER_TAG = "biotp-runlog"


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a pipeline component."""
    return logging.getLogger(f"biotp.{name}" if not name.startswith("biotp") else name)


def configure_logging(log_path: Path | None, level: str = "INFO") -> None:
    """Send log records to stdout, and to log_path when one is given.

    Safe to call more than once: handlers this function installed are replaced
    rather than stacked, so a second call cannot double every line.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    setattr(console, _HANDLER_TAG, True)
    root.addHandler(console)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_TAG, True)
        root.addHandler(file_handler)


def _git(*args: str) -> str | None:
    """Run a git command, returning None when git or the repo is unavailable."""
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state() -> dict[str, Any]:
    """Capture the code version, including whether the tree was dirty.

    `dirty` matters more than the commit: a number produced from uncommitted
    edits is not reproducible from the commit alone, and the manifest should say
    so rather than imply a clean provenance it cannot support.
    """
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def package_versions() -> dict[str, str]:
    """Version of each recorded package, without importing the heavy ones."""
    from importlib import metadata

    # sklearn's distribution name differs from its import name.
    distributions = {"sklearn": "scikit-learn", "esm": "fair-esm"}
    versions: dict[str, str] = {}
    for name in RECORDED_PACKAGES:
        try:
            versions[name] = metadata.version(distributions.get(name, name))
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def environment_state() -> dict[str, Any]:
    """Capture the machine and library versions the run executed against."""
    from biotp.utils import get_device

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": get_device(prefer_gpu=True),
        "packages": package_versions(),
    }


@dataclass
class RunLog:
    """Handle passed to a pipeline body: a logger, plus what to put in the manifest."""

    name: str
    log: logging.Logger
    log_path: Path | None
    manifest_path: Path | None
    params: dict[str, Any]
    started_at: str
    records: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    error: dict[str, str] | None = None
    _start_monotonic: float = field(default_factory=time.monotonic)

    def record(self, key: str, value: Any) -> None:
        """Attach a fact to the manifest, and log it.

        Use for the numbers a writeup would cite: row counts, class counts,
        realized split sizes, headline metrics.
        """
        self.records[key] = value
        self.log.info("%s = %s", key, value)

    @contextmanager
    def step(self, description: str) -> Iterator[None]:
        """Time a named stage, logging its start and end, recording its duration.

        A failing step is logged and timed before the exception propagates, so the
        manifest shows which stage died rather than only that the run failed.
        """
        self.log.info("start: %s", description)
        started = time.monotonic()
        try:
            yield
        except BaseException as error:
            elapsed = time.monotonic() - started
            self.steps.append(
                {
                    "name": description,
                    "duration_seconds": round(elapsed, 3),
                    "failed": True,
                }
            )
            self.log.error("failed after %.1fs: %s (%s)", elapsed, description, error)
            raise
        elapsed = time.monotonic() - started
        self.steps.append({"name": description, "duration_seconds": round(elapsed, 3)})
        self.log.info("done in %.1fs: %s", elapsed, description)

    def manifest(self) -> dict[str, Any]:
        """The manifest as a plain dict, safe to serialize."""
        payload: dict[str, Any] = {
            "run": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "duration_seconds": round(time.monotonic() - self._start_monotonic, 3),
            "params": _plain(self.params),
            "steps": self.steps,
            "records": _plain(self.records),
            "git": git_state(),
            "environment": environment_state(),
            "log_file": str(self.log_path) if self.log_path else None,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload

    def write_manifest(self, path: Path) -> Path:
        """Write the manifest to an explicit path, e.g. beside committed results."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n")
        return path


def _plain(value: Any) -> Any:
    """Coerce values into something json.dumps accepts, without silent loss."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # numpy scalars and anything else exotic; str() beats failing to write at all.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except (ValueError, TypeError):
            pass
    return str(value)


@contextmanager
def run_context(
    name: str,
    log_dir: Path = DEFAULT_LOG_DIR,
    params: dict[str, Any] | None = None,
    level: str = "INFO",
) -> Iterator[RunLog]:
    """Wrap a pipeline run: configure logging, then write a manifest on the way out.

    Args:
        name: short slug identifying the pipeline, used in both filenames.
        log_dir: where the log and manifest go. Gitignored by default.
        params: the run's configuration, typically `vars(args)`.
        level: logging level for this run.

    Yields:
        A RunLog to log through and to record facts on.

    The manifest is written whether the body succeeds or raises, so a crashed run
    leaves evidence rather than nothing. The exception is re-raised untouched.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{name}-{stamp}.log"
    manifest_path = log_dir / f"{name}-{stamp}.json"

    configure_logging(log_path, level=level)
    logger = get_logger(name)

    run = RunLog(
        name=name,
        log=logger,
        log_path=log_path,
        manifest_path=manifest_path,
        params=params or {},
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    git = git_state()
    logger.info("run %r starting", name)
    logger.info(
        "commit %s on %s (dirty=%s)", git["commit"], git["branch"], git["dirty"]
    )
    logger.info("log file: %s", log_path)

    try:
        yield run
    except BaseException as error:
        run.status = "failed"
        run.error = {"type": type(error).__name__, "message": str(error)[:500]}
        run.write_manifest(manifest_path)
        logger.exception("run %r failed", name)
        logger.info("manifest: %s", manifest_path)
        raise
    run.status = "completed"
    run.write_manifest(manifest_path)
    logger.info(
        "run %r completed in %.1fs", name, time.monotonic() - run._start_monotonic
    )
    logger.info("manifest: %s", manifest_path)
