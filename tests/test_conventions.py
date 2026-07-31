"""Repo-wide conventions checked mechanically, so they survive refactors.

Three things are pinned here: the shared API surface promised in PLANNING.md
still exists under the expected names, the stub APIs stay free of default
argument values so implementing them forces every call site to be explicit, and
CI still runs the network suite that guards the embedding path.

biotp.utils is deliberately outside the no-defaults check: `get_device(prefer_gpu=True)`
is an existing, intentional default. The convention is about not adding defaults
to these APIs as they grow, not about banning defaults everywhere.
"""

from __future__ import annotations

import fnmatch
import inspect
import itertools
import shlex
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

import biotp
from biotp import embeddings, evaluation, release, training

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

STUB_MODULES = [embeddings, evaluation, release, training]

EXPECTED_FUNCTIONS = {
    "biotp.embeddings": {
        "load_esm2",
        "embed_sequences",
        "cached_embeddings",
        # The text arms of grounding-multimodal need a second encoder under the
        # same cache contract; see PLANNING.md and issue #1.
        "load_sentence_encoder",
        "embed_texts",
        "cached_text_embeddings",
        # The cache key covers the embedding code, not only its inputs; these
        # build the code half of the key. See issue #4 and docs/embedding-cache.md.
        "sequence_embedding_spec",
        "text_embedding_spec",
    },
    "biotp.evaluation": {
        "grouped_split",
        "spearman",
        "classification_metrics",
        # Added for the localization writeup: a per-class breakdown and the
        # majority-class floor any arm has to clear.
        "per_class_f1",
        "majority_class_accuracy",
    },
    "biotp.release": {"build_model_card", "push_to_hub"},
    "biotp.training": {"build_head", "train", "predict"},
}


def public_functions(module: ModuleType) -> list[object]:
    """Return functions defined in this module, skipping imports and privates."""
    return [
        obj
        for name, obj in vars(module).items()
        if inspect.isfunction(obj)
        and not name.startswith("_")
        and obj.__module__ == module.__name__
    ]


def module_id(module: ModuleType) -> str:
    return module.__name__


def test_version_is_exported() -> None:
    assert biotp.__version__


@pytest.mark.parametrize("module", STUB_MODULES, ids=module_id)
def test_module_exposes_its_planned_api(module: ModuleType) -> None:
    names = {func.__name__ for func in public_functions(module)}  # type: ignore[attr-defined]
    assert names == EXPECTED_FUNCTIONS[module.__name__]


@pytest.mark.parametrize("module", STUB_MODULES, ids=module_id)
def test_stub_apis_declare_no_default_arguments(module: ModuleType) -> None:
    offenders = sorted(
        f"{func.__name__}({name}={param.default!r})"  # type: ignore[attr-defined]
        for func in public_functions(module)
        for name, param in inspect.signature(func).parameters.items()  # type: ignore[arg-type]
        if param.default is not inspect.Parameter.empty
    )
    assert not offenders, f"defaults added to {module.__name__}: {offenders}"


# --- CI actually runs the network suite (issue #8) ----------------------------
#
# `test_embed_sequences_matches_the_frozen_reference` is the only check that
# would catch the embedding path drifting away from the code that produced the
# committed results. It is `@pytest.mark.network`, and the default pytest run
# deselects network tests, so it protects nothing unless CI opts back in.
#
# The way that gap comes back is nobody noticing: a workflow gets deleted during
# a cleanup, its triggers get trimmed to save CI minutes, or the marker gets
# renamed, and the suite stays green because the test simply stops being
# selected. These tests fail in each of those cases, which is the whole point of
# pinning a convention mechanically rather than in prose.
#
# The guards themselves are only worth having if they fail when CI is broken, so
# each mutation they claim to catch has been checked by making it.

NETWORK_MARKER = "network"

# Concrete files rather than directory prefixes. The question a path filter has
# to answer is "would a pull request touching this file run the anchor", and that
# is only answerable against a real path.
#
# utils.py is here because "the embedding path" is wider than embeddings.py:
# `get_device` lives in utils and decides which backend produces the vectors, so
# a change there can move the numbers while embeddings.py stays untouched.
FILES_THAT_MUST_TRIGGER_THE_ANCHOR = (
    "src/biotp/embeddings.py",
    "src/biotp/utils.py",
    "tests/data/reference_embeddings.npz",
)

# A pull_request trigger carrying no `paths:` filter runs on everything.
MATCHES_EVERY_FILE = ["**"]


def workflows() -> dict[str, Any]:
    """Every workflow under .github/workflows, parsed, keyed by filename."""
    assert WORKFLOW_DIR.is_dir(), f"no workflows directory at {WORKFLOW_DIR}"

    paths = sorted(
        [*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")],
        key=lambda path: path.name,
    )
    assert paths, f"no workflow files found in {WORKFLOW_DIR}"
    return {path.name: yaml.safe_load(path.read_text()) for path in paths}


def triggers_of(workflow: dict[str, Any]) -> dict[str, Any]:
    """The `on:` block, normalized to a dict.

    YAML 1.1 reads a bare `on` as the boolean True, so the key can arrive either
    way depending on the parser. The list form (`on: [push]`) carries no
    per-event configuration, so it normalizes to empty bodies.
    """
    for key in ("on", True):
        if key in workflow:
            triggers = workflow[key]
            if isinstance(triggers, list):
                return {event: None for event in triggers}
            assert isinstance(triggers, dict), f"unexpected `on:` block: {triggers!r}"
            return triggers
    raise AssertionError(f"workflow has no `on:` block: {sorted(workflow)}")


def run_commands(workflow: dict[str, Any]) -> list[str]:
    """Every `run:` command in a workflow, flattened across its jobs."""
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict), f"workflow has no jobs: {jobs!r}"

    commands: list[str] = []
    for name, job in jobs.items():
        steps = job.get("steps")
        assert isinstance(steps, list), f"job {name!r} has no steps"
        commands.extend(step["run"] for step in steps if "run" in step)
    return commands


def shell_tokens(command: str) -> list[str]:
    """Tokenize a `run:` command, tolerating anything shlex cannot parse.

    Tokenizing rather than matching substrings is what makes quoting irrelevant,
    so `pytest -m 'network'` and `pytest -m network` are read the same way.
    """
    try:
        return shlex.split(command)
    except ValueError:
        return []


def marker_expression(tokens: list[str]) -> str | None:
    """The value of pytest's `-m` flag, or None when the invocation has none."""
    for flag, value in itertools.pairwise(tokens):
        if flag == "-m":
            return value
    return None


def workflows_running(marker: str | None) -> dict[str, Any]:
    """Workflows with a pytest step selecting exactly this marker expression.

    `marker=None` means the default suite: a pytest invocation carrying no `-m`
    at all. Whether a command *is* pytest is checked separately from what it
    selects, because collapsing the two makes every `run:` step in the file look
    like a pytest run with no marker, and then every workflow appears to run the
    tests. That is the same conflation as reading a missing `pull_request` key as
    an absent path filter, and it hides the same kind of regression.
    """
    matched = {}
    for name, workflow in workflows().items():
        for command in run_commands(workflow):
            tokens = shell_tokens(command)
            if "pytest" in tokens and marker_expression(tokens) == marker:
                matched[name] = workflow
                break
    return matched


def pull_request_paths(workflow: dict[str, Any]) -> list[str] | None:
    """The workflow's pull_request path filter.

    Returns None when the workflow has no pull_request trigger at all, which is
    the case that matters: it means no pull request can ever run this workflow.
    That is a different thing from a pull_request trigger with no `paths:`, which
    runs on every pull request, and conflating the two is how a guard passes
    while the job it guards has stopped running.
    """
    triggers = triggers_of(workflow)
    if "pull_request" not in triggers:
        return None

    configuration = triggers["pull_request"]
    if configuration is None:
        # `pull_request:` with an empty body is valid, and means every PR.
        return MATCHES_EVERY_FILE
    assert isinstance(
        configuration, dict
    ), f"unexpected pull_request block: {configuration!r}"

    return configuration.get("paths") or MATCHES_EVERY_FILE


def pattern_selects(pattern: str, file_path: str) -> bool:
    """Whether a GitHub `paths:` pattern selects this file.

    fnmatch's `*` spans `/`, which approximates GitHub's `**`. Matching as a glob
    rather than as a literal prefix is what lets the filter be broadened, say
    from `src/biotp/embeddings.py` to `src/biotp/**`, without failing a test that
    the broadening does not actually violate.
    """
    if fnmatch.fnmatch(file_path, pattern):
        return True
    # A pattern naming a directory, written without a glob.
    return file_path.startswith(pattern.rstrip("*").rstrip("/") + "/")


def test_some_workflow_runs_the_network_suite() -> None:
    """Without this, the frozen-reference anchor never runs outside a laptop."""
    assert workflows_running(NETWORK_MARKER), (
        "no workflow runs `pytest -m network`, so "
        "test_embed_sequences_matches_the_frozen_reference only runs when a human "
        "remembers to ask for it. See issue #8."
    )


def test_some_workflow_runs_the_offline_suite() -> None:
    """The default suite carries every other test, including these ones."""
    assert workflows_running(
        None
    ), "no workflow runs the default `pytest` suite, so nothing checks it on a PR"


@pytest.mark.parametrize("required_file", FILES_THAT_MUST_TRIGGER_THE_ANCHOR)
def test_the_network_workflow_triggers_on_embedding_changes(required_file: str) -> None:
    """A pull request touching the embedding path must run the anchor.

    Trimming the workflow's triggers is at least as likely as deleting it, and it
    leaves the file sitting there looking like protection, so this checks that a
    pull_request trigger exists as well as that its filter covers the file.
    """
    candidates = workflows_running(NETWORK_MARKER)
    assert candidates, "no workflow runs the network suite at all"

    triggering = [
        name
        for name, workflow in candidates.items()
        for patterns in [pull_request_paths(workflow)]
        if patterns is not None
        and any(pattern_selects(pattern, required_file) for pattern in patterns)
    ]
    assert triggering, (
        f"no network-suite workflow runs on a pull request touching "
        f"{required_file!r}; checked {sorted(candidates)}"
    )


def test_the_network_marker_is_registered(pytestconfig: pytest.Config) -> None:
    """Checked against pytest's resolved configuration, not the file it came from.

    `--strict-markers` turns a renamed or mistyped marker into an error rather
    than a silent selection of nothing, which is the other way a load-bearing
    test stops running while the suite stays green.
    """
    declared = [entry.split(":")[0] for entry in pytestconfig.getini("markers")]
    assert NETWORK_MARKER in declared, f"{NETWORK_MARKER} is no longer registered"
    assert pytestconfig.getoption(
        "strict_markers"
    ), "--strict-markers is not in effect, so a mistyped marker selects nothing"
