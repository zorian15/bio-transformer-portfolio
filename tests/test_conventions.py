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

import inspect
import tomllib
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
# a cleanup, or the marker gets renamed, and the suite stays green because the
# test simply stops being selected. These tests fail loudly in that case, which
# is the whole point of pinning a convention mechanically rather than in prose.

NETWORK_MARKER = "network"

# Changing any of these means the anchor no longer runs when the embedding code
# changes, so the path filter has to keep covering them.
PATHS_THAT_MUST_TRIGGER_THE_ANCHOR = (
    "src/biotp/embeddings.py",
    "tests/data/",
)


def workflows() -> dict[str, Any]:
    """Every workflow under .github/workflows, parsed, keyed by filename."""
    assert WORKFLOW_DIR.is_dir(), f"no workflows directory at {WORKFLOW_DIR}"
    parsed = {}
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        parsed[path.name] = yaml.safe_load(path.read_text())
    assert parsed, f"no workflow files found in {WORKFLOW_DIR}"
    return parsed


def run_steps(workflow: dict[str, Any]) -> list[str]:
    """Every `run:` command in a workflow, flattened across its jobs."""
    commands = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "run" in step:
                commands.append(step["run"])
    return commands


def workflows_running_the_network_suite() -> dict[str, Any]:
    """Workflows that select the network tests rather than the default suite."""
    return {
        name: workflow
        for name, workflow in workflows().items()
        if any(f"-m {NETWORK_MARKER}" in command for command in run_steps(workflow))
    }


def test_some_workflow_runs_the_network_suite() -> None:
    """Without this, the frozen-reference anchor never runs anywhere but locally."""
    assert workflows_running_the_network_suite(), (
        "no workflow runs `pytest -m network`, so "
        "test_embed_sequences_matches_the_frozen_reference only runs when a human "
        "remembers to ask for it. See issue #8."
    )


@pytest.mark.parametrize("required_path", PATHS_THAT_MUST_TRIGGER_THE_ANCHOR)
def test_the_network_workflow_triggers_on_embedding_changes(required_path: str) -> None:
    """A pull request touching the embedding path must run the anchor."""
    matching = workflows_running_the_network_suite()

    triggering = []
    for name, workflow in matching.items():
        # `on` parses as the boolean True under YAML 1.1, which reads "on" as a
        # keyword. Accept either spelling rather than depending on the parser.
        triggers = workflow.get("on", workflow.get(True, {}))
        paths = triggers.get("pull_request", {}).get("paths")
        if paths is None:
            # No filter at all means it runs on every pull request, which covers
            # the embedding path too.
            triggering.append(name)
        elif any(pattern.startswith(required_path) for pattern in paths):
            triggering.append(name)

    assert triggering, (
        f"no network-suite workflow triggers on changes to {required_path!r}; "
        f"checked {sorted(matching)}"
    )


def test_the_network_marker_is_registered() -> None:
    """`--strict-markers` turns a renamed or typo'd marker into an error."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    pytest_config = config["tool"]["pytest"]["ini_options"]

    declared = [entry.split(":")[0] for entry in pytest_config["markers"]]
    assert NETWORK_MARKER in declared, f"{NETWORK_MARKER} is no longer registered"
    assert "--strict-markers" in pytest_config["addopts"], (
        "without --strict-markers a mistyped marker silently selects nothing, "
        "which is how a load-bearing test stops running without failing"
    )
