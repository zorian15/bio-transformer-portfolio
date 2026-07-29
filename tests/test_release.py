"""Contract tests for biotp.release (currently a scaffold stub).

push_to_hub deliberately has no behavioral test here: exercising it for real
would upload to the Hugging Face Hub from the test suite. The skipped test below
marks where a mocked Hub client goes once the function is implemented.

See test_embeddings.py for how the `stub` marker and xfail_strict interact.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from biotp import release

stub = pytest.mark.xfail(
    raises=NotImplementedError, reason="release is a scaffold stub"
)

METRICS = {"spearman": 0.41, "baseline_spearman": 0.44}
LIMITATIONS = "Single seed, one protein family held out; not validated in vitro."


def test_build_model_card_requires_limitations() -> None:
    """Limitations are mandatory, so no release can quietly omit its caveats."""
    limitations = inspect.signature(release.build_model_card).parameters["limitations"]
    assert limitations.default is inspect.Parameter.empty


def test_push_to_hub_private_has_no_default() -> None:
    """Visibility must be stated per release so nothing leaks by accident."""
    private = inspect.signature(release.push_to_hub).parameters["private"]
    assert private.default is inspect.Parameter.empty


@stub
def test_build_model_card_returns_markdown_naming_the_project() -> None:
    card = release.build_model_card(
        project="grounding-multimodal",
        summary="Frozen ESM-2 plus text embeddings for subcellular localization.",
        training_data="Swiss-Prot entries with curated function text.",
        metrics=METRICS,
        limitations=LIMITATIONS,
    )
    assert isinstance(card, str)
    assert "grounding-multimodal" in card


@stub
def test_build_model_card_reports_metrics_and_limitations() -> None:
    """Including a negative result: the card must not drop an unflattering number."""
    card = release.build_model_card(
        project="grounding-multimodal",
        summary="Frozen ESM-2 plus text embeddings for subcellular localization.",
        training_data="Swiss-Prot entries with curated function text.",
        metrics=METRICS,
        limitations=LIMITATIONS,
    )
    assert LIMITATIONS in card
    for name in METRICS:
        assert name in card


@pytest.mark.skip(reason="needs a mocked Hub client; a live call would upload for real")
def test_push_to_hub_returns_the_repo_url(tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.touch()
    url = release.push_to_hub(
        repo_id="example/grounding-multimodal",
        weights_path=weights,
        model_card="# card",
        private=True,
    )
    assert url.startswith("https://")
