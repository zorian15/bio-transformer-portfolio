"""Open-science release helpers: model cards and Hugging Face Hub uploads.

Every project ships weights plus a model card. These helpers keep releases
uniform and reproducible across the three projects. See the release checklist
in PLANNING.md.
"""

from __future__ import annotations

from pathlib import Path


def build_model_card(
    project: str,
    summary: str,
    training_data: str,
    metrics: dict,
    limitations: str,
) -> str:
    """Render a model card (Markdown) from structured fields.

    metrics is reported honestly, including negative or null results. limitations
    is required, not optional, so no release omits its caveats.
    """
    raise NotImplementedError


def push_to_hub(
    repo_id: str,
    weights_path: Path,
    model_card: str,
    private: bool,
) -> str:
    """Upload weights and model card to the Hugging Face Hub.

    Args:
        repo_id: target repo, e.g. "zorian15/grounding-multimodal".
        weights_path: local checkpoint to upload.
        model_card: rendered card text; written as the repo README.
        private: whether the repo starts private. No default, so each release
            states its visibility explicitly rather than leaking by accident.

    Returns:
        The URL of the created or updated Hub repo.
    """
    raise NotImplementedError
