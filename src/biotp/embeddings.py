"""ESM-2 embedding extraction with on-disk caching.

Frozen embeddings are the cheap backbone for every project: extract once, cache,
reuse. See PLANNING.md ("Shared infrastructure").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_esm2(model_name: str):
    """Load an ESM-2 checkpoint and its alphabet/tokenizer.

    Args:
        model_name: an ESM-2 identifier, e.g. "esm2_t12_35M_UR50D".

    Returns:
        The model and whatever tokenizer/alphabet the caller needs to batch
        sequences. Exact return type is decided during implementation.
    """
    raise NotImplementedError


def embed_sequences(model, sequences: list[str], batch_size: int) -> np.ndarray:
    """Return per-sequence embeddings (mean-pooled over residues).

    Args:
        model: a loaded ESM-2 model.
        sequences: amino-acid strings.
        batch_size: sequences per forward pass.

    Returns:
        Array of shape (len(sequences), embedding_dim), where embedding_dim is
        fixed by the checkpoint (e.g. 320 for esm2_t6_8M, 480 for esm2_t12_35M,
        640 for esm2_t30_150M, 1280 for esm2_t33_650M), not a caller argument.
        Read it from the model if needed. To force a chosen output width,
        add a projection in the downstream head, not here.
    """
    raise NotImplementedError


def cached_embeddings(
    sequences: list[str], model_name: str, cache_path: Path, batch_size: int
) -> np.ndarray:
    """Load embeddings from cache_path, or compute and cache them if absent.

    The cache key must cover both the sequences and model_name so a changed
    model never silently reuses stale vectors.
    """
    raise NotImplementedError
