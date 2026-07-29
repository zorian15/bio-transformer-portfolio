"""Contract tests for biotp.embeddings (currently a scaffold stub).

Behavioral tests carry the `stub` marker below. `xfail_strict` is enabled in
pyproject.toml, so each one becomes a hard failure the moment the function stops
raising NotImplementedError: that failure is the prompt to delete the marker and
keep the assertions, which are written against the intended contract.

Signature-level tests are not marked, because they check design decisions that
already hold today and should survive implementation.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from biotp import embeddings

# Small enough to iterate with locally; see PLANNING.md for the sizing ladder.
SMALL_CHECKPOINT = "esm2_t12_35M_UR50D"
SEQUENCES = ["MKTFFVLLL", "MGSSHHHHHH", "MEEPQSDPSV"]

stub = pytest.mark.xfail(
    raises=NotImplementedError, reason="embeddings is a scaffold stub"
)


def test_embed_sequences_takes_no_embedding_dim_argument() -> None:
    """Embedding width is fixed by the checkpoint, never chosen by the caller.

    A caller wanting a different width adds a projection in the downstream head,
    so no width parameter should ever appear here.
    """
    params = set(inspect.signature(embeddings.embed_sequences).parameters)
    assert params == {"model", "sequences", "batch_size"}


def test_cached_embeddings_receives_everything_its_cache_key_needs() -> None:
    """Both the sequences and the model name must be available to key the cache.

    Without model_name in the key, switching checkpoints would silently reuse
    stale vectors, which is the one failure mode this cache must not have.
    """
    params = set(inspect.signature(embeddings.cached_embeddings).parameters)
    assert {"sequences", "model_name"} <= params


@stub
def test_load_esm2_returns_a_model() -> None:
    assert embeddings.load_esm2(SMALL_CHECKPOINT) is not None


@stub
def test_embed_sequences_returns_one_row_per_sequence() -> None:
    model = embeddings.load_esm2(SMALL_CHECKPOINT)
    out = embeddings.embed_sequences(model, SEQUENCES, batch_size=2)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[0] == len(SEQUENCES)


@stub
def test_embed_sequences_is_invariant_to_batch_size() -> None:
    """Batching is a throughput knob and must not change the vectors."""
    model = embeddings.load_esm2(SMALL_CHECKPOINT)
    one_at_a_time = embeddings.embed_sequences(model, SEQUENCES, batch_size=1)
    all_at_once = embeddings.embed_sequences(
        model, SEQUENCES, batch_size=len(SEQUENCES)
    )
    np.testing.assert_allclose(one_at_a_time, all_at_once, rtol=1e-5, atol=1e-6)


@stub
def test_cached_embeddings_writes_then_reuses_the_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "embeddings.npy"
    first = embeddings.cached_embeddings(
        SEQUENCES, SMALL_CHECKPOINT, cache_path, batch_size=2
    )
    assert cache_path.exists()

    second = embeddings.cached_embeddings(
        SEQUENCES, SMALL_CHECKPOINT, cache_path, batch_size=2
    )
    np.testing.assert_array_equal(first, second)


@stub
def test_cached_embeddings_does_not_reuse_vectors_across_models(tmp_path: Path) -> None:
    """A changed model must never silently hit a cache written by another one."""
    cache_path = tmp_path / "embeddings.npy"
    small = embeddings.cached_embeddings(
        SEQUENCES, "esm2_t6_8M_UR50D", cache_path, batch_size=2
    )
    larger = embeddings.cached_embeddings(
        SEQUENCES, SMALL_CHECKPOINT, cache_path, batch_size=2
    )
    # 8M is width 320 and 35M is width 480, so a stale hit shows up as equal widths.
    assert small.shape[1] != larger.shape[1]
