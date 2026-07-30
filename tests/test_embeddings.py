"""Tests for biotp.embeddings.

Tests that load real checkpoints are marked `network` and deselected by default
(see pyproject.toml), since they download ESM-2 weights; run them with
`pytest -m network`. Everything else runs offline against stub encoders, which is
what lets the cache contract be tested properly: the interesting behavior is
*when a recompute is triggered*, and that needs a counter, not a real model.

Signature-level tests pin design decisions that should outlive any rewrite.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from biotp import embeddings

SMALL_CHECKPOINT = "esm2_t12_35M_UR50D"
TINY_CHECKPOINT = "esm2_t6_8M_UR50D"
SEQUENCES = ["MKTFFVLLL", "MGSSHHHHHH", "MEEPQSDPSV"]

# Widths are properties of the checkpoints, not of this test file.
EXPECTED_WIDTHS = {SMALL_CHECKPOINT: 480, TINY_CHECKPOINT: 320}


class StubSentenceEncoder:
    """Minimal stand-in for a sentence-transformers model."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.encoded: list[str] = []

    def get_embedding_dimension(self) -> int:
        return self.width

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        self.encoded.extend(texts)
        # Distinct, deterministic, non-zero rows so zero means "not encoded".
        return np.array(
            [[float(len(text))] * self.width for text in texts], dtype=np.float32
        )


def test_embed_sequences_takes_no_embedding_dim_argument() -> None:
    """Embedding width is fixed by the checkpoint, never chosen by the caller.

    A caller wanting a different width adds a projection in the downstream head,
    so no width parameter should ever appear here.
    """
    params = set(inspect.signature(embeddings.embed_sequences).parameters)
    assert params == {"model", "sequences", "batch_size"}


def test_cached_embeddings_receives_everything_its_cache_key_needs() -> None:
    params = set(inspect.signature(embeddings.cached_embeddings).parameters)
    assert {"sequences", "model_name"} <= params


def test_truncation_limit_leaves_room_for_bos_and_eos() -> None:
    """ESM-2 was trained with 1024 positions, two of which are special tokens."""
    assert embeddings.MAX_SEQUENCE_LENGTH == 1022


def spec_for(model_name: str = "model-a") -> dict:
    return embeddings.sequence_embedding_spec(model_name)


def test_cache_key_changes_with_the_model() -> None:
    """The one failure mode this cache must not have: stale vectors after a swap."""
    assert embeddings._cache_key(
        SEQUENCES, spec_for("model-a")
    ) != embeddings._cache_key(SEQUENCES, spec_for("model-b"))


def test_cache_key_changes_with_the_inputs() -> None:
    assert embeddings._cache_key(SEQUENCES, spec_for()) != embeddings._cache_key(
        [*SEQUENCES, "MKV"], spec_for()
    )


def test_cache_key_changes_with_input_order() -> None:
    """Row i of the output must correspond to item i of the input."""
    assert embeddings._cache_key(SEQUENCES, spec_for()) != embeddings._cache_key(
        list(reversed(SEQUENCES)), spec_for()
    )


def test_cache_key_is_stable_across_calls() -> None:
    assert embeddings._cache_key(SEQUENCES, spec_for()) == embeddings._cache_key(
        SEQUENCES, spec_for()
    )


def test_cache_key_is_not_fooled_by_concatenation() -> None:
    """["ab", "c"] and ["a", "bc"] are different inputs and must key differently."""
    assert embeddings._cache_key(["ab", "c"], spec_for()) != embeddings._cache_key(
        ["a", "bc"], spec_for()
    )


@pytest.fixture
def counting_embedder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the ESM-2 path with a fake, recording each model actually loaded."""
    loaded: list[str] = []

    def fake_load_esm2(model_name: str) -> Any:
        loaded.append(model_name)
        return model_name

    def fake_embed_sequences(
        model: Any, sequences: list[str], batch_size: int
    ) -> np.ndarray:
        width = EXPECTED_WIDTHS.get(str(model), 8)
        return np.ones((len(sequences), width), dtype=np.float32)

    monkeypatch.setattr(embeddings, "load_esm2", fake_load_esm2)
    monkeypatch.setattr(embeddings, "embed_sequences", fake_embed_sequences)
    return loaded


def test_cached_embeddings_computes_then_reuses(
    tmp_path: Path, counting_embedder: list[str]
) -> None:
    cache_path = tmp_path / "seq.npz"

    first = embeddings.cached_embeddings(SEQUENCES, SMALL_CHECKPOINT, cache_path, 2)
    assert cache_path.exists()
    assert counting_embedder == [SMALL_CHECKPOINT]

    second = embeddings.cached_embeddings(SEQUENCES, SMALL_CHECKPOINT, cache_path, 2)
    np.testing.assert_array_equal(first, second)
    assert counting_embedder == [SMALL_CHECKPOINT], "cache hit should not recompute"


def test_cached_embeddings_recomputes_when_the_model_changes(
    tmp_path: Path, counting_embedder: list[str]
) -> None:
    """A changed model must never silently hit a cache written by another one."""
    cache_path = tmp_path / "seq.npz"
    small = embeddings.cached_embeddings(SEQUENCES, SMALL_CHECKPOINT, cache_path, 2)
    tiny = embeddings.cached_embeddings(SEQUENCES, TINY_CHECKPOINT, cache_path, 2)

    assert counting_embedder == [SMALL_CHECKPOINT, TINY_CHECKPOINT]
    assert small.shape[1] != tiny.shape[1]


def test_cached_embeddings_recomputes_when_the_inputs_change(
    tmp_path: Path, counting_embedder: list[str]
) -> None:
    cache_path = tmp_path / "seq.npz"
    before = embeddings.cached_embeddings(SEQUENCES, SMALL_CHECKPOINT, cache_path, 2)
    after = embeddings.cached_embeddings(
        [*SEQUENCES, "MKV"], SMALL_CHECKPOINT, cache_path, 2
    )
    assert after.shape[0] == before.shape[0] + 1
    assert len(counting_embedder) == 2


def test_embed_texts_maps_empty_text_to_a_zero_vector() -> None:
    """Empty annotation is meaningful, and encoding "" would be a confound.

    Every un-annotated protein would otherwise share one identical non-zero
    vector that a head can learn as a "missing annotation" flag.
    """
    encoder = StubSentenceEncoder(width=4)
    out = embeddings.embed_texts(encoder, ["FUNCTION: kinase.", "", "   "], 2)

    assert out.shape == (3, 4)
    assert out[0].any()
    assert not out[1].any()
    assert not out[2].any()
    assert encoder.encoded == ["FUNCTION: kinase."], "blank text should not be encoded"


def test_embed_texts_preserves_row_order() -> None:
    encoder = StubSentenceEncoder(width=3)
    out = embeddings.embed_texts(encoder, ["", "ab", "", "abcd"], 2)

    assert not out[0].any()
    assert out[1][0] == pytest.approx(2.0)
    assert not out[2].any()
    assert out[3][0] == pytest.approx(4.0)


def test_embed_texts_handles_all_text_being_empty() -> None:
    """No populated rows means no encoder call at all, not a crash."""
    encoder = StubSentenceEncoder(width=5)
    out = embeddings.embed_texts(encoder, ["", ""], 2)

    assert out.shape == (2, 5)
    assert not out.any()
    assert encoder.encoded == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_embed_texts_rejects_nonpositive_batch_size(batch_size: int) -> None:
    with pytest.raises(AssertionError, match="batch_size"):
        embeddings.embed_texts(StubSentenceEncoder(width=2), ["a"], batch_size)


def test_embed_texts_rejects_empty_input() -> None:
    with pytest.raises(AssertionError, match="no texts"):
        embeddings.embed_texts(StubSentenceEncoder(width=2), [], 2)


def test_cached_text_embeddings_reuses_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "text.npz"
    loaded: list[str] = []

    def fake_loader(model_name: str) -> StubSentenceEncoder:
        loaded.append(model_name)
        return StubSentenceEncoder(width=6)

    monkeypatch.setattr(embeddings, "load_sentence_encoder", fake_loader)

    texts = ["FUNCTION: one.", ""]
    first = embeddings.cached_text_embeddings(texts, "stub-encoder", cache_path, 2)
    second = embeddings.cached_text_embeddings(texts, "stub-encoder", cache_path, 2)

    np.testing.assert_array_equal(first, second)
    assert loaded == ["stub-encoder"], "cache hit should not reload the encoder"


@pytest.mark.network
def test_load_esm2_reports_width_and_layer_from_the_checkpoint() -> None:
    bundle = embeddings.load_esm2(SMALL_CHECKPOINT)
    assert bundle.embedding_dim == EXPECTED_WIDTHS[SMALL_CHECKPOINT]
    assert bundle.repr_layer == 12
    assert bundle.device in {"cuda", "mps", "cpu"}


@pytest.mark.network
def test_load_esm2_rejects_a_non_esm2_name() -> None:
    with pytest.raises(AssertionError, match="not an ESM-2 checkpoint"):
        embeddings.load_esm2("bert-base-uncased")


@pytest.mark.network
def test_embed_sequences_returns_one_row_per_sequence() -> None:
    bundle = embeddings.load_esm2(SMALL_CHECKPOINT)
    out = embeddings.embed_sequences(bundle, SEQUENCES, batch_size=2)

    assert out.shape == (len(SEQUENCES), EXPECTED_WIDTHS[SMALL_CHECKPOINT])
    assert np.isfinite(out).all()


@pytest.mark.network
def test_embed_sequences_is_invariant_to_batch_size() -> None:
    """Batching is a throughput knob and must not change the vectors."""
    bundle = embeddings.load_esm2(SMALL_CHECKPOINT)
    one_at_a_time = embeddings.embed_sequences(bundle, SEQUENCES, batch_size=1)
    all_at_once = embeddings.embed_sequences(
        bundle, SEQUENCES, batch_size=len(SEQUENCES)
    )
    np.testing.assert_allclose(one_at_a_time, all_at_once, rtol=1e-4, atol=1e-5)


@pytest.mark.network
def test_embed_sequences_excludes_padding_from_pooling() -> None:
    """A short sequence must embed the same alone as beside a long one."""
    bundle = embeddings.load_esm2(SMALL_CHECKPOINT)
    alone = embeddings.embed_sequences(bundle, [SEQUENCES[0]], batch_size=1)
    padded = embeddings.embed_sequences(bundle, SEQUENCES, batch_size=len(SEQUENCES))
    np.testing.assert_allclose(alone[0], padded[0], rtol=1e-4, atol=1e-5)


@pytest.mark.network
def test_embed_sequences_truncates_overlong_sequences() -> None:
    """Past the position limit the call must truncate rather than fail."""
    bundle = embeddings.load_esm2(TINY_CHECKPOINT)
    long_sequence = "M" + "A" * (embeddings.MAX_SEQUENCE_LENGTH + 500)
    out = embeddings.embed_sequences(bundle, [long_sequence], batch_size=1)
    assert out.shape == (1, EXPECTED_WIDTHS[TINY_CHECKPOINT])
    assert np.isfinite(out).all()


# --- Cache key covers the embedding code, not only its inputs (issue #4) -------
#
# The bug these pin down: the key used to hash only (sequences, model_name), so
# editing the embedding code left every cache file looking valid and the cache
# kept serving vectors from the old implementation. Every test below fails
# against that earlier key.


def test_cache_key_changes_with_the_truncation_limit() -> None:
    """The exact case demonstrated in issue #4: 0.48 divergence, silently served."""
    spec = embeddings.sequence_embedding_spec(SMALL_CHECKPOINT)
    longer = {**spec, "max_sequence_length": spec["max_sequence_length"] + 1}
    assert embeddings._cache_key(SEQUENCES, spec) != embeddings._cache_key(
        SEQUENCES, longer
    )


def test_cache_key_changes_with_the_implementation_version() -> None:
    spec = embeddings.sequence_embedding_spec(SMALL_CHECKPOINT)
    bumped = {**spec, "impl_version": spec["impl_version"] + 1}
    assert embeddings._cache_key(SEQUENCES, spec) != embeddings._cache_key(
        SEQUENCES, bumped
    )


@pytest.mark.parametrize("field", ["pooling", "repr_layer_policy", "dtype"])
def test_cache_key_changes_with_any_semantic_field(field: str) -> None:
    spec = embeddings.sequence_embedding_spec(SMALL_CHECKPOINT)
    altered = {**spec, field: "something-else"}
    assert embeddings._cache_key(SEQUENCES, spec) != embeddings._cache_key(
        SEQUENCES, altered
    )


def test_text_and_sequence_specs_never_collide() -> None:
    """Same model name, different modality, must not share a cache entry."""
    assert embeddings._cache_key(
        SEQUENCES, embeddings.sequence_embedding_spec("m")
    ) != embeddings._cache_key(SEQUENCES, embeddings.text_embedding_spec("m"))


def test_text_spec_records_the_empty_text_rule() -> None:
    """Mapping empty text to a zero vector is a semantic choice, so it is keyed."""
    spec = embeddings.text_embedding_spec(embeddings.DEFAULT_SENTENCE_ENCODER)
    assert spec["empty_text"] == "zero_vector"
    other = {**spec, "empty_text": "encode_empty_string"}
    assert embeddings._cache_key(["a"], spec) != embeddings._cache_key(["a"], other)


def test_changing_the_truncation_limit_forces_a_recompute(
    tmp_path: Path, counting_embedder: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end reproduction of issue #4, at the cached_embeddings level.

    Before the fix this returned the first call's vectors with no warning, because
    the key saw identical sequences and an identical model name.
    """
    cache_path = tmp_path / "seq.npz"

    monkeypatch.setattr(embeddings, "MAX_SEQUENCE_LENGTH", 100)
    embeddings.cached_embeddings(SEQUENCES, SMALL_CHECKPOINT, cache_path, 2)
    assert counting_embedder == [SMALL_CHECKPOINT]

    monkeypatch.setattr(embeddings, "MAX_SEQUENCE_LENGTH", 200)
    embeddings.cached_embeddings(SEQUENCES, SMALL_CHECKPOINT, cache_path, 2)
    assert (
        counting_embedder == [SMALL_CHECKPOINT] * 2
    ), "changing the truncation limit must recompute, not serve stale vectors"


def test_changing_batch_size_alone_still_hits_the_cache(
    tmp_path: Path, counting_embedder: list[str]
) -> None:
    """The deliberate exclusion: batching is a throughput knob, not a semantic one.

    Pinned so nobody "fixes" the key later by adding batch_size, which would force
    a 100-minute recompute for no correctness gain.
    """
    cache_path = tmp_path / "seq.npz"
    embeddings.cached_embeddings(SEQUENCES, SMALL_CHECKPOINT, cache_path, 2)
    embeddings.cached_embeddings(SEQUENCES, SMALL_CHECKPOINT, cache_path, 8)
    assert counting_embedder == [SMALL_CHECKPOINT], "batch_size must not be keyed"


def test_cache_file_records_the_spec_that_produced_it(
    tmp_path: Path, counting_embedder: list[str]
) -> None:
    """A cache file should explain itself rather than hold an opaque hash."""
    cache_path = tmp_path / "seq.npz"
    embeddings.cached_embeddings(SEQUENCES, SMALL_CHECKPOINT, cache_path, 2)

    with np.load(cache_path, allow_pickle=False) as payload:
        assert embeddings.SPEC_FIELD in payload
        stored = json.loads(str(payload[embeddings.SPEC_FIELD]))
    assert stored == embeddings.sequence_embedding_spec(SMALL_CHECKPOINT)


def test_cache_miss_reports_which_field_changed(
    tmp_path: Path,
    counting_embedder: list[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A recompute should never be mysterious."""
    cache_path = tmp_path / "seq.npz"
    monkeypatch.setattr(embeddings, "MAX_SEQUENCE_LENGTH", 100)
    embeddings.cached_embeddings(SEQUENCES, SMALL_CHECKPOINT, cache_path, 2)

    monkeypatch.setattr(embeddings, "MAX_SEQUENCE_LENGTH", 200)
    with caplog.at_level("INFO"):
        embeddings.cached_embeddings(SEQUENCES, SMALL_CHECKPOINT, cache_path, 2)

    assert "max_sequence_length" in caplog.text
    assert "100" in caplog.text and "200" in caplog.text


def test_cache_miss_distinguishes_changed_inputs_from_changed_code() -> None:
    spec = embeddings.sequence_embedding_spec(SMALL_CHECKPOINT)
    assert "inputs themselves changed" in embeddings._describe_cache_miss(spec, spec)
    assert "predates spec tracking" in embeddings._describe_cache_miss(None, spec)


# Golden key. This deliberately fails whenever anything in the spec changes,
# including an EMBEDDING_IMPL_VERSION bump. That is the point: the version
# constant is the one part of the key that depends on a human remembering, so a
# change to it must show up as a test to update rather than passing silently.
# See the CLAUDE.md convention. If this test fails, confirm the spec change was
# intended, then update the expected key here in the same commit.
GOLDEN_SPEC_KEY = "8af2a753c16252d8cf7dfd41e7f19b4c9b01953aae290e56ee81ef24bd660857"
GOLDEN_ITEMS = ["MKTFFVLLL", "MGSSHHHHHH"]


def test_cache_key_is_stable_for_the_recorded_spec() -> None:
    spec = embeddings.sequence_embedding_spec("esm2_t12_35M_UR50D")
    assert embeddings._cache_key(GOLDEN_ITEMS, spec) == GOLDEN_SPEC_KEY, (
        "the embedding spec changed. If that was intended, bump "
        "EMBEDDING_IMPL_VERSION where required and update GOLDEN_SPEC_KEY here."
    )
