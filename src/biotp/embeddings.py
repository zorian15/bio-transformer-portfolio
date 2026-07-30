"""Frozen embedding extraction with on-disk caching, for sequences and for text.

Frozen embeddings are the cheap backbone for every project: extract once, cache,
reuse. All head training and evaluation then runs on cached vectors, so the
expensive step happens once and anywhere (SLURM GPU or laptop) while iteration
stays fast. See PLANNING.md ("Shared infrastructure").

Two encoders live here because the grounding project needs both arms keyed the
same way: ESM-2 over amino-acid sequences, and a small sentence encoder over
UniProt annotation text.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biotp.runlog import get_logger
from biotp.utils import get_device

# ESM-2 was trained with 1024 positions, two of which the BOS and EOS tokens
# take, so 1022 residues is the longest input that fits without extrapolating
# past the training regime. Longer sequences are truncated, which matters for
# localization: signal peptides sit at the N-terminus and survive truncation.
MAX_SEQUENCE_LENGTH = 1022

# Small, fast, and widely used; 384-dimensional. Chosen for MVP iteration speed
# rather than quality, and cheap to swap since it is only a caller argument.
DEFAULT_SENTENCE_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

CACHE_KEY_FIELD = "cache_key"
EMBEDDINGS_FIELD = "embeddings"
SPEC_FIELD = "spec"

# How often the sequence loop reports progress. Tuned so a multi-minute run says
# something every few seconds without flooding the log.
PROGRESS_EVERY_N_BATCHES = 25

# Bump when a change alters the numerical output of embedding but is not visible
# in one of the named spec fields below: a different pooling rule, a different
# normalization, a bug fix that moves the vectors. Forgetting to bump means every
# existing cache silently keeps serving vectors from the old implementation.
#
# Two safeguards, because this is the one part of the cache key that relies on a
# human noticing. `test_cache_key_is_stable_for_the_recorded_spec` pins the exact
# key for a fixed spec, so any change here fails a test rather than passing
# quietly, and CLAUDE.md carries the review convention.
#
# History:
#   1: the MVP implementation, dataset-order batching and per-sequence pooling.
#   2: length-bucketed batching and on-device masked pooling (issue #3). The
#      pooling rule is unchanged in intent, but the arithmetic and the batch
#      composition both moved, and ESM-2 is only padding-invariant to about 1e-5,
#      so the vectors are not bit-identical to v1's.
EMBEDDING_IMPL_VERSION = 2


def sequence_embedding_spec(model_name: str) -> dict[str, Any]:
    """Everything about the sequence path that changes the resulting vectors.

    This is the cache key's semantic half. A field belongs here if changing it
    changes the output; `batch_size` is deliberately absent because batching is a
    throughput knob that provably does not (see the batch-invariance test), and
    including it would force needless recomputation.

    `repr_layer_policy` records the rule rather than the layer number, since the
    number is determined by the checkpoint and would require loading the model to
    read, which would defeat the point of checking the cache first.
    """
    return {
        "kind": "esm2_sequence",
        "impl_version": EMBEDDING_IMPL_VERSION,
        "model_name": model_name,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "repr_layer_policy": "final",
        "pooling": "mean_over_residues_excluding_special_tokens",
        "dtype": "float32",
    }


def text_embedding_spec(model_name: str) -> dict[str, Any]:
    """Everything about the text path that changes the resulting vectors.

    `empty_text` is a genuine semantic choice, not an implementation detail: empty
    annotations map to a zero vector rather than an encoding of the empty string,
    so changing that rule changes the vectors for every un-annotated protein.
    """
    return {
        "kind": "sentence_text",
        "impl_version": EMBEDDING_IMPL_VERSION,
        "model_name": model_name,
        "empty_text": "zero_vector",
        "dtype": "float32",
    }


@dataclass(frozen=True)
class Esm2Bundle:
    """A loaded ESM-2 checkpoint plus everything needed to batch and pool it."""

    model: Any
    alphabet: Any
    batch_converter: Any
    embedding_dim: int
    repr_layer: int
    max_sequence_length: int
    device: str


def load_esm2(model_name: str) -> Esm2Bundle:
    """Load an ESM-2 checkpoint, its alphabet, and its batch converter.

    Args:
        model_name: an ESM-2 identifier, e.g. "esm2_t12_35M_UR50D".

    Returns:
        An Esm2Bundle carrying the model in eval mode on the best available
        device, along with the embedding width and final layer index read off the
        checkpoint rather than guessed by the caller.
    """
    import esm  # Imported lazily so importing biotp stays cheap.

    assert model_name.startswith("esm2_"), f"not an ESM-2 checkpoint: {model_name!r}"
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)

    embedding_dim = getattr(model, "embed_dim", None)
    repr_layer = getattr(model, "num_layers", None)
    assert isinstance(embedding_dim, int), f"no embed_dim on {model_name!r}"
    assert isinstance(repr_layer, int), f"no num_layers on {model_name!r}"

    device = get_device(prefer_gpu=True)
    model = model.to(device)
    model.eval()

    return Esm2Bundle(
        model=model,
        alphabet=alphabet,
        batch_converter=alphabet.get_batch_converter(),
        embedding_dim=embedding_dim,
        repr_layer=repr_layer,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        device=device,
    )


def _length_bucketed_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    """Group sequence indices into batches of similar length, longest batch first.

    Batching in dataset order pads every batch to its longest member, and with a
    median length of 421 against a p90 of 995 that meant pushing 2.0x more residue
    slots through the model than the data contains, at 3.0x the attention cost
    (issue #3). Grouping similar lengths together removes almost all of it.

    Longest first, so peak memory is claimed by the first batch. An input that
    cannot fit then fails in the first seconds rather than forty minutes in.

    Returns:
        Lists of indices into `lengths`. Every index appears exactly once, so the
        caller can scatter results back into the original order.
    """
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"

    by_length = sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))
    return [
        by_length[start : start + batch_size]
        for start in range(0, len(by_length), batch_size)
    ]


def _mean_pool_residues(representations: Any, lengths: Any) -> Any:
    """Mean over residue positions for a whole batch at once, on-device.

    Args:
        representations: (batch, positions, width) activations from the model.
        lengths: (batch,) residue counts, excluding BOS, EOS, and padding.

    Position 0 is BOS and position len+1 is EOS, so residues occupy [1, len]. A
    mask built from the lengths is what keeps a short sequence in a batch of long
    ones unaffected by its neighbours; the batch-invariance and padding-isolation
    tests pin that. Pooling the batch in one masked reduction, rather than slicing
    each sequence in a Python loop, also cuts the device-to-host transfers from one
    per sequence to one per batch.
    """
    import torch

    positions = torch.arange(representations.shape[1], device=representations.device)
    is_residue = (positions.unsqueeze(0) >= 1) & (
        positions.unsqueeze(0) <= lengths.unsqueeze(1)
    )
    assert int(is_residue.sum()) == int(
        lengths.sum()
    ), "residue mask does not cover exactly the residues"

    masked = representations * is_residue.unsqueeze(-1).to(representations.dtype)
    return masked.sum(dim=1) / lengths.unsqueeze(1).to(representations.dtype)


def embed_sequences(
    model: Esm2Bundle, sequences: list[str], batch_size: int
) -> np.ndarray:
    """Return per-sequence embeddings, mean-pooled over residues.

    Args:
        model: a bundle from load_esm2.
        sequences: amino-acid strings.
        batch_size: sequences per forward pass.

    Returns:
        Array of shape (len(sequences), embedding_dim), where embedding_dim is
        fixed by the checkpoint (e.g. 320 for esm2_t6_8M, 480 for esm2_t12_35M,
        640 for esm2_t30_150M, 1280 for esm2_t33_650M), not a caller argument.
        To force a chosen output width, add a projection in the downstream head.

    Pooling covers residue positions only: the BOS and EOS tokens and any padding
    are excluded, so a short sequence in a batch of long ones is unaffected by its
    neighbours. Sequences longer than the checkpoint's limit are truncated.

    Batches are formed by length rather than in dataset order, and results are
    scattered back so row i is always the embedding of sequence i. That reordering
    is invisible to callers by construction, and pinned by
    `test_embed_sequences_preserves_input_order_under_bucketing`, because a
    permuted embedding matrix trains happily and reports plausible metrics against
    labels that belong to other proteins.
    """
    import torch

    assert sequences, "embed_sequences received no sequences"
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"

    truncated = [sequence[: model.max_sequence_length] for sequence in sequences]
    assert all(truncated), "at least one sequence was empty"
    lengths = [len(sequence) for sequence in truncated]

    out = np.empty((len(truncated), model.embedding_dim), dtype=np.float32)
    batches = _length_bucketed_batches(lengths, batch_size)

    # Work is padded residue slots, not sequences: the longest batches run first,
    # so a sequence-count rate would start out pessimistic and drift for the whole
    # run. Counting slots keeps the estimate stable from the first report.
    slots_per_batch = [
        max(lengths[index] for index in indices) * len(indices) for indices in batches
    ]
    total_slots = sum(slots_per_batch)

    log = get_logger("embeddings")
    log.info(
        "embedding %d sequences on %s in %d length-bucketed batches of up to %d "
        "(%d padded residue slots, %d residues)",
        len(truncated),
        model.device,
        len(batches),
        batch_size,
        total_slots,
        sum(lengths),
    )
    started = time.monotonic()
    done_slots = 0
    done_sequences = 0

    for batch_index, indices in enumerate(batches):
        batch = [truncated[index] for index in indices]
        _, _, tokens = model.batch_converter(
            [(str(index), sequence) for index, sequence in zip(indices, batch)]
        )
        tokens = tokens.to(model.device)

        with torch.inference_mode():
            result = model.model(tokens, repr_layers=[model.repr_layer])
        representations = result["representations"][model.repr_layer]

        batch_lengths = torch.tensor(
            [len(sequence) for sequence in batch], device=representations.device
        )
        pooled = _mean_pool_residues(representations, batch_lengths)
        # Scatter back to the caller's order, undoing the length bucketing.
        out[indices] = pooled.float().cpu().numpy()

        done_slots += slots_per_batch[batch_index]
        done_sequences += len(batch)

        # Periodic progress with an estimate, because this is the step that can run
        # for tens of minutes and silence is indistinguishable from a hang.
        done = batch_index + 1
        if done % PROGRESS_EVERY_N_BATCHES == 0 or done == len(batches):
            elapsed = time.monotonic() - started
            remaining = elapsed / done_slots * (total_slots - done_slots)
            log.info(
                "  %d/%d batches, %d/%d sequences, %.1f seq/s, ~%.0fs remaining",
                done,
                len(batches),
                done_sequences,
                len(truncated),
                done_sequences / elapsed,
                remaining,
            )

    assert np.isfinite(out).all(), "ESM-2 produced non-finite embeddings"
    log.info(
        "embedded %d sequences in %.1fs", len(truncated), time.monotonic() - started
    )
    return out


def load_sentence_encoder(model_name: str) -> Any:
    """Load a sentence-transformers model for the text arms, on the best device."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=get_device(prefer_gpu=True))


def embed_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    """Return one embedding per text, with empty strings mapped to zero vectors.

    Empty text is common and meaningful here: a protein with no curated function
    annotation genuinely has nothing to ground on. Encoding "" would hand every
    such protein an identical non-zero vector that the head could learn as a
    "missing annotation" feature, which is a confound rather than grounding, so
    those rows get an explicit zero vector instead.
    """
    assert texts, "embed_texts received no texts"
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"

    populated = [(index, text) for index, text in enumerate(texts) if text.strip()]
    # sentence-transformers renamed this accessor; accept either spelling.
    if hasattr(model, "get_embedding_dimension"):
        width = int(model.get_embedding_dimension())
    else:
        width = int(model.get_sentence_embedding_dimension())
    out = np.zeros((len(texts), width), dtype=np.float32)

    log = get_logger("embeddings")
    log.info(
        "embedding %d texts (%d non-empty, %d zero vectors)",
        len(texts),
        len(populated),
        len(texts) - len(populated),
    )
    started = time.monotonic()

    if populated:
        encoded = model.encode(
            [text for _, text in populated],
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for row, (index, _) in enumerate(populated):
            out[index] = encoded[row]

    assert np.isfinite(out).all(), "sentence encoder produced non-finite embeddings"
    log.info("embedded %d texts in %.1fs", len(texts), time.monotonic() - started)
    return out


def _cache_key(items: list[str], spec: dict[str, Any]) -> str:
    """Hash the inputs together with everything about the code that shapes them.

    Hashing the inputs alone would be enough if the transformation never changed,
    since embedding is a pure function of (items, model). It does change, and then
    the same inputs legitimately produce different vectors while an inputs-only key
    stays identical, so the cache serves the previous implementation's answer with
    no warning. Folding the spec in makes that impossible.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode())
    digest.update(str(len(items)).encode())
    for item in items:
        digest.update(b"\x00")
        digest.update(item.encode())
    return digest.hexdigest()


def _describe_cache_miss(
    stored_spec: dict[str, Any] | None, spec: dict[str, Any]
) -> str:
    """Explain why a cache was rejected, so a recompute is never mysterious."""
    if stored_spec is None:
        return "cache predates spec tracking"

    changed = [
        f"{field}: {stored_spec.get(field, '<absent>')!r} -> {value!r}"
        for field, value in sorted(spec.items())
        if stored_spec.get(field) != value
    ]
    dropped = sorted(set(stored_spec) - set(spec))
    if dropped:
        changed.append(f"fields removed: {dropped}")
    if not changed:
        return "same spec, so the inputs themselves changed"
    return "; ".join(changed)


def _load_if_key_matches(
    cache_path: Path, expected_key: str, spec: dict[str, Any]
) -> np.ndarray | None:
    """Return cached embeddings when the stored key matches, else None."""
    if not cache_path.exists():
        return None

    log = get_logger("embeddings")
    with np.load(cache_path, allow_pickle=False) as payload:
        stored_key = str(payload[CACHE_KEY_FIELD])
        if stored_key == expected_key:
            embeddings = np.asarray(payload[EMBEDDINGS_FIELD])
            log.info("cache hit: %s %s", cache_path, embeddings.shape)
            return embeddings

        stored_spec = (
            json.loads(str(payload[SPEC_FIELD])) if SPEC_FIELD in payload else None
        )
        log.info(
            "cache miss, recomputing %s (%s)",
            cache_path,
            _describe_cache_miss(stored_spec, spec),
        )
        return None


def _write_cache(
    cache_path: Path, embeddings: np.ndarray, cache_key: str, spec: dict[str, Any]
) -> None:
    """Write vectors plus the spec that produced them, so the file self-describes."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Annotated as Any because savez types **kwds as array-like, and two are strings.
    payload: dict[str, Any] = {
        EMBEDDINGS_FIELD: embeddings,
        CACHE_KEY_FIELD: cache_key,
        SPEC_FIELD: json.dumps(spec, sort_keys=True, separators=(",", ":")),
    }
    np.savez(cache_path, **payload)


def cached_embeddings(
    sequences: list[str], model_name: str, cache_path: Path, batch_size: int
) -> np.ndarray:
    """Load embeddings from cache_path, or compute and cache them if absent.

    The cache key covers the sequences, the model name, and every parameter of the
    embedding code that changes the output (see `sequence_embedding_spec`), so
    neither a changed input set nor a changed implementation can silently return
    stale vectors. `batch_size` is excluded on purpose: it does not affect output.
    """
    spec = sequence_embedding_spec(model_name)
    cache_key = _cache_key(sequences, spec)
    cached = _load_if_key_matches(cache_path, cache_key, spec)
    if cached is not None:
        return cached

    model = load_esm2(model_name)
    embeddings = embed_sequences(model, sequences, batch_size)
    _write_cache(cache_path, embeddings, cache_key, spec)
    return embeddings


def cached_text_embeddings(
    texts: list[str], model_name: str, cache_path: Path, batch_size: int
) -> np.ndarray:
    """Text-arm counterpart to cached_embeddings, under the same cache contract."""
    spec = text_embedding_spec(model_name)
    cache_key = _cache_key(texts, spec)
    cached = _load_if_key_matches(cache_path, cache_key, spec)
    if cached is not None:
        return cached

    model = load_sentence_encoder(model_name)
    embeddings = embed_texts(model, texts, batch_size)
    _write_cache(cache_path, embeddings, cache_key, spec)
    return embeddings
