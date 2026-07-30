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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
    """
    import torch

    assert sequences, "embed_sequences received no sequences"
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"

    truncated = [sequence[: model.max_sequence_length] for sequence in sequences]
    assert all(truncated), "at least one sequence was empty"

    out = np.empty((len(truncated), model.embedding_dim), dtype=np.float32)

    for start in range(0, len(truncated), batch_size):
        batch = truncated[start : start + batch_size]
        _, _, tokens = model.batch_converter(
            [(str(index), sequence) for index, sequence in enumerate(batch)]
        )
        tokens = tokens.to(model.device)

        with torch.inference_mode():
            result = model.model(tokens, repr_layers=[model.repr_layer])
        representations = result["representations"][model.repr_layer]

        for index, sequence in enumerate(batch):
            # Position 0 is BOS and position len+1 is EOS, so residues are [1, len].
            residues = representations[index, 1 : len(sequence) + 1]
            out[start + index] = residues.mean(dim=0).float().cpu().numpy()

    assert np.isfinite(out).all(), "ESM-2 produced non-finite embeddings"
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
    return out


def _cache_key(items: list[str], model_name: str) -> str:
    """Hash the inputs and the model together, so neither can change unnoticed."""
    digest = hashlib.sha256()
    digest.update(model_name.encode())
    digest.update(str(len(items)).encode())
    for item in items:
        digest.update(b"\x00")
        digest.update(item.encode())
    return digest.hexdigest()


def _load_if_key_matches(cache_path: Path, expected_key: str) -> np.ndarray | None:
    """Return cached embeddings when the stored key matches, else None."""
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as payload:
        stored_key = str(payload[CACHE_KEY_FIELD])
        if stored_key != expected_key:
            print(f"cache key changed, recomputing {cache_path}")
            return None
        return np.asarray(payload[EMBEDDINGS_FIELD])


def _write_cache(cache_path: Path, embeddings: np.ndarray, cache_key: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Annotated as Any because savez types **kwds as array-like, and the key is a str.
    payload: dict[str, Any] = {
        EMBEDDINGS_FIELD: embeddings,
        CACHE_KEY_FIELD: cache_key,
    }
    np.savez(cache_path, **payload)


def cached_embeddings(
    sequences: list[str], model_name: str, cache_path: Path, batch_size: int
) -> np.ndarray:
    """Load embeddings from cache_path, or compute and cache them if absent.

    The cache key covers both the sequences and model_name, so a changed model or
    a changed input set recomputes rather than silently returning stale vectors.
    """
    cache_key = _cache_key(sequences, model_name)
    cached = _load_if_key_matches(cache_path, cache_key)
    if cached is not None:
        return cached

    model = load_esm2(model_name)
    embeddings = embed_sequences(model, sequences, batch_size)
    _write_cache(cache_path, embeddings, cache_key)
    return embeddings


def cached_text_embeddings(
    texts: list[str], model_name: str, cache_path: Path, batch_size: int
) -> np.ndarray:
    """Text-arm counterpart to cached_embeddings, under the same cache contract."""
    cache_key = _cache_key(texts, model_name)
    cached = _load_if_key_matches(cache_path, cache_key)
    if cached is not None:
        return cached

    model = load_sentence_encoder(model_name)
    embeddings = embed_texts(model, texts, batch_size)
    _write_cache(cache_path, embeddings, cache_key)
    return embeddings
