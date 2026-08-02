"""Pieces shared by the encoder stubs in more than one test module.

Three test modules stand in for ESM-2: `test_embeddings.py`, `test_train_lora.py`
and `test_zero_shot.py`. Each needs the same tokenization arithmetic, and the
copies had drifted: only the embeddings stub poisoned its non-residue positions,
so a special-token read was loud on the frozen path and invisible on the LoRA
path. The missing position guard fixed in issue #11 survived partly because of
that asymmetry. What lives here is what all three must agree on; what stays in
each module is what genuinely differs.

Two conventions worth stating, because both look like omissions:

This file holds no pytest fixtures. The suite's convention is plain module-level
helpers called directly inside a test, and converting these to fixtures would
churn every call site without making anything clearer.

Test modules reach these by name, as `from conftest import tokenize_batch`. That
resolves because `tests/` has no `__init__.py`, so pytest's prepend import mode
puts `tests/` on `sys.path` and imports this file as the top-level module
`conftest`, and mypy computes the same module name for the same reason. A second
`conftest.py` anywhere under `tests/` would collide on that basename and break
both; put shared code here rather than in a sibling conftest.
"""

from __future__ import annotations

from typing import Any

# The 20 standard residues, in the conventional order. Deliberately not imported
# from `biotp.zero_shot`, which holds its own copy as a frozenset: the zero-shot
# tests build their stub alphabet from this string and then exercise
# `parse_substitutions`, which validates against the module's copy. Sharing one
# constant would make an alphabet typo in the module invisible to the test whose
# job is to catch it.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Special tokens, matching the layout of a real ESM-2 alphabet: BOS at position
# 0, EOS after the last residue, PAD filling the rest of the batch.
BOS = 0
PAD = 1
EOS = 2

# Written into every non-residue slot of a stub's output. Large enough that a
# pooled or gathered vector which touched one is obviously wrong rather than
# subtly wrong, and far enough from any real activation that a test can assert
# on the sign alone.
POISON = -1e6

# Residue ids starting above the special tokens, as in a real ESM-2 alphabet.
RESIDUE_TOKEN_IDS = {residue: 4 + offset for offset, residue in enumerate(AMINO_ACIDS)}

# The ids `test_embeddings.py` uses, where a residue's token id is its ASCII
# code. That module's expectations are hand-computed in terms of `ord`, so the
# mapping has to stay ASCII; going through a dict rather than calling `ord`
# turns a non-standard residue into a loud KeyError instead of a plausible
# number.
ASCII_TOKEN_IDS = {residue: ord(residue) for residue in AMINO_ACIDS}


def tokenize_batch(batch: list[tuple[str, str]], token_ids: dict[str, int]) -> Any:
    """Tokenize one (label, sequence) batch the way fair-esm's converter does.

    Args:
        batch: (label, sequence) pairs, as `esm.Alphabet.get_batch_converter`
            takes them.
        token_ids: residue to token id. Parameterised rather than fixed because
            the embedding tests hand-compute their expectations from ASCII codes
            while the other two modules use ESM-2-like ids.

    Returns:
        (labels, sequences, tokens), matching fair-esm's three-tuple. tokens has
        shape (batch, longest + 2) and is padded to the longest member.

    The layout is the load-bearing part: BOS at token 0, residue i at token
    i + 1, EOS at token len + 1, PAD after that. Every readout in the codebase
    offsets by that same +1, so a stub that got it wrong here would agree with a
    reader that got it wrong there and the pair would look correct.
    """
    import torch

    assert batch, "tokenize_batch received an empty batch"
    longest = max(len(sequence) for _, sequence in batch)
    tokens = torch.full((len(batch), longest + 2), PAD, dtype=torch.long)
    for row, (_, sequence) in enumerate(batch):
        tokens[row, 0] = BOS
        for column, residue in enumerate(sequence):
            assert (
                residue in token_ids
            ), f"no token id for residue {residue!r} in sequence {sequence!r}"
            tokens[row, column + 1] = token_ids[residue]
        tokens[row, len(sequence) + 1] = EOS
    return [label for label, _ in batch], [sequence for _, sequence in batch], tokens


def poison_non_residues(representations: Any, tokens: Any) -> Any:
    """Overwrite every BOS, EOS and PAD slot of a stub's output with POISON.

    Args:
        representations: (batch, positions, width) activations.
        tokens: (batch, positions) token ids from `tokenize_batch`.

    A padding vector and a residue vector coming out of an unpoisoned stub look
    equally plausible, which is how a readout that reads the wrong slot passes
    its tests. Poisoning makes that read produce a number no assertion can
    mistake for a result.

    Applied to a model's output, never to its hidden states: the special
    positions still participate as attention keys and values, so a real module
    trains normally and gradients still reach its parameters through the residue
    rows.
    """
    import torch

    is_special = ((tokens == BOS) | (tokens == PAD) | (tokens == EOS)).unsqueeze(-1)
    return torch.where(
        is_special, torch.full_like(representations, POISON), representations
    )
