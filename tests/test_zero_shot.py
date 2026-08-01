"""Tests for biotp.zero_shot.

The stub masked LM makes every score computable in closed form. Its logits are
`vocab_index * (1 + token_position)`, and a masked-marginal score is a difference
of log-softmax entries at one position, where the log-partition term cancels:

    log p(mut) - log p(wt) = (mut_index - wt_index) * (1 + token_position)

So an expected score is arithmetic, and the position factor means an off-by-one
in the BOS offset changes the answer rather than hiding. That matters because the
alternative, checking the scores are merely "reasonable", would pass for a
function that scored the residue next door.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import RESIDUE_TOKEN_IDS, tokenize_batch

from biotp import zero_shot

STUB_MASK_IDX = 32
STUB_VOCAB = 33
STUB_LIMIT = 64

WILDTYPE = "MKTFFVLLLACDE"


class StubAlphabet:
    """Minimal stand-in for esm.data.Alphabet.

    Only the mask token and the residue lookup, which is all `masked_marginal_
    scores` reads off an alphabet. The special-token ids live in conftest,
    because the tokenizer that writes them is shared.
    """

    def __init__(self) -> None:
        self.mask_idx = STUB_MASK_IDX

    def get_idx(self, token: str) -> int:
        assert token in RESIDUE_TOKEN_IDS, f"stub alphabet has no token {token!r}"
        return RESIDUE_TOKEN_IDS[token]


class StubMaskedLM:
    """A masked LM whose logits are a closed-form function of position and vocab.

    Records the token positions it saw masked, so a test can assert that scoring
    masked exactly the mutated positions and no others.
    """

    def __init__(self) -> None:
        self.masked_token_positions: list[int] = []
        self.batches: list[int] = []

    def batch_converter(self, batch: list[tuple[str, str]]) -> tuple[Any, Any, Any]:
        # Recorded here rather than in the shared helper: only this module
        # asserts on how scoring batched its work.
        self.batches.append(len(batch))
        labels, texts, tokens = tokenize_batch(batch, RESIDUE_TOKEN_IDS)
        return labels, texts, tokens

    def __call__(self, tokens: Any, repr_layers: list[int]) -> dict[str, Any]:
        import torch

        for row in range(tokens.shape[0]):
            masked = (tokens[row] == STUB_MASK_IDX).nonzero().flatten().tolist()
            self.masked_token_positions.extend(int(index) for index in masked)

        positions = torch.arange(tokens.shape[1], dtype=torch.float32)
        vocab = torch.arange(STUB_VOCAB, dtype=torch.float32)
        logits = vocab.view(1, 1, -1) * (1.0 + positions).view(1, -1, 1)
        return {"logits": logits.expand(tokens.shape[0], -1, -1).contiguous()}


def stub_bundle() -> Any:
    """Wrap StubMaskedLM in the bundle zero_shot expects."""
    from biotp.embeddings import Esm2Bundle

    model = StubMaskedLM()
    return Esm2Bundle(
        model=model,
        alphabet=StubAlphabet(),
        batch_converter=model.batch_converter,
        embedding_dim=8,
        repr_layer=1,
        max_sequence_length=STUB_LIMIT,
        device="cpu",
    )


def expected_score(substitutions: list[tuple[int, str, str]]) -> float:
    """The closed form the stub implies, summed over a variant's substitutions."""
    total = 0.0
    for position, wildtype_aa, mutant_aa in substitutions:
        gap = RESIDUE_TOKEN_IDS[mutant_aa] - RESIDUE_TOKEN_IDS[wildtype_aa]
        total += gap * (1.0 + (position + 1))
    return total


# --- Parsing -----------------------------------------------------------------


def test_parses_a_single_substitution_to_zero_based() -> None:
    assert zero_shot.parse_substitutions("A24G", one_based=True) == [(23, "A", "G")]


def test_parses_a_multiple_substitution() -> None:
    assert zero_shot.parse_substitutions("A24G:L56P", one_based=True) == [
        (23, "A", "G"),
        (55, "L", "P"),
    ]


def test_honours_zero_based_input() -> None:
    """ProteinGym numbers from one; a caller with zero-based data says so."""
    assert zero_shot.parse_substitutions("A24G", one_based=False) == [(24, "A", "G")]


def test_rejects_a_malformed_mutant() -> None:
    for mutant in ["", "24", "AG", "A-1G", "AXG", "24AG"]:
        with pytest.raises(AssertionError):
            zero_shot.parse_substitutions(mutant, one_based=True)


def test_rejects_a_one_based_position_of_zero() -> None:
    """Position 0 under one-based numbering would silently become -1."""
    with pytest.raises(AssertionError):
        zero_shot.parse_substitutions("A0G", one_based=True)


# --- Scoring -----------------------------------------------------------------


def test_scores_match_the_closed_form() -> None:
    bundle = stub_bundle()
    variants = [
        [(0, "M", "A")],
        [(3, "F", "W")],
        [(2, "T", "C"), (9, "A", "Y")],
    ]

    scores = zero_shot.masked_marginal_scores(bundle, WILDTYPE, variants, batch_size=4)

    np.testing.assert_allclose(
        scores, [expected_score(variant) for variant in variants], rtol=1e-5
    )


def test_a_synonymous_substitution_scores_zero() -> None:
    """Mutating a residue to itself is a no-op, and the log-ratio says so."""
    bundle = stub_bundle()
    scores = zero_shot.masked_marginal_scores(
        bundle, WILDTYPE, [[(0, "M", "M")]], batch_size=1
    )
    np.testing.assert_allclose(scores, [0.0], atol=1e-6)


def test_masks_each_mutated_position_exactly_once() -> None:
    """The cost is one forward per distinct position, not one per variant.

    Two variants at the same site must share a forward pass; the whole reason
    masked-marginals is affordable at 650M is that it scales with the protein,
    not with the size of the assay.
    """
    bundle = stub_bundle()
    variants = [[(3, "F", "W")], [(3, "F", "A")], [(0, "M", "C")], [(3, "F", "Y")]]

    zero_shot.masked_marginal_scores(bundle, WILDTYPE, variants, batch_size=8)

    masked = sorted(bundle.model.masked_token_positions)
    assert masked == [1, 4], "expected one mask per distinct position, at BOS offset"


def test_scores_are_invariant_to_batch_size() -> None:
    bundle_small = stub_bundle()
    bundle_large = stub_bundle()
    variants = [[(index, WILDTYPE[index], "W")] for index in range(len(WILDTYPE))]

    one_at_a_time = zero_shot.masked_marginal_scores(
        bundle_small, WILDTYPE, variants, batch_size=1
    )
    all_at_once = zero_shot.masked_marginal_scores(
        bundle_large, WILDTYPE, variants, batch_size=len(variants)
    )

    np.testing.assert_allclose(one_at_a_time, all_at_once, rtol=1e-5)


def test_rejects_a_wildtype_that_disagrees_with_the_sequence() -> None:
    """The commonest real failure: an off-by-one between assay and reference.

    ProteinGym positions index the assay's target sequence. Pairing them with a
    UniProt sequence that has an extra N-terminal methionine shifts everything by
    one, and every score stays finite and plausible. The variant string carries
    the wild-type residue precisely so this is checkable, so it is checked.
    """
    bundle = stub_bundle()
    with pytest.raises(AssertionError):
        zero_shot.masked_marginal_scores(
            bundle, WILDTYPE, [[(0, "K", "A")]], batch_size=1
        )


def test_rejects_a_position_past_the_context_limit() -> None:
    bundle = stub_bundle()
    overlong = "A" * (STUB_LIMIT + 10)
    with pytest.raises(AssertionError):
        zero_shot.masked_marginal_scores(
            bundle, overlong, [[(STUB_LIMIT + 5, "A", "G")]], batch_size=1
        )


def test_rejects_an_empty_variant() -> None:
    """A variant with no substitutions is the wild type, not a score of zero."""
    bundle = stub_bundle()
    with pytest.raises(AssertionError):
        zero_shot.masked_marginal_scores(bundle, WILDTYPE, [[]], batch_size=1)


def test_rejects_no_variants() -> None:
    bundle = stub_bundle()
    with pytest.raises(AssertionError):
        zero_shot.masked_marginal_scores(bundle, WILDTYPE, [], batch_size=1)
