"""Tests for biotp.training.train_lora.

Unlike the embedding tests, these cannot run against a duck-typed stub: `peft`
attaches by walking real `nn.Module` children and matching leaf names, so the
fixture is a genuine two-layer transformer-shaped module using fair-esm's naming
(`layers.N.self_attn.q_proj`). That naming is the thing under test as much as the
training loop is, since a `target_modules` list that matches nothing produces a
model that trains, converges, and has adapted nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import (
    AMINO_ACIDS,
    POISON,
    RESIDUE_TOKEN_IDS,
    poison_non_residues,
    tokenize_batch,
)

from biotp import training
from biotp.embeddings import Esm2Bundle
from biotp.utils import set_seed

WIDTH = 16
VOCAB = 33
LIMIT = 64

# The single reference every split in this module mutates.
WILDTYPE = "MKTFFVLLLACDEFGHIKLM"


def build_tiny_encoder() -> Any:
    """A transformer-shaped module using fair-esm's attention leaf names."""
    import torch
    from torch import nn

    class TinyAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(WIDTH, WIDTH)
            self.k_proj = nn.Linear(WIDTH, WIDTH)
            self.v_proj = nn.Linear(WIDTH, WIDTH)
            self.out_proj = nn.Linear(WIDTH, WIDTH)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            weights = torch.softmax(
                self.q_proj(hidden) @ self.k_proj(hidden).transpose(1, 2) / WIDTH**0.5,
                dim=-1,
            )
            out: torch.Tensor = self.out_proj(weights @ self.v_proj(hidden))
            return out

    class TinyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = TinyAttention()
            self.fc1 = nn.Linear(WIDTH, WIDTH)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return hidden + self.fc1(torch.relu(self.self_attn(hidden)))

    class TinyEsm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(VOCAB, WIDTH)
            self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])

        def forward(
            self, tokens: torch.Tensor, repr_layers: list[int]
        ) -> dict[str, Any]:
            hidden = self.embed_tokens(tokens)
            for layer in self.layers:
                hidden = layer(hidden)
            # Poisoned on the way out, never before the layers: the special
            # positions still serve as attention keys and values, so the module
            # trains exactly as it did unpoisoned and gradients still reach the
            # adapters through the residue rows. Unconditional on purpose. An
            # opt-out is what let this fixture and the embedding one diverge,
            # and a padding vector that looks like a residue vector is how the
            # missing position guard survived review.
            poisoned = poison_non_residues(hidden, tokens)
            return {"representations": {layer: poisoned for layer in repr_layers}}

    return TinyEsm()


def batch_converter(batch: list[tuple[str, str]]) -> tuple[Any, Any, Any]:
    labels, texts, tokens = tokenize_batch(batch, RESIDUE_TOKEN_IDS)
    return labels, texts, tokens


def tiny_bundle() -> Esm2Bundle:
    return Esm2Bundle(
        model=build_tiny_encoder(),
        alphabet=None,
        batch_converter=batch_converter,
        embedding_dim=WIDTH,
        repr_layer=2,
        max_sequence_length=LIMIT,
        device="cpu",
    )


def toy_split(count: int, seed: int) -> training.VariantSplit:
    """Variants of one wild type, with a target that is a function of the mutation.

    The target is learnable from the mutated residue alone, so a run that fails
    to reduce loss has a broken optimizer rather than an impossible task.
    """
    rng = np.random.default_rng(seed)

    sequences: list[str] = []
    positions: list[int] = []
    targets: list[float] = []
    for _ in range(count):
        index = int(rng.integers(0, len(WILDTYPE)))
        mutant = AMINO_ACIDS[int(rng.integers(0, len(AMINO_ACIDS)))]
        sequences.append(WILDTYPE[:index] + mutant + WILDTYPE[index + 1 :])
        positions.append(index)
        targets.append(float(RESIDUE_TOKEN_IDS[mutant]))

    return training.VariantSplit(
        sequences=sequences,
        positions=positions,
        targets=np.asarray(targets, dtype=np.float32),
        wildtype=None,
    )


def run(
    max_epochs: int = 3, **overrides: Any
) -> tuple[Esm2Bundle, Any, dict[str, Any]]:
    encoder = tiny_bundle()
    head = training.build_head(WIDTH, 1, "regression")
    kwargs: dict[str, Any] = {
        "encoder": encoder,
        "head": head,
        "train_data": toy_split(24, seed=0),
        "val_data": toy_split(8, seed=1),
        "readout": "at_position",
        "max_epochs": max_epochs,
        "lr": 1e-2,
        "batch_size": 8,
        "lora_rank": 4,
        "lora_alpha": 8,
        "target_modules": ("q_proj", "v_proj"),
        "seed": 0,
    }
    kwargs.update(overrides)
    return training.train_lora(**kwargs)


def ragged_split(count: int, seed: int) -> training.VariantSplit:
    """Variants of differing lengths, for the readout that pools every residue.

    The mean readout takes no positions, so the wild type's length stops being
    fixed and a batch genuinely has to be padded. The target is the mean residue
    id, which is what a correct pooling of this stub recovers up to the encoder's
    own transform, so the task stays learnable.
    """
    rng = np.random.default_rng(seed)

    sequences: list[str] = []
    targets: list[float] = []
    for _ in range(count):
        length = int(rng.integers(4, len(WILDTYPE) + 1))
        sequence = "".join(
            AMINO_ACIDS[int(rng.integers(0, len(AMINO_ACIDS)))] for _ in range(length)
        )
        sequences.append(sequence)
        targets.append(
            float(np.mean([RESIDUE_TOKEN_IDS[residue] for residue in sequence]))
        )

    return training.VariantSplit(
        sequences=sequences,
        positions=None,
        targets=np.asarray(targets, dtype=np.float32),
        wildtype=None,
    )


# --- What the fixture itself can detect ----------------------------------------


@pytest.mark.parametrize("position", [-1, len(WILDTYPE)])
def test_the_encoder_stub_makes_a_special_token_read_loud(position: int) -> None:
    """A guard on the guard: reading BOS or EOS must produce POISON, not a vector.

    `validate_positions` rejects both of these positions before training sees
    them, so this reaches `_encode_batch` directly to get past it. The point is
    not that the readout is wrong, it is that this fixture can *tell*. Before
    issue #14 it could not: an unpoisoned stub returns a plausible vector from a
    padding slot, which is how the missing position guard survived PR #13's
    review. If someone drops the poisoning, this fails.
    """
    encoder = tiny_bundle()

    out = training._encode_batch(encoder, [WILDTYPE], [position], "at_position", None)

    assert (out < POISON / 2).all()


# --- The mean readout on the LoRA path -----------------------------------------


def test_mean_readout_pools_exactly_the_residues() -> None:
    """Hand-computed against the stub's own output, so the mask is pinned exactly.

    Poisoning makes the failure unmissable rather than merely detectable: a mask
    that covered BOS, EOS or a padding slot lands near POISON instead of a few
    percent off.
    """
    import torch

    encoder = tiny_bundle()
    sequences = [WILDTYPE, WILDTYPE[:9]]

    _, _, tokens = encoder.batch_converter(
        [(str(index), sequence) for index, sequence in enumerate(sequences)]
    )
    with torch.no_grad():
        representations = encoder.model(tokens, repr_layers=[encoder.repr_layer])[
            "representations"
        ][encoder.repr_layer]
        expected = torch.stack(
            [
                representations[row, 1 : len(sequence) + 1].mean(dim=0)
                for row, sequence in enumerate(sequences)
            ]
        )
        out = training._encode_batch(encoder, sequences, None, "mean", None)

    torch.testing.assert_close(out, expected)


def test_mean_readout_trains_end_to_end_on_ragged_lengths() -> None:
    """The only path exercising the `positions is None` branch of the batch loop."""
    _, _, history = run(
        max_epochs=8,
        readout="mean",
        train_data=ragged_split(24, seed=0),
        val_data=ragged_split(8, seed=1),
    )
    assert history["readout"] == "mean"
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_mean_readout_rejects_positions() -> None:
    """Positions the readout cannot honour are an error, not something ignored."""
    with pytest.raises(AssertionError, match="cannot honour"):
        run(readout="mean", train_data=toy_split(24, seed=0))


def test_lora_attaches_to_the_named_modules() -> None:
    """A target_modules list that matches nothing must not pass silently.

    This is the failure the whole rung-2-to-rung-3 comparison rests on: an
    encoder with no adapters attached still trains its head, still converges, and
    reports a number indistinguishable from the frozen rung.
    """
    _, _, history = run()
    assert history["trainable_encoder_parameters"] > 0
    # 2 layers x 2 adapted modules x (lora_A, lora_B).
    assert history["lora_parameter_tensors"] == 8


def test_the_base_encoder_stays_frozen() -> None:
    """Exactly the adapters are trainable, and nothing else.

    A ratio threshold would be the obvious assertion and would be wrong here: the
    LoRA share of a model is a property of its scale, and on this deliberately
    tiny fixture rank-4 adapters are a sixth of the weights. On 35M they are well
    under a percent. The scale-free invariant is that the trainable set *is* the
    adapter set.
    """
    encoder, _, history = run()

    still_trainable = [
        name
        for name, parameter in encoder.model.named_parameters()
        if "lora_" not in name and parameter.requires_grad
    ]
    assert not still_trainable, f"base weights left trainable: {still_trainable[:5]}"

    adapter_parameters = sum(
        parameter.numel()
        for name, parameter in encoder.model.named_parameters()
        if "lora_" in name
    )
    assert history["trainable_encoder_parameters"] == adapter_parameters
    assert history["trainable_encoder_parameters"] < history["encoder_parameters"]


def test_base_weights_are_unchanged_by_training() -> None:
    """The invariant that makes adapter-only checkpointing correct.

    The best-epoch snapshot saves only the adapters, because the base cannot
    differ between epochs. Cloning the whole encoder instead would allocate a
    full copy on-device every time validation improved: about 140 MB at 35M and
    2.6 GB at 650M. That shortcut is valid exactly as long as the base really is
    frozen, so this pins it rather than trusting it.

    peft wraps modules in place and keeps the original Parameter object as the
    wrapped layer's base, so holding a reference across the call compares the
    same tensor rather than a same-named one.
    """
    import torch

    encoder = tiny_bundle()
    watched = encoder.model.layers[0].self_attn.q_proj.weight
    before = watched.detach().clone()

    trained, _, history = run(max_epochs=4, encoder=encoder)

    torch.testing.assert_close(watched, before)
    assert history["best_epoch"] >= 0, "nothing trained, so the check is vacuous"

    adapters = [
        parameter
        for name, parameter in trained.model.named_parameters()
        if "lora_B" in name
    ]
    assert adapters, "no adapter weights found"
    assert any(
        bool((parameter != 0).any()) for parameter in adapters
    ), "every lora_B is still zero, so training moved nothing and the base check is vacuous"


def test_training_reduces_loss() -> None:
    _, _, history = run(max_epochs=8)
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_restores_the_best_validation_epoch() -> None:
    _, _, history = run(max_epochs=6)
    assert history["best_epoch"] == int(np.argmin(history["val_loss"]))


def test_history_records_the_split_sizes_and_mode() -> None:
    _, _, history = run()
    assert history["mode"] == "lora"
    assert history["n_train"] == 24
    assert history["n_val"] == 8


def test_rejects_a_target_module_that_matches_nothing() -> None:
    """peft raises here, and that error is deliberately not softened.

    An encoder with no adapters attached still trains its head and still reports
    a plausible number, which would be rung 2's result filed under rung 3.
    """
    with pytest.raises(ValueError):
        run(target_modules=("no_such_projection",))


def test_rejects_mismatched_targets_and_sequences() -> None:
    broken = training.VariantSplit(
        sequences=["MKT", "MKA"],
        positions=[0, 1],
        targets=np.asarray([1.0], dtype=np.float32),
        wildtype=None,
    )
    with pytest.raises(AssertionError):
        run(train_data=broken)


def test_rejects_an_empty_split() -> None:
    empty = training.VariantSplit(
        sequences=[],
        positions=[],
        targets=np.asarray([], dtype=np.float32),
        wildtype=None,
    )
    with pytest.raises(AssertionError):
        run(train_data=empty)


def test_at_position_readout_requires_positions() -> None:
    without = training.VariantSplit(
        sequences=["MKT", "MKA"],
        positions=None,
        targets=np.asarray([1.0, 2.0], dtype=np.float32),
        wildtype=None,
    )
    with pytest.raises(AssertionError):
        run(train_data=without)


@pytest.mark.parametrize("rank", [0, -1])
def test_rejects_a_nonpositive_rank(rank: int) -> None:
    with pytest.raises(AssertionError):
        run(lora_rank=rank)


def test_train_points_at_train_lora_rather_than_raising_blindly() -> None:
    """The old guard said LoRA was unimplemented; it now says where it lives."""
    head = training.build_head(WIDTH, 1, "regression")
    features = np.zeros((4, WIDTH), dtype=np.float32)
    targets = np.zeros(4, dtype=np.float32)

    with pytest.raises(NotImplementedError, match="train_lora"):
        training.train(
            head,
            (features, targets),
            (features, targets),
            "lora",
            max_epochs=1,
            lr=1e-3,
        )


# --- Guards the frozen rung had and this one did not (PR #13 review) ----------


@pytest.mark.parametrize("position", [-1, 99])
def test_rejects_an_out_of_range_position(position: int) -> None:
    """The LoRA path must reject what the frozen path rejects.

    Before this guard, `_check_split` validated only the *count* of positions, so
    an out-of-range index reached `torch.gather` and came back as whatever token
    sat at that slot: a padding vector when the batch held a longer sequence, the
    BOS vector for -1. Both train, and both report a plausible Spearman.
    """
    split = training.VariantSplit(
        sequences=["MKTFF", "ACDEF"],
        positions=[position, 0],
        targets=np.asarray([1.0, 2.0], dtype=np.float32),
        wildtype=None,
    )
    with pytest.raises(AssertionError):
        run(train_data=split)


def test_rejects_a_position_lost_to_truncation() -> None:
    split = training.VariantSplit(
        sequences=["A" * (LIMIT + 10)],
        positions=[LIMIT + 5],
        targets=np.asarray([1.0], dtype=np.float32),
        wildtype=None,
    )
    with pytest.raises(AssertionError):
        run(train_data=split)


def test_rejects_an_encoder_that_already_carries_adapters() -> None:
    """peft wraps in place, so a reused bundle would stack adapter sets.

    A sweep over readout, N and seed is the obvious shape for rung 3, and peft
    only warns in this case. Every run after the first would silently be a
    continuation of its predecessor rather than an independent fit.
    """
    encoder = tiny_bundle()
    run(encoder=encoder)

    with pytest.raises(AssertionError, match="already carries LoRA adapters"):
        run(encoder=encoder)


def test_the_seed_changes_the_batch_order() -> None:
    """The ladder's seed axis has to be real on rung 3, not just on rung 2.

    The frozen rung draws its permutation from the global torch RNG, so it
    already responds to seeding. A rung 3 whose shuffle depended only on the
    epoch would report three identical runs as a three-seed spread, and the
    standard deviations in the results table would be fiction.
    """

    def losses(seed: int) -> list[float]:
        # `seed` governs the batch order only. Head initialisation and dropout
        # draw from the global torch RNG, which is the caller's to set, exactly
        # as Project 1's run_arms does it. Pinning it here isolates the one
        # source of randomness under test.
        set_seed(1234)
        return run(max_epochs=3, seed=seed)[2]["train_loss"]

    assert losses(0) == losses(0), "same seed must reproduce"
    assert losses(0) != losses(1), "different seeds must give different batch orders"


# --- Difference-at-position, the third pre-registered readout ------------------


def difference_split(count: int, seed: int) -> training.VariantSplit:
    base = toy_split(count, seed)
    return training.VariantSplit(
        sequences=base.sequences,
        positions=base.positions,
        targets=base.targets,
        wildtype=WILDTYPE,
    )


def test_difference_readout_is_zero_against_the_wild_type_itself() -> None:
    """The closed form: a variant identical to the reference differs by nothing.

    Cheap and exact, and it catches the two ways this readout goes wrong, an
    off-by-one on either the mutant or the reference row, without needing to know
    what the encoder computes.
    """
    import torch

    encoder = tiny_bundle()
    out = training._encode_batch(
        encoder, [WILDTYPE], [7], "difference_at_position", WILDTYPE
    )
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_difference_readout_trains_end_to_end() -> None:
    _, _, history = run(
        max_epochs=4,
        readout="difference_at_position",
        train_data=difference_split(24, seed=0),
        val_data=difference_split(8, seed=1),
    )
    assert history["readout"] == "difference_at_position"
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_difference_readout_requires_a_wildtype() -> None:
    with pytest.raises(AssertionError, match="no wildtype"):
        run(
            readout="difference_at_position",
            train_data=toy_split(24, seed=0),
            val_data=toy_split(8, seed=1),
        )


def test_a_wildtype_on_another_readout_is_rejected() -> None:
    """Silently ignoring it would let a caller think the difference was taken."""
    with pytest.raises(AssertionError, match="does not"):
        run(readout="at_position", train_data=difference_split(24, seed=0))


def test_rejects_a_wildtype_of_the_wrong_length() -> None:
    split = training.VariantSplit(
        sequences=["MKTFF"],
        positions=[0],
        targets=np.asarray([1.0], dtype=np.float32),
        wildtype="MKTFFVLLL",
    )
    with pytest.raises(AssertionError, match="substitutions preserve length"):
        run(readout="difference_at_position", train_data=split)


# --- predict_lora --------------------------------------------------------------


def fitted() -> tuple[Esm2Bundle, Any]:
    set_seed(7)
    encoder, head, _ = run(max_epochs=2)
    return encoder, head


def test_predict_returns_one_value_per_variant() -> None:
    encoder, head = fitted()
    split = toy_split(10, seed=3)
    out = training.predict_lora(encoder, head, split, "at_position", batch_size=4)
    assert out.shape == (10,)
    assert np.isfinite(out).all()


def test_predict_follows_the_input_order() -> None:
    """Row i of the output must be the prediction for variant i.

    Reversing the split must reverse the predictions exactly. A batching bug that
    permuted rows would still produce finite, plausible numbers and a Spearman
    that merely looked disappointing.
    """
    encoder, head = fitted()
    split = toy_split(9, seed=4)
    flipped = training.VariantSplit(
        sequences=list(reversed(split.sequences)),
        positions=list(reversed(split.positions or [])),
        targets=split.targets[::-1].copy(),
        wildtype=None,
    )

    forward = training.predict_lora(encoder, head, split, "at_position", batch_size=4)
    backward = training.predict_lora(
        encoder, head, flipped, "at_position", batch_size=4
    )
    np.testing.assert_allclose(forward, backward[::-1], rtol=1e-5)


def test_predict_is_invariant_to_batch_size() -> None:
    """Dropout must be off, so two batchings agree exactly rather than nearly."""
    encoder, head = fitted()
    split = toy_split(12, seed=5)
    one = training.predict_lora(encoder, head, split, "at_position", batch_size=1)
    many = training.predict_lora(encoder, head, split, "at_position", batch_size=12)
    np.testing.assert_allclose(one, many, rtol=1e-5)


def test_predict_supports_the_difference_readout() -> None:
    encoder, head, _ = run(
        max_epochs=2,
        readout="difference_at_position",
        train_data=difference_split(24, seed=0),
        val_data=difference_split(8, seed=1),
    )
    out = training.predict_lora(
        encoder, head, difference_split(6, seed=9), "difference_at_position", 4
    )
    assert out.shape == (6,)


def test_predict_rejects_an_empty_split() -> None:
    encoder, head = fitted()
    empty = training.VariantSplit(
        sequences=[],
        positions=[],
        targets=np.asarray([], dtype=np.float32),
        wildtype=None,
    )
    with pytest.raises(AssertionError):
        training.predict_lora(encoder, head, empty, "at_position", batch_size=4)


def test_predict_rejects_an_out_of_range_position() -> None:
    """The same guard training uses, so scoring cannot be laxer than fitting."""
    encoder, head = fitted()
    broken = training.VariantSplit(
        sequences=["MKTFF"],
        positions=[99],
        targets=np.asarray([0.0], dtype=np.float32),
        wildtype=None,
    )
    with pytest.raises(AssertionError):
        training.predict_lora(encoder, head, broken, "at_position", batch_size=4)
