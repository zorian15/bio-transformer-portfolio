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

from biotp import training
from biotp.embeddings import Esm2Bundle

WIDTH = 16
VOCAB = 33
LIMIT = 64
BOS, PAD, EOS = 0, 1, 2
TOKENS = {aa: 4 + offset for offset, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}


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
            return {"representations": {layer: hidden for layer in repr_layers}}

    return TinyEsm()


def batch_converter(batch: list[tuple[str, str]]) -> tuple[Any, Any, Any]:
    import torch

    longest = max(len(sequence) for _, sequence in batch)
    tokens = torch.full((len(batch), longest + 2), PAD, dtype=torch.long)
    for row, (_, sequence) in enumerate(batch):
        tokens[row, 0] = BOS
        for column, residue in enumerate(sequence):
            tokens[row, column + 1] = TOKENS[residue]
        tokens[row, len(sequence) + 1] = EOS
    return [label for label, _ in batch], [text for _, text in batch], tokens


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
    residues = "ACDEFGHIKLMNPQRSTVWY"
    wildtype = "MKTFFVLLLACDEFGHIKLM"

    sequences: list[str] = []
    positions: list[int] = []
    targets: list[float] = []
    for _ in range(count):
        index = int(rng.integers(0, len(wildtype)))
        mutant = residues[int(rng.integers(0, len(residues)))]
        sequences.append(wildtype[:index] + mutant + wildtype[index + 1 :])
        positions.append(index)
        targets.append(float(TOKENS[mutant]))

    return training.VariantSplit(
        sequences=sequences,
        positions=positions,
        targets=np.asarray(targets, dtype=np.float32),
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
    }
    kwargs.update(overrides)
    return training.train_lora(**kwargs)


def test_lora_attaches_to_the_named_modules() -> None:
    """A target_modules list that matches nothing must not pass silently.

    This is the failure the whole rung-2-to-rung-3 comparison rests on: an
    encoder with no adapters attached still trains its head, still converges, and
    reports a number indistinguishable from the frozen rung.
    """
    _, _, history = run()
    assert history["trainable_encoder_parameters"] > 0
    assert history["lora_modules_adapted"] == 8, "2 layers x (q_proj, v_proj) x 2"


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
    )
    with pytest.raises(AssertionError):
        run(train_data=broken)


def test_rejects_an_empty_split() -> None:
    empty = training.VariantSplit(
        sequences=[], positions=[], targets=np.asarray([], dtype=np.float32)
    )
    with pytest.raises(AssertionError):
        run(train_data=empty)


def test_at_position_readout_requires_positions() -> None:
    without = training.VariantSplit(
        sequences=["MKT", "MKA"],
        positions=None,
        targets=np.asarray([1.0, 2.0], dtype=np.float32),
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
