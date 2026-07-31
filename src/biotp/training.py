"""Fine-tuning harness: linear probe, LoRA, or full fine-tune behind one interface.

MVP work uses frozen embeddings + a linear/MLP head (the "linear_probe" mode);
LoRA and full fine-tune are the ramp. See PLANNING.md.

`train` covers "linear_probe" and takes precomputed features. `train_lora`
(issue #11) covers LoRA and takes sequences instead, because adapting the encoder
puts it inside the training loop and its frozen embeddings no longer exist. Full
fine-tuning still raises rather than silently doing something else, so a caller
asking for it gets an error instead of a linear probe wearing its name.

The two paths deliberately share `build_head`, so a ladder comparing them differs
only in whether the encoder was allowed to adapt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from biotp.utils import get_device

FinetuneMode = Literal["linear_probe", "lora", "full"]
Task = Literal["regression", "classification"]

# One hidden layer, wide enough to combine two concatenated embeddings without
# the capacity to memorize the training set. Not tuned; if a result turns on this
# number, that belongs in the log as a finding.
HEAD_HIDDEN_DIM = 256
HEAD_DROPOUT = 0.1

# Minibatch size for head training. The data is small enough that this affects
# optimization noise rather than memory.
BATCH_SIZE = 256

# Stop when validation loss has not improved for this many epochs. Best weights
# are restored afterwards, so a generous max_epochs costs time, not quality.
EARLY_STOPPING_PATIENCE = 10


def build_head(input_dim: int, output_dim: int, task: Task) -> Any:
    """Build a small MLP head mapping an embedding to the task output.

    input_dim is the embedding width from the encoder (fixed by the checkpoint;
    see embeddings.embed_sequences). output_dim is task-defined (1 for scalar
    regression, n_classes for classification).

    The returned module carries its own `task` attribute, so `train` picks the loss
    from the head itself and the two cannot disagree.
    """
    import torch
    from torch import nn

    assert input_dim > 0, f"input_dim must be positive, got {input_dim}"
    assert output_dim > 0, f"output_dim must be positive, got {output_dim}"
    assert task in ("regression", "classification"), f"unknown task {task!r}"

    class TaskHead(nn.Module):
        """An MLP head that remembers which task it was built for."""

        def __init__(self) -> None:
            super().__init__()
            self.task = task
            self.layers = nn.Sequential(
                nn.Linear(input_dim, HEAD_HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(HEAD_DROPOUT),
                nn.Linear(HEAD_HIDDEN_DIM, output_dim),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            output: torch.Tensor = self.layers(inputs)
            return output

    return TaskHead()


def _as_tensors(data: tuple[np.ndarray, np.ndarray], task: str, device: str) -> tuple:
    """Move one (X, y) pair onto the device with the dtype the loss expects."""
    import torch

    features, labels = data
    assert len(features) == len(labels), "features and labels have different lengths"
    assert len(features) > 0, "received an empty split"

    x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    if task == "classification":
        y = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    else:
        y = torch.as_tensor(
            np.asarray(labels), dtype=torch.float32, device=device
        ).reshape(-1, 1)
    return x, y


def train(
    model: Any,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    mode: FinetuneMode,
    max_epochs: int,
    lr: float,
) -> tuple[Any, dict]:
    """Train under the chosen mode and return the fitted model plus history.

    mode selects what is trainable: only the head (linear_probe), LoRA adapters
    (lora), or all weights (full). No default is given, forcing each call site to
    state its regime explicitly.

    Args:
        model: a head from build_head.
        train_data: (features, labels) as arrays; features are frozen embeddings.
        val_data: same shape, used only for model selection.
        mode: which regime to train under.
        max_epochs: upper bound; early stopping usually ends the run sooner.
        lr: Adam learning rate.

    Returns:
        (model, history). The returned model carries the weights from the best
        validation epoch rather than the last, so a run that overfits late is not
        reported at its worst point. history holds per-epoch losses, the best
        epoch, and realized split sizes.
    """
    import torch
    from torch import nn

    if mode == "lora":
        raise NotImplementedError(
            "LoRA does not run through `train`, which takes precomputed features: "
            "adapting the encoder means the encoder is in the loop, so the "
            "sequences are needed rather than their frozen embeddings. Use "
            "`train_lora` instead."
        )
    if mode != "linear_probe":
        raise NotImplementedError(
            f"mode {mode!r} is not implemented yet; only 'linear_probe' is. "
            f"See PLANNING.md for where full fine-tuning fits."
        )

    assert max_epochs > 0, f"max_epochs must be positive, got {max_epochs}"
    assert lr > 0, f"lr must be positive, got {lr}"
    task = getattr(model, "task", None)
    assert task in ("regression", "classification"), (
        "model has no usable `task` attribute; build it with build_head so the "
        "loss cannot disagree with the head"
    )

    device = get_device(prefer_gpu=True)
    model = model.to(device)

    x_train, y_train = _as_tensors(train_data, task, device)
    x_val, y_val = _as_tensors(val_data, task, device)

    loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "n_train": len(x_train),
        "n_val": len(x_val),
        "mode": mode,
    }
    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None

    for epoch in range(max_epochs):
        model.train()
        permutation = torch.randperm(len(x_train), device=device)
        epoch_loss = 0.0

        for start in range(0, len(x_train), BATCH_SIZE):
            batch = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)

        model.eval()
        with torch.inference_mode():
            val_loss = float(loss_fn(model(x_val), y_val))

        history["train_loss"].append(epoch_loss / len(x_train))
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= EARLY_STOPPING_PATIENCE:
            break

    assert best_state is not None, "training completed without a best epoch"
    model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    history["epochs_run"] = len(history["val_loss"])
    return model, history


def predict(model: Any, features: np.ndarray) -> np.ndarray:
    """Return predictions for frozen embeddings: class indices, or scalar values.

    Separate from `train` so evaluation can never accidentally run with dropout
    active or gradients enabled.
    """
    import torch

    assert len(features) > 0, "predict received no features"
    task = getattr(model, "task", None)
    assert task in ("regression", "classification"), "model has no usable `task`"

    device = get_device(prefer_gpu=True)
    model = model.to(device)
    model.eval()

    x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    with torch.inference_mode():
        outputs = model(x)
        if task == "classification":
            return np.asarray(outputs.argmax(dim=1).cpu().numpy())
        return np.asarray(outputs.reshape(-1).cpu().numpy())


@dataclass(frozen=True)
class VariantSplit:
    """One split's sequences, readout positions, and regression targets.

    Grouped into a type rather than passed as a bare tuple because the three move
    together and are matched by index: a silent misalignment between a sequence
    and its target trains happily and reports a plausible Spearman.

    positions is None for the mean readout and one zero-based residue index per
    sequence otherwise, mirroring `embeddings.embed_sequences`.
    """

    sequences: list[str]
    positions: list[int] | None
    targets: np.ndarray


def _check_split(split: VariantSplit, readout: str, name: str) -> None:
    """Fail loudly on a split that would train but mean nothing."""
    assert split.sequences, f"{name} split is empty"
    assert len(split.sequences) == len(split.targets), (
        f"{name} split has {len(split.sequences)} sequences and "
        f"{len(split.targets)} targets; they are matched by index"
    )
    if readout == "mean":
        assert split.positions is None, (
            f"{name} split carries positions, but the mean readout pools every "
            "residue and cannot honour them"
        )
    else:
        assert (
            split.positions is not None
        ), f"{name} split has no positions, which readout {readout!r} requires"
        assert len(split.positions) == len(split.sequences), (
            f"{name} split has {len(split.positions)} positions for "
            f"{len(split.sequences)} sequences"
        )


def _encode_batch(
    encoder: Any, sequences: list[str], positions: list[int] | None, readout: str
) -> Any:
    """Tokenize, forward, and collapse one batch, keeping gradients attached.

    Deliberately not `embeddings.embed_sequences`, which wraps its forward in
    `torch.inference_mode()`. That is right for the frozen rungs and fatal here:
    a LoRA run under inference mode produces no gradients and silently trains
    only the head, which is exactly the rung-2 result wearing rung 3's name.

    The collapse itself is imported rather than reimplemented, so both rungs read
    the same residue through the same code.
    """
    import torch

    from biotp.embeddings import _mean_pool_residues, _select_residue

    truncated = [sequence[: encoder.max_sequence_length] for sequence in sequences]
    _, _, tokens = encoder.batch_converter(
        [(str(index), sequence) for index, sequence in enumerate(truncated)]
    )
    tokens = tokens.to(encoder.device)

    result = encoder.model(tokens, repr_layers=[encoder.repr_layer])
    representations = result["representations"][encoder.repr_layer]

    if readout == "mean":
        lengths = torch.tensor(
            [len(sequence) for sequence in truncated], device=representations.device
        )
        collapsed: Any = _mean_pool_residues(representations, lengths)
    else:
        assert positions is not None  # Established by _check_split; narrows for mypy.
        collapsed = _select_residue(
            representations,
            torch.tensor(positions, device=representations.device),
        )
    return collapsed


def train_lora(
    encoder: Any,
    head: Any,
    train_data: VariantSplit,
    val_data: VariantSplit,
    readout: str,
    max_epochs: int,
    lr: float,
    batch_size: int,
    lora_rank: int,
    lora_alpha: int,
    target_modules: tuple[str, ...],
) -> tuple[Any, Any, dict]:
    """Fine-tune LoRA adapters on the encoder alongside the head.

    Rung 3 of the DMS ladder in issue #11. The head is the same `build_head`
    module rung 2 uses, and the readout is the same one, so the only difference
    between the two rungs is whether the encoder was allowed to adapt. That is
    what makes the rung-2-to-rung-3 delta attributable to adaptation rather than
    to a change of architecture.

    Args:
        encoder: an Esm2Bundle. Its `.model` is wrapped in place by peft.
        head: a head from build_head, with task "regression".
        train_data: sequences, positions, and targets to fit on.
        val_data: same shape, used only for model selection.
        readout: "mean" or "at_position", matching embeddings.Readout.
        max_epochs: upper bound; early stopping usually ends the run sooner.
        lr: Adam learning rate, applied to adapters and head together.
        batch_size: sequences per forward pass. The binding constraint is the
            attention matrix, so this is a memory knob rather than a speed one.
        lora_rank: adapter rank.
        lora_alpha: adapter scaling.
        target_modules: leaf module names to adapt. fair-esm's ESM-2 exposes
            q_proj, k_proj, v_proj and out_proj under layers.N.self_attn.

    Returns:
        (encoder, head, history). Both carry the weights from the best validation
        epoch. history holds per-epoch losses, the best epoch, split sizes, and
        the parameter counts that show the adapters actually attached.

    Raises:
        ValueError: from peft, when target_modules matches nothing. That case is
            not softened into a warning: an encoder with no adapters trains, and
            converges, and reports a number indistinguishable from rung 2.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from torch import nn

    assert max_epochs > 0, f"max_epochs must be positive, got {max_epochs}"
    assert lr > 0, f"lr must be positive, got {lr}"
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
    assert lora_rank > 0, f"lora_rank must be positive, got {lora_rank}"
    assert lora_alpha > 0, f"lora_alpha must be positive, got {lora_alpha}"
    assert target_modules, "target_modules is empty, so nothing would be adapted"

    task = getattr(head, "task", None)
    assert task == "regression", (
        f"train_lora expects a regression head, got task {task!r}; build it with "
        "build_head so the loss cannot disagree with the head"
    )
    assert readout in ("mean", "at_position"), f"unknown readout {readout!r}"

    _check_split(train_data, readout, "train")
    _check_split(val_data, readout, "val")

    # The bundle is the source of truth for where its model already lives, set by
    # load_esm2. Re-deriving it here would let the two disagree, and a model on
    # one device with its tokens on another fails deep inside the embedding
    # lookup rather than at the point of confusion.
    device = encoder.device
    encoder_parameters = sum(
        parameter.numel() for parameter in encoder.model.parameters()
    )

    encoder.model.requires_grad_(False)
    adapted = get_peft_model(
        encoder.model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules),
            bias="none",
        ),
    )
    encoder = replace(encoder, model=adapted.to(device))
    head = head.to(device)

    trainable = [
        parameter for parameter in encoder.model.parameters() if parameter.requires_grad
    ]
    trainable_encoder_parameters = sum(parameter.numel() for parameter in trainable)
    assert trainable_encoder_parameters > 0, (
        f"no encoder parameter is trainable after attaching LoRA to "
        f"{target_modules}; the adapters did not attach"
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [*trainable, *head.parameters()],
        lr=lr,
    )

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "n_train": len(train_data.sequences),
        "n_val": len(val_data.sequences),
        "mode": "lora",
        "readout": readout,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(target_modules),
        "encoder_parameters": encoder_parameters,
        "trainable_encoder_parameters": trainable_encoder_parameters,
        "lora_modules_adapted": sum(
            1 for name, _ in encoder.model.named_parameters() if "lora_" in name
        ),
    }
    best_val = float("inf")
    best_epoch = -1
    best_state: tuple[dict, dict] | None = None

    def evaluate(split: VariantSplit) -> float:
        """Mean loss over a split, with dropout off and gradients detached."""
        encoder.model.eval()
        head.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(split.sequences), batch_size):
                stop = start + batch_size
                positions = (
                    None if split.positions is None else split.positions[start:stop]
                )
                features = _encode_batch(
                    encoder, split.sequences[start:stop], positions, readout
                )
                targets = torch.as_tensor(
                    split.targets[start:stop], dtype=torch.float32, device=device
                ).reshape(-1, 1)
                total += float(loss_fn(head(features), targets)) * len(targets)
        return total / len(split.sequences)

    for epoch in range(max_epochs):
        encoder.model.train()
        head.train()
        order = np.random.default_rng(epoch).permutation(len(train_data.sequences))
        epoch_loss = 0.0

        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            sequences = [train_data.sequences[index] for index in batch]
            positions = (
                None
                if train_data.positions is None
                else [train_data.positions[index] for index in batch]
            )
            targets = torch.as_tensor(
                train_data.targets[batch], dtype=torch.float32, device=device
            ).reshape(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            features = _encode_batch(encoder, sequences, positions, readout)
            loss = loss_fn(head(features), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)

        history["train_loss"].append(epoch_loss / len(train_data.sequences))
        val_loss = evaluate(val_data)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = (
                {
                    key: value.detach().clone()
                    for key, value in encoder.model.state_dict().items()
                },
                {
                    key: value.detach().clone()
                    for key, value in head.state_dict().items()
                },
            )
        elif epoch - best_epoch >= EARLY_STOPPING_PATIENCE:
            break

    assert best_state is not None, "training completed without a best epoch"
    encoder.model.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    history["epochs_run"] = len(history["val_loss"])
    return encoder, head, history
