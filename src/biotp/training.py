"""Fine-tuning harness: linear probe, LoRA, or full fine-tune behind one interface.

MVP work uses frozen embeddings + a linear/MLP head (the "linear_probe" mode);
LoRA and full fine-tune are the ramp. See PLANNING.md.

Only "linear_probe" is implemented. The other modes raise rather than silently
doing something else, so a caller asking for LoRA gets an error instead of a
linear probe wearing LoRA's name.
"""

from __future__ import annotations

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

    if mode != "linear_probe":
        raise NotImplementedError(
            f"mode {mode!r} is not implemented yet; only 'linear_probe' is. "
            f"See PLANNING.md for where LoRA and full fine-tuning fit."
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
