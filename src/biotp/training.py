"""Fine-tuning harness: linear probe, LoRA, or full fine-tune behind one interface.

MVP work uses frozen embeddings + a linear/MLP head (the "linear_probe" mode);
LoRA and full fine-tune are the ramp. See PLANNING.md.
"""

from __future__ import annotations

from typing import Literal

FinetuneMode = Literal["linear_probe", "lora", "full"]


def build_head(
    input_dim: int, output_dim: int, task: Literal["regression", "classification"]
):
    """Build a small MLP head mapping an embedding to the task output.

    input_dim is the embedding width from the encoder (fixed by the checkpoint;
    see embeddings.embed_sequences). output_dim is task-defined (1 for scalar
    regression, n_classes for classification).
    """
    raise NotImplementedError


def train(
    model,
    train_data,
    val_data,
    mode: FinetuneMode,
    max_epochs: int,
    lr: float,
):
    """Train under the chosen mode and return the fitted model plus history.

    mode selects what is trainable: only the head (linear_probe), LoRA adapters
    (lora), or all weights (full). No default is given, forcing each call site to
    state its regime explicitly.
    """
    raise NotImplementedError
