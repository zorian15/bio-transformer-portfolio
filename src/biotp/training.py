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

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Generic, Literal, TypeVar

import numpy as np

from biotp.embeddings import validate_positions
from biotp.utils import get_device

FinetuneMode = Literal["linear_probe", "lora", "full"]
Task = Literal["regression", "classification"]

# One hidden layer, wide enough to combine two concatenated embeddings without
# the capacity to memorize the training set. Not tuned; if a result turns on this
# number, that belongs in the log as a finding.
HEAD_HIDDEN_DIM = 256
HEAD_DROPOUT = 0.1

# Project 1's minibatch size for head training, kept as a named constant so that
# call site can pass it explicitly and stay bit-for-bit. It is deliberately no
# longer a default inside `train`: batching at a module constant while
# `train_lora` took its batch size as an argument meant an epoch was a different
# number of gradient updates on the two rungs, and early-stopping patience is
# counted in epochs. See issue #33.
BATCH_SIZE = 256

# Stop when validation loss has not improved for this many epochs. Best weights
# are restored afterwards, so a generous max_epochs costs time, not quality.
# Read only by _BestEpochTracker, which is what keeps the two rungs on one rule.
EARLY_STOPPING_PATIENCE = 10


def _clone_state(state: dict[str, Any]) -> dict[str, Any]:
    """Detach and copy a state dict, so later steps cannot move a saved snapshot."""
    return {key: value.detach().clone() for key, value in state.items()}


def _initial_history(mode: FinetuneMode, n_train: int, n_val: int) -> dict[str, Any]:
    """The history fields every training function reports, whatever it trained.

    Shared so a caller reading a manifest finds the same five keys under the same
    names regardless of which rung produced it. Each function adds its own on top.
    """
    return {
        "train_loss": [],
        "val_loss": [],
        "n_train": n_train,
        "n_val": n_val,
        "mode": mode,
    }


# What a snapshot holds. Generic rather than Any so the pair of callbacks is
# checked against each other: a restore that cannot accept what its own snapshot
# produces is caught at the call site rather than at the end of a long run.
Snapshot = TypeVar("Snapshot")


class _BestEpochTracker(Generic[Snapshot]):
    """Best-epoch selection, the early-stopping rule, and the restore afterwards.

    Both rungs of the DMS ladder run their loop through one of these. That is the
    point of the class rather than a nicety: `train` and `train_lora` differ in
    exactly one respect by design, and a stopping rule or an improvement test
    changed in one and not the other would move the measured rung-2-to-rung-3
    delta while every test still passed and every number still looked reasonable.

    Patience is read from EARLY_STOPPING_PATIENCE here rather than taken as an
    argument, so the two call sites cannot pass different values.

    The epoch index is counted here rather than supplied, so `best_epoch` is by
    construction an index into the validation losses this object was handed, and
    cannot drift from the caller's own loop variable.

    Args:
        snapshot: called on each improving epoch; returns whatever should be
            restored later. The only real difference between the two call sites:
            the frozen rung checkpoints one state dict, the LoRA rung checkpoints
            the adapters and the head as a pair.
        restore: called once by `finish`, with the best epoch's snapshot.
    """

    def __init__(
        self, snapshot: Callable[[], Snapshot], restore: Callable[[Snapshot], None]
    ) -> None:
        self._snapshot = snapshot
        self._restore = restore
        self.best_val = float("inf")
        # Deliberately -1 rather than None, so the patience arithmetic below is
        # reachable before any improvement. A NaN validation loss never improves
        # on the initial infinity, and that run should stop after patience epochs
        # and then fail loudly in `finish`, not burn every epoch first.
        self.best_epoch = -1
        self.epochs_seen = 0
        # Any rather than `Snapshot | None`, because a snapshot callback is
        # entitled to return None and this attribute must not be the thing that
        # decides whether an epoch improved. `best_epoch` is that thing.
        self._best_state: Any = None

    def update(self, val_loss: float) -> bool:
        """Record one epoch's validation loss; return True when training should stop.

        Strictly less-than, so a tie leaves the best epoch at the first minimum.
        Both call sites' tests locate that epoch with a first-minimum rule.
        """
        epoch = self.epochs_seen
        self.epochs_seen += 1

        if val_loss < self.best_val:
            self.best_val = val_loss
            self.best_epoch = epoch
            self._best_state = self._snapshot()
            return False
        return epoch - self.best_epoch >= EARLY_STOPPING_PATIENCE

    def finish(self, history: dict[str, Any]) -> None:
        """Restore the best epoch's weights and write the three keys both rungs report.

        The consistency assertions cover the one coupling this seam leaves open:
        the loop owns the loss lists and this object owns the epoch count, so a
        loop that appended in the wrong place would otherwise report a best epoch
        indexing a different list than the one it names.
        """
        assert self.best_epoch >= 0, "training completed without a best epoch"
        assert len(history["val_loss"]) == self.epochs_seen, (
            f"history recorded {len(history['val_loss'])} validation losses but "
            f"{self.epochs_seen} epochs ran; best_epoch indexes that list"
        )
        assert history["val_loss"][self.best_epoch] == self.best_val, (
            f"best_val_loss {self.best_val} is not history['val_loss']"
            f"[{self.best_epoch}]; the loop and the tracker saw different losses"
        )

        self._restore(self._best_state)
        history["best_epoch"] = self.best_epoch
        history["best_val_loss"] = self.best_val
        history["epochs_run"] = self.epochs_seen


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
    batch_size: int,
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
        batch_size: samples per gradient update. No default: patience is counted
            in epochs, so two callers batching differently get different
            optimisation budgets from the same stopping rule, and the difference
            is invisible in the history. A ladder comparing this against
            `train_lora` has to set both from one place.

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
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
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

    history = _initial_history(mode, len(x_train), len(x_val))
    # Recorded because it is now a caller's choice rather than a constant, and it
    # changes how many updates an epoch is worth.
    history["batch_size"] = batch_size
    tracker = _BestEpochTracker(
        lambda: _clone_state(model.state_dict()),
        model.load_state_dict,
    )

    for _ in range(max_epochs):
        model.train()
        permutation = torch.randperm(len(x_train), device=device)
        epoch_loss = 0.0

        for start in range(0, len(x_train), batch_size):
            batch = permutation[start : start + batch_size]
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

        should_stop = tracker.update(val_loss)
        if should_stop:
            break

    tracker.finish(history)
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


# The readouts rung 3 supports. Deliberately a superset of `embeddings.Readout`:
# difference-at-position is `mutant[i] - wildtype[i]`, and on the frozen rung the
# caller composes it from two cached matrices. Inside a training loop it cannot
# be composed that way, because the encoder moves every step and the wild-type
# vector has to be recomputed under the current adapters.
#
# It has to exist here regardless of that inconvenience: issue #11 pre-registers
# three readouts as a full axis precisely so no readout is selected after seeing
# results, and a rung 3 offering fewer than rung 2 would bias the headline delta
# downward exactly as a selection step would.
LoraReadout = Literal["mean", "at_position", "difference_at_position"]

# The runtime counterpart of LoraReadout, for callers reading a readout out of a
# config file or a SLURM array index, where the Literal cannot help.
LORA_READOUTS: tuple[str, ...] = ("mean", "at_position", "difference_at_position")


@dataclass(frozen=True)
class VariantSplit:
    """One split's sequences, readout positions, targets, and wild type.

    Grouped into a type rather than passed as a bare tuple because these move
    together and are matched by index: a silent misalignment between a sequence
    and its target trains happily and reports a plausible Spearman.

    positions is None for the mean readout and one zero-based residue index per
    sequence otherwise, mirroring `embeddings.embed_sequences`.

    wildtype is the unmutated reference, needed only by the difference readout
    and None otherwise. It is one sequence rather than one per variant because a
    DMS assay has a single reference, which is also why the difference readout
    costs one extra row per batch rather than one extra forward per variant.
    """

    sequences: list[str]
    positions: list[int] | None
    targets: np.ndarray
    wildtype: str | None


@dataclass(frozen=True)
class LoraSpec:
    """Which adapters to attach and how large to make them.

    Grouped into a type because these three move together and mean nothing
    apart: they map one-for-one onto `peft.LoraConfig(r=, lora_alpha=,
    target_modules=)`. Everything else `train_lora` takes describes how to
    optimize rather than what to adapt, and stays a parameter.

    Named `LoraSpec` rather than `LoraConfig` to avoid peft's own symbol, which
    `train_lora` imports inside its body and which would otherwise shadow this
    class exactly where it is used. "Spec" here is adapter hyperparameters; it
    is unrelated to the embedding specs that feed the cache key.

    Validation lives in the constructor rather than in `train_lora` so a SLURM
    array task fails while parsing its configuration, not after loading a 650M
    checkpoint. No field takes a default: the point of grouping the parameters
    is not to acquire defaults through the back door.

    Attributes:
        rank: adapter rank.
        alpha: adapter scaling.
        target_modules: leaf module names to adapt. fair-esm's ESM-2 exposes
            q_proj, k_proj, v_proj and out_proj under layers.N.self_attn.
    """

    rank: int
    alpha: int
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        assert self.rank > 0, f"rank must be positive, got {self.rank}"
        assert self.alpha > 0, f"alpha must be positive, got {self.alpha}"
        # A bare string is iterable, so `target_modules="q_proj"` would pass an
        # emptiness check and reach peft as ['q', '_', 'p', 'r', 'o', 'j'],
        # matching nothing. peft's error names the characters, not the mistake.
        assert isinstance(self.target_modules, tuple), (
            f"target_modules must be a tuple of module names, got "
            f"{type(self.target_modules).__name__}"
        )
        assert (
            self.target_modules
        ), "target_modules is empty, so nothing would be adapted"
        for name in self.target_modules:
            assert (
                isinstance(name, str) and name
            ), f"target_modules holds a non-name entry {name!r}"

    def as_history_block(self) -> dict[str, Any]:
        """This config as one JSON-safe block, for a history dict or a manifest.

        `target_modules` becomes a list rather than a tuple so a manifest written
        and then read back compares equal to the one that produced it. Read it
        back with `from_history_block`, not `LoraSpec(**block)`: a list is
        exactly what `__post_init__` refuses, and that guard is the one thing
        standing between `target_modules="q_proj"` and six single-character
        names reaching peft.
        """
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": list(self.target_modules),
        }

    @classmethod
    def from_history_block(cls, block: dict[str, Any]) -> LoraSpec:
        """Rebuild a spec from `as_history_block`, after a trip through JSON.

        The inverse the SLURM array needs: a job reads its configuration out of
        a manifest or a job spec, and gets back an object that has re-run every
        check rather than a dict nobody validated.

        The key set is checked rather than ignored, so a block that gained or
        lost a field fails here instead of silently dropping it.
        """
        expected = {"rank", "alpha", "target_modules"}
        assert (
            set(block) == expected
        ), f"expected keys {sorted(expected)}, got {sorted(block)}"
        return cls(
            rank=block["rank"],
            alpha=block["alpha"],
            target_modules=tuple(block["target_modules"]),
        )


def _check_split(
    split: VariantSplit, readout: LoraReadout, max_sequence_length: int, name: str
) -> None:
    """Fail loudly on a split that would train but mean nothing.

    Delegates the position range check to `embeddings.validate_positions`, the
    same guard the frozen rung uses. Checking only the *count* here is what let
    an out-of-range position reach `torch.gather` and come back as a padding
    vector, which trains and reports a plausible Spearman.
    """
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
        validate_positions(split.sequences, split.positions, max_sequence_length)

    if readout == "difference_at_position":
        assert split.wildtype, (
            f"{name} split has no wildtype, which readout {readout!r} requires: "
            "the difference is against the unmutated reference"
        )
        # Substitutions preserve length, so a mismatch means the split paired its
        # variants with the wrong reference, and every difference vector would be
        # a comparison against the wrong residue.
        for index, sequence in enumerate(split.sequences):
            assert len(sequence) == len(split.wildtype), (
                f"{name} sequence {index} is {len(sequence)} residues but the "
                f"wildtype is {len(split.wildtype)}; substitutions preserve length"
            )
    else:
        assert split.wildtype is None, (
            f"{name} split carries a wildtype, but readout {readout!r} does not "
            "use one; passing it would be silently ignored"
        )


def _encode_batch(
    encoder: Any,
    sequences: list[str],
    positions: list[int] | None,
    readout: LoraReadout,
    wildtype: str | None,
) -> Any:
    """Tokenize, forward, and collapse one batch, keeping gradients attached.

    Deliberately not `embeddings.embed_sequences`, which wraps its forward in
    `torch.inference_mode()`. That is right for the frozen rungs and fatal here:
    a LoRA run under inference mode produces no gradients and silently trains
    only the head, which is exactly the rung-2 result wearing rung 3's name.

    The collapse itself is imported rather than reimplemented, so both rungs read
    the same residue through the same code.

    Under the difference readout the wild type rides along as one extra row in
    the same batch, so it passes through the current adapters rather than a stale
    copy of them, and costs one row rather than one forward per variant.
    """
    import torch

    from biotp.embeddings import mean_pool_residues, select_residue

    truncated = [sequence[: encoder.max_sequence_length] for sequence in sequences]
    rows = list(truncated)
    if readout == "difference_at_position":
        assert wildtype is not None  # Established by _check_split; narrows for mypy.
        rows.append(wildtype[: encoder.max_sequence_length])

    _, _, tokens = encoder.batch_converter(
        [(str(index), sequence) for index, sequence in enumerate(rows)]
    )
    tokens = tokens.to(encoder.device)

    result = encoder.model(tokens, repr_layers=[encoder.repr_layer])
    representations = result["representations"][encoder.repr_layer]

    if readout == "mean":
        lengths = torch.tensor(
            [len(sequence) for sequence in truncated], device=representations.device
        )
        collapsed: Any = mean_pool_residues(representations, lengths)
        return collapsed

    assert positions is not None  # Established by _check_split; narrows for mypy.
    index = torch.tensor(positions, device=representations.device)

    if readout == "at_position":
        return select_residue(representations, index)

    # Residue i sits at token i + 1, matching select_residue's own offset.
    mutant = select_residue(representations[:-1], index)
    reference = representations[-1][index + 1]
    return mutant - reference


def train_lora(
    encoder: Any,
    head: Any,
    train_data: VariantSplit,
    val_data: VariantSplit,
    readout: LoraReadout,
    max_epochs: int,
    lr: float,
    batch_size: int,
    lora: LoraSpec,
    seed: int,
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
        lora: which adapters to attach and how large. See LoraSpec, which also
            validates them at construction.
        seed: draws the batch order. Required, because the ladder's seed axis is
            only real if rung 3 actually varies with it: the frozen rung draws
            from the global torch RNG and so responds to seeding, and a rung 3
            that ignored the seed would report three identical runs as a spread.

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
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from torch import nn

    assert max_epochs > 0, f"max_epochs must be positive, got {max_epochs}"
    assert lr > 0, f"lr must be positive, got {lr}"
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
    # The adapter hyperparameters checked themselves when the LoraSpec was built.

    # peft wraps in place, so the caller's bundle is modified. Calling twice on
    # one bundle stacks a second adapter set on the first run's weights, and peft
    # only warns. That shape is the obvious one for a sweep over readout, N and
    # seed, and it would quietly make every run after the first a continuation of
    # its predecessor.
    assert not hasattr(encoder.model, "peft_config"), (
        "encoder.model already carries LoRA adapters, so this bundle has been "
        "through train_lora before. Load a fresh bundle per run: wrapping again "
        "would stack adapters on the previous run's weights."
    )

    task = getattr(head, "task", None)
    assert task == "regression", (
        f"train_lora expects a regression head, got task {task!r}; build it with "
        "build_head so the loss cannot disagree with the head"
    )
    assert (
        readout in LORA_READOUTS
    ), f"unknown readout {readout!r}; expected one of {sorted(LORA_READOUTS)}"

    _check_split(train_data, readout, encoder.max_sequence_length, "train")
    _check_split(val_data, readout, encoder.max_sequence_length, "val")

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
            r=lora.rank,
            lora_alpha=lora.alpha,
            target_modules=list(lora.target_modules),
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
        f"{lora.target_modules}; the adapters did not attach"
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [*trainable, *head.parameters()],
        lr=lr,
    )

    history = _initial_history(
        "lora", len(train_data.sequences), len(val_data.sequences)
    )
    history.update(
        {
            "readout": readout,
            # Recorded on both rungs, not just this one: the two histories are
            # read side by side when the ladder's delta is computed, and a field
            # present in one schema and absent from the other is a trap for
            # whatever reads them next.
            "batch_size": batch_size,
            # One nested block rather than three loose keys, so a manifest
            # reader finds the adapter configuration in one place and a SLURM
            # array can round-trip the same object it was given.
            "lora": lora.as_history_block(),
            "seed": seed,
            "encoder_parameters": encoder_parameters,
            "trainable_encoder_parameters": trainable_encoder_parameters,
            # Tensors, not modules: peft adds two per adapted module (lora_A and
            # lora_B). Named for what it counts, because this lands in a run
            # manifest where the name is all a later reader has.
            "lora_parameter_tensors": sum(
                1 for name, _ in encoder.model.named_parameters() if "lora_" in name
            ),
        }
    )

    def snapshot() -> tuple[dict, dict]:
        """Adapters and head, not the whole encoder.

        The base is frozen, so it cannot differ between epochs, and cloning it
        would allocate a full copy on-device every time validation improved:
        about 140 MB at 35M and 2.6 GB at 650M, to preserve well under a
        megabyte of adapters.
        """
        return (
            _clone_state(get_peft_model_state_dict(encoder.model)),
            _clone_state(head.state_dict()),
        )

    def restore(state: tuple[dict, dict]) -> None:
        set_peft_model_state_dict(encoder.model, state[0])
        head.load_state_dict(state[1])

    tracker = _BestEpochTracker(snapshot, restore)

    def evaluate(split: VariantSplit) -> float:
        """Mean squared error over a split, through the same path scoring uses.

        Routed via `predict_lora` rather than reimplementing the batching, so
        model selection during training and the predictions reported afterwards
        cannot diverge. A validation loss computed one way and a Spearman
        computed another is a difference that never shows up as a failure.
        """
        predictions = predict_lora(encoder, head, split, readout, batch_size)
        residuals = predictions - np.asarray(split.targets, dtype=np.float64)
        return float(np.mean(residuals**2))

    for epoch in range(max_epochs):
        encoder.model.train()
        head.train()
        # Seeded by (seed, epoch) together: the epoch alone would give every run
        # the same batch order regardless of seed, so the ladder's three seeds
        # would be three identical runs reported as a spread.
        order = np.random.default_rng([seed, epoch]).permutation(
            len(train_data.sequences)
        )
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
            features = _encode_batch(
                encoder, sequences, positions, readout, train_data.wildtype
            )
            loss = loss_fn(head(features), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)

        history["train_loss"].append(epoch_loss / len(train_data.sequences))
        val_loss = evaluate(val_data)
        history["val_loss"].append(val_loss)

        should_stop = tracker.update(val_loss)
        if should_stop:
            break

    tracker.finish(history)
    return encoder, head, history


def predict_lora(
    encoder: Any,
    head: Any,
    split: VariantSplit,
    readout: LoraReadout,
    batch_size: int,
) -> np.ndarray:
    """Return one scalar prediction per variant from an adapted encoder and head.

    The fine-tuned counterpart to `predict`, which takes precomputed features and
    so cannot serve a rung whose encoder is part of the model. Kept separate from
    `train_lora` for the same reason `predict` is separate from `train`:
    evaluation must not be able to run with dropout active or gradients enabled,
    and the only reliable way to guarantee that is for it to live somewhere that
    never trains.

    Args:
        encoder: an Esm2Bundle whose model carries LoRA adapters, as returned by
            train_lora.
        head: the fitted regression head returned alongside it.
        split: the variants to score. `targets` is ignored and may hold anything;
            only the sequences, positions and wild type are read.
        readout: must match the one the model was trained under. A model fitted
            on one readout and scored under another produces finite, plausible,
            meaningless numbers.
        batch_size: sequences per forward pass.

    Returns:
        Array of shape (len(split.sequences),), in the split's own order.
    """
    import torch

    assert split.sequences, "predict_lora received an empty split"
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
    assert (
        readout in LORA_READOUTS
    ), f"unknown readout {readout!r}; expected one of {sorted(LORA_READOUTS)}"
    _check_split(split, readout, encoder.max_sequence_length, "predict")

    encoder.model.eval()
    head.eval()

    outputs: list[np.ndarray] = []
    # `no_grad`, not the `inference_mode` that `train` and `predict` use. This
    # function also runs inside train_lora's epoch loop, and ESM-2's rotary
    # embedding writes its sin/cos tables onto the module during the forward,
    # keeping them until the sequence length or the device changes. A table
    # created under inference mode is an inference tensor, and the next epoch's
    # backward refuses to save one, so validation would poison training.
    # `no_grad` produces ordinary tensors and has no such restriction.
    #
    # The failure is a crash rather than a wrong number, and it is currently
    # unreachable on the DMS ladder because substitutions preserve length, so
    # the cached length never changes after the first forward. This is the one
    # scoring path documented as shared between in-loop validation and final
    # scoring, so it is written to be safe in the stricter of the two.
    with torch.no_grad():
        for start in range(0, len(split.sequences), batch_size):
            stop = start + batch_size
            positions = None if split.positions is None else split.positions[start:stop]
            features = _encode_batch(
                encoder,
                split.sequences[start:stop],
                positions,
                readout,
                split.wildtype,
            )
            outputs.append(head(features).reshape(-1).cpu().numpy())

    predictions = np.concatenate(outputs).astype(np.float64)
    assert len(predictions) == len(split.sequences), (
        f"predicted {len(predictions)} values for {len(split.sequences)} variants; "
        "batching dropped or duplicated rows"
    )
    return predictions
