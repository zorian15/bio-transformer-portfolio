"""biotp: shared infrastructure for the bio-transformer-portfolio experiments.

Modules:
    embeddings  -- load ESM-2 and extract cached per-sequence embeddings.
    training    -- linear-probe / LoRA / full fine-tune behind one interface.
    evaluation  -- leakage-aware splits and metrics.
    text_ablation -- strip annotation bookkeeping and ablate label-stating text.
    release     -- Hugging Face Hub upload and model-card helpers.
    utils       -- device selection and seeding (implemented).

embeddings, evaluation, text_ablation and utils are implemented. training is
implemented for the linear-probe mode only, and raises NotImplementedError for
LoRA and full fine-tuning; release is still a scaffold stub. See PLANNING.md for
the intended behavior of each.
"""

__version__ = "0.0.1"
