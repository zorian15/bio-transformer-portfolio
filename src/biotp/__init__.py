"""biotp: shared infrastructure for the bio-transformer-portfolio experiments.

Modules:
    embeddings  -- load ESM-2 and extract cached per-sequence embeddings.
    training    -- linear-probe / LoRA / full fine-tune behind one interface.
    evaluation  -- leakage-aware splits and metrics.
    release     -- Hugging Face Hub upload and model-card helpers.

These are scaffold stubs. See PLANNING.md for the intended behavior of each.
"""

__version__ = "0.0.1"
