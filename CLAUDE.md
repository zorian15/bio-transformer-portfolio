# CLAUDE.md — bio-transformer-portfolio

Read `PLANNING.md` first; it is the source of truth for scope and sequencing.

## What this repo is
Three small, openly-released transformer-and-biology experiments (see `PLANNING.md`). Build order: `grounding-multimodal` first, then `dms-benchmark`, then `tcr-antibody-lm`. They share the `biotp` package in `src/`.

## Environment
- Use **mamba**, not venv. The env is named **`biollm`**.
- Create/update from `environment.yml`: `mamba env create -f environment.yml` (or `mamba env update -f environment.yml`), then `mamba activate biollm`, then `pip install -e .`.

## Experiment workflow
- Every meaningful run or decision gets an entry in that project's `DECISION_LOG.md` (newest on top): question/hypothesis, setup (data, model, config), result (metric, plot ref), decision/next step.
- MVP first: get an end-to-end number with frozen embeddings and a small head before adding fine-tuning, fusion, or scale.
- Evaluation splits are leakage-aware by construction (held-out proteins / families / epitopes, not random rows). Treat this as a first-class feature, not an afterthought.

## Code conventions
- Format with Black; lint with ruff; type-check with mypy; test with pytest. All Python should pass these.
- Prefer assertions over silent failures; fail loudly on unexpected inputs.
- When adding a parameter to an existing function, do not give it a default; let old call sites break so they are caught.

## Writing conventions (READMEs, writeups, model cards)
- No prose em-dashes; use commas, semicolons, or colons. Keep en-dash ranges and compound modifiers.
- Report results honestly, including negative results. A clean "it did not help, and here is why" is a legitimate deliverable.

## Open science
Each project ships: public code, model weights + model card on the Hugging Face Hub, and a short honest writeup. See the release checklist in `PLANNING.md`.
