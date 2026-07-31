# CLAUDE.md — bio-transformer-portfolio

Read `PLANNING.md` first; it is the source of truth for scope and sequencing.

## What this repo is
Three small, openly-released transformer-and-biology experiments (see `PLANNING.md`). Build order matches the project numbering: `grounding-multimodal` (Project 1) first, then `dms-benchmark` (Project 2), then `tcr-antibody-lm` (Project 3). They share the `biotp` package in `src/`.

## Environment
- Use **mamba**, not venv. The env is named **`biollm`**.
- Create/update from `environment.yml`: `mamba env create -f environment.yml` (or `mamba env update -f environment.yml`), then `mamba activate biollm`, then `pip install -e .`.
- PyTorch comes from **conda-forge** via `environment.yml`; never `pip install torch`. The pip wheel bundles a second OpenMP runtime, and importing torch then aborts the process with `OMP: Error #15` instead of raising. `tests/test_environment.py` guards this.
- Device: `biotp.utils.get_device()` returns cuda, then mps, then cpu. Locally set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- SLURM: submit `slurm/*.sbatch` for GPU-heavy one-offs while cluster access lasts; use the same `biollm` env on the cluster. Default partition `campus-new`; `chorus` for L40S.

## Artifacts and storage
- git tracks only small files: code, configs, metrics, small figures, DECISION_LOGs. Large binaries are gitignored.
- Move embedding caches with rsync (they are regenerable); push final checkpoints to the Hugging Face Hub. No git-LFS/DVC/cloud bucket for now. See PLANNING.md.

## Run logging
- Every pipeline script runs its body inside `biotp.runlog.run_context(...)`, and logs through `get_logger(...)` rather than `print`. Bare prints into a redirected file are block-buffered, so a long step looks identical to a hang.
- Each run writes `logs/<name>-<timestamp>.log` plus a JSON manifest holding params, per-step timings, recorded counts, git commit and dirty flag, device, and package versions. `logs/` is gitignored; runs that produce committed metrics also drop a manifest copy beside them.
- Record anything a writeup would cite with `run.record(...)`, so `DECISION_LOG.md` entries cite a manifest rather than memory. See `docs/run-logging.md`.

## Embedding cache invalidation
The embedding cache key covers the inputs **and** the code that shapes them, via the spec dicts in `biotp.embeddings` (`sequence_embedding_spec`, `text_embedding_spec`). Changing a named field there invalidates caches automatically. `EMBEDDING_IMPL_VERSION` covers changes that no named field captures, and it is the one part of the key that depends on a human noticing.

**Review checklist, whenever a diff touches `biotp/embeddings.py`:**
- Ask first: could this change the numbers a cached vector would hold? Pooling, truncation, normalization, layer selection, dtype, the empty-text rule, or any bug fix that moves the vectors, all mean yes.
- If yes and no named spec field already captures it, **bump `EMBEDDING_IMPL_VERSION` in the same commit**, and say so in the commit message.
- If yes, `test_cache_key_is_stable_for_the_recorded_spec` will fail. That failure is the reminder, not a nuisance: confirm the change was intended, then update `GOLDEN_SPEC_KEY` in the same commit. Never update the golden key on its own to make a red test green.
- If no, the key should be unchanged and that test should still pass. A surprising failure means the change was less cosmetic than it looked.
- Stale caches are the silent failure mode: results keep computing, numbers stay plausible, and they describe code that is no longer in the repo. Prefer an unnecessary bump (costs a recompute) to a missed one (costs the truth of a result).

## Experiment workflow
- Every meaningful run or decision gets an entry in that project's `DECISION_LOG.md` (newest on top): question/hypothesis, setup (data, model, config), result (metric, plot ref), decision/next step.
- MVP first: get an end-to-end number with frozen embeddings and a small head before adding fine-tuning, fusion, or scale.
- Evaluation splits are leakage-aware by construction (held-out proteins / families / epitopes, not random rows). Treat this as a first-class feature, not an afterthought.

## Code conventions
- Format with Black; lint with ruff; type-check with mypy; test with pytest. All Python should pass these.
- Prefer assertions over silent failures; fail loudly on unexpected inputs.
- When adding a parameter to an existing function, do not give it a default; let old call sites break so they are caught.

## Docs site
- `docs/` is built into a MkDocs site by `mkdocs.yml`, using the `readthedocs` theme that ships with mkdocs. Preview with `mkdocs serve`; `site/` is gitignored.
- **Any page added to `docs/` must also be added to the `nav:` block in `mkdocs.yml`**, or it builds but is unreachable from the sidebar.
- CI runs `mkdocs build --strict`, so a broken internal link fails the build. Link to files outside `docs/` (scripts, `PLANNING.md`) by absolute GitHub URL, since relative paths escaping `docs_dir` cannot resolve on the site.
- Each project's `DECISION_LOG.md` is symlinked in as `docs/<project>/decision-log.md`. Edit the real file under `projects/`, never the symlink, and keep those logs free of relative Markdown links, which would resolve differently in the two locations.
- `mkdocs` is pinned `<2` in `environment.yml`: MkDocs 2.0 drops the plugin and theming systems with no migration path, so unpinning would break this config at the next env rebuild.
- **Figures are generated, never hand-drawn.** `projects/grounding-multimodal/scripts/make_figures.py` reads the committed `results/*.csv` and `results/*.json` and writes SVG into `docs/<project>/figures/`. Re-run it whenever those results change, and commit the SVGs; CI builds the docs without matplotlib and will serve whatever is in git. A figure that disagrees with the table beside it means the script was not re-run.
- Metrics cited in a writeup get a mathematical definition, via `pymdownx.arithmatex` (`\(...\)` inline, `\[...\]` display). `docs/grounding-multimodal/method.md` is the pattern to follow.

## Writing conventions (READMEs, writeups, model cards)
- No prose em-dashes; use commas, semicolons, or colons. Keep en-dash ranges and compound modifiers.
- Report results honestly, including negative results. A clean "it did not help, and here is why" is a legitimate deliverable.

## Open science
Each project ships: public code, model weights + model card on the Hugging Face Hub, and a short honest writeup. See the release checklist in `PLANNING.md`.
