# bio-transformer-portfolio

A portfolio of three small, openly-released machine-learning experiments at the intersection of transformers and biology. The aim is hands-on transformer work with careful, leakage-aware evaluation, released as public code and model weights.

Each project stands alone but shares one infrastructure package (`biotp`) and follows the same open-science release checklist. See `PLANNING.md` for the full plan and rationale.

## The three projects

Numbering matches build order, so Project 1 is the one to build first. That order is deliberate: Project 1's sequence-only baseline is essentially Project 2's core pipeline, and Project 3 reuses the same machinery.

1. **[grounding-multimodal](projects/grounding-multimodal/)** (flagship, first) — does grounding a protein-sequence representation in text (functional annotations) beat sequence-only, once controls rule out label leakage? A protein-domain take on the broader "does language grounding help" question. [Decisions](projects/grounding-multimodal/DECISION_LOG.md)
2. **[dms-benchmark](projects/dms-benchmark/)** — when does a fine-tuned protein language model beat a simple structured baseline on deep mutational scanning fitness, and how does that change with the amount of labeled data? [Decisions](projects/dms-benchmark/DECISION_LOG.md)
3. **[tcr-antibody-lm](projects/tcr-antibody-lm/)** — can a fine-tuned protein LM predict TCR-epitope specificity (or antibody escape), and does it generalize to unseen epitopes? [Decisions](projects/tcr-antibody-lm/DECISION_LOG.md)

## Layout

| Path | Role |
|---|---|
| `PLANNING.md` | Source-of-truth plan for all three projects |
| `src/biotp/` | Shared infrastructure: ESM-2 embeddings, fine-tune harness, leakage-aware eval, HF release helpers |
| `tests/` | Tests for `biotp`; the stub modules carry strict-`xfail` contract tests (see below) |
| `projects/*/` | One folder per project: `README.md` (scope) + `DECISION_LOG.md` (experiment log) |
| `data/` | Datasets (gitignored); see `data/README.md` for what to fetch |
| `notebooks/` | Exploratory notebooks |
| `slurm/` | sbatch templates for GPU-heavy one-offs; see `slurm/README.md` |

## Quickstart

```bash
mamba env create -f environment.yml   # creates the 'biollm' env
mamba activate biollm
pip install -e .                       # installs the biotp package (editable)
# then fetch data per data/README.md and start with projects/grounding-multimodal
```

## Working conventions
See `CLAUDE.md`. In short: log every meaningful run in the relevant project's `DECISION_LOG.md`; Python is formatted with Black and checked with ruff/mypy/pytest; results are reported honestly, negatives included.

## Tests
`pytest` from the repo root. `biotp.utils` is implemented and tested normally. The other four modules are scaffold stubs, so their tests come in two kinds:

- **Signature and convention tests** run for real today. They pin design decisions that should outlive implementation: embedding width comes from the checkpoint rather than a caller argument, `train(mode=...)` and `push_to_hub(private=...)` have no defaults, and `build_model_card` cannot omit `limitations`.
- **Behavioral tests** are written against the intended contract and marked `xfail(raises=NotImplementedError)`. With `xfail_strict = true` set in `pyproject.toml`, each one flips to a hard failure the moment the stub starts working, which is the cue to delete the marker and keep the assertions rather than leave a test quietly skipped forever.

`tests/test_environment.py` guards one environment invariant: PyTorch must come from conda-forge, not pip. The pip wheel bundles a second `libomp.dylib` alongside the env's own, and importing torch then aborts the process with `OMP: Error #15` instead of raising, which takes down the whole test run rather than failing one test. Since that crash cannot be caught where it happens, the test checks the packaging structurally and reports how to fix it.

## Running locally vs on SLURM
The same code runs on the MacBook (Apple MPS) or a SLURM GPU node (CUDA); `biotp.utils.get_device()` selects automatically (set `PYTORCH_ENABLE_MPS_FALLBACK=1` locally). Use SLURM for the heavy one-offs (embedding extraction, fine-tunes) via the templates in `slurm/`, move results back with rsync, and push final checkpoints to Hugging Face. Large artifacts are never committed to git. See `PLANNING.md` (Compute and robustness; Artifacts and storage).

## Releasing to Hugging Face
Each project's final step publishes weights and a model card to the Hugging Face Hub (see `biotp.release`). Setup is only needed at release time; downloading pretrained models (ESM-2) and training need no account.

One-time setup:
1. Create a free account at huggingface.co and pick a username; it becomes part of every model URL (e.g. `your-username/grounding-multimodal`), so choose one you are happy to put on a CV.
2. Create an access token with **write** permission under Settings, then Access Tokens.
3. Log in locally so the libraries find the token: `huggingface-cli login` (caches the token), or set `HF_TOKEN` in your environment.

Then `biotp.release.push_to_hub(...)` can upload. Hosting public models and datasets is free, and large files are handled automatically via Git LFS.

## Docs and writeups
`docs/` holds each project's writeup as a short research report, built into a browsable site with [MkDocs](https://www.mkdocs.org/) and its `readthedocs` theme: sidebar navigation, prev/next paging above and below each page, and full-text search. Metrics are defined mathematically (MathJax via `pymdownx.arithmatex`), and the headline tables are paired with figures.

```bash
mamba activate biollm
mkdocs serve     # live preview at http://127.0.0.1:8000
mkdocs build     # static site into site/ (gitignored)
```

Pushing to `main` builds and publishes to GitHub Pages via `.github/workflows/docs.yml`; pull requests build in strict mode but do not publish, so a broken link fails review rather than the live site. Each project's `DECISION_LOG.md` is symlinked into `docs/` and appears in the site as that project's experiment log, so the log has one home and two readers.

Still to do: link the published site from the portfolio tab of the personal site (zorian15.github.io). See PLANNING.md.
