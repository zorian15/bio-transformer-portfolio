# bio-transformer-portfolio

A portfolio of three small, openly-released machine-learning experiments at the intersection of transformers and biology. The aim is hands-on transformer work with careful, leakage-aware evaluation, released as public code and model weights.

Each project stands alone but shares one infrastructure package (`biotp`) and follows the same open-science release checklist. See `PLANNING.md` for the full plan and rationale.

## The three projects

Numbering matches build order, so Project 1 is the one to build first. That order is deliberate: Project 1's sequence-only baseline is essentially Project 2's core pipeline, and Project 3 reuses the same machinery.

1. **[grounding-multimodal](projects/grounding-multimodal/)** (flagship, first) — does grounding a protein-sequence representation in text (functional annotations) beat sequence-only, once controls rule out label leakage? A protein-domain take on the broader "does language grounding help" question. [Decisions](projects/grounding-multimodal/DECISION_LOG.md)
2. **[dms-benchmark](projects/dms-benchmark/)** — when does a fine-tuned protein language model beat a simple structured baseline on deep mutational scanning fitness, and how does that change with the amount of labeled data? [Decisions](projects/dms-benchmark/DECISION_LOG.md)
3. **[epistasis-plm-torchdms](projects/epistasis-plm-torchdms/)** — masked-marginal scoring is additive over sites by construction, so a protein LM used zero-shot cannot represent epistasis at all. Can a supervised one, and how does it compare to `torchdms`? [Decisions](projects/epistasis-plm-torchdms/DECISION_LOG.md)

## Layout

| Path | Role |
|---|---|
| `PLANNING.md` | Source-of-truth plan for all three projects |
| `src/biotp/` | Shared infrastructure: ESM-2 embeddings, zero-shot masked-marginal scoring, fine-tune harness (linear probe and LoRA), leakage-aware eval, annotation-text ablation, HF release helpers |
| `tests/` | Tests for `biotp`; the remaining stub module carries strict-`xfail` contract tests (see below) |
| `projects/*/` | One folder per project: `README.md` (scope) + `DECISION_LOG.md` (experiment log) |
| `data/` | Datasets (gitignored); see `data/README.md` for what to fetch |
| `notebooks/` | Exploratory notebooks |
| `slurm/` | `submit-*.sh` sbatch scripts for GPU-heavy one-offs; see `slurm/README.md` |

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
`pytest` from the repo root. The default run deselects tests marked `network`, which download model checkpoints; `pytest -m network` selects exactly those, and CI runs them for you (see below). `biotp.utils`, `biotp.runlog`, `biotp.embeddings`, `biotp.zero_shot`, `biotp.training` (linear probe and LoRA), `biotp.evaluation` and `biotp.text_ablation` are implemented and tested normally. `biotp.release` is still a scaffold stub, so its tests come in two kinds:

- **Signature and convention tests** run for real today, and they cover the implemented modules too. They pin design decisions that should outlive implementation: embedding width comes from the checkpoint rather than a caller argument, `train(mode=...)`, `embed_sequences(readout=..., positions=...)` and `push_to_hub(private=...)` have no defaults, and `build_model_card` cannot omit `limitations`.
- **Behavioral tests** are written against the intended contract and marked `xfail(raises=NotImplementedError)`. With `xfail_strict = true` set in `pyproject.toml`, each one flips to a hard failure the moment the stub starts working, which is the cue to delete the marker and keep the assertions rather than leave a test quietly skipped forever.

`tests/test_environment.py` guards one environment invariant: PyTorch must come from conda-forge, not pip. The pip wheel bundles a second `libomp.dylib` alongside the env's own, and importing torch then aborts the process with `OMP: Error #15` instead of raising, which takes down the whole test run rather than failing one test. Since that crash cannot be caught where it happens, the test checks the packaging structurally and reports how to fix it. It is macOS-specific and skips elsewhere, including on CI's Linux runners, which install from `environment.yml` and so cannot hit the hazard in the first place.

## Continuous integration
Three GitHub Actions workflows in `.github/workflows/`:

| Workflow | What it runs | When |
|---|---|---|
| `tests.yml` | `black --check` and `ruff` on a plain Python, then `mypy` and the offline `pytest` suite inside the `biollm` env built from `environment.yml` | every push to `main` and every pull request |
| `embedding-anchor.yml` | `pytest -m network`, with the ESM-2 checkpoints cached between runs | pull requests touching the embedding path, plus weekly and on demand |
| `docs.yml` | `mkdocs build --strict`; publishes to GitHub Pages from `main` only | every push to `main` and every pull request |

The network suite is split out because it is the only one that downloads model weights, which wants a cache and a narrower trigger. Its load-bearing member is `test_embed_sequences_matches_the_frozen_reference`, which checks the current embedding code against reference vectors committed in `tests/data/`. That check is what makes a bump of `EMBEDDING_IMPL_VERSION` defensible, and it used to run only when someone remembered to type `pytest -m network`, which is a poor guard for a failure mode whose defining feature is that nothing looks wrong. See [`docs/embedding-cache.md`](docs/embedding-cache.md).

`tests/test_conventions.py` pins that arrangement mechanically: it fails if no workflow runs the network suite, if the workflow that does stops triggering on embedding changes, or if the `network` marker stops being registered. A workflow deleted in a cleanup breaks a test rather than quietly reducing coverage.

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
