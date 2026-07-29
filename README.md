# bio-transformer-portfolio

A portfolio of three small, openly-released machine-learning experiments at the intersection of transformers and biology. The aim is hands-on transformer work with careful, leakage-aware evaluation, released as public code and model weights.

Each project stands alone but shares one infrastructure package (`biotp`) and follows the same open-science release checklist. See `PLANNING.md` for the full plan and rationale.

## The three projects

Build order is deliberate: Project 2 first, because its sequence-only baseline is essentially Project 1, and Project 3 reuses the same machinery.

1. **[grounding-multimodal](projects/grounding-multimodal/)** (flagship, first) — does grounding a protein-sequence representation in text (functional annotations) beat sequence-only, once controls rule out label leakage? Mirrors the CellOLMo question in the protein domain. [Decisions](projects/grounding-multimodal/DECISION_LOG.md)
2. **[dms-benchmark](projects/dms-benchmark/)** — when does a fine-tuned protein language model beat a simple structured baseline on deep mutational scanning fitness, and how does that change with the amount of labeled data? [Decisions](projects/dms-benchmark/DECISION_LOG.md)
3. **[tcr-antibody-lm](projects/tcr-antibody-lm/)** — can a fine-tuned protein LM predict TCR-epitope specificity (or antibody escape), and does it generalize to unseen epitopes? [Decisions](projects/tcr-antibody-lm/DECISION_LOG.md)

## Layout

| Path | Role |
|---|---|
| `PLANNING.md` | Source-of-truth plan for all three projects |
| `src/biotp/` | Shared infrastructure: ESM-2 embeddings, fine-tune harness, leakage-aware eval, HF release helpers |
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

## Running locally vs on SLURM
The same code runs on the MacBook (Apple MPS) or a SLURM GPU node (CUDA); `biotp.utils.get_device()` selects automatically (set `PYTORCH_ENABLE_MPS_FALLBACK=1` locally). Use SLURM for the heavy one-offs (embedding extraction, fine-tunes) via the templates in `slurm/`, move results back with rsync, and push final checkpoints to Hugging Face. Large artifacts are never committed to git. See `PLANNING.md` (Compute and robustness; Artifacts and storage).

## Releasing to Hugging Face
Each project's final step publishes weights and a model card to the Hugging Face Hub (see `biotp.release`). Setup is only needed at release time; downloading pretrained models (ESM-2) and training need no account.

One-time setup:
1. Create a free account at huggingface.co and pick a username; it becomes part of every model URL (e.g. `your-username/grounding-multimodal`), so choose one you are happy to put on a CV.
2. Create an access token with **write** permission under Settings, then Access Tokens.
3. Log in locally so the libraries find the token: `huggingface-cli login` (caches the token), or set `HF_TOKEN` in your environment.

Then `biotp.release.push_to_hub(...)` can upload. Hosting public models and datasets is free, and large files are handled automatically via Git LFS.

## Docs and writeups (planned, not built yet)
A `docs/` folder will hold each project's writeup as a short research report, published via GitHub Pages (gh-pages) so the reports are browsable on the web. The intent is to link them from the portfolio tab of the personal site (zorian15.github.io). This is a later step; see PLANNING.md for where it sits in the roadmap.
