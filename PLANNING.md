# Transformer Portfolio Plan (3 projects)

This is the source-of-truth plan for this repo. Each project lives under `projects/`, shares the `biotp` package under `src/`, and keeps its own `DECISION_LOG.md`.

Goal: build practical, openly-released transformer experiments at the intersection of machine learning and biology, drawing on a background in viral-protein and adaptive-immunity modeling. Each project ends in a public repo, model weights, and a short writeup, with an emphasis on careful, leakage-aware evaluation and reproducibility.

Order: **build in numbered order**, Project 1, then Project 2, then Project 3. The numbers are assigned to match the build order, so "Project 1" always means "the one to do first." Rationale for that order: Project 1's honest baseline is the sequence-only protein-LM pipeline, which is also the core of Project 2, so Project 1 delivers most of Project 2 as a byproduct. Project 3 then reuses the same fine-tuning and evaluation harness on multi-mutant DMS data.

Guiding principle: MVP first. Get an end-to-end result with frozen embeddings and a small head before adding fine-tuning, fusion, or scale. Ramp complexity only after something runs end to end.

## Compute and robustness (resolved)
Two-path design: use SLURM GPUs freely while available, but keep every step runnable on the laptop so nothing breaks when access ends. The same code runs in both places; only the device and the model size differ.

Primary machine: M5 MacBook Air, 24 GB unified memory, Apple Silicon (PyTorch MPS backend, no CUDA). SLURM GPU nodes (CUDA) are available now, but assume access may become limited within a few months.

- Device-agnostic code: `biotp.utils.get_device()` picks cuda, then mps, then cpu, so scripts run unchanged on a GPU node or the Mac. Set PYTORCH_ENABLE_MPS_FALLBACK=1 locally for ops that lack MPS kernels.
- SLURM-ready from day one: `slurm/` holds sbatch templates for the GPU-heavy one-offs (embedding extraction, LoRA or full fine-tunes). Submit these while access lasts; the larger ESM-2 sizes are exactly what they are for.
- Decouple expensive from iterative. Extract ESM-2 embeddings once (on SLURM when available, else on the laptop for small checkpoints), cache to disk, and run all head training and evaluation on the cached vectors. The costly step runs anywhere and only once; fast iteration runs on the Mac indefinitely.
- Front-load while access lasts. Run GPU-heavy jobs at every size that might be needed and cache their outputs (embeddings, checkpoints), so later work is not blocked if SLURM access ends. A rented cloud GPU or Colab is the fallback afterward.
- ESM-2 sizing: iterate with esm2_t12_35M (dim 480) or esm2_t30_150M (dim 640) locally; treat 650M (dim 1280) as a "run on SLURM, cache the output" job; skip 3B and larger.
- Prefer parameter-efficient fine-tuning (LoRA) and small checkpoints, so artifacts stay portable and fit the laptop.

## Artifacts and storage (resolved)
- git tracks only small, text-ish files: code, configs, metrics tables, small figures, and the DECISION_LOG files. Large binaries are gitignored.
- Embedding caches are regenerable, so stage rather than version them: compute on SLURM (or the laptop for small checkpoints), rsync/scp to the laptop when needed, recompute if lost.
- Trained checkpoints go to the Hugging Face Hub (durable, survives losing SLURM, doubles as the public release). Keep local copies under gitignored `weights/`.
- No git-LFS, DVC, or cloud bucket for now; revisit only if artifact volume outgrows rsync + HF.

---

## Shared infrastructure (build once, reuse across all three)
Lives in `src/biotp/`.
- ESM-2 embedding extraction (`embeddings.py`): load a small ESM-2 checkpoint (e.g. 35M or 150M), produce per-sequence embeddings, cache to disk. Frozen embeddings keep everything cheap.
- Fine-tuning harness (`training.py`): swap between linear probe, LoRA, and full fine-tune behind one interface.
- Evaluation harness with leakage-aware splits (`evaluation.py`): held-out entities (proteins / families / epitopes / donors), not random rows. Leakage-aware splits are the methodological backbone of these experiments, so make them a first-class feature.
- Leakage ablation for annotation text (`text_ablation.py`): strip database bookkeeping from curated prose, split it into sentences, drop the sentences stating the label according to a caller-supplied vocabulary, and report how much was removed. Leakage-aware splits handle a leak *across* the split; this handles a leak *inside* the input. The vocabulary stays with the project that owns the task: subcellular compartments for Project 1. Project 3 no longer has a text arm, so nothing there uses it.
- Release template (`release.py`): repo layout, environment lockfile, model card, weights pushed to the Hugging Face Hub, reproducible run scripts.

---

## Project 1 (flagship, first): Does language grounding help protein representations?
Lives in `projects/grounding-multimodal/`. Takes up a live question in the field, whether grounding a representation in biological language beats molecular-data-only, and asks it in the protein domain.

- **Question:** Does adding text (functional annotations) to a protein-sequence representation measurably improve a downstream task over sequence-only, and does the gain survive controls that rule out label leakage?
- **Data:** Swiss-Prot / UniProt entries with sequence plus curated function text (or GO terms / keywords). Pick a downstream task with clean labels: subcellular localization (DeepLoc), EC-number or GO-term classification, or a protein-family property. Optionally focus on viral proteins for a domain-flavored variant.
- **Model (MVP then extend):**
  - MVP: frozen ESM-2 sequence embedding + frozen text embedding (a small sentence encoder) concatenated into a small MLP head.
  - Extend: CLIP-style contrastive alignment between sequence and text; light fine-tuning of one encoder.
- **Baselines:** sequence-only (= Project 2's core pipeline), text-only, and a shuffled/random-text control.
- **Eval:** held-out proteins/families; the key result is sequence+text vs sequence-only. Run the shuffled-text and text-only controls to prove any gain is real grounding, not annotation leaking the label. This ablation rigor is the scientifically credible core of the result.
- **Deliverable:** repo + weights + a writeup whose headline is the honest answer (including "it did not help, and here is why," which is a legitimate and interesting result).
- **Skills exercised:** using transformer encoders, multimodal fusion, contrastive learning, controlled evaluation.
- **Main risk:** text leakage (annotations encode the label). Turn it into the analysis's spine via controls rather than hiding it.

## Project 2 (second): Protein-LM fine-tune vs structured baseline on DMS fitness
Lives in `projects/dms-benchmark/`. Extends Project 1's sequence pipeline into a benchmark that tests the priors-versus-scale question empirically.

- **Question:** When does a fine-tuned protein LM beat a simple structured/biophysical baseline on deep mutational scanning fitness, and how does the answer change with the amount of labeled data?
- **Data:** ProteinGym (public DMS benchmark) and/or an in-house viral DMS dataset.
- **Model:** ESM-2 zero-shot (masked-marginal log-likelihood scoring) vs fine-tuned (LoRA or regression head) vs a simple additive / site-independent baseline (or a biophysically-structured model).
- **Eval:** Spearman on held-out mutations, plus data-efficiency curves (performance vs number of training labels). The data-efficiency curve is the money plot: it quantifies the "every label costs a wet-lab measurement" argument.
- **Deliverable:** repo + writeup with the benchmark and the crossover point where scale overtakes structure.
- **Skills exercised:** PLM fine-tuning (LoRA), zero-shot scoring, benchmarking.

## Project 3 (third): Epistasis, or what a protein LM structurally cannot see
Lives in `projects/epistasis-plm-torchdms/`. Points the Project 2 machinery at multi-mutant SARS-CoV-2 RBD data and compares it against `torchdms`, a DMS model written by this repo's author.

- **Question:** Deep mutational scanning of combinations is where epistasis lives. Can a protein LM capture it, and how does it compare to a model built for the job?
- **The observation the project is built on:** masked-marginal scoring is **additive over sites by construction**. Each term masks one position of the *wild-type* sequence, so nothing in the sum ever sees the other mutation, and the score of a double mutant is identically the sum of its singles. Verified on real data at exactly 0.00e+00 deviation. This is not an empirical weakness to be measured; it is a property of the scoring rule. The standard way people use protein LMs zero-shot cannot represent epistasis at all.
- **Data:** Starr/Bloom SARS-CoV-2 RBD DMS, the barcoded libraries carrying variable numbers of mutations. ProteinGym ships only the single-mutant summary (`SPIKE_SARS2_Starr_2020_binding` and `_expression`, `includes_multiple_mutants` false), so the multi-mutant data comes from the Bloom lab release upstream of it.
- **Arms:**
  - PLM zero-shot masked-marginals: provably additive, the null.
  - PLM embeddings of the full mutant sequence + head: the embedding of a double is not the sum of the singles' embeddings, so this *can* express epistasis. Whether it does is the open question.
  - `torchdms` additive: an explicit additive model fit to the data, which separates "additive because the biology is" from "additive because the model cannot do otherwise."
  - `torchdms` with a latent nonlinearity: global epistasis, the thing the tool was built for.
- **Eval:** train on singles, test on multiples. Held-out mutational *combinations*, leakage-aware by construction, and the question the field actually asks.
- **Deliverable:** repo + weights + writeup.
- **Skills exercised:** fine-tuning, leakage-aware generalization, and a comparison against a purpose-built structured model rather than a strawman.
- **Known constraint:** `torchdms` pins `python_requires=">=3.8,<3.10"` and hard-pins `pandas==1.4.2`, so it cannot live in the Python 3.11 `biollm` env. It ships a `tdms` console entry point, so it runs in its own environment behind a subprocess boundary rather than as an import.

### Deferred: TCR / antibody LM for specificity or escape
The original Project 3, kept here because the reasoning should outlive the decision. It asked whether a fine-tuned protein LM predicts TCR-epitope specificity better than a simple baseline, on VDJdb / IEDB, split on held-out epitopes. It was replaced because the epistasis project exercises the same skills (fine-tuning, leakage-aware splits, viral protein data) with far more domain depth, and compares against a tool this repo's author wrote rather than a strawman. Worth returning to; nothing about it was wrong.

---

## Sequencing and rough effort
1. Shared infra + Project 1 MVP (frozen embeddings, sequence-only vs sequence+text): the biggest single step, since it stands up the whole stack. Get a number end to end before adding contrastive/fine-tuning.
2. Project 1 full (controls, contrastive, writeup, release).
3. Project 2: reuse the sequence pipeline, add zero-shot + LoRA + the data-efficiency curve. Smaller increment.
4. Project 3: swap in multi-mutant RBD data, add the singles-to-multiples split, and wire `torchdms` in behind a subprocess boundary. Domain reuse, plus a comparison against a purpose-built structured model.

## Open-science release checklist (per project)
- Public repo with README, reproducible run scripts, environment lockfile.
- Model weights + model card on the Hugging Face Hub.
- Short writeup (blog or short arXiv note); honest results, including negatives.
- A concise, linkable summary of what was trained and released.

## Docs and hosting
Each project's writeup lives in `docs/` as a short research report. These are built into a browsable site with MkDocs and its `readthedocs` theme (`mkdocs.yml`), giving sidebar navigation, prev/next paging, and full-text search over the reports. `.github/workflows/docs.yml` publishes to GitHub Pages on every push to `main` and builds pull requests in strict mode without publishing, so broken internal links fail CI instead of shipping.

The per-project `DECISION_LOG.md` files remain the raw material for these reports, and each is symlinked into `docs/` so it appears in the site as that project's experiment log without being duplicated.

Remaining: link the published site from the portfolio tab of the personal site (zorian15.github.io).

## Open questions to resolve before starting
- Use a clean public benchmark (ProteinGym, DeepLoc, VDJdb) or bring an in-house viral DMS dataset for a domain-flavored version?
- Target: ship one polished project, or all three at lighter depth? Affects how much to invest in each writeup.
