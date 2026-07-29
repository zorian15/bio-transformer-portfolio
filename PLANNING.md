# Transformer Portfolio Plan (3 projects)

This is the source-of-truth plan for this repo. Each project lives under `projects/`, shares the `biotp` package under `src/`, and keeps its own `DECISION_LOG.md`.

Goal: close the hands-on transformer/LLM gap with shippable, openly-released artifacts that lean on Zorian's viral-protein and adaptive-immunity expertise. Each project ends in a public repo + model weights + a short writeup. Together they signal exactly what the target roles (Ai2, Anthropic, EvolutionaryScale, Terray, Nabla, A-Alpha) screen for: practical transformer work, careful evaluation, and open science.

Order: **Project 2 first** (per preference), then Project 1, then Project 3. Rationale: Project 2's honest baseline is the sequence-only protein-LM pipeline, which is the core of Project 1, so #2 delivers most of #1 as a byproduct. #3 reuses the same fine-tuning and evaluation harness on immune data.

Assumed compute: one modern GPU (lab/cluster or cloud), frozen-embedding tricks to stay cheap. Adjust if your budget differs.

Guiding principle: MVP first. Get an end-to-end result with frozen embeddings and a small head before adding fine-tuning, fusion, or scale. Ramp complexity only after something runs end to end.

---

## Shared infrastructure (build once, reuse across all three)
Lives in `src/biotp/`.
- ESM-2 embedding extraction (`embeddings.py`): load a small ESM-2 checkpoint (e.g. 35M or 150M), produce per-sequence embeddings, cache to disk. Frozen embeddings keep everything cheap.
- Fine-tuning harness (`training.py`): swap between linear probe, LoRA, and full fine-tune behind one interface.
- Evaluation harness with leakage-aware splits (`evaluation.py`): held-out entities (proteins / families / epitopes / donors), not random rows. This is where your study-design and confounding instinct is a real edge, so make splits a first-class feature.
- Release template (`release.py`): repo layout, environment lockfile, model card, weights pushed to the Hugging Face Hub, reproducible run scripts.

---

## Project 2 (flagship, first): Does language grounding help protein representations?
Lives in `projects/grounding-multimodal/`. Mirrors the CellOLMo question ("does grounding in biological language beat molecular-data-only?") in the protein domain you know.

- **Question:** Does adding text (functional annotations) to a protein-sequence representation measurably improve a downstream task over sequence-only, and does the gain survive controls that rule out label leakage?
- **Data:** Swiss-Prot / UniProt entries with sequence plus curated function text (or GO terms / keywords). Pick a downstream task with clean labels: subcellular localization (DeepLoc), EC-number or GO-term classification, or a protein-family property. Optionally focus on viral proteins for a domain-flavored variant.
- **Model (MVP then extend):**
  - MVP: frozen ESM-2 sequence embedding + frozen text embedding (a small sentence encoder) concatenated into a small MLP head.
  - Extend: CLIP-style contrastive alignment between sequence and text; light fine-tuning of one encoder.
- **Baselines:** sequence-only (= Project 1's core pipeline), text-only, and a shuffled/random-text control.
- **Eval:** held-out proteins/families; the key result is sequence+text vs sequence-only. Run the shuffled-text and text-only controls to prove any gain is real grounding, not annotation leaking the label. This ablation rigor is the scientifically credible core and your differentiator.
- **Deliverable:** repo + weights + a writeup whose headline is the honest answer (including "it did not help, and here is why," which is a legitimate and interesting result).
- **Skills built:** using transformer encoders, multimodal fusion, contrastive learning, controlled evaluation. Strongest signal for Ai2 and foundation-model / multimodal roles.
- **Main risk:** text leakage (annotations encode the label). Turn it into the paper's spine via controls rather than hiding it.

## Project 1 (second): Protein-LM fine-tune vs structured baseline on DMS fitness
Lives in `projects/dms-benchmark/`. Extends Project 2's sequence pipeline into a benchmark that tests your priors-vs-scale thesis empirically.

- **Question:** When does a fine-tuned protein LM beat a simple structured/biophysical baseline on deep mutational scanning fitness, and how does the answer change with the amount of labeled data?
- **Data:** ProteinGym (public DMS benchmark) and/or one of your own viral DMS datasets.
- **Model:** ESM-2 zero-shot (masked-marginal log-likelihood scoring) vs fine-tuned (LoRA or regression head) vs a simple additive / site-independent baseline (or a polyclonal-style structured model).
- **Eval:** Spearman on held-out mutations, plus data-efficiency curves (performance vs number of training labels). The data-efficiency curve is the money plot: it is your "every label costs a wet-lab measurement" argument, quantified.
- **Deliverable:** repo + writeup with the benchmark and the crossover point where scale overtakes structure.
- **Skills built:** PLM fine-tuning (LoRA), zero-shot scoring, benchmarking. Ties directly to your Terray narrative.

## Project 3 (third): TCR / antibody LM for specificity or escape
Lives in `projects/tcr-antibody-lm/`. Points the same machinery at your adaptive-immunity specialty.

- **Question:** Can a fine-tuned protein LM predict TCR-epitope specificity (or antibody escape) better than a simple baseline, and does it generalize to unseen epitopes?
- **Data:** VDJdb / IEDB for TCR-epitope pairs, or an antibody escape / DMS dataset in your polyclonal wheelhouse.
- **Model:** ESM-2 or a TCR-specific transformer fine-tuned for the task; baseline = k-mer or simple classifier.
- **Eval:** the hard and honest split is held-out epitopes (no epitope shared between train and test), which exposes the memorization-vs-generalization gap that naive splits hide. Your bias/leakage instinct is the whole point here.
- **Deliverable:** repo + weights + writeup.
- **Skills built:** transformer fine-tuning on immune-repertoire data; leans into your differentiator. Signal for immuno-ML and antibody-design roles (Nabla, A-Alpha).

---

## Sequencing and rough effort
1. Shared infra + Project 2 MVP (frozen embeddings, sequence-only vs sequence+text): the biggest single step, since it stands up the whole stack. Get a number end to end before adding contrastive/fine-tuning.
2. Project 2 full (controls, contrastive, writeup, release).
3. Project 1: reuse the sequence pipeline, add zero-shot + LoRA + the data-efficiency curve. Smaller increment.
4. Project 3: swap in immune data, add leakage-aware epitope split. Domain reuse.

## Open-science release checklist (per project)
- Public repo with README, reproducible run scripts, environment lockfile.
- Model weights + model card on the Hugging Face Hub.
- Short writeup (blog or short arXiv note); honest results, including negatives.
- A one-line entry you can put on every relevant application ("trained and released ...").

## Docs and hosting (planned, later step)
Each project's writeup should live in a `docs/` folder as a short research report, published via GitHub Pages (gh-pages) so the reports are browsable on the web. Longer term, link these from the portfolio tab of the personal site (zorian15.github.io). Not built yet; revisit once the first project has a result worth writing up. The per-project `DECISION_LOG.md` files are the raw material for these reports.

## Open questions to resolve before starting
- Compute budget (single GPU vs cluster vs cloud credits)? Sets the ESM-2 size and whether Project 2 uses frozen or fine-tuned encoders.
- Use a clean public benchmark (ProteinGym, DeepLoc, VDJdb) or bring one of your own viral DMS datasets for a domain-flavored version?
- Target: ship one polished project, or all three at lighter depth? Affects how much to invest in each writeup.
