# bio-transformer-portfolio

Three small, openly-released transformer-and-biology experiments. These are short
research reports, written as the work happens, so they stay honest about what is
settled and what is still open.

Each project ships public code, weights and a model card on the Hugging Face Hub,
and a writeup whose headline is the honest answer, including a clean null result
if that is what happens.

## Where to start

If you want the science, read
[Project 1's introduction](grounding-multimodal/introduction.md) and then
[its results](grounding-multimodal/results.md). If you want the machinery the
experiments run on, read [run logging](run-logging.md) and the
[embedding cache](embedding-cache.md).

If a transformer concept these reports lean on is unfamiliar, the
[appendix](appendix.md) explains the ones that change what a number means:
what a protein language model actually returns and why the readout is a
choice, why fine-tuning cannot use the embedding cache, what LoRA does and does
not buy, how zero-shot scoring works without training, and why the splits here
hold out sites rather than rows.

Each project also keeps an **experiment log**: a chronological record of runs and
the decisions they drove, newest first. The reports describe the setup; the logs
record what actually happened, including the wrong turns.

## Status

| Project | Reports | Experiment log |
|---|---|---|
| 1. grounding-multimodal | Introduction, Data, Method, Ablation filter, Results | [Log](grounding-multimodal/decision-log.md) |
| 2. dms-benchmark | Not started | [Log](dms-benchmark/decision-log.md) |
| 3. tcr-antibody-lm | Not started | [Log](tcr-antibody-lm/decision-log.md) |

Build order follows the project numbering, and Projects 2 and 3 have not begun,
so their logs currently hold only the entry template. Scope and sequencing live
in [`PLANNING.md`](https://github.com/zorian15/bio-transformer-portfolio/blob/main/PLANNING.md).

## Current headline result

Project 1 asks whether grounding a protein-sequence representation in text
improves subcellular localization over sequence alone, and whether any gain
survives controls that rule out label leakage.

Adding free-text function annotations to a frozen ESM-2 representation improves
macro-F1 from 0.616 to 0.740, roughly 20x the seed spread, and a shuffled-text
control lands *below* sequence-only. So the gain is tied to each protein's own
annotation rather than to the extra dimensions.

That is the narrow claim, and it holds. The broader claim, that this is
*grounding* rather than annotations quietly restating the label, has now been
measured by [ablating the localization-stating sentences and
re-running](grounding-multimodal/results.md#separating-grounding-from-leakage).
About a seventh of the gain is the prose naming the compartment. Most of the rest
is not leakage, but neither is it robust: a length-matched random-sentence
control shows the bulk of the gain depends on having the whole annotation,
particularly for the rare compartments. The filter and its judgement calls are
documented in [Ablation filter](grounding-multimodal/ablation.md).

## Reading these locally

The site is built with [MkDocs](https://www.mkdocs.org/) and its `readthedocs`
theme:

```bash
mamba activate biollm
mkdocs serve      # live-reloading preview at http://127.0.0.1:8000
mkdocs build      # static site into site/ (gitignored)
```

Figures are generated from the committed result files, not drawn by hand, so a
plot cannot drift from the table beside it:

```bash
python projects/grounding-multimodal/scripts/make_figures.py
```

Pushing to `main` deploys the site to GitHub Pages automatically.
