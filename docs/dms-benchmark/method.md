# Method: the three-rung ladder

**Status:** pipeline implemented; the fine-tuned rung runs on SLURM. See
[Results](results.md) for what has landed.

This project asks what fine-tuning a protein language model buys over using the
same model with no gradient updates at all, and how the answer changes with the
number of labels. For a protein LM, "no gradient updates" is what prompting
means, so the gap between those two is the quantity of interest.

The [appendix](../appendix.md) explains the machinery this page assumes: what a
protein LM returns, why the readout is a choice, why the fine-tuned rung cannot
use the embedding cache, and what LoRA does and does not buy.

## The ladder

| rung | what it is | what the step isolates |
|---|---|---|
| 1 | ESM-2 zero-shot masked-marginals | the pretrained prior, no labels |
| 2 | frozen ESM-2 embeddings + MLP head | what supervision buys at a fixed representation |
| 3 | LoRA on ESM-2 + the same head | what adapting the representation buys on top |

Each step changes exactly one thing. The headline is rung 2 to rung 3; rung 1 is
what makes that number legible, because without a floor "supervision reached
Spearman 0.6" says nothing.

**All three rungs run the same checkpoints**, `esm2_t12_35M_UR50D` and
`esm2_t33_650M_UR50D`. This is the invariant the design rests on, and since issue
#34 it is stated as "the same checkpoints on both supervised rungs" rather than
"one checkpoint": checkpoint is a grid axis now, not a fixed constant, but the
two supervised rungs still walk that axis together, so every rung-3 row has a
rung-2 row at its own size and the rung 2 to rung 3 delta at either size never
conflates adaptation with model scale. Checkpoint is the outermost axis of the
supervised grid, so the two sizes are complete, disjoint halves of the array
rather than interleaved. 650M exists to answer the objection a 35M-only result
leaves standing, that the encoder was too small for adaptation to matter. Rung
1 costs almost nothing at either size because masked-marginal scoring needs one
forward pass per *distinct mutated position* rather than one per variant.

**Rung 3's adapters** are rank 8, \(\alpha = 16\), attached to the `q_proj` and
`v_proj` projections of every attention block. Those three values travel together
as a single frozen `LoraSpec` (`LORA_SPEC` in `run_arms.py`) rather than as loose
constants, so a SLURM array task carries one serializable object per job and the
run manifest records the adapter configuration as one nested `lora` block. The
spec validates itself at construction, which means a malformed configuration
fails while the job parses its arguments rather than after loading a checkpoint.

## Scoring

Rung 1 scores a variant by how much likelihood the model moves from the wild-type
residue to the mutant one, at each mutated position, with that position masked:

\[
s(v) = \sum_{i \in M(v)} \Big[ \log p\big(a_i^{\text{mut}} \mid x_{\setminus i}\big)
                             - \log p\big(a_i^{\text{wt}}  \mid x_{\setminus i}\big) \Big]
\]

Rungs 2 and 3 predict the assay score directly from a representation of the
mutant. All three are compared by Spearman rank correlation against the measured
DMS score on the held-out fold, which is scale-free and so does not care that
rung 1 produces log-odds while rungs 2 and 3 produce fitted values.

## Splits

ProteinGym ships five folds per assay under three schemes, and all three are
reported. Fold 0 is the test set, fold 1 is validation for early stopping, and
folds 2-4 are the pool training subsets are drawn from. That is a single held-out
fold rather than full five-fold cross-validation: rotating the test fold would
multiply every arm by five for a precision the headline does not need.

| scheme | what it holds out | measured on `R1AB_SARS2_Flynn_2022` |
|---|---|---|
| `random` | random variants | folds 0 and 1 share **291** of ~297 positions |
| `modulo` | positions spread across the sequence | share **0** |
| `contiguous` | a contiguous block of positions | share **0** |

Under `random` a model can learn that a given site tolerates nothing and score
well on held-out mutations at that same site without transferring anything. The
other two make that impossible by construction. `make_splits` asserts the
disjointness rather than trusting it, because a silent violation would turn a
memorization result into a generalization one with nothing anomalous in the
number.

The *shape* across the three schemes is the finding, more than any single value.

## Readout

A protein LM returns one vector per residue, so something must collapse them into
one vector per variant. For a single substitution that choice matters a great
deal, since mean pooling moves a 300-residue protein's vector by about one part
in 300 when one residue changes.

Three readouts run on both supervised rungs, as a pre-registered axis rather than
a tuned knob:

- `mean`, the average over residues, comparable to published supervised baselines
- `at_position`, the vector at the mutated residue
- `difference_at_position`, \(\text{mutant}[i] - \text{wildtype}[i]\)

Selecting one on validation and carrying it forward would tune rung 2 and leave
rung 3 untuned, shrinking exactly the difference the experiment measures.

## Data efficiency

Each supervised arm is fit at \(N \in \{32, 128, 512, 2048\}\) labelled variants,
drawn independently per \(N\) from the training pool, with three seeds. The draw
deliberately does not depend on the readout, so the three readouts are compared
on identical training sets.

Each arm records the realized number of **distinct training positions** alongside
\(N\). Under `contiguous` a small draw covers few sites, so a flat point on the
curve may be a site-coverage limit rather than a label-count limit, and only that
number tells the two apart.

## Cohort

Three ProteinGym assays under a filter fixed before any data was downloaded:
single-substitution only, target length \(\le 400\), single-mutant count in
\([2000, 8000]\), then the alphabetically-first assay within each of three taxa.

| assay | taxon | length | variants | sites |
|---|---|---:|---:|---:|
| `R1AB_SARS2_Flynn_2022` | Virus | 306 | 5,725 | 303 |
| `A4GRB6_PSEAI_Chen_2020` | Prokaryote | 266 | 5,004 | 266 |
| `CCR5_HUMAN_Gill_2023` | Human | 352 | 6,137 | 323 |

A protease, a beta-lactamase and a chemokine receptor. Inputs come from
ProteinGym v1.3 pinned by DOI (`10.5281/zenodo.15293562`), so the record behind a
committed result is immutable.

## Reproducibility details worth stating

**Numbering.** ProteinGym mutant strings are 1-based against the target sequence
each assay shipped with, which is not always the UniProt canonical one. Pairing
the numbering with a different sequence shifts every position by a constant and
leaves every score finite and plausible. The reference file carries `target_seq`,
so the sequence is taken from there and never fetched separately, and the
agreement is asserted at preparation time for every variant.

**One optimiser for both supervised rungs.** Rungs 2 and 3 select the best epoch
and stop early through the same implementation, and they now read the same batch
size and learning rate from one place. The ladder's whole claim is that
consecutive rungs differ in exactly one respect, so a patience or improvement
test changed in one loop and not the other would move the measured rung 2 to
rung 3 delta while every test still passed and every number still looked
reasonable.

Sharing the implementation was not sufficient on its own, which is worth
recording because it looked sufficient. Rung 2 batched at 256 with lr 1e-3 while
rung 3 used 8 and 1e-4, and patience is counted in *epochs*: an epoch was 5x to
25x more gradient updates on rung 3, so the two rungs shared a stopping constant
and not a stopping rule. Removing that difference cut the measured delta by about
a quarter. See issue #33 and the 2026-08-02 entries in the
[experiment log](decision-log.md).

Passing identical optimiser settings at each call site was itself only a
coincidence one layer down: `train` and `train_lora` each built their own
`torch.optim.Adam`, so a hyperparameter such as `weight_decay` added to either
trainer alone would have moved the rung 2 to rung 3 delta with every test still
passing. Both rungs now construct their optimiser through one private
`_build_optimizer`, so a setting added there reaches both rungs or neither, and
the property is structural rather than merely tested. See issue #37.

**What still differs, stated plainly.** Rung 2 reads cached embeddings while
rung 3 re-encodes every step, which is inherent to adapting the encoder and
cannot be removed. And the rung-2 head carries no explicit weight decay, only
dropout and early stopping, which leaves it below a ridge regression on the same
features for the mean readout specifically. So "differ in exactly one respect" is
now true of the optimiser and not yet true of the regularisation.

**Subsampling.** Training draws are seeded from a CRC of the configuration rather
than Python's `hash`, which is randomized per process. Seeding from `hash` would
have drawn a different training subset on every invocation while every number
downstream stayed entirely plausible.
