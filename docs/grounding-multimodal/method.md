# Method: splits, head, and the six-arm runner

**Status:** implemented and tested; results pending. See
[introduction.md](introduction.md) for the question and
[data.md](data.md) for the inputs.

This page describes the machinery: how splits are made, what the trained model
actually is, and how the six arms are run and compared. All of it is shared
infrastructure in the `biotp` package except the runner, which is
project-specific.

## Splitting: `biotp.evaluation.grouped_split`

Random row splits are the standard way to get an optimistic protein result.
Protein families are large and internally similar, so a random split lets a model
recognize a close relative it already saw in training and score well without
generalizing at all.

`grouped_split` takes a `group_key` callable and guarantees no group spans two
splits. The group is whatever leakage unit the task has: protein family here,
epitope for the TCR project, donor for repertoire data.

Three properties it enforces, each with a test:

- **Groups stay disjoint.** This is the entire purpose.
- **Every record lands in exactly one split.** No silent dropping.
- **The result depends on the seed, not on input order.** Groups are sorted into a
  canonical order before shuffling, so re-ordering the input list cannot change a
  seeded split.

Fractions must sum to 1.0, asserted rather than renormalized, so a typo becomes an
error instead of a slightly different experiment. Exact proportions are
unreachable with grouped data since whole groups move together: the realized sizes
approximate the request, and callers should read them rather than assume. Asking
for a split that the available groups cannot fill also fails loudly, rather than
returning an empty split that would look like a successful run.

**How this project uses it.** The test split is DeepLoc's own, inherited rather
than re-derived. Train and validation come from the remaining pool, grouped on
accession, which makes that particular split effectively random. That is the
documented limitation from [data.md](data.md): the train/validation boundary is
not family-grouped yet. It is used only for model selection, and the reported
numbers come from the homology-partitioned test set, which is unaffected.

## The trained model: `biotp.training`

Both encoders are frozen, so the only trained component is a small head over
concatenated embeddings:

```
concatenated embedding -> Linear(dim, 256) -> ReLU -> Dropout(0.1) -> Linear(256, 10)
```

The head is deliberately small. With frozen features and 11,085 training proteins,
a larger head would mostly buy the capacity to memorize. Neither the hidden width
nor the dropout rate is tuned; if a conclusion turns on either, that is a finding
for the log rather than a knob to quietly adjust.

Three design choices worth naming:

**The head carries its own task.** `build_head` attaches a `task` attribute, and
`train` reads the loss from the head rather than from a separate argument. Cross
entropy and MSE cannot end up paired with the wrong output shape, because there is
only one place the task is recorded.

**Model selection restores the best epoch, not the last.** Validation loss is
checked every epoch, the best weights are kept, and early stopping fires after 10
epochs without improvement. A run that overfits late is therefore reported at its
best point rather than its final one, and `max_epochs` becomes a time budget
rather than a hyperparameter that silently changes results.

**Unimplemented modes raise.** `train` accepts `linear_probe`, `lora`, or `full`,
and only `linear_probe` exists today. The other two raise `NotImplementedError`
rather than falling back, so a call asking for LoRA cannot receive a linear probe
under LoRA's name. `mode` has no default, so every call site states its regime.

`predict` is separate from `train` so evaluation cannot accidentally run with
dropout active or gradients enabled.

## Metrics: `biotp.evaluation`

`classification_metrics` returns accuracy, balanced accuracy, and
macro- or micro-averaged precision, recall, and F1. `per_class_f1` gives the
per-compartment breakdown, and `majority_class_accuracy` gives the floor.

### Definitions

Take \(N\) test proteins, true labels \(y_i\) and predictions \(\hat{y}_i\) drawn
from \(C = 10\) compartments. For a class \(c\), count the true positives, false
positives and false negatives:

\[
\mathrm{TP}_c = \sum_{i=1}^{N} \mathbb{1}[y_i = c \wedge \hat{y}_i = c],
\quad
\mathrm{FP}_c = \sum_{i=1}^{N} \mathbb{1}[y_i \neq c \wedge \hat{y}_i = c],
\quad
\mathrm{FN}_c = \sum_{i=1}^{N} \mathbb{1}[y_i = c \wedge \hat{y}_i \neq c].
\]

**Accuracy** is the fraction of proteins placed in the right compartment:

\[
\mathrm{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i = \hat{y}_i].
\]

**Per-class F1** is the harmonic mean of precision and recall for that class:

\[
P_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c},
\qquad
R_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FN}_c},
\qquad
F1_c = 2 \cdot \frac{P_c \cdot R_c}{P_c + R_c}.
\]

**Macro-F1**, the headline metric, averages those per-class scores with equal
weight per class rather than per protein:

\[
\text{macro-}F1 = \frac{1}{C} \sum_{c=1}^{C} F1_c.
\]

That unweighted average is the whole point. The classes are imbalanced 26-fold
(Nucleus 29.2% against Peroxisome 1.1%), so a protein-weighted average would let
a model that ignores the rare compartments entirely still score well. Under
macro-F1, Peroxisome and Nucleus each carry \(1/10\) of the score, and failing a
rare class costs as much as failing a common one.

**Balanced accuracy** is the same rebalancing applied to recall alone, which
makes it macro-recall:

\[
\mathrm{BalAcc} = \frac{1}{C} \sum_{c=1}^{C} R_c.
\]

**The majority-class floor** is what a model scores by always predicting the most
common compartment, and is the number every arm has to beat to mean anything:

\[
\mathrm{Acc}_{\text{majority}} = \frac{1}{N} \max_{c} \sum_{i=1}^{N} \mathbb{1}[y_i = c].
\]

On this test split that is 0.291 (Nucleus). Note it is an *accuracy* floor, not a
macro-F1 floor: a constant predictor scores macro-F1 \(\approx 0.045\), since it
earns a nonzero \(F1_c\) on one class out of ten.

Where a metric carries \(\pm\), it is the standard deviation across the three
seeds, not a confidence interval over proteins. It measures pipeline sensitivity
to initialization and split, which is the quantity that decides whether an arm
gap is real.

Two deliberate omissions and one substitution:

- **AUROC is absent.** It needs predicted scores, and these functions take hard
  labels. Computing it here would mean inventing the scores it needs.
- **Balanced accuracy stands in** as the rare-class-sensitive companion to
  accuracy that hard labels do support.
- **Undefined per-class scores count as zero** rather than raising. With a class
  holding 1.1% of the data, "never predicted" is an expected outcome to measure,
  not a crash.

## The six-arm runner

`projects/grounding-multimodal/scripts/run_arms.py` is the experiment itself.

Its job is to make the six arms differ in exactly one respect: which feature
blocks the head is allowed to see. Everything else is deliberately shared.

**Three feature blocks are embedded once**, cached to disk, and reused by every
arm and every seed:

| Block | Encoder | Width |
|---|---|---:|
| `sequence` | ESM-2 35M | 480 |
| `text_free` | MiniLM over function text | 384 |
| `text_structured` | MiniLM over GO terms and keywords | 384 |

Because the cache is keyed on the inputs, the encoder name, and the embedding code
itself (see [embedding cache](../embedding-cache.md)), re-running to change the
head or add an arm costs seconds. Only the first run pays for encoding, and a
change to how the vectors are computed invalidates the cache rather than silently
reusing it.

**Each arm selects blocks and concatenates them**, so the head's input width is
480, 384, or 864 depending on the arm. The shuffled-text control is the one arm
that transforms a block: it permutes the text rows across proteins, so every
protein keeps its own sequence but receives a different protein's annotation. The
permutation is seeded and applied to the whole dataset rather than within a split,
because the thing being broken is the sequence-to-text pairing itself.

**Every arm runs at three seeds** (0, 1, 2). The seed drives head initialization,
minibatch order, the train/validation split, and the control's permutation, so the
spread across seeds measures the whole pipeline's sensitivity rather than just
initialization. Results report mean and standard deviation, and a gap between arms
smaller than that spread is not a win.

**Fair-comparison invariants**, each of which would quietly distort a comparison
if violated:

- One head architecture, one learning rate, one epoch budget, one early-stopping
  rule for all arms.
- Identical train/validation/test row indices across arms within a seed.
- Embeddings computed once and shared, so no arm gets a differently-encoded view
  of the same protein.
- The test split is untouched during training and model selection.

**Outputs** land in `projects/grounding-multimodal/results/`: a CSV of per-arm
aggregates, a Markdown table with the majority-class floor stated alongside, and
per-class F1 as JSON. These are small text files and are committed, so the numbers
in the writeup are traceable to a specific run.

## Reproducing

```bash
mamba activate biollm
python projects/grounding-multimodal/scripts/prepare_data.py   # once
python projects/grounding-multimodal/scripts/run_arms.py       # all proteins
python projects/grounding-multimodal/scripts/run_arms.py --annotated-only
```

The second invocation restricts to the 91.1% of proteins that have free-text
function annotation, which is the other half of the comparability question raised
in [data.md](data.md). It subsets after embedding, so it reuses the same cached
vectors rather than recomputing them.

Both scripts log to `logs/` and write a run manifest recording the git commit,
device, package versions, per-step timings, and the counts and metrics they
produced. The runner also copies its manifest into `results/`, so a committed
number can be traced to the run that produced it. See
[run-logging.md](../run-logging.md).
