# Does language grounding help protein representations?

**Status:** data pipeline and frozen embeddings are built and tested; head training
and evaluation are in progress (issue #1). No results yet.

## The question

Protein language models learn from amino-acid sequence alone. Curated databases
also describe proteins in natural language: what a protein does, which processes
it participates in, which compartment it occupies. If that text carries
information the sequence does not, then a representation combining both should
outperform a sequence-only one.

This project tests that on subcellular localization, and treats one specific
failure mode as the main event rather than a footnote: the annotation text may
simply restate the label. A model that reads "Cell membrane" from a keyword list
and predicts "Cell membrane" has learned nothing about proteins.

So the question is two-part:

1. Does adding text to a protein-sequence representation improve localization
   prediction over sequence alone?
2. Does any improvement survive controls designed to rule out label leakage?

## Why this is interesting biologically

A protein's function depends on where in the cell it acts. A kinase in the
nucleus and a kinase at the plasma membrane participate in different biology,
and mislocalization is itself a disease mechanism. Localization is therefore a
genuine functional property, not an arbitrary label.

It is also a property that sequence plausibly determines. Proteins carry
targeting information in their sequence: N-terminal signal peptides route
proteins into the secretory pathway, mitochondrial and plastid transit peptides
target those organelles, and nuclear localization signals are short basic motifs.
Hydrophobic stretches indicate membrane insertion. A sequence model has real
signal to work with, which makes sequence-only a strong baseline rather than a
strawman.

The interesting part is what text could add. Curated function descriptions
summarize decades of experimental work, including evidence that leaves no trace
in the sequence: interaction partners, pathway membership, conditional behavior.
Whether a small encoder can extract any of that, and whether it helps beyond
sequence, is an open empirical question.

## Why this is interesting as a machine-learning problem

Framed plainly, this is 10-class single-label classification over about 14,000
proteins, with two frozen pretrained encoders and a small trained head. Three
things make it less routine than that sounds.

**Multimodal fusion with mismatched modalities.** The sequence encoder (ESM-2)
and the text encoder (a sentence transformer) were pretrained on unrelated
corpora with unrelated objectives. Their embedding spaces are not aligned, so
naive concatenation is the honest starting point and any gain has to come from
the head learning to use both.

**Label leakage as the central confound.** The text is not an independent
observation of the protein; it is a human-written summary produced by curators
who knew the localization. Some annotation fields state the answer outright. This
is not a defect to be engineered away, it is the thing worth measuring: how much
of an apparent multimodal gain is real grounding, and how much is the label
leaking through the text channel?

**Missing data that is not missing at random.** Well-studied proteins have rich
annotations; obscure ones have little or none. Annotation richness therefore
correlates with how much is known about a protein, which may correlate with
label difficulty. Any comparison across arms has to account for this.

## Objectives

1. Establish an honest sequence-only baseline on held-out proteins.
2. Measure whether adding text improves on it.
3. Distinguish real grounding from label leakage using controls, and quantify the
   difference between free-text annotations and structured annotation terms.

A null result satisfies these objectives. "Text did not help beyond sequence, and
here is the evidence" is a valid outcome and will be reported as one.

## Arms and controls

Six conditions, all sharing the same frozen encoders, head architecture, and
splits, so differences are attributable to the inputs:

| Arm | Inputs | Purpose |
|---|---|---|
| sequence-only | ESM-2 embedding | The baseline to beat |
| text-only, free text | function description | How much does text alone explain? |
| text-only, structured | GO terms and keywords | Leakage upper bound |
| sequence + free text | both, concatenated | The headline comparison |
| sequence + structured | both, concatenated | Same, with leaky text |
| shuffled-text control | sequence + text from a random other protein | Detects gains that are not about this protein's text |

The controls are what make the headline interpretable. If sequence+text beats
sequence-only, the shuffled-text arm says whether the gain came from this
protein's annotation or merely from adding a well-behaved extra input. If
text-only already performs near the top, the text is doing the work by itself,
which points to leakage rather than complementary information.

## Evaluation criteria

**Split.** Reported numbers come from DeepLoc's official test partition, which
its authors built by homology rather than at random, so test proteins are not
close relatives of training proteins. It is used once, for final numbers, and not
for tuning. See [data.md](data.md) for details and one current limitation in the
train/validation boundary.

**Metrics.** Accuracy plus macro-averaged F1. Macro-F1 carries most of the weight
because the classes are heavily imbalanced: the largest holds 29.2% of proteins
and the smallest 1.1%, so accuracy alone rewards ignoring rare compartments.
Per-class F1 is reported alongside, since a method that helps only for
well-annotated compartments is a different finding from one that helps broadly.

**Reference points.** Two floors keep the numbers interpretable. Predicting the
majority class always gives 0.292 accuracy. The sequence-only arm is the baseline
that the grounded arms have to beat to matter at all.

**What counts as a real effect.** A difference between arms is only reported as
real if it holds across multiple random seeds for the head, since a small gap on
2,773 test proteins is within seed noise. Where an effect is inside that noise,
it will be described that way rather than as a win.
