# When does a fine-tuned protein language model beat the prior it started from?

Project 2 of the [portfolio](https://github.com/zorian15/bio-transformer-portfolio/blob/main/PLANNING.md), tracked as issue
[#11](https://github.com/zorian15/bio-transformer-portfolio/issues/11). See [Data](data.md) for the
inputs, [Method](method.md) for the ladder, and [Results](results.md) for what came out.

## The question

Deep mutational scanning measures the fitness of thousands of single-residue
variants of one protein at once. A protein language model can score those
variants without ever seeing a label, from the likelihood it assigns each
substitution. It can also be trained on some of them and asked to predict the
rest.

The question is where the second thing starts to be worth doing:

> Given a fixed protein language model, how many labelled variants does it take
> before supervision beats the model's own zero-shot prior, and does adapting the
> representation buy anything the frozen one does not already give you?

That is deliberately narrower than "is ESM-2 good at DMS", which the field has
answered many times. The interesting quantity is the *crossover*, and how it
moves when the evaluation stops letting the model see the residue positions it
will be tested on.

## Why this is interesting biologically

Every label in a DMS assay is a wet-lab measurement. A curve that says
"supervision overtakes the prior at roughly N labels" is a statement about how
much bench work buys how much predictive power, which is the form a biologist
can actually act on. "Model A beats model B on this benchmark" is not.

The second reason is that fitness is not a property of a residue in isolation.
An assay measures a protein in one context: one organism, one selection
pressure, one temperature. A model that has learned general protein constraints
and a model that has memorised which sites of *this* protein tolerate mutation
will look identical under a random train/test split, and will diverge the moment
you ask about a site neither has seen. Which of those two a supervised gain
represents is the thing worth knowing, and it is decided by the split rather
than by the model.

## Why this is interesting as a machine-learning problem

The pretrained prior is a strong, free baseline that costs no labels, which is a
setting most benchmark work does not have. That changes what a result means: a
supervised model has to clear a floor that already encodes real biology, not a
floor of chance.

It is also a clean setting for asking what fine-tuning actually does. A frozen
encoder with a trained head and a LoRA-adapted encoder with the *same* head
differ in exactly one respect. If the delta between them is small, that is a
finding about where the useful information already sits, not a failed
experiment.

And the data-efficiency axis makes the comparison say something about regime
rather than ranking. Two methods can trade places as the label count grows, and
the point at which they cross is more informative than either endpoint.

## Objectives

1. Measure the zero-shot prior honestly, with masked marginals rather than the
   cheaper wild-type-marginal approximation, and validate the implementation
   against ProteinGym's published numbers before trusting anything built on it.
2. Measure what supervision buys at a fixed representation: frozen embeddings, a
   small head, across a range of training-set sizes.
3. Measure what adapting the representation buys on top, with LoRA on the same
   encoder feeding the same head.
4. Report all three under three cross-validation schemes, two of which hold out
   residue positions rather than rows, so a gain that depends on having seen the
   test sites is visible as such.

## The three rungs

| rung | what it is | labels used |
|---|---|---|
| 1 | ESM-2 masked-marginal scoring | none |
| 2 | frozen ESM-2 embeddings, MLP head | N per arm |
| 3 | LoRA-adapted ESM-2, same head | N per arm |

Each rung changes exactly one thing from the one below it, which is what makes
the differences attributable. Rungs 2 and 3 share `build_head` and, since issue
[#14](https://github.com/zorian15/bio-transformer-portfolio/issues/14), a single
implementation of early stopping, so they cannot silently diverge on anything
except the one axis under test. See [Method](method.md).

## What counts as an answer

A crossover point, per assay and per split scheme, with the honest caveats
attached. Specifically:

- The label count at which rung 2 overtakes rung 1, or a statement that it does
  not within the range tested.
- The rung-2-to-rung-3 delta, which is the headline: what adapting the
  representation is worth once supervision is already in play.
- How both change between `random` folds and the position-disjoint ones.

A negative or null result is a result here. "Adapting the encoder did not help
at this scale, and here is the evidence" answers the question asked. The
[experiment log](decision-log.md) records the decisions as they were made,
including the ones that turned out to be wrong.

## What this project is not

It is not a leaderboard entry. The cohort is three assays chosen by a
pre-registered filter (see [Data](data.md)), which is enough to ask the question
across taxa and nowhere near enough to rank methods in general.

It is also not a comparison against a biophysical or structure-based baseline,
which the original plan floated. The ladder compares a model against its own
prior and against its own frozen representation, which is a sharper question and
one this cohort can actually support.
