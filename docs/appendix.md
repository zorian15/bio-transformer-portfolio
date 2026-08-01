# Appendix: concepts worth a second pass

The other pages describe what was run and what it found. This one explains the
transformer machinery those pages lean on, for a reader who knows the biology and
the statistics but has not spent much time inside language models.

Nothing here is novel and none of it is specific to this repo. It is collected
because each of these concepts changes what a number *means*, and a reader who
takes them as black boxes cannot tell a sound result from a broken one.

---

## What a protein language model actually returns

ESM-2 is a transformer over amino acids. Feed it a 400-residue protein and it does
**not** hand back one vector describing the protein. It returns **one vector per
residue**, plus two more for the `BOS` and `EOS` bookkeeping tokens. At the 35M
checkpoint each vector is 480 numbers wide, so the raw output is roughly a
400 × 480 matrix.

Every residue's vector is contextualized: attention has already mixed in the rest
of the protein, so position 200 "knows" about position 12. But it remains anchored
on whatever residue sits at position 200.

A downstream head needs **one** vector per example. So something has to collapse
400 vectors into 1, and **that collapse is the readout**. It is not a setting on
the model. It is a choice the caller makes about how to read the model's output,
and this repo makes it an explicit argument
([`embed_sequences`](https://github.com/zorian15/bio-transformer-portfolio/blob/main/src/biotp/embeddings.py))
with no default, so it cannot be chosen by accident.

## Why the readout was invisible in Project 1 and is an axis in Project 2

Project 1 asks where a protein lives in the cell. The whole protein is the unit of
interest, so averaging its residue vectors gives a fair summary of "what kind of
protein is this." Mean pooling was an obvious default that nobody had to think
about.

Project 2 asks how fit a *variant* is, and a variant is a protein identical to the
wild type except at one position:

```
wild type : [v1  v2  v3  ...  v200   ...  v400]  -> mean -> W
variant   : [v1  v2  v3  ...  v200'  ...  v400]  -> mean -> W + small
```

Every variant in an assay maps to almost the same vector. Attention does propagate
a substitution to neighbouring positions, so it is not literally one slot in 400
that changes, but after averaging the perturbation is heavily diluted. The head
must then learn from a low-variance direction buried inside a large constant that
every training example shares.

Three readouts are therefore run, and the choice between them is treated as an
axis of the experiment rather than a knob to tune:

| readout | what it is | why it might win |
|---|---|---|
| `mean` | average over residues | comparable to published supervised baselines; robust to positional idiosyncrasy |
| `at_position` | the vector at the mutated residue | highest signal-to-noise for a single substitution, still globally contextualized |
| `difference_at_position` | \(\text{mutant}[i] - \text{wildtype}[i]\) | cancels the constant wild-type component explicitly instead of asking the head to learn to ignore it |

`at_position` looks obviously best and may not be. Residue vectors carry positional
and local-context flavour, so under position-disjoint splits (below) the head is
asked to generalize to sites whose vectors look unlike anything it trained on, and
pooled vectors may transfer better. Nobody knows in advance, so all three run
everywhere and the interaction is a reported result.

**Selecting a readout on validation and carrying it forward would be worse than
either choice**, because it would tune the frozen arm and leave the fine-tuned arm
untuned, shrinking exactly the difference the experiment exists to measure.

## The embedding cache, and why one arm cannot use it

The encoder's forward pass is the expensive step and the head is nearly free.
Project 1's own numbers: embedding 13,858 proteins took 21 minutes, while the
eighteen head fits took 56 seconds. So the pipeline computes vectors once, writes
them to disk, and iterates on heads indefinitely. See
[embedding-cache.md](embedding-cache.md) for how the key is built and why it
covers the code as well as the inputs.

The three arms of the DMS ladder relate to that cache in three different ways, and
the differences drive the entire compute plan:

| arm | uses the cache? | why |
|---|---|---|
| zero-shot | no | produces one scalar per variant, not a vector; there is nothing embedding-shaped to cache |
| frozen + head | yes, ideally | weights never change, so vectors never change; embed once, then every combination of training-set size, seed and split is a tiny head fit |
| LoRA + head | **cannot** | the weights change every gradient step, so there is no stable output to cache |

That last row is the important one. It is not an optimization that was skipped; it
is ruled out by what fine-tuning *is*. Every epoch re-runs the encoder over every
training variant from scratch, which is why the fine-tuned arm is the whole compute
budget of the benchmark.

## LoRA: adapting a frozen model

Full fine-tuning updates every weight. You need a gradient for each, plus Adam's
two extra copies of optimizer state, so working memory runs about four times the
model size, and each run yields an entirely new multi-gigabyte model.

LoRA freezes every original weight and adds a thin detour beside chosen matrices.
Take one attention projection \(W\), 480 × 480, about 230,000 parameters. Rather
than updating it, factor the *update* into two thin matrices:

\[
h = Wx + \frac{\alpha}{r}\,B\,(A x)
\]

where \(A\) is \(r \times 480\), \(B\) is \(480 \times r\), and the rank \(r\) is
small (4 in this project). That is 3,840 trainable numbers standing in for 230,000.

Three things worth knowing:

**\(B\) is initialized to zero.** At step zero the detour contributes nothing and
the model is bit-for-bit the pretrained one. Training departs from the prior rather
than from some perturbed version of it. This is also why the test suite asserts
that some \(B\) is non-zero after training: if they were all still zero, nothing
had adapted, and a test that only checked "loss went down" would pass against a
completely frozen encoder.

**The low-rank bet** is that useful task-specific adaptation lives in a small
subspace rather than needing all 230,000 degrees of freedom. Empirically this holds
up well.

**Small parameter count is itself a regularizer**, which is what makes a
data-efficiency curve starting at 32 labels meaningful. Full fine-tuning of 35M
parameters on 32 examples would memorize immediately.

!!! warning "LoRA saves memory and storage, not compute"
    A common assumption is that cutting trainable parameters by 100x cuts training
    time by something like 100x. It does not. The backward pass still traverses
    every layer to reach the adapters in the earliest one, so a LoRA step costs
    roughly what a full fine-tuning step costs in floating-point work. What
    collapses is optimizer state, gradient memory, and the size of the artifact you
    ship. Any wall-clock estimate derived from a parameter-count ratio will be
    badly optimistic.

## Masked-marginal scoring: using a model without training it

The zero-shot arm never updates a weight. It exploits what the model was pretrained
to do: fill in a masked position.

Mask residue \(i\) of the wild-type sequence, ask the model for its distribution
over what belongs there, and score a variant by how much probability moves from the
wild-type residue to the mutant one, summed over the positions it mutates:

\[
s(v) = \sum_{i \in M(v)} \Big[ \log p\big(a_i^{\text{mut}} \mid x_{\setminus i}\big)
                             - \log p\big(a_i^{\text{wt}}  \mid x_{\setminus i}\big) \Big]
\]

Read it as a log-odds ratio: positive means the model finds the mutant more
plausible than the residue evolution actually put there. It is a statement about
the model's prior over protein sequences, not a prediction of any particular assay,
which is why results are reported as Spearman correlation within an assay rather
than as a calibrated value.

**The cost structure is the surprising part.** One forward pass is needed per
*distinct mutated position*, not per variant, because every variant touching a site
reads the same distribution. A deep mutational scan of a 300-residue protein needs
about 300 forward passes whether it measured 2,000 variants or 8,000. That is what
makes this arm affordable even at the largest checkpoint, and it is pinned by a
test so an implementation that quietly scaled with the assay would fail rather than
just get slow.

The cheaper alternative, wild-type marginals, takes a single unmasked forward pass
and reads the same distribution off the wild-type logits. It is weaker, and it is
deliberately not implemented, so a number carrying the masked-marginal name cannot
have come from it.

## Position-disjoint splits: why they change the problem

A random split of a deep mutational scan puts some substitutions at residue 200 in
training and others at residue 200 in test. A model can then learn "position 200
tolerates almost nothing" and score well on held-out mutations at that site without
any transferable understanding.

Position-disjoint splits hold out whole *sites*. ProteinGym ships three schemes and
this project reports all three rather than choosing one:

| scheme | held out | what it asks |
|---|---|---|
| `random` | random variants | do labels help, given data at the same sites |
| `modulo` | positions spread across the sequence | does site-level knowledge transfer |
| `contiguous` | a contiguous block of positions | does it transfer to a whole unseen region |

The three are progressively harder, and the *shape* across them is more informative
than any single number. Training may buy a great deal on `random` and little on
`contiguous`, which would say supervision is largely memorizing site-specific
effects rather than learning transferable ones. That is a real and useful finding,
and it is invisible to anyone reporting only a random split.

This is the same principle as Project 1's homology-partitioned protein split and
Project 3's held-out epitopes: the split is chosen so that the easy shortcut is
unavailable. See
[method.md](grounding-multimodal/method.md) for how the shared
`grouped_split` helper enforces it.

**Reporting all three, and saying so before running anything, is what keeps the
result honest.** Choosing the split after seeing which one flatters the conclusion
is the same error as choosing a readout after seeing the results, one level up.
