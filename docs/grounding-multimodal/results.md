# Results: does language grounding help?

**Short answer: yes, and the gain survives the control.** Adding free-text
function annotations to a frozen ESM-2 representation improves macro-F1 from
0.617 to 0.740 on held-out proteins, and pairing each protein with a *different*
protein's text destroys the gain, which is what rules out the boring explanation.

The separate structured-annotation arm shows what blatant label leakage looks
like, and it scores far higher still, which is the main reason to read the
headline with care. See [Interpretation](#interpretation).

Produced by the run of 2026-07-30; provenance in
`projects/grounding-multimodal/results/run_manifest_all.json`.

## Headline

2,773 held-out proteins from DeepLoc's homology-partitioned test split. Mean over
3 seeds, standard deviation across seeds. Majority-class accuracy floor: 0.291.
Macro-F1 is the metric to read, since the classes are 26-fold imbalanced.

| Arm | Accuracy | Macro-F1 | Balanced acc. |
|---|---:|---:|---:|
| sequence-only | 0.757 ± 0.003 | 0.617 ± 0.008 | 0.599 ± 0.007 |
| text-only, free text | 0.690 ± 0.005 | 0.617 ± 0.015 | 0.585 ± 0.010 |
| text-only, structured | 0.936 ± 0.001 | 0.912 ± 0.001 | 0.899 ± 0.003 |
| **sequence + free text** | **0.835 ± 0.008** | **0.740 ± 0.007** | **0.716 ± 0.009** |
| sequence + structured | 0.940 ± 0.001 | 0.906 ± 0.006 | 0.893 ± 0.006 |
| shuffled-text control | 0.737 ± 0.003 | 0.583 ± 0.003 | 0.568 ± 0.001 |

Three things to take from this table.

**The grounded arm beats sequence-only by 0.123 macro-F1.** Seed spread is about
0.008, so the gap is roughly fifteen times the noise. This is not a seed
artifact.

**The shuffled control lands *below* sequence-only**, at 0.583 against 0.617.
This is the result that makes the headline mean something. The control keeps each
protein's own sequence but hands it another protein's annotation, so it has
exactly the same 864 input dimensions and exactly the same head capacity as the
grounded arm. If the gain came merely from feeding the model more numbers, the
control would capture it. Instead the extra 384 dimensions of mismatched text are
mild noise and cost a little accuracy. The improvement is therefore tied to *this
protein's* annotation, which is the definition of grounding for this experiment.

**Free text alone matches sequence alone on macro-F1** (0.617 versus 0.617) while
scoring lower on accuracy (0.690 versus 0.757). The two modalities are roughly
comparable in information content but distribute it differently across classes,
which is consistent with them being complementary rather than redundant.

## Where the gain comes from

Per-class F1 on the all-proteins cohort, ordered by sequence-only performance:

| Compartment | Share | sequence-only | sequence+free text | Δ |
|---|---:|---:|---:|---:|
| Extracellular | 14.2% | 0.909 | 0.923 | +0.014 |
| Nucleus | 29.2% | 0.835 | 0.899 | +0.064 |
| Plastid | 5.5% | 0.833 | 0.860 | +0.027 |
| Mitochondrion | 10.9% | 0.789 | 0.879 | +0.090 |
| Cell membrane | 9.7% | 0.778 | 0.809 | +0.031 |
| Cytoplasm | 18.3% | 0.647 | 0.743 | +0.096 |
| Endoplasmic reticulum | 6.2% | 0.549 | 0.685 | +0.136 |
| Lysosome/Vacuole | 2.3% | 0.309 | 0.586 | +0.277 |
| Golgi apparatus | 2.6% | 0.301 | 0.442 | +0.141 |
| Peroxisome | 1.1% | 0.244 | 0.519 | +0.275 |

**The gain is concentrated in the classes the sequence model handles worst**, and
those are largely the rare ones. Peroxisome and Lysosome/Vacuole roughly double.
Extracellular, where sequence already scores 0.909 because secretion signals are
blatant in the N-terminus, gains almost nothing.

This is a coherent story rather than a curiosity. Rare compartments give a
sequence model few examples to learn their targeting motifs from, while a curator
writing a functional description will still say what the protein does and where it
acts. Text is most useful exactly where sequence evidence is thinnest.

The shuffled control reinforces it from the other side: its worst damage is also
in the rare classes, dropping Lysosome/Vacuole from 0.309 to 0.143. Rare classes
have the least signal to spare, so they are the most sensitive both to real
information and to noise.

## The annotation-coverage question

8.9% of proteins have no free-text function annotation and receive a zero vector,
which raised the question of whether the headline should be reported on all
proteins or only on annotated ones. Both were run:

| Cohort | n test | sequence-only | sequence+free text | Δ |
|---|---:|---:|---:|---:|
| All proteins | 2,773 | 0.617 | 0.740 | +0.123 |
| Annotated only | 2,534 | 0.623 | 0.749 | +0.126 |

**The choice does not matter.** The gain is 0.123 on one cohort and 0.126 on the
other, well inside seed noise. Reporting the all-proteins number is therefore
honest and loses nothing.

One difference is worth noting: text-only rises from 0.617 to 0.664 when
un-annotated proteins are excluded, which is expected, since on the full cohort
that arm is asked to classify 239 test proteins from a zero vector. That it
affects the text-only arm but not the combined arm suggests the head learns to
lean on sequence when text is uninformative.

## Interpretation

The narrow claim is supported: **adding curated function text to a frozen
protein-sequence representation measurably improves subcellular localization, and
the improvement is specific to the protein's own text rather than an artifact of
extra input dimensions.**

The broader claim, that this constitutes grounding in biological knowledge, is
**not yet established**, and the structured arm is why.

GO terms and keywords score 0.912 on their own, close to a ceiling. Those fields
frequently contain the label verbatim: `Q9H400` is labelled Cell membrane and its
keywords literally include "Cell membrane". So that arm mostly measures the
model's ability to read an answer key, and it usefully bounds what pure leakage
looks like on this task: about 0.91.

Free text sits at 0.740, between sequence-only at 0.617 and the leakage ceiling at
0.912. That position is consistent with two different stories that this experiment
cannot yet separate:

1. Function prose carries genuine functional information that complements
   sequence, or
2. Function prose sometimes states the localization outright, and the arm is
   partially reading the answer, just less reliably than the structured fields.

Story 2 is entirely plausible. Curated descriptions routinely mention compartment
in passing: the example quoted in [data.md](data.md) contains "across cell
membrane" for a protein labelled Cell membrane. The per-class pattern does not
settle it either, since a curator is arguably *more* likely to state the location
explicitly for an unusual compartment like peroxisome, which would predict the
same rare-class concentration observed above.

Distinguishing them requires an ablation that removes localization-stating
sentences from the free text and re-runs. Until that is done, the honest summary
is: text helps, the help is specific to the protein, and how much of it is
grounding rather than leakage is unmeasured.

Two smaller observations:

- **Adding sequence to the structured arm does not help.** 0.906 with sequence
  against 0.912 without, a difference within about one standard deviation. Once
  the text contains the answer, the sequence contributes nothing.
- **Every arm clears the 0.291 majority floor comfortably**, so no arm is
  degenerate.

## Limitations

- The free-text arm may contain label leakage, unquantified. This is the main
  caveat and the obvious next experiment.
- One encoder pair (ESM-2 35M, all-MiniLM-L6-v2), one head configuration, one
  learning rate. No hyperparameter search, so these are not best-achievable
  numbers for any arm.
- Fusion is plain concatenation. Contrastive alignment, per `PLANNING.md`, is
  untried.
- The train/validation boundary is not family-grouped, so model selection may be
  mildly optimistic. Test numbers come from DeepLoc's homology-partitioned split
  and are unaffected.
- Text is fed to the encoder as UniProt returns it, including `{ECO:...}`
  evidence codes and a constant `FUNCTION: ` prefix (see [data.md](data.md)).
  Cleaning these could change the free-text arms.

## Cost and reproduction

| Step | Time |
|---|---|
| Embedding 13,858 sequences (ESM-2 35M, MPS) | 6,024s (~100 min) |
| Embedding free text and structured text | 62s + 73s |
| 18 arm-seed fits on cached vectors | ~56s total |
| Full run | 6,231s (~104 min) |
| Re-run on the annotated cohort, all caches hit | **44s** |

The 44-second second run is the caching design working as intended: the expensive
step is paid once and every later question is cheap. Embedding throughput of 2.4
sequences/second is poor and tracked in
[#3](https://github.com/zorian15/bio-transformer-portfolio/issues/3).

```bash
python projects/grounding-multimodal/scripts/prepare_data.py
python projects/grounding-multimodal/scripts/run_arms.py
python projects/grounding-multimodal/scripts/run_arms.py --annotated-only
```
