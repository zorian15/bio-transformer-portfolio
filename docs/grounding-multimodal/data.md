# Data: inputs, labels, and splits

Everything below is produced by
[`projects/grounding-multimodal/scripts/prepare_data.py`](../../projects/grounding-multimodal/scripts/prepare_data.py),
which writes a single table to `data/processed/deeploc_annotated.parquet`, one row
per protein. Raw downloads and processed tables are gitignored; the script
regenerates them. Counts on this page come from the build of 2026-07-29.

## Labels: where the ground truth comes from

Labels are the subcellular compartment assignments from **DeepLoc 1.0**, a
published benchmark whose labels derive from curated UniProt subcellular location
annotations. Using an existing benchmark rather than re-deriving labels from
UniProt means the label definitions and the test partition are someone else's
published choices, which makes the numbers comparable to prior work.

Source: `https://services.healthtech.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta`,
downloaded 2026-07-29. It needs no license form.

Each FASTA header encodes the accession, the compartment, a membrane or soluble
code, and whether the protein belongs to the official test partition:

```
>Q9H400 Cell.membrane-M test
>Q5I0E9 Cytoplasm-S
```

The ten compartments and their frequencies after preprocessing:

| Compartment | Proteins | Share |
|---|---:|---:|
| Nucleus | 4,043 | 29.2% |
| Cytoplasm | 2,542 | 18.3% |
| Extracellular | 1,973 | 14.2% |
| Mitochondrion | 1,510 | 10.9% |
| Cell membrane | 1,340 | 9.7% |
| Endoplasmic reticulum | 862 | 6.2% |
| Plastid | 757 | 5.5% |
| Golgi apparatus | 356 | 2.6% |
| Lysosome/Vacuole | 321 | 2.3% |
| Peroxisome | 154 | 1.1% |

The imbalance is roughly 26-fold between the largest and smallest class, which is
why macro-F1 rather than accuracy carries the weight in evaluation.

### One preprocessing decision worth stating

Of 14,004 proteins in the file, 146 are annotated in both cytoplasm and nucleus
(`Cytoplasm-Nucleus`). These are dropped, leaving **13,858 single-label
proteins**. Dropping them keeps the task honestly single-label rather than
quietly forcing dual-localized proteins into one compartment. It costs 1.0% of
the data, and the count is reported by the build script rather than absorbed
silently. Treating localization as multi-label is a legitimate alternative and a
possible follow-up.

## Inputs

Each protein has one sequence input and several text inputs. All encoders are
frozen: they are used as fixed feature extractors, and only a small head is
trained. This keeps every experiment cheap enough to iterate on a laptop and
means differences between arms come from the inputs, not from different amounts
of training.

### Sequence

The amino-acid sequence, from the same DeepLoc FASTA, encoded with **ESM-2**
(`esm2_t12_35M_UR50D`, 480-dimensional) and mean-pooled over residues into one
vector per protein.

Two details that affect correctness:

- Pooling covers residue positions only. The BOS and EOS tokens and any batch
  padding are excluded, so a short protein embedded alongside long ones gets the
  same vector as it would alone. This is verified in the test suite to within
  float32 noise.
- Sequences longer than **1,022 residues** are truncated, that being ESM-2's
  1,024-position limit minus the two special tokens. Truncation keeps the
  N-terminus, which is where signal and transit peptides sit, so the most
  localization-relevant region survives.

The 35M checkpoint is a deliberate choice for iteration speed rather than peak
accuracy. Larger checkpoints are a later step, and because embeddings are cached
on disk, swapping one in does not change any downstream code.

### Text

Text comes from the **UniProt REST API** (`rest.uniprot.org/uniprotkb/search`),
queried 2026-07-29 for the fields `cc_function`, `go_c`, `go_p`, `go_f`, and
`keyword`. UniProt served 13,973 of the 14,004 requested accessions; the
remainder are entries it no longer serves.

Two text conditions are used, and they behave very differently:

**Free-text function description** (`cc_function`), a curator-written prose
summary. This is the condition the research question is really about. A complete
example, exactly as UniProt returns it (`P93004`):

> FUNCTION: Water channel required to facilitate the transport of water across
> cell membrane. May be involved in the osmoregulation in plants under high
> osmotic stress such as under a high salt condition.
> {ECO:0000269|PubMed:10102577, ECO:0000269|PubMed:9276952}

**Structured annotation terms**, GO terms plus keywords. These are controlled
vocabulary rather than prose, and they frequently contain the label verbatim. For
`Q9H400`, the keywords include `Cell membrane` and `Membrane`, and the GO
cellular-component field includes `extracellular space`. This condition is
included precisely because it is leaky: it bounds how well a model can do by
reading the answer, which is the reference the free-text arm is interpreted
against.

Both are encoded with `all-MiniLM-L6-v2` (384-dimensional) and combined with the
sequence embedding by concatenation.

### An unresolved preprocessing question: evidence codes

The example above shows something that affects 96.0% of the function
descriptions: UniProt appends provenance markers such as
`{ECO:0000269|PubMed:10102577}` to the prose. They average 14% of each field's
character content.

These codes are database bookkeeping, not biology. They say which experiment
supports a claim, and no two proteins share them in a way a sentence encoder
could use. They are also not harmless filler: `all-MiniLM-L6-v2` truncates its
input, so evidence codes displace real text on longer descriptions. Every
description also begins with the literal prefix `FUNCTION: `, which is constant
across all 12,626 proteins and therefore carries no information.

Whether to strip these before encoding is not yet decided, and the current
pipeline leaves the text as UniProt returns it. Stripping is very likely correct,
but it changes every text embedding, so it belongs in a deliberate rebuild with
the before-and-after numbers recorded rather than as a quiet edit.

### Coverage, and why it matters

| Field | Non-empty | Share |
|---|---:|---:|
| GO cellular component | 13,824 | 99.8% |
| Keywords | 13,843 | 99.9% |
| Free-text function | 12,626 | 91.1% |

The gap matters. 1,232 proteins have no curated function text, so the free-text
arm has nothing to ground on for 8.9% of the data, while the structured arm has
near-complete coverage. Missingness is not random: proteins without function text
are generally less-studied ones.

Those proteins are given an explicit **zero vector** rather than an encoding of
the empty string. Encoding `""` would hand every un-annotated protein the same
distinctive non-zero vector, which a head can learn as a "this protein is
obscure" flag. That would be a confound dressed up as grounding.

This leaves an open question, recorded rather than resolved: whether to report the
headline comparison on the 91.1% with annotations, on all proteins with zeros, or
both. The gap between those two numbers is itself informative about how much the
grounded arm depends on annotation availability.

### Isoforms

68 accessions are isoforms, such as `P22462-2`. Queried directly, UniProt returns
these with empty annotation fields, because the text lives on the parent entry.
They inherit the parent entry's text, and the column
`annotation_from_parent_entry` marks them so they can be excluded in one filter.
The caveat: two isoforms of one gene can have different localization labels but
identical inherited text, giving the text arm contradictory supervision for those
rows. At 0.5% of the data this is noted rather than fixed.

## Splits

| Split | Proteins | Source |
|---|---:|---|
| Test | 2,773 | DeepLoc's official test partition |
| Train + validation pool | 11,085 | The remainder |

**Test.** DeepLoc's authors partitioned their data by homology rather than at
random, so a test protein is not a close relative of a training protein. This
matters more than it might seem: protein families are large and internally
similar, so a random split lets a model recognize a family member it has already
seen and score well without generalizing. Because the partition is inherited from
the benchmark, the property is built in rather than something asserted here. The
exact clustering procedure and identity threshold are described in the DeepLoc
publication.

The test set is used once, for final reported numbers, and never for model
selection.

**Train and validation.** Carved from the remaining 11,085 proteins.

**A known limitation, stated plainly:** this train/validation boundary is not yet
family-grouped. Related proteins can therefore land on both sides of it, which
makes validation numbers optimistic and unsuitable for claims about
generalization. They are used only for model selection, and reported numbers come
from the homology-partitioned test set, which is unaffected. Grouping the
train/validation split by sequence-similarity cluster is a planned follow-up.

## Output table

One row per protein, with the columns each arm needs:

| Column | Description |
|---|---|
| `accession` | UniProt accession, including isoform suffix where present |
| `sequence` | Amino-acid sequence |
| `localization` | One of the ten compartments, the classification target |
| `solubility` | `M`, `S`, or `U`; carried along, unused by this task |
| `is_test` | Whether the protein is in DeepLoc's official test partition |
| `function_text` | Free-text function description |
| `go_cellular_component`, `go_biological_process`, `go_molecular_function` | GO terms |
| `keywords` | UniProt keywords |
| `entry_accession` | Parent accession used for the annotation lookup |
| `annotation_from_parent_entry` | Whether text was inherited from a parent entry |
| `has_function_text` | Whether free-text function is present |

## Reproducing

```bash
mamba activate biollm
python projects/grounding-multimodal/scripts/prepare_data.py
```

The DeepLoc download is cached after the first run; pass `--force-download` to
refetch, or `--skip-uniprot` to parse the FASTA without querying UniProt. The
script prints every count on this page, so a rebuild that disagrees with these
numbers is visible immediately. UniProt is updated continuously, so annotation
coverage will drift over time even though the DeepLoc labels are fixed.
