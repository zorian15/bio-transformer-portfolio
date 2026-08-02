# Data: assays, numbering, and folds

What the three rungs are computed from, where it came from, and the two things
about it that fail silently if you get them wrong. See
[Introduction](introduction.md) for the question and [Method](method.md) for
what the experiment does with these inputs.

## Source and provenance

[ProteinGym](https://proteingym.org) v1.3, taken from its Zenodo deposit, DOI
[10.5281/zenodo.15293562](https://doi.org/10.5281/zenodo.15293562).

Two files are needed and both come from that record:

| file | what it carries |
|---|---|
| `DMS_substitutions.csv` | the assay-level reference: taxon, target sequence, length, mutant counts |
| `cv_folds_singles_substitutions.zip` | the per-variant tables **including** fold assignments |

Zenodo rather than the lab web server, for two reasons. A DOI names an immutable
record, so "which inputs produced this table" has an answer that outlives a URL;
the lab host also publishes unversioned paths that resolve, which would let the
inputs move under committed results with nothing to see in a diff. Separately,
the lab host presents its certificate chain out of order, which Python's OpenSSL
declines to build a path through even with certifi's bundle passed explicitly,
while curl's LibreSSL accepts it. Working around that would have meant either
weakening verification or depending on whichever TLS stack the machine happens
to have. Both files were checked byte-for-byte against the lab host before the
switch.

The fold archive is a strict superset of the main substitutions archive, so it
is the only download needed for the variant tables.

## Which assays, and why those

The filter was **pre-registered on issue
[#11](https://github.com/zorian15/bio-transformer-portfolio/issues/11) before any
assay was downloaded**. These are the experiment's terms, not tuning knobs:
changing one after seeing a result is precisely the selection bias
pre-registration exists to prevent.

```
target length          <= 400 residues
single mutants         >= 2000 and <= 8000
taxa                   Virus, Prokaryote, Human
within each taxon      the alphabetically-first DMS_id
```

Length bounds cost as well as scope, since attention is quadratic in it.

**One amendment, on 2026-07-31, before any outcome data existed.** The original
rule was "one viral plus two non-viral", which returned two *Pseudomonas*
enzymes out of three and made a generality claim thin for no gain. It became one
assay per taxon across three taxa. The amendment is recorded on issue #11; it is
stated here because a reader assessing whether the cohort was chosen to flatter
a result should not have to go find it.

### The three assays

| assay | taxon | organism | length | variants | positions |
|---|---|---|---:|---:|---:|
| `R1AB_SARS2_Flynn_2022` | Virus | SARS-CoV-2 | 306 | 5,725 | 303 |
| `A4GRB6_PSEAI_Chen_2020` | Prokaryote | *Pseudomonas aeruginosa* | 266 | 5,004 | 266 |
| `CCR5_HUMAN_Gill_2023` | Human | *Homo sapiens* | 352 | 6,137 | 323 |

16,866 variants in total. Three assays is enough to ask whether an effect holds
across taxa and nowhere near enough to rank methods in general, which is why
[Introduction](introduction.md) frames the deliverable as a crossover point
rather than a leaderboard.

## The numbering hazard

ProteinGym mutant strings such as `A24G` are **1-based against the target
sequence that assay shipped with**, which is not always the UniProt canonical
sequence for that protein.

Pairing the numbering with a different sequence shifts every position by a
constant. Nothing raises: every score stays finite, every Spearman stays
plausible, and the model is scored on the wrong residue throughout. This is the
failure mode the whole pipeline is shaped around avoiding.

Two things close it. The reference file carries `target_seq`, so the sequence is
taken from there and never fetched separately. And `prepare_data.py` asserts the
agreement between each mutant's stated wild-type residue and the target sequence
at that position, at preparation time rather than 40 minutes into a run.

The prepared table therefore carries `position` as a **zero-based** index, along
with the parsed `wildtype_aa` and `mutant_aa`, so nothing downstream re-derives
them from the string and nothing has to remember which convention it is in.

## Folds, and why two of the three schemes matter more

ProteinGym ships three five-fold cross-validation schemes, and all three are
reported:

| scheme | how folds are formed | what it tests |
|---|---|---|
| `fold_random_5` | rows at random | performance when the test sites were seen in training |
| `fold_modulo_5` | by residue position, modulo | generalisation to unseen positions |
| `fold_contiguous_5` | by contiguous position blocks | the same, with contiguous held-out regions |

`modulo` and `contiguous` are **position-disjoint by construction**: no residue
position appears in more than one fold. That is the leakage-aware default this
repo uses everywhere, and here it is what separates a model that learned protein
constraints from one that memorised which sites of this particular protein
tolerate mutation. Under `random`, folds share almost every position, so a model
can score well without transferring anything.

`run_arms.py` asserts the disjointness rather than trusting it, because the
scheme name is a string in a CSV column and a mislabelled one would produce a
number that looks like generalisation and is not.

Using ProteinGym's own folds rather than deriving equivalents keeps the numbers
comparable to published supervised baselines. Fold sizes are close to even; for
`R1AB_SARS2_Flynn_2022` under `modulo` they run 1,134 to 1,154 variants.

How the folds are assigned to roles (test, validation, training pool) is a
Method question and lives in [Method](method.md).

## Output table

`data/processed/proteingym_variants.parquet`, one row per variant:

| column | meaning |
|---|---|
| `dms_id` | which assay |
| `mutant` | ProteinGym's mutant string, 1-based |
| `mutated_sequence` | the full variant sequence |
| `DMS_score` | the measured fitness, the regression target |
| `DMS_score_bin` | ProteinGym's binarised version, unused here |
| `fold_random_5`, `fold_modulo_5`, `fold_contiguous_5` | fold index per scheme |
| `position` | **zero-based** residue index, parsed |
| `wildtype_aa`, `mutant_aa` | parsed from the mutant string, checked against the target |

Alongside it, `data/processed/proteingym_assays.json` carries the per-assay
metadata: taxon, target sequence, length, variant and position counts, source
organism, plus the ProteinGym version and DOI and the filter that selected the
cohort. That sidecar is what makes a committed result traceable to the inputs
that produced it without re-running anything.

Both are gitignored, as regenerable artifacts. Raw downloads land in
`data/raw/`, also gitignored.

## Reproducing

```bash
python projects/dms-benchmark/scripts/prepare_data.py
```

Re-running skips the download when the raw file is already present; pass
`--force-download` to refetch. The run writes a log and a manifest under `logs/`
recording the inputs, the filter, the selected assays and the package versions,
per the [run logging](../run-logging.md) convention.

Rung 3 needs only these two prepared files, about 600 KB together: it calls
`load_esm2` directly and never touches the embedding cache, which is what rungs
1 and 2 use. That matters when staging to a cluster; see
[slurm/README.md](https://github.com/zorian15/bio-transformer-portfolio/blob/main/slurm/README.md).
