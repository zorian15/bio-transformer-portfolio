# Decision Log: grounding-multimodal

Chronological record of experiments and the decisions they drove. Newest entries on top. One entry per meaningful run or decision. This log is the raw material for the eventual writeup.

## Entry template
```
### YYYY-MM-DD: <short title>
- **Question / hypothesis:** what this run was meant to answer.
- **Setup:** data, model, config (checkpoint, mode, lr, split).
- **Result:** metric(s), and a pointer to any figure or output.
- **Decision / next step:** what this changes about the plan.
```

---

<!-- newest entries below this line -->

### 2026-07-30: MVP six-arm result. Free text helps, and the control holds

- **Question / hypothesis:** does adding text to a frozen sequence representation improve localization over sequence-only, and does the gain survive controls that rule out label leakage? Issue #1.
- **Setup:** DeepLoc 1.0, 13,858 single-label proteins, 10 classes. Frozen ESM-2 `esm2_t12_35M_UR50D` (480-d) and `all-MiniLM-L6-v2` (384-d), concatenated into a 256-unit MLP head, `linear_probe` mode, lr 1e-3, max 200 epochs with patience 10 and best-val-epoch restore. Test is DeepLoc's homology-partitioned split (2,773 proteins), never used for selection. 3 seeds. Provenance: `results/run_manifest_all.json`.
- **Result:** macro-F1 on the held-out test set, mean over seeds, majority floor 0.291.

  | Arm | Macro-F1 | vs sequence-only |
  |---|---:|---:|
  | sequence-only | 0.617 ± 0.008 | |
  | text-only, free text | 0.617 ± 0.015 | +0.000 |
  | text-only, structured | 0.912 ± 0.001 | +0.295 |
  | sequence + free text | 0.740 ± 0.007 | **+0.123** |
  | sequence + structured | 0.906 ± 0.006 | +0.289 |
  | shuffled-text control | 0.583 ± 0.003 | -0.034 |

  Full tables and per-class breakdown in `docs/grounding-multimodal/results.md` and `results/arms_all.md`.

  - The headline gain is roughly 15x the seed spread, so it is not noise.
  - The shuffled control lands *below* sequence-only. Same 864 dimensions, same head capacity, but text paired with the wrong protein. So the gain is not "more numbers helps"; it is tied to the protein's own annotation.
  - The gain concentrates in the classes sequence handles worst, which are largely the rare ones: Peroxisome 0.244 to 0.519, Lysosome/Vacuole 0.309 to 0.586, while Extracellular moves 0.909 to 0.923.
  - Cohort choice is a non-issue: the gain is +0.123 on all proteins and +0.126 on the annotated-only subset, both inside seed noise. The open question from the previous entry is settled; report the all-proteins number.
  - Adding sequence to the structured arm does not help (0.906 with, 0.912 without), so once the text states the answer the sequence is redundant.
- **Decision / next step:**
  - Report the narrow claim, which is supported: text helps, and the help is specific to this protein's text.
  - Do **not** yet claim grounding. The structured arm bounds blatant leakage at 0.912, and free text at 0.740 sits between that and sequence-only at 0.617. That is equally consistent with "prose carries real functional information" and with "prose sometimes states the location outright". The experiment cannot currently separate them.
  - Next experiment, and the one that decides the writeup's headline: ablate localization-stating sentences from the free text and re-run. If the gain survives, it is grounding; if it collapses toward sequence-only, it was leakage.
  - Blocked-on-nothing follow-ups already filed: embedding throughput (#3, 2.4 seq/s) and the cache key not covering embedding code (#4).
- **Cost:** 6,024s to embed sequences, 6,231s for the full run. The annotated-only re-run took 44s on cache hits, which is the frozen-embedding design paying off exactly as `PLANNING.md` intended.

### 2026-07-29: dataset built, arms and encoders fixed for the MVP

- **Question / hypothesis:** what data and which arms make the "does language grounding help" question answerable, with leakage measured rather than assumed? Tracked in issue #1.
- **Setup:**
  - Task: DeepLoc 1.0 subcellular localization, 10 classes. `deeploc_data.fasta` downloaded 2026-07-29 from `https://services.healthtech.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta`, no license form required.
  - Text: UniProt REST (`rest.uniprot.org/uniprotkb/search`), queried 2026-07-29, fields `accession,cc_function,go_c,go_p,go_f,keyword`.
  - Encoders, both frozen: ESM-2 `esm2_t12_35M_UR50D` (480-d) for sequence, `all-MiniLM-L6-v2` (384-d) for text.
  - Built by `projects/grounding-multimodal/scripts/prepare_data.py` into `data/processed/deeploc_annotated.parquet`.
- **Result:** dataset is usable, and the leakage concern is confirmed rather than hypothetical.
  - 14,004 proteins parsed; 146 dual-localized (`Cytoplasm-Nucleus`) dropped, leaving 13,858 single-label across exactly 10 classes.
  - Official homology-partitioned test split: 2,773. Train/val pool: 11,085.
  - Majority-class accuracy floor is 0.292 (Nucleus), which is the number any arm has to beat to mean anything.
  - Annotation coverage: GO cellular component 99.8%, keywords 99.9%, free-text function only 91.1%.
  - UniProt served 13,973 of 14,004 entries; 68 isoform accessions inherit their parent entry's text, flagged by `annotation_from_parent_entry`.
  - Leakage is explicit in the structured fields: Q9H400's keywords include "Cell membrane" and its GO CC includes "extracellular space", so those fields carry the label verbatim.
  - Embedding path verified against real weights: batch-size invariance and padding isolation both within 7e-07 (float32 noise), and the cache recomputes when either the model or the input set changes.
- **Decision / next step:**
  - Run both grounded arms, free text and structured, instead of picking one. The structured arm is expected to score well for the wrong reason, and that contrast is the cleanest available evidence about leakage.
  - Empty text embeds as a zero vector rather than an encoding of `""`, so 1,232 un-annotated proteins cannot share one distinctive "missing annotation" vector for the head to exploit.
  - Truncate sequences at 1,022 residues, the ESM-2 position limit minus BOS and EOS. Acceptable here because localization signal peptides sit at the N-terminus and survive truncation.
  - Next: implement `biotp.evaluation` and `biotp.training`, then run the six arms and report per-arm test numbers.
- **Open questions:**
  - Arm comparability: 8.9% of proteins have no function text. Report the headline on the annotated subset only, on all proteins with zero vectors, or both? Leaning toward both, since the gap between them is itself informative.
  - Train/val boundary is not family-grouped yet, so validation numbers may be optimistic. Test numbers, the reported ones, use DeepLoc's homology-reduced partition and are unaffected.
