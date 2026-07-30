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

### 2026-07-30: embedding is 4.8x faster, and it was all padding

- **Question / hypothesis:** embedding 13,858 proteins took 100 minutes at 2.3 seq/s, which dominated the pipeline and would have made the 150M and 650M checkpoints in `PLANNING.md` impractical. Issue #3 proposed three causes: padding waste from unsorted batching, per-sequence device synchronisation, and too small a batch size. Which of them actually costs the time?
- **Setup:** ESM-2 `esm2_t12_35M_UR50D` on Apple MPS, the same 13,858 single-label proteins. Harness: `scripts/benchmark_embeddings.py`, in three modes (phase profile, alternating A/B, throughput). Fast loop is a 1,000-protein sample drawn proportionally per length decile, seed 0, so it matches the full length distribution (sample mean 473.7 against 473.9, median 421 against 421) rather than being an easier subset. Authoritative measure is the `build feature blocks` step in `results/run_manifest_all.json`, before and after.
- **Result:** 4.79x on the sequence-embedding step, and the answer to "which cause" is essentially "one of them".

  **Profile first, before changing anything.** With device syncs separating the phases:

  | phase | seconds (200 proteins) | share |
  |---|---:|---:|
  | tokenize | 0.42 | 0.6% |
  | forward | 71.08 | 99.3% |
  | pool and transfer | 0.08 | 0.1% |

  So candidate 2, per-sequence device synchronisation, was never a real cost: 16 transfers per batch total 0.08s. The hypothesis mistook an accounting artifact for a bottleneck, because an un-synced forward pass hides inside whichever later line first touches its result, which was the `.cpu()` call. Pooling was still vectorised, since it is a few lines and its share grows as the forward gets cheaper, but it is honestly worth ~0.1% here and no more.

  **Padding was the whole story**, because it lives inside the forward pass. At `batch_size=16` over the full cohort, dataset-order batching pushed 13,237,858 padded residue slots through the model against 6,567,584 actual residues, a 2.01x waste, and 3.04x the necessary attention cost. Length-bucketed batching brought padded slots to 6,571,040, which is 1.0005x the data itself.

  **Authoritative comparison**, both runs on MPS with the same checkpoint and cohort, manifests in `results/`:

  | | before | after | speedup |
  |---|---:|---:|---:|
  | sequence embedding | 6,024.2s (2.30 seq/s) | 1,256.6s (11.03 seq/s) | **4.79x** |
  | build feature blocks | 6,169.6s | 1,337.6s | 4.61x |
  | full run, 18 arm-seed fits included | 6,230.9s | 1,380.7s | 4.51x |

  Projected 650M wall time, at the parameter-count ratio and assuming the pipeline stays compute-bound: about 30 hours before, about 6.3 hours after. That is the number that decides whether the larger checkpoints are reachable on this hardware, and it moves from "no" to "overnight".

  **The results are unchanged, which is the point.** The v2 vectors are not bit-identical to v1's, so every arm was re-fit:

  | Arm | Macro-F1 before | Macro-F1 after |
  |---|---:|---:|
  | sequence-only | 0.617 ± 0.008 | 0.616 ± 0.004 |
  | text-only, free text | 0.617 ± 0.015 | 0.617 ± 0.015 |
  | text-only, structured | 0.912 ± 0.001 | 0.912 ± 0.001 |
  | sequence + free text | **0.740 ± 0.007** | **0.740 ± 0.006** |
  | sequence + structured | 0.906 ± 0.006 | 0.906 ± 0.005 |
  | shuffled-text control | 0.583 ± 0.003 | 0.578 ± 0.006 |

  The headline +0.123 gain reproduces to three decimals, and the control still lands below sequence-only. Nothing in the 2026-07-30 entry above needs revising.

- **Two things worth recording, because both cost time to learn:**
  - **Bigger batches are worse here, not better.** Candidate 3 was backwards. `batch_size=64` had to be abandoned after 12 minutes on 1,000 proteins, against 84 seconds at 16, and it drove the machine to 8.3 GB of 9.2 GB swap. On MPS the binding constraint is the attention matrix, not device utilisation: a batch of 64 at 1022 positions materialises roughly 5 GB of attention. Settled on 8, where the medians beat 16 slightly (37.4s against 39.2s per 300 proteins) and, more importantly, three repeats fell within 0.3s of each other while 16 spread over 14s.
  - **Single-shot before/after timing on a laptop is not a measurement.** An identical configuration came out 3.7x apart (83.8s and 306.7s) either side of the run that exhausted swap. The fix was `--mode ab`, which runs the old and new implementations alternately in one process and takes the median, so drift hits both arms. Under alternation the 300-protein speedup is 3.38x at batch 16; the full run reaches 4.79x because bucketing works better the larger the pool, with 13,858 proteins leaving essentially no padding at all. The A/B also asserts the two implementations agree numerically, so a "speedup" that changed the answers would fail rather than be reported.
- **Decision / next step:**
  - Take the 4.79x, and treat the 650M checkpoint as reachable on this hardware.
  - `EMBEDDING_IMPL_VERSION` is 2. Every cache invalidated automatically and recomputed, which is exactly what #4 was built for: the "after" run would otherwise have hit the cache and reported a speedup that measured nothing. `docs/embedding-cache.md` records v2 as the worked example.
  - Correctness is anchored, not assumed: `tests/data/reference_embeddings.npz` holds 24 vectors written by the v1 code before the rewrite, and the v2 code still reproduces them within float32 tolerance.
  - Not done, and deliberately: the per-modality cache version. One `EMBEDDING_IMPL_VERSION` covers both encoders, so this sequence-side change also invalidated the text caches, costing 72 seconds of needless recompute. Splitting it would save that and add a second constant for a human to remember. Over-invalidation is the safe direction, so it stays as is.

### 2026-07-30: MVP six-arm result. Free text helps, and the control holds

- **Question / hypothesis:** does adding text to a frozen sequence representation improve localization over sequence-only, and does the gain survive controls that rule out label leakage? Issue #1.
- **Setup:** DeepLoc 1.0, 13,858 single-label proteins, 10 classes. Frozen ESM-2 `esm2_t12_35M_UR50D` (480-d) and `all-MiniLM-L6-v2` (384-d), concatenated into a 256-unit MLP head, `linear_probe` mode, lr 1e-3, max 200 epochs with patience 10 and best-val-epoch restore. Test is DeepLoc's homology-partitioned split (2,773 proteins), never used for selection. 3 seeds. Provenance: `results/run_manifest_all.json`.
- **Result:** macro-F1 on the held-out test set, mean over seeds, majority floor 0.291 **on the test split** (2,773 proteins). Not to be confused with the 0.292 recorded on 2026-07-29, which is the floor across the whole 13,858-protein dataset; the two are different quantities that happen to land a thousandth apart. The annotated-only test cohort has a different floor again, 0.305.

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
