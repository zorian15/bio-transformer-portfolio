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

### 2026-07-31: a seventh of the free-text gain is leakage, and the control is what showed it

- **Question / hypothesis:** the MVP found free text worth +0.124 macro-F1 over sequence-only, with the shuffled control confirming the gain is tied to each protein's own text. That leaves the mechanism open (issue #5): is the prose carrying function (grounding), or naming the compartment (leakage)? Ablate the sentences mentioning any of the ten compartments and re-run.
- **Setup:** new shared module `biotp.text_ablation` (clean, split into sentences, drop by a caller-supplied vocabulary, report what went). The DeepLoc compartment vocabulary lives in `scripts/run_arms.py` as `COMPARTMENT_TERMS`, `ABLATION_LEXICON_VERSION` 1, 85 terms across 10 compartments plus a `localization_language` group, with 8 exclusion phrases. Six arms became twelve: `cleaned` (bookkeeping stripped, nothing ablated), `ablated`, and `random-ablated` (the same *number* of sentences removed per protein, chosen at random), each with and without sequence. Same encoders, head, splits and seeds (0, 1, 2) as the MVP. Both cohorts re-run. Artifacts: `results/arms_all.csv`, `results/ablation_all.json`, `results/run_manifest_all.json`, and the `_annotated` counterparts.
- **Result:** the ablation is a trim, not a corpus deletion. 30.6% of the 12,626 annotated proteins lose at least one sentence, 13.5% of sentences go, 83.6% of characters remain, and the median protein loses nothing. But 900 proteins (7.1%) end up with no text at all.

  | Arm | Macro-F1 | vs sequence-only |
  |---|---:|---:|
  | sequence-only | 0.616 ± 0.004 | |
  | sequence + free text | 0.740 ± 0.006 | +0.124 |
  | sequence + cleaned text | 0.743 ± 0.004 | +0.127 |
  | sequence + random-ablated text | 0.674 ± 0.013 | +0.058 |
  | sequence + ablated text | 0.655 ± 0.013 | +0.039 |

  **Stripping evidence codes does nothing.** 0.743 against 0.740, inside one standard deviation, despite the `{ECO:...}` markers and `FUNCTION: ` prefix being 22% of the corpus by character count. That resolves the open preprocessing question standing in `docs/grounding-multimodal/data.md` since the MVP, and resolves it as a negative.

  **The headline would have been wrong without the length-matched control.** Read against the unfiltered arm, the ablation destroys 69% of the gain, which reads as a damning leakage result. Read against a control removing an equal *number* of randomly chosen sentences from the same proteins, it does not: the random arm falls to 0.674, four fifths of the way down. Decomposing the +0.124: 53% is lost by removing 13.5% of the sentences at all, 15% by those sentences being the ones that name the compartment, and 31% survives both.

  **The leakage component is small but consistent.** 0.019 macro-F1 is smaller than either arm's seed spread, so the standard deviations do not settle it. The per-seed pairing does: ablated sits below the control in all three seeds, by 0.018, 0.013 and 0.026. Seeds share a split and an initialisation stream, so the paired difference resolves far better than the marginal spreads suggest.

- **Two things that changed the answer, both worth recording:**
  - **A punctuation bug hid the confound.** Evidence codes usually follow a sentence-final period, so stripping one leaves an orphan `.` that survives as its own sentence. A first pass without that cleanup reported 0.5% of proteins emptied by the filter. The true figure is 7.1%, a fourteen-fold difference, and the orphan fragments were all of it. It matters because `embed_texts` maps empty text to a zero vector and the emptied population is sharply class-skewed: Mitochondrion 26.4% and Peroxisome 22.4%, against Nucleus 2.6%. The ablated arm would have been handicapped hardest in exactly the rare compartments carrying the gain, and that handicap would have been read as leakage.
  - **Prefer recall in the vocabulary, and pay for it with the control.** A missed synonym leaves the answer in the text and biases toward "grounding" invisibly; an over-removed sentence biases toward "leakage" but shows up as both arms falling together, which is readable. So the vocabulary is deliberately aggressive. `chromatin` is the one term left out against that rule, at 654 mentions: it is a near-perfect Nucleus indicator, but "chromatin remodeling" is exactly the functional prose the hypothesis is about, so it is measured in the sentinel probe (4.6% of surviving texts still carry it) rather than cut.
- **Decision / next step:**
  - The honest headline is now: text helps, a seventh of the help is the prose naming the compartment, and most of the rest is not robust to losing part of the annotation. `docs/grounding-multimodal/results.md` says so, and `docs/grounding-multimodal/ablation.md` documents the filter and its judgement calls.
  - Every previously committed arm reproduced **bit-for-bit** in both cohorts, so the new rows can be read alongside the old. That check is cheap and it is the only thing licensing a combined table.
  - No `EMBEDDING_IMPL_VERSION` bump, deliberately. The filter runs upstream of the encoder and the diff does not touch `biotp/embeddings.py`, so the four new text variants invalidate by their inputs and land in their own cache files. `test_cache_key_is_stable_for_the_recorded_spec` passing unmodified is the evidence.
  - Not resolved: `text-only-ablated` still scores 0.483 against a 0.291 floor, so the filtered prose is far from information-free about localization. This design cannot say how much of that is residual leakage against genuine function signal.
  - Open, and worth a follow-up: three seeds is thin for a *ratio* like "31% of the gain survives". The leakage claim rests on the per-seed sign test rather than the ratio's precision. Five seeds would now cost about a minute, since the sequence vectors are cached.

### 2026-07-30: embedding is 4.8x faster end to end, and padding was the cause

- **Question / hypothesis:** embedding 13,858 proteins took 100 minutes at 2.3 seq/s, which dominated the pipeline and would have made the 150M and 650M checkpoints in `PLANNING.md` impractical. Issue #3 proposed three causes: padding waste from unsorted batching, per-sequence device synchronisation, and too small a batch size. Which of them actually costs the time?
- **Setup:** ESM-2 `esm2_t12_35M_UR50D` on Apple MPS, the same 13,858 single-label proteins. Harness: `scripts/benchmark_embeddings.py`, in four modes (phase profile, alternating A/B, throughput, and writing the reference anchor). Profiling manifests are committed as `results/embedding_profile_v1.json` and `_v2.json`, and the A/B as `results/embedding_benchmark_ab.json`, so the tables below cite artifacts rather than memory. Fast loop is a 1,000-protein sample drawn proportionally per length decile, seed 0, so it matches the full length distribution (sample mean 473.7 against 473.9, median 421 against 421) rather than being an easier subset. Authoritative measure is the `build feature blocks` step in `results/run_manifest_all.json`, before and after.
- **Result:** the sequence-embedding step went from 100 minutes to 21, a 4.79x end-to-end improvement, of which roughly 3x is the code change under controlled measurement and the rest is the batch retune and a larger bucketing pool. The answer to "which of the three candidates" is essentially "one of them".

  **Profile first, before changing anything.** With device syncs separating the phases:

  | phase | seconds (200 proteins) | share |
  |---|---:|---:|
  | tokenize | 0.42 | 0.6% |
  | forward | 71.08 | 99.3% |
  | pool and transfer | 0.08 | 0.1% |

  So candidate 2, per-sequence device synchronisation, was never a real cost: 16 transfers per batch total 0.08s. The hypothesis mistook an accounting artifact for a bottleneck, because an un-synced forward pass hides inside whichever later line first touches its result, which was the `.cpu()` call. Pooling was still vectorised, since it is a few lines and its share grows as the forward gets cheaper, but it is honestly worth ~0.1% here and no more.

  **Padding was the cause**, because it lives inside the forward pass. The waste grows with batch size, since a larger batch is likelier to contain one long sequence, so the figures have to be quoted per batch size:

  | batch size | dataset-order slots | bucketed slots | waste | attention cost |
  |---:|---:|---:|---:|---:|
  | 8 (what the pipeline now runs) | 12,152,018 | 6,571,040 | 1.85x | 2.64x |
  | 16 (what it ran before) | 13,237,858 | 6,575,040 | 2.01x | 3.04x |

  Against 6,567,584 actual residues, bucketing leaves 1.0005x at batch 8: essentially no padding at all.

  **Authoritative comparison**, both runs on MPS with the same checkpoint and cohort, manifests in `results/`:

  | | before | after | ratio |
  |---|---:|---:|---:|
  | sequence embedding | 6,024.2s (2.30 seq/s) | 1,256.6s (11.03 seq/s) | **4.79x** |
  | build feature blocks | 6,169.6s | 1,337.6s | 4.61x |
  | full run, 18 arm-seed fits included | 6,230.9s | 1,380.7s | 4.51x |

  **That 4.79x is the pipeline, not the code change in isolation, and the two should not be confused.** The before run used `EMBED_BATCH_SIZE=16` and the after run uses 8, so it moves two variables. The controlled measurement, holding batch size fixed and alternating the implementations, is **2.92x at batch 8 and 3.38x at batch 16** on 300 proteins. The remainder comes from two places: the batch retune, which helped v1 too (109.2s against 132.6s per 300 proteins), and pool size, since bucketing has more similar-length neighbours to work with over 13,858 proteins than over 300. What is fair to claim is that the step now takes 21 minutes instead of 100, and that padding is why.

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

  The headline gain is +0.124 (0.7400 against 0.6157), against +0.123 before; the change is in the fourth decimal and the rounding tipped. The control still lands below sequence-only, and the annotated-only cohort was re-run too, giving +0.130 there against +0.126 before. Nothing in the 2026-07-30 entry above needs revising.

  **Aggregates are stable, rare classes are not.** Sequence-only Peroxisome F1 went 0.244 to 0.167 between two runs whose only difference is the embedding code path. That class has about 30 test proteins, so single-protein flips move it by 0.03 or more, and the same instability is why its "roughly doubles with text" claim is now "roughly triples" (0.167 to 0.519). Read the per-class numbers in `docs/grounding-multimodal/results.md` as noisy at that resolution; the macro-F1 aggregates, which average over ten classes, moved by 0.001.

- **Two things worth recording, because both cost time to learn:**
  - **Bigger batches are worse here, not better.** Candidate 3 was backwards. `batch_size=64` had to be abandoned after 12 minutes on 1,000 proteins, against 84 seconds at 16, and it drove the machine to 8.3 GB of 9.2 GB swap. On MPS the binding constraint is the attention matrix, not device utilisation: a batch of 64 at 1022 positions materialises roughly 5 GB of attention.

    Settled on 8, and the reason is peak memory rather than the stopwatch. Peak attention scales as batch x heads x positions squared, so 8 halves the worst case against 16, which is arithmetic rather than a measurement and does not care how warm the machine was. The timing evidence is weaker than it first looked: 8 did post a better median (37.4s against 39.2s per 300 proteins) with three repeats inside 0.3s while 16 spread over 14s, but the two batch sizes were measured in sequence rather than interleaved, and 16 ran first during the thermal ramp. Batch 16's best round (32.5s) beats every batch-8 round. So the honest reading is that 8 and 16 are within noise of each other on speed, and 8 wins on headroom.
  - **Single-shot before/after timing on a laptop is not a measurement.** An identical configuration came out 3.7x apart (83.8s and 306.7s) either side of the run that exhausted swap. An early draft of this entry reported 4.72x on the fast loop off a baseline taken while the machine was degraded. The fix was `--mode ab`, which runs the old and new implementations against the same machine state, flipping which goes first each repeat, and takes the median. It also asserts the two implementations agree numerically, so a "speedup" that changed the answers fails rather than getting reported. Anything measured by a single before/after pair separated by minutes, including the batch-size comparison above, should be read as indicative and not much more.
- **Decision / next step:**
  - Take the 4.79x, and treat the 650M checkpoint as reachable on this hardware.
  - `EMBEDDING_IMPL_VERSION` is 2. Every cache invalidated automatically and recomputed, which is exactly what #4 was built for: the "after" run would otherwise have hit the cache and reported a speedup that measured nothing. `docs/embedding-cache.md` records v2 as the worked example.
  - Correctness is anchored, not assumed: `tests/data/reference_embeddings.npz` holds 24 vectors written by the v1 code before the rewrite, and the v2 code still reproduces them within float32 tolerance.
  - Both cohorts were re-run, `_all` and `_annotated`, so no committed metric is left describing v1 vectors. Wall-clock configuration now lands in the manifest too (`embed_batch_size`, `text_batch_size`), because this entry's own headline showed how easily a run that changed two things reads from the artifacts as a run that changed one.
  - Open gap, filed as a follow-up: the reference anchor is `@pytest.mark.network` and there is no CI, so the v1-versus-v2 numerical check runs only when someone remembers `pytest -m network`. That is the test which makes the version bump defensible, and it should not depend on memory. **Closed 2026-07-31 (issue #8):** `.github/workflows/embedding-anchor.yml` now runs the network suite on every pull request touching the embedding path, and weekly. The sentence above is left standing rather than rewritten, because it records what was true on the day; this note says what changed since.
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
