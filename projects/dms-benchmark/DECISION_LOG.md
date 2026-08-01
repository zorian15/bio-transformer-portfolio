# Decision Log: dms-benchmark

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

### 2026-08-01: the rung-3 estimate was wrong because it never modelled validation

- **Question / hypothesis:** smoke-test one fine-tuned configuration locally, to validate the path before submitting 81 SLURM jobs and to turn #11's ~7 GPU-hour estimate into a measurement.
- **Setup:** `R1AB_SARS2_Flynn_2022`, `fold_modulo_5`, `at_position`, N=128, seed 0, `esm2_t12_35M_UR50D`, LoRA rank 8 alpha 16 on `q_proj`/`v_proj`, lr 1e-4, batch 8, on MPS.
- **Result:** the first attempt was **killed after eleven minutes without finishing**. The cause is not the training set.

  ProteinGym's folds are 975 to 1,271 variants each, and fold 1 is the validation set. The fine-tuned rung re-encodes all of it every epoch, because unlike the frozen rung it has no cache to read from. At N=32 that is validating on **thirty times more data than it trains on**, and validation is roughly 90% of the run. The estimate in #11 counted training passes and ignored validation entirely.

  Validation is now subsampled to **256 variants**, fixed per (assay, scheme) and independent of N and seed so every arm selects its stopping epoch against the same held-out variants.

  **The cap is applied to both supervised rungs, not just the expensive one.** The ladder's whole claim is that its rungs differ in exactly one thing; selecting the stopping epoch on different data would be a second difference. Rung 2 was re-run under the cap, in 37.5s rather than 1201s because the embeddings were already cached, which is the first real demonstration that the assay-level cache does what it was built for.

  **The cap changes nothing that matters.** `A4GRB6` random at N=2048 went 0.734 to 0.733, `R1AB` 0.577 to 0.562, and no qualitative conclusion moves. 256 variants estimate a validation loss perfectly well.

  With the cap, one N=128 configuration takes **441.7s**. Extrapolating over the N curve, where per-epoch encoder work is roughly `3N + 256` forward-equivalents:

  | | estimate |
  |---|---:|
  | per (assay, scheme, readout, seed), full N curve | 1.76 h on MPS |
  | 81 jobs | 143 MPS-hours |
  | on an L40S at 8x to 15x | **10 to 18 GPU-hours** |

  So #11's ~7 GPU-hours was optimistic by roughly a factor of two, and the reason was a cost the estimate never modelled rather than a mis-measured one.

  **A first real data point, on one configuration:**

  | rung | Spearman |
  |---|---:|
  | zero-shot 35M | −0.073 |
  | frozen + head | 0.116 |
  | LoRA + head | **0.179** |

  One point is not a result. It is the first evidence that adapting the encoder buys something the frozen representation does not, on a position-disjoint split where rung 2 was struggling, which is exactly where the ladder's headline lives.

- **Decision / next step:** the rung-3 path works end to end and is affordable. Submit the array in the SLURM PR. Quote 10-18 GPU-hours rather than 7, and say why the earlier number was wrong.

### 2026-08-01: supervision beats the prior only where the test sites were seen, and the readout mattered more than the label count

- **Question / hypothesis:** rung 2 of the #11 ladder. What does supervision buy at a fixed representation, how does it scale with labels, and does the answer survive holding out residue positions rather than rows?
- **Setup:** 324 arms, the full pre-registered grid: 3 assays x 3 CV schemes x 3 readouts x N in {32, 128, 512, 2048} x 3 seeds. Frozen `esm2_t12_35M_UR50D` embeddings into the same `build_head` MLP the fine-tuned rung uses, `linear_probe` mode, lr 1e-3, max 200 epochs with patience 10 and best-val-epoch restore. Fold 0 tests, fold 1 selects, folds 2-4 supply the draws. 1201s total on MPS, most of it the one-off embedding pass at 24-46 seq/s. Artifacts: `results/frozen.csv`, manifest `logs/dms-run-arms-20260801T074617Z.json`, figures in `docs/dms-benchmark/figures/`.

- **Result: the headline is uncomfortable and it is the point of the design.** On `random`, supervision climbs steeply with labels and overtakes zero-shot on all three assays. On the two position-disjoint schemes it mostly does not.

  | assay | scheme | N=2048 | zero-shot 35M |
  |---|---|---:|---:|
  | `A4GRB6` | random | **0.734** | 0.516 |
  | | modulo | 0.357 | **0.488** |
  | | contiguous | 0.024 | **0.237** |
  | `CCR5` | contiguous | 0.241 | **0.395** |
  | `R1AB` | random | **0.577** | −0.026 |

  So 2,048 labels are worth a great deal when you already have data at the sites you care about, and worth little when you do not. A benchmark reporting only a random split shows the first half of that and none of the second. `R1AB` is the exception, winning under every scheme, but only because its prior is at zero and anything clears it.

- **The readout axis paid for itself, decisively.** Mean pooling is worse than `at_position` in **all nine** assay-scheme cells, often by a factor of two, and on `A4GRB6` under `contiguous` it is negative (−0.146 against 0.088).

  | assay | scheme | mean | at_position | difference |
  |---|---|---:|---:|---:|
  | `A4GRB6` | random | 0.548 | **0.855** | 0.800 |
  | | modulo | 0.068 | **0.605** | 0.399 |
  | `R1AB` | random | 0.281 | **0.753** | 0.696 |

  Mean pooling is what the field reaches for by default, and what this pipeline would have used had the parameter been given one. Making the readout a pre-registered axis cost three times the rung-2 compute, which was twenty minutes, and the alternative was reporting roughly half the achievable performance while believing it a property of protein language models. `at_position` wins seven cells and `difference_at_position` two, both under `contiguous`, so there is no single right answer to have selected either.

- **Site coverage, not label count, is what saturates.** The per-arm record of distinct training positions explains the flat curves. On `R1AB`, `modulo` and `contiguous` cap at 181 sites no matter how many labels are drawn, because the training folds contain only that many, while `random` reaches 303. Past N=512 the extra labels are repeat measurements at sites already covered.

  This is why recording it was worth the line of code. Without it, `A4GRB6` under `contiguous` sitting at 0.02 across every label count reads as a broken pipeline rather than as a model that has seen 181 of 266 sites and cannot transfer to the remaining block.

- **One implementation note that mattered more than it looks.** `residue_features` originally embedded only the rows of the current arm, so each of the 324 arms hashed a different item list and took a cache miss. Fixed to embed each assay once per (checkpoint, readout) and index into the matrix. Without it rung 2 would have recomputed embeddings 324 times instead of 9.

- **Decision / next step:** rungs 1 and 2 are reported in `docs/dms-benchmark/results.md`. Rung 3 next, on SLURM. The honest framing for the writeup is already visible: the interesting quantity is not "does fine-tuning win" but "does anything transfer to residues you have no labels for", and at a fixed representation the answer is largely no.

### 2026-08-01: rung 1 reproduces the published benchmark, and the cohort spans three very different priors

- **Question / hypothesis:** does the masked-marginal implementation actually compute what it claims? This is the only external check the design has: ProteinGym publishes per-assay zero-shot Spearman for every ESM-2 size, so rung 1 can be validated against it before anything downstream is trusted. Issue #15.
- **Setup:** ProteinGym v1.3, pinned by DOI `10.5281/zenodo.15293562`. Three assays under the pre-registered filter (single-substitution, target length ≤ 400, single-mutant count in [2000, 8000]), one per taxon by alphabetically-first `DMS_id`: `R1AB_SARS2_Flynn_2022` (virus, 306 aa, 5,725 variants), `A4GRB6_PSEAI_Chen_2020` (prokaryote, 266 aa, 5,004), `CCR5_HUMAN_Gill_2023` (human, 352 aa, 6,137). Masked-marginal scoring on fold 0 of each of the three CV schemes, at `esm2_t12_35M_UR50D` and `esm2_t33_650M_UR50D`. 18 configurations, 341.7s total on MPS. Artifacts: `results/zero_shot.csv`, manifest `logs/dms-run-arms-20260801T073952Z.json`.

- **Result:** the implementation reproduces the published numbers on the `random` scheme, which is the one closest to scoring a whole assay:

  | assay | published 35M | measured | published 650M | measured |
  |---|---:|---:|---:|---:|
  | `A4GRB6_PSEAI_Chen_2020` | 0.528 | 0.516 | 0.738 | 0.725 |
  | `CCR5_HUMAN_Gill_2023` | 0.358 | 0.353 | 0.347 | 0.336 |
  | `R1AB_SARS2_Flynn_2022` | −0.026 | −0.026 | 0.105 | 0.113 |

  The published figures cover the whole assay and these cover a fifth of it, so exact agreement is not expected. Nothing is off by a sign, a scale, or a residue, which is what the check is for.

  **The cohort accidentally spans the whole range of prior strength**, which is more useful than a uniform one would have been. `A4GRB6` is the regime protein LMs are usually presented in: 0.52 at 35M rising to 0.73 at 650M. `CCR5` sits at about 0.35 regardless of size, and 650M is *worse* than 35M at two schemes of three. `R1AB` has no usable prior at either size, −0.03 and 0.11, and ProteinGym's table shows it does not become useful until 3B (0.498).

  **Scale is not monotone.** Nineteen times the parameters helps enormously on one assay, does nothing on another, and is insufficient on the third. That is worth stating because the opposite is easy to assume.

- **Two things worth recording:**
  - **The −0.026 looked like a bug and was not.** First reaction to a negative Spearman was that the scoring was wrong. Checking the published table showed ProteinGym reports exactly −0.026 for that assay at 35M. Worth noting because the instinct to debug a surprising number is right, and the correct first move was to find the external reference rather than start editing code.
  - **This assay stays in the cohort.** Rung 1 at a floor of nothing means the headline for `R1AB` will be "training beat a baseline that was at zero", which is a weaker claim than it sounds. The temptation is to swap the assay. That would be selection on outcome: the assay was chosen by a pre-registered rule before any of this was known, and these numbers are now known. It stays, and the three assays are reported separately rather than averaged so the difference stays visible.

- **Decision / next step:** rung 1 is done and trustworthy. Rung 2 is running. The `R1AB` case is a caution for the writeup, not a reason to change the design.

### 2026-08-01: both open questions closed by contact with the data

- **Question / hypothesis:** issue #11 left two things to be settled by downloading rather than by reasoning: whether ProteinGym's assay CSVs ship cross-validation fold assignments, and whether the length filter leaves a viral assay in the pool.
- **Setup:** ProteinGym v1.3 reference table and archives.
- **Result:**
  - **Folds are not in the assay CSVs.** Those carry only `mutant`, `mutated_sequence`, `DMS_score`, `DMS_score_bin`. The fold columns ship in a separate archive, `cv_folds_singles_substitutions.zip`, whose CSVs are a strict superset with `fold_random_5`, `fold_modulo_5` and `fold_contiguous_5` added. So that archive is the *only* download needed, 13 MB rather than 43, and ProteinGym's own folds are used rather than derived substitutes, which keeps the numbers comparable to published supervised baselines.
  - **The filter leaves 33 of 217 assays**, two of them viral. No relaxation needed.
  - **The numbering trap is not present here**, but the guard stays. The reference file carries `target_seq`, the assay's own reference sequence, so there is no opportunity to pair the wrong one. Verified across all 16,866 variants in the three selected assays: zero wild-type mismatches and zero mutated-sequence mismatches.
  - **Verified the fold schemes do what they claim.** On `R1AB`, folds 0 and 1 share 291 of about 297 residue positions under `random`, and **zero** under both `modulo` and `contiguous`. `make_splits` now asserts that rather than trusting it.
- **Two things that changed the implementation:**
  - **Source switched to Zenodo.** The lab web server presents its certificate chain out of order, and Python's OpenSSL 3.6 declines to build a path through it even with certifi's bundle passed explicitly, while curl's LibreSSL accepts the same chain. Rather than weaken verification or depend on whichever TLS stack the machine has, the inputs come from ProteinGym's Zenodo deposit, which is also better provenance: a DOI names an immutable record. Both files were checked byte-for-byte against the lab host first.
  - **The assay-selection rule was amended before any outcome existed.** The original "one viral plus two non-viral, alphabetically first" returned two *Pseudomonas* enzymes out of three. Amended to alphabetically-first within each taxon, giving virus, prokaryote and human. Recorded on issue #11 with an explicit note that only assay metadata had been examined at the time. That note is what makes the amendment defensible, and it would not have been an hour later.
- **Decision / next step:** data preparation is settled and pinned. `prepare_data.py` runs end to end in 21s.
