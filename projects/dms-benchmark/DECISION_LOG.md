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

### 2026-08-02: matching the optimiser removed a quarter of the delta, and moved the problem

- **Question / hypothesis:** the previous entry found rungs 2 and 3 did not differ in exactly one thing: patience is counted in epochs, and rung 2 batched at 256 with lr 1e-3 against rung 3's 8 and 1e-4, so an epoch was 5x to 25x more gradient updates on rung 3. Does the rung-2-to-rung-3 delta survive equalising that? Issue #33.
- **Setup:** `train` now takes a required `batch_size`, and both supervised rungs read one `SUPERVISED_BATCH_SIZE` / `SUPERVISED_LEARNING_RATE` pair, set to rung 3's values because rung 3's batch size is a memory constraint and rung 2's is free. Rung 2 re-run over the full 324-configuration grid, 771s. Rung 3 unchanged and not re-run: its settings did not move. Project 1 reproduces bit-for-bit on both cohorts, which is what licenses reading the two projects' numbers together.
- **Result:** the confound was real and is now smaller, but it is not gone.

  | | mean Spearman over 324 |
  |---|---:|
  | rung 2, old optimiser | 0.2236 |
  | **rung 2, shared optimiser** | **0.2348** |
  | ridge, converged floor | 0.2619 |
  | LoRA | 0.2634 |

  | paired contrast, seed-averaged (n=108) | mean | Wilcoxon p |
  |---|---:|---:|
  | new rung 2 − old rung 2 | +0.0112 | 0.005 |
  | **LoRA − old rung 2** | +0.0399 | 8.6e-5 |
  | **LoRA − new rung 2** | **+0.0287** | 0.002 |
  | new rung 2 − ridge | −0.0271 | 0.107 |

  **Matching the optimiser removed about a quarter of the apparent delta** (+0.040 to +0.029). What survives is significant overall, and by scheme it is significant *only* on the split that shares residue positions between folds: `random` +0.056 (p<0.001), `contiguous` +0.015 (p=0.30), `modulo` +0.015 (p=0.50). That sharpens the previous entry's finding rather than overturning it.

  **The prediction I made was wrong, usefully.** I expected rung 2 to approach ridge once its optimiser matched, which would have shown it converged. It moved +0.011 and remains 0.027 short (p=0.107). So the delta is still not cleanly attributable, and the reason has changed: it is no longer an optimisation-budget difference.

  **Where the remaining gap actually lives.** Broken out by readout, ridge minus the rung-2 MLP is +0.104 for `mean`, +0.009 for `at_position`, and **−0.031 for `difference_at_position`**, where the MLP wins. So this is not a general under-fit. It is one badly-conditioned readout: a mean-pooled 480-dimensional vector over a 300-residue protein differing at a single position, where ridge's tuned penalty does something the MLP's dropout and early stopping do not. The rung-2 head has no explicit weight decay. Filed as issue #35, because it is a regularisation problem rather than the optimisation one #33 named, and fixing it will move rung 2's numbers again.

  **Two smaller things review caught.** Eleven of 324 rung-2 runs now reach the 200-epoch cap while still improving (best epochs 189 to 199, nine of them the `mean` readout); no rung-3 run does. Those are lower bounds and would inflate the delta if they mattered, but excluding them moves it from +0.0287 to +0.0292, so they do not. And `train` recorded `batch_size` in its history while `train_lora` did not, which left the two rungs reporting different schemas to whatever reads them side by side; both record it now.

- **Decision / next step:** #33 is satisfied: an epoch-count difference can no longer produce different optimisation budgets, and `tests/test_dms_run_arms.py` asserts both rungs read the shared constants and that the old per-rung names are gone, so the agreement is a property rather than a coincidence. `method.md` now states plainly what still differs rather than claiming one respect. The headline stands where the previous entry left it: against a converged frozen baseline LoRA buys nothing (+0.0015, p=0.28), and the delta against the trained head is concentrated in the leaky split. Both figures changed materially, not just in matplotlib churn, and are committed.

### 2026-08-02: rung 3 ran, and the delta it appeared to show is rung 2's under-fit

- **Question / hypothesis:** what does adapting the encoder buy over a frozen representation, across three assays, three splits, three readouts and four training sizes? This is the headline the ladder was built to produce.
- **Setup:** the full 324-configuration rung-3 grid on SLURM L40S, `esm2_t12_35M_UR50D`, LoRA rank 8 alpha 16 on `q_proj`/`v_proj`, lr 1e-4, batch 8. All 324 shards produced, no NaN, adapters attached in every run (184,320 trainable parameters), early stopping fired everywhere (11 to 110 epochs against a 200 cap).
- **Result:** the naive comparison says fine-tuning helps. Against rung 2 as published, LoRA gains **+0.0399 Spearman on average** (Wilcoxon p=8.6e-5, 69% of configurations), and the gain looked concentrated in the position-sharing split: `random` +0.070, `contiguous` +0.029, `modulo` +0.021.

  **That reading does not survive scrutiny, for two separate reasons.**

  **The two rungs do not differ in exactly one thing.** Rung 2 trains at batch 256, lr 1e-3; rung 3 at batch 8, lr 1e-4. Early stopping is shared, but patience is counted in *epochs*, and an epoch is a different number of gradient updates on each rung: 5x more for rung 3 at N=32, rising to 25x at N=2048. 14.8% of rung-2 runs select `best_epoch <= 2`, essentially at initialisation. So the delta mixes "adapting the encoder helped" with "rung 3 optimised for longer", and the claim in `method.md` that consecutive rungs differ in exactly one respect is, as things stand, false. The two rungs share the stopping *constant*, not the stopping *rule*.

  **A converged baseline on the same frozen features recovers the whole gain.** `frozen_reference.py` fits ridge regression on the identical cached embeddings, the identical splits and subsample draws, with the penalty chosen on the identical validation fold. It is converged by construction. Results in `results/frozen_ridge.csv`:

  | contrast | mean | Wilcoxon p | wins |
  |---|---:|---:|---:|
  | LoRA − rung 2 as published | +0.0399 | 8.6e-5 | 69% |
  | ridge − rung 2 as published | +0.0383 | 0.017 | 57% |
  | **LoRA − ridge** | **+0.0015** | **0.28** | 59% |

  Mean Spearman: rung 2 0.224, ridge 0.262, LoRA 0.263. **Against a properly-fit frozen representation, LoRA buys nothing on average.** The apparent gain is the size of rung 2's under-fit.

  **What does survive.** The scheme ordering: LoRA beats ridge only on `fold_random_5` (+0.0195, p=0.032), and not on either position-disjoint scheme (`contiguous` +0.007 p=0.96, `modulo` −0.022 p=0.86). The paired scheme contrast `random − modulo` is +0.049, p=0.0022. That is consistent with adaptation buying site-specific memorisation: predicting each test variant by the mean score of its position in the training pool reaches Spearman 0.80 / 0.43 / 0.80 under `random` and is undefined under the disjoint schemes, which have 0% position overlap by construction.

  **Three corrections to what I first wrote, all found by the skeptic pass.**

  1. I read "contiguous p=0.017 vs modulo p=0.57" as the two disjoint schemes disagreeing. That is the difference-of-significance fallacy. Paired, `contiguous − modulo` is +0.008, p=0.52: there is nothing to explain.
  2. I claimed the gain was largest for the `mean` readout (+0.080) and near zero for `difference_at_position` (+0.007). Against ridge the ordering **reverses**: `difference_at_position` +0.036 (p<0.001), `at_position` +0.019, `mean` −0.051. The `mean` readout is where rung 2's baseline is broken, not where LoRA helps.
  3. The 108 seed-averaged configurations are not 108 independent observations: they rest on 3 assays and 3 test folds per scheme, and the per-assay deltas change sign. Under an assay-clustered bootstrap only `random` excludes zero (`contiguous` [−0.011, +0.073], `modulo` [−0.013, +0.078], `random` [+0.029, +0.115]). Effective n for any scheme-level claim is nearer 3 than 36.

  Two smaller corrections: R1AB_SARS2's anti-correlated prior is specific to the 35M checkpoint (650M gives +0.11 / +0.06 / +0.21), and should not be described as a property of the assay. CCR5's prior beating both supervised rungs almost everywhere does hold: 0/12 rung-2, 1/12 rung-3 and 1/12 ridge scheme-by-N cells clear it.

- **Decision / next step:** **do not report the +0.040 as an effect of adaptation.** The publishable finding is the negative one: on this cohort, at 35M, LoRA does not beat a converged frozen-feature baseline, and whatever advantage it has is confined to the split that lets it see the test positions. Before the ladder can attribute anything to adaptation, rung 2 must be re-run with rung 3's optimiser, or patience redefined in gradient steps rather than epochs. `frozen_ridge.csv` stays as a permanent sanity floor for rung 2, because it is nearly free once the embeddings are cached and it is the check that catches this class of failure. Filed as its own issue rather than patched here, since re-running rung 2 changes committed numbers.

### 2026-08-02: rung 3 becomes a 324-task SLURM array (issue #20)

- **Question / hypothesis:** turn rung 3 into a job array without inheriting the two problems the #14 validation surfaced: a repo that disagreed with itself about what a job is, and a result-writing path that would silently lose rows under concurrency.
- **Setup:** no experiment. `slurm/submit-finetune.sh` becomes a real array; `run_arms.py` gains `--task-id`, `--grid-size` and `--aggregate`.
- **Result:**

  **A job is one configuration, so the array is 324 tasks**, not the 81 the 2026-08-01 entry assumed. That entry costed a job as one `(assay, scheme, readout, seed)` running the full N curve, while the runner's own docstring described one configuration per invocation. Both readings were in the tree. 324 wins because the runner already works that way, a preempted task costs one configuration rather than four, and short tasks backfill better. Total GPU-hours are unchanged: still 143 MPS-hours, still 10 to 18 on an L40S. Restated per task, at 3N + 256 per-epoch encoder work: the N=2048 arms are 18x the N=32 ones and take about 70% of a curve between them, so the worst task is ~1.2 h on MPS and the cheapest ~4 minutes, which is 5 to 10 minutes and well under a minute respectively on an L40S. MaxArraySize here is 50001, so 324 needs no chunking.

  **The array index maps to a configuration in Python, not in the batch script.** `--task-id` indexes `grid`, which was already deterministic. An off-by-one in a shell expression would be invisible until the results came up short, and nothing in a batch file is reachable by a test.

  **One configuration now writes one file.** Each task writes `results/<rung>_shards/<configuration>.csv`, and `--aggregate` combines them. The old path read-modify-wrote a single CSV under `not args.all`, which is exactly the array path: two tasks finishing close together both read the pre-existing file and both wrote it, and the loser vanished. The result would have been a well-formed CSV with fewer rows than jobs that reported success, with nothing cross-checking the two. Shards are keyed by configuration rather than task id, so a requeued task overwrites its own shard instead of duplicating a row, and the name stays meaningful if the grid is ever reordered.

  **Aggregation asserts rather than trusts.** It refuses to write unless every configuration produced a shard, and names the ones that did not; it also rejects shards belonging to no configuration in the current grid, since a leftover from an earlier grid would report an arm this run never asked for. Verified on real data: with 2 of 324 shards present it named 10 missing and said "and 312 more", and wrote no partial `frozen.csv`.

  **Nothing moved.** A task's output is identical to the committed frozen-rung numbers for the same configuration, to the last digit, and `--rung frozen --all` still reproduces `frozen.csv` byte-for-byte.

  **One defect found by using the thing rather than testing it.** `--grid-size` originally printed inside `run_context`, whose logger writes to stdout, so the command substitution the batch script uses to size `--array` would have captured a dozen log lines along with the number. It is now handled before the run context: a query that changes nothing should not open a run or leave a manifest.

  **The same race survived one layer down, and review caught it.** Fixing the CSV write left `run_context` untouched, and it derives both the log and the manifest path from the run name plus a timestamp with one-second resolution. Every task used the name `dms-run-arms`, so under `%16` the sixteen tasks starting in a given second shared a filename: the manifest is written with `write_text` and would be overwritten, the log handler appends and would interleave. Reproduced directly: three runs in one second left two manifests, with the middle task's record gone. That would have defeated this issue's own requirement that a task's manifest record what it ran. Tasks now run under `dms-run-arms-task<id>`. Worth stating as a general hazard rather than a local fix: `run_context` is not safe for two concurrent runs sharing a name, and the next array to be written will need the same precaution.

  **The cost of 324 over 81, quantified rather than asserted.** Per-task fixed overhead is about 20 s of imports plus `load_esm2`, so roughly 1.8 allocation-hours across the array against 10 to 18 GPU-hours of work. For the N=32 arms, startup plausibly exceeds compute. That is the price of the finer restart granularity, and it is worth knowing before the same choice is made for a larger grid.

  **The walltime is conditional, not derived.** The 1 h per task assumes early stopping fires. `MAX_EPOCHS = 200` with patience 10 means an arm that never stops early runs about 10x the estimate at N=2048. The failure is loud, since the task dies without a shard and `--aggregate` names it, which is the right shape but not the same as the limit being right.

  **Coverage review found the wiring untested, for the second time.** The pure helpers were fully covered while every new line in `main()` was not, including the shard-writing branch that is the entire point of this issue: reverting it to the old read-modify-write would have failed no test. That is the same asymmetry as the manifest gap in #14, library half tested and script half assumed. `main()` is now exercised end to end with `load_inputs` and `evaluate` stubbed, and three tests that restated their target's implementation were replaced rather than kept for the count.

- **Decision / next step:** the submission machinery is done and the cluster is not. Nothing in issue #20's third item has been exercised: the env has never been solved there, conda-forge has never been checked for a CUDA build on that platform, `peft` has never run on CUDA in this project, the data has not been staged, and the checkpoint cache has not been pre-warmed. `slurm/README.md` carries that as an ordered checklist. The run that fills in `lora.csv`, and the rung-2-to-rung-3 delta it produces, is the next entry.

### 2026-08-01: refactor the training layer before the SLURM array (issue #14)

- **Question / hypothesis:** three findings deferred from PR #13's reviews, none of them a correctness bug, all of them places where the discipline `embeddings.py` got right did not cross into `training.py`. The question is whether they can be cleaned up without moving a single number, since the SLURM array is the first place a diverged early-stopping rule would corrupt a real result rather than a toy one.
- **Setup:** no experiment. Refactor only: a frozen `LoraSpec` grouping the adapter hyperparameters, a shared `_BestEpochTracker` owning best-epoch selection and the stopping rule for both `train` and `train_lora`, and a `tests/conftest.py` consolidating three diverged encoder stubs.
- **Result:** every committed artifact is byte-identical, re-run against the final state of the branch.

  | artifact | rung / cohort | outcome |
  |---|---|---|
  | `grounding-multimodal/results/arms_all.csv`, `arms_all.md`, `ablation_all.json`, `per_class_f1_all.json` | Project 1, all | byte-identical |
  | `grounding-multimodal/results/arms_annotated.csv`, `arms_annotated.md`, `ablation_annotated.json`, `per_class_f1_annotated.json` | Project 1, annotated | byte-identical |
  | `dms-benchmark/results/frozen.csv` | rung 2 | byte-identical |
  | `dms-benchmark/results/zero_shot.csv` | rung 1 | every Spearman identical, see below |

  The shared loop is bit-identical by construction rather than by luck: `_BestEpochTracker` draws no randomness and changes no arithmetic, so the number and order of RNG draws in both loops is unchanged. The re-runs confirm that rather than discover it.

  **Rung 3 was smoke-tested against real ESM-2 and real peft**, since the offline suite only exercises it against a toy `nn.Module`. The recorded configuration (`R1AB_SARS2_Flynn_2022`, `fold_modulo_5`, `at_position`, N=128, seed 0) returns Spearman **0.1792351850878362**, identical to the last digit to the pre-refactor run in `logs/dms-run-arms-20260801T082036Z.json`, and reproduced twice. So the LoRA path is bit-for-bit too, not only the frozen one. `trainable_encoder_parameters = 184320` confirms the adapters attached through the new `LoraSpec` plumbing rather than silently not attaching, and `best_epoch = 12` with `epochs_run = 23` puts the stop at exactly epoch 22, which is the shared tracker's `best + EARLY_STOPPING_PATIENCE` boundary confirmed on real data rather than in a unit test. 357.7s and 390.1s on MPS, consistent with the 441.7s measured before the validation cap analysis.

  **That smoke run found a half-delivered payoff.** Issue #14 asked for two things from `LoraSpec`: one object per SLURM job, and a manifest carrying a single nested block instead of loose keys. The nested block existed in `train_lora`'s history, but `run_arms.py` recorded none of it, so a rung-3 manifest still described a run without saying what was adapted and the only record of rank and alpha was the commit hash. That is the original complaint, untouched. `run.record("lora", LORA_SPEC.as_history_block())` closes it, scoped to the lora rung. No test would have caught this: the unit tests assert on `history["lora"]`, and nothing asserts on what the script records.

  **One pre-existing discrepancy, surfaced here and fixed in its own commit.** Re-running rung 1 reproduces all eighteen Spearman values exactly, but emits an extra `n_val` column. `zero_shot.csv` was committed in `c0e0c63`; the `n_val` column entered the row schema in the *later* commit `5ce2c90`, which capped validation and re-ran rung 2 but not rung 1. The committed file was therefore stale in its columns, not in its numbers. It has been regenerated: all eighteen rows byte-identical on every shared column, `n_val` added and uniformly 256. Note that rung 1 trains nothing, so its `n_val` records the size of a split it never uses; the column is uniform across rungs by design, and rung 2's file already carried it.

  The figures were re-run as the convention requires and are **not** committed. `make_figures.py` reads `zero_shot.csv`, but its output differs only in matplotlib's embedded `<dc:date>` and its randomly generated element ids: after normalising those two, every coordinate is identical. Committing that churn would assert a change to the figures that did not happen. Worth knowing for next time: these SVGs are not byte-reproducible, so "re-run the script and commit the result" always produces a diff, and only a normalised comparison distinguishes a real one.

  **No `EMBEDDING_IMPL_VERSION` bump, deliberately.** The diff to `embeddings.py` is two renames plus one docstring: no change to pooling, truncation, normalization, layer selection, dtype, or the empty-text rule. `test_cache_key_is_stable_for_the_recorded_spec` passing with `GOLDEN_SPEC_KEY` untouched is the evidence.

  **One history format change.** `train_lora`'s history reports the adapter configuration as a nested `"lora": {"rank", "alpha", "target_modules"}` block, replacing the loose `lora_rank` / `lora_alpha` / `target_modules` keys. Nothing in the repo read the old keys, but manifests already written under `logs/` carry them, so a reader comparing an old manifest to a new one should expect the shape to differ.

  **What the fixture consolidation bought, beyond tidiness.** Only the embeddings stub poisoned its non-residue slots, so a special-token read was loud on the frozen path and invisible on the LoRA path. That asymmetry is part of why the missing position guard survived PR #13's review. `TinyEsm` now poisons unconditionally, and a test reads BOS and EOS through `_encode_batch` to prove the fixture can tell; against the unpoisoned fixture it returns an ordinary-looking vector. The LoRA path also had no coverage of the `mean` readout at all, which the poisoned fixture makes worth adding.

- **Decision / next step:** the training layer is ready for the SLURM array PR. `LORA_SPEC` in `projects/dms-benchmark/scripts/run_arms.py` is the one object an array task now needs to carry, in place of three module constants that were not reachable from the CLI. Review caught that the round trip the array will actually perform did not work: `as_history_block` widens `target_modules` to a list for JSON, which is exactly what the constructor refuses, so `LoraSpec(**block)` raised. `from_history_block` is the inverse, and it re-runs every check rather than handing back a dict nobody validated. Without it the array PR would have met that assertion under time pressure, where the obvious fix is to loosen the one guard that catches a bare-string `target_modules`. Deliberately still out of scope: the `MeanReadout()` / `AtPosition(positions)` sum type that would make `readout` and `positions` an unrepresentable-invalid-state pair. It collapses several of these findings at once and is worth revisiting if a fourth readout ever lands, since adding one currently means editing five sites across two modules.

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
