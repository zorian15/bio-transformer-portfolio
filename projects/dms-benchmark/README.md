# dms-benchmark (Project 1)

When does a fine-tuned protein language model beat a simple structured/biophysical baseline on deep mutational scanning fitness, and how does the answer change with the amount of labeled data? Tests the priors-vs-scale thesis empirically. Reuses the sequence pipeline built in `grounding-multimodal`.

- **Data:** ProteinGym (public DMS benchmark) and/or one of your own viral DMS datasets.
- **Model:** ESM-2 zero-shot (masked-marginal log-likelihood scoring) vs fine-tuned (LoRA or regression head) vs a simple additive / site-independent baseline (or a polyclonal-style structured model).
- **Eval:** Spearman on held-out mutations, plus data-efficiency curves (performance vs number of training labels). The data-efficiency curve is the money plot: the "every label costs a wet-lab measurement" argument, quantified.
- **Deliverable:** repo + writeup with the benchmark and the crossover point where scale overtakes structure.

See `../../PLANNING.md` for full context. Log runs in `DECISION_LOG.md`.
