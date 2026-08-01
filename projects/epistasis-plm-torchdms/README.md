# epistasis-plm-torchdms (Project 3)

Deep mutational scanning of *combinations* is where epistasis lives. Can a protein language model capture it, and how does it compare to a model built for the job?

The observation the project is built on: **masked-marginal scoring is additive over sites by construction.** Each term masks one position of the wild-type sequence, so nothing in the sum ever sees the other mutation, and the score of a double mutant is identically the sum of its singles. Measured deviation from additivity on real data: exactly `0.00e+00`. That is a property of the scoring rule rather than an empirical weakness, which means the standard zero-shot use of a protein LM cannot represent epistasis at all.

- **Data:** Starr/Bloom SARS-CoV-2 RBD DMS, the barcoded libraries carrying variable numbers of mutations. ProteinGym ships only the single-mutant summary, so this comes from the Bloom lab release upstream of it.
- **Arms:**
  - PLM zero-shot masked-marginals, provably additive: the null.
  - PLM embeddings of the full mutant sequence + head, which *can* express epistasis, since a double's embedding is not the sum of its singles'.
  - `torchdms` additive, which separates "additive because the biology is" from "additive because the model cannot do otherwise."
  - `torchdms` with a latent nonlinearity: global epistasis.
- **Eval:** train on singles, test on multiples. Held-out mutational combinations, leakage-aware by construction, and the question the field actually asks.
- **Deliverable:** repo + weights + writeup.

`torchdms` pins `python_requires=">=3.8,<3.10"` and hard-pins `pandas==1.4.2`, so it cannot live in the Python 3.11 `biollm` env. It ships a `tdms` console entry point and runs in its own environment behind a subprocess boundary.

Replaces the original Project 3 (TCR / antibody LM for specificity or escape), which `../../PLANNING.md` records as a deferred future direction rather than dropping.

See `../../PLANNING.md` for full context. Log runs in `DECISION_LOG.md`.
