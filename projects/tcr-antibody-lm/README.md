# tcr-antibody-lm (Project 3)

Can a fine-tuned protein LM predict TCR-epitope specificity (or antibody escape) better than a simple baseline, and does it generalize to unseen epitopes? Points the shared machinery at the adaptive-immunity specialty.

- **Data:** VDJdb / IEDB for TCR-epitope pairs, or an antibody escape / DMS dataset in the polyclonal wheelhouse.
- **Model:** ESM-2 or a TCR-specific transformer fine-tuned for the task; baseline is a k-mer or simple classifier.
- **Eval:** the hard and honest split is held-out epitopes (no epitope shared between train and test), which exposes the memorization-vs-generalization gap that naive splits hide. Use `biotp.evaluation.grouped_split` with the epitope as the group key.
- **Deliverable:** repo + weights + writeup.

See `../../PLANNING.md` for full context. Log runs in `DECISION_LOG.md`.
