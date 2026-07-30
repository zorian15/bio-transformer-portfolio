# docs/

Short research reports, one per project. These are written as the work happens, so
they stay honest about what is settled and what is still open.

Publishing to GitHub Pages (`gh-pages`) is planned but not set up yet; for now
these are plain Markdown files read in the repo. See `PLANNING.md`.

## Shared infrastructure

- [Run logging](run-logging.md): how pipeline runs are logged, and the manifest
  each one writes for provenance.

## Project 1: grounding-multimodal

- [Introduction](grounding-multimodal/introduction.md): the question, why it is
  interesting biologically and as a machine-learning problem, and how it will be
  judged.
- [Data](grounding-multimodal/data.md): inputs, labels, provenance, preprocessing
  decisions, and the train/validation/test split.
- [Results](grounding-multimodal/results.md): the six-arm numbers, what the control
  shows, and what is still unresolved.
- [Method](grounding-multimodal/method.md): how splits are made, what the trained
  head is, the metrics, and how the six-arm runner keeps the comparison fair.

The corresponding experiment log is
[`projects/grounding-multimodal/DECISION_LOG.md`](../projects/grounding-multimodal/DECISION_LOG.md),
which records individual runs and the decisions they drove. These docs describe
the setup; the log records what happened.
