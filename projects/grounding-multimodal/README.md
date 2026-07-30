# grounding-multimodal (Project 1, flagship)

Does grounding a protein-sequence representation in text (functional annotations) measurably improve a downstream task over sequence-only, and does the gain survive controls that rule out label leakage? A protein-domain take on the broader "does language grounding help" question.

- **Data:** Swiss-Prot / UniProt entries with sequence plus curated function text (or GO terms / keywords). Downstream task with clean labels: subcellular localization (DeepLoc), EC-number or GO-term classification, or a protein-family property. Optional viral-protein variant for domain flavor.
- **Model (MVP then extend):** MVP is frozen ESM-2 sequence embedding + frozen text embedding (small sentence encoder) concatenated into a small MLP head. Extend to CLIP-style contrastive alignment and light encoder fine-tuning.
- **Baselines:** sequence-only, text-only, and a shuffled/random-text control.
- **Eval:** held-out proteins/families. The headline is sequence+text vs sequence-only; the shuffled-text and text-only controls prove any gain is real grounding, not annotation leaking the label.
- **Deliverable:** repo + weights + a writeup whose headline is the honest answer, including a clean null result if that is what happens.
- **Main risk:** text leakage (annotations encode the label). Make controls the spine of the analysis.

## Writeup
- [Introduction](../../docs/grounding-multimodal/introduction.md): the question, biological and ML framing, objectives, evaluation criteria.
- [Data](../../docs/grounding-multimodal/data.md): inputs, labels, provenance, preprocessing decisions, splits.
- [Method](../../docs/grounding-multimodal/method.md): splits, head, metrics, and the six-arm runner.
- [Results](../../docs/grounding-multimodal/results.md): six-arm numbers and interpretation.

See `../../PLANNING.md` for full context. Log runs in `DECISION_LOG.md`.
