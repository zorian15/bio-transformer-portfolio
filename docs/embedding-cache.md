# Embedding cache: what invalidates it, and why

Shared infrastructure in `biotp.embeddings`. Embedding 13,858 proteins takes about
100 minutes, so results are computed once and cached; every later question runs on
the cached vectors in seconds. That makes the cache load-bearing rather than an
optimization, and makes a wrong cache hit expensive in the way that matters.

## The rule

A cache entry is reused only when **both** halves of its key match:

1. **The inputs**: the exact sequences or texts, including their order and count.
2. **The spec**: every parameter of the embedding code that changes the output.

The spec is built by `sequence_embedding_spec` and `text_embedding_spec`:

| Field | Why it is keyed |
|---|---|
| `model_name` | Different checkpoint, different vectors |
| `impl_version` | Catches changes no other field describes |
| `max_sequence_length` | Truncation changes long proteins |
| `repr_layer_policy` | Reading a different layer changes everything |
| `pooling` | Mean over residues versus anything else |
| `dtype` | Output precision |
| `empty_text` (text only) | Zero vector versus encoding `""` |

`batch_size` is deliberately **not** keyed. Batching is a throughput knob that
provably does not change the output, verified by the batch-invariance test, so
including it would force needless recomputation. There is a test pinning that
exclusion so nobody "fixes" it later.

## Why the spec half exists

Embedding is a pure function of its inputs, so hashing the inputs alone is
sufficient *while the function stays fixed*. Edit the transformation and the same
inputs legitimately produce different vectors, while an inputs-only key stays
identical. The cache then reports a hit and returns the previous implementation's
answer, silently.

That was a real defect (issue #4). Changing `MAX_SEQUENCE_LENGTH` from 100 to 200
produced vectors differing by 0.48, and the cache served the old ones anyway.

Two ways it bites, in increasing severity:

- **A benchmark measures nothing.** A performance change that should be timed hits
  the cache instead and reports a near-zero duration.
- **Results describe code that is not in the repository.** Metrics, figures, and
  `DECISION_LOG.md` entries all cite an implementation nobody can reproduce from
  the commit. Every step succeeds and the numbers look plausible.

## Cache files explain themselves

Each `.npz` stores the spec that produced it alongside the vectors, so you can
read a cache file and know what made it:

```python
import json, numpy as np
with np.load("data/processed/embeddings/sequence_esm2_35m.npz") as f:
    print(json.loads(str(f["spec"])))
```

A miss also says why, rather than only that it happened:

```
cache miss, recomputing ... (max_sequence_length: 100 -> 200)
cache miss, recomputing ... (same spec, so the inputs themselves changed)
cache miss, recomputing ... (cache predates spec tracking)
```

## The part that still needs a human

`impl_version` is the residual risk: a change to how residues are pooled is not
visible in any named field, so someone has to bump it. Two safeguards:

- `test_cache_key_is_stable_for_the_recorded_spec` pins the exact key for a fixed
  spec. Any spec change, including a version bump, fails that test, so the change
  surfaces as something to confirm rather than passing quietly.
- `CLAUDE.md` carries the review checklist for diffs touching `embeddings.py`.

Neither is airtight, and that is the honest position. What they do is convert a
silent runtime failure into a visible code-review question, and the asymmetry
justifies it: an unnecessary bump costs a recompute, a missed one costs the truth
of a result.
