# Embedding cache: what invalidates it, and why

Shared infrastructure in `biotp.embeddings`. Embedding 13,858 proteins takes about
100 minutes, so results are computed once and cached; every later question runs on
the cached vectors in seconds. That makes the cache load-bearing rather than an
optimization, and makes a wrong cache hit expensive in the way that matters.

## The rule

A cache entry is reused only when **both** halves of its key match:

1. **The inputs**: the exact sequences or texts, including their order and count,
   and the residue positions when one is requested. Under the `at_position`
   readout the position is part of the input's identity, since the same sequence
   read at two positions is two different vectors, so the hashed item is
   `f"{sequence}@{position}"` rather than the sequence alone. That join is
   injective: `@` is outside both the 20-residue alphabet and the decimal digits,
   so it appears exactly once and splits the item back into the pair that made
   it, and two distinct (sequence, position) pairs cannot collapse onto one key.
2. **The spec**: every parameter of the embedding code that changes the output.

The spec is built by `sequence_embedding_spec(model_name, readout)` and
`text_embedding_spec`:

| Field | Why it is keyed |
|---|---|
| `model_name` | Different checkpoint, different vectors |
| `impl_version` | Catches changes no other field describes |
| `max_sequence_length` | Truncation changes long proteins |
| `repr_layer_policy` | Reading a different layer changes everything |
| `pooling` | Carries the readout: mean over residues versus a single residue |
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
visible in any named field, so someone has to bump it. Three safeguards:

- `test_cache_key_is_stable_for_the_recorded_spec` pins the exact key for a fixed
  spec. Any spec change, including a version bump, fails that test, so the change
  surfaces as something to confirm rather than passing quietly.
- `test_embed_sequences_matches_the_frozen_reference` checks the current code
  against 24 vectors written by the pre-v2 implementation, so a change that moves
  the numbers is measured rather than assumed. It needs model weights, so it is
  marked `network` and the default `pytest` run skips it; CI opts back in on every
  pull request that touches the embedding path, and weekly
  (`.github/workflows/embedding-anchor.yml`). Until that workflow existed the
  check ran only when someone typed `pytest -m network`, which is a poor guard for
  a failure mode whose defining feature is that nothing looks wrong.
- `CLAUDE.md` carries the review checklist for diffs touching `embeddings.py`.

None of this is airtight, and that is the honest position. CI can tell you the
vectors moved; it cannot tell you whether they were *supposed* to, and so it
cannot decide the version bump for you. What the three do together is convert a
silent runtime failure into a visible code-review question, and the asymmetry
justifies it: an unnecessary bump costs a recompute, a missed one costs the truth
of a result.

## Version history

| `impl_version` | Change | Vectors moved? |
|---:|---|---|
| 1 | The MVP implementation: dataset-order batching, per-sequence pooling | baseline |
| 2 | Length-bucketed batching and on-device masked pooling (issue #3) | yes, at ~1e-6 |

Version 2 is the case the mechanism was built for, and it is worth reading as a
worked example. The *pooling rule* did not change: it is still the mean over
residue positions, excluding BOS, EOS, and padding, so the named `pooling` field
reads exactly as before and the key would not have moved on its own. What changed
underneath is that pooling became a masked sum over the batch instead of a slice
per sequence, and batches are now grouped by length, so a given protein rides in
a differently padded batch than it used to. ESM-2 is only padding-invariant to
about 1e-5, so the vectors are close but not bit-identical.

That is precisely the shape of change `impl_version` exists to catch: no named
field describes it, and every existing cache file would otherwise have kept
serving v1 vectors while the repository described v2 code. It also mattered for
the benchmark that motivated the change, since an "after" run that hit the cache
would have reported a speedup measuring nothing at all.

The v1 vectors are not lost to the argument, incidentally.
`tests/data/reference_embeddings.npz` holds 24 of them, written before the
rewrite, and `test_embed_sequences_matches_the_frozen_reference` checks that v2
still reproduces them within float32 tolerance. The version bump says "these are
different"; the anchor says "and here is how much".

## A change that needed no bump: the readout parameter

Project 2 (issue #11) added a second readout, `at_position`, which reads one
residue instead of pooling all of them. `embed_sequences` and `cached_embeddings`
therefore take `readout` and `positions`, neither with a default, so a caller that
has not thought about the readout cannot get one by accident.

This is the opposite case to v2, and it is worth naming as such. A named field
already describes the change: `pooling` carries the readout, so the two readouts
land in separate cache files under different keys, and nothing stale can be
served. The `mean` value is byte-identical to the string that predates the
parameter, so every vector cached for Project 1 stays valid and its arms
reproduce bit-for-bit. `EMBEDDING_IMPL_VERSION` stays at 2, and
`test_cache_key_is_stable_for_the_recorded_spec` passes unmodified, which is the
evidence for that claim rather than a nuisance to route around.
