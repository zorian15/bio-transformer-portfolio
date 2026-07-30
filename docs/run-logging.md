# Run logging and provenance

Shared infrastructure in `biotp.runlog`, used by every pipeline script in every
project. Two things it produces per run: a readable log, and a machine-readable
manifest.

## Why, concretely

Both of these were real problems on this project before the module existed.

**Progress was invisible.** Bare `print` into a redirected file is
block-buffered, so embedding 13,858 sequences produced a 0-byte log for the best
part of an hour. There was no way to tell a working run from a hung one.
`logging` handlers flush per record, and the embedding loop now reports batch
progress with a throughput estimate.

**Provenance was remembered rather than recorded.** `DECISION_LOG.md` entries are
supposed to state the setup that produced a number: data, model, config, device,
code version. Reconstructing that after the fact is how logs quietly become
wrong. Now every run writes a JSON manifest, and log entries can cite it.

## What a run produces

```
logs/prepare-data-20260730T014346Z.log     # human-readable, flushed live
logs/prepare-data-20260730T014346Z.json    # manifest
```

`logs/` is gitignored, since logs are regenerable. For runs that produce
committed metrics, the runner also drops a copy of the manifest beside them
(`results/run_manifest_all.json`), so a number in a writeup can be traced to the
commit and machine that produced it.

The manifest holds:

| Field | Contents |
|---|---|
| `status` | `completed` or `failed` |
| `started_at`, `duration_seconds` | When and how long |
| `params` | The script's arguments, usually `vars(args)` |
| `steps` | Each named stage, its duration, and whether it failed |
| `records` | Facts the script chose to record: counts, split sizes, metrics |
| `git` | Commit, branch, and whether the tree was dirty |
| `environment` | Python, platform, device, and versions of eight key packages |
| `error` | Exception type and message, on failed runs only |

`git.dirty` matters as much as the commit. A number produced from uncommitted
edits is not reproducible from the commit alone, and the manifest says so rather
than implying a clean provenance it cannot support.

## Using it

```python
from biotp.runlog import get_logger, run_context

log = get_logger("my-pipeline")

with run_context("my-pipeline", params=vars(args)) as run:
    with run.step("load data"):
        table = load(...)
    run.record("rows", len(table))
```

- `run.step(name)` times a stage and logs its start and end. A stage that raises
  is timed, logged, and marked `failed` in the manifest before the exception
  propagates.
- `run.record(key, value)` logs a fact and puts it on the manifest. Use it for
  anything a writeup would cite. Numpy scalars are coerced, so recording a
  `np.float32` metric will not break the JSON write.
- `run.write_manifest(path)` writes an extra copy somewhere specific.

## Failure is recorded, not swallowed

The manifest is written whether the body succeeds or raises, and the exception is
re-raised untouched. A crashed run therefore leaves evidence rather than nothing,
and the failing stage is identifiable:

```json
{
  "status": "failed",
  "error": {"type": "AssertionError", "message": "unexpected localization ..."},
  "steps": [
    {"name": "download DeepLoc FASTA", "duration_seconds": 0.0},
    {"name": "parse FASTA", "duration_seconds": 0.001, "failed": true}
  ]
}
```

This is the difference between "the run died" and "the run died in the parser on
input it refused to guess about," and it is why logging completion matters as much
as logging progress.
