# slurm/

`sbatch` submission scripts for the GPU-heavy one-offs, meant to be submitted while SLURM access lasts. The same code runs locally on the MacBook via MPS (see `biotp.utils.get_device`); these just point it at a CUDA node for speed and for the larger ESM-2 sizes. Use the same `biollm` mamba env on the cluster.

## This cluster
- Default partition `campus-new`: GTX 1080 Ti / RTX 2080 Ti, ~11 GB. Fine for embedding extraction up to ESM-2 650M at a modest batch size.
- `chorus`: L40S, 48 GB, 4 per node. Use for the largest models and for fine-tunes.
- `short` for quick jobs; `interactive` for interactive sessions.
- Account is `matsen_e` (left out of the templates for now); add `#SBATCH --account=matsen_e` if the scheduler requires it. Available QOS include `normal`.

## Templates
- `submit-embed.sh`: extract and cache ESM-2 embeddings (the main one-off). Run it at every model size you might want, so the cached vectors survive even if cluster access ends.
- `submit-finetune.sh`: rung 3 of the DMS ladder as a 324-task job array, one configuration per task. See "Running rung 3 as an array" below.

Edit the `#SBATCH` directives (partition, account, time, resources) for your cluster before submitting, then `sbatch slurm/submit-embed.sh`.

## Running rung 3 as an array

`submit-finetune.sh` is a job array over the whole rung-3 grid: one task is one configuration, 324 of them. The mapping from `$SLURM_ARRAY_TASK_ID` to a configuration lives in `run_arms.py` and is covered by tests, so nothing here does arithmetic on the index.

### Building the environment

Two situations, and which one you are in is decided by a single question: can conda reach its channels from this machine?

```bash
curl -sS -o /dev/null -w '%{http_code}\n' https://conda.anaconda.org/conda-forge/noarch/repodata.json
```

| result | you are in | go to |
|---|---|---|
| `200` | **Situation A**, the documented path | [A: conda](#situation-a-conda-works) |
| anything else | **Situation B**, conda is unreachable | [B: venv fallback](#situation-b-conda-cannot-reach-its-channels) |

Run the check rather than assuming. Situation B looks like a broken repo or a broken `environment.yml` from the inside, and it is neither.

### Situation A: conda works

```bash
mamba env create -f environment.yml
conda activate biollm      # conda.sh provides this; mamba activate needs mamba.sh
pip install -e .
```

Nothing else to configure. Leave `BIOTP_ENV_ACTIVATE` unset and the batch scripts will activate `biollm` themselves. Skip to [verifying the environment](#verifying-the-environment-either-situation).

### Situation B: conda cannot reach its channels

**On the rhino nodes conda fails, and the reason is worth recording** so the next person does not spend an afternoon on it. `conda.anaconda.org` resolves to AAAA records only, and IPv6 egress from those nodes does not work, so every repodata fetch dies mid-TLS with `SSL_ERROR_SYSCALL` / connection reset. It is not a proxy, a firewall rule against conda, or anything in `environment.yml`: `curl -4` to the same host succeeds, and IPv4 hosts such as github.com work normally, which is why `git clone` works on a node where `mamba` cannot. Diagnose with:

```bash
getent hosts conda.anaconda.org                                   # AAAA only?
curl -4 -sS -o /dev/null -w 'v4: %{http_code}\n' https://conda.anaconda.org/conda-forge/noarch/repodata.json
curl    -sS -o /dev/null -w 'default: %{http_code}\n' https://conda.anaconda.org/conda-forge/noarch/repodata.json
```

The knobs that force IPv4 preference are all root-owned, so this is not fixable from a user account. PyPI does answer, so the working path is a venv.

`tests/test_environment.py` forbids pip-installed torch, but it is `skipif(sys.platform != "darwin")`: the OpenMP hazard it guards is a macOS `libomp.dylib` collision, and PyPI's linux wheels ship CUDA. On Linux this is a legitimate build, not a safety rule dodged.

```bash
# 1. Python 3.11, matching environment.yml. uv fetches a standalone CPython from
#    GitHub, which works where conda's channels do not.
#
#    The version is not cosmetic. On 3.12 the resolver finds no pyarrow wheel,
#    falls back to its sdist, and tries to build libcst, which needs a Rust
#    compiler. The error you get says "can't find Rust compiler", three layers
#    away from the actual cause.
pip install uv
uv python install 3.11

# 2. --seed installs pip INTO the venv. Without it, `pip` inside the activated
#    venv falls through PATH to whatever else is there, usually conda's, and
#    installs into that instead. It looks like it is working: the prompt says
#    (biollm-venv) and pip reports success, while the wheels say cp312 and land
#    somewhere else entirely. `which pip` is the check.
#
#    Fast filesystem, not $HOME: the bundled CUDA wheels (cublas, cudnn, nccl,
#    cusparselt, triton) come to 4-6 GB, which is a large fraction of a typical
#    home quota, and compute nodes read /fh/fast faster.
uv venv --seed --python 3.11 /fh/fast/<lab>/<user>/biollm-venv

# 3. Activate it, and only it. Two active environments make PATH ambiguous, and
#    a leftover (base) is what the pip fall-through above depends on.
conda deactivate 2>/dev/null || true
source /fh/fast/<lab>/<user>/biollm-venv/bin/activate
which python pip           # both must be under the venv path
python -V                  # 3.11.x

# 4. Install. --only-binary=:all: so pip reports "no wheel for X" rather than
#    attempting a compile and burying the real problem in someone else's build
#    log. Watch the download lines say cp311; cp312 means step 3 did not take.
pip install --only-binary=:all: \
  torch \
  "numpy==2.4.6" "pandas==3.0.3" "scipy==1.17.1" "scikit-learn==1.8.0" \
  pyarrow "matplotlib==3.10.9" \
  "peft==0.20.0" "fair-esm==2.0.0" \
  "pytest==9.1.1" "pyyaml==6.0.3"
pip install -e .
```

Already made the venv without `--seed`? You do not need to rebuild it: `uv pip install pip` puts one in, then `hash -r` so the shell forgets the path it cached for the other one.

Two things about those pins. They are the laptop's versions for everything that touches the numbers, so the two environments differ in as little as possible. But **conda-forge and PyPI version packages independently, so a conda env's pins do not transfer as a set**: `pyarrow` is deliberately unpinned because the laptop's 25.0.0 comes from conda-forge and PyPI's line stops around 20, and `torch` is unpinned for the same reason. Both are recorded in every run manifest, so what actually ran is never in doubt.

The list is what rung 3 imports, not a copy of `environment.yml`. `transformers` and `sentence-transformers` are omitted: only Project 1's text arms use them, through a function-local import, and `runlog` records absent packages rather than failing.

Finally, point the batch scripts at it rather than editing them:

```bash
export BIOTP_ENV_ACTIVATE=/fh/fast/<lab>/<user>/biollm-venv/bin/activate
```

Unset, they run `conda activate biollm`. Set, they source that file and exit non-zero if it does not exist, so a wrong path fails once at submit-test time instead of 324 times in parallel. Put the export in your shell profile, or re-run it in every session that submits.

If `uv venv` reports `Directory not empty` deleting an existing venv, the old one is still activated in your shell and NFS has silly-renamed its open files to `.nfsXXXX`. `deactivate`, then `rm -rf`.

### Verifying the environment (either situation)

```bash
python -c "import sys, torch, pyarrow, peft, esm; print(sys.prefix); print(torch.__file__)"
pytest -q tests/test_train_lora.py
```

Both printed paths must be inside the environment you meant to build. A `torch.__file__` under `miniforge3` while the prompt says otherwise is the fall-through described above, and everything downstream will use the wrong interpreter.

`torch.cuda.is_available()` returns `False` on a login node with no GPU, so it is not a useful check there. The single-task run at the end of bring-up is the real one.

### The rest of bring-up

```bash
# 1. Stage the data. Rung 3 needs exactly two files, about 600 KB: it calls
#    load_esm2 directly and never touches the embedding cache, so the
#    dms_embeddings/ (63 MB) and embeddings/ (169 MB) trees are for rungs 1-2
#    and are dead weight here. Run this from the machine that has the data.
rsync -avP data/processed/proteingym_variants.parquet \
           data/processed/proteingym_assays.json \
           user@cluster:/path/to/bio-transformer-portfolio/data/processed/

# 2. Pre-warm the checkpoint cache, or 324 tasks each try to download ESM-2 on
#    first use. Somewhere the compute nodes can read, and not $HOME: 324 tasks
#    reading ~130 MB each over NFS is worth avoiding.
export TORCH_HOME=/fh/fast/<lab>/<user>/torch-cache
python -c "from biotp.embeddings import load_esm2; load_esm2('esm2_t12_35M_UR50D')"

# 3. Run one task by hand before submitting 324. The #SBATCH lines are comments
#    to bash, so this takes exactly the path the scheduler would, including the
#    environment activation, which is the part most likely to be wrong.
#
#    Task 267 is the configuration recorded in DECISION_LOG.md, so it has a known
#    answer to check against rather than only "it did not crash". On a login node
#    with no GPU, prefix with: srun -p chorus --gres=gpu:1
SLURM_ARRAY_TASK_ID=267 bash slurm/submit-finetune.sh
cat projects/dms-benchmark/results/lora_shards/lora-R1AB_SARS2_Flynn_2022-fold_modulo_5-at_position-n128-seed0-esm2_t12_35M_UR50D.csv
```

Three things to read off that row, in order of what they catch:

- `trainable_encoder_parameters` is **184320**. Anything else and the adapters did not attach, which still trains the head and still reports a plausible number.
- `spearman` is near **0.179**. Not identical: that reference was measured on MPS, and CUDA differs in floating point. Treat a few thousandths as expected and a sign flip as a problem.
- `epochs_run` is well under 200. The one-hour walltime assumes early stopping fires; if a run goes the distance, the N=2048 arms will hit the limit.

Confirm you got a CUDA build while you are here, since a CPU build does not error, it just runs the array about 50x slow:

```bash
python -c "from biotp.utils import get_device; print(get_device(prefer_gpu=True))"   # expect: cuda
```

### Submitting

```bash
# The array bound comes from the code, never from memory. Prints just the number.
python projects/dms-benchmark/scripts/run_arms.py --rung lora --grid-size   # 324

ARRAY_JOB=$(sbatch --parsable slurm/submit-finetune.sh)

# Combine the shards once every task has succeeded. afterok means a failed task
# blocks aggregation rather than letting it report a short result.
#
# --wrap runs in a fresh shell that has not sourced conda and does not inherit
# DATA_ROOT or RESULTS_DIR, so the activation and the paths have to be spelled
# out. A wrap that just called python would fail on import, or worse, aggregate
# into the default results directory rather than the one the array wrote to.
sbatch --dependency=afterok:"$ARRAY_JOB" --wrap \
  "source \"${BIOTP_ENV_ACTIVATE:-/dev/null}\" 2>/dev/null || \
   { source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate biollm; } && \
   python projects/dms-benchmark/scripts/run_arms.py --rung lora --aggregate \
     --data-root \"${DATA_ROOT:-data}\" \
     --results-dir \"${RESULTS_DIR:-projects/dms-benchmark/results}\""
```

Each task writes `results/lora_shards/<configuration>.csv` and nothing else; `--aggregate` combines them into `results/lora.csv`. Shards are named after the configuration rather than the task id, so requeueing a task overwrites its own shard instead of duplicating a row.

`--aggregate` refuses to write unless every configuration produced a shard, and names the ones that did not. That is the point of the split: the previous design had every task read-modify-write one CSV, and two tasks finishing close together silently dropped a row into a file that still looked complete.

If submission is rejected for accounting, add `#SBATCH --account=<account>` to `submit-finetune.sh`; the templates deliberately omit it.

### Watching it, and finishing

```bash
squeue -u "$USER"
sacct -j "$ARRAY_JOB" --format=JobID,State,ExitCode | grep -v COMPLETED   # failures only
ls projects/dms-benchmark/results/lora_shards/*.csv | wc -l               # want 324
```

`afterok` means any failed task blocks aggregation, deliberately, so a short result cannot appear quietly. If some fail, fix the cause, resubmit just those indices, and then aggregate by hand since the dependency job will already have been cancelled:

```bash
sbatch --array=17,42,101 slurm/submit-finetune.sh
python projects/dms-benchmark/scripts/run_arms.py --rung lora --aggregate
```

Then bring back the one file that matters. `lora.csv` is small and tracked; the shards are gitignored intermediates and stay on the cluster:

```bash
rsync -avP user@cluster:/path/to/.../projects/dms-benchmark/results/lora.csv \
           ./projects/dms-benchmark/results/
python projects/dms-benchmark/scripts/make_figures.py   # reads all three rungs
```

## Getting results back (rsync + Hugging Face)
Per the storage plan: git tracks only small files; large artifacts move by rsync, and final checkpoints go to Hugging Face. Nothing large is committed.

```bash
# From the laptop, pull cached embeddings the job wrote on the cluster:
rsync -avP user@cluster:/path/to/bio-transformer-portfolio/data/processed/ ./data/processed/

# Final trained checkpoints go to the Hugging Face Hub (durable, doubles as
# the public release), via biotp.release.push_to_hub(...).
```

Embedding caches are regenerable, so if one is lost you can simply recompute it; only trained checkpoints need the durable Hugging Face home.
