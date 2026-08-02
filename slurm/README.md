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

Try conda first, since that is what the repo documents:

```bash
mamba env create -f environment.yml && conda activate biollm && pip install -e .
```

**On the rhino nodes this fails, and the reason is worth recording** so the next person does not spend an afternoon on it. `conda.anaconda.org` resolves to AAAA records only, and IPv6 egress from those nodes does not work, so every repodata fetch dies mid-TLS with `SSL_ERROR_SYSCALL` / connection reset. It is not a proxy, a firewall rule against conda, or anything in `environment.yml`: `curl -4` to the same host succeeds, and IPv4 hosts such as github.com work normally, which is why `git clone` works on a node where `mamba` cannot. Diagnose with:

```bash
getent hosts conda.anaconda.org                                   # AAAA only?
curl -4 -sS -o /dev/null -w 'v4: %{http_code}\n' https://conda.anaconda.org/conda-forge/noarch/repodata.json
curl    -sS -o /dev/null -w 'default: %{http_code}\n' https://conda.anaconda.org/conda-forge/noarch/repodata.json
```

The knobs that force IPv4 preference are all root-owned, so this is not fixable from a user account. PyPI does answer, so the working path is a venv.

### The venv fallback

Only for a machine where conda cannot reach its channels. `tests/test_environment.py` forbids pip-installed torch, but it is `skipif(sys.platform != "darwin")`: the OpenMP hazard it guards is a macOS `libomp.dylib` collision, and PyPI's linux wheels ship CUDA. On Linux this is a legitimate build, not a workaround around a safety rule.

```bash
# 1. Python 3.11, matching environment.yml. uv fetches a standalone CPython from
#    GitHub, which works where conda's channels do not. On 3.12 the resolver
#    finds no pyarrow wheel, falls back to its sdist, and tries to build libcst,
#    which needs a Rust compiler. That error names Rust, not the interpreter.
pip install uv && uv python install 3.11

# 2. Put it on the fast filesystem, not in $HOME. The bundled CUDA wheels
#    (cublas, cudnn, nccl, cusparselt, triton) come to 4-6 GB, which is a large
#    fraction of a typical home quota, and compute nodes read /fh/fast faster.
uv venv --python 3.11 /fh/fast/<lab>/<user>/biollm-venv
source /fh/fast/<lab>/<user>/biollm-venv/bin/activate

# 3. --only-binary=:all: so pip reports "no wheel for X" rather than attempting a
#    compile and burying the real problem in someone else's build log. Versions
#    pinned to the laptop's, so the two environments differ in as little as
#    possible; torch is left free because the laptop's build is conda-forge's.
pip install --upgrade pip
pip install --only-binary=:all: \
  torch \
  "numpy==2.4.6" "pandas==3.0.3" "scipy==1.17.1" "scikit-learn==1.8.0" \
  "pyarrow==25.0.0" "matplotlib==3.10.9" \
  "peft==0.20.0" "fair-esm==2.0.0" \
  "pytest==9.1.1" "pyyaml==6.0.3"
pip install -e .
```

That list is what rung 3 imports, not a copy of `environment.yml`. `transformers` and `sentence-transformers` are omitted: only Project 1's text arms use them, through a function-local import, and `runlog` records them as absent rather than failing.

Point the batch scripts at it, rather than editing them:

```bash
export BIOTP_ENV_ACTIVATE=/fh/fast/<lab>/<user>/biollm-venv/bin/activate
```

Unset, they use `conda activate biollm`. Set, they source that file and exit non-zero if it does not exist, so a wrong path fails once at submit-test time instead of 324 times in parallel.

If `uv venv` reports `Directory not empty` deleting an existing venv, the old one is still activated in your shell and NFS has silly-renamed its open files to `.nfsXXXX`. `deactivate`, then `rm -rf`.

### The rest of bring-up

```bash
# 1. Confirm a CUDA build, not the CPU one. A CPU build does not error, it just
#    runs the whole array ~50x slow, which is the expensive way to find out.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from biotp.utils import get_device; print(get_device(prefer_gpu=True))"   # expect: cuda

# 2. Confirm peft attaches on CUDA. The offline suite only proves it on CPU
#    against a toy module.
pytest -q tests/test_train_lora.py

# 3. Stage the data. Rung 3 needs exactly two files, about 600 KB: it calls
#    load_esm2 directly and never touches the embedding cache, so the
#    dms_embeddings/ (63 MB) and embeddings/ (169 MB) trees are for rungs 1-2
#    and are dead weight here.
rsync -avP data/processed/proteingym_variants.parquet \
           data/processed/proteingym_assays.json \
           user@cluster:/path/to/bio-transformer-portfolio/data/processed/

# 4. Pre-warm the checkpoint cache, or 324 tasks each try to download ESM-2 on
#    first use. Put it somewhere the compute nodes can read.
export TORCH_HOME="$HOME/.cache/torch"
python -c "from biotp.embeddings import load_esm2; load_esm2('esm2_t12_35M_UR50D')"

# 5. Run one task by hand before submitting 324. The #SBATCH lines are comments
#    to bash, so this takes exactly the path the scheduler would, including the
#    environment activation.
SLURM_ARRAY_TASK_ID=0 bash slurm/submit-finetune.sh
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

If tasks fail, fix the cause and resubmit only those indices, then aggregate:

```bash
sbatch --array=17,42,101 slurm/submit-finetune.sh
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
