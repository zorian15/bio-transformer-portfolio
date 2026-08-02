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

### Bring-up, before the first submission

None of this has been exercised on the cluster yet. Do it once, in order, and stop at the first thing that surprises you.

```bash
# 1. Build the env from the same file the laptop uses.
mamba env create -f environment.yml
mamba activate biollm
pip install -e .          # safe here; a cluster checkout is not a worktree

# 2. Confirm you got a CUDA build, not the CPU one. conda-forge resolves this
#    per platform, and a CPU build would run the whole array 50x slow rather
#    than failing, which is the expensive way to find out.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from biotp.utils import get_device; print(get_device(prefer_gpu=True))"   # expect: cuda

# 3. Confirm peft attaches on CUDA. The offline suite only proves this against a
#    toy module on CPU.
pytest -q tests/test_train_lora.py

# 4. Stage the data. It lives only on the laptop; --data-root points at it.
rsync -avP ./data/processed/ user@cluster:/path/to/bio-transformer-portfolio/data/processed/

# 5. Pre-warm the checkpoint cache, or 324 tasks each try to download ESM-2 on
#    first use. Put it somewhere the compute nodes can read.
export TORCH_HOME="$HOME/.cache/torch"
python -c "from biotp.embeddings import load_esm2; load_esm2('esm2_t12_35M_UR50D')"

# 6. Run one task by hand before submitting 324. The #SBATCH lines are comments
#    to bash, so this takes exactly the path the scheduler would.
SLURM_ARRAY_TASK_ID=0 bash slurm/submit-finetune.sh
```

### Submitting

```bash
# The array bound comes from the code, never from memory. Prints just the number.
python projects/dms-benchmark/scripts/run_arms.py --rung lora --grid-size   # 324

ARRAY_JOB=$(sbatch --parsable slurm/submit-finetune.sh)

# Combine the shards once every task has succeeded. afterok means a failed task
# blocks aggregation rather than letting it report a short result.
sbatch --dependency=afterok:"$ARRAY_JOB" --wrap \
  "python projects/dms-benchmark/scripts/run_arms.py --rung lora --aggregate"
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
