#!/usr/bin/env bash
#SBATCH --job-name=biotp-lora
#SBATCH --partition=chorus         # L40S (48 GB), best for fine-tunes / larger models
#SBATCH --gres=gpu:1               # one card per task; LoRA on 35M needs nothing more
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-323%16
#SBATCH --output=slurm-%x-%A_%a.out
# 'chorus' has the 48 GB L40S cards; fall back to 'campus-new' (11 GB) for small
# models or LoRA on ESM-2 <= 150M. Check limits with: scontrol show partition <name>.
# If a job is rejected for QOS, add a line like: #SBATCH --qos=normal
#
# --array=0-323 is rung 3's whole grid: 3 assays x 3 schemes x 3 readouts x 4
# training sizes x 3 seeds. Do not retype 324 from memory; regenerate it with
#   python projects/dms-benchmark/scripts/run_arms.py --rung lora --grid-size
# and change the bound in the same commit that changes an axis. A bound set too
# low finishes cleanly having skipped configurations, which is exactly the silent
# shortfall --aggregate refuses to write.
#
# MaxArraySize on this cluster is 50001 (scontrol show config | grep MaxArraySize),
# so 324 is nowhere near the limit and needs no chunking.
#
# %16 caps concurrently running tasks. It is a courtesy to the partition rather
# than a correctness constraint: tasks are independent and write to separate
# files, so any value is safe. Raise it if the queue is empty.
#
# --time is per task, not for the whole array, and tasks are far from uniform.
# Per-epoch encoder work scales as 3N + 256, so across N in (32, 128, 512, 2048)
# the per-epoch costs are 352, 640, 1792 and 6400: the N=2048 arms are 18x the
# N=32 ones and take about 70% of a full N curve between them. Against the
# measured 1.76 h per (assay, scheme, readout, seed) curve on MPS, that is ~1.2 h
# for the worst task and ~4 minutes for the cheapest. An L40S is 8-15x faster,
# putting the worst case at 5 to 10 minutes. One hour is slack, not an estimate.
set -euo pipefail

# Same env as local. biotp.utils.get_device() picks cuda here, mps on the laptop.
# PYTORCH_ENABLE_MPS_FALLBACK is deliberately absent: it is a laptop-only setting
# and exporting it on a CUDA node would be cargo cult.
source "$(conda info --base)/etc/profile.d/conda.sh"
mamba activate biollm

# Point the checkpoint caches at storage the compute nodes share, so 324 tasks do
# not each try to download ESM-2 on first use. Pre-warm once before submitting;
# see slurm/README.md.
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

DATA_ROOT="${DATA_ROOT:-data}"
RESULTS_DIR="${RESULTS_DIR:-projects/dms-benchmark/results}"

# One array task is one configuration. The index maps onto the grid inside
# run_arms.py rather than being unpacked into flags by shell arithmetic here, so
# the mapping is covered by tests/test_dms_run_arms.py. An off-by-one in a shell
# expression would be invisible until the results came up short.
#
# Each task writes its own shard under <results-dir>/lora_shards/, named after the
# configuration. Nothing here writes lora.csv. Combine the shards afterwards:
#   python projects/dms-benchmark/scripts/run_arms.py --rung lora --aggregate
# which refuses to write unless every configuration produced one.
python projects/dms-benchmark/scripts/run_arms.py \
    --rung lora \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --data-root "${DATA_ROOT}" \
    --results-dir "${RESULTS_DIR}"

# To debug one configuration without the scheduler, run a task by hand:
#   SLURM_ARRAY_TASK_ID=0 bash slurm/submit-finetune.sh
# The #SBATCH lines are comments to bash, so this takes exactly the same path.
