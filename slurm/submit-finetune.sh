#!/usr/bin/env bash
#SBATCH --job-name=biotp-lora
#SBATCH --partition=chorus         # L40S (48 GB), best for fine-tunes / larger models
#SBATCH --gres=gpu:1               # one card per task; LoRA on 35M needs nothing more
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-647%16
# Into slurm-logs/, which is tracked but whose contents are gitignored.
# SLURM resolves this path before the script runs and will not create the
# directory, so it is kept in the repo by slurm-logs/.gitkeep. Paths are
# relative to the submitting directory, so submit from the repo root.
#SBATCH --output=slurm-logs/%x-%A_%a.out
# 'chorus' has the 48 GB L40S cards; fall back to 'campus-new' (11 GB) for small
# models or LoRA on ESM-2 <= 150M. Check limits with: scontrol show partition <name>.
# If a job is rejected for QOS, add a line like: #SBATCH --qos=normal
#
# --array=0-647 is rung 3's whole grid: 2 checkpoints x 3 assays x 3 schemes x 3
# readouts x 4 training sizes x 3 seeds. Do not retype 648 from memory;
# regenerate it with
#   python projects/dms-benchmark/scripts/run_arms.py --rung lora --grid-size
# and change the bound in the same commit that changes an axis. A bound set too
# low finishes cleanly having skipped configurations, which is exactly the silent
# shortfall --aggregate refuses to write.
# tests/test_dms_run_arms.py::test_the_documented_lora_array_bound_is_the_real_one
# compares this literal against the grid, so the two cannot drift.
#
# Checkpoint is the outermost grid axis, so tasks 0-323 are 650M and 324-647
# are 35M. SLURM dispatches an array in ascending index order, so this puts the
# expensive half first on purpose: 650M is the deliverable and 35M is a
# reproduction check of an already-published result. An array that is cancelled
# or preempted part way then leaves the result we came for. Note that a range
# list like --array=324-647,0-323 would NOT achieve this; SLURM schedules by
# index, not by the order the ranges are written.
#
# To run one half alone: --array=0-323%16 is 650M, --array=324-647%16 is 35M.
#
# MaxArraySize on this cluster is 50001 (scontrol show config | grep MaxArraySize),
# so 648 is nowhere near the limit and needs no chunking.
#
# %16 caps concurrently running tasks. It is a courtesy to the partition rather
# than a correctness constraint: tasks are independent and write to separate
# files, so any value is safe. Raise it if the queue is empty.
#
# --time is per task, not for the whole array, and tasks are far from uniform in
# two dimensions now.
#
# By N: per-epoch encoder work scales as 3N + 256, so across N in (32, 128, 512,
# 2048) the per-epoch costs are 352, 640, 1792 and 6400. The N=2048 arms are 18x
# the N=32 ones and take about 70% of a full N curve between them. Against the
# measured 1.76 h per (assay, scheme, readout, seed) curve on MPS, that is ~1.2 h
# for the worst 35M task and ~4 minutes for the cheapest. An L40S is 8-15x
# faster, putting the worst 35M case at 5 to 10 minutes.
#
# By checkpoint: 650M is 33 layers x 1280 wide against 35M's 12 x 480, so
# roughly (33/12) * (1280/480)^2 ~ 19x the FLOPs per token. That puts the worst
# 650M task near 2 h rather than 10 minutes, which is why one hour is no longer
# slack but a ceiling the expensive half would hit. Six hours is the new slack.
# A task that exceeds it is killed and writes no shard, and --aggregate then
# names the missing configuration rather than writing a short file, so the
# failure is loud; it is still a wasted allocation.
set -euo pipefail

# Same env as local. biotp.utils.get_device() picks cuda here, mps on the laptop.
# PYTORCH_ENABLE_MPS_FALLBACK is deliberately absent: it is a laptop-only setting
# and exporting it on a CUDA node would be cargo cult.
# Activate the environment. BIOTP_ENV_ACTIVATE points at an activation script for
# sites where conda is not the working answer: this cluster's conda channels
# resolve to IPv6-only and are unreachable, so the env there is a venv built from
# PyPI (see slurm/README.md). Unset, this uses the conda env the repo documents.
#
# Checked rather than assumed: a wrong path would otherwise fail identically in
# every one of the array's tasks, with the real cause buried in 324 log files.
if [ -n "${BIOTP_ENV_ACTIVATE:-}" ]; then
    test -f "${BIOTP_ENV_ACTIVATE}" || {
        echo "BIOTP_ENV_ACTIVATE=${BIOTP_ENV_ACTIVATE} is not a file" >&2
        exit 1
    }
    # shellcheck source=/dev/null
    source "${BIOTP_ENV_ACTIVATE}"
else
    # Resolved into a variable first, and every step allowed to fail. A command
    # substitution inside an `if` condition is NOT covered by the condition's
    # exemption from `set -e`, so `if source "$(conda info --base)/..."` exits the
    # script the moment conda is absent, which is exactly the case the preflight
    # below exists to explain. Verified rather than assumed.
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "${conda_base}/etc/profile.d/conda.sh"
        conda activate biollm || true
    fi
fi

# Preflight. Both checks exist because their failures are otherwise reported as
# something generic, once per task, with the real cause nowhere in the message.
if ! python -c "import biotp" 2>/dev/null; then
    echo "preflight: cannot import biotp after activating the environment." >&2
    echo "  BIOTP_ENV_ACTIVATE = ${BIOTP_ENV_ACTIVATE:-<unset>}" >&2
    echo "  python             = $(command -v python || echo '<none on PATH>')" >&2
    echo >&2
    echo "Either export BIOTP_ENV_ACTIVATE to your venv's activate script, or" >&2
    echo "build the conda env named biollm. Forgetting the export is the common" >&2
    echo "case: unset, this script falls back to conda, and on a machine where" >&2
    echo "conda was never usable that fails with no indication why." >&2
    echo "See slurm/README.md." >&2
    exit 1
fi

# Only meaningful when the scheduler actually allocated a GPU. Run by hand on a
# login node, CUDA_VISIBLE_DEVICES is unset and this is skipped, which keeps the
# documented single-task debug path working.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        echo "preflight: a GPU was allocated (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})" >&2
        echo "but torch cannot see it, so this is a CPU build or a driver mismatch." >&2
        echo "That does not fail on its own, it runs roughly 50x slow, which across" >&2
        echo "324 tasks is the expensive way to discover it. Check with:" >&2
        echo "  python -c 'import torch; print(torch.__version__, torch.version.cuda)'" >&2
        exit 1
    fi
fi

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
