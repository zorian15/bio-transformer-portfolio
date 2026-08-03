#!/usr/bin/env bash
#SBATCH --job-name=biotp-frozen
#SBATCH --partition=chorus         # L40S (48 GB); the 650M embedding pass wants the room
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# Into slurm-logs/, which is tracked but whose contents are gitignored.
# SLURM resolves this path before the script runs and will not create the
# directory, so it is kept in the repo by slurm-logs/.gitkeep. Paths are
# relative to the submitting directory, so submit from the repo root.
#SBATCH --output=slurm-logs/%x-%j.out
#
# Rung 2 of the DMS ladder: frozen embeddings into the same MLP head rung 3
# trains. One job rather than an array, because unlike rung 3 this rung is cheap
# once its embeddings exist and the embedding cache is the expensive part: a
# single process embeds each (assay, checkpoint, readout) once and then reuses
# it across all 108 arms at that size. Split across array tasks, every task
# would take the same cache miss and the 650M encoder would be run hundreds of
# times over. That was measured on the 35M grid: embedding per-arm rather than
# per-assay would have recomputed 324 times instead of 9.
#
# Why this exists at all: `submit-embed.sh` is an unwired stub, and rung 2 at
# 650M is not something to run on a laptop. The 35M grid takes ~750 s on MPS
# with a warm cache; at 650M the embedding pass alone is the dominant cost and
# an L40S is 8-15x faster.
#
# Walltime: the 35M half is ~750 s warm. The 650M half adds one forward pass per
# variant per readout over 16,866 variants, then trains 324 arms whose head is
# the same size but whose input is 1280-wide rather than 480. Four hours is
# slack rather than an estimate.
#
# Run after this finishes, from the repo root:
#   python projects/dms-benchmark/scripts/run_arms.py --rung zero_shot --all
# Rung 1 is nearly free at any size and needs no GPU allocation of its own.
set -euo pipefail

# Same env as local. biotp.utils.get_device() picks cuda here, mps on the laptop.
# PYTORCH_ENABLE_MPS_FALLBACK is deliberately absent: it is a laptop-only setting
# and exporting it on a CUDA node would be cargo cult.
# Activate the environment. BIOTP_ENV_ACTIVATE points at an activation script for
# sites where conda is not the working answer: this cluster's conda channels
# resolve to IPv6-only and are unreachable, so the env there is a venv built from
# PyPI (see slurm/README.md). Unset, this uses the conda env the repo documents.
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
    # below exists to explain.
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "${conda_base}/etc/profile.d/conda.sh"
        conda activate biollm || true
    fi
fi

# Preflight. Both checks exist because their failures are otherwise reported as
# something generic with the real cause nowhere in the message. This job is a
# single long run rather than an array, so a silent CPU fallback here does not
# fail: it finishes hours late, which is worse.
if ! python -c "import biotp" 2>/dev/null; then
    echo "preflight: cannot import biotp after activating the environment." >&2
    echo "  BIOTP_ENV_ACTIVATE = ${BIOTP_ENV_ACTIVATE:-<unset>}" >&2
    echo "  python             = $(command -v python || echo '<none on PATH>')" >&2
    echo >&2
    echo "Either export BIOTP_ENV_ACTIVATE to your venv's activate script, or" >&2
    echo "build the conda env named biollm. See slurm/README.md." >&2
    exit 1
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        echo "preflight: a GPU was allocated (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})" >&2
        echo "but torch cannot see it, so this is a CPU build or a driver mismatch." >&2
        echo "The 650M embedding pass on CPU is the expensive way to discover it." >&2
        echo "  python -c 'import torch; print(torch.__version__, torch.version.cuda)'" >&2
        exit 1
    fi
fi

# --all rather than --task-id: one process, one owner of frozen.csv, no shards
# to aggregate afterwards. See the note above on why this rung is not an array.
python projects/dms-benchmark/scripts/run_arms.py \
    --rung frozen \
    --all
