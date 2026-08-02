#!/usr/bin/env bash
#SBATCH --job-name=biotp-embed
#SBATCH --partition=campus-new     # default (gtx1080ti / rtx2080ti, ~11 GB)
#SBATCH --gres=gpu:1               # any GPU; pin a type with e.g. gpu:rtx2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Into slurm-logs/, which is tracked but whose contents are gitignored.
# SLURM resolves this path before the script runs and will not create the
# directory, so it is kept in the repo by slurm-logs/.gitkeep. Paths are
# relative to the submitting directory, so submit from the repo root.
#SBATCH --output=slurm-logs/%x-%j.out
# Alternatives: 'short' for quick jobs; 'chorus' for L40S (48 GB) on the largest
# ESM-2 sizes. Check limits with: scontrol show partition <name>.
# If a job is rejected for QOS, add a line like: #SBATCH --qos=normal
set -euo pipefail

# Same env as local. biotp.utils.get_device() picks cuda here, mps on the laptop.
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

# GPU-heavy one-off: extract and cache ESM-2 embeddings into data/processed/
# so the vectors can be rsync'd to the laptop afterward (see slurm/README.md).
# Run at every size you might want while SLURM access lasts. Embedding extraction
# fits the 11 GB cards up to ~650M with a modest batch size.
# TODO: wire to the project's real extraction entry point once it exists, e.g.:
#   python projects/grounding-multimodal/extract_embeddings.py \
#       --model esm2_t33_650M_UR50D \
#       --sequences data/raw/sequences.fasta \
#       --out data/processed/esm2_650M.npy
echo "TODO: replace with the embedding-extraction command"
