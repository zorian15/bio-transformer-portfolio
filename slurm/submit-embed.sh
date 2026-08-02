#!/usr/bin/env bash
#SBATCH --job-name=biotp-embed
#SBATCH --partition=campus-new     # default (gtx1080ti / rtx2080ti, ~11 GB)
#SBATCH --gres=gpu:1               # any GPU; pin a type with e.g. gpu:rtx2080ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
# Alternatives: 'short' for quick jobs; 'chorus' for L40S (48 GB) on the largest
# ESM-2 sizes. Check limits with: scontrol show partition <name>.
# If a job is rejected for QOS, add a line like: #SBATCH --qos=normal
set -euo pipefail

# Same env as local. biotp.utils.get_device() picks cuda here, mps on the laptop.
# `conda activate`, not `mamba activate`: the mamba shell function comes from
# mamba.sh, which this has not sourced, while conda.sh always provides conda
# activate. The env is the same one either way.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biollm

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
