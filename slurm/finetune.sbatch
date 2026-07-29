#!/usr/bin/env bash
#SBATCH --job-name=biotp-finetune
#SBATCH --partition=chorus         # L40S (48 GB), best for fine-tunes / larger models
#SBATCH --gres=gpu:1               # bump to gpu:2+ only if a run needs it
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
# 'chorus' has the 48 GB L40S cards; fall back to 'campus-new' (11 GB) for small
# models or LoRA on ESM-2 <= 150M. Check limits with: scontrol show partition <name>.
# If a job is rejected for QOS, add a line like: #SBATCH --qos=normal
set -euo pipefail

# Same env as local. biotp.utils.get_device() picks cuda here, mps on the laptop.
source "$(conda info --base)/etc/profile.d/conda.sh"
mamba activate biollm

# GPU-heavy fine-tune (LoRA or full). Checkpoint to weights/ (gitignored), then
# push the final artifact to the Hugging Face Hub via biotp.release.push_to_hub(...).
# TODO: wire to the project's real training entry point once it exists, e.g.:
#   python projects/dms-benchmark/finetune.py \
#       --model esm2_t30_150M_UR50D --mode lora \
#       --data data/processed/proteingym --out weights/dms_lora
echo "TODO: replace with the fine-tune command"
