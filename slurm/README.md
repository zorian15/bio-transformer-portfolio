# slurm/

`sbatch` templates for the GPU-heavy one-offs, meant to be submitted while SLURM access lasts. The same code runs locally on the MacBook via MPS (see `biotp.utils.get_device`); these just point it at a CUDA node for speed and for the larger ESM-2 sizes. Use the same `biollm` mamba env on the cluster.

## This cluster
- Default partition `campus-new`: GTX 1080 Ti / RTX 2080 Ti, ~11 GB. Fine for embedding extraction up to ESM-2 650M at a modest batch size.
- `chorus`: L40S, 48 GB, 4 per node. Use for the largest models and for fine-tunes.
- `short` for quick jobs; `interactive` for interactive sessions.
- Account is `matsen_e` (left out of the templates for now); add `#SBATCH --account=matsen_e` if the scheduler requires it. Available QOS include `normal`.

## Templates
- `embed.sbatch`: extract and cache ESM-2 embeddings (the main one-off). Run it at every model size you might want, so the cached vectors survive even if cluster access ends.
- `finetune.sbatch`: a LoRA or full fine-tune for runs too heavy for the laptop.

Edit the `#SBATCH` directives (partition, account, time, resources) for your cluster before submitting, then `sbatch slurm/embed.sbatch`.

## Getting results back (rsync + Hugging Face)
Per the storage plan: git tracks only small files; large artifacts move by rsync, and final checkpoints go to Hugging Face. Nothing large is committed.

```bash
# From the laptop, pull cached embeddings the job wrote on the cluster:
rsync -avP user@cluster:/path/to/bio-transformer-portfolio/data/processed/ ./data/processed/

# Final trained checkpoints go to the Hugging Face Hub (durable, doubles as
# the public release), via biotp.release.push_to_hub(...).
```

Embedding caches are regenerable, so if one is lost you can simply recompute it; only trained checkpoints need the durable Hugging Face home.
