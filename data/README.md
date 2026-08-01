# data/

Datasets live here and are gitignored (only this README is tracked). Keep raw downloads under `data/raw/` and derived/processed files under `data/processed/`. Never commit data or model weights.

## Where to fetch each project's data

**grounding-multimodal (Project 1)**
- Swiss-Prot / UniProt sequences plus function annotations (function text, GO terms, keywords): uniprot.org (bulk download or the REST API).
- A clean downstream benchmark, e.g. DeepLoc for subcellular localization.

**dms-benchmark (Project 2)**
- ProteinGym: the public deep mutational scanning benchmark (substitutions and indels), from the ProteinGym release.
- Optionally, one of your own viral DMS datasets for a domain-flavored variant.

**epistasis-plm-torchdms (Project 3)**
- Starr/Bloom SARS-CoV-2 RBD DMS, the barcoded libraries carrying variable numbers
  of mutations per variant. ProteinGym ships only the single-mutant summary
  (`SPIKE_SARS2_Starr_2020_binding` and `_expression`), so the multi-mutant data
  comes from the Bloom lab release upstream of it.
- `torchdms` runs in its own environment: it pins `python_requires=">=3.8,<3.10"`,
  so it cannot share the Python 3.11 `biollm` env.

## Conventions
- Record the exact download date and version/URL of each dataset in the relevant project's `DECISION_LOG.md`, since these resources are updated over time.
- Preprocessing that produces `data/processed/` should be a script or notebook checked into the project, so the derived data is reproducible from raw.
