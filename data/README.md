# data/

Datasets live here and are gitignored (only this README is tracked). Keep raw downloads under `data/raw/` and derived/processed files under `data/processed/`. Never commit data or model weights.

## Where to fetch each project's data

**grounding-multimodal (Project 2)**
- Swiss-Prot / UniProt sequences plus function annotations (function text, GO terms, keywords): uniprot.org (bulk download or the REST API).
- A clean downstream benchmark, e.g. DeepLoc for subcellular localization.

**dms-benchmark (Project 1)**
- ProteinGym: the public deep mutational scanning benchmark (substitutions and indels), from the ProteinGym release.
- Optionally, one of your own viral DMS datasets for a domain-flavored variant.

**tcr-antibody-lm (Project 3)**
- VDJdb and/or IEDB for TCR-epitope pairs.
- Optionally an antibody escape / DMS dataset from the polyclonal work.

## Conventions
- Record the exact download date and version/URL of each dataset in the relevant project's `DECISION_LOG.md`, since these resources are updated over time.
- Preprocessing that produces `data/processed/` should be a script or notebook checked into the project, so the derived data is reproducible from raw.
