"""Build the dms-benchmark dataset: ProteinGym DMS assays under the issue #11 filter.

Produces one table of variants keyed by assay, plus a sidecar of per-assay
metadata. The zero-shot arm reads the wild-type sequence and the parsed
substitutions; the supervised arms read the same rows plus ProteinGym's
cross-validation fold assignments.

Two things this script exists to get right, both of which fail silently otherwise:

**Numbering.** ProteinGym mutant strings such as `A24G` are 1-based against the
target sequence that assay shipped with, which is not always the UniProt
canonical one. Pairing the numbering with a different sequence shifts every
position by a constant and leaves every score finite and plausible. The reference
file carries `target_seq`, so the sequence is taken from there and never fetched
separately, and the agreement is asserted here rather than 40 minutes into a run.

**Fold assignments.** The main substitutions archive does *not* carry them. They
ship in `cv_folds_singles_substitutions.zip`, whose CSVs are a strict superset of
the main ones, so that archive is the only download needed. Using ProteinGym's
own folds rather than deriving them keeps the numbers comparable to published
supervised baselines.

Both inputs come from ProteinGym's Zenodo deposit, pinned by DOI, so the record
that produced a committed result is immutable and citable. See the constants
below for why that is the source rather than the lab web server.

Run from the repo root:

    python projects/dms-benchmark/scripts/prepare_data.py

Raw downloads land in `data/raw/`, the prepared table in `data/processed/`, both
gitignored. Re-running skips the download when the raw file is present; pass
--force-download to refetch.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from biotp.runlog import DEFAULT_LOG_DIR, RunLog, get_logger, run_context
from biotp.zero_shot import parse_substitutions

# Fetched from ProteinGym's Zenodo deposit rather than the lab web server, for
# two reasons.
#
# Provenance: a DOI names an immutable record, so "which inputs produced this
# table" has an answer that outlives a URL. The lab host also publishes
# unversioned paths that resolve, which would let the inputs move under committed
# results with nothing to see in a diff.
#
# Practical: the lab host presents its certificate chain out of order, and
# Python's OpenSSL 3.6 declines to build a path through it even with certifi's
# bundle passed explicitly, while curl's LibreSSL accepts the same chain. Working
# around that would have meant either weakening verification or depending on
# whichever TLS stack the machine happens to have. Zenodo needs neither.
#
# Both files were checked byte-for-byte against the lab host before the switch.
PROTEINGYM_VERSION = "v1.3"
PROTEINGYM_DOI = "10.5281/zenodo.15293562"
PROTEINGYM_RECORD = "15293562"
PROTEINGYM_BASE = f"https://zenodo.org/records/{PROTEINGYM_RECORD}/files"
REFERENCE_FILENAME = "DMS_substitutions.csv"
CV_FOLDS_ARCHIVE = "cv_folds_singles_substitutions.zip"
CV_FOLDS_MEMBER_DIR = "cv_folds_singles_substitutions"

# The pre-registered filter from issue #11. These are the experiment's terms, not
# tuning knobs: they were fixed before any assay was downloaded, and changing one
# after seeing a result is the selection bias the pre-registration exists to
# prevent. Length also bounds cost, since attention is quadratic in it.
MAX_TARGET_LENGTH = 400
MIN_SINGLE_MUTANTS = 2000
MAX_SINGLE_MUTANTS = 8000

# One assay per taxon, the alphabetically-first DMS_id within each. Amended on
# 2026-07-31, before any outcome data existed, from "one viral plus two
# non-viral": the flat rule returned two Pseudomonas enzymes out of three, which
# makes a generality claim thin for no gain. See the amendment on issue #11.
SELECTED_TAXA = ("Virus", "Prokaryote", "Human")

# ProteinGym's three cross-validation schemes. All three are reported; `modulo`
# and `contiguous` hold out residue positions rather than rows, which is the
# leakage-aware default this repo uses everywhere.
FOLD_COLUMNS = ("fold_random_5", "fold_modulo_5", "fold_contiguous_5")

REQUIRED_ASSAY_COLUMNS = ("mutant", "mutated_sequence", "DMS_score", *FOLD_COLUMNS)

REQUIRED_REFERENCE_COLUMNS = (
    "DMS_id",
    "taxon",
    "target_seq",
    "seq_len",
    "includes_multiple_mutants",
    "DMS_number_single_mutants",
)

log = get_logger("dms-prepare-data")


def download(url: str, destination: Path, force: bool) -> Path:
    """Fetch `url` to `destination`, skipping the transfer if it is already there."""
    if destination.exists() and not force:
        log.info(f"using cached {destination} ({destination.stat().st_size} bytes)")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"downloading {url}")
    response = requests.get(url, timeout=600)
    assert response.status_code == 200, (
        f"GET {url} returned {response.status_code}. The inputs are pinned to "
        f"Zenodo record {PROTEINGYM_RECORD} (ProteinGym {PROTEINGYM_VERSION}, DOI "
        f"{PROTEINGYM_DOI}); check that the record and file name still resolve."
    )
    destination.write_bytes(response.content)
    log.info(f"wrote {destination} ({destination.stat().st_size} bytes)")
    return destination


def load_reference(path: Path) -> pd.DataFrame:
    """Read ProteinGym's assay-level reference table."""
    reference = pd.read_csv(path)
    missing = set(REQUIRED_REFERENCE_COLUMNS) - set(reference.columns)
    assert not missing, (
        f"reference file {path} is missing {sorted(missing)}; ProteinGym's schema "
        "changed and the filter below no longer means what it says"
    )
    return reference


def filtered_pool(reference: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-registered filter, returning the eligible assays.

    Sorted by DMS_id so the result does not depend on the order rows happen to
    appear in the reference file, which is what makes the selection below
    reproducible rather than incidentally stable.
    """
    assert reference["includes_multiple_mutants"].dtype == bool, (
        "includes_multiple_mutants is not boolean, so `~` would not mean what it "
        "reads as; check ProteinGym's schema"
    )
    pool = reference[
        (~reference["includes_multiple_mutants"])
        & (reference["seq_len"] <= MAX_TARGET_LENGTH)
        & (
            reference["DMS_number_single_mutants"].between(
                MIN_SINGLE_MUTANTS, MAX_SINGLE_MUTANTS
            )
        )
    ]
    return pool.sort_values("DMS_id").reset_index(drop=True)


def select_assays(pool: pd.DataFrame) -> pd.DataFrame:
    """Take the alphabetically-first assay within each pre-registered taxon.

    Deterministic by construction: no sampling, no seed, and no dependence on
    input order. A taxon with nothing left in the pool is an error rather than a
    quietly shorter benchmark.
    """
    chosen = []
    for taxon in SELECTED_TAXA:
        candidates = pool[pool["taxon"] == taxon]
        assert not candidates.empty, (
            f"no {taxon} assay survives the filter (length <= {MAX_TARGET_LENGTH}, "
            f"single mutants in [{MIN_SINGLE_MUTANTS}, {MAX_SINGLE_MUTANTS}]). "
            "Relax the filter on issue #11 before looking at any result, not after."
        )
        chosen.append(candidates.iloc[0])

    selected = pd.DataFrame(chosen).reset_index(drop=True)
    assert selected["DMS_id"].is_unique, "an assay was selected for two taxa"
    return selected


def extract_assay_csvs(archive: Path, dms_ids: list[str], destination: Path) -> None:
    """Pull only the selected assays out of the folds archive."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = {Path(name).name: name for name in bundle.namelist()}
        for dms_id in dms_ids:
            wanted = f"{dms_id}.csv"
            assert wanted in members, (
                f"{wanted} is not in {archive.name}. The reference file lists this "
                "assay but the folds archive does not carry it, so the two "
                "downloads disagree."
            )
            target = destination / wanted
            with bundle.open(members[wanted]) as source:
                target.write_bytes(source.read())


def load_assay(path: Path, dms_id: str) -> pd.DataFrame:
    """Read one assay's variants, asserting the fold columns are present."""
    assay = pd.read_csv(path)
    missing = set(REQUIRED_ASSAY_COLUMNS) - set(assay.columns)
    assert not missing, (
        f"{path.name} is missing {sorted(missing)}. The fold columns live only in "
        f"{CV_FOLDS_ARCHIVE}; the main substitutions archive does not carry them, "
        "and silently deriving substitutes would make these numbers "
        "incomparable to published baselines."
    )
    assay = assay.copy()
    assay.insert(0, "dms_id", dms_id)
    return assay


def parse_variants(assay: pd.DataFrame, target_seq: str, dms_id: str) -> pd.DataFrame:
    """Add zero-based position and residue columns, checking them against the reference.

    This is the numbering guard. A substitution whose stated wild-type residue
    disagrees with `target_seq` means the assay's numbering has been paired with
    the wrong sequence, and every score computed afterwards would describe a
    different residue while looking entirely reasonable.
    """
    positions: list[int] = []
    wildtype_residues: list[str] = []
    mutant_residues: list[str] = []

    for mutant, mutated_sequence in zip(assay["mutant"], assay["mutated_sequence"]):
        substitutions = parse_substitutions(str(mutant), one_based=True)
        assert len(substitutions) == 1, (
            f"{dms_id} variant {mutant!r} has {len(substitutions)} substitutions; "
            "this cohort is filtered to single substitutions only"
        )
        position, wildtype_aa, mutant_aa = substitutions[0]

        assert 0 <= position < len(target_seq), (
            f"{dms_id} variant {mutant!r} indexes position {position} of a "
            f"{len(target_seq)}-residue reference"
        )
        assert target_seq[position] == wildtype_aa, (
            f"{dms_id} variant {mutant!r} states wild-type {wildtype_aa!r} at "
            f"position {position}, but the reference has "
            f"{target_seq[position]!r}: the assay numbering and target_seq disagree"
        )
        assert str(mutated_sequence)[position] == mutant_aa, (
            f"{dms_id} variant {mutant!r} does not carry {mutant_aa!r} at position "
            f"{position} of its own mutated_sequence"
        )

        positions.append(position)
        wildtype_residues.append(wildtype_aa)
        mutant_residues.append(mutant_aa)

    parsed = assay.copy()
    parsed["position"] = positions
    parsed["wildtype_aa"] = wildtype_residues
    parsed["mutant_aa"] = mutant_residues
    return parsed


def build_metadata(selected: pd.DataFrame, variants: pd.DataFrame) -> dict[str, Any]:
    """Per-assay metadata, including the wild-type sequence the arms need."""
    metadata: dict[str, Any] = {
        "proteingym_version": PROTEINGYM_VERSION,
        "proteingym_doi": PROTEINGYM_DOI,
        "filter": {
            "max_target_length": MAX_TARGET_LENGTH,
            "min_single_mutants": MIN_SINGLE_MUTANTS,
            "max_single_mutants": MAX_SINGLE_MUTANTS,
            "selected_taxa": list(SELECTED_TAXA),
            "rule": "alphabetically-first DMS_id within each taxon",
        },
        "assays": {},
    }
    for row in selected.itertuples():
        rows = variants[variants["dms_id"] == row.DMS_id]
        metadata["assays"][row.DMS_id] = {
            "taxon": row.taxon,
            "target_seq": row.target_seq,
            "target_length": int(row.seq_len),
            "variants": len(rows),
            "distinct_positions": int(rows["position"].nunique()),
            "source_organism": getattr(row, "source_organism", None),
        }
    return metadata


def record_counts(run: RunLog, selected: pd.DataFrame, variants: pd.DataFrame) -> None:
    """Put every number a writeup would cite onto the run manifest."""
    run.record("proteingym_version", PROTEINGYM_VERSION)
    run.record("proteingym_doi", PROTEINGYM_DOI)
    run.record("assays_selected", len(selected))
    run.record("variants_total", len(variants))
    for dms_id, rows in variants.groupby("dms_id"):
        run.record(f"variants_{dms_id}", len(rows))
        run.record(f"distinct_positions_{dms_id}", int(rows["position"].nunique()))
        for column in FOLD_COLUMNS:
            run.record(f"folds_{column}_{dms_id}", int(rows[column].nunique()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="refetch the ProteinGym files even when they are already present",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    raw = args.data_root / "raw"
    processed = args.data_root / "processed"

    with run_context(
        "dms-prepare-data", log_dir=args.log_dir, params=vars(args)
    ) as run:
        with run.step("download reference"):
            reference_path = download(
                f"{PROTEINGYM_BASE}/{REFERENCE_FILENAME}?download=1",
                raw / REFERENCE_FILENAME,
                args.force_download,
            )
            reference = load_reference(reference_path)

        with run.step("apply the pre-registered filter"):
            pool = filtered_pool(reference)
            selected = select_assays(pool)
            run.record("assays_in_reference", len(reference))
            run.record("assays_passing_filter", len(pool))
            log.info(
                "filter left %d of %d assays; selected %s",
                len(pool),
                len(reference),
                ", ".join(f"{r.DMS_id} ({r.taxon})" for r in selected.itertuples()),
            )

        with run.step("download cross-validation folds"):
            archive = download(
                f"{PROTEINGYM_BASE}/{CV_FOLDS_ARCHIVE}?download=1",
                raw / CV_FOLDS_ARCHIVE,
                args.force_download,
            )
            extract_assay_csvs(
                archive, selected["DMS_id"].tolist(), raw / CV_FOLDS_MEMBER_DIR
            )

        with run.step("parse variants and check numbering"):
            frames = []
            for row in selected.itertuples():
                assay = load_assay(
                    raw / CV_FOLDS_MEMBER_DIR / f"{row.DMS_id}.csv", row.DMS_id
                )
                frames.append(parse_variants(assay, row.target_seq, row.DMS_id))
            variants = pd.concat(frames, ignore_index=True)

        with run.step("write outputs"):
            processed.mkdir(parents=True, exist_ok=True)
            table_path = processed / "proteingym_variants.parquet"
            variants.to_parquet(table_path, index=False)

            metadata_path = processed / "proteingym_assays.json"
            metadata_path.write_text(
                json.dumps(build_metadata(selected, variants), indent=2) + "\n"
            )
            log.info(f"wrote {table_path} and {metadata_path}")

        record_counts(run, selected, variants)

    return 0


if __name__ == "__main__":
    sys.exit(main())
