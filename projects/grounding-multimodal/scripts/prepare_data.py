"""Build the grounding-multimodal dataset: DeepLoc labels joined to UniProt text.

Produces one table with everything the MVP arms need, keyed by accession:
sequence and localization label from DeepLoc 1.0, plus free-text function
annotations, GO terms, and keywords from UniProt. The sequence arms read the
sequence, the grounded arms read the text columns, and both draw the label and
the official test-split flag from the same rows.

Run from the repo root:

    python projects/grounding-multimodal/scripts/prepare_data.py

Raw downloads land in `data/raw/`, the joined table in `data/processed/`, both
gitignored. Re-running skips the download when the raw file is already present;
pass --force-download to refetch.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

DEEPLOC_URL = (
    "https://services.healthtech.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta"
)
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

# The ten single-compartment classes of the DeepLoc 1.0 localization task.
SINGLE_LOCALIZATIONS = frozenset(
    {
        "Cell.membrane",
        "Cytoplasm",
        "Endoplasmic.reticulum",
        "Extracellular",
        "Golgi.apparatus",
        "Lysosome/Vacuole",
        "Mitochondrion",
        "Nucleus",
        "Peroxisome",
        "Plastid",
    }
)

# Proteins annotated in both cytoplasm and nucleus carry this compound label.
# They are excluded from the single-label task; DECISION_LOG.md records the
# rationale and the exact count dropped.
DUAL_LOCALIZATION = "Cytoplasm-Nucleus"

# Membrane-bound, soluble, or unknown. Retained for the binary DeepLoc task,
# which the MVP does not use but which costs nothing to carry along.
SOLUBILITY_CODES = frozenset({"M", "S", "U"})

# What to ask UniProt for. These are API field ids, which are NOT the column
# titles that come back in the TSV; sending the titles is a 400.
UNIPROT_REQUEST_FIELDS = (
    "accession",
    "cc_function",
    "go_c",
    "go_p",
    "go_f",
    "keyword",
)

# TSV column titles UniProt returns, mapped to the names used downstream.
UNIPROT_COLUMN_RENAMES = {
    "Entry": "accession",
    "Function [CC]": "function_text",
    "Gene Ontology (cellular component)": "go_cellular_component",
    "Gene Ontology (biological process)": "go_biological_process",
    "Gene Ontology (molecular function)": "go_molecular_function",
    "Keywords": "keywords",
}

ANNOTATION_COLUMNS = tuple(
    name for name in UNIPROT_COLUMN_RENAMES.values() if name != "accession"
)

UNIPROT_BATCH_SIZE = 100
UNIPROT_MAX_RETRIES = 4
UNIPROT_RETRY_BACKOFF_SECONDS = 2.0


@dataclass(frozen=True)
class DeepLocRecord:
    """One DeepLoc entry: sequence, label, and which split it belongs to."""

    accession: str
    sequence: str
    localization: str
    solubility: str
    is_test: bool


def download_deeploc(destination: Path, force: bool) -> Path:
    """Download the DeepLoc 1.0 FASTA, skipping the fetch if already present."""
    if destination.exists() and not force:
        print(f"raw FASTA already present, reusing {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {DEEPLOC_URL}")
    response = requests.get(DEEPLOC_URL, timeout=120)
    response.raise_for_status()
    assert response.content, "DeepLoc download returned an empty body"
    destination.write_bytes(response.content)
    print(f"wrote {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
    return destination


def parse_header(header: str) -> tuple[str, str, str, bool]:
    """Parse a DeepLoc FASTA header into accession, localization, solubility, split.

    Headers look like "Q9H400 Cell.membrane-M test", where the trailing token is
    present only for the official test partition. Anything not matching this
    grammar is an error rather than something to skip quietly.
    """
    tokens = header.split()
    assert len(tokens) in (2, 3), f"unexpected header field count: {header!r}"

    accession, label_token = tokens[0], tokens[1]
    is_test = len(tokens) == 3
    if is_test:
        assert tokens[2] == "test", f"unexpected third header token: {header!r}"

    localization, _, solubility = label_token.rpartition("-")
    assert localization, f"could not split localization from label: {label_token!r}"
    assert (
        solubility in SOLUBILITY_CODES
    ), f"unexpected solubility code {solubility!r} in {label_token!r}"
    assert (
        localization in SINGLE_LOCALIZATIONS or localization == DUAL_LOCALIZATION
    ), f"unexpected localization {localization!r} in {label_token!r}"

    return accession, localization, solubility, is_test


def parse_deeploc_fasta(text: str) -> list[DeepLocRecord]:
    """Parse the DeepLoc FASTA into records, keeping dual-localization entries.

    Filtering to the single-label task happens later so the caller can report how
    many records were dropped rather than losing that count silently.
    """
    records: list[DeepLocRecord] = []
    header: str | None = None
    sequence_lines: list[str] = []

    def flush() -> None:
        if header is None:
            return
        accession, localization, solubility, is_test = parse_header(header)
        sequence = "".join(sequence_lines).strip().upper()
        assert sequence, f"{accession} has an empty sequence"
        assert sequence.isalpha(), f"{accession} has non-alphabetic residues"
        records.append(
            DeepLocRecord(
                accession=accession,
                sequence=sequence,
                localization=localization,
                solubility=solubility,
                is_test=is_test,
            )
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:]
            sequence_lines = []
        else:
            assert header is not None, "sequence data appeared before any header"
            sequence_lines.append(line)
    flush()

    assert records, "parsed no records from the DeepLoc FASTA"
    accessions = [record.accession for record in records]
    assert len(accessions) == len(set(accessions)), "duplicate accessions in FASTA"
    return records


def base_accession(accession: str) -> str:
    """Strip a UniProt isoform suffix, so P22462-2 looks up entry P22462.

    Queried directly, an isoform accession returns a row with empty annotation
    fields; the parent entry is what carries the function text and GO terms. The
    localization label still comes from DeepLoc per isoform, so only the text is
    inherited, and `annotation_from_parent_entry` marks where that happened.
    """
    return accession.split("-", 1)[0]


def _describe(error: requests.RequestException) -> str:
    """Summarize a failed request without dumping a multi-kilobyte URL."""
    message = str(error)[:200]
    response = error.response
    if response is None:
        return message
    return f"{message} | body: {response.text[:200]}"


def _is_transient(error: requests.RequestException) -> bool:
    """Decide whether retrying could plausibly help.

    Timeouts, dropped connections, and 5xx are worth another attempt. A 4xx is
    deterministic, so retrying only delays the traceback and hides the cause.
    """
    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True
    response = error.response
    return response is not None and response.status_code >= 500


def _request_uniprot_batch(accessions: list[str]) -> str:
    """Fetch one batch of UniProt annotations as TSV, retrying transient faults."""
    params = {
        "query": f"accession:({' OR '.join(accessions)})",
        "fields": ",".join(UNIPROT_REQUEST_FIELDS),
        "format": "tsv",
        "size": str(len(accessions)),
    }

    last_error: requests.RequestException | None = None
    for attempt in range(UNIPROT_MAX_RETRIES):
        try:
            response = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=120)
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            if not _is_transient(error):
                raise RuntimeError(
                    f"UniProt rejected the request: {_describe(error)}"
                ) from error
            last_error = error
            sleep_for = UNIPROT_RETRY_BACKOFF_SECONDS * (attempt + 1)
            print(
                f"  transient failure ({_describe(error)}), retry in {sleep_for:.0f}s"
            )
            time.sleep(sleep_for)

    raise RuntimeError(
        f"UniProt request failed after {UNIPROT_MAX_RETRIES} attempts: "
        f"{_describe(last_error) if last_error else 'unknown error'}"
    ) from last_error


def fetch_uniprot_annotations(accessions: list[str]) -> pd.DataFrame:
    """Fetch function text, GO terms, and keywords, indexed by entry accession.

    Accessions UniProt no longer serves are simply absent from the result; the
    caller decides what that means.
    """
    unique = sorted(set(accessions))
    frames: list[pd.DataFrame] = []
    total_batches = (len(unique) + UNIPROT_BATCH_SIZE - 1) // UNIPROT_BATCH_SIZE

    for batch_index in range(total_batches):
        start = batch_index * UNIPROT_BATCH_SIZE
        batch = unique[start : start + UNIPROT_BATCH_SIZE]
        tsv = _request_uniprot_batch(batch)

        frame = pd.read_csv(
            io.StringIO(tsv),
            sep="\t",
            dtype=str,
            quoting=csv.QUOTE_NONE,
            keep_default_na=False,
        )
        missing = set(UNIPROT_COLUMN_RENAMES) - set(frame.columns)
        assert not missing, f"UniProt response missing columns: {missing}"
        renamed = frame.rename(columns=UNIPROT_COLUMN_RENAMES)
        frames.append(renamed[["accession", *ANNOTATION_COLUMNS]])

        if batch_index % 20 == 0 or batch_index == total_batches - 1:
            print(
                f"  batch {batch_index + 1}/{total_batches}: "
                f"requested {len(batch)}, received {len(frame)}"
            )

    annotations = pd.concat(frames, ignore_index=True)
    annotations = annotations.drop_duplicates(subset="accession")
    return annotations.set_index("accession")


def build_table(
    records: list[DeepLocRecord], annotations: pd.DataFrame
) -> pd.DataFrame:
    """Join DeepLoc records to UniProt annotations, one row per protein."""
    deeploc = pd.DataFrame(
        {
            "accession": [record.accession for record in records],
            "sequence": [record.sequence for record in records],
            "localization": [record.localization for record in records],
            "solubility": [record.solubility for record in records],
            "is_test": [record.is_test for record in records],
        }
    )
    deeploc["entry_accession"] = deeploc["accession"].map(base_accession)
    deeploc["annotation_from_parent_entry"] = (
        deeploc["entry_accession"] != deeploc["accession"]
    )

    table = deeploc.join(annotations, on="entry_accession")
    for column in ANNOTATION_COLUMNS:
        # Absent accessions join as NaN; an empty string is the honest value.
        table[column] = table[column].fillna("")
        # Collapse whitespace so the text survives a TSV round trip intact.
        table[column] = table[column].str.replace(r"\s+", " ", regex=True).str.strip()

    table["has_function_text"] = table["function_text"].str.len() > 0
    assert len(table) == len(records), "join changed the row count"
    return table


def report(table: pd.DataFrame) -> None:
    """Print the counts that decide whether this dataset is usable."""
    single = table[table["localization"] != DUAL_LOCALIZATION]
    print("\n--- dataset summary ---")
    print(f"proteins parsed:              {len(table)}")
    print(f"dual-localization dropped:    {len(table) - len(single)}")
    print(f"single-label proteins:        {len(single)}")
    print(f"  official test split:        {int(single['is_test'].sum())}")
    print(f"  train/val pool:             {int((~single['is_test']).sum())}")
    print(f"localization classes:         {single['localization'].nunique()}")
    print(
        "isoforms using parent text:   "
        f"{int(single['annotation_from_parent_entry'].sum())}"
    )

    for column in ("function_text", "go_cellular_component", "keywords"):
        non_empty = int((single[column].str.len() > 0).sum())
        share = non_empty / len(single) * 100
        print(f"non-empty {column:24} {non_empty:6} ({share:.1f}%)")

    print("\nclass balance (single-label):")
    for name, count in single["localization"].value_counts().items():
        print(f"  {name!s:24} {count:5} ({count / len(single) * 100:.1f}%)")

    majority = single["localization"].value_counts().iloc[0] / len(single)
    print(f"\nmajority-class accuracy floor: {majority:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="root of the gitignored data directory (default: data)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="refetch the DeepLoc FASTA even if it is already on disk",
    )
    parser.add_argument(
        "--skip-uniprot",
        action="store_true",
        help="parse the FASTA only, without querying UniProt (for quick checks)",
    )
    args = parser.parse_args()

    raw_fasta = args.data_root / "raw" / "deeploc_data.fasta"
    output_path = args.data_root / "processed" / "deeploc_annotated.parquet"

    download_deeploc(raw_fasta, force=args.force_download)
    records = parse_deeploc_fasta(raw_fasta.read_text())
    print(f"parsed {len(records)} DeepLoc records")

    if args.skip_uniprot:
        annotations = pd.DataFrame(
            columns=list(ANNOTATION_COLUMNS), dtype=str
        ).rename_axis("accession")
    else:
        print(f"fetching UniProt annotations in batches of {UNIPROT_BATCH_SIZE}")
        annotations = fetch_uniprot_annotations(
            [base_accession(record.accession) for record in records]
        )
        print(f"received annotations for {len(annotations)} entries")

    table = build_table(records, annotations)
    report(table)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    print(f"\nwrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
