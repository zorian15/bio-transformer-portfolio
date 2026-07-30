"""Tests for the grounding-multimodal DeepLoc parser.

The script lives under `projects/`, not in the installed `biotp` package, because
data preparation is project-specific by convention (see `data/README.md`). It is
loaded by path here so the parsing logic still gets real tests: the header
grammar carries the label, the solubility code, and the official test-split flag,
and a silent misparse there would quietly corrupt every downstream number.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import requests

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "grounding-multimodal"
    / "scripts"
    / "prepare_data.py"
)


def load_prepare_data() -> ModuleType:
    """Import the project script by path, since it is not an installed module."""
    assert SCRIPT_PATH.exists(), f"expected the data script at {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare_data = load_prepare_data()

FASTA = """\
>Q9H400 Cell.membrane-M test
MGLPVSWAPP
ALWVLGCCAL
>Q5I0E9 Cytoplasm-S
MEVLEEPAPG
>P63033 Cytoplasm-Nucleus-U
MMKTLSSGNC
"""


def test_parse_header_reads_accession_label_and_split() -> None:
    accession, localization, solubility, is_test = prepare_data.parse_header(
        "Q9H400 Cell.membrane-M test"
    )
    assert accession == "Q9H400"
    assert localization == "Cell.membrane"
    assert solubility == "M"
    assert is_test is True


def test_parse_header_defaults_to_train_when_no_test_marker() -> None:
    _, _, _, is_test = prepare_data.parse_header("Q5I0E9 Cytoplasm-S")
    assert is_test is False


def test_parse_header_keeps_compound_localization_intact() -> None:
    """Only the final hyphen separates the solubility code."""
    _, localization, solubility, _ = prepare_data.parse_header(
        "P63033 Cytoplasm-Nucleus-U"
    )
    assert localization == prepare_data.DUAL_LOCALIZATION
    assert solubility == "U"


@pytest.mark.parametrize(
    "header",
    [
        "Q9H400",  # Missing the label entirely.
        "Q9H400 Cell.membrane-M train",  # Unrecognized third token.
        "Q9H400 Cell.membrane-X",  # Invalid solubility code.
        "Q9H400 Nucleolus-S",  # Localization outside the DeepLoc classes.
        "Q9H400 Cell.membrane-M test extra",  # Too many tokens.
    ],
)
def test_parse_header_rejects_malformed_headers(header: str) -> None:
    """Unexpected input fails loudly rather than being skipped."""
    with pytest.raises(AssertionError):
        prepare_data.parse_header(header)


def test_parse_deeploc_fasta_joins_wrapped_sequence_lines() -> None:
    records = prepare_data.parse_deeploc_fasta(FASTA)
    assert len(records) == 3
    assert records[0].sequence == "MGLPVSWAPPALWVLGCCAL"


def test_parse_deeploc_fasta_preserves_order_and_labels() -> None:
    records = prepare_data.parse_deeploc_fasta(FASTA)
    assert [record.accession for record in records] == ["Q9H400", "Q5I0E9", "P63033"]
    assert [record.is_test for record in records] == [True, False, False]


def test_parse_deeploc_fasta_keeps_dual_localization_records() -> None:
    """Dropping them is the caller's decision, so the parser must not do it."""
    records = prepare_data.parse_deeploc_fasta(FASTA)
    localizations = [record.localization for record in records]
    assert prepare_data.DUAL_LOCALIZATION in localizations


def test_parse_deeploc_fasta_rejects_duplicate_accessions() -> None:
    with pytest.raises(AssertionError, match="duplicate accessions"):
        prepare_data.parse_deeploc_fasta(
            ">Q9H400 Cytoplasm-S\nMKV\n>Q9H400 Nucleus-U\nMKV\n"
        )


def test_parse_deeploc_fasta_rejects_empty_sequence() -> None:
    with pytest.raises(AssertionError, match="empty sequence"):
        prepare_data.parse_deeploc_fasta(">Q9H400 Cytoplasm-S\n\n")


def test_parse_deeploc_fasta_rejects_sequence_before_header() -> None:
    with pytest.raises(AssertionError, match="before any header"):
        prepare_data.parse_deeploc_fasta("MKV\n>Q9H400 Cytoplasm-S\nMKV\n")


def test_parse_deeploc_fasta_rejects_empty_input() -> None:
    with pytest.raises(AssertionError, match="parsed no records"):
        prepare_data.parse_deeploc_fasta("")


def test_single_localizations_are_the_ten_deeploc_classes() -> None:
    assert len(prepare_data.SINGLE_LOCALIZATIONS) == 10
    assert prepare_data.DUAL_LOCALIZATION not in prepare_data.SINGLE_LOCALIZATIONS


@pytest.mark.parametrize(
    ("accession", "expected"),
    [
        ("P22462-2", "P22462"),
        ("Q99N50-11", "Q99N50"),
        ("Q9H400", "Q9H400"),
    ],
)
def test_base_accession_strips_isoform_suffix(accession: str, expected: str) -> None:
    """Isoform accessions return empty annotations; the parent entry has the text."""
    assert prepare_data.base_accession(accession) == expected


def test_request_fields_are_api_ids_not_column_titles() -> None:
    """Sending the human-readable TSV titles as `fields` is a 400 from UniProt.

    The two vocabularies are easy to conflate because the response is keyed by the
    titles, so pin that the request uses lowercase api ids with no spaces.
    """
    for field in prepare_data.UNIPROT_REQUEST_FIELDS:
        assert field == field.lower()
        assert " " not in field
        assert field not in prepare_data.UNIPROT_COLUMN_RENAMES


def test_annotation_columns_exclude_the_join_key() -> None:
    assert "accession" not in prepare_data.ANNOTATION_COLUMNS
    assert "function_text" in prepare_data.ANNOTATION_COLUMNS


def _http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(f"{status_code} error", response=response)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (requests.Timeout("timed out"), True),
        (requests.ConnectionError("dropped"), True),
        (_http_error(500), True),
        (_http_error(503), True),
        (_http_error(400), False),
        (_http_error(404), False),
    ],
)
def test_is_transient_only_retries_recoverable_failures(
    error: requests.RequestException, expected: bool
) -> None:
    """A 400 is deterministic, so retrying it only delays and buries the cause."""
    assert prepare_data._is_transient(error) is expected


def test_describe_truncates_long_error_text() -> None:
    """A failed batch must not dump a multi-kilobyte URL into the log."""
    response = requests.Response()
    response.status_code = 400
    response._content = b"x" * 5000
    error = requests.HTTPError("y" * 5000, response=response)
    assert len(prepare_data._describe(error)) < 500
