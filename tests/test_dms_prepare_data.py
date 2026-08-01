"""Tests for the dms-benchmark ProteinGym preparation script.

Loaded by path, like the grounding-multimodal parser tests, because data
preparation is project-specific by convention and lives outside the installed
package.

Everything here runs offline against small hand-built frames. The two behaviours
worth real tests are the ones that would otherwise fail silently: the
pre-registered filter and selection rule must be reproducible and independent of
input order, and the numbering guard must reject an assay paired with the wrong
reference sequence.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "dms-benchmark"
    / "scripts"
    / "prepare_data.py"
)


def load_prepare_data() -> ModuleType:
    """Import the project script by path, since it is not an installed module."""
    spec = importlib.util.spec_from_file_location("dms_prepare_data", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare_data = load_prepare_data()

# A 12-residue stand-in. Short enough to write substitutions against by hand.
TARGET = "MKTFFVLLLACD"


def reference_row(
    dms_id: str,
    taxon: str,
    seq_len: int = len(TARGET),
    singles: int = 3000,
    multiples: bool = False,
) -> dict:
    return {
        "DMS_id": dms_id,
        "taxon": taxon,
        "target_seq": TARGET,
        "seq_len": seq_len,
        "includes_multiple_mutants": multiples,
        "DMS_number_single_mutants": singles,
    }


def reference_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def assay_frame(mutants: list[str], target: str = TARGET) -> pd.DataFrame:
    """Build an assay table whose mutated_sequence agrees with each mutant."""
    sequences = []
    for mutant in mutants:
        position = int(mutant[1:-1]) - 1
        sequences.append(target[:position] + mutant[-1] + target[position + 1 :])
    return pd.DataFrame(
        {
            "dms_id": "TEST_ASSAY",
            "mutant": mutants,
            "mutated_sequence": sequences,
            "DMS_score": [0.0] * len(mutants),
            "fold_random_5": [0] * len(mutants),
            "fold_modulo_5": [0] * len(mutants),
            "fold_contiguous_5": [0] * len(mutants),
        }
    )


# --- The pre-registered filter ------------------------------------------------


def test_filter_drops_assays_with_multiple_mutants() -> None:
    frame = reference_frame(
        [
            reference_row("KEEP_VIRUS", "Virus"),
            reference_row("DROP_VIRUS", "Virus", multiples=True),
        ]
    )
    pool = prepare_data.filtered_pool(frame)
    assert pool["DMS_id"].tolist() == ["KEEP_VIRUS"]


def test_filter_drops_long_targets() -> None:
    frame = reference_frame(
        [
            reference_row("KEEP", "Virus", seq_len=prepare_data.MAX_TARGET_LENGTH),
            reference_row("DROP", "Virus", seq_len=prepare_data.MAX_TARGET_LENGTH + 1),
        ]
    )
    assert prepare_data.filtered_pool(frame)["DMS_id"].tolist() == ["KEEP"]


@pytest.mark.parametrize(
    "singles,kept",
    [
        (prepare_data.MIN_SINGLE_MUTANTS - 1, False),
        (prepare_data.MIN_SINGLE_MUTANTS, True),
        (prepare_data.MAX_SINGLE_MUTANTS, True),
        (prepare_data.MAX_SINGLE_MUTANTS + 1, False),
    ],
)
def test_filter_bounds_are_inclusive(singles: int, kept: bool) -> None:
    frame = reference_frame([reference_row("A", "Virus", singles=singles)])
    assert (len(prepare_data.filtered_pool(frame)) == 1) is kept


def test_filter_rejects_a_non_boolean_multiple_mutants_column() -> None:
    """Strings would make `~` mean the opposite of how it reads."""
    frame = reference_frame([reference_row("A", "Virus")])
    frame["includes_multiple_mutants"] = ["False"]
    with pytest.raises(AssertionError):
        prepare_data.filtered_pool(frame)


# --- Selection ----------------------------------------------------------------


def test_selection_takes_the_alphabetically_first_assay_per_taxon() -> None:
    frame = reference_frame(
        [
            reference_row("ZZZ_VIRUS", "Virus"),
            reference_row("AAA_VIRUS", "Virus"),
            reference_row("MMM_PROK", "Prokaryote"),
            reference_row("BBB_PROK", "Prokaryote"),
            reference_row("QQQ_HUMAN", "Human"),
        ]
    )
    selected = prepare_data.select_assays(prepare_data.filtered_pool(frame))
    assert selected["DMS_id"].tolist() == ["AAA_VIRUS", "BBB_PROK", "QQQ_HUMAN"]


def test_selection_does_not_depend_on_reference_row_order() -> None:
    """The rule must be reproducible, not incidentally stable.

    ProteinGym's reference file is not sorted, so a selection that inherited its
    order would change the benchmark's assays if upstream ever reordered a row,
    with nothing in the diff to show it.
    """
    rows = [
        reference_row("AAA_VIRUS", "Virus"),
        reference_row("ZZZ_VIRUS", "Virus"),
        reference_row("BBB_PROK", "Prokaryote"),
        reference_row("QQQ_HUMAN", "Human"),
    ]
    forward = prepare_data.select_assays(
        prepare_data.filtered_pool(reference_frame(rows))
    )
    reversed_ = prepare_data.select_assays(
        prepare_data.filtered_pool(reference_frame(list(reversed(rows))))
    )
    assert forward["DMS_id"].tolist() == reversed_["DMS_id"].tolist()


def test_selection_fails_loudly_when_a_taxon_is_empty() -> None:
    """A quietly shorter benchmark is worse than a stopped one."""
    frame = reference_frame(
        [reference_row("AAA_VIRUS", "Virus"), reference_row("BBB_PROK", "Prokaryote")]
    )
    with pytest.raises(AssertionError, match="Human"):
        prepare_data.select_assays(prepare_data.filtered_pool(frame))


# --- The numbering guard ------------------------------------------------------


def test_parse_variants_produces_zero_based_positions() -> None:
    parsed = prepare_data.parse_variants(assay_frame(["M1A", "C11G"]), TARGET, "TEST")
    assert parsed["position"].tolist() == [0, 10]
    assert parsed["wildtype_aa"].tolist() == ["M", "C"]
    assert parsed["mutant_aa"].tolist() == ["A", "G"]


def test_parse_variants_rejects_a_shifted_reference() -> None:
    """The failure this whole function exists for.

    A construct lacking its initiator methionine shifts every position by one.
    Scores computed against it stay finite and plausible and describe the wrong
    residue throughout, so the mismatch has to be an error at prep time.
    """
    shifted = TARGET[1:] + "A"
    with pytest.raises(AssertionError, match="disagree"):
        prepare_data.parse_variants(assay_frame(["M1A"]), shifted, "TEST")


def test_parse_variants_rejects_a_position_past_the_reference() -> None:
    """A truncated reference must fail on the index before anything else.

    The range check has to come first: a position outside the sequence would
    otherwise reach the residue comparison and raise `IndexError` from the string
    subscript, which says far less about what actually went wrong.
    """
    with pytest.raises(AssertionError, match="indexes position"):
        prepare_data.parse_variants(assay_frame(["C11G"]), "MKT", "TEST")


def test_parse_variants_rejects_a_mutated_sequence_that_disagrees() -> None:
    """The mutant string and the sequence beside it must describe one variant."""
    assay = assay_frame(["M1A"])
    assay.loc[0, "mutated_sequence"] = TARGET
    with pytest.raises(AssertionError, match="does not carry"):
        prepare_data.parse_variants(assay, TARGET, "TEST")


def test_parse_variants_rejects_a_multiple_substitution() -> None:
    """This cohort is filtered to singles, so a colon here means the filter slipped."""
    assay = assay_frame(["M1A"])
    assay.loc[0, "mutant"] = "M1A:K2R"
    with pytest.raises(AssertionError, match="substitutions"):
        prepare_data.parse_variants(assay, TARGET, "TEST")


# --- Fold columns -------------------------------------------------------------


def test_load_assay_requires_the_fold_columns(tmp_path: Path) -> None:
    """The main ProteinGym archive lacks these; only the folds archive has them.

    Deriving substitutes would still run and would quietly stop being comparable
    to published supervised baselines, so a missing column is an error.
    """
    assay = assay_frame(["M1A"]).drop(columns=["fold_modulo_5"])
    path = tmp_path / "TEST_ASSAY.csv"
    assay.to_csv(path, index=False)

    with pytest.raises(AssertionError, match="fold_modulo_5"):
        prepare_data.load_assay(path, "TEST_ASSAY")


def test_load_assay_tags_rows_with_the_assay_id(tmp_path: Path) -> None:
    path = tmp_path / "TEST_ASSAY.csv"
    assay_frame(["M1A", "K2R"]).drop(columns=["dms_id"]).to_csv(path, index=False)

    loaded = prepare_data.load_assay(path, "TEST_ASSAY")
    assert loaded["dms_id"].unique().tolist() == ["TEST_ASSAY"]


def test_the_three_pre_registered_schemes_are_required() -> None:
    """All three are reported; none may be dropped without amending issue #11."""
    assert set(prepare_data.FOLD_COLUMNS) == {
        "fold_random_5",
        "fold_modulo_5",
        "fold_contiguous_5",
    }
