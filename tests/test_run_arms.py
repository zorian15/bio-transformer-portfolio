"""Tests for the arm runner's structure and its localization vocabulary (issue #5).

Two separate things live here.

The **structural** tests pin the wiring that fails silently: a feature block no
arm knows how to shuffle, a seed-dependent block that resolves to the wrong seed,
a duplicated arm name that `summarize` would quietly merge. None of these raise on
their own; they produce a plausible-looking number for the wrong experiment.

The **vocabulary** tests are the apparatus's ground truth. LOCALIZATION_FIXTURE is
hand-labeled, and every sentence in it is quoted from the real DeepLoc/UniProt
corpus. Recall is asserted at exactly 1.0 because a missed synonym leaves the
answer in the text and biases the ablation toward "grounding" invisibly, which is
the one failure this experiment cannot detect from its own results.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from biotp.text_ablation import ablate_sentences, compile_term_pattern

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "projects"
    / "grounding-multimodal"
    / "scripts"
    / "run_arms.py"
)


def load_run_arms() -> ModuleType:
    """Import the project script by path, since it is not an installed module."""
    assert SCRIPT_PATH.exists(), f"expected the runner at {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("run_arms", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_arms = load_run_arms()

DEEPLOC_CLASSES = {
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


def lexicon_pattern() -> re.Pattern[str]:
    return compile_term_pattern(
        tuple(term for group in run_arms.COMPARTMENT_TERMS.values() for term in group)
    )


def exclusion_pattern() -> re.Pattern[str]:
    return compile_term_pattern(run_arms.COMPARTMENT_EXCLUSIONS)


def is_removed(sentence: str) -> bool:
    """True when the real lexicon would drop this sentence from the free text."""
    result = ablate_sentences(sentence, lexicon_pattern(), exclusion_pattern())
    return bool(result.removed)


# --- The hand-labeled corpus fixture ------------------------------------------

# Sentences the filter MUST remove. All quoted from the corpus. Several are not
# location claims about the labeled protein at all ("mitochondrial stress" on a
# Cell.membrane protein): removing them is the prefer-recall doctrine working as
# designed, and the length-matched random control is what pays for it.
LOCALIZATION_POSITIVES = (
    "Water channel required to facilitate the transport of water across cell membrane.",
    "Plasma membrane ceramidase that hydrolyzes sphingolipid ceramides into sphingosine.",
    (
        "Binds to type I regulatory subunits of protein kinase A and anchors them to the "
        "plasma membrane."
    ),
    (
        "Mitochondrial protein that plays a critical role in the sodium-dependent calcium "
        "efflux from mitochondrion."
    ),
    (
        "Plays a role in the learned avoidance behavior of animals exposed to food that "
        "induces mitochondrial stress."
    ),
    "Required for peroxisomal protein import which maintains the function of peroxisomes.",
    "Affinity for prostaglandins, peroxisomal proliferators and clofibrate is moderate.",
    "May play a role in lysosomal ion flux and osmoregulation.",
    "Required for protein transport from Golgi to vacuoles.",
    "Involved in post-Golgi vesicle transport.",
    "May be involved in endoplasmic reticulum exit trafficking of proteins.",
    (
        "Through zinc cellular uptake also plays a role in the adaptation of cells to "
        "endoplasmic reticulum stress."
    ),
    (
        "When activated, the interaction between both proteins is affected and DDX3X "
        "relocalizes to the nucleus."
    ),
    "Receptor for ADIPOQ, an essential hormone secreted by adipocytes.",
    "Required for vacuolar catabolite degradation of fructose-1,6-bisphosphatase.",
    "Mediates chloroplast movement via chloroplast actin filaments.",
    (
        "Kinesin-like protein required for chloroplast movements and anchor to the plasma "
        "membrane."
    ),
    (
        "May also function intracellularly and mediate the transport from endosomes to "
        "cytosol of iron endocytosed by transferrin."
    ),
    "Upon extracellular pH drop this channel elicits transient inward currents.",
    (
        "Required for apical extracellular matrix organization and epithelial junction "
        "maintenance."
    ),
    "Extracts GDP-bound YPT7 from vacuolar membranes, antagonizing membrane fusion.",
    "Nuclear CGAS is inactivated by chromatin via direct interaction with nucleosomes.",
    # `perinuclear` is listed in its own right, so this is a removal, not a
    # near-miss on `nuclear`. It states a location and belongs here.
    "Concentrates in perinuclear puncta during the stress response.",
)

# Sentences the filter MUST keep. The first group are the verified traps: each
# contains a lexicon term, or a word that looks like one, without stating where
# the protein is. The second group are ordinary function claims.
LOCALIZATION_NEGATIVES = (
    # Traps, each encoding one design decision.
    "Reductase required for adipogenesis and activation of PPARG nuclear receptor.",
    "Stimulates nuclear receptor mediated transcription.",
    "Nuclear factor which might have a role in spermatogenesis.",
    (
        "All CD3 chains contain immunoreceptor tyrosine-based activation motifs in their "
        "cytoplasmic domain."
    ),
    (
        "Exports S-geranylgeranyl-glutathione in lymphoid cells and stromal compartments "
        "of lymphoid organs."
    ),
    "Required for cell wall integrity.",
    "Regulates cell wall composition and structure.",
    "Acts as a coreceptor with CD4 for HIV-1 virus envelope protein.",
    "Involved in zinc transport from the intestinal lumen to the pseudocoelum.",
    "Plays an essential role in the yeast secretory pathway.",
    "Functions in the late secretory pathway.",
    "Required for GLUTAMINE DUMPER 1-induced amino acid secretion and homeostasis.",
    (
        "May regulate the levels of polyamines on chromosomal DNA, which would modify "
        "chromatin structure and affect transcription."
    ),
    # Ordinary function prose, which the ablation must not touch.
    "Transcription factor involved in starch synthesis.",
    "Peptide transporter.",
    "Mediates the transport of di- and tripeptides.",
    "Inhibits trypsin, plasmin, factor VIIa/tissue factor and weakly factor Xa.",
    "Has no effect on thrombin.",
    "High affinity transporter with low selectivity.",
    "Catalyzes the reduction of 3'-oxosphinganine to sphinganine.",
    "E3 ubiquitin-protein ligase that monoubiquitinates H2B to form H2BK143ub1.",
    "Involved in the control of seed dormancy and germination.",
    "Major component of the cell cycle transcription factor complex MBF.",
)


def test_filter_removes_every_hand_labeled_positive() -> None:
    """Recall must be 1.0: a survivor here is a leak the experiment cannot see."""
    survivors = [s for s in LOCALIZATION_POSITIVES if not is_removed(s)]
    assert not survivors, (
        f"{len(survivors)} localization-stating sentences survived the filter, so "
        f"the ablated arm can still read the label: {survivors}"
    )


@pytest.mark.parametrize(
    "sentence", LOCALIZATION_NEGATIVES, ids=range(len(LOCALIZATION_NEGATIVES))
)
def test_filter_keeps_every_hand_labeled_negative(sentence: str) -> None:
    """Each is asserted separately, because each encodes its own design decision."""
    assert not is_removed(sentence), (
        f"the filter removed functional prose that states no location: {sentence!r}"
    )


def test_perinuclear_is_a_term_not_an_accident_of_matching() -> None:
    """`nuclear` must not fire inside `perinuclear`; the lexicon lists it separately."""
    assert "perinuclear" in run_arms.COMPARTMENT_TERMS["Nucleus"]


# --- The vocabulary's shape ---------------------------------------------------


def test_lexicon_covers_every_deeploc_class() -> None:
    """A silently unrepresented compartment would leak only for that class."""
    compartments = set(run_arms.COMPARTMENT_TERMS) - {"localization_language"}
    assert compartments == DEEPLOC_CLASSES


def test_sentinel_terms_are_disjoint_from_the_lexicon() -> None:
    """A term in both would make the false-negative probe report zero by design."""
    lexicon = {
        term.lower() for group in run_arms.COMPARTMENT_TERMS.values() for term in group
    }
    overlap = lexicon & {term.lower() for term in run_arms.SENTINEL_TERMS}
    assert not overlap, f"filtered terms cannot also be sentinels: {sorted(overlap)}"


def test_chromatin_is_measured_rather_than_filtered() -> None:
    """A deliberate false negative: `chromatin remodeling` is the prose under study.

    Recorded as a test so the decision has to be re-made, rather than drifting in
    with a later lexicon edit. See docs/grounding-multimodal/ablation.md.
    """
    lexicon = {
        term.lower() for group in run_arms.COMPARTMENT_TERMS.values() for term in group
    }
    assert "chromatin" not in lexicon
    assert "chromatin" in run_arms.SENTINEL_TERMS


# --- Arm and block wiring -----------------------------------------------------


def test_every_arm_block_is_sequence_or_a_declared_text_block() -> None:
    referenced = {name for arm in run_arms.ARMS for name in arm.blocks}
    assert referenced <= run_arms.TEXT_BLOCKS | {"sequence"}


def test_arm_names_are_unique() -> None:
    """`summarize` groups by name, so a duplicate merges two arms' seeds silently."""
    names = [arm.name for arm in run_arms.ARMS]
    assert len(set(names)) == len(names)


def test_the_ablation_arms_are_present_alongside_the_unfiltered_ones() -> None:
    """The issue's requirement: the comparison stays visible rather than replacing."""
    names = {arm.name for arm in run_arms.ARMS}
    assert {"sequence+free-text", "text-only-free"} <= names
    assert {
        "sequence+free-text-cleaned",
        "sequence+free-text-ablated",
        "sequence+free-text-random-ablated",
    } <= names


@pytest.mark.parametrize("block_name", sorted(run_arms.TEXT_BLOCKS))
def test_the_shuffle_rule_covers_every_text_block(block_name: str) -> None:
    """A text block the shuffle rule misses turns the control into a grounded arm.

    The old rule was a `startswith("text")` prefix test, which happens to work for
    today's names and would fail silently for a block called `free_text_...`.
    """
    rows = 6
    identity = np.eye(rows, dtype=np.float32)
    blocks = {f"{block_name}_seed0": identity, block_name: identity}
    arm = run_arms.Arm("probe", (block_name,), True, "test")

    features = run_arms.assemble_features(arm, blocks, np.arange(rows), 0)
    assert not np.array_equal(features, identity), (
        f"{block_name} was not permuted despite shuffle_text=True"
    )


def test_seed_dependent_blocks_resolve_to_that_seeds_copy() -> None:
    """Each seed must get its own random draw, not one draw reused three times."""
    blocks = {
        "text_free_random_ablated_seed0": np.zeros((3, 2), dtype=np.float32),
        "text_free_random_ablated_seed1": np.ones((3, 2), dtype=np.float32),
    }
    arm = run_arms.Arm("probe", ("text_free_random_ablated",), False, "test")

    for seed, expected in [(0, 0.0), (1, 1.0)]:
        features = run_arms.assemble_features(arm, blocks, np.arange(3), seed)
        assert float(features[0, 0]) == expected


def test_assemble_features_fails_loudly_on_a_missing_block() -> None:
    """A typo'd block name must raise rather than train on the wrong features."""
    arm = run_arms.Arm("probe", ("text_free_cleaned",), False, "test")
    with pytest.raises(AssertionError, match="which was not built"):
        run_arms.assemble_features(arm, {"sequence": np.zeros((2, 2))}, np.arange(2), 0)


def test_seed_dependent_blocks_are_never_shuffled() -> None:
    """Permuting and redrawing on the same seed would confound two controls."""
    assert not any(
        arm.shuffle_text and set(arm.blocks) & run_arms.SEED_DEPENDENT_BLOCKS
        for arm in run_arms.ARMS
    )
