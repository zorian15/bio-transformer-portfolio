"""Tests for the sentence-ablation apparatus (issue #5).

These pin the mechanics: cleaning, splitting, the filter's contract, the
length-matched control, and the summary. The compartment vocabulary itself is
project-1 scientific judgement and is tested against a hand-labeled corpus
fixture in tests/test_run_arms.py, next to where it is defined.

Every string here is taken from the real DeepLoc/UniProt corpus unless it is
labeled as constructed, because the failure modes that matter are the ones the
actual data produces.
"""

from __future__ import annotations

import pytest

from biotp import text_ablation

# A real entry, quoted exactly as UniProt returns it (P93004). It carries the
# constant FUNCTION: prefix, a compartment mention, and a trailing evidence-code
# block after a sentence-final period, which is the orphan-punctuation case.
P93004 = (
    "FUNCTION: Water channel required to facilitate the transport of water across "
    "cell membrane. May be involved in the osmoregulation in plants under high "
    "osmotic stress such as under a high salt condition. "
    "{ECO:0000269|PubMed:10102577, ECO:0000269|PubMed:9276952}."
)


# --- Cleaning UniProt bookkeeping ---------------------------------------------


def test_clean_annotation_text_strips_evidence_codes() -> None:
    """Evidence codes are on 96% of descriptions and are pure provenance."""
    cleaned = text_ablation.clean_annotation_text(P93004)
    assert "ECO:" not in cleaned
    assert "PubMed" not in cleaned
    assert cleaned.startswith("Water channel required")


def test_clean_annotation_text_strips_inline_pubmed_citations() -> None:
    """Inline citations are interleaved mid-sentence on 4,042 proteins."""
    text = (
        "FUNCTION: Transcription factor involved in starch synthesis "
        "(PubMed:12953112). Acts as a transcriptional activator in sugar "
        "signaling (PubMed:16167901, PubMed:12953112)."
    )
    cleaned = text_ablation.clean_annotation_text(text)
    assert cleaned == (
        "Transcription factor involved in starch synthesis. Acts as a "
        "transcriptional activator in sugar signaling."
    )


def test_clean_annotation_text_strips_parenthetical_evidence_qualifiers() -> None:
    """`(By similarity)` says how well attested a claim is, not what it says."""
    text = "FUNCTION: May play a role in lysosomal ion flux (By similarity)."
    assert (
        text_ablation.clean_annotation_text(text)
        == "May play a role in lysosomal ion flux."
    )


def test_clean_annotation_text_keeps_content_qualifiers() -> None:
    """`(Microbial infection)` and isoform tags qualify the claim, not its evidence."""
    text = (
        "FUNCTION: [Isoform 1]: (Microbial infection) Acts as a coreceptor with "
        "CD4 for HIV-1 virus envelope protein."
    )
    cleaned = text_ablation.clean_annotation_text(text)
    assert "(Microbial infection)" in cleaned
    assert "[Isoform 1]" in cleaned


def test_clean_annotation_text_strips_every_function_prefix_not_only_the_first() -> (
    None
):
    """406 entries carry several function comments joined by `.; FUNCTION: `."""
    text = "FUNCTION: Peptide transporter.; FUNCTION: Mediates peptide transport."
    cleaned = text_ablation.clean_annotation_text(text)
    assert "FUNCTION" not in cleaned
    assert cleaned == "Peptide transporter. Mediates peptide transport."


def test_clean_annotation_text_leaves_no_orphan_punctuation() -> None:
    """The bug that made a first pass undercount emptied proteins tenfold.

    Evidence codes usually follow a sentence-final period, so removing one leaves
    a stray `.` that survives as its own sentence. A fully ablated protein then
    still looks non-empty, which is exactly the statistic the ablation turns on.
    """
    cleaned = text_ablation.clean_annotation_text(P93004)
    assert ".." not in cleaned
    assert " ." not in cleaned
    assert not cleaned.endswith(" .")
    assert cleaned.endswith("high salt condition.")


def test_clean_annotation_text_of_pure_bookkeeping_is_empty() -> None:
    """Constructed: text that is nothing but provenance has no content to keep."""
    text = "FUNCTION: {ECO:0000269|PubMed:10102577}."
    assert text_ablation.clean_annotation_text(text) == ""


# --- Sentence splitting -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Binds several ions, e.g. calcium and zinc, at neutral pH.",
        "Acts on the terminal residue, i.e. the C-terminus, of the substrate.",
        "Found in Bacillus sp. and related organisms.",
        "Found in Candida spp. under anaerobic growth.",
        "Cleaves the substrate in S. cerevisiae during sporulation.",
        "Reduces activity approx. two-fold under oxidative stress.",
    ],
)
def test_split_sentences_does_not_break_on_abbreviations(text: str) -> None:
    """A period inside an abbreviation is not a sentence boundary."""
    assert text_ablation.split_sentences(text) == [text]


def test_split_sentences_does_not_break_on_decimal_numbers() -> None:
    """293 descriptions contain a decimal; splitting one fragments the claim."""
    text = "Reduces activity 2.5-fold at pH 7.4 under oxidative stress."
    assert text_ablation.split_sentences(text) == [text]


def test_split_sentences_does_not_break_inside_brackets() -> None:
    """Constructed: a semicolon inside brackets separates list items, not claims."""
    text = "Catalyzes two reactions (EC 1.1.1.1; EC 2.2.2.2) in one active site."
    assert text_ablation.split_sentences(text) == [text]


def test_split_sentences_separates_on_periods_and_semicolons() -> None:
    """Semicolons join independent clauses; a finer unit removes less collateral."""
    text = "Peptide transporter. Mediates transport; has no effect on thrombin."
    assert text_ablation.split_sentences(text) == [
        "Peptide transporter.",
        "Mediates transport;",
        "has no effect on thrombin.",
    ]


def test_split_sentences_drops_no_characters() -> None:
    """Guards the class of splitter bug that silently eats text."""
    cleaned = text_ablation.clean_annotation_text(P93004)
    rejoined = " ".join(text_ablation.split_sentences(cleaned))
    assert rejoined == cleaned


def test_split_sentences_handles_text_without_a_terminal_period() -> None:
    assert text_ablation.split_sentences("Peptide transporter") == [
        "Peptide transporter"
    ]


def test_split_sentences_drops_punctuation_only_fragments() -> None:
    """Constructed: a caller counting sentences should be counting claims."""
    assert text_ablation.split_sentences("Peptide transporter. . ;") == [
        "Peptide transporter."
    ]


def test_split_sentences_of_empty_text_is_empty() -> None:
    assert text_ablation.split_sentences("") == []


# --- Term matching ------------------------------------------------------------


def test_compile_term_pattern_matches_case_insensitively() -> None:
    pattern = text_ablation.compile_term_pattern(("golgi",))
    assert pattern.search("Transported to the Golgi.")
    assert pattern.search("post-golgi vesicle transport")


def test_compile_term_pattern_respects_word_boundaries() -> None:
    """`nuclear` must not fire inside `perinuclear`, or the counts are fiction."""
    pattern = text_ablation.compile_term_pattern(("nuclear",))
    assert pattern.search("Localizes to the nuclear pore.")
    assert not pattern.search("Concentrates in perinuclear puncta.")


def test_compile_term_pattern_matches_across_hyphens() -> None:
    """A hyphen is not a word character, so `golgi` fires inside `trans-Golgi`."""
    pattern = text_ablation.compile_term_pattern(("golgi",))
    assert pattern.search("Resident of the trans-Golgi network.")


def test_compile_term_pattern_prefers_the_longest_match() -> None:
    """Otherwise a short term shadows a long one and misattributes removals."""
    pattern = text_ablation.compile_term_pattern(("golgi", "trans-golgi network"))
    match = pattern.search("The trans-Golgi network.")
    assert match is not None
    assert match.group(0).lower() == "trans-golgi network"


def test_compile_term_pattern_matches_regular_plurals() -> None:
    """A vocabulary enumerating plurals would eventually miss one, invisibly."""
    pattern = text_ablation.compile_term_pattern(("peroxisome",))
    assert pattern.search("Required for import into peroxisomes.")


def test_compile_term_pattern_reports_the_listed_term_not_the_inflection() -> None:
    """Otherwise per-term counts split across singular and plural."""
    pattern = text_ablation.compile_term_pattern(("peroxisome",))
    assert pattern.findall("Targets peroxisomes and the peroxisome.") == [
        "peroxisome",
        "peroxisome",
    ]


def test_compile_term_pattern_rejects_an_empty_vocabulary() -> None:
    with pytest.raises(AssertionError, match="received no terms"):
        text_ablation.compile_term_pattern(())


# --- Ablation -----------------------------------------------------------------


def test_ablate_sentences_removes_only_the_matching_sentence() -> None:
    pattern = text_ablation.compile_term_pattern(("cell membrane",))
    result = text_ablation.ablate_sentences(P93004, pattern, None)

    assert len(result.removed) == 1
    assert result.removed[0].startswith("Water channel required")
    assert len(result.kept) == 1
    assert result.kept[0].startswith("May be involved in the osmoregulation")
    assert result.matched_terms == ("cell membrane",)


def test_ablate_sentences_counts_characters_over_sentences_not_raw_input() -> None:
    """Before and after must be commensurate, or retention reports a phantom loss."""
    pattern = text_ablation.compile_term_pattern(("cell membrane",))
    result = text_ablation.ablate_sentences(P93004, pattern, None)

    sentences = text_ablation.split_sentences(result.cleaned)
    assert result.characters_before == sum(len(s) for s in sentences)
    assert result.characters_after == sum(len(s) for s in result.kept)
    assert 0 < result.characters_after < result.characters_before


def test_ablate_sentences_keeps_a_sentence_whose_only_match_is_excluded() -> None:
    """`nuclear receptor` names a protein family, not a compartment."""
    pattern = text_ablation.compile_term_pattern(("nuclear",))
    exclusions = text_ablation.compile_term_pattern(("nuclear receptor",))
    result = text_ablation.ablate_sentences(
        "Stimulates nuclear receptor mediated transcription.", pattern, exclusions
    )
    assert result.removed == ()
    assert len(result.kept) == 1


def test_exclusions_only_rescue_sentences_with_no_other_match() -> None:
    """Constructed: an excluded phrase must not launder a real location claim."""
    pattern = text_ablation.compile_term_pattern(("nuclear", "nucleus"))
    exclusions = text_ablation.compile_term_pattern(("nuclear receptor",))
    result = text_ablation.ablate_sentences(
        "The nuclear receptor is retained in the nucleus.", pattern, exclusions
    )
    assert result.kept == ()
    assert result.matched_terms == ("nucleus",)


def test_ablate_sentences_can_empty_a_single_sentence_protein() -> None:
    """22% of annotated texts are one sentence, so this is common, not an edge case."""
    pattern = text_ablation.compile_term_pattern(("mitochondrial",))
    result = text_ablation.ablate_sentences(
        "FUNCTION: Mitochondrial carrier protein.", pattern, None
    )
    assert result.kept == ()
    assert result.characters_after == 0


# --- The length-matched random control ----------------------------------------


def test_random_ablation_removes_exactly_the_requested_count() -> None:
    text = "One claim. Two claims. Three claims. Four claims."
    kept = text_ablation.random_ablate_sentences(text, 2, seed=0)
    assert len(text_ablation.split_sentences(kept)) == 2


def test_random_ablation_is_deterministic_in_the_seed() -> None:
    text = "One claim. Two claims. Three claims. Four claims."
    first = text_ablation.random_ablate_sentences(text, 2, seed=7)
    second = text_ablation.random_ablate_sentences(text, 2, seed=7)
    assert first == second


def test_random_ablation_does_not_depend_on_call_order() -> None:
    """The draw is seeded per protein, so a cohort subset cannot change it.

    If the control drew from one stream shared across proteins, running the
    annotated-only cohort would hand every protein a different draw than the full
    run, and the two cohorts would stop being comparable.
    """
    corpus = {
        "P1": "One claim. Two claims. Three claims.",
        "P2": "Alpha claim. Beta claim. Gamma claim.",
        "P3": "Red claim. Green claim. Blue claim.",
    }
    forward = {
        name: text_ablation.random_ablate_sentences(
            text, 1, text_ablation.sentence_seed(name, 0)
        )
        for name, text in corpus.items()
    }
    reverse = {
        name: text_ablation.random_ablate_sentences(
            text, 1, text_ablation.sentence_seed(name, 0)
        )
        for name, text in reversed(list(corpus.items()))
    }
    assert forward == reverse


def test_random_ablation_of_every_sentence_returns_empty_text() -> None:
    """The control must be able to empty a protein, or it is not length-matched."""
    assert text_ablation.random_ablate_sentences("Only claim.", 1, seed=0) == ""


def test_random_ablation_rejects_removing_more_than_there_are() -> None:
    """A caller and the ablation disagreeing on the split must fail loudly."""
    with pytest.raises(AssertionError, match="asked to remove"):
        text_ablation.random_ablate_sentences("One claim. Two claims.", 3, seed=0)


def test_sentence_seed_is_stable_across_processes() -> None:
    """Pinned literally, because the builtin `hash` is salted per process.

    `biotp.utils.set_seed` sets PYTHONHASHSEED, which does nothing to the running
    interpreter. A salted seed here would redraw the control and invalidate the
    text cache on every run, which reads from the artifacts as a changed experiment.
    """
    assert text_ablation.sentence_seed("P93004", 0) == 17774277182579145964


def test_sentence_seed_separates_run_seeds_and_proteins() -> None:
    assert text_ablation.sentence_seed("P93004", 0) != text_ablation.sentence_seed(
        "P93004", 1
    )
    assert text_ablation.sentence_seed("P93004", 0) != text_ablation.sentence_seed(
        "Q9H400", 0
    )


# --- Summary and the false-negative probe -------------------------------------


def make_result(
    kept: int, removed: int, terms: tuple[str, ...]
) -> text_ablation.AblationResult:
    """Build an AblationResult with controlled sentence and character counts."""
    return text_ablation.AblationResult(
        cleaned="",
        kept=tuple("x" * 10 for _ in range(kept)),
        removed=tuple("y" * 10 for _ in range(removed)),
        matched_terms=terms,
        characters_before=(kept + removed) * 10,
        characters_after=kept * 10,
    )


def test_ablation_summary_counts_proteins_emptied_by_the_filter() -> None:
    """The number the ablation turns on: an emptied protein becomes a zero vector."""
    results = [make_result(2, 1, ("nucleus",)), make_result(0, 1, ("mitochondrial",))]
    summary = text_ablation.ablation_summary(results, ["Nucleus", "Mitochondrion"])

    assert summary["proteins_emptied"]["count"] == 1
    assert summary["proteins_emptied"]["share"] == 0.5
    assert summary["proteins_trimmed"]["count"] == 2


def test_ablation_summary_reports_per_class_shares() -> None:
    """A filter that trims 13% overall can still gut one class."""
    results = [make_result(2, 0, ()), make_result(0, 1, ("mitochondrial",))]
    summary = text_ablation.ablation_summary(results, ["Nucleus", "Mitochondrion"])

    assert summary["per_class"]["Nucleus"]["trimmed_share"] == 0.0
    assert summary["per_class"]["Mitochondrion"]["emptied_share"] == 1.0


def test_ablation_summary_reports_character_retention() -> None:
    results = [make_result(3, 1, ("nucleus",))]
    summary = text_ablation.ablation_summary(results, ["Nucleus"])

    assert summary["characters"]["corpus_retention"] == pytest.approx(0.75)
    assert summary["sentences"] == {"before": 4, "removed": 1, "share": 0.25}


def test_ablation_summary_rejects_mismatched_groups() -> None:
    """A misalignment would attribute one protein's removal to another's class."""
    with pytest.raises(AssertionError, match="results and"):
        text_ablation.ablation_summary([make_result(1, 0, ())], ["Nucleus", "Plastid"])


def test_term_mention_counts_reports_zero_for_an_absent_term() -> None:
    """A missing key reads as 'no leakage' at a glance, so absent terms report 0."""
    counts = text_ablation.term_mention_counts(
        ["Binds the chromatin remodeling complex."], ("chromatin", "peroxisome")
    )
    assert counts == {"chromatin": 1, "peroxisome": 0}


def test_term_mention_counts_counts_texts_not_occurrences() -> None:
    """The probe answers 'how many proteins still say this', not 'how often'."""
    counts = text_ablation.term_mention_counts(
        ["Chromatin binds chromatin.", "No mention here."], ("chromatin",)
    )
    assert counts == {"chromatin": 1}
