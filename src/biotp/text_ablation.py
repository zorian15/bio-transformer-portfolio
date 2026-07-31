"""Ablate label-stating sentences from curated annotation prose.

A text arm that beats a sequence-only baseline has two possible explanations:
the text carries genuine information the sequence lacks (grounding), or the text
states the answer and the model is reading it (leakage). Telling them apart needs
an ablation: remove the sentences that state the label, re-run, and see how much
of the gain survives. See issue #5 and docs/grounding-multimodal/ablation.md.

This module is the apparatus, not the experiment. It knows how to strip UniProt
bookkeeping, split annotation prose into sentences, drop the sentences matching a
caller-supplied vocabulary, and report exactly how much it removed. The vocabulary
itself is the caller's, because which words state the label depends on the task:
subcellular compartments for grounding-multimodal, epitope names elsewhere.

Two asymmetries govern the design and are worth stating once.

A **false negative**, a synonym the vocabulary misses, leaves the answer in the
text and biases the result toward "grounding" invisibly. A **false positive**
removes real functional content, which looks like leakage but is visible: pair the
ablation with a length-matched random-sentence control (`random_ablate_sentences`)
and an over-aggressive vocabulary shows up as both arms dropping together. So
prefer recall over precision in the vocabulary, and always run the control.

Removing sentences can leave a protein with **no text at all**, and callers must
handle that deliberately rather than discover it. `biotp.embeddings.embed_texts`
maps empty text to a zero vector, so an emptied protein silently becomes a
"no annotation" row. If the emptied population is class-skewed, and here it is,
that hands the ablated arm a handicap unrelated to leakage. `ablation_summary`
therefore reports the emptied count as a first-class number.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# UniProt appends provenance markers to the prose and interleaves inline
# citations. Both are database bookkeeping rather than biology, and they average
# a fifth of each field's characters, so they displace real text under a sentence
# encoder's token limit. See docs/grounding-multimodal/data.md.
EVIDENCE_CODES = re.compile(r"\{ECO:[^}]*\}")
# Any parenthetical holding a PubMed or ECO reference is a citation, wherever the
# reference sits inside it. Anchoring on the opening paren instead would miss
# `(see PubMed:123)`, and would leave `(, PubMed:456)` behind once an ECO block
# earlier in the same parenthetical had already been stripped.
INLINE_CITATIONS = re.compile(r"\([^()]*(?:PubMed:|ECO:|Ref\.)[^()]*\)")
FUNCTION_PREFIX = re.compile(r"FUNCTION:\s*")

# Parenthetical evidence qualifiers, which say how confident the curator is rather
# than anything about the protein. They are the same class of bookkeeping as the
# ECO codes and are stripped with them. `(Microbial infection)` and `[Isoform N]:`
# are deliberately left alone: those qualify what the sentence is about, not how
# well it is attested.
EVIDENCE_QUALIFIERS = re.compile(r"\((?:By similarity|Probable|Potential)\)")

# Periods that do not end a sentence. Each entry was counted in the corpus before
# being included; see the DECISION_LOG entry for issue #5.
ABBREVIATIONS = ("e.g.", "i.e.", "spp.", "sp.", "cf.", "approx.", "vs.", "etc.")

# Private codepoints stand in for a protected period or semicolon while the
# splitter runs, so the split pattern can stay a simple lookbehind. They are
# restored before anything is returned, and text containing one already would be
# corrupted, so their absence is asserted rather than assumed. The two characters
# need separate stand-ins: sharing one would restore a protected semicolon as a
# period and silently rewrite the text.
_PROTECTED = {".": "\ue000", ";": "\ue001"}


@dataclass(frozen=True)
class AblationResult:
    """One protein's before and after, with enough detail to audit the filter.

    `characters_before` and `characters_after` count sentence characters, not raw
    input characters, so the two are commensurate: splitting discards the
    whitespace between sentences, and comparing a raw length against a joined
    length would report a loss that never happened.
    """

    cleaned: str
    kept: tuple[str, ...]
    removed: tuple[str, ...]
    matched_terms: tuple[str, ...]
    characters_before: int
    characters_after: int


def clean_annotation_text(text: str) -> str:
    """Strip UniProt bookkeeping and the punctuation that stripping leaves behind.

    Removes `{ECO:...}` evidence codes, inline `(PubMed:...)` citations,
    parenthetical evidence qualifiers such as `(By similarity)`, and every
    `FUNCTION: ` prefix rather than only the first, since entries with several
    function comments arrive joined by `.; FUNCTION: `.

    The orphan-punctuation pass is load-bearing, not cosmetic. Evidence codes
    usually follow a sentence-final period, so removing one leaves `"... condition. ."`
    and the stray `"."` survives as a sentence of its own. That fragment inflates
    the sentence count and, worse, keeps a fully ablated protein looking non-empty,
    which is exactly the statistic the ablation turns on.

    A handful of entries (4 of 12,626 here) cite a reference in running prose,
    "According to PubMed:18817736, shows only specificity for ...". Those are left
    alone deliberately: the citation is the clause's subject, so deleting the token
    leaves broken grammar, and there is no reading of that sentence that drops it
    cleanly. Only bracketed citations are removed.
    """
    assert not any(
        mark in text for mark in _PROTECTED.values()
    ), "input contains a private protection codepoint"

    cleaned = EVIDENCE_CODES.sub("", text)
    cleaned = INLINE_CITATIONS.sub("", cleaned)
    cleaned = EVIDENCE_QUALIFIERS.sub("", cleaned)
    cleaned = FUNCTION_PREFIX.sub("", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([.;,:])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"([.;])[.;]+", r"\1", cleaned)
    cleaned = re.sub(r"^[.;,:\s]+", "", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    """Split annotation prose into sentences, keeping abbreviations intact.

    Sentences are the ablation's unit because they are the smallest span that
    reads as a claim. Semicolons split too, since UniProt uses them to separate
    independent clauses, and a finer unit removes less collateral content.

    Four period-like characters do not terminate a sentence and are protected:
    known abbreviations, a single capital followed by a lowercase word (the
    `S. cerevisiae` genus abbreviation), decimal points, and any punctuation
    inside brackets, where `(EC 1.1.1.1; EC 2.2.2.2)` would otherwise fragment.

    Punctuation-only fragments are dropped rather than returned, so a caller
    counting sentences counts claims.
    """
    assert not any(
        mark in text for mark in _PROTECTED.values()
    ), "input contains a private protection codepoint"

    protected = text
    for abbreviation in ABBREVIATIONS:
        # The lookbehind is load-bearing, and so is replacing via the matched text
        # rather than the literal. Without the boundary, `vs.` matches inside
        # `MAVS.` and `etc.` inside `petC.`, which both suppresses a real sentence
        # boundary and, because the replacement was a lowercase literal under
        # IGNORECASE, silently rewrote gene names (`MAVS.` became `MAvs.`).
        protected = re.sub(
            rf"(?<!\w){re.escape(abbreviation)}",
            lambda match: match.group(0).replace(".", _PROTECTED["."]),
            protected,
            flags=re.IGNORECASE,
        )
    protected = re.sub(
        r"\b([A-Z])\.(?=\s+[a-z])", lambda m: m.group(1) + _PROTECTED["."], protected
    )
    protected = re.sub(r"(?<=\d)\.(?=\d)", _PROTECTED["."], protected)
    protected = _protect_inside_brackets(protected)

    parts = re.split(r"(?<=[.;])\s+", protected)
    sentences = [_restore(part).strip() for part in parts]
    return [sentence for sentence in sentences if sentence.strip(" .;,:")]


def compile_term_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    """Compile one case-insensitive alternation over `terms`.

    Longer phrases come first so the reported match is the most specific one:
    without that, `golgi` would shadow `trans-golgi network` and the per-term
    counts would misattribute every removal.

    Boundaries are lookarounds rather than `\\b` so hyphenated compounds behave:
    `golgi` matches inside `trans-Golgi` (a hyphen is not a word character) while
    `nuclear` does not match inside `perinuclear`.

    A regular plural is matched without being listed, so `peroxisome` also fires on
    `peroxisomes`. This is a recall decision: a vocabulary that had to enumerate
    plurals would eventually miss one, and a missed term is the invisible failure.
    Irregular plurals (`nuclei`, `mitochondria`) still need their own entry. The
    match reported to callers is the listed term rather than the inflected form, so
    per-term counts do not split across singular and plural.

    This is a plain function rather than a cached one on purpose. A cache wrapper
    is not `inspect.isfunction`, so it would drop out of the public-API check in
    tests/test_conventions.py and stop being policed.
    """
    assert terms, "compile_term_pattern received no terms"
    assert all(term.strip() for term in terms), "a term is empty or whitespace"

    ordered = sorted(set(terms), key=lambda term: (-len(term), term))
    alternation = "|".join(re.escape(term) for term in ordered)
    return re.compile(rf"(?<!\w)({alternation})s?(?!\w)", re.IGNORECASE)


def ablate_sentences(
    text: str, pattern: re.Pattern[str], exclusions: re.Pattern[str] | None
) -> AblationResult:
    """Clean, split, and drop every sentence `pattern` matches.

    `exclusions` names phrases that contain a vocabulary term without stating a
    location, such as `nuclear receptor` or `cytoplasmic tail`. A sentence is kept
    only when an exclusion accounts for *all* of its matches, so
    "the nuclear receptor is retained in the nucleus" is still removed. Pass None
    for no exclusions.

    Every exclusion is a deliberate false negative, the dangerous direction, so
    keep that list short and test each entry by name.
    """
    cleaned = clean_annotation_text(text)
    sentences = split_sentences(cleaned)

    kept: list[str] = []
    removed: list[str] = []
    matched: list[str] = []
    for sentence in sentences:
        candidate = sentence if exclusions is None else exclusions.sub(" ", sentence)
        hits = pattern.findall(candidate)
        if hits:
            removed.append(sentence)
            matched.extend(hit.lower() for hit in hits)
        else:
            kept.append(sentence)

    return AblationResult(
        cleaned=cleaned,
        kept=tuple(kept),
        removed=tuple(removed),
        matched_terms=tuple(sorted(set(matched))),
        characters_before=sum(len(sentence) for sentence in sentences),
        characters_after=sum(len(sentence) for sentence in kept),
    )


def random_ablate_sentences(text: str, removals: int, seed: int) -> str:
    """Drop `removals` sentences chosen uniformly at random, seeded.

    The length-matched control for `ablate_sentences`. Removing localization
    sentences removes text as well as answers, so a drop in the ablated arm is
    uninterpretable on its own; this removes the same *number* of sentences from
    the same protein, which also means it empties the same proteins and hands out
    the same population of zero vectors.

    `seed` must already encode the protein's identity, so the draw does not depend
    on row order or cohort membership. Derive it with `sentence_seed`, never with
    the builtin `hash`, which is salted per process.
    """
    assert removals >= 0, f"removals must be >= 0, got {removals}"

    sentences = split_sentences(clean_annotation_text(text))
    assert removals <= len(sentences), (
        f"asked to remove {removals} of {len(sentences)} sentences; the caller "
        "and the ablation disagree about how this text splits"
    )

    dropped = set(random.Random(seed).sample(range(len(sentences)), removals))
    return " ".join(
        sentence for index, sentence in enumerate(sentences) if index not in dropped
    )


def sentence_seed(identifier: str, seed: int) -> int:
    """Derive a stable per-protein seed from an accession and a run seed.

    `biotp.utils.set_seed` sets PYTHONHASHSEED, which has no effect on the already
    running interpreter, so the builtin `hash` is not stable across processes.
    Using it here would redraw the random control on every run and invalidate the
    text cache each time, which reads from the artifacts as a changed experiment.
    """
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def ablation_summary(
    results: list[AblationResult], groups: list[str]
) -> dict[str, Any]:
    """Aggregate per-protein results into the numbers a writeup would cite.

    `groups` is one label per result, here the true compartment, because a filter
    that trims 14% of the corpus overall can still gut one class: the per-class
    breakdown is what makes the aggregate interpretable.

    The emptied count is reported separately from the removal count because the
    two mean different things. A protein that loses one of five sentences is
    trimmed; a protein that loses its only sentence becomes a zero vector.
    """
    assert results, "ablation_summary received no results"
    assert len(results) == len(groups), (
        f"got {len(results)} results and {len(groups)} groups; a mismatch would "
        "silently attribute one protein's removal to another's class"
    )

    # A protein whose whole field is bookkeeping (`FUNCTION: {ECO:...}.`) cleans to
    # nothing, and there is no honest retention to report for it: counting it as
    # 1.0 would say "kept everything" about text the cleaner deleted, and counting
    # it as 0.0 would blame the filter. It would also be tallied as emptied but not
    # as trimmed, breaking the subset relation the removal figure draws. No such
    # protein exists in this corpus, so this asserts rather than picking a default.
    # If one ever appears, the caller must decide, loudly, which bucket it belongs in.
    empty_before = [
        index for index, result in enumerate(results) if not result.characters_before
    ]
    assert not empty_before, (
        f"{len(empty_before)} results have no characters before filtering, so their "
        f"retention is undefined (first at index {empty_before[0]}); such a protein "
        "was emptied by the cleaner rather than by the ablation and needs its own "
        "bucket, not a default retention"
    )

    retention = np.array(
        [result.characters_after / result.characters_before for result in results]
    )
    sentences_before = sum(len(r.kept) + len(r.removed) for r in results)
    sentences_removed = sum(len(r.removed) for r in results)
    characters_before = sum(r.characters_before for r in results)

    per_class: dict[str, dict[str, Any]] = {}
    for result, group in zip(results, groups):
        stats = per_class.setdefault(group, {"proteins": 0, "trimmed": 0, "emptied": 0})
        stats["proteins"] += 1
        stats["trimmed"] += bool(result.removed)
        stats["emptied"] += not result.kept

    term_removals: dict[str, int] = {}
    for result in results:
        for term in result.matched_terms:
            term_removals[term] = term_removals.get(term, 0) + 1

    trimmed = sum(1 for result in results if result.removed)
    emptied = sum(1 for result in results if not result.kept)
    # An emptied protein is always a trimmed one. The removal figure draws the
    # emptied share inside the trimmed bar and would render nonsense otherwise.
    assert emptied <= trimmed, (
        f"{emptied} proteins emptied but only {trimmed} trimmed; emptied must be a "
        "subset of trimmed, and the removal figure draws it as one"
    )
    return {
        "proteins": len(results),
        "proteins_trimmed": {"count": trimmed, "share": trimmed / len(results)},
        "proteins_emptied": {"count": emptied, "share": emptied / len(results)},
        "sentences": {
            "before": sentences_before,
            "removed": sentences_removed,
            "share": sentences_removed / sentences_before if sentences_before else 0.0,
        },
        "characters": {
            "before": characters_before,
            "after": sum(r.characters_after for r in results),
            "corpus_retention": (
                sum(r.characters_after for r in results) / characters_before
                if characters_before
                else 1.0
            ),
            "retention_mean": float(retention.mean()),
            "retention_median": float(np.median(retention)),
            "retention_p10": float(np.percentile(retention, 10)),
        },
        "per_class": {
            name: {
                **stats,
                "trimmed_share": stats["trimmed"] / stats["proteins"],
                "emptied_share": stats["emptied"] / stats["proteins"],
            }
            for name, stats in sorted(per_class.items())
        },
        "removals_by_term": dict(sorted(term_removals.items())),
    }


def term_mention_counts(texts: list[str], terms: tuple[str, ...]) -> dict[str, int]:
    """Count how many of `texts` mention each term, including the terms that miss.

    The false-negative probe. Run it over the *surviving* text with a vocabulary
    deliberately left out of the filter, and the result says which location words
    the ablation still leaves behind. Terms with no hits are reported as zero
    rather than omitted, since a missing key reads as "no leakage" at a glance.
    """
    assert terms, "term_mention_counts received no terms"

    counts = {term: 0 for term in terms}
    for term in terms:
        pattern = compile_term_pattern((term,))
        for text in texts:
            if pattern.search(text):
                counts[term] += 1
    return counts


def _protect_inside_brackets(text: str) -> str:
    """Mask sentence-terminating punctuation inside brackets so it cannot split."""
    depth = 0
    out: list[str] = []
    for character in text:
        if character in "([":
            depth += 1
        elif character in ")]":
            depth = max(0, depth - 1)
        protect = depth and character in _PROTECTED
        out.append(_PROTECTED[character] if protect else character)
    return "".join(out)


def _restore(text: str) -> str:
    """Put every protected period and semicolon back as itself."""
    for character, mark in _PROTECTED.items():
        text = text.replace(mark, character)
    return text
