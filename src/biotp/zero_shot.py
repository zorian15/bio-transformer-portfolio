"""Zero-shot variant scoring from a protein LM's masked-token likelihood.

This is the "no training" arm of the DMS ladder in issue #11: the pretrained
model is used exactly as it comes, with no gradient updates and no labels. For a
protein LM that is what prompting means, so the gap between this and a fine-tune
is the quantity the ladder exists to measure.

The scoring rule is the masked-marginal one, which is ProteinGym's standard for
ESM. Mask one wild-type position at a time, read the model's distribution over
what belongs there, and score a variant by how much likelihood it moves from the
wild-type residue to the mutant one:

    s(v) = sum over mutated positions i of  log p(mut_i | x_masked_i)
                                          - log p(wt_i  | x_masked_i)

Cost is one forward pass per *distinct mutated position*, not per variant, since
every variant touching a position reads the same distribution. A deep mutational
scan of a 300-residue protein is therefore about 300 forwards however many
thousand variants it measured, which is what makes this affordable at 650M.

The cheaper alternative, wild-type marginals, takes a single unmasked forward
pass and reads the same distribution off the wild-type logits. It is weaker, and
it is not implemented here: this module has one scoring rule so that a number
carrying its name cannot have come from the other one.

Nothing here touches the embedding cache. The output is a scalar per variant, not
a representation, so there is nothing to key and nothing to invalidate.
"""

from __future__ import annotations

import time

import numpy as np

from biotp.embeddings import PROGRESS_EVERY_N_BATCHES, Esm2Bundle
from biotp.runlog import get_logger

# A substitution as (zero-based position, wild-type residue, mutant residue). The
# wild-type residue is carried rather than looked up so it can be checked against
# the sequence, which is the only guard against an off-by-one between an assay's
# numbering and the reference sequence it is paired with.
Substitution = tuple[int, str, str]

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def parse_substitutions(mutant: str, one_based: bool) -> list[Substitution]:
    """Parse a mutant string such as "A24G" or "A24G:L56P".

    Args:
        mutant: colon-separated substitutions, each a wild-type residue, a
            position, and a mutant residue.
        one_based: whether the positions count from one. ProteinGym's do. There
            is no default, because an off-by-one here shifts every score by one
            residue and leaves them all finite and plausible.

    Returns:
        One (zero-based position, wild-type residue, mutant residue) triple per
        substitution, in the order written.
    """
    assert mutant, "parse_substitutions received an empty mutant string"

    parsed: list[Substitution] = []
    for field in mutant.split(":"):
        assert len(field) >= 3, f"malformed substitution {field!r} in {mutant!r}"
        wildtype_aa, digits, mutant_aa = field[0], field[1:-1], field[-1]
        assert wildtype_aa in AMINO_ACIDS, f"bad wild-type residue in {field!r}"
        assert mutant_aa in AMINO_ACIDS, f"bad mutant residue in {field!r}"
        assert digits.isdigit(), f"non-numeric position in {field!r}"

        position = int(digits)
        if one_based:
            assert position >= 1, f"one-based position must be >= 1 in {field!r}"
            position -= 1
        parsed.append((position, wildtype_aa, mutant_aa))

    return parsed


def _distinct_positions(variants: list[list[Substitution]]) -> list[int]:
    """Every position any variant mutates, sorted, each appearing once."""
    positions = {position for variant in variants for position, _, _ in variant}
    return sorted(positions)


def masked_marginal_scores(
    model: Esm2Bundle,
    wildtype: str,
    variants: list[list[Substitution]],
    batch_size: int,
) -> np.ndarray:
    """Score each variant by the masked-marginal log-likelihood ratio.

    Args:
        model: a bundle from load_esm2. Its alphabet supplies the mask token.
        wildtype: the reference sequence the positions index.
        variants: one list of substitutions per variant. A single substitution is
            a one-element list; multi-mutants carry several.
        batch_size: masked copies of the sequence per forward pass.

    Returns:
        Array of shape (len(variants),). Higher means the model prefers the
        mutant over the wild type at those positions. The scale is log-odds and
        is not comparable across proteins, which is why the benchmark reports
        Spearman within an assay rather than a raw value.

    Every mutated position is masked exactly once and its distribution reused by
    each variant touching it, so cost scales with the protein rather than with
    the number of variants.

    A substitution whose stated wild-type residue disagrees with `wildtype` is an
    error. That mismatch is the signature of an assay numbering paired with the
    wrong reference sequence, and every score it produces is finite, plausible,
    and describes the residue next door.
    """
    import torch

    assert variants, "masked_marginal_scores received no variants"
    assert wildtype, "masked_marginal_scores received an empty wild-type sequence"
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
    for index, variant in enumerate(variants):
        assert (
            variant
        ), f"variant {index} has no substitutions; the wild type is not a variant"

    truncated = wildtype[: model.max_sequence_length]
    for position, wildtype_aa, _ in (sub for variant in variants for sub in variant):
        assert 0 <= position < len(truncated), (
            f"position {position} is outside the sequence, which has "
            f"{len(truncated)} residues after truncation to "
            f"{model.max_sequence_length}"
        )
        assert truncated[position] == wildtype_aa, (
            f"variant states wild-type {wildtype_aa!r} at position {position}, but "
            f"the sequence has {truncated[position]!r}; the assay numbering and "
            "the reference sequence disagree"
        )

    positions = _distinct_positions(variants)
    log = get_logger("zero-shot")
    log.info(
        "masked-marginal scoring: %d variants over %d distinct positions on %s",
        len(variants),
        len(positions),
        model.device,
    )
    started = time.monotonic()

    # log_probabilities[position] is the model's distribution over the vocabulary
    # at that masked position. Keyed by position rather than by variant, which is
    # what lets variants sharing a site share a forward pass.
    log_probabilities: dict[int, np.ndarray] = {}
    batches = [
        positions[start : start + batch_size]
        for start in range(0, len(positions), batch_size)
    ]

    for batch_index, batch in enumerate(batches):
        _, _, tokens = model.batch_converter(
            [(str(position), truncated) for position in batch]
        )
        tokens = tokens.to(model.device)
        for row, position in enumerate(batch):
            # Residue i sits at token i + 1, because token 0 is BOS.
            tokens[row, position + 1] = model.alphabet.mask_idx

        with torch.inference_mode():
            result = model.model(tokens, repr_layers=[])
        logits = result["logits"]

        for row, position in enumerate(batch):
            row_logits = logits[row, position + 1].float()
            log_probabilities[position] = (
                torch.log_softmax(row_logits, dim=-1).cpu().numpy()
            )

        done = batch_index + 1
        if done % PROGRESS_EVERY_N_BATCHES == 0 or done == len(batches):
            log.info("  %d/%d position batches", done, len(batches))

    scores = np.empty(len(variants), dtype=np.float64)
    for index, variant in enumerate(variants):
        total = 0.0
        for position, wildtype_aa, mutant_aa in variant:
            distribution = log_probabilities[position]
            total += float(
                distribution[model.alphabet.get_idx(mutant_aa)]
                - distribution[model.alphabet.get_idx(wildtype_aa)]
            )
        scores[index] = total

    assert np.isfinite(
        scores
    ).all(), "masked-marginal scoring produced non-finite scores"
    log.info("scored %d variants in %.1fs", len(variants), time.monotonic() - started)
    return scores
