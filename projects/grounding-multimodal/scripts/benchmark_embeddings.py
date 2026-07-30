"""Measure, attribute, and compare the cost of `biotp.embeddings.embed_sequences`.

Written for issue #3, where embedding 13,858 proteins at 2.4 seq/s dominated the
Project 1 pipeline. The issue's requirement is a comparison, so this script exists
to make the before and after numbers apples to apples: same sample, same device,
same checkpoint, recorded to JSON with a fingerprint that fails loudly if the two
runs did not in fact embed the same sequences.

Four modes:

    --mode profile     attribute time to tokenization, forward, and pooling
    --mode benchmark   seq/s on a fixed length-matched sample, per batch size
    --mode ab          time the old and new implementations against each other
    --mode reference   freeze a small correctness anchor under tests/data/

`--mode ab` is the one to trust for a speedup claim. Single-shot before/after
timing on a laptop measures the laptop: an identical configuration here came out
3.7x apart either side of a run that exhausted swap. The A/B runs both
implementations against the same machine state and checks they agree numerically.

The fast loop is a 1,000-protein sample drawn to match the full length
distribution, not the shortest 1,000, which would flatter any fix to padding
waste. The authoritative measure is still the full run: compare the
`build feature blocks` step in `results/run_manifest_all.json`.

Run from the repo root, after prepare_data.py:

    python projects/grounding-multimodal/scripts/benchmark_embeddings.py \
        --mode benchmark --tag baseline

Nothing here goes through the embedding cache. These modes call `embed_sequences`
directly, so a timing can never be a cache hit in disguise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biotp.embeddings import (
    MAX_SEQUENCE_LENGTH,
    Esm2Bundle,
    _length_bucketed_batches,
    _mean_pool_residues,
    embed_sequences,
    load_esm2,
)
from biotp.runlog import DEFAULT_LOG_DIR, get_logger, run_context

# The checkpoint Project 1 actually runs, so the numbers transfer to the pipeline.
SEQUENCE_ENCODER = "esm2_t12_35M_UR50D"

# The correctness anchor only needs determinism, not capacity, so it uses the
# smallest ESM-2 and stays a ~60 KB file that git can hold.
REFERENCE_CHECKPOINT = "esm2_t6_8M_UR50D"
REFERENCE_SEQUENCES = 24

SAMPLE_SIZE = 1000
SAMPLE_SEED = 0
LENGTH_STRATA = 10

# 650M against the 35M actually benchmarked. Used only for the roadmap projection,
# which assumes the pipeline stays compute-bound and is a planning estimate rather
# than a measurement.
PARAMETER_RATIO_650M = 18

# Phase attribution runs every batch twice, once mirrored and once for real, so it
# uses a smaller sample than the throughput benchmark.
PROFILE_SEQUENCES = 200

DUAL_LOCALIZATION = "Cytoplasm-Nucleus"

log = get_logger("benchmark-embeddings")


def load_sequences(data_root: Path) -> list[str]:
    """Load the same single-label protein cohort the pipeline embeds."""
    path = data_root / "processed" / "deeploc_annotated.parquet"
    assert path.exists(), f"missing {path}; run prepare_data.py first"

    table = pd.read_parquet(path)
    single = table[table["localization"] != DUAL_LOCALIZATION].reset_index(drop=True)
    assert len(single) > 0, "no single-label proteins found"
    return single["sequence"].tolist()


def truncated_lengths(sequences: list[str], limit: int) -> list[int]:
    """Residue counts after truncation, which is what the model actually sees."""
    return [min(len(sequence), limit) for sequence in sequences]


def length_summary(lengths: list[int]) -> dict[str, Any]:
    """The distribution that drives the cost, in the form the issue reports it."""
    ordered = sorted(lengths)
    return {
        "count": len(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "median": ordered[len(ordered) // 2],
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
        "total_residues": sum(lengths),
    }


def length_matched_sample(lengths: list[int], size: int, seed: int) -> list[int]:
    """Draw indices whose length distribution matches the full set's.

    A uniform random draw would match only in expectation. Stratifying by length
    decile makes it hold for the single draw actually used, which matters here
    because the whole hypothesis is that cost is driven by the length spread.

    Returns indices in dataset order, so the sample reproduces the batch
    composition production sees rather than an accidentally tidier one.
    """
    assert size <= len(lengths), f"cannot draw {size} from {len(lengths)}"

    by_length = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    rng = np.random.default_rng(seed)

    chosen: list[int] = []
    for stratum in np.array_split(np.asarray(by_length), LENGTH_STRATA):
        take = min(round(size * len(stratum) / len(by_length)), len(stratum))
        chosen.extend(int(index) for index in rng.choice(stratum, take, replace=False))

    # Per-stratum rounding can land a few either side of the target.
    if len(chosen) > size:
        chosen = [int(index) for index in rng.choice(chosen, size, replace=False)]
    elif len(chosen) < size:
        already = set(chosen)
        spare = [index for index in by_length if index not in already]
        short_by = size - len(chosen)
        chosen.extend(
            int(index) for index in rng.choice(spare, short_by, replace=False)
        )

    assert len(set(chosen)) == size, "sample contains duplicates"
    return sorted(chosen)


def fingerprint(sequences: list[str]) -> str:
    """Identify the exact sample, so a before/after pair cannot silently differ."""
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(b"\x00")
        digest.update(sequence.encode())
    return digest.hexdigest()[:16]


def padding_cost(lengths: list[int], batch_size: int) -> dict[str, Any]:
    """Residue slots and attention units pushed through the model, two orderings.

    Feed-forward work scales with padded slots and attention with their square, so
    these two numbers bound what length-bucketed batching can and cannot buy.
    """

    def cost(order: list[int]) -> tuple[int, int]:
        slots = 0
        attention = 0
        for start in range(0, len(order), batch_size):
            batch = [lengths[index] for index in order[start : start + batch_size]]
            padded = max(batch)
            slots += padded * len(batch)
            attention += padded * padded * len(batch)
        return slots, attention

    useful = sum(lengths)
    dataset_slots, dataset_attention = cost(list(range(len(lengths))))
    bucketed_slots, bucketed_attention = cost(
        sorted(range(len(lengths)), key=lambda index: -lengths[index])
    )
    return {
        # Every figure below is specific to this batch size, and the ratios move a
        # lot with it: quoting one without the other is how a batch-8 slot count
        # ends up labelled as a batch-16 result.
        "batch_size": batch_size,
        "useful_residues": useful,
        "dataset_order": {
            "padded_slots": dataset_slots,
            "attention": dataset_attention,
        },
        "bucketed": {"padded_slots": bucketed_slots, "attention": bucketed_attention},
        "slot_waste_ratio": round(dataset_slots / useful, 3),
        "slot_reduction": round(dataset_slots / bucketed_slots, 3),
        "attention_reduction": round(dataset_attention / bucketed_attention, 3),
    }


def synchronize(device: str) -> None:
    """Block until queued device work finishes, so a phase timing means something.

    Without this the forward pass is asynchronous and its cost lands in whichever
    later line first touches the result, which on this code is the host transfer.
    Timing the phases separately requires making them separate in wall-clock too.
    """
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def profile_phases(
    model: Esm2Bundle, sequences: list[str], batch_size: int
) -> dict[str, Any]:
    """Attribute embedding time to tokenization, forward, and pooling plus transfer.

    This mirrors the loop inside `embed_sequences` rather than calling it, because
    the phases only separate in wall-clock if a device sync sits between them, and
    `embed_sequences` should not carry sync points it does not need. The batching
    and pooling come from the library's own helpers, so the only thing the mirror
    reimplements is the timing.

    Kept honest two ways: the mirror's vectors must equal the real function's, and
    its total must land within 25% of the real total. A mirror that has drifted
    fails an assertion instead of reporting a fiction.
    """
    import torch

    truncated = [sequence[: model.max_sequence_length] for sequence in sequences]
    lengths = [len(sequence) for sequence in truncated]
    phases = {"tokenize": 0.0, "forward": 0.0, "pool_and_transfer": 0.0}
    out = np.empty((len(truncated), model.embedding_dim), dtype=np.float32)

    started = time.monotonic()
    for indices in _length_bucketed_batches(lengths, batch_size):
        batch = [truncated[index] for index in indices]

        mark = time.monotonic()
        _, _, tokens = model.batch_converter(
            [(str(index), sequence) for index, sequence in zip(indices, batch)]
        )
        tokens = tokens.to(model.device)
        synchronize(model.device)
        phases["tokenize"] += time.monotonic() - mark

        mark = time.monotonic()
        with torch.inference_mode():
            result = model.model(tokens, repr_layers=[model.repr_layer])
        representations = result["representations"][model.repr_layer]
        synchronize(model.device)
        phases["forward"] += time.monotonic() - mark

        mark = time.monotonic()
        batch_lengths = torch.tensor(
            [len(sequence) for sequence in batch], device=representations.device
        )
        out[indices] = (
            _mean_pool_residues(representations, batch_lengths).float().cpu().numpy()
        )
        phases["pool_and_transfer"] += time.monotonic() - mark

    mirrored_total = time.monotonic() - started

    started = time.monotonic()
    real = embed_sequences(model, sequences, batch_size)
    real_total = time.monotonic() - started

    np.testing.assert_allclose(out, real, rtol=1e-5, atol=1e-6)
    drift = abs(mirrored_total - real_total) / real_total
    assert drift < 0.25, (
        f"phase mirror is {drift:.0%} off the real function "
        f"({mirrored_total:.1f}s vs {real_total:.1f}s); it no longer models it"
    )

    return {
        "sequences": len(truncated),
        "batch_size": batch_size,
        "mirrored_total_seconds": round(mirrored_total, 2),
        "real_total_seconds": round(real_total, 2),
        "phases_seconds": {name: round(value, 2) for name, value in phases.items()},
        "phases_share": {
            name: round(value / mirrored_total, 4) for name, value in phases.items()
        },
    }


def embed_sequences_v1(
    model: Esm2Bundle, sequences: list[str], batch_size: int
) -> np.ndarray:
    """The pre-#3 implementation: dataset-order batches, pooled one sequence at a time.

    Kept here, and only here, as the comparison arm for `--mode ab`. Timing the
    current code against a number recorded twenty minutes earlier measures the
    machine as much as the code: on this laptop an identical configuration came out
    3.7x apart either side of a run that exhausted swap. The only way to compare
    implementations honestly is to run both against the same machine state, which
    means the old one has to still exist somewhere.

    It is deliberately not in `biotp`: nothing should import it, and no cache should
    ever be written from it.
    """
    import torch

    truncated = [sequence[: model.max_sequence_length] for sequence in sequences]
    out = np.empty((len(truncated), model.embedding_dim), dtype=np.float32)

    for start in range(0, len(truncated), batch_size):
        batch = truncated[start : start + batch_size]
        _, _, tokens = model.batch_converter(
            [(str(index), sequence) for index, sequence in enumerate(batch)]
        )
        tokens = tokens.to(model.device)

        with torch.inference_mode():
            result = model.model(tokens, repr_layers=[model.repr_layer])
        representations = result["representations"][model.repr_layer]

        for index, sequence in enumerate(batch):
            residues = representations[index, 1 : len(sequence) + 1]
            out[start + index] = residues.mean(dim=0).float().cpu().numpy()

    return out


def timed(
    implementation: Any, model: Esm2Bundle, sequences: list[str], batch_size: int
) -> tuple[float, np.ndarray]:
    """Run one implementation once, returning wall time and its vectors."""
    started = time.monotonic()
    out = implementation(model, sequences, batch_size)
    synchronize(model.device)
    return time.monotonic() - started, out


def compare_implementations(
    model: Esm2Bundle, sequences: list[str], batch_size: int, repeats: int
) -> dict[str, Any]:
    """Time the old and new implementations alternately, on identical input.

    Alternating matters more than repeating: thermal state, memory pressure, and
    whatever else the laptop is doing all drift over minutes, and a before/after
    pair separated by half an hour cannot tell that drift apart from the change
    being measured. Interleaving makes both arms share whatever the machine is
    doing, and the median over repeats absorbs the rest.

    The order flips every repeat, rather than always running v1 first. Within a
    single repeat the second arm inherits whatever the first one did to the
    machine, so a fixed order biases one arm consistently and a median cannot
    remove it. Flipping makes that bias cancel across repeats instead.

    The two arms must also agree numerically, which is checked here rather than
    assumed: a speedup that changed the answers would not be a speedup.
    """
    assert repeats > 0, f"repeats must be positive, got {repeats}"

    # One untimed pass per arm, so neither pays for lazily compiled kernels.
    warmup = sequences[: min(len(sequences), 8)]
    timed(embed_sequences_v1, model, warmup, batch_size)
    timed(embed_sequences, model, warmup, batch_size)

    old_times: list[float] = []
    new_times: list[float] = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            old_seconds, old_out = timed(
                embed_sequences_v1, model, sequences, batch_size
            )
            new_seconds, new_out = timed(embed_sequences, model, sequences, batch_size)
        else:
            new_seconds, new_out = timed(embed_sequences, model, sequences, batch_size)
            old_seconds, old_out = timed(
                embed_sequences_v1, model, sequences, batch_size
            )
        np.testing.assert_allclose(new_out, old_out, rtol=1e-4, atol=1e-5)

        old_times.append(old_seconds)
        new_times.append(new_seconds)
        log.info(
            "  repeat %d/%d: v1 %.1fs (%.2f seq/s), v2 %.1fs (%.2f seq/s), %.2fx",
            repeat + 1,
            repeats,
            old_seconds,
            len(sequences) / old_seconds,
            new_seconds,
            len(sequences) / new_seconds,
            old_seconds / new_seconds,
        )

    old_median = statistics.median(old_times)
    new_median = statistics.median(new_times)
    return {
        "mode": "ab",
        "batch_size": batch_size,
        "sequences": len(sequences),
        "repeats": repeats,
        "v1_seconds": [round(value, 2) for value in old_times],
        "v2_seconds": [round(value, 2) for value in new_times],
        "v1_median_seconds": round(old_median, 2),
        "v2_median_seconds": round(new_median, 2),
        "v1_sequences_per_second": round(len(sequences) / old_median, 3),
        "v2_sequences_per_second": round(len(sequences) / new_median, 3),
        "speedup": round(old_median / new_median, 3),
    }


def benchmark(
    model: Esm2Bundle, sequences: list[str], batch_size: int
) -> dict[str, Any]:
    """Time `embed_sequences` end to end and report throughput."""
    started = time.monotonic()
    out = embed_sequences(model, sequences, batch_size)
    elapsed = time.monotonic() - started

    assert out.shape == (len(sequences), model.embedding_dim), "unexpected shape"
    return {
        "batch_size": batch_size,
        "sequences": len(sequences),
        "seconds": round(elapsed, 2),
        "sequences_per_second": round(len(sequences) / elapsed, 3),
    }


def project_full_run(seconds_per_sequence: float, population: int) -> dict[str, Any]:
    """Extrapolate the sample's rate to the full cohort, and to the 650M checkpoint.

    The 650M multiplier is the parameter ratio, which assumes the pipeline stays
    compute-bound. It is a planning estimate for `PLANNING.md`, not a measurement,
    and is labelled as such wherever it is reported.
    """
    full = seconds_per_sequence * population
    return {
        "full_cohort_seconds": round(full, 1),
        "full_cohort_minutes": round(full / 60, 1),
        "projected_650m_minutes": round(full * PARAMETER_RATIO_650M / 60, 1),
    }


def evenly_spaced_by_length(sequences: list[str], count: int) -> list[int]:
    """Pick indices spanning the length range, deterministically and without an RNG.

    The anchor should exercise the interesting cases by construction: the shortest
    sequences, the longest one (which is truncated), and a spread in between.
    """
    assert count <= len(sequences), f"cannot pick {count} from {len(sequences)}"
    by_length = sorted(
        range(len(sequences)), key=lambda index: (len(sequences[index]), index)
    )
    positions = np.linspace(0, len(by_length) - 1, count).round().astype(int)
    picked = sorted({by_length[position] for position in positions})
    assert len(picked) == count, "evenly spaced picks collided"
    return picked


def write_reference(sequences: list[str], destination: Path) -> dict[str, Any]:
    """Freeze vectors from the current implementation as a correctness anchor.

    Saved before changing the embedding code, so the rewrite is checked against
    what the pipeline actually produced rather than against itself.
    """
    picked = evenly_spaced_by_length(sequences, REFERENCE_SEQUENCES)
    chosen = [sequences[index] for index in picked]

    model = load_esm2(REFERENCE_CHECKPOINT)
    vectors = embed_sequences(model, chosen, batch_size=4)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        sequences=np.asarray(chosen),
        embeddings=vectors,
        checkpoint=REFERENCE_CHECKPOINT,
    )
    return {
        "path": str(destination),
        "checkpoint": REFERENCE_CHECKPOINT,
        "sequences": len(chosen),
        "lengths": [len(sequence) for sequence in chosen],
        "fingerprint": fingerprint(chosen),
    }


def compare(previous: dict[str, Any], current: dict[str, Any]) -> str:
    """Render the before/after table the issue asks for, guarding the comparison.

    Only `--mode benchmark` payloads are comparable this way. A profile payload
    has no `runs`, and an A/B payload already carries both arms and its own
    speedup, so pointing this at either is a mistake worth an error rather than a
    confusing table or a KeyError from somewhere deeper.
    """
    for label, payload in (("previous", previous), ("current", current)):
        assert payload.get("mode") == "benchmark", (
            f"{label} payload is mode {payload.get('mode')!r}, not 'benchmark'; "
            "an A/B payload already reports its own speedup"
        )
        assert payload["runs"], f"{label} payload has no timed runs"

    assert previous["sample_fingerprint"] == current["sample_fingerprint"], (
        "the two runs embedded different sequences, so their times are not "
        f"comparable ({previous['sample_fingerprint']} vs "
        f"{current['sample_fingerprint']})"
    )
    assert previous["device"] == current["device"], "different devices"
    assert previous["checkpoint"] == current["checkpoint"], "different checkpoints"

    lines = [
        f"| {'run':<24} | batch | seq/s | sample s | projected full-run min | speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    baseline_rate = previous["runs"][0]["sequences_per_second"]
    for label, payload in (("before", previous), ("after", current)):
        for entry in payload["runs"]:
            projection = project_full_run(
                1.0 / entry["sequences_per_second"], payload["population"]
            )
            lines.append(
                f"| {label + ' (' + payload['tag'] + ')':<24} "
                f"| {entry['batch_size']} "
                f"| {entry['sequences_per_second']:.2f} "
                f"| {entry['seconds']:.1f} "
                f"| {projection['full_cohort_minutes']:.1f} "
                f"| {entry['sequences_per_second'] / baseline_rate:.2f}x |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("profile", "benchmark", "ab", "reference"), required=True
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="alternating rounds per batch size in --mode ab",
    )
    parser.add_argument("--tag", type=str, help="label for this run's output file")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("projects/grounding-multimodal/results"),
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path("tests/data/reference_embeddings.npz"),
    )
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16])
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="an earlier benchmark JSON to tabulate this run against",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    assert (
        args.mode == "reference" or args.tag
    ), "--tag is required, so before and after land in different files"

    with run_context(
        f"benchmark-embeddings-{args.mode}", log_dir=args.log_dir, params=vars(args)
    ) as run:
        with run.step("load sequences"):
            population = load_sequences(args.data_root)
            run.record("population", len(population))

        if args.mode == "reference":
            with run.step("write reference vectors"):
                summary = write_reference(population, args.reference_path)
                run.record("reference", summary)
            log.info("reference written to %s", args.reference_path)
            return 0

        lengths = truncated_lengths(population, MAX_SEQUENCE_LENGTH)
        indices = length_matched_sample(lengths, args.sample_size, SAMPLE_SEED)
        sample = [population[index] for index in indices]
        sample_lengths = [lengths[index] for index in indices]

        # One entry per batch size, because the padding ratios move with it and a
        # single unlabelled figure invites quoting one batch size's numerator
        # against another's denominator.
        padding = [padding_cost(lengths, size) for size in args.batch_sizes]
        sample_digest = fingerprint(sample)

        run.record("full_lengths", length_summary(lengths))
        run.record("sample_lengths", length_summary(sample_lengths))
        run.record("sample_fingerprint", sample_digest)
        run.record("padding_cost", padding)

        with run.step(f"load {SEQUENCE_ENCODER}"):
            model = load_esm2(SEQUENCE_ENCODER)
            run.record("device", model.device)

        results: list[dict[str, Any]] = []
        if args.mode == "profile":
            assert (
                len(args.batch_sizes) == 1
            ), f"--mode profile takes one batch size, got {args.batch_sizes}"
            with run.step("profile phases"):
                profile = profile_phases(
                    model, sample[:PROFILE_SEQUENCES], args.batch_sizes[0]
                )
                run.record("profile", profile)
        elif args.mode == "ab":
            for batch_size in args.batch_sizes:
                with run.step(f"a/b batch_size={batch_size}"):
                    entry = compare_implementations(
                        model, sample, batch_size, args.repeats
                    )
                    results.append(entry)
                    run.record(f"ab_batch_{batch_size}", entry)
        else:
            for batch_size in args.batch_sizes:
                with run.step(f"benchmark batch_size={batch_size}"):
                    entry = benchmark(model, sample, batch_size)
                    entry["projection"] = project_full_run(
                        entry["seconds"] / entry["sequences"], len(population)
                    )
                    results.append(entry)
                    run.record(f"batch_{batch_size}", entry)

        payload = {
            "tag": args.tag,
            "mode": args.mode,
            "checkpoint": SEQUENCE_ENCODER,
            "device": model.device,
            "population": len(population),
            "sample_size": args.sample_size,
            "sample_fingerprint": sample_digest,
            "sample_lengths": length_summary(sample_lengths),
            "full_lengths": length_summary(lengths),
            "padding_cost": padding,
            "runs": results,
            "profile": run.records.get("profile"),
        }
        args.results_dir.mkdir(parents=True, exist_ok=True)
        destination = args.results_dir / f"embedding_benchmark_{args.tag}.json"
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        log.info("wrote %s", destination)

        if args.compare_to:
            assert args.compare_to.exists(), f"missing {args.compare_to}"
            table = compare(json.loads(args.compare_to.read_text()), payload)
            log.info("\n%s", table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
