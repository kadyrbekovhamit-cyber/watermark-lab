"""Learn conditional preferences from counts, without access to a generator key.

The synthetic generator uses the project's pinned reference arithmetic. The
learner and evaluator accept only aggregate observations. This is neither a
Claude detector nor an implementation of exact secret-key recovery.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

SCHEMA = "conditional-choice-corpus-v1"
MAX_BYTES = 5_000_000
MAX_ROWS = 4096
MAX_COUNT = 1_000_000
SMOOTHING = 0.5


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def paired_cells(rows, width):
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("Each split needs 1 to 4096 aggregate rows")
    cells = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Rows must be objects")
        context, prefix, group = (row.get(k) for k in ("context", "prefix", "group"))
        if any(not isinstance(x, str) or not 1 <= len(x) <= 100 for x in (context, prefix)):
            raise ValueError("context and prefix must be nonempty identifiers of at most 100 characters")
        if group not in ("target", "control"):
            raise ValueError("group must be target or control")
        counts = row.get("counts")
        if (not isinstance(counts, list) or len(counts) != width or
                any(type(n) is not int or not 0 <= n <= MAX_COUNT for n in counts) or
                not 4 <= sum(counts) <= MAX_COUNT):
            raise ValueError("counts must match choices, with 4 to 1000000 total observations per row")
        cell = cells.setdefault((context, prefix), {})
        if group in cell:
            raise ValueError("Duplicate context/prefix/group aggregate")
        cell[group] = counts[:]
    for cell in cells.values():
        if set(cell) != {"target", "control"}:
            raise ValueError("Every context/prefix needs both target and matched control")
        if sum(cell["target"]) != sum(cell["control"]):
            raise ValueError("Target and control sample sizes must match within each cell")
    return cells


def validate_corpus(corpus):
    if not isinstance(corpus, dict) or corpus.get("schema") != SCHEMA:
        raise ValueError("Unsupported corpus schema")
    choices = corpus.get("choices")
    if (not isinstance(choices, list) or not 2 <= len(choices) <= 32 or
            any(not isinstance(x, str) or not 1 <= len(x) <= 100 for x in choices) or
            len(set(choices)) != len(choices)):
        raise ValueError("choices must contain 2 to 32 distinct string identifiers")
    provenance = corpus.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("kind") not in ("synthetic", "controlled", "unknown"):
        raise ValueError("Declare provenance.kind as synthetic, controlled or unknown")
    train = paired_cells(corpus.get("training"), len(choices))
    validation = paired_cells(corpus.get("validation"), len(choices))
    train_prefixes = {p for _, p in train}
    test_prefixes = {p for _, p in validation}
    if train_prefixes & test_prefixes:
        raise ValueError("Training and validation prefix identifiers must be disjoint")
    if len(train_prefixes) < 2 or len(test_prefixes) < 2:
        raise ValueError("Use at least two training and two held-out prefix identifiers")
    return train, validation


def learn_preferences(training, width):
    """Fit smoothed context-specific target/control log ratios. No key input."""
    totals = {}
    for (context, _), groups in training.items():
        table = totals.setdefault(context, {g: [0] * width for g in ("target", "control")})
        for group in table:
            table[group] = [a + b for a, b in zip(table[group], groups[group])]
    learned = {}
    for context, groups in totals.items():
        target, control = groups["target"], groups["control"]
        nt, nc = sum(target) + width * SMOOTHING, sum(control) + width * SMOOTHING
        learned[context] = [math.log((a + SMOOTHING) / nt) - math.log((b + SMOOTHING) / nc)
                            for a, b in zip(target, control)]
    return learned


def evaluate(learned, validation, permutations=4095, seed=20260916):
    if type(permutations) is not int or not 99 <= permutations <= 9999:
        raise ValueError("Use 99 to 9999 preselected label-swap permutations")
    differences, correct, covered, total = [], 0.0, 0, 0
    unseen_contexts = set()
    for (context, _), groups in sorted(validation.items()):
        n = sum(groups["target"])
        total += 2 * n
        if context not in learned:
            unseen_contexts.add(context)
            continue  # Abstention, not a no-watermark decision.
        scores = learned[context]
        target_mean = sum(c * s for c, s in zip(groups["target"], scores)) / n
        control_mean = sum(c * s for c, s in zip(groups["control"], scores)) / n
        differences.append(target_mean - control_mean)
        covered += 2 * n
        for count, score in zip(groups["target"], scores):
            correct += count * (1 if score > 0 else 0.5 if score == 0 else 0)
        for count, score in zip(groups["control"], scores):
            correct += count * (1 if score < 0 else 0.5 if score == 0 else 0)
    effect, p = None, None
    if differences:
        observed = sum(differences)
        effect = observed / len(differences)
        rng = random.Random(seed)
        exceed = 0
        for _ in range(permutations):
            swapped = sum(d if rng.getrandbits(1) else -d for d in differences)
            exceed += swapped >= observed - 1e-12
        p = (1 + exceed) / (permutations + 1)
    return {"validation_observations": total, "covered_observations": covered,
            "coverage": covered / total, "unseen_context_count": len(unseen_contexts),
            "paired_cells_scored": len(differences), "mean_paired_log_ratio_difference": effect,
            "balanced_accuracy_on_covered": correct / covered if covered else None,
            "paired_label_swap_p": p, "permutations": permutations,
            "permutation_seed": seed, "paired_cell_differences": differences,
            "inference": "Distribution comparison only: positive differences are not specific evidence of watermarking",
            "null_assumptions": "Frozen learner; independent validation cells; target/control groups exchangeable within each cell under the null",
            "unseen_context_policy": "Abstain; no learned preference and no vendor verdict"}


def analyze(corpus, permutations=4095):
    train, validation = validate_corpus(corpus)
    learned = learn_preferences(train, len(corpus["choices"]))
    return {"scope": "conditional_preference_learning_only", "claude_status": "unavailable",
            "key_recovered": False, "corpus_sha256": digest(corpus),
            "provenance_kind": corpus["provenance"]["kind"],
            "training_observations": sum(sum(v) for c in train.values() for v in c.values()),
            "training_contexts": len(learned), "smoothing": SMOOTHING,
            "evaluation": evaluate(learned, validation, permutations),
            "limitations": ["Declared provenance, sample independence and genuine matched controls cannot be verified from aggregate counts",
                            "Human prose, another model or a different deployment is not automatically an unwatermarked control",
                            "A context lookup table does not reveal a key or generalize to unseen contexts",
                            "Repeated experiments or selected thresholds require separate multiplicity control"],
            "input_text_exported": False, "external_requests": False}


def synthetic_corpus(scenario, training_repetitions=128, contexts=32, validation_repetitions=256):
    """Fresh single-choice trials, not natural language or a full LLM deployment.

    Prefix effects are simulated probability vectors. Every response has one
    context occurrence, so no within-response repeated-context masking arises.
    Public demonstration keys are used by this generator, never by the learner.
    """
    from watermark_lab import demo_profile, ngram_g_values, tournament_probabilities
    if scenario not in ("reference_watermark", "no_watermark", "ordinary_context_effect"):
        raise ValueError("Unknown synthetic scenario")
    choices = [101, 211, 307, 419]
    prefix_probs = [[.4, .3, .2, .1], [.1, .2, .3, .4], [.3, .1, .4, .2], [.2, .4, .1, .3]]
    context_rng = random.Random(81287)
    windows = [context_rng.sample(range(1024), 4) for _ in range(contexts)]
    bits = [[ngram_g_values(window + [c], demo_profile()) for c in choices] for window in windows]
    splits = {"training": [], "validation": []}
    for split, prefixes, repetitions in (("training", [0, 1], training_repetitions),
                                       ("validation", [2, 3], validation_repetitions)):
        for prefix in prefixes:
            for context in range(contexts):
                for group in ("target", "control"):
                    probs = prefix_probs[prefix][:]
                    if group == "target" and scenario == "reference_watermark":
                        probs = tournament_probabilities(probs, bits[context])
                    elif group == "target" and scenario == "ordinary_context_effect":
                        # An explicit non-keyed context-dependent preference.
                        probs[context % len(choices)] *= 5
                    rng = random.Random(int(digest(["choice-seed-v1", scenario, split, prefix, context, group])[:16], 16))
                    counts = [0] * len(choices)
                    for choice in rng.choices(range(len(choices)), weights=probs, k=repetitions):
                        counts[choice] += 1
                    splits[split].append({"context": "context-%03d" % context,
                                          "prefix": "prefix-%d" % prefix, "group": group, "counts": counts})
    return {"schema": SCHEMA, "choices": [str(c) for c in choices],
            "provenance": {"kind": "synthetic", "scenario": scenario,
                           "generator": "Finite-choice sampler using pinned SynthID arithmetic or explicit controls; not Claude",
                           "tokenizer_id": "synthetic-integer-vocabulary-1024-v1",
                           "context_seed": 81287, "training_repetitions_per_group_cell": training_repetitions,
                           "validation_repetitions_per_group_cell": validation_repetitions}, **splits}


def benchmark():
    scenarios = {}
    for scenario in ("reference_watermark", "no_watermark", "ordinary_context_effect"):
        curve = []
        for repetitions in (8, 32, 128):
            corpus = synthetic_corpus(scenario, repetitions)
            result = analyze(corpus)
            # The same independent validation grid is reused for descriptive curves.
            # Only the preselected maximum training budget is the primary run.
            if repetitions != 128:
                result["evaluation"].pop("paired_label_swap_p")
                result["role"] = "descriptive_learning_curve; no hypothesis-test result"
            else:
                result["role"] = "preselected_primary_budget_for_this_synthetic_scenario"
            curve.append(result)
        scenarios[scenario] = curve
    corpus = synthetic_corpus("reference_watermark")
    train, validation = validate_corpus(corpus)
    learned = learn_preferences(train, len(corpus["choices"]))
    unseen = {(context + "-unseen", prefix): groups for (context, prefix), groups in validation.items()}
    return {"experiment": "conditional-preference-learning-v1", "claude_status": "unavailable",
            "key_recovered": False, "natural_language_samples": 0, "target_model_calls": 0,
            "design": {"contexts": 32, "choices": 4, "training_prefixes": 2, "validation_prefixes": 2,
                       "training_repetitions": [8, 32, 128], "validation_repetitions": 256,
                       "all_runs_reported": True, "primary_budget_selected_before_execution": 128,
                       "learner_inputs": "Counts and context/prefix/group identifiers only; no keys or g-values",
                       "not_a_reproduction_of_eth_experiments": True},
            "scenarios": scenarios, "unseen_context_check": evaluate(learned, unseen),
            "interpretation": "Recovering some context preferences is possible in this controlled setup; the ordinary-context-effect control demonstrates non-specificity. No exact key recovery or Claude test."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("benchmark", help="Run bounded synthetic learning experiments offline")
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--fixture", type=Path)
    a = sub.add_parser("analyze", help="Analyze an explicit matched aggregate corpus locally")
    a.add_argument("corpus", type=Path)
    a.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "benchmark":
            if args.fixture and args.fixture.resolve() == args.output.resolve():
                raise ValueError("Report and corpus fixture must use different output paths")
            report = benchmark()
            if args.fixture:
                args.fixture.parent.mkdir(parents=True, exist_ok=True)
                args.fixture.write_text(json.dumps(synthetic_corpus("reference_watermark"), indent=2) + "\n")
        else:
            if not args.corpus.is_file() or args.corpus.stat().st_size > MAX_BYTES:
                raise ValueError("Corpus must be a regular JSON file of at most 5 MB")
            if args.output.resolve() == args.corpus.resolve():
                raise ValueError("Output must not overwrite the input corpus")
            report = analyze(json.loads(args.corpus.read_text(encoding="utf-8")))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print("Saved %s; Claude status unavailable; no key recovered." % args.output)


if __name__ == "__main__":
    main()
