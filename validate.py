"""Bounded single-process validation; never calls a model or an external API."""
import ast
import hashlib
import json
import math
import os
import random
import statistics
import time
import types
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np
import watermark_lab as lab

ROOT = Path(__file__).resolve().parent


def reference_functions():
    """Execute only three pinned arithmetic functions, on a small NumPy adapter.

    No PyTorch/Transformers import or full upstream model execution is claimed.
    """
    torch = types.SimpleNamespace(LongTensor=lambda x: np.array(x, dtype=np.int64),
                                  FloatTensor=np.ndarray, add=np.add, mul=np.multiply)
    scope = {"torch": torch}
    tree = ast.parse((ROOT / "reference/hashing_function.py").read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "accumulate_hash")
    # Only annotations are replaced, to avoid importing PyTorch.
    def compile_function(node, namespace):
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "pinned-reference", "exec"), namespace)
        return namespace[node.name]
    accumulate = compile_function(fn, scope)
    scope["hashing_function"] = types.SimpleNamespace(accumulate_hash=accumulate)
    tree = ast.parse((ROOT / "reference/logits_processing.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SynthIDLogitsProcessor")
    get_gvals = compile_function(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_gvals"), scope)
    mean_tree = ast.parse((ROOT / "reference/detector_mean.py").read_text())
    mean = compile_function(next(node for node in mean_tree.body if isinstance(node, ast.FunctionDef) and node.name == "mean_score"), {"jnp": np})
    return accumulate, get_gvals, mean


def check_reference():
    lock = json.loads((ROOT / "reference/SOURCE_LOCK.json").read_text())
    for entry in lock["files"]:
        assert hashlib.sha256((ROOT / entry["local_path"]).read_bytes()).hexdigest() == entry["sha256"]
    accumulate, gvals, mean = reference_functions()
    rng = random.Random(90210)
    profile = lab.demo_profile()
    cases = [[rng.randrange(0, 1 << 40) for _ in range(5)] for _ in range(256)]
    cases += [[0] * 5, [(1 << 63) - 1] * 5, [0, 1, 2, (1 << 63) - 1, 0]]
    grams = np.array(cases, dtype=np.int64)
    with np.errstate(over="ignore"):
        hashed = accumulate(np.full(len(cases), lab.hash_iv(profile), dtype=np.int64), grams)
        keyed = accumulate(hashed[:, None], np.array(profile["keys"], dtype=np.int64)[None, :, None])
        reference_bits = gvals(None, keyed)
    actual = np.array([lab.ngram_g_values(row, profile) for row in cases])
    assert np.array_equal(actual, reference_bits)
    mask = np.array([[i % 3 != 0 for i in range(len(cases))]])
    reference_mean = float(mean(reference_bits[None, :, :], mask)[0])
    scalar_mean = sum(sum(row) for row, enabled in zip(actual.tolist(), mask[0]) if enabled) / (sum(mask[0]) * len(profile["keys"]))
    assert abs(reference_mean - scalar_mean) < 1e-15
    return {"upstream_commit": lock["commit"], "ngram_cases": len(cases),
            "binary_scores_compared": int(actual.size), "all_exact": True,
            "masked_mean_agrees": True,
            "method": "Pinned arithmetic function bodies executed with a minimal NumPy adapter; not a full PyTorch runtime test"}


def summarize(results):
    scores = [r["mean_g_score"] for r in results]
    return {"documents": len(results), "detected": sum(r["status"] == "reference_signal_detected" for r in results),
            "inconclusive": sum(r["status"] not in ("reference_signal_detected", "reference_signal_not_detected") for r in results),
            "mean_score": statistics.mean(scores), "min_score": min(scores), "max_score": max(scores)}


def main():
    started = time.monotonic()
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    parity = check_reference()
    print("Reference arithmetic parity:", parity["binary_scores_compared"], "exact bit matches", flush=True)
    profile = lab.demo_profile()
    tokenizer = profile["tokenizer_id"]
    null_scores = []
    for seed in range(10000, 10199):
        result = lab.inspect_tokens(lab.generate_demo(seed, False), profile, tokenizer)
        assert result["scored_contexts"] == 124
        null_scores.append(result["mean_g_score"])
    calibration = {"profile_fingerprint": lab.fingerprint(profile), "tokenizer_id": tokenizer,
                   "scored_contexts": 124, "input_tokens": 128, "null_scores": null_scores,
                   "population": "Synthetic IID uniform integer tokens in [0,1023], length 128; seed range 10000:10199",
                   "seed_range": [10000, 10199], "not_real_language_calibration": True}
    (evidence / "demo-calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print("Calibration complete: 199 independent synthetic documents", flush=True)
    groups = {}
    for kind, marked, foreign, first in [("unmarked", False, False, 20000),
                                        ("marked_correct_key", True, False, 30000),
                                        ("marked_other_key", True, True, 40000)]:
        results = []
        for seed in range(first, first + 100):
            tokens = lab.generate_demo(seed, marked, foreign=foreign)
            result = lab.inspect_tokens(tokens, profile, tokenizer, calibration=calibration)
            results.append({"seed": seed, "status": result["status"],
                            "mean_g_score": result["mean_g_score"],
                            "empirical_tail_rank": result["empirical_tail_rank"],
                            "scored_contexts": result["scored_contexts"]})
            if seed == first:
                payload = {"token_ids": tokens, "profile": profile, "tokenizer_id": tokenizer,
                           "calibration": calibration}
                fixture = {"kind": kind, "source": "synthetic sampler, not Claude", "payload": payload,
                           "result": result}
                (evidence / ("demo-" + kind + ".json")).write_text(json.dumps(fixture, indent=2) + "\n")
        groups[kind] = {"summary": summarize(results), "seed_range": [first, first + 100], "results": results}
        print(kind, json.dumps(groups[kind]["summary"]), flush=True)
    report = {"arithmetic_parity": parity, "calibration_documents": 199,
              "fixed_alpha": lab.ALPHA, "tokens_per_document": 128,
              "scored_contexts_per_document": 124, "keys": 8,
              "groups": groups, "elapsed_seconds": round(time.monotonic() - started, 3),
              "scope": "Synthetic reference-scheme experiment only. No claim of Claude detection, natural-language false-positive rate, or key recovery.",
              "selection": "Fixed seed ranges and threshold; all 300 held-out outcomes retained, no cherry-picking"}
    (evidence / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved evidence/benchmark.json", flush=True)


if __name__ == "__main__":
    main()
