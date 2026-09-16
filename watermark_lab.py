"""Local SynthID reference scorer. No claim of compatibility with Claude's key.

Scalar adaptations of the DeepMind reference arithmetic, Apache-2.0:
see reference/SOURCE_LOCK.json, reference/LICENSE and NOTICE.
No third-party runtime dependencies; Python 3.9+.
"""
import hashlib
import json
import math
import random
import re
import struct
import unicodedata
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALGORITHM = "deepmind-synthid-addb4a15-int64-le-v1"
MULTIPLIER = 6364136223846793005
MASK64 = (1 << 64) - 1
MAX_TEXT = 100_000
MAX_TOKENS = 4096
MIN_CONTEXTS = 64  # Application policy, not a claimed vendor requirement.
ALPHA = 0.01


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("%s must be an integer in [%s, %s]" % (name, low, high))
    return value


def signed64(value):
    value &= MASK64
    return value - (1 << 64) if value >= 1 << 63 else value


def accumulate_hash(current, data):
    for token in data:
        current = signed64((current + token) * MULTIPLIER + 1)
    return current


def g_bit(value):
    # Signed overflow followed by arithmetic right shift, exactly as pinned code.
    for _ in range(12):
        value = accumulate_hash(value, [1]) >> 5
    return (value >> 30) % 2


def validate_profile(profile):
    if not isinstance(profile, dict) or profile.get("algorithm") != ALGORITHM:
        raise ValueError("Unsupported algorithm: a compatible explicit profile is required")
    keys = profile.get("keys")
    if not isinstance(keys, list) or not 1 <= len(keys) <= 32:
        raise ValueError("keys must contain 1 to 32 signed int64 integers")
    for key in keys:
        integer(key, "key", -(1 << 63), (1 << 63) - 1)
    if len(set(keys)) != len(keys):
        raise ValueError("keys must be distinct")
    n = integer(profile.get("ngram_len"), "ngram_len", 2, 32)
    history = integer(profile.get("context_history_size"), "context_history_size", 1, 4096)
    tokenizer = profile.get("tokenizer_id")
    if not isinstance(tokenizer, str) or not 1 <= len(tokenizer) <= 200:
        raise ValueError("A nonempty tokenizer_id is required")
    return {"algorithm": ALGORITHM, "keys": keys[:], "ngram_len": n,
            "context_history_size": history, "tokenizer_id": tokenizer}


def fingerprint(profile):
    encoded = json.dumps(validate_profile(profile), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def hash_iv(profile):
    # Upstream uses native int64 bytes. We explicitly support little-endian profiles.
    data = struct.pack("<" + "q" * len(profile["keys"]), *profile["keys"])
    return int.from_bytes(hashlib.sha256(data).digest(), "big") % ((1 << 63) - 1)


def ngram_g_values(ngram, profile):
    h = accumulate_hash(hash_iv(profile), ngram)
    return [g_bit(accumulate_hash(h, [key])) for key in profile["keys"]]


def inspect_tokens(tokens, profile, tokenizer_id, eos_token_id=None, calibration=None):
    profile = validate_profile(profile)
    if tokenizer_id != profile["tokenizer_id"]:
        raise ValueError("Tokenizer mismatch: use the exact IDs and tokenizer of the generator")
    if not isinstance(tokens, list) or len(tokens) > MAX_TOKENS:
        raise ValueError("token_ids must be a list of at most %s integers" % MAX_TOKENS)
    for token in tokens:
        integer(token, "token_id", 0, (1 << 63) - 1)
    original_length = len(tokens)
    if eos_token_id is not None:
        integer(eos_token_id, "eos_token_id", 0, (1 << 63) - 1)
        if eos_token_id in tokens:
            tokens = tokens[:tokens.index(eos_token_id)]
    n, size, iv = profile["ngram_len"], profile["context_history_size"], hash_iv(profile)
    history = deque([0] * size)
    counts = Counter(history)
    layer_sums = [0] * len(profile["keys"])
    used, skipped, rows = 0, 0, []
    for position in range(n - 1, len(tokens)):
        window = tokens[position - n + 1:position + 1]
        context_hash = accumulate_hash(iv, window[:-1])
        repeated = counts[context_hash] > 0
        outgoing = history.pop()
        counts[outgoing] -= 1
        if not counts[outgoing]:
            del counts[outgoing]
        history.appendleft(context_hash)
        counts[context_hash] += 1
        bits = ngram_g_values(window, profile)
        if repeated:
            skipped += 1
        else:
            used += 1
            layer_sums = [a + b for a, b in zip(layer_sums, bits)]
        if len(rows) < 256:
            rows.append({"position": position, "token_id": tokens[position],
                         "g_mean": sum(bits) / len(bits), "included": not repeated})
    score = sum(layer_sums) / (used * len(layer_sums)) if used else None
    result = {"scope": "supplied_reference_profile_only", "claude_status": "unavailable",
              "status": "insufficient_contexts" if used < MIN_CONTEXTS else "uncalibrated",
              "profile_fingerprint": fingerprint(profile), "tokenizer_id": tokenizer_id,
              "input_tokens": original_length, "tokens_before_eos": len(tokens),
              "scored_contexts": used, "skipped_repeated_contexts": skipped,
              "minimum_contexts_policy": MIN_CONTEXTS, "mean_g_score": score,
              "per_layer_mean": [s / used for s in layer_sums] if used else [],
              "token_trace": rows, "trace_truncated": max(0, len(tokens) - n + 1) > 256,
              "empirical_tail_rank": None, "calibration_note": "No matching calibration supplied"}
    if calibration is not None:
        if not isinstance(calibration, dict):
            raise ValueError("calibration must be an object")
        scores = calibration.get("null_scores")
        if not isinstance(scores, list) or not 99 <= len(scores) <= 10000:
            raise ValueError("Calibration requires 99 to 10000 independent null scores")
        if any(type(s) not in (int, float) or not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError("Invalid calibration scores")
        compatible = (calibration.get("profile_fingerprint") == result["profile_fingerprint"]
                      and calibration.get("tokenizer_id") == tokenizer_id
                      and calibration.get("scored_contexts") == used
                      and calibration.get("input_tokens") == original_length)
        if compatible and used >= MIN_CONTEXTS:
            rank = (1 + sum(s >= score for s in scores)) / (1 + len(scores))
            result.update(empirical_tail_rank=rank, alpha=ALPHA,
                          status="reference_signal_detected" if rank <= ALPHA else "reference_signal_not_detected",
                          calibration_note="Valid only for the supplied null population, fixed profile and single test",
                          calibration_samples=len(scores), calibration_population=calibration.get("population", "unspecified"))
        else:
            result["calibration_note"] = "Calibration mismatch or too few scored contexts; no detection verdict"
    return result


def analyze_text(text):
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise ValueError("text must be a string of at most %s characters" % MAX_TEXT)
    counts, examples = Counter(), []
    for i, char in enumerate(text):
        code = ord(char)
        flagged = (unicodedata.category(char) == "Cf" or char in "\u00a0\u202f\u034f"
                   or 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF)
        if flagged:
            name = "U+%04X %s" % (code, unicodedata.name(char, "UNNAMED"))
            counts[name] += 1
            if len(examples) < 80:
                examples.append({"position": i, "codepoint": "U+%04X" % code,
                                 "name": unicodedata.name(char, "UNNAMED")})
    return {"claude_status": "unavailable", "status": "vendor_verification_unavailable",
            "reason": "Claude key, exact tokenization/profile and authorized detector access are not available",
            "characters": len(text), "words_approx": len(re.findall(r"\w+", text, re.UNICODE)),
            "unicode_findings": dict(counts), "unicode_examples": examples,
            "unicode_note": "Formatting characters can be legitimate; they do not establish AI origin",
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "modified": False, "external_requests": False}


def demo_profile(foreign=False):
    return {"algorithm": ALGORITHM,
            "keys": [17, 263, 1021, 4099, 8191, 16381, 32749, 65521] if not foreign else
                    [37, 271, 1031, 4111, 8209, 16411, 32771, 65537],
            "ngram_len": 5, "context_history_size": 1024,
            "tokenizer_id": "synthetic-uniform-1024-v1"}


def tournament_probabilities(probs, bits):
    p = list(probs)
    for layer in range(len(bits[0])):
        mass = sum(prob * g[layer] for prob, g in zip(p, bits))
        p = [prob * (1 + g[layer] - mass) for prob, g in zip(p, bits)]
    return p


def generate_demo(seed, marked, length=128, foreign=False):
    """Synthetic token sampler, NOT an LLM or Claude-generated text."""
    rng = random.Random(seed)
    profile = demo_profile(foreign)
    tokens = [rng.randrange(1024) for _ in range(4)]
    while len(tokens) < length:
        if not marked:
            tokens.append(rng.randrange(1024))
            continue
        candidates = rng.sample(range(1024), 16)
        gs = [ngram_g_values(tokens[-4:] + [token], profile) for token in candidates]
        probs = tournament_probabilities([1 / 16] * 16, gs)
        tokens.append(rng.choices(candidates, weights=probs, k=1)[0])
    return tokens[:length]


def inspect_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON object required")
    return inspect_tokens(payload.get("token_ids"), payload.get("profile"),
                          payload.get("tokenizer_id"), payload.get("eos_token_id"),
                          payload.get("calibration"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("text").add_argument("file", type=Path)
    sub.add_parser("tokens").add_argument("file", type=Path)
    args = parser.parse_args()
    if args.file.stat().st_size > 512_000:
        parser.error("Input exceeds 512 KB")
    try:
        raw = args.file.read_text(encoding="utf-8")
        if args.command == "text":
            result = analyze_text(raw)
        else:
            payload = json.loads(raw)
            if isinstance(payload, dict) and "payload" in payload:
                payload = payload["payload"]
            result = inspect_payload(payload)
    except (ValueError, UnicodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
