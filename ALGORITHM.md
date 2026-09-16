# Supported reference algorithm

[Русский разбор](ALGORITHM.ru.md)

This describes the public DeepMind snapshot at commit `addb4a158143c7c6851a1308f78b89fceed59683`, not a recovered production Claude configuration. See [the source lock](reference/SOURCE_LOCK.json) and [key-search findings](KEY_SEARCH.md).

## Context-dependent signal

For each possible next token, a generator derives binary g-values from a key and a short token context. A sequence of probability updates biases sampling toward some of those values. A detector with a matching profile recomputes the bits for the observed sequence and measures the resulting signal.

A word has no permanent watermark label: its bits depend on its token ID, context and profile. Invisible characters are a separate formatting issue.

## Exact supported arithmetic

The profile contains the n-gram length `n`, an ordered list of `d` signed-int64 keys, a context-history size and the tokenizer ID. This adapter fixes little-endian key serialization; all mixing explicitly wraps signed 64-bit integers.

1. Initialize `IV = int(SHA256(int64_le(keys)), big_endian) mod (2^63 - 1)`.
2. Starting from IV, accumulate the n-gram token IDs with `h = int64((h + token) * 6364136223846793005 + 1)`.
3. For each layer, accumulate its key as one additional value.
4. Repeat 12 times: `h = int64((h + 1) * 6364136223846793005 + 1) >> 5`, using an arithmetic right shift.
5. Return `g = (h >> 30) mod 2`.

This is a scalar adaptation of the pinned public implementation. Older integrations using a sampling table can produce different values. The name SynthID alone does not guarantee compatibility.

## Probability update in the synthetic generator

For one layer, candidate probabilities `p_i` and binary scores `g_i`:

```text
m = sum_i p_i * g_i
p'_i = p_i * (1 + g_i - m)
```

Repeat by layer. The update preserves a normalized nonnegative distribution. With only one possible token there is no probability mass to redistribute.

Our small synthetic sampler draws 16 distinct candidate IDs from a vocabulary of 1,024 at each step, applies eight updates, and samples from the result. Its first four tokens are random context. It is not a language model. These fixed examples have no repeated four-token contexts; the generator does not implement the general repeat-skipping mechanism. The detector does implement bounded repeated-context masking.

## Scoring and masking

The first `n - 1` tokens cannot be scored. For each later token, hash the preceding `n - 1` IDs. Exclude a token's score when that context hash is already in the bounded history. Initialization with zero-filled history and history updates follow the selected reference. If supplied, the first EOS and everything after it are excluded.

```text
S = sum_t sum_l mask_t * g[t,l] / (d * sum_t mask_t)
```

`S` is the mean g-score, not an authorship probability. Removing repeated contexts does not make overlapping windows independent.

## Calibration and decisions

The application compares S against an independent unmarked reference population:

```text
r = (1 + number of null scores >= S) / (M + 1)
supplied-profile signal detected if r <= 0.01
```

The +1 term prevents a zero empirical rank; ties are conservative. Interpretation requires a fixed profile, a single test and exchangeability of the test sample with the null calibration population under the null. The software cannot verify those assumptions from matching metadata.

Searching many keys, fragments or thresholds needs separate multiplicity control and independent validation. It is not implemented as a secret-key recovery method. The minimum of 64 scored contexts and 99 calibration values is application policy. Without matching calibration there is no detection verdict.

## What validation covers

The pinned `accumulate_hash`, `get_gvals` and `mean_score` function bodies are executed using a minimal NumPy adapter. There are 259 n-gram cases, including signed-overflow edges, and 2,072 exact g-bit matches. This tests arithmetic and a masked mean, not a complete upstream PyTorch runtime or production model.

The fixed synthetic benchmark uses 199 calibration documents and 100 held-out samples in each of three groups. All outcomes are published in [evidence/benchmark.json](evidence/benchmark.json); [README.md](README.md) contains the observed false positive and the other group results.

Neither reference agreement nor a synthetic benchmark establishes detection of Claude, Gemini or ChatGPT. This project has no production provider key/profile.

## References

- [DeepMind source at the pinned commit](https://github.com/google-deepmind/synthid-text/tree/addb4a158143c7c6851a1308f78b89fceed59683).
- [SynthID-Text paper, Nature 2024](https://www.nature.com/articles/s41586-024-08025-4) and [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11499265/).
- [Claude's description](https://www.anthropic.com/news/claude-text-watermark); production-profile status is tracked separately in [KEY_SEARCH.md](KEY_SEARCH.md).
