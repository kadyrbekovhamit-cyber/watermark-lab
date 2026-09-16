# Watermark Lab

A small, offline research tool for scoring **known profiles of the open SynthID-Text scheme**.

**No production Claude key has been recovered. This project cannot verify whether arbitrary text contains Claude's watermark.** See the dated [key-search report](KEY_SEARCH.md) for the evidence and the next research step.

[Русская инструкция](README.ru.md) · [Algorithm](ALGORITHM.md) · [Research findings](KEY_SEARCH.md)

## Run locally

The application uses Python 3.9+ and the standard library. No model download, GPU or API key is needed.

```sh
git clone https://github.com/kadyrbekovhamit-cyber/watermark-lab.git
cd watermark-lab
python3 -B server.py --port 4548
```

Open [127.0.0.1:4548](http://127.0.0.1:4548). Stop the server with Ctrl+C. The UI is in Russian; JSON output and technical documentation are in English.

| Mode | What it does |
| --- | --- |
| Your text / Ваш текст | Counts text and flags special Unicode characters; Claude status remains unavailable. |
| Control experiment / Контрольный эксперимент | Rescores three synthetic fixtures: matching key, no watermark, and a different key. |
| Known profile / Известный профиль | Scores imported token IDs using an explicit compatible profile; requires matching calibration for a detection verdict. |

No style-based authorship percentage is generated. A Unicode finding is not a statistical watermark finding.

## Measured results

One fixed profile, eight public demonstration keys, 128 synthetic tokens per document, 124 scored contexts, threshold 0.01. Calibration used 199 separate unmarked samples. Every outcome in the 300-sample held-out evaluation is retained.

| Held-out group | Samples | Supplied-profile detections |
| --- | ---: | ---: |
| Matching-key watermark | 100 | 100 |
| Unmarked | 100 | **1 false positive** |
| Other-key watermark | 100 | 0 |

**These are synthetic results, not accuracy measurements on natural language or Claude.** A negative result with the wrong key does not establish that a text has no watermark.

Evidence: [all outcomes](evidence/benchmark.json), [calibration](evidence/demo-calibration.json), [matching-key fixture](evidence/demo-marked_correct_key.json).

The scorer matched **2,072 reference g-bits** across 259 n-grams, plus a masked-mean check. Validation runs the bodies of three pinned DeepMind arithmetic functions through a minimal NumPy adapter. It does not execute a full PyTorch model or validate every upstream feature.

## CLI and inputs

```sh
python3 -B watermark_lab.py text document.txt
python3 -B watermark_lab.py tokens evidence/demo-marked_correct_key.json
```

A token request is an object with `token_ids`, `tokenizer_id`, `profile`, and optional `calibration` and `eos_token_id`. The UI and CLI also accept the saved fixture wrapper containing `payload`.

```json
{
  "token_ids": [10, 32, 45, 78, 12],
  "tokenizer_id": "synthetic-uniform-1024-v1",
  "profile": {
    "algorithm": "deepmind-synthid-addb4a15-int64-le-v1",
    "keys": [17, 263, 1021, 4099, 8191, 16381, 32749, 65521],
    "ngram_len": 5,
    "context_history_size": 1024,
    "tokenizer_id": "synthetic-uniform-1024-v1"
  }
}
```

This short format example has too few contexts for a verdict. Use the full fixture above for a working calibrated example.

Token IDs must come from the generator's exact tokenizer. Splitting prose into words does not supply those IDs. Matching tokenizer names checks consistency of the declaration, not the provenance of the tokens. The supported algorithm is one pinned reference snapshot; other SynthID implementations may differ.

Calibration must match the profile fingerprint, tokenizer, input length and number of scored contexts. Its independent unmarked samples must represent the population being tested. The program cannot establish those sampling assumptions from JSON metadata. Demonstration calibration is only for its synthetic population. Multiple hypothesis searches need separate correction.

Limits: 4,096 tokens; 100,000 text characters; 512,000-byte files/HTTP bodies; at least 64 scored contexts and 99 calibration scores for a verdict. These are application policies, not provider requirements. EOS and subsequent tokens are excluded when its ID is supplied.

## Verification

NumPy is needed only for the reference comparison and experiment, not for running the UI or CLI.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-test.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 -B -m unittest -v test_watermark_lab.py
node --check web/app.js
```

The 25 tests cover arithmetic agreement, repeated contexts, EOS, validation, calibration, HTTP handling and preservation of the unavailable Claude verdict. Node is optional for the JavaScript syntax check.

Reproduce the bounded synthetic benchmark with `python3 -B validate.py`. It runs sequentially and rewrites its generated evidence files. There are no model or vendor calls.

## Privacy and interpretation

The single-process server listens on `127.0.0.1` only, rejects nonlocal origins/hosts, and does not log submissions. Browser requests go to the local server; there are no external fonts, telemetry or media. Source links open only when clicked.

Text and supplied keys are processed in memory. Downloading a report is an explicit user action. Token reports include up to 256 input token IDs; text reports include a SHA-256 digest and special-character positions. Exported reports are not automatically anonymous. Browser/OS memory handling is outside the application's control.

A mean g-score or empirical tail rank is not an authorship probability. All reports keep `claude_status: unavailable`.

## Sources and license

Anthropic describes Claude's watermark as a SynthID-Text variant and its detection API as private preview in its [official explanation](https://www.anthropic.com/news/claude-text-watermark), checked 2026-09-16. This repository provides no private API adapter or provider key.

The scalar arithmetic is adapted from [DeepMind SynthID-Text](https://github.com/google-deepmind/synthid-text/tree/addb4a158143c7c6851a1308f78b89fceed59683), pinned at `addb4a158143c7c6851a1308f78b89fceed59683`. Exact source files and SHA-256 hashes are in [reference/SOURCE_LOCK.json](reference/SOURCE_LOCK.json).

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Independent research project; no Anthropic or Google endorsement.
