# External reference experiment: independent recount

2026-09-17. This is a reproduction of one published **GPT-2** experiment, not a
Claude detector or key-recovery result. [Upstream MIT-licensed research](https://github.com/jensabrahamsson/text-watermark-laboratory/tree/989064945295141dbc85b8acbc3a9026f7827a83).

The pinned dataset contains 12 prompt families × 4 draws × 2 labels = 96 files.
The labels identify generation with a public SynthID instance on/off. The scorer
gets only token counts and labels. Each complete prompt family is held out when
fitting the remaining families. It never receives keys, hash initialization or
g-values. Generation and the official keyed detector were not rerun.

| Fixed evaluation | Reproduced outcome |
| --- | ---: |
| Marked group has higher mean score than its unmarked twin group | 9 / 12 families |
| Individual marked file has score > 0 | 25 / 48 |
| Individual unmarked file has score <= 0 | 22 / 48 |
| Correct individual decisions overall | **47 / 96 (48.96%)** |

These are different endpoints. Group ranking is not isolated-document accuracy.
The simple fixed-zero classifier gives 26 false positives and 23 false negatives.
No new significance claim is made: training folds overlap, and the expected
upstream results were known before reproduction. Results do not describe all
key-free methods or production Claude. No model weights were downloaded,
and no model API was called.

Our independent count implementation also agrees with 96 outputs of the reviewed
upstream pure functions: maximum difference 3.06e-16. We execute only an explicit
AST allowlist of those arithmetic/container definitions, not the upstream package,
model loaders or command-line application. Source and data hashes are checked.

Published text re-tokenizes to 126–128 tokens, although primary generation metadata
says 128. Every saved string round-trips exactly. We preserve and report every
primary-draw discrepancy; original generation IDs cannot be inferred from the
strings. No samples were discarded to match the metadata.

## Reproduce

Python 3.10+; the audit was run on Python 3.12 with `tokenizers==0.23.2`.
The application itself retains its original requirements. From the repository root:

```sh
python3 research/public-pair-audit-2026-09-17/fetch.py
python3 -m pip install --only-binary=:all: --no-deps --no-cache-dir \
  --target private/public-pair-audit-2026-09-17/deps tokenizers==0.23.2
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B research/public-pair-audit-2026-09-17/reproduce.py
```

`fetch.py` explicitly downloads about 1.45 MB of pinned source, data and a tokenizer
specification. The package installation is separate. Neither command downloads a
language model. `reproduce.py` then runs offline and saves `reproduction-local.json`
in the Git-ignored workspace. An existing report at that path is replaced.
Both scripts accept `--workspace PATH` to use another directory.

Recorded evidence: [all 96 scores and folds](reproduction.json),
[protocol and data-check deviation](PROTOCOL.md), [source lock](source-lock.json),
[tokenizer lock](tokenizer-lock.json), [upstream license](LICENSE.upstream).
Private user attachments are not part of this corpus or publication.
