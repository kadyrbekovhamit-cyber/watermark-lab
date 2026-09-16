# Learning preferences from thousands of observations

[По-русски](PREFERENCE_LEARNING.ru.md)

**Result, 2026-09-16:** a learner without generator keys recovered some conditional
choice preferences in a small synthetic experiment. It did not recover a secret
key. No natural-language answers or live Claude calls were used.

This addresses a narrower question than finding a provider's secret: can repeated
observations reveal which continuations are preferred after a particular context?
The learner sees target/control counts and context identifiers. The generator's
keys and intermediate g-values never enter the learner or evaluator.

## Design and results

The generator uses the project's pinned DeepMind arithmetic for one scenario.
It samples one of four artificial token IDs after each of 32 four-token contexts.
These are fresh single-choice trials, not full LLM responses. Each response has
only one context occurrence; within-response repetition masking does not arise.

Two simulated prefixes are used for training and two different prefixes for
validation. Prefixes have different baseline choice probabilities. The context
identifiers are deliberately shared across training and validation. A smoothed
log target/control frequency ratio is learned for each context and choice.

Training budgets include **both groups**: 1,024, 4,096 and 16,384 observations
(at the maximum, 8,192 target and 8,192 control). Validation uses 32,768 separate
observations per scenario. The same held-out grid is reused for the descriptive
learning curve. All scenarios, budgets and fixed seeds are retained.

Balanced accuracy below means classifying **one synthetic choice** as target or
control, on previously seen contexts. It is not accuracy on articles or Claude.

| Training observations, both groups | Reference watermark | Identical distributions, no mark | Ordinary context preference, no mark |
| ---: | ---: | ---: | ---: |
| 1,024 | 68.45% | 50.33% | 62.09% |
| 4,096 | 70.88% | 49.81% | 65.74% |
| 16,384 | 71.20% | 49.87% | 66.85% |

In the last column, a context-specific choice receives five times its baseline
weight. There is **no watermark key** behind that preference. This counterexample
shows why learning persistent preferences is not enough to establish watermarking.
With unknown context identifiers, the lookup learner abstains: coverage is zero.
That is a coverage-policy check, not an accuracy estimate for new contexts.

At the preselected maximum budget, one-sided paired group-label swaps give
p=0.000244 for the reference scenario, p=0.718018 for the null and p=0.000244 for
the ordinary-preference scenario. These compare distributions under the stated
null, not watermark provenance. There are 4,095 Monte Carlo swaps; 0.000244 is
their minimum reported p-value. No broader false-positive rate is estimated by
this one fixed null run. Earlier budgets are descriptive, with no reported p-value.

The test requires a frozen learner, independent validation cells, and target and
control groups exchangeable within each cell under the null. It swaps complete
groups in each context/prefix cell and equally weights the cell mean differences.
JSON counts cannot establish those assumptions. Subsequent searches over prompts,
contexts, thresholds or keys need separate validation and multiplicity control.

## Run offline

Python 3.9+, standard library only. This CLI module is separate from the browser
interface. Its commands neither collect model answers nor contact external APIs.

```sh
python3 -B preference_probe.py benchmark \
  --output private/preference-benchmark.json \
  --fixture private/preference-corpus-synthetic.json

python3 -B preference_probe.py analyze \
  private/preference-corpus-synthetic.json \
  --output private/preference-result.json

python3 -B -m unittest -v test_preference_probe.py
```

Outputs are written to the explicitly supplied paths. Existing output files may
be replaced; report and input paths must differ. `private/` is Git-ignored.
Published evidence contains only generated data:
[complete results](evidence/preference-benchmark.json) and
[maximum-budget reference corpus](evidence/preference-corpus-synthetic.json).

## Input format

The runnable fixture shows the full `conditional-choice-corpus-v1` format:

- `choices`: 2–32 distinct string identifiers, with stable meaning and order.
- `provenance.kind`: `synthetic`, `controlled` or `unknown`; a declaration, not verification.
- `training` and `validation`: arrays of objects with `context`, `prefix`,
  `group` (`target` or `control`) and nonnegative integer `counts` in choice order.
- Every context/prefix cell must contain both groups with equal sample sizes.
- At least two prefix identifiers per split; no prefix overlap between splits.
  Repeating known contexts across prefixes is intentional. Raw examples must still
  be independently collected and kept out of the opposite split.
- Limits: 5 MB input, 4,096 rows per split, 4–1,000,000 observations per row.

The report exports counts, aggregate metrics and an input digest, not prompts,
keys or context identifiers. This is not a guarantee of anonymization. Keep any
private source corpus outside the publication; only the synthetic fixture belongs
in this repository.

## What a real corpus would need

A folder of unrelated Claude answers cannot directly supply this experiment's
matched repetitions and controls. Record model/version, generation date, endpoint,
sampling settings, context/tokenization assumptions, failures and selection rules.
Another model, human text or a different deployment is not automatically a valid
unwatermarked control. A persistent ordinary model preference can also be learned.

The bounded live pilot remains in [PROBE_PROTOCOL.md](PROBE_PROTOCOL.md). It checks
format/diversity only. There is no usable live target corpus in this project and
no budgeted large collection. Exact-key recovery would need independently
verifiable agreement with the actual provider deployment. The output therefore
always says `claude_status: unavailable` and `key_recovered: false`.

## Research context

[Watermark Stealing in Large Language Models, ICML 2024](https://proceedings.mlr.press/v235/jovanovic24a.html)
studies learning approximate watermark preferences from outputs. Approximation
and exact secret-key recovery are distinct outcomes.
[ETH's SynthID investigation](https://www.sri.inf.ethz.ch/blog/probingsynthid)
uses a locally watermarked model; it does not recover Claude's production key.
This small finite-choice demonstration is **not a reproduction** of either study.
The broader source review is in [KEY_SEARCH.md](KEY_SEARCH.md).
