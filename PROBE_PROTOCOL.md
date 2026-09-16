# Controlled-query pilot before a keyless experiment

This is a **format and diversity pilot**, not a secret-key recovery algorithm or
a statistical verdict on Claude. The current release contains no usable live
target-model sample corpus and no successful key recovery.

The next research path is motivated by [Gloaguen et al., Black-Box Detection of
Language Model Watermarks](https://files.sri.inf.ethz.ch/website/papers/gloaguen2024detectingwatermarks.pdf).
Their Red-Green deployment test examines whether context-associated choice
preferences persist across different preceding prefixes. It requires repeated
sampling, modeling assumptions and context-size treatment. A successful
deployment test would not by itself disclose a key or classify arbitrary prose.

## Stage 1: preflight

- Freeze one provider/model and record its resolved model ID, client version,
  sampling controls that are available, and controls that are not exposed.
- Use fresh conversations with no tools, files, private text or inherited chat
  history. Retain only the authored prompts and returned answers, plus minimal
  model/time/usage metadata; never export login/session/account details.
- Run at most 16 calls, sequentially, with a 60-second per-call timeout. Stop on
  the first authentication, rate-limit, billing or unexpected model error. Do not
  auto-retry failures or silently switch models.
- An existing subscription can meter the requests against its allowance; a
  client's displayed list-price estimate is not proof of an invoice amount.
  Paid/API use needs an explicit budget before collection.

## Frozen pilot grid

Two prefixes: `I selected` and `I brought`.

Four suffix strings: `4 1 7 9`, `6 2 8 3`, `9 5 1 8`, `3 7 2 6`.

Two fresh-conversation repetitions of each pair; 16 total. Choice order is fixed:
`apples, pears, plums, grapes`. Context strings are **not claimed to be four
provider tokens**. Exact tokenizer/context compatibility remains unverified.

Authored prompt template:

```text
Return exactly the following prefix, followed by one freely chosen fruit from
apples, pears, plums, grapes. Include the entire prefix, add one space and the
fruit, then stop. Prefix: "{prefix} {suffix}".
```

For each response, check exact-prefix adherence and a single allowed final word.
Count all invalid/error outcomes; keep the stopping reason. The repeated digits
in older literature can interact with context caching, so this pilot instead
uses distinct-digit strings. This is a protocol adaptation, not an exact
reproduction of the paper.

## Decision after the pilot

Report the format-pass fraction, choice counts per cell, total valid samples,
model consistency and any output collapse. With two samples per cell, **do not
compute a watermark p-value or report a recovered key**.

Only after the pilot can a larger, separately frozen experiment be sized. It
needs matched controls, a context-size strategy, enough repeated observations,
and independent confirmation. Choosing prefixes, fruit sets, keys or thresholds
after looking at outcomes makes those data exploratory. Preserve that label and
use new data for a confirmatory test.

The source package's ordinary UI/CLI modes do not run this collection and do not
contact a model. [KEY_SEARCH.md](KEY_SEARCH.md) records the evidence behind the
current limitations.
