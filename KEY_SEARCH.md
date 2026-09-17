# Claude watermark key: search result and next experiment

**Checked: 2026-09-16. No verified production Claude watermark key or compatible public profile was found in the sources below. No Claude key was recovered by this project.** This is the result of a bounded public-source search, not proof that no such information exists anywhere.

## What the sources actually establish

**2026-09-17 follow-up:** [Russian research update](research/2026-09-17-search.ru.md)
and [independent recount of a public paired GPT-2 dataset](research/public-pair-audit-2026-09-17/README.md).
The recount reproduces 9/12 prompt-group rankings but only 47/96 correct isolated
file decisions at the fixed zero threshold. No live Claude answers or keys were
obtained. Fable 5.1 documentation now gives a model-specific watermark statement;
the newly reviewed TTP-Detect method requires paired controls/provider cooperation.

| Primary source | Finding | Consequence for this project |
| --- | --- | --- |
| [Anthropic: text watermark](https://www.anthropic.com/news/claude-text-watermark), updated 2026-09-01 | Describes a SynthID-Text variant and a detection API in private preview; the article does not disclose its production key/profile. | Keep Claude verification unavailable. An API access application is not access already granted. |
| [DeepMind reference implementation](https://github.com/google-deepmind/synthid-text/tree/addb4a158143c7c6851a1308f78b89fceed59683) | Publishes a configurable implementation; our snapshot derives an initialization value from the ordered key list. | Public algorithms and example keys do not establish a provider's deployed configuration. |
| [ETH Zürich: Probing SynthID-Text](https://www.sri.inf.ethz.ch/blog/probingsynthid), 2024-12-20 | Studies black-box detection of scheme deployment, and learned approximations for spoofing/scrubbing. Its SynthID experiments use a locally watermarked model. | A promising research method, not a recovered Claude secret. |
| [Black-Box Detection of Language Model Watermarks](https://www.sri.inf.ethz.ch/publications/gloaguen2024detectingwatermarks), ICLR 2025 | Tests for watermark deployment through repeated model queries. Its reported Claude 3 result concerns an older model and period. | Do not transfer that historical result to current Claude models, or to arbitrary-document authorship. |
| [natzir/synthid-text-watermark-attacks](https://github.com/natzir/synthid-text-watermark-attacks) | Explicitly uses the author's own keys and substitute local models. | Its experiments do not reveal the production Claude key. |

## Three different research goals

1. **Recover the exact key/profile.** Requires independently verifiable evidence that a candidate matches the actual deployment. No candidate meeting that standard was found.
2. **Detect whether a queried model deploys a watermark.** ETH's method investigates this without knowing the key. It needs controlled repeated sampling and correct treatment of context size/caching. It does not automatically classify a standalone passage.
3. **Learn an approximation of watermark preferences.** The watermark-stealing literature studies this with corpora of model outputs. Approximation, successful spoofing and exact secret-key recovery are different claims. No such corpus experiment on Claude was performed here.

## Why a blind integer loop is not a recovery result

In our pinned reference, every layer key also contributes to the SHA-256-derived initialization value. Guessing a single layer while holding an arbitrary initialization value fixed is not a valid test of a complete candidate profile. Claude's actual profile, exact tokenizer and deployment parameters are unknown. This project makes no claim that every possible attack is infeasible; it has no evidence that an unrestricted brute-force run would be useful.

Selecting the largest score from many trial keys creates selection bias. A candidate would need a frozen selection procedure, an independent validation corpus and correction for the number of attempted hypotheses. A high score on a hand-picked example is insufficient.

## Concrete next experiment

An offline [conditional-preference learning module](PREFERENCE_LEARNING.md) is now
implemented. It tests three synthetic scenarios at three training budgets, with
held-out prefixes, no learner access to generator keys, and an explicit ordinary
preference counterexample. It supplies no live Claude corpus or key-recovery
evidence. This demonstration is separate from the target-model pilot below.

Before model sampling, record a named model/version, date, endpoint, sampling parameters, query budget, stopping rule and matched controls. Use fresh synthetic prompts and retain raw outputs with their provenance. Reproduce the published deployment-detection test on a controlled model first, then evaluate the target under the same documented assumptions. Keep exact-key recovery and arbitrary-text detection as separate, unproven endpoints.

The initial public-source search did not call a vendor model, submit an access request, search private credentials, or spend money. The next step is specified in [PROBE_PROTOCOL.md](PROBE_PROTOCOL.md): a bounded format/diversity pilot before any statistical deployment test. No usable live target-model answers or recovered keys have been obtained in this project. Ordinary tool use stays offline.

Search queries and source URLs are preserved in [evidence/key-search-2026-09-16.json](evidence/key-search-2026-09-16.json). Our implemented arithmetic is documented in [ALGORITHM.md](ALGORITHM.md).
