# Reproduction protocol, fixed before local scoring

This is a reproduction of an already-published result, not a blind discovery.
The upstream README already reports 9/12 prompt rankings, 25/48 marked-positive
files and 22/48 unmarked-nonpositive files for the corrected hard last-4 method.

Use all 96 generated files in the pinned 12-prompt × 4-draw paired GPT-2 dataset.
Hold out the complete prompt family (both labels and all draws). Re-tokenize with
the GPT-2 tokenizer; check counts against the published primary-draw metadata.
Fit target/control conditional counts for suffix lengths 1 through min(4, i),
storing each real suffix once. Smoothing alpha=0.5. At scoring, select hard
last-4, fall back to the corresponding unigram distribution when unseen, and
exclude token zero. No prompt tokens, watermark keys, hash initialization or
g-values enter training or scoring. Do not tune the threshold: isolated sign >0
for marked, <=0 for unmarked. Compare mean LRs for the two groups per prompt.

Report every file score, confusion counts and group ranks. Outcomes are
descriptive; overlapping LOO training folds are not independent experiments.
No new significance claim or universal accuracy claim. These are downloaded
public generated GPT-2 samples; they are not live Claude samples. The supplied
private attachments are outside this experiment.

Pre-scoring data check: the first marked UTF-8 file re-tokenizes to 126 tokens,
while generation metadata says 128. The complete string round-trips without
change. Original generation token counts are therefore recorded as a discrepancy,
not forced onto re-tokenized strings. All files stay in the experiment; report
all 24 primary-draw count comparisons. This matches the upstream key-free path's
re-tokenization of saved strings; it does not reproduce original generation IDs.
