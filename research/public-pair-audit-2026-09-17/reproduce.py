"""Independent recount of a pinned public experiment; no keys or model calls.

Method and dataset: Jens Abrahamsson's MIT-licensed text-watermark-laboratory.
The implementation below is a direct count/likelihood calculation; source
function parity is an additional check, not the experiment's decision input.
"""
import ast
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import types

DOCUMENTS = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', type=Path, default=Path('private/public-pair-audit-2026-09-17'))
ROOT = parser.parse_args().workspace.resolve()
sys.path.insert(0, str(ROOT / 'deps'))
from tokenizers import Tokenizer

K, ALPHA = 4, 0.5

def fit(sequences):
    buckets, single = defaultdict(Counter), Counter()
    for seq in sequences:
        single.update(seq)
        for i, token in enumerate(seq):
            for width in range(1, min(K, i) + 1):
                buckets[tuple(seq[i-width:i])][token] += 1
    return buckets, single

def score(seq, marked, unmarked, vocabulary):
    contributions = []
    for i in range(1, len(seq)):
        ctx = tuple(seq[max(0, i-K):i])
        probs = []
        for buckets, single in (marked, unmarked):
            bucket = buckets.get(ctx, single)
            probs.append(math.log((bucket[seq[i]] + ALPHA) /
                                  (sum(bucket.values()) + ALPHA * vocabulary)))
        contributions.append(probs[0] - probs[1])
    return sum(contributions) / len(contributions) if contributions else 0.0

def reviewed_reference():
    # Only the reviewed pure arithmetic/data-container definitions. No upstream
    # imports, tokenizer/model loading, file access or CLI code are executed.
    from dataclasses import dataclass, field
    import typing
    path = ROOT / 'upstream/src/text_watermark_tools/blind.py'
    source = path.read_text()
    names = {'NextTokenTable', 'BlindModel', '_ctx', '_scored_ctx', '_add_sequence',
             'fit_table', 'fit_blind', '_bucket_log_prob', 'score_index_range',
             'backoff_contexts', '_log_prob', 'likelihood_ratio'}
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    module = types.ModuleType('reviewed_count_reference')
    sys.modules[module.__name__] = module
    module.__dict__.update({'dataclass': dataclass, 'field': field, 'Counter': Counter,
                           'math': math, 'Sequence': typing.Sequence, 'Iterable': typing.Iterable,
                           'DEFAULT_CONTEXT_LEN': 1, 'DEFAULT_ALPHA': 0.5, 'FIRST_TOKEN_CTX': (-1,)})
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()

def main():
    lock = json.loads((DOCUMENTS / 'source-lock.json').read_text())
    for item in lock['records']:
        raw = (ROOT / 'upstream' / item['path']).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
    token_lock = json.loads((ROOT / 'tokenizer-lock.json').read_text())
    data = (ROOT / 'gpt2-tokenizer.json').read_bytes()
    assert hashlib.sha256(data).hexdigest() == token_lock['sha256']
    tokenizer = Tokenizer.from_str(data.decode())
    pairs = ROOT / 'upstream/experiments/2026-08-17-pair-12x4'
    metadata = json.loads((pairs / 'results.json').read_text())
    assert metadata['model_name'] == 'gpt2' and metadata['instance'] == 'public-deepmind-30'
    families, token_counts, primary_count_checks = {}, [], []
    for row in metadata['rows']:
        stem = row['stem']
        family = {}
        for group, ending, key in [('marked', 'marked', 'marked'), ('unmarked', 'unmarked-gen', 'unmarked_gen')]:
            seqs = []
            for draw in range(1, 5):
                name = stem + '-' + ending + ('' if draw == 1 else '-'+str(draw)) + '.txt'
                text = (pairs / name).read_text()
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                assert tokenizer.decode(ids, skip_special_tokens=False) == text
                if draw == 1:
                    primary_count_checks.append({'file': name, 'generation_metadata_tokens': row[key]['n_tokens'],
                                                 'retokenized_text_tokens': len(ids),
                                                 'match': len(ids) == row[key]['n_tokens']})
                assert 16 < len(ids) < 256, name
                token_counts.append(len(ids))
                seqs.append(ids)
            family[group] = seqs
        families[stem] = family
    assert len(families) == 12 and len(token_counts) == 96
    reference, ref_sha = reviewed_reference()
    scores, folds, differences = [], [], []
    for held, family in families.items():
        train = {g: [seq for stem, item in families.items() if stem != held for seq in item[g]]
                 for g in ['marked', 'unmarked']}
        marked, unmarked = fit(train['marked']), fit(train['unmarked'])
        vocabulary = max(len(set(marked[1]) | set(unmarked[1])), 2)
        ref = reference.fit_blind(train['marked'], train['unmarked'], context_len=K, alpha=ALPHA, backoff=False)
        assert not ref.used_keys and not ref.used_hash_iv and not ref.used_g_values
        group_scores = {}
        for group in ['marked', 'unmarked']:
            group_scores[group] = []
            for draw, seq in enumerate(family[group], 1):
                value = score(seq, marked, unmarked, vocabulary)
                expected = reference.likelihood_ratio(seq, ref)
                differences.append(abs(value - expected))
                group_scores[group].append(value)
                scores.append({'prompt_family': held, 'group': group, 'draw': draw,
                               'tokens': len(seq), 'log_ratio': value})
        means = {g: sum(v)/len(v) for g, v in group_scores.items()}
        folds.append({'prompt_family': held, 'marked_mean': means['marked'],
                      'unmarked_mean': means['unmarked'], 'marked_group_ranks_above': means['marked'] > means['unmarked']})
    assert max(differences) < 1e-12
    tp = sum(r['group'] == 'marked' and r['log_ratio'] > 0 for r in scores)
    tn = sum(r['group'] == 'unmarked' and r['log_ratio'] <= 0 for r in scores)
    report = {'scope': 'reproduction_of_one_public_GPT2_reference_instance',
              'claude_status': 'unavailable', 'key_recovered': False,
              'live_model_calls': 0, 'model_weights_downloaded': False,
              'source': json.loads((ROOT / 'source.json').read_text()),
              'tokenizer': token_lock, 'prompt_families': 12, 'files': 96,
              'token_count_range': [min(token_counts), max(token_counts)],
              'generation_count_checks': primary_count_checks,
              'tokenization_note': 'Published UTF-8 strings round-trip exactly; re-tokenized lengths need not equal original generation token counts. No token-ID provenance is inferred.',
              'method': {'history': K, 'alpha': ALPHA, 'backoff': 'unigram_when_exact_context_absent',
                         'split': 'leave_entire_prompt_family_out', 'isolated_threshold': 0,
                         'first_token_excluded': True, 'source_results_known_before_reproduction': True},
              'reference_parity': {'sha256': ref_sha, 'scores_compared': len(scores),
                                   'max_absolute_difference': max(differences),
                                   'scope': 'reviewed pure definitions only; full upstream package not run'},
              'results': {'prompt_group_wins': sum(f['marked_group_ranks_above'] for f in folds),
                          'true_positive': tp, 'false_negative': 48-tp, 'true_negative': tn,
                          'false_positive': 48-tn, 'balanced_accuracy': (tp+tn)/96},
              'limitations': ['Not production Claude or secret-key recovery',
                             'Descriptive reproduction; overlapping LOO fits and no new significance test',
                             'Published text is re-tokenized; generation itself was not rerun',
                             'Good prompt-group ranking does not imply reliable isolated-document classification'],
              'folds': folds, 'file_scores': scores}
    report['source']['executable_code_run'] = 'reviewed pure definitions for parity only'
    (ROOT / 'reproduction-local.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'results': report['results'], 'parity': report['reference_parity'], 'token_counts': report['token_count_range']}, indent=2))

if __name__ == '__main__':
    main()
