#!/usr/bin/env python3
"""Independently audit the frozen three-seed geometry/retrieval 2x2 experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'docs/tools'))
# Independent query-at-a-time, mean-per-signer reference, not production reranking.
from analyze_retrieval_pilot import reference_scores, order, metrics, paired
from run_rgb_pilot import inside, require, sha, verify_run

PROTOCOL = 'poet-factorial-retrieval-v1'
SEEDS = (20260922, 20260923, 20260924)
MODES = ('original', 'pure_memory', 'fixed_blend')
CELLS = ('B', 'B_plus_R', 'G', 'G_plus_R')
COMPARISONS = {
    'B_plus_R_vs_B': ('B', 'B_plus_R'),
    'G_vs_B': ('B', 'G'),
    'G_plus_R_vs_G': ('G', 'G_plus_R'),
    'G_plus_R_vs_B_plus_R': ('B_plus_R', 'G_plus_R'),
    'G_plus_R_vs_B': ('B', 'G_plus_R'),
}
IDENTITY = ('index', 'gloss', 'signer_id', 'split', 'source_path', 'video_sha256')
SCORES = {'memory_scores': 'memory_kernel_scores', 'base_probabilities': 'base_probability',
          'memory_probabilities': 'memory_probability', 'blend_probabilities': 'blend_probability'}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def assert_close(got, expected, message):
    a, b = np.asarray(got), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, rtol=1e-12, atol=1e-14), message)


def check_prereg(prereg):
    require(prereg.get('protocol') == PROTOCOL and prereg.get('registered_before_factorial_scores') is True
            and prereg.get('test_used') is False and prereg.get('optimizer_steps') == 0,
            'Wrong preregistration protocol/scope')
    require(prereg.get('seeds') == list(SEEDS) and prereg.get('primary_method') == 'geometry_fixed_blend',
            'Seeds or primary comparison differ')
    recipe = prereg['retrieval']
    require(recipe['beta'] == 10 and recipe['blend_memory_weight'] == .25
            and recipe['blend_base_weight'] == .75 and recipe['candidate_count'] == 5,
            'Frozen retrieval recipe differs')
    gate = prereg['retention_gate']
    require(gate['comparisons'] == ['G_plus_R_vs_G', 'G_plus_R_vs_B_plus_R']
            and gate['metrics'] == ['top1', 'macro_top1'] and gate['all_seeds_strictly_positive'] is True
            and all(gate[k] is True for k in ('zero_failed_required', 'historical_predictions_exact_required',
                                             'same_top5_within_model_required', 'independent_verification_required'))
            and gate['automatic_further_experiments'] is False, 'Retention criterion changed')
    models = prereg['models']
    require(len(models) == 6 and len({m['model_id'] for m in models}) == 6, 'Need six unique frozen models')
    require({(m['method'], m['seed']) for m in models} == {(m, s) for m in ('baseline', 'geometry') for s in SEEDS},
            'Missing, duplicate, or selected method/seed')
    for m in models:
        require(m['model_id'] == f"{m['method']}_seed{m['seed']}", 'Model identity differs')
    data = prereg['data']
    require(data['train_count'] == 40154 and data['val_count'] == 10304 and data['classes'] == 2731,
            'Wrong full-split population')
    return models


def full_metrics(targets, prediction):
    """Bincount reference with class and signer-subpopulation support."""
    result = metrics(targets, prediction)
    counts = np.bincount(targets, minlength=2731)
    hit1 = prediction[:, 0] == targets
    hit5 = (prediction == targets[:, None]).any(axis=1)
    n1 = np.bincount(targets, weights=hit1, minlength=2731)
    n5 = np.bincount(targets, weights=hit5, minlength=2731)
    present = np.flatnonzero(counts)
    result.update(candidate_classes=2731, observed_classes=len(present), per_class={
        str(c): {'samples': int(counts[c]), 'correct_top1': int(n1[c]), 'correct_top5': int(n5[c]),
                 'top1': float(n1[c] / counts[c]), 'top5': float(n5[c] / counts[c])} for c in present})
    return result


def check_metrics(stored, expected):
    require(set(stored) == set(expected), 'Metric field inventory differs')
    for key, value in expected.items():
        if key == 'per_class':
            require(set(stored[key]) == set(value), 'Metric class inventory differs')
            for c, entry in value.items():
                require(set(stored[key][c]) == set(entry), 'Per-class field inventory differs')
                for k, v in entry.items():
                    assert_close(stored[key][c][k], v, 'Per-class metric differs')
        else:
            assert_close(stored[key], value, 'Aggregate metric differs: ' + key)


def detailed_pair(records, targets, original, candidate):
    b, c = original[:, 0] == targets, candidate[:, 0] == targets
    details = {}
    for kind, mask in (('fixed', ~b & c), ('regressed', b & ~c),
                       ('changed_top1', original[:, 0] != candidate[:, 0])):
        details[kind] = [{'row': int(i), 'index': records[i]['index'], 'identity': records[i]['identity'],
                          'video_sha256': records[i]['video_sha256'], 'signer_id': records[i]['signer_id'],
                          'target': int(targets[i]), 'original': int(original[i, 0]),
                          'candidate': int(candidate[i, 0])} for i in np.flatnonzero(mask)]
    net = len(details['fixed']) - len(details['regressed'])
    return {'samples': len(targets), 'fixed_count': len(details['fixed']), 'regressed_count': len(details['regressed']),
            'net_correct': net, 'changed_top1_count': len(details['changed_top1']), **details,
            'delta_top1_percentage_points': 100. * net / len(targets)}


def check_pair(stored, expected, *, macro=False):
    for key, value in expected.items():
        if isinstance(value, list):
            require(stored[key] == value, 'Paired example identities differ: ' + key)
        else:
            assert_close(stored[key], value, 'Paired statistic differs: ' + key)


def verify_history(records, arrays, historical):
    require(len(records) == len(historical), 'Historical reference population differs')
    for i, (row, old) in enumerate(zip(records, historical)):
        require(old.get('status') == 'ok', 'Historical prediction failed')
        require(all(row[k] == old[k] for k in IDENTITY), 'Historical identity/order differs')
        require(old['logits_sha256'] == row['logits_sha256'], 'Historical full-logit SHA differs')
        require([x['class_id'] for x in old['top5']] == arrays['top5_class_ids'][i].tolist(),
                'Historical Top5 ranks differ')
        require([x['probability'] for x in old['top5']] == row['top5_probabilities'],
                'Historical Top5 probabilities differ')


def verify_feature_arrays(data, count):
    require(set(data) == {'embeddings', 'top5_class_ids', 'top5_logits'}, 'Unexpected feature arrays')
    for key, shape, dtype in (('embeddings', (count, 128), np.float32),
                              ('top5_class_ids', (count, 5), np.int64), ('top5_logits', (count, 5), np.float32)):
        require(data[key].shape == shape and data[key].dtype == dtype and np.isfinite(data[key]).all(),
                'Wrong feature shape/dtype/finite contract: ' + key)
    ids = data['top5_class_ids']
    require(bool(((ids >= 0) & (ids < 2731)).all())
            and bool((np.diff(np.sort(ids, axis=1), axis=1) > 0).all()), 'Wrong/duplicate candidate class')
    require(bool((np.diff(data['top5_logits'], axis=1) <= 0).all()), 'Original logits not ranked')


def load_feature_container(run, split, prereg, prereg_sha):
    directory = inside(run) / 'features'
    manifest = read(directory / 'manifest.json')
    require(manifest.get('protocol') == PROTOCOL and manifest.get('status') == 'completed'
            and manifest.get('scope') == 'formal' and manifest.get('split') == split
            and manifest.get('prereg_sha256') == prereg_sha, 'Wrong feature container scope/state')
    require(set(manifest['models']) == {m['model_id'] for m in prereg['models']}, 'Feature model inventory differs')
    return directory, manifest


def load_model_features(directory, container, split, model, prereg, prereg_sha, tracked):
    entry = container['models'][model['model_id']]
    path = inside(entry['manifest_path'], directory, relative=True)
    require(sha(path) == entry['manifest_sha256'], 'Per-model manifest SHA differs')
    manifest = read(path)
    require(manifest.get('status') == 'completed' and manifest.get('scope') == 'formal'
            and manifest.get('protocol') == PROTOCOL and manifest.get('failed') == 0
            and manifest.get('test_used') is False and manifest.get('optimizer_steps') == 0
            and all(manifest.get(k) is True for k in ('weights_unchanged', 'gradients_absent', 'model_eval')),
            'Wrong per-model feature state')
    require(manifest.get('prereg_sha256') == prereg_sha and manifest['feature_dim'] == 128
            and manifest['class_ids'] == list(range(2731)), 'Feature vocabulary/protocol differs')
    for key in ('model_id', 'method', 'seed', 'checkpoint_sha256', 'model_state_sha256'):
        require(manifest[key] == model[key], 'Feature model identity differs: ' + key)
    require(manifest['checkpoint_epoch'] == model['checkpoint_epoch'], 'Checkpoint epoch differs')
    expected_geometry = prereg['data'][split + '_geometry_sha256'] if model['method'] == 'geometry' else None
    require(manifest['geometry_manifest_sha256'] == expected_geometry, 'Geometry manifest binding differs')
    require(manifest['training_protocol_sha256'] == model['protocol_sha256'], 'Training protocol differs')
    require(manifest['weights_state_sha256_before'] == manifest['weights_state_sha256_after']
            == model['model_state_sha256'], 'Model state changed')
    for name, expected in manifest['source_sha256'].items():
        require(sha(inside(name)) == expected, 'Extraction frozen source changed: ' + name)
        tracked[name] = expected
    require(set(manifest['splits']) == {split}, 'Unexpected feature split')
    part = manifest['splits'][split]
    require(part['samples'] == prereg['data'][split + '_count'] and part['failed'] == 0
            and part['complete_official_split'] is True, 'Wrong feature population')
    metadata = inside(part['metadata_path'], path.parent, relative=True)
    featurefile = inside(part['features_path'], path.parent, relative=True)
    require(sha(metadata) == part['metadata_sha256'] and sha(featurefile) == part['features_sha256'],
            'Feature artifact SHA differs')
    records = rows(metadata)
    with np.load(featurefile, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    verify_feature_arrays(data, part['samples'])
    cache_path = inside(prereg['data'][split + '_cache_manifest'])
    require(sha(cache_path) == prereg['data'][split + '_cache_sha256'], 'Cache index changed')
    source = read(cache_path)
    require(len(records) == len(source['samples']) == part['samples'], 'Missing/extra feature rows')
    for i, (row, item) in enumerate(zip(records, source['samples'])):
        require(row['row'] == i and row['identity'] == item['identity']
                and all(row[k] == item[k] for k in IDENTITY), 'Feature identity/order differs')
        require(manifest['label_map'][str(row['class_id'])] == row['gloss'], 'Feature label map mismatch')
        require(row['features_sha256'] == item['features_sha256'] and row['pose_sha256'] == item['pose_sha256'],
                'Source pose payload differs')
    require(len({r['signer_id'] for r in records}) == (35 if split == 'train' else 6)
            and len({r['video_sha256'] for r in records}) == len(records)
            and {r['class_id'] for r in records} == set(range(2731)), 'Feature signer/video/class population differs')
    return records, data, manifest


def shared_rows_fingerprint(records):
    fields = IDENTITY + ('row', 'identity', 'class_id', 'pose_sha256', 'features_sha256', 'input_sha256',
                         'source_frame_indices_sha256', 'source_frame_indices', 'cache_manifest_sha256', 'source_manifest_sha256')
    text = json.dumps([{k: r[k] for k in fields} for r in records], sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def derive_factorial(records, targets, models, predictions):
    """All seeds retained; interaction is difference of retrieval effects."""
    by_seed = {}
    for seed in SEEDS:
        b = next(m['model_id'] for m in models if m['seed'] == seed and m['method'] == 'baseline')
        g = next(m['model_id'] for m in models if m['seed'] == seed and m['method'] == 'geometry')
        ranks = {'B': predictions[b]['original'], 'B_plus_R': predictions[b]['fixed_blend'],
                 'G': predictions[g]['original'], 'G_plus_R': predictions[g]['fixed_blend']}
        cells = {key: full_metrics(targets, value) for key, value in ranks.items()}
        comparisons = {}
        for name, (before, after) in COMPARISONS.items():
            comparisons[name] = {**detailed_pair(records, targets, ranks[before], ranks[after]),
                'delta_macro_top1_percentage_points': 100. * (cells[after]['macro_top1'] - cells[before]['macro_top1'])}
        interaction_net = cells['G_plus_R']['correct_top1'] - cells['G']['correct_top1'] - cells['B_plus_R']['correct_top1'] + cells['B']['correct_top1']
        interaction = {
            'net_correct': interaction_net,
            'top1_percentage_points': 100. * interaction_net / len(targets),
            'macro_top1_percentage_points': 100. * ((cells['G_plus_R']['macro_top1'] - cells['G']['macro_top1'])
                                                   - (cells['B_plus_R']['macro_top1'] - cells['B']['macro_top1']))}
        retain = all(comparisons[key]['net_correct'] > 0
                     and comparisons[key]['delta_macro_top1_percentage_points'] > 0
                     for key in ('G_plus_R_vs_G', 'G_plus_R_vs_B_plus_R'))
        by_seed[str(seed)] = {'cells': cells, 'comparisons': comparisons, 'interaction': interaction,
                             'combination_strictly_better_than_both': bool(retain)}
    aggregate = {'cells': {}, 'comparisons': {}, 'interaction': {}}
    for section, names, keys in (
            ('cells', CELLS, ('top1', 'macro_top1', 'top5', 'macro_top5', 'correct_top1')),
            ('comparisons', tuple(COMPARISONS), ('net_correct', 'delta_top1_percentage_points', 'delta_macro_top1_percentage_points'))):
        for name in names:
            aggregate[section][name] = {}
            for key in keys:
                values = [by_seed[str(seed)][section][name][key] for seed in SEEDS]
                aggregate[section][name][key] = {'mean': float(np.mean(values)),
                    'sample_std': float(np.std(values, ddof=1)), 'n': 3}
    for key in ('net_correct', 'top1_percentage_points', 'macro_top1_percentage_points'):
        values = [by_seed[str(seed)]['interaction'][key] for seed in SEEDS]
        aggregate['interaction'][key] = {'mean': float(np.mean(values)),
            'sample_std': float(np.std(values, ddof=1)), 'n': 3}
    return {'by_seed': by_seed, 'aggregate': aggregate,
            'all_seeds_strictly_better': all(x['combination_strictly_better_than_both'] for x in by_seed.values())}


def verify_model_output(out, model, val, vx, reference, predictions, prereg_sha):
    summary = read(out / 'summary.json')
    require(summary.get('status') == 'completed' and summary.get('scope') == 'factorial_retrieval_full2731'
            and summary.get('failed') == 0 and summary.get('optimizer_steps') == 0
            and summary.get('test_used') is False and summary.get('prereg_sha256') == prereg_sha,
            'Wrong model rerank scope/state')
    require(summary.get('train_samples') == 40154 and summary.get('val_samples') == 10304
            and summary.get('candidate_classes') == 2731 and summary.get('topk') == 5
            and summary.get('protocol') == PROTOCOL, 'Wrong per-model rerank population')
    for key in ('model_id', 'method', 'seed', 'checkpoint_sha256'):
        require(summary[key] == model[key], 'Rerank model identity differs: ' + key)
    targets = np.array([x['class_id'] for x in val])
    max_error = {}
    with np.load(out / 'scores.npz', allow_pickle=False) as z:
        require(set(z.files) == set(SCORES.values()) | {'candidate_ids'}, 'Unexpected score array inventory')
        require(np.array_equal(z['candidate_ids'], vx['top5_class_ids']), 'Scored candidates differ')
        for key, name in SCORES.items():
            require(z[name].dtype == np.float64, 'Score dtype differs')
            assert_close(z[name], reference[key], 'Independent retrieval math differs: ' + name)
            max_error[name] = float(np.abs(z[name] - reference[key]).max())
    output_metrics, output_pairs = {}, {}
    for mode, pred in predictions.items():
        require(np.array_equal(np.sort(pred, axis=1), np.sort(vx['top5_class_ids'], axis=1))
                and np.array_equal((pred == targets[:, None]).any(axis=1),
                                   (vx['top5_class_ids'] == targets[:, None]).any(axis=1)), 'Within-model Top5 changed')
        saved = rows(out / mode / 'predictions.jsonl')
        require(len(saved) == len(val), 'Prediction row count differs')
        for i, (got, item) in enumerate(zip(saved, val)):
            require(got['row'] == i and got['index'] == item['index'] and got['identity'] == item['identity']
                    and got['split'] == 'val' and got['video_sha256'] == item['video_sha256']
                    and got['signer_id'] == item['signer_id'] and got['target'] == item['class_id']
                    and got['gloss'] == item['gloss'] and got['method'] == mode
                    and got['model_id'] == model['model_id'] and got['checkpoint_sha256'] == model['checkpoint_sha256']
                    and got['original_full_logits_sha256'] == item['logits_sha256'], 'Prediction identity differs')
            require(got['original_candidate_ids'] == vx['top5_class_ids'][i].tolist()
                    and got['original_logits'] == vx['top5_logits'][i].tolist()
                    and got['ranked_class_ids'] == pred[i].tolist() and got['prediction'] == int(pred[i, 0]),
                    'Independent prediction ranking differs')
            require(got['correct_top1'] == bool(pred[i, 0] == targets[i])
                    and got['correct_top5'] == bool((pred[i] == targets[i]).any()), 'Prediction correctness differs')
            require(got['ranking_positions'] == [got['original_candidate_ids'].index(int(c)) for c in pred[i]],
                    'Prediction ranking positions differ')
            for key, name in SCORES.items():
                assert_close(got[name], reference[key][i], 'Saved prediction scores differ')
        measured = full_metrics(targets, pred)
        require(read(out / mode / 'metrics.json') == summary['metrics'][mode], 'Metric sidecar differs')
        check_metrics(summary['metrics'][mode], measured)
        output_metrics[mode] = {k: v for k, v in measured.items() if k != 'per_class'}
        stored_signers = read(out / mode / 'by_signer.json')
        require(stored_signers == summary['by_signer'][mode]
                and set(stored_signers) == {r['signer_id'] for r in val}, 'Signer inventory/sidecar differs')
        for signer in sorted(stored_signers):
            ix = np.array([i for i, item in enumerate(val) if item['signer_id'] == signer])
            check_metrics(stored_signers[signer], full_metrics(targets[ix], pred[ix]))
        if mode != 'original':
            stored = summary['paired_comparisons'][mode]
            require(read(out / mode / 'paired_vs_original.json') == stored, 'Paired sidecar differs')
            expected_pair = detailed_pair(val, targets, predictions['original'], pred)
            expected_pair['delta_macro_top1_percentage_points'] = 100. * (measured['macro_top1'] - output_metrics['original']['macro_top1'])
            check_pair(stored, expected_pair)
            output_pairs[mode] = paired(targets, predictions['original'], pred)
    margin = reference['base_probabilities'][:, 0] - reference['base_probabilities'][:, 1]
    locked = margin > 1./3. + 1e-12
    require(np.array_equal(predictions['original'][locked, 0], predictions['fixed_blend'][locked, 0]),
            'Fixed blending violated the margin bound')
    return {'metrics': output_metrics, 'paired': output_pairs, 'max_absolute_score_error': max_error,
            'margin_bound': {'locked_samples': int(locked.sum()), 'potentially_changeable_samples': int((~locked).sum()),
                             'errors_among_potentially_changeable': int(((~locked) & (predictions['original'][:, 0] != targets)).sum())}}, summary


def check_statistics_tree(got, expected):
    require(set(got) == set(expected), 'Aggregate statistics inventory differs')
    for key, value in expected.items():
        if isinstance(value, dict):
            check_statistics_tree(got[key], value)
        else:
            assert_close(got[key], value, 'Mean/sample standard deviation differs')


def verify_factorial(stored, expected):
    check_statistics_tree(stored['aggregate'], expected['aggregate'])
    require(set(stored['by_seed']) == {str(s) for s in SEEDS}, 'Factorial seed inventory differs')
    for seed, reference in expected['by_seed'].items():
        got = stored['by_seed'][seed]
        require(set(got['cells']) == set(CELLS) and set(got['comparisons']) == set(COMPARISONS),
                'Factorial cells/comparisons differ')
        for name, value in reference['cells'].items():
            check_metrics(got['cells'][name], value)
        for name, value in reference['comparisons'].items():
            check_pair(got['comparisons'][name], value, macro=True)
        for key, value in reference['interaction'].items():
            assert_close(got['interaction'][key], value, 'Factorial interaction differs')
        require(got['combination_strictly_better_than_both'] is reference['combination_strictly_better_than_both'],
                'Per-seed combination criterion differs')


def analyze(args):
    launch_path = inside(args.launch)
    launch, state = read(launch_path), read(launch_path.parent / 'state.json')
    require(state['state'] == 'completed', 'Factorial lane has not completed')
    prereg_path = inside(launch['prereg_path'])
    prereg_sha = sha(prereg_path)
    require(prereg_sha == launch['prereg_sha256'], 'Launch preregistration SHA differs')
    prereg = read(prereg_path)
    models = check_prereg(prereg)
    tracked = {str(p.relative_to(ROOT)): sha(p) for p in (launch_path, launch_path.parent / 'state.json', prereg_path)}
    for name, expected in launch['frozen_files'].items():
        require(sha(inside(name)) == expected, 'Frozen launch input changed: ' + name)
    runs = {name: inside(getattr(args, key)) for name, key in
            (('train_extraction', 'train_run'), ('val_extraction', 'val_run'), ('rerank', 'rerank_run'))}
    proof = {}
    for name, run in runs.items():
        require(state['stages'][name]['state'] == 'completed' and inside(state['stages'][name]['run_dir']) == run,
                'Analysis run differs from completed lane stage')
        for leaf in ('manifest.json', 'artifacts.json', 'config.json', 'code_manifest.json', 'exit_code.txt'):
            tracked[str((run / leaf).relative_to(ROOT))] = sha(run / leaf)
        proof[name] = verify_run(run)
    td, tm = load_feature_container(runs['train_extraction'], 'train', prereg, prereg_sha)
    vd, vm = load_feature_container(runs['val_extraction'], 'val', prereg, prereg_sha)
    out = runs['rerank'] / 'rerank'
    summary = read(out / 'summary.json')
    require(summary.get('status') == 'completed' and summary.get('protocol') == PROTOCOL
            and summary.get('scope') == 'factorial_retrieval_full2731' and summary.get('failed') == 0
            and summary.get('test_used') is False and summary.get('optimizer_steps') == 0
            and summary.get('prereg_sha256') == prereg_sha, 'Wrong overall rerank scope/state')
    require(set(summary['models']) == {m['model_id'] for m in models}, 'Rerank model inventory differs')
    require(summary['train_manifest_sha256'] == sha(td / 'manifest.json')
            and summary['val_manifest_sha256'] == sha(vd / 'manifest.json'), 'Rerank extraction bindings differ')
    require(read(out / 'factorial.json') == summary['factorial'], 'Factorial sidecar differs')
    cfg = read(out / 'config.json')
    require(sha(out / 'config.json') == summary['config_sha256']
            and cfg['protocol'] == PROTOCOL and cfg['prereg_sha256'] == prereg_sha
            and cfg['train_manifest_sha256'] == sha(td / 'manifest.json')
            and cfg['val_manifest_sha256'] == sha(vd / 'manifest.json')
            and cfg['beta'] == 10 and cfg['memory_weight'] == .25 and cfg['base_weight'] == .75
            and cfg['model_ids'] == [m['model_id'] for m in models], 'Rerank input/config differs')
    for item in summary['inputs'] + summary['source_files']:
        require(sha(inside(item['path'])) == item['sha256'], 'Rerank source/input SHA differs')
        tracked[item['path']] = item['sha256']
    predictions, model_results, canonical_val, common = {}, {}, None, {}
    for model in models:
        mid = model['model_id']
        train, tx, _ = load_model_features(td, tm, 'train', model, prereg, prereg_sha, tracked)
        val, vx, _ = load_model_features(vd, vm, 'val', model, prereg, prereg_sha, tracked)
        for split, records in (('train', train), ('val', val)):
            fingerprint = shared_rows_fingerprint(records)
            if split in common:
                require(common[split] == fingerprint, 'Cross-model pose/identity/input/order differs')
            common[split] = fingerprint
        require(not ({x['video_sha256'] for x in train} & {x['video_sha256'] for x in val})
                and not ({x['signer_id'] for x in train} & {x['signer_id'] for x in val}), 'Memory/query overlap')
        if canonical_val is None:
            canonical_val = val
        else:
            require([{k: r[k] for k in IDENTITY + ('identity', 'class_id')} for r in val]
                    == [{k: r[k] for k in IDENTITY + ('identity', 'class_id')} for r in canonical_val],
                    'Six models do not have identical validation identities/order')
        old_path = inside(model['reference_predictions_path'])
        require(sha(old_path) == model['reference_predictions_sha256'], 'Historical reference SHA differs')
        tracked[str(old_path.relative_to(ROOT))] = model['reference_predictions_sha256']
        verify_history(val, vx, rows(old_path))
        ix = np.asarray(sorted(range(len(train)), key=lambda i: train[i]['video_sha256']))
        reference = reference_scores(tx['embeddings'][ix], np.array([train[i]['class_id'] for i in ix]),
            np.array([train[i]['signer_id'] for i in ix]), vx['embeddings'], vx['top5_class_ids'], vx['top5_logits'])
        pred = {'original': vx['top5_class_ids'], 'pure_memory': order(vx['top5_class_ids'], reference['memory_probabilities']),
                'fixed_blend': order(vx['top5_class_ids'], reference['blend_probabilities'])}
        model_results[mid], stored = verify_model_output(out / 'models' / mid, model, val, vx, reference, pred, prereg_sha)
        require(stored == summary['models'][mid], 'Per-model summary copies differ')
        require(stored['train_manifest_sha256'] == tm['models'][mid]['manifest_sha256']
                and stored['val_manifest_sha256'] == vm['models'][mid]['manifest_sha256'], 'Per-model input bindings differ')
        predictions[mid] = pred
        print(json.dumps({'independently_verified_model': mid, 'prediction_rows': len(val) * 3}), flush=True)
    targets = np.asarray([r['class_id'] for r in canonical_val])
    factorial = derive_factorial(canonical_val, targets, models, predictions)
    verify_factorial(summary['factorial'], factorial)
    signal = summary['retention_signal']
    require(summary['factorial']['retention_signal'] == signal, 'Retention signal copies differ')
    require(signal['per_seed'] == {seed: item['combination_strictly_better_than_both']
                                  for seed, item in factorial['by_seed'].items()}
            and signal['all_seeds_strictly_positive'] is True
            and signal['automatic_further_experiments'] is False
            and signal['statistical_significance_test'] is False, 'Retention signal scope differs')
    require(signal['mathematical_conditions_met'] is factorial['all_seeds_strictly_better']
            and signal['independent_verification_required'] is True
            and signal['independent_verification_completed'] is False and signal['combination_retained'] is False,
            'Production retention signal differs or skipped independent verification')
    for name, expected in launch['frozen_files'].items():
        require(sha(inside(name)) == expected, 'Frozen input changed during analysis: ' + name)
    for name, expected in tracked.items():
        require(sha(inside(name)) == expected, 'Analysis input changed during analysis: ' + name)
    for name, run in runs.items():
        require(verify_run(run) == proof[name], 'Completed artifacts changed during independent analysis')
    return {'status': 'verified', 'protocol': PROTOCOL, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'test_used': False, 'optimizer_steps': 0, 'prereg_sha256': prereg_sha, 'models': model_results,
            'factorial': factorial,
            'retention_signal': {'mathematical_conditions_met': factorial['all_seeds_strictly_better'],
                'independent_verification_required': True, 'independent_verification_completed': True,
                'combination_retained': factorial['all_seeds_strictly_better'], 'automatic_further_experiments': False,
                'statistical_significance_claim': False},
            'verification': {'frozen_files': len(launch['frozen_files']), 'all_185472_prediction_rows_exact': True,
                'historical_61824_top5_ranks_full_logits_and_probabilities_exact': True,
                'all_within_model_top5_sets_and_hits_unchanged': True, 'all_class_and_signer_metrics_recomputed': True,
                'three_seeds_preserved': list(SEEDS), 'input_and_artifact_bytes_unchanged': True},
            'artifact_verification': proof, 'input_sha256': tracked,
            'runs': {name: str(run.relative_to(ROOT)) for name, run in runs.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('launch', 'train-run', 'val-run', 'rerank-run', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    args = p.parse_args()
    output = inside(args.output)
    require(not output.exists(), 'Independent analysis output is immutable')
    result = analyze(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'combination_retained': result['retention_signal']['combination_retained']}))


if __name__ == '__main__':
    main()
