#!/usr/bin/env python3
"""Audit exact 10-to-20 epoch continuation and separate budget from geometry effects.

Read-only: no model inference, retraining, search, TTA, retrieval, or test access.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'docs/tools'))
from analyze_paired_seeds import score, paired, aggregate, identity
from run_rgb_pilot import verify_run, sha, require
from run_paired_seed_lane import verify_predictions

KIND = 'poet-extended-training-v1'
SEEDS = (20260922, 20260923, 20260924)
METHODS = ('baseline', 'geometry')
FIELDS = ('top1_percent', 'macro_top1_percent', 'top5_percent', 'macro_top5_percent')
SHA = re.compile(r'^[0-9a-f]{64}$')


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def resolve(value, root=ROOT, remote_root=None):
    path = Path(value)
    require('..' not in path.parts, 'Unsafe input path')
    if path.is_absolute() and remote_root is not None and path.is_relative_to(Path(remote_root)):
        path = Path(root) / path.relative_to(Path(remote_root))
    elif not path.is_absolute():
        path = Path(root) / path
    path = path.resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Input escapes workspace')
    return path


def check_plan(plan):
    require(plan.get('schema_version') == 1 and plan.get('kind') == KIND
            and plan.get('seeds') == list(SEEDS) and plan.get('test_used') is False,
            'Wrong extension plan identity or seed/test scope')
    counters = {'initial_epochs': 10, 'epochs': 20, 'training_samples': 40154, 'validation_samples': 10304,
                'initial_optimizer_steps': 3140, 'optimizer_steps': 6280,
                'additional_optimizer_steps': 3140, 'samples_seen': 803080}
    require(all(plan.get(k) == v for k, v in counters.items()), 'Training budget differs from finite extension')
    require(set(plan['lanes']) == {'gpu0', 'gpu1'}, 'Wrong extension lanes')
    jobs = []
    for lane, method in (('gpu0', 'baseline'), ('gpu1', 'geometry')):
        entries = plan['lanes'][lane]['jobs']
        require([j['seed'] for j in entries] == list(SEEDS) and all(j['method'] == method for j in entries),
                'Missing, duplicate, reordered, or selected paired seeds')
        jobs.extend(entries)
    require(len({j['job_id'] for j in jobs}) == 6, 'Duplicate job identity')
    return jobs


def valid_metrics(metrics, count=10304, classes=2731):
    require(metrics['attempted'] == metrics['inferred'] == count and metrics['failed'] == 0
            and metrics['vocabulary_classes'] == classes and metrics['missing_classes'] == [],
            'Incomplete validation denominator or class inventory')
    for level in ('per_instance', 'per_class_observed', 'per_class_full_vocabulary'):
        for key in ('top1', 'top5'):
            value = metrics[level][key]
            require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
                    'Invalid epoch metric')
    require(metrics['per_class_observed']['denominator_classes'] == classes, 'Observed vocabulary differs')


def validate_history(parent_history, new_history, best, *, count=10304, classes=2731):
    require([x['epoch'] for x in parent_history] == list(range(11)), 'Parent history must be epochs 0..10')
    require([x['epoch'] for x in new_history] == list(range(21)), 'Continuation history must be epochs 0..20')
    require(new_history[:11] == parent_history, 'Inherited 0..10 history changed')
    for row in new_history:
        require(row['split'] == 'val', 'History contains non-validation evaluation')
        valid_metrics(row['metrics'], count, classes)
        if row['epoch']:
            require(row['training']['samples'] == 40154 and row['training']['optimizer_steps'] == 314,
                    'Epoch training population/budget differs')
    selected = max(new_history, key=lambda x: x['metrics']['per_class_full_vocabulary']['top1'])
    require(best == {'epoch': selected['epoch'], 'score': selected['metrics']['per_class_full_vocabulary']['top1']},
            'Best must be earliest maximum macro Top1 across all 0..20 epochs')
    return selected


def selected_predictions_path(record, parent, child, root=ROOT, remote_root=None):
    owner = Path(parent) if record['epoch'] <= 10 else Path(child)
    expected = owner / 'training/evaluation' / ('epoch-%04d' % record['epoch']) / 'predictions.jsonl'
    actual = resolve(record['predictions_path'], root, remote_root)
    require(actual == expected.resolve(), 'Selected-epoch prediction owner/path differs')
    return actual


def compare_config(parent, current, item, parent_checkpoint, root=ROOT, remote_root=None):
    ignored = {'epochs', 'output_dir', 'resume'}
    left = {k: v for k, v in parent.items() if k not in ignored}
    right = {k: v for k, v in current.items() if k not in ignored}
    # The first baseline predates the explicit default module_lr=None key.
    left.setdefault('module_lr', None)
    right.setdefault('module_lr', None)
    path_keys = ('train_cache_manifest', 'val_cache_manifest', 'train_geometry_manifest',
                 'val_geometry_manifest', 'attribute_targets', 'checkpoint_dir')
    for config in (left, right):
        for key in path_keys:
            if config.get(key) is not None:
                config[key] = str(resolve(config[key], root, remote_root))
    require(left == right, 'Continuation altered training configuration beyond target/output/resume')
    require(current['epochs'] == 20 and parent['epochs'] == 10
            and current['method'] == item['method'] and current['seed'] == current['head_seed'] == item['seed']
            and current['engineering_smoke'] is False, 'Wrong continuation configuration')
    require(resolve(current['resume'], root, remote_root) == Path(parent_checkpoint).resolve(),
            'Continuation did not start from the frozen parent last checkpoint')


def verify_per_class(path, rows, expected_classes=2731):
    with Path(path).open(newline='') as stream:
        table = list(csv.DictReader(stream))
    require(len(table) == expected_classes and [int(x['class_id']) for x in table] == list(range(expected_classes)),
            'Per-class table inventory differs')
    labels = {int(x['class_id']): x['gloss'] for x in table}
    require(len(set(labels.values())) == expected_classes, 'Per-class label map is not bijective')
    totals = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        predictions = row['top5']
        require(all(type(p['class_id']) is int and labels.get(p['class_id']) == p['gloss'] for p in predictions),
                'Prediction class ID/gloss mismatch')
        require(len({p['class_id'] for p in predictions}) == 5
                and [p['probability'] for p in predictions] == sorted([p['probability'] for p in predictions], reverse=True)
                and SHA.fullmatch(row.get('logits_sha256', '')), 'Malformed prediction ranking/logit identity')
        bucket = totals[row['gloss']]
        bucket[0] += 1
        bucket[1] += predictions[0]['gloss'] == row['gloss']
        bucket[2] += row['gloss'] in [p['gloss'] for p in predictions]
    require(set(totals) == set(labels.values()), 'Missing validation class')
    for row in table:
        n, one, five = totals[row['gloss']]
        expected = {'attempted': n, 'inferred': n, 'failed': 0, 'top1_correct': one, 'top5_correct': five,
                    'errors_top1': n - one, 'errors_top5': n - five}
        require(all(int(row[k]) == v for k, v in expected.items()), 'Per-class count differs')
        require(abs(float(row['top1']) - one / n) < 1e-12 and abs(float(row['top5']) - five / n) < 1e-12,
                'Per-class accuracy differs')
    return labels


def verify_metric_values(stored, measured):
    valid_metrics(stored, measured['samples'], measured['classes'])
    for level, prefix in (('per_instance', ''), ('per_class_full_vocabulary', 'macro_'), ('per_class_observed', 'macro_')):
        for top in ('top1', 'top5'):
            require(abs(100 * stored[level][top] - measured[prefix + top + '_percent']) < 1e-9,
                    'Independent metric recomputation differs')
    require(stored['top1_correct'] == measured['top1_correct'] and stored['top5_correct'] == measured['top5_correct'],
            'Independent correct counts differ')


def compare_predictions(before, after, before_metrics, after_metrics):
    result = paired(before, after)
    for signer in result['by_signer_descriptive_only'].values():
        signer['before_correct'] = signer.pop('baseline_correct')
        signer['after_correct'] = signer.pop('geometry_correct')
    result['metric_differences_percentage_points'] = {k: after_metrics[k] - before_metrics[k] for k in FIELDS}
    result['fixed_examples'], result['regressed_examples'] = [], []
    for left, right in zip(before, after):
        a, b, target = left['top5'][0]['gloss'], right['top5'][0]['gloss'], left['gloss']
        kind = 'fixed_examples' if a != target and b == target else 'regressed_examples' if a == target and b != target else None
        if kind:
            result[kind].append({'index': left['index'], 'video_sha256': left['video_sha256'],
                'signer_id': left['signer_id'], 'target': target, 'before': a, 'after': b})
    return result


def geometry_effect_change(b10, g10, b20, g20):
    return {key: (g20[key] - b20[key]) - (g10[key] - b10[key]) for key in FIELDS}


def descriptive_statistics(values):
    require(len(values) >= 2 and all(math.isfinite(v) for v in values), 'Need finite paired seeds')
    return {'mean': statistics.fmean(values), 'sample_sd': statistics.stdev(values),
            'per_seed': values, 'all_strictly_positive': all(x > 0 for x in values)}


def aggregate_extended(results):
    require(len(results) >= 2, 'Need at least two seeds for sample SD')
    out = {'seeds': [r['seed'] for r in results], 'n_seeds': len(results),
           'standard_deviation_definition': 'sample standard deviation across seeds (ddof=1)',
           'units': 'method means in percent; differences in percentage points'}
    for budget in ('ten_epochs', 'twenty_epochs'):
        adapted = [{'seed': r['seed'], **r[budget]} for r in results]
        out[budget] = aggregate(adapted)
    out['training_budget_effect'] = {method: {field: descriptive_statistics([
        row['comparisons'][method + '20_vs_' + method + '10']['metric_differences_percentage_points'][field]
        for row in results]) for field in FIELDS} for method in METHODS}
    out['geometry_effect_change'] = {field: descriptive_statistics([r['geometry_effect_change'][field] for r in results])
                                     for field in FIELDS}
    return out


def check_step_trace(path, cohort, seed, method, history, *, initial_epoch=10, final_epoch=20, batch_size=128, allow_legacy_trace=False):
    """Stream true batch evidence; fingerprint common B/G sampling without geometry-specific payload."""
    count = len(cohort)
    per_epoch = math.ceil(count / batch_size)
    expected_steps = (final_epoch - initial_epoch) * per_epoch
    steps, visits = 0, initial_epoch * count
    epoch_indices, common = [], hashlib.sha256()
    epoch_orders = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            epoch = initial_epoch + 1 + steps // per_epoch
            offset = steps % per_epoch
            n = min(batch_size, count - offset * batch_size)
            steps += 1
            require(epoch <= final_epoch and row['epoch'] == epoch
                    and row['global_optimizer_step'] == initial_epoch * per_epoch + steps, 'Step/epoch counter differs')
            indices = row['indices']
            require(len(indices) == n and all(type(i) is int and 0 <= i < count for i in indices), 'Malformed training batch')
            visits += n
            require(row['samples_seen'] == visits, 'Sample-visit counter differs')
            videos = [cohort[i]['video_sha256'] for i in indices]
            require(row['video_sha256'] == videos, 'Batch video identities differ from training cohort')
            seeds = [int.from_bytes(hashlib.sha256(json.dumps(['temporal', seed, epoch, v],
                     separators=(',', ':')).encode()).digest()[:4], 'big') for v in videos]
            require(row['temporal_seeds'] == seeds and len(row['batch_trace']) == n
                    and len(row['temporal_indices_sha256']) == n, 'Temporal sampling evidence differs')
            for i, trace in enumerate(row['batch_trace']):
                require(trace['temporal_seed'] == seeds[i]
                        and trace['source_indices_sha256'] == row['temporal_indices_sha256'][i]
                        and SHA.fullmatch(trace['source_indices_sha256']) and SHA.fullmatch(trace['input_sha256']),
                        'Batch trace input/time linkage differs')
                if method == 'geometry':
                    require(SHA.fullmatch(trace.get('geometry_sha256', '')), 'Missing geometry batch evidence')
            if 'optimizer_groups' not in row:
                require(allow_legacy_trace and method == 'baseline' and seed == 20260922,
                        'Optimizer-group evidence missing outside frozen legacy baseline')
                groups = [('all', row['lr'])]
            else:
                groups = [(g['name'], g['lr']) for g in row['optimizer_groups']]
            expected_groups = [('base', 3e-5), ('new-module', 3e-4)] if method == 'geometry' else [('all', 3e-5)]
            require(groups == expected_groups and row['lr'] == 3e-5 and row['auxiliary'] == 0
                    and all(math.isfinite(row[k]) for k in ('loss', 'word_ce', 'gradient_l2')),
                    'Optimizer/loss configuration differs')
            record = {k: row[k] for k in ('epoch', 'global_optimizer_step', 'samples_seen', 'indices',
                                          'video_sha256', 'temporal_seeds', 'temporal_indices_sha256')}
            record['input_sha256'] = [t['input_sha256'] for t in row['batch_trace']]
            common.update(json.dumps(record, sort_keys=True, separators=(',', ':')).encode() + b'\n')
            epoch_indices.extend(indices)
            if offset == per_epoch - 1:
                require(sorted(epoch_indices) == list(range(count)), 'Missing or repeated training sample within epoch')
                digest = hashlib.sha256(struct.pack('<' + 'q' * len(epoch_indices), *epoch_indices)).hexdigest()
                require(history[epoch]['training']['order_sha256'] == digest, 'History sampler-order SHA differs')
                epoch_orders[str(epoch)] = digest
                epoch_indices = []
    require(steps == expected_steps and visits == final_epoch * count and not epoch_indices, 'Incomplete new training trace')
    return {'new_optimizer_steps': steps, 'new_sample_visits': visits - initial_epoch * count,
            'final_optimizer_step': final_epoch * per_epoch, 'final_samples_seen': visits,
            'common_sampling_and_pose_sha256': common.hexdigest(), 'epoch_order_sha256': epoch_orders}


def verify_input_run(run, *, parent_item=None, plan=None, root=ROOT):
    """Only a declared parent uses the narrowly frozen legacy-bytecode audit."""
    if parent_item is None:
        return verify_run(run)
    require(plan is not None and Path(run).resolve() == resolve(parent_item['parent_training_run'], root, plan['workspace']),
            'Parent audit exception cannot be applied to a different run')
    from run_extended_training_lane import verify_parent_run
    return verify_parent_run(run, parent_item, plan, root)


def analyze(root, plan_path):
    root = Path(root).resolve()
    plan_path = resolve(plan_path, root)
    plan = read(plan_path)
    jobs = check_plan(plan)
    plan_hash = sha(plan_path)
    remote_root = Path(plan['workspace'])
    tracked, run_proofs, parent_audits = {}, {}, {}
    def track(path, expected=None):
        path = resolve(path, root, remote_root)
        actual = sha(path)
        require(expected is None or actual == expected, 'Frozen or indexed file changed: ' + str(path))
        name = str(path.relative_to(root))
        require(name not in tracked or tracked[name] == actual, 'Input changed during initial read')
        tracked[name] = actual
        return path
    track(plan_path, plan_hash)
    for name, digest in plan['frozen_files'].items():
        track(name, digest)
    def audit_run(run, parent_item=None):
        run = resolve(run, root, remote_root)
        key = str(run.relative_to(root))
        if key not in run_proofs:
            for leaf in ('manifest.json', 'artifacts.json', 'config.json', 'code_manifest.json', 'exit_code.txt'):
                track(run / leaf)
            run_proofs[key] = verify_input_run(run, parent_item=parent_item, plan=plan, root=root)
            if parent_item is not None:
                parent_audits[key] = parent_item
        else:
            require(parent_audits.get(key) == parent_item, 'Run audit role changed')
        return run
    cohort = read(track(plan['validation_manifest']))['samples']
    train_cohort = read(track(plan['training_manifest']))['samples']
    require(len(cohort) == 10304 and len(train_cohort) == 40154
            and len({r['gloss'] for r in cohort}) == 2731 and all(r['split'] == 'val' for r in cohort)
            and all(r['split'] == 'train' for r in train_cohort), 'Wrong full data populations')
    require(not ({r['video_sha256'] for r in train_cohort} & {r['video_sha256'] for r in cohort})
            and not ({r['signer_id'] for r in train_cohort} & {r['signer_id'] for r in cohort}), 'Train/val overlap')
    completed = {}
    for lane in ('gpu0', 'gpu1'):
        state_path = track(plan_path.parent / lane / 'state.json')
        state = read(state_path)
        require(state['state'] == 'completed' and state['plan_sha256'] == plan_hash, 'Lane incomplete or plan mismatch')
        requested = plan['lanes'][lane]['jobs']
        require(len(state['jobs']) == len(requested), 'Lane job population differs')
        for item, state_job in zip(requested, state['jobs']):
            require(all(state_job[k] == item[k] for k in ('job_id', 'method', 'seed'))
                    and state_job['state'] == 'completed'
                    and all(state_job[part]['state'] == 'completed' for part in ('training', 'evaluation')),
                    'Job state/identity differs')
            completed[item['job_id']] = state_job
    # Supplement rather than replace runner verification with independent calculations below.
    from run_extended_training_lane import validate_plan, verify_training, verify_evaluation
    for lane in ('gpu0', 'gpu1'):
        validate_plan(plan, lane, root)
    references, traces = {}, {}
    for item in jobs:
        method, seed = item['method'], item['seed']
        state_job = completed[item['job_id']]
        parent = audit_run(item['parent_training_run'], parent_item=item)
        old_eval = audit_run(item['parent_evaluation_run'])
        child = audit_run(state_job['training']['run_dir'])
        new_eval = audit_run(state_job['evaluation']['run_dir'])
        runner_proof = verify_training(child, item, plan, root)
        runner_eval = verify_evaluation(child, new_eval, item, runner_proof, plan)
        ps, cs = read(track(parent / 'training/summary.json')), read(track(child / 'training/summary.json'))
        require((ps['completed_epoch'], ps['global_optimizer_step'], ps['samples_seen']) == (10, 3140, 401540)
                and (cs['completed_epoch'], cs['global_optimizer_step'], cs['samples_seen']) == (20, 6280, 803080)
                and ps['state'] == cs['state'] == 'completed' and ps['scope'] == cs['scope'] == 'formal_finetuning',
                'Parent or child training budget/state differs')
        parent_checkpoint = track(item['parent_checkpoint'], item['parent_checkpoint_sha256'])
        require(parent_checkpoint == parent / 'training/checkpoints/last.pth', 'Resume must use parent last')
        compare_config(ps['config'], cs['config'], item, parent_checkpoint, root, remote_root)
        resume = cs['resume']
        require(resolve(resume['checkpoint'], root, remote_root) == parent_checkpoint
                and resume['sha256'] == item['parent_checkpoint_sha256'] and resume['completed_epoch'] == 10
                and resolve(resume['parent_output'], root, remote_root) == parent / 'training', 'Exact resume lineage differs')
        parent_protocol, child_protocol = (read(track(run / 'training/protocol.json')) for run in (parent, child))
        require(parent_protocol == child_protocol and parent_protocol['sha256'] == item['parent_protocol_sha256']
                == ps['protocol_sha256'] == cs['protocol_sha256']
                and canonical_sha(child_protocol['protocol']) == child_protocol['sha256'], 'Training protocol changed on continuation')
        ph = read(track(parent / 'training/metrics_history.json'))['epochs']
        ch = read(track(child / 'training/metrics_history.json'))['epochs']
        selected = validate_history(ph, ch, cs['best'])
        for row in ch:
            path = selected_predictions_path(row, parent, child, root, remote_root)
            track(path, row['predictions_sha256'])
        ci = read(track(child / 'training/checkpoints/index.json'))
        pi = read(track(parent / 'training/checkpoints/index.json'))
        require(ci['last']['epoch'] == 20 and ci['best']['epoch'] == selected['epoch']
                and pi['last']['epoch'] == 10 and pi['last']['sha256'] == item['parent_checkpoint_sha256'],
                'Checkpoint index epochs/lineage differ')
        for run, index in ((parent, pi), (child, ci)):
            for kind in ('last', 'best'):
                path = track(index[kind]['path'], index[kind]['sha256'])
                require(path.is_relative_to(run / 'training/checkpoints'), 'Checkpoint escapes owning training run')
            track(run / 'training/checkpoints/best.pth', index['best']['sha256'])
        if selected['epoch'] <= 10:
            require(ci['best']['sha256'] == pi['best']['sha256'] and cs['best'] == ps['best'],
                    'Retained old best checkpoint was changed')
        trace = check_step_trace(track(child / 'training/steps.jsonl'), train_cohort, seed, method, ch,
            allow_legacy_trace=(method == 'baseline' and seed == 20260922
                and item['trainer_script'] == item['parent_training_run'] + '/code/scripts/finetune_methods.py'))
        traces.setdefault(seed, {})[method] = trace
        old_best = max(ph, key=lambda x: x['metrics']['per_class_full_vocabulary']['top1'])
        require(ps['best'] == {'epoch': old_best['epoch'], 'score': old_best['metrics']['per_class_full_vocabulary']['top1']},
                'Old best selection differs')
        measures, all_rows = {}, {}
        for budget, run, training, best, index in (('ten_epochs', old_eval, parent, old_best, pi),
                                                  ('twenty_epochs', new_eval, child, selected, ci)):
            directory = run / 'evaluation'
            summary = read(track(directory / 'summary.json'))
            stored = read(track(directory / 'metrics.json'))
            require(summary['status'] == 'completed' and summary['scope'] == 'official_validation'
                    and summary['attribute_targets_read'] is False and summary['method'] == method
                    and summary['checkpoint_epoch'] == best['epoch'] and summary['checkpoint_sha256'] == index['best']['sha256']
                    and summary['protocol_sha256'] == item['parent_protocol_sha256']
                    and stored == summary['metrics'] == best['metrics'], 'Evaluation model/metrics linkage differs')
            prediction_path = track(directory / 'predictions.jsonl')
            selected_path = selected_predictions_path(best, parent, child, root, remote_root)
            track(selected_path, best['predictions_sha256'])
            verify_predictions(selected_path, prediction_path, 10304)
            require(read(track(selected_path.parent / 'metrics.json'))['metrics'] == stored,
                    'Selected epoch metric sidecar differs')
            require(track(selected_path.parent / 'per_class.csv').read_bytes() == track(directory / 'per_class.csv').read_bytes(),
                    'Independent per-class table differs from selected epoch')
            rows = jsonl(prediction_path)
            measured = score(rows, cohort, 2731)
            verify_metric_values(stored, measured)
            verify_per_class(directory / 'per_class.csv', rows)
            measures[budget] = {'training_run': str(training.relative_to(root)), 'evaluation_run': str(run.relative_to(root)),
                'best_epoch': best['epoch'], 'checkpoint_sha256': index['best']['sha256'], 'recomputed': measured,
                'selected_predictions_path': str(selected_path.relative_to(root)),
                'selected_from_parent_history': best['epoch'] <= 10}
            all_rows[budget] = rows
        references[(seed, method)] = {'measures': measures, 'predictions': all_rows,
                                      'runner_training_verification': runner_proof, 'runner_evaluation_verification': runner_eval}
        print(json.dumps({'verified_continuation': item['job_id'], 'best_epoch': selected['epoch'],
                          'new_steps': trace['new_optimizer_steps'], 'validation_predictions': 10304}), flush=True)
    result_rows = []
    for seed in SEEDS:
        require(traces[seed]['baseline']['common_sampling_and_pose_sha256']
                == traces[seed]['geometry']['common_sampling_and_pose_sha256'], 'Paired methods saw different batches/temporal inputs')
        row = {'seed': seed, 'role': 'earlier_development_seed' if seed == SEEDS[0] else 'later_development_seed',
               'comparisons': {}, 'training_trace': traces[seed]}
        for budget in ('ten_epochs', 'twenty_epochs'):
            row[budget] = {method: references[(seed, method)]['measures'][budget] for method in METHODS}
        pairs = (('G10_vs_B10', 'baseline', 'ten_epochs', 'geometry', 'ten_epochs'),
                 ('G20_vs_B20', 'baseline', 'twenty_epochs', 'geometry', 'twenty_epochs'),
                 ('baseline20_vs_baseline10', 'baseline', 'ten_epochs', 'baseline', 'twenty_epochs'),
                 ('geometry20_vs_geometry10', 'geometry', 'ten_epochs', 'geometry', 'twenty_epochs'))
        for name, bm, bb, am, ab in pairs:
            row['comparisons'][name] = compare_predictions(references[(seed, bm)]['predictions'][bb],
                references[(seed, am)]['predictions'][ab], row[bb][bm]['recomputed'], row[ab][am]['recomputed'])
        row['geometry_effect_change'] = geometry_effect_change(
            row['ten_epochs']['baseline']['recomputed'], row['ten_epochs']['geometry']['recomputed'],
            row['twenty_epochs']['baseline']['recomputed'], row['twenty_epochs']['geometry']['recomputed'])
        result_rows.append(row)
    for name, digest in tracked.items():
        require(sha(root / name) == digest, 'Frozen/input file changed during analysis: ' + name)
    for name, proof in run_proofs.items():
        require(verify_input_run(root / name, parent_item=parent_audits.get(name), plan=plan, root=root) == proof,
                'Run artifacts changed during independent analysis: ' + name)
    return {'schema_version': 1, 'kind': KIND, 'status': 'verified', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'validation_selected_best_of_existing_10_and_exact_continued_20_epochs', 'test_used': False,
        'tta_used': False, 'retrieval_used': False, 'rgb_used': False, 'automatic_further_experiments': False,
        'statistical_significance_claim': False, 'plan_sha256': plan_hash, 'analyzer_sha256': sha(__file__),
        'paired_seeds': result_rows, 'three_seeds_descriptive': aggregate_extended(result_rows),
        'two_later_seeds_descriptive': aggregate_extended(result_rows[1:]),
        'verification': {'new_training_runs': 6, 'new_independent_evaluations': 6,
            'new_independent_prediction_rows_exact': 61824, 'old_reference_prediction_rows_verified': 61824,
            'all_inherited_history_0_to_10_exact': True, 'all_histories_0_to_20_complete': True,
            'new_steps_per_model': 3140, 'total_steps_per_model': 6280, 'total_sample_visits_per_model': 803080,
            'paired_actual_sampling_and_temporal_inputs_exact': True, 'frozen_files': len(plan['frozen_files']),
            'full_artifact_verification_before_and_after': True},
        'run_verification': run_proofs, 'verified_input_sha256': tracked,
        'interpretation_limits': [
            'Same validation has already been used repeatedly for development and best-epoch selection; this is not test.',
            'All three seeds are retained; seed 20260922 informed earlier development, and the later two are shown separately.',
            '20-epoch models use twice the 10-epoch training budget; their absolute improvement is not automatically a geometry contribution.',
            'Geometry contribution is G20-B20 at equal budget; its change is (G20-B20)-(G10-B10).',
            'Best selection includes epochs 0..20 with ties keeping the earliest, so a pre-extension best may remain selected.',
            'Means and sample SD across the same validation cohort are descriptive, not significance or confidence intervals.',
            'Only the authorized fixed continuation is evaluated; no automated further training, tuning, retrieval, TTA, RGB, or test.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', '--root', type=Path, default=ROOT)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    output = resolve(args.output, root)
    require(not output.exists(), 'Analysis output is immutable')
    result = analyze(root, args.plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(output),
                      'three_seeds_descriptive': result['three_seeds_descriptive']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
