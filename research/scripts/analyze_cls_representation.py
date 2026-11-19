#!/usr/bin/env python3
"""Representation x classifier analysis for the preregistered CLS study.

Reads saved validation predictions only: the six existing handcrafted cells, the
three new CLS cells from the frozen runner and the official-head replay. Computes
Top-1/Top-5/macro, per-signer metrics, paired fixed/regressed counts with exact
McNemar p, signer-stratified bootstrap CIs, the preregistered gap decomposition
and classifier costs. No model is loaded, nothing is refit and no test data is
read. scripts/verify_cls_representation.py recomputes the point results with the
standard library only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from math import comb
from pathlib import Path

import numpy as np

from evaluate_pose_cache import sha256, write_json

WORKSPACE = Path(__file__).resolve().parents[1]
METHODS = ('knn', 'linear_svm', 'random_forest')
SHORT = {'knn': 'knn', 'linear_svm': 'svm', 'random_forest': 'rf'}
IDENTITY_KEYS = ('index', 'gloss', 'signer_id', 'split', 'source_path', 'video_sha256')
HANDCRAFTED = {
    'ho': 'runs/20261002T161254Z-classical-formal-three-methods-dac0c02f/experiment',
    'hc': 'runs/20261010T065827Z-handedness-arma-canonical-refit-489d698b/experiment',
}
REPRESENTATION_NAMES = {'ho': 'handcrafted 1,020-D (original)', 'hc': 'handcrafted 1,020-D (handedness-canonical)',
                        'cls': 'official PoET 128-D CLS'}
PRIMARY = [('P1', 'hc_svm', 'cls_svm', 'representation effect, Linear SVM'),
           ('P2', 'hc_knn', 'cls_knn', 'representation effect, kNN'),
           ('P3', 'hc_rf', 'cls_rf', 'representation effect, bounded RF'),
           ('P4', 'cls_svm', 'head', 'classifier effect on fixed CLS: official head vs Linear SVM'),
           ('P5', 'cls_knn', 'head', 'official head vs kNN on CLS'),
           ('P6', 'cls_rf', 'head', 'official head vs bounded RF on CLS'),
           ('P7', 'hc_svm', 'head', 'total gap')]
SECONDARY = [('P1o', 'ho_svm', 'cls_svm', 'representation effect vs original descriptor, Linear SVM'),
             ('P2o', 'ho_knn', 'cls_knn', 'representation effect vs original descriptor, kNN'),
             ('P3o', 'ho_rf', 'cls_rf', 'representation effect vs original descriptor, bounded RF'),
             ('S1', 'cls_knn', 'cls_svm', 'within CLS: Linear SVM vs kNN'),
             ('S2', 'cls_rf', 'cls_svm', 'within CLS: Linear SVM vs bounded RF')]
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261011
BOOTSTRAP_CHUNK = 500
LABELS = ((0.75, 'predominantly representation'), (0.50, 'mostly representation'),
          (0.25, 'mostly classifier'), (float('-inf'), 'predominantly classifier'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def mcnemar_exact(fixed: int, regressed: int) -> float:
    n = fixed + regressed
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(fixed, regressed) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def label_fraction(value: float | None) -> str | None:
    if value is None:
        return None
    for threshold, name in LABELS:
        if value >= threshold:
            return name
    raise AssertionError('unreachable')


def load_outcomes(path: Path, rows: list, label_to_id: dict) -> dict:
    """Per-clip Top-1/Top-5 correctness and ranked-class count, in row order."""
    lines = path.read_text(encoding='utf-8').splitlines()
    if len(lines) != len(rows):
        raise ValueError(f'{path}: {len(lines)} predictions for {len(rows)} rows')
    top1 = np.zeros(len(rows), bool)
    top5 = np.zeros(len(rows), bool)
    ranked = np.zeros(len(rows), np.int64)
    for i, (line, row) in enumerate(zip(lines, rows)):
        item = json.loads(line)
        if any(item.get(key) != row[key] for key in IDENTITY_KEYS) or item.get('status') != 'ok':
            raise ValueError(f'{path}: identity/order/status mismatch at row {i}')
        ids = item['top5_ids']
        if not 1 <= len(ids) <= 5 or len(set(ids)) != len(ids):
            raise ValueError(f'{path}: invalid ranked IDs at row {i}')
        target = label_to_id[row['gloss']]
        top1[i] = ids[0] == target
        top5[i] = target in ids
        ranked[i] = len(ids)
    return {'top1': top1, 'top5': top5, 'ranked': ranked}


def cell_metrics(outcome: dict, class_ids: np.ndarray, signers: np.ndarray, num_classes: int) -> dict:
    n = len(class_ids)
    counts = np.bincount(class_ids, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError('Every class must have validation clips for the full-vocabulary macro')
    macro = {key: float(np.mean(np.bincount(class_ids, weights=outcome[key].astype(np.float64),
                                            minlength=num_classes) / counts)) for key in ('top1', 'top5')}
    by_signer = {}
    for signer in sorted(set(signers.tolist())):
        mask = signers == signer
        by_signer[signer] = {'count': int(mask.sum()), 'top1_correct': int(outcome['top1'][mask].sum()),
                             'top5_correct': int(outcome['top5'][mask].sum()),
                             'top1': float(outcome['top1'][mask].mean()), 'top5': float(outcome['top5'][mask].mean())}
    return {'count': n, 'top1_correct': int(outcome['top1'].sum()), 'top5_correct': int(outcome['top5'].sum()),
            'top1': float(outcome['top1'].mean()), 'top5': float(outcome['top5'].mean()),
            'macro_top1': macro['top1'], 'macro_top5': macro['top5'],
            'rows_with_fewer_than_5_ranked': int((outcome['ranked'] < 5).sum()), 'by_signer': by_signer}


def paired(base: np.ndarray, other: np.ndarray) -> dict:
    fixed = int((~base & other).sum())
    regressed = int((base & ~other).sum())
    return {'fixed': fixed, 'regressed': regressed, 'net': fixed - regressed,
            'delta_top1_pp': 100 * (fixed - regressed) / len(base), 'mcnemar_exact_p': mcnemar_exact(fixed, regressed)}


def stratified_bootstrap(base: np.ndarray, other: np.ndarray, signers: np.ndarray, rng: np.random.Generator,
                         resamples: int = BOOTSTRAP_RESAMPLES, chunk: int = BOOTSTRAP_CHUNK) -> dict:
    """Percentile CI of the Top-1 difference; clips resampled within each signer."""
    difference = other.astype(np.int64) - base.astype(np.int64)
    strata = [difference[signers == s] for s in sorted(set(signers.tolist()))]
    totals = np.zeros(resamples, np.int64)
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        for values in strata:
            picks = rng.integers(0, len(values), size=(size, len(values)))
            totals[start:start + size] += values[picks].sum(axis=1)
    deltas = 100 * totals / len(difference)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {'ci95_pp': [float(low), float(high)], 'resamples': resamples, 'method': 'signer-stratified clip bootstrap, percentile'}


def decomposition(top1: dict) -> dict:
    """Preregistered estimands in percentage points (A = validation Top-1)."""
    a = {key: 100 * value for key, value in top1.items()}
    result = {'accuracy_pp': a}
    for row in ('hc', 'ho'):
        paths = {}
        for method in METHODS:
            c = SHORT[method]
            gap = a['head'] - a[f'{row}_{c}']
            rep = a[f'cls_{c}'] - a[f'{row}_{c}']
            clf = a['head'] - a[f'cls_{c}']
            fraction = rep / gap if gap else None
            paths[method] = {'total_gap_pp': gap, 'representation_effect_pp': rep, 'classifier_effect_pp': clf,
                             'representation_fraction': fraction, 'classifier_fraction': None if fraction is None else 1 - fraction,
                             'label': label_fraction(fraction)}
        result[row] = paths
    primary = result['hc']['linear_svm']
    labels = {m: result['hc'][m]['label'] for m in METHODS}
    result['primary'] = {**primary, 'path': 'Linear SVM, handcrafted canonical -> CLS -> official head',
                         'robustness_labels_hc': labels,
                         'robustness_agrees_with_primary': all(v == primary['label'] for v in labels.values())}
    return result


def method_evidence(directory: Path, experiment_verification: dict) -> dict:
    state = json.loads((directory / 'state.json').read_text())
    reload = json.loads((directory / 'reload_verification.json').read_text())
    details = json.loads((directory / 'method_details.json').read_text())
    fit = json.loads((directory / 'fit_info.json').read_text())
    if state.get('status') != 'completed' or reload.get('status') != 'verified':
        raise ValueError(f'{directory}: method not completed or reload replay not verified')
    if experiment_verification.get('status') != 'verified':
        raise ValueError(f'{directory.parent}: frozen verifier did not verify')
    evidence = {'state': state['status'], 'reload_verification': reload['status'],
                'model_sha256': reload.get('model_sha256'), 'timings': details['timings'],
                'converged': fit.get('converged'), 'warnings': len(fit.get('warnings', []))}
    if 'n_iter_by_estimator' in fit:
        iterations = fit['n_iter_by_estimator']
        evidence.update(n_iter_max=max(iterations), binary_estimators=len(iterations),
                        binary_estimators_with_warnings=len(fit.get('binary_warnings', [])))
    return evidence


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def track(path: Path, expected: str | None = None) -> Path:
        path = path.resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'SHA256 mismatch: {path}')
        inputs[str(path.relative_to(WORKSPACE)) if path.is_relative_to(WORKSPACE) else str(path)] = digest
        return path

    dataset_dir = args.cls_dataset_dir.resolve()
    document = json.loads(track(dataset_dir / 'dataset.json', args.expected_dataset_sha256).read_text())
    rows_path = track(dataset_dir / 'val_rows.json', document['val']['artifacts']['val_rows.json']['sha256'])
    rows = json.loads(rows_path.read_text())
    labels = document['id_to_label']
    label_to_id = {label: i for i, label in enumerate(labels)}
    class_ids = np.array([label_to_id[row['gloss']] for row in rows], np.int64)
    signers = np.array([row['signer_id'] for row in rows])
    experiments = {**{key: WORKSPACE / value for key, value in HANDCRAFTED.items()},
                   'cls': args.grid_experiment_dir.resolve()}
    outcomes, cells, evidence = {}, {}, {}
    for row_key, experiment in experiments.items():
        state = json.loads(track(experiment / 'state.json').read_text())
        if state.get('status') != 'completed' or state.get('results_verified') is not True:
            raise ValueError(f'{experiment}: runner not completed with verified results')
        verification = json.loads(track(experiment / 'verification.json').read_text())
        for method in METHODS:
            name = f'{row_key}_{SHORT[method]}'
            directory = experiment / method
            outcomes[name] = load_outcomes(track(directory / 'predictions.jsonl'), rows, label_to_id)
            cells[name] = cell_metrics(outcomes[name], class_ids, signers, len(labels))
            saved = json.loads(track(directory / 'metrics.json').read_text())
            checks = {'top1': saved['per_instance']['top1'], 'top5': saved['per_instance']['top5'],
                      'macro_top1': saved['per_class_full_vocabulary']['top1'],
                      'macro_top5': saved['per_class_full_vocabulary']['top5']}
            for key, value in checks.items():
                if abs(cells[name][key] - value) > 1e-12:
                    raise ValueError(f'{name} {key}: recomputed {cells[name][key]} vs saved {value}')
            for key in ('top1', 'top5'):
                if abs(verification['metrics'][method]['per_instance'][key] - cells[name][key]) > 1e-12:
                    raise ValueError(f'{name} {key}: differs from frozen verification.json')
            evidence[name] = method_evidence(directory, verification)
            cells[name].update(representation=REPRESENTATION_NAMES[row_key], classifier=method)
    head_dir = args.head_dir.resolve()
    replay = json.loads(track(args.head_replay_json).read_text())
    if replay.get('status') != 'verified' or replay['splits']['val']['gate_G3']['status'] != 'passed':
        raise ValueError('Official head replay did not pass G3')
    outcomes['head'] = load_outcomes(track(head_dir / 'predictions.jsonl'), rows, label_to_id)
    cells['head'] = cell_metrics(outcomes['head'], class_ids, signers, len(labels))
    saved = json.loads(track(head_dir / 'metrics.json').read_text())
    if abs(saved['per_instance']['top1'] - cells['head']['top1']) > 1e-12 or cells['head']['top1_correct'] != 8340:
        raise ValueError('Official head metrics do not reproduce 8340/10304')
    cells['head'].update(representation=REPRESENTATION_NAMES['cls'], classifier='official PoET linear head')
    head_details = json.loads(track(head_dir / 'method_details.json').read_text())
    evidence['head'] = {'timings': head_details['timings'], 'train_top1': replay['splits']['train']['top1'],
                        'train_top5': replay['splits']['train']['top5'], 'gate_G3': 'passed'}
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    comparisons = []
    for tier, items in (('primary', PRIMARY), ('secondary', SECONDARY)):
        for key, base, other, meaning in items:
            entry = {'id': key, 'tier': tier, 'base': base, 'other': other, 'meaning': meaning,
                     **paired(outcomes[base]['top1'], outcomes[other]['top1'])}
            entry.update(stratified_bootstrap(outcomes[base]['top1'], outcomes[other]['top1'], signers, rng))
            entry['by_signer_delta_top1_pp'] = {
                s: 100 * float(outcomes[other]['top1'][signers == s].mean() - outcomes[base]['top1'][signers == s].mean())
                for s in sorted(set(signers.tolist()))}
            comparisons.append(entry)
    top1 = {name: cell['top1'] for name, cell in cells.items()}
    manifests = {}
    for split, run_path in (('train', 'runs/20260926T202515Z-retrieval-pilot-train-extraction-5911f3b9'),
                            ('val', 'runs/20260926T203453Z-retrieval-pilot-val-extraction-993a38f2')):
        manifest = json.loads(track(WORKSPACE / run_path / 'features/manifest.json').read_text())
        manifests[split] = manifest.get('elapsed_seconds')
    descriptor = json.loads(track(WORKSPACE / 'runs/20261002T160743Z-classical-full-features-3470975b/features/dataset.json').read_text())
    result = {'schema_version': 1, 'status': 'completed', 'created_at_utc': utc(), 'test_used': False,
              'preregistration_sha256': args.preregistration_sha256, 'cells': cells, 'evidence': evidence,
              'comparisons': comparisons, 'decomposition': decomposition(top1),
              'representation_cost_context': {
                  'descriptor_extraction_wall_seconds_50458_clips_8_cpu_workers': descriptor.get('wall_seconds'),
                  'cls_extraction_elapsed_seconds_gpu0': manifests,
                  'scope': 'Both start from cached MediaPipe poses; neither includes video decoding or MediaPipe.'},
              'bootstrap': {'seed': BOOTSTRAP_SEED, 'generator': 'numpy PCG64', 'resamples': BOOTSTRAP_RESAMPLES,
                            'order': 'one generator consumed in comparison order P1..P7, P1o..P3o, S1, S2'},
              'notes': ['McNemar and bootstrap treat clips as exchangeable within signer; 6 validation signers; descriptive only.',
                        'kNN Top-5 counts only voted classes; see rows_with_fewer_than_5_ranked.'],
              'inputs': dict(sorted(inputs.items()))}
    write_json(output / 'analysis.json', result)
    print(json.dumps({'status': 'completed', 'top1': top1, 'primary': result['decomposition']['primary']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cls-dataset-dir', type=Path, required=True)
    parser.add_argument('--expected-dataset-sha256', required=True)
    parser.add_argument('--grid-experiment-dir', type=Path, required=True)
    parser.add_argument('--head-dir', type=Path, required=True)
    parser.add_argument('--head-replay-json', type=Path, required=True)
    parser.add_argument('--preregistration-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
