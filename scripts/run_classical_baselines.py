#!/usr/bin/env python3
"""One bounded kNN/SVM/RF experiment, with train-only scaling and replay checks.

Optionally use run_logged.py to retain a local run record. Output directories are never reused.
Each classifier is a separate child; a failed child does not erase other results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from classical_models import (DEFAULT_PARAMETERS, fit_train_scaler, transform_features,
                              fit_classifier, predict_topk, save_model, load_model)
from evaluate_pose_cache import MetricAccumulator, sha256, verify_file, write_json

WORKSPACE = Path(__file__).resolve().parents[1]
METHODS = ('knn', 'linear_svm', 'random_forest')
IDENTITY_KEYS = ('index', 'gloss', 'signer_id', 'split', 'source_path', 'video_sha256')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def parameters():
    result = {name: dict(DEFAULT_PARAMETERS[name]) for name in METHODS}
    result['linear_svm'].update(parallel_ovr=True, n_jobs=8)
    result['random_forest'].update(n_jobs=8)
    return result


def record_metrics(out, rows, labels, predictions, fit_info, timings):
    acc = MetricAccumulator(dict(enumerate(labels)))
    confusion, signer = Counter(), {}
    with (out / 'predictions.jsonl').open('x') as stream:
        for i, row in enumerate(rows):
            valid = predictions['valid_mask'][i]
            ids = predictions['topk_indices'][i][valid].astype(int).tolist()
            scores = predictions['topk_scores'][i][valid].astype(float).tolist()
            if not ids:
                raise ValueError('No valid predicted class for a sample')
            acc.add(row['gloss'], ids)
            result = {key: row[key] for key in IDENTITY_KEYS}
            result.update(status='ok', top5_ids=ids, top5_scores=scores,
                          score_kind=predictions['score_kind'])
            if 'neighbor_indices' in predictions:
                result['neighbor_train_rows'] = predictions['neighbor_indices'][i].astype(int).tolist()
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            correct = labels[ids[0]] == row['gloss']
            entry = signer.setdefault(row['signer_id'], {'count': 0, 'correct': 0})
            entry['count'] += 1
            entry['correct'] += int(correct)
            if not correct:
                confusion[(row['gloss'], labels[ids[0]])] += 1
    metrics, per_class = acc.summary()
    write_json(out / 'metrics.json', metrics)
    write_json(out / 'per_class.json', per_class)
    write_json(out / 'method_details.json', {'fit_info': fit_info, 'timings': timings,
                                           'score_kind': predictions['score_kind'], 'split': rows[0]['split']})
    write_json(out / 'error_analysis.json', {
        'top_confusions': [{'true': k[0], 'predicted': k[1], 'count': v} for k, v in confusion.most_common(40)],
        'by_signer': {k: {**v, 'top1': v['correct'] / v['count']} for k, v in signer.items()},
        'interpretation': 'Development validation diagnostics, not unseen test performance.'})
    return metrics


def run_method(experiment, method, device):
    import joblib
    out = experiment / method
    out.mkdir(exist_ok=False)
    plan = json.loads((experiment / 'plan.json').read_text())
    state = {'status': 'running', 'method': method, 'stage': 'loading', 'started_at_utc': timestamp(), 'pid': os.getpid()}
    write_json(out / 'state.json', state)
    try:
        for name, digest in plan['source_sha256'].items():
            verify_file(WORKSPACE / name, digest, 'frozen source')
        for name, digest in plan['prepared_sha256'].items():
            verify_file(experiment / name, digest, 'shared prepared input')
        X = np.load(experiment / 'train_scaled.npy', mmap_mode='r')
        y = np.load(experiment / 'train_labels.npy', allow_pickle=False)
        Q = np.load(experiment / 'query_scaled.npy', mmap_mode='r')
        rows = json.loads((experiment / 'query_rows.json').read_text())
        dataset = json.loads((experiment / 'dataset.json').read_text())
        scaler = joblib.load(experiment / 'scaler.joblib')
        state['stage'] = 'fitting'; write_json(out / 'state.json', state)
        stamp = time.perf_counter()
        model = fit_classifier(method, X, y, len(dataset['id_to_label']), **plan['parameters'][method])
        fit_seconds = time.perf_counter() - stamp
        write_json(out / 'fit_info.json', model.fit_info)
        digest = save_model(model, out / 'model.joblib', scaler=scaler)
        state.update(stage='predicting', fit_seconds=fit_seconds); write_json(out / 'state.json', state)
        stamp = time.perf_counter()
        prediction = predict_topk(model, Q, top_k=5, device=device if method == 'knn' else 'cpu',
                                  batch_size=128, reference_chunk_size=8192)
        prediction_seconds = time.perf_counter() - stamp
        state['stage'] = 'reload_replay'; write_json(out / 'state.json', state)
        reloaded, scaler2 = load_model(out / 'model.joblib', expected_sha256=digest)
        if not np.array_equal(scaler.mean_, scaler2.mean_) or not np.array_equal(scaler.scale_, scaler2.scale_):
            raise ValueError('Reloaded scaler differs')
        stamp = time.perf_counter()
        repeated = predict_topk(reloaded, Q, top_k=5, device=device if method == 'knn' else 'cpu',
                               batch_size=128, reference_chunk_size=8192)
        replay_seconds = time.perf_counter() - stamp
        for name in ('topk_indices', 'valid_mask'):
            if not np.array_equal(prediction[name], repeated[name]):
                raise ValueError(f'Reloaded predictions disagree: {name}')
        mask = prediction['valid_mask']
        np.testing.assert_allclose(prediction['topk_scores'][mask], repeated['topk_scores'][mask], rtol=1e-10, atol=1e-10)
        timings = {'fit_seconds': fit_seconds, 'predict_seconds': prediction_seconds,
                   'replay_seconds': replay_seconds, 'query_count': len(Q),
                   'predict_ms_per_clip': prediction_seconds / len(Q) * 1000,
                   'model_bytes': (out / 'model.joblib').stat().st_size,
                   'scope': 'Cached standardized descriptors; excludes video/MediaPipe and shared descriptor/scaler costs.'}
        metrics = record_metrics(out, rows, dataset['id_to_label'], prediction, model.fit_info, timings)
        write_json(out / 'reload_verification.json', {'status': 'verified', 'rows': len(Q),
            'topk_ids_exact': True, 'score_rtol': 1e-10, 'score_atol': 1e-10, 'model_sha256': digest})
        state.update(status='completed', stage='complete', ended_at_utc=timestamp(),
                     top1=metrics['per_instance']['top1'], model_sha256=digest)
        write_json(out / 'state.json', state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error), ended_at_utc=timestamp())
        write_json(out / 'state.json', state)
        raise


def run(args):
    import joblib
    import sklearn
    feature_dir = args.feature_dir.resolve()
    feature_doc = json.loads((feature_dir / 'dataset.json').read_text())
    if feature_doc['status'] != 'completed' or feature_doc['test_used']:
        raise ValueError('Feature dataset is not completed train/validation data')
    if not args.smoke and (feature_doc['purpose'] != 'formal_train_validation_features'
                          or feature_doc['train']['count'] != 40154 or feature_doc['val']['count'] != 10304):
        raise ValueError('Formal run requires the full original train and validation splits')
    for split in ('train',) if args.smoke else ('train', 'val'):
        for row in feature_doc[split]['artifacts'].values():
            verify_file(feature_dir / row['path'], row['sha256'], 'feature dataset artifact')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'stage': 'prepare', 'started_at_utc': timestamp(), 'pid': os.getpid()}
    write_json(output / 'state.json', state)
    X = np.load(feature_dir / 'train_features.npy', allow_pickle=False)
    y = np.load(feature_dir / 'train_labels.npy', allow_pickle=False)
    dataset = dict(feature_doc)
    if args.smoke:
        # Engineering only: fit the small train slice and predict its first rows.
        # This is not a validation estimate or a hyperparameter selection step.
        unique, y = np.unique(y, return_inverse=True)
        dataset['id_to_label'] = [feature_doc['id_to_label'][int(i)] for i in unique]
        Q = X[:32]
        rows = json.loads((feature_dir / 'train_rows.json').read_text())[:32]
    else:
        Q = np.load(feature_dir / 'val_features.npy', allow_pickle=False)
        rows = json.loads((feature_dir / 'val_rows.json').read_text())
    stamp = time.perf_counter()
    scaler = fit_train_scaler(X, split='train')
    np.save(output / 'train_scaled.npy', transform_features(scaler, X))
    np.save(output / 'query_scaled.npy', transform_features(scaler, Q))
    np.save(output / 'train_labels.npy', np.asarray(y, dtype=np.int64))
    joblib.dump(scaler, output / 'scaler.joblib', compress=0)
    write_json(output / 'query_rows.json', rows)
    write_json(output / 'dataset.json', dataset)
    names = ('train_scaled.npy', 'query_scaled.npy', 'train_labels.npy', 'scaler.joblib', 'query_rows.json', 'dataset.json')
    source_names = ('scripts/run_classical_baselines.py', 'scripts/classical_models.py',
                    'scripts/classical_features.py', 'scripts/verify_classical_results.py',
                    'scripts/evaluate_pose_cache.py')
    plan = {'schema_version': 1, 'purpose': 'engineering_smoke' if args.smoke else 'formal_train_validation',
            'methods': list(METHODS), 'parameters': parameters(), 'created_at_utc': timestamp(),
            'feature_dataset_path': str(feature_dir / 'dataset.json'),
            'feature_dataset_sha256': sha256(feature_dir / 'dataset.json'),
            'feature_spec': feature_doc['feature_spec'], 'scaler_fit_split': 'train',
            'scaler_seconds': time.perf_counter() - stamp, 'test_used': False,
            'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
            'query_device_knn': args.device, 'prepared_sha256': {n: sha256(output / n) for n in names},
            'source_sha256': {n: sha256(WORKSPACE / n) for n in source_names},
            'no_parameter_search': True, 'seed': 20261002}
    write_json(output / 'plan.json', plan)
    processes = []
    # Explicit per-method directories and the enclosing new run guard prevent
    # an accidental replay from overwriting successful siblings.
    try:
        for method in METHODS:
            log = (output / f'{method}.log').open('x')
            command = [sys.executable, str(Path(__file__).resolve()), '--single-method', method,
                       '--output-dir', str(output), '--device', args.device]
            try:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            except BaseException:
                log.close()
                raise
            processes.append((method, child, log))
            state['stage'] = 'dispatching'
            state.setdefault('children', {})[method] = {'pid': child.pid, 'command': command}
            write_json(output / 'state.json', state)
    except BaseException:
        # Account for successfully dispatched siblings before reporting a
        # launch error; never forget a live classifier or blindly relaunch it.
        for method, child, log in processes:
            code = child.wait()
            log.close()
            state.setdefault('children', {})[method] = {'pid': child.pid, 'returncode': code}
        write_json(output / 'state.json', state)
        raise
    state.update(stage='classifiers')
    write_json(output / 'state.json', state)
    errors = []
    for method, child, log in processes:
        code = child.wait()
        log.close()
        state['children'][method]['returncode'] = code
        write_json(output / 'state.json', state)
        if code:
            errors.append({'method': method, 'returncode': code})
    if errors:
        state.update(status='failed', errors=errors, ended_at_utc=timestamp())
        write_json(output / 'state.json', state)
        raise RuntimeError(f'Classifiers failed; preserve sibling results: {errors}')
    summaries = {method: json.loads((output / method / 'metrics.json').read_text()) for method in METHODS}
    write_json(output / 'comparison.json', summaries)
    if not args.smoke:
        state['stage'] = 'independent_verification'; write_json(output / 'state.json', state)
        subprocess.run([sys.executable, str(WORKSPACE / 'scripts/verify_classical_results.py'),
                        '--experiment-dir', str(output)], check=True)
    state.update(status='completed', stage='complete', ended_at_utc=timestamp(),
                 results_verified=not args.smoke, purpose=plan['purpose'])
    write_json(output / 'state.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--single-method', choices=METHODS)
    args = parser.parse_args()
    if args.single_method:
        run_method(args.output_dir.resolve(), args.single_method, args.device)
    elif args.feature_dir is None:
        parser.error('--feature-dir is required')
    else:
        try:
            run(args)
        except BaseException as error:
            state_path = args.output_dir.resolve() / 'state.json'
            # A rejected replay must not overwrite an earlier completed run.
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get('pid') == os.getpid() and state.get('status') == 'running':
                    state.update(status='failed', error_type=type(error).__name__, error=str(error),
                                 ended_at_utc=timestamp())
                    write_json(state_path, state)
            raise
