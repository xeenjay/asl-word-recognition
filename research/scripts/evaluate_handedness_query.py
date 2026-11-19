#!/usr/bin/env python3
"""Arm 0: score handedness-canonicalized validation descriptors with the frozen models.

No model or scaler is refit. Rows of unmirrored signers must reproduce the saved
formal predictions exactly (IDs, kNN neighbours) with scores within 1e-10, which
checks that only the mirrored clips changed. Writes the standard prediction
layout so verify_classical_results.py can independently recompute the metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np

from classical_handedness import VERSION
from classical_models import load_model, predict_topk, transform_features
from evaluate_pose_cache import sha256, verify_file, write_json
from run_classical_baselines import IDENTITY_KEYS, METHODS, record_metrics

WORKSPACE = Path(__file__).resolve().parents[1]
BASELINE = WORKSPACE / 'runs/20261002T161254Z-classical-formal-three-methods-dac0c02f/experiment'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'stage': 'verify_inputs', 'started_at_utc': utc(), 'pid': os.getpid()}
    write_json(output / 'state.json', state)
    try:
        experiment = args.experiment_dir.resolve()
        plan = json.loads((experiment / 'plan.json').read_text())
        if plan.get('purpose') != 'formal_train_validation' or plan.get('test_used'):
            raise ValueError('Baseline must be the formal train/validation experiment')
        for name, digest in plan['prepared_sha256'].items():
            verify_file(experiment / name, digest, 'baseline prepared input')
        baseline_rows = json.loads((experiment / 'query_rows.json').read_text())
        labels = json.loads((experiment / 'dataset.json').read_text())['id_to_label']

        features = args.feature_dir.resolve()
        document = json.loads((features / 'dataset.json').read_text())
        if document.get('variant') != VERSION or document.get('status') != 'completed' or document.get('test_used'):
            raise ValueError('Feature directory is not a completed handedness-canonical dataset')
        for row in document['val']['artifacts'].values():
            verify_file(features / row['path'], row['sha256'], 'handedness validation artifact')
        rows = json.loads((features / 'val_rows.json').read_text())
        if rows != baseline_rows:
            raise ValueError('Validation rows differ from the baseline query rows')
        decisions = json.loads((features / 'handedness_decisions.json').read_text())['decisions']['val']
        mirrored = np.array([bool(decisions[r['signer_id']]['mirror']) for r in rows])
        X = np.load(features / 'val_features.npy', allow_pickle=False)

        scaler = joblib.load(experiment / 'scaler.joblib')
        Q = transform_features(scaler, X)
        reference = np.load(experiment / 'query_scaled.npy', allow_pickle=False)
        if Q[~mirrored].tobytes() != reference[~mirrored].tobytes():
            raise ValueError('Unmirrored scaled rows differ from the baseline query matrix')
        if Q[mirrored].tobytes() == reference[mirrored].tobytes() and mirrored.any():
            raise ValueError('Mirrored rows are unchanged; canonicalization was not applied')

        write_json(output / 'dataset.json', document)
        write_json(output / 'plan.json', {
            'schema_version': 1, 'purpose': 'query_only_handedness_diagnostic', 'methods': list(METHODS),
            'parameters': plan['parameters'], 'feature_spec': document['feature_spec'],
            'baseline_experiment': str(experiment), 'baseline_plan_sha256': sha256(experiment / 'plan.json'),
            'feature_dataset_sha256': sha256(features / 'dataset.json'), 'scaler': 'baseline train scaler (not refit)',
            'models': 'baseline models (not refit)', 'test_used': False, 'created_at_utc': utc(),
            'mirrored_validation_rows': int(mirrored.sum()),
            'source_sha256': {name: sha256(WORKSPACE / name) for name in (
                'scripts/evaluate_handedness_query.py', 'scripts/classical_handedness.py',
                'scripts/classical_models.py', 'scripts/run_classical_baselines.py',
                'scripts/verify_classical_results.py')}})
        summary = {}
        for method in METHODS:
            state.update(stage=f'predict_{method}'); write_json(output / 'state.json', state)
            method_state = json.loads((experiment / method / 'state.json').read_text())
            model, saved_scaler = load_model(experiment / method / 'model.joblib',
                                             expected_sha256=method_state['model_sha256'])
            if (saved_scaler is None or not np.array_equal(saved_scaler.mean_, scaler.mean_)
                    or not np.array_equal(saved_scaler.scale_, scaler.scale_)):
                raise ValueError(f'{method} saved scaler differs from the experiment scaler')
            stamp = time.perf_counter()
            prediction = predict_topk(model, Q, top_k=5, device=args.device if method == 'knn' else 'cpu',
                                      batch_size=128, reference_chunk_size=8192)
            seconds = time.perf_counter() - stamp
            baseline = read_jsonl(experiment / method / 'predictions.jsonl')
            unchanged = 0
            for i, old in enumerate(baseline):
                if any(old[k] != rows[i][k] for k in IDENTITY_KEYS):
                    raise ValueError(f'{method}: baseline prediction order differs at {i}')
                if mirrored[i]:
                    continue
                valid = prediction['valid_mask'][i]
                ids = prediction['topk_indices'][i][valid].astype(int).tolist()
                scores = prediction['topk_scores'][i][valid].astype(float)
                if ids != old['top5_ids'] or not np.allclose(scores, old['top5_scores'], rtol=1e-10, atol=1e-10):
                    raise ValueError(f'{method}: unmirrored row {i} does not replay the baseline prediction')
                if method == 'knn' and prediction['neighbor_indices'][i].astype(int).tolist() != old['neighbor_train_rows']:
                    raise ValueError(f'knn: unmirrored row {i} neighbours differ')
                unchanged += 1
            (output / method).mkdir()
            fit_info = dict(json.loads((experiment / method / 'fit_info.json').read_text()), refit=False)
            metrics = record_metrics(output / method, rows, labels, prediction, fit_info,
                                     {'predict_seconds': seconds, 'query_count': len(Q),
                                      'scope': 'Frozen baseline model; canonicalized validation descriptors only.'})
            summary[method] = {'top1': metrics['per_instance']['top1'], 'top5': metrics['per_instance']['top5'],
                               'unmirrored_rows_replayed_exactly': unchanged,
                               'mirrored_rows': int(mirrored.sum()), 'model_sha256': method_state['model_sha256']}
            print(json.dumps({method: summary[method]}), flush=True)
        state['stage'] = 'independent_verification'; write_json(output / 'state.json', state)
        subprocess.run([sys.executable, str(WORKSPACE / 'scripts/verify_classical_results.py'),
                        '--experiment-dir', str(output)], check=True)
        write_json(output / 'summary.json', {'status': 'completed', 'methods': summary, 'test_used': False})
        state.update(status='completed', stage='complete', ended_at_utc=utc())
        write_json(output / 'state.json', state)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error), ended_at_utc=utc())
        write_json(output / 'state.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--experiment-dir', type=Path, default=BASELINE)
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args())
