#!/usr/bin/env python3
"""Replay the official PoET linear head on stored CLS vectors (gate G3 + reference row).

The official checkpoint's classifier is MLP1, a single nn.Linear(128, 2731). This
script reads only its weight and bias with the audited, SHA-pinned loader and
computes logits = CLS @ W.T + b in float64 NumPy on CPU. No encoder forward pass,
no gradient, no training and no test data.

Gate G3 (preregistered): on validation the replay must give exactly 8340 Top-1
correct, top-5 IDs equal to the stored extraction output except rows whose stored
adjacent top-5 logits differ by < 1e-4 (at most 10), and max |logit difference|
<= 1e-3. Validation predictions are written with the frozen classical runner's own
record_metrics, so the reference row shares the classical evaluation path. The
train replay is descriptive only (caveat 1).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from evaluate_pose_cache import sha256, verify_file, write_json
from prepare_cls_representation_dataset import (EXTRACTIONS, FEATURE_DIM, NUM_CLASSES,
                                                OFFICIAL_CHECKPOINT_SHA256)

WORKSPACE = Path(__file__).resolve().parents[1]
CHECKPOINT = WORKSPACE / 'assets/checkpoints/ASL_Citizen_PoET.pth'
EXPECTED_VAL_TOP1 = 8340
NEAR_TIE = 1e-4
MAX_NEAR_TIE_ROWS = 10
MAX_LOGIT_DIFF = 1e-3


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_head(path: Path = CHECKPOINT):
    verify_file(path, OFFICIAL_CHECKPOINT_SHA256, 'official PoET checkpoint')
    from _poet_runtime import load_checkpoint
    checkpoint = load_checkpoint(path)
    parameters = checkpoint['model_parameters']
    if (parameters.get('mlp_classifier') != 'MLP1' or parameters.get('transformer_hidden_size') != FEATURE_DIM
            or checkpoint.get('num_classes') != NUM_CLASSES):
        raise ValueError('Official head is not MLP1 nn.Linear(128, 2731)')
    state = checkpoint['model_state_dict']
    head_keys = sorted(k for k in state if k.startswith('mlp_classifier.'))
    if head_keys != ['mlp_classifier.L.bias', 'mlp_classifier.L.weight']:
        raise ValueError(f'Unexpected head parameters: {head_keys}')
    weight = state['mlp_classifier.L.weight'].detach().cpu().numpy().astype(np.float64)
    bias = state['mlp_classifier.L.bias'].detach().cpu().numpy().astype(np.float64)
    if weight.shape != (NUM_CLASSES, FEATURE_DIM) or bias.shape != (NUM_CLASSES,):
        raise ValueError('Head parameter shapes differ')
    if not (np.isfinite(weight).all() and np.isfinite(bias).all()):
        raise ValueError('Nonfinite head parameters')
    return weight, bias, {'mlp_classifier': 'MLP1', 'weight_shape': list(weight.shape),
                          'checkpoint_sha256': OFFICIAL_CHECKPOINT_SHA256}


def head_topk(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, k: int = 5,
              batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Float64 logits; stable descending order, ties by lower class ID."""
    ids = np.empty((len(features), k), np.int64)
    values = np.empty((len(features), k), np.float64)
    for start in range(0, len(features), batch):
        logits = features[start:start + batch].astype(np.float64) @ weight.T + bias
        order = np.argsort(-logits, axis=1, kind='stable')[:, :k]
        ids[start:start + batch] = order
        values[start:start + batch] = np.take_along_axis(logits, order, axis=1)
    return ids, values


def compare_with_stored(ids6: np.ndarray, values6: np.ndarray, stored_ids: np.ndarray,
                        stored_logits: np.ndarray) -> dict:
    """Compare replayed top-5 with the stored top-5.

    A mismatched row is tolerated only where adjacent logits nearly tie, either
    among the stored five or among the replayed top six (the 5th/6th boundary).
    """
    ids, values = ids6[:, :5], values6[:, :5]
    stored = stored_logits.astype(np.float64)
    mismatch = (ids != stored_ids).any(axis=1)
    near_tie = ((np.diff(-stored, axis=1) < NEAR_TIE).any(axis=1)
                | (np.diff(-values6, axis=1) < NEAR_TIE).any(axis=1))
    unexplained = np.flatnonzero(mismatch & ~near_tie).astype(int).tolist()
    same_rows = ~mismatch
    max_diff = float(np.max(np.abs(values[same_rows] - stored[same_rows]))) if same_rows.any() else float('inf')
    return {'rows': int(len(ids)), 'top5_mismatch_rows': np.flatnonzero(mismatch).astype(int).tolist(),
            'top5_mismatch_unexplained_rows': unexplained,
            'top1_mismatch_rows': int((ids[:, 0] != stored_ids[:, 0]).sum()),
            'near_tie_rows': int(near_tie.sum()), 'position_match_fraction': float((ids == stored_ids).mean()),
            'max_abs_logit_diff_on_matching_rows': max_diff}


def gate_g3(val_check: dict, top1_correct: int) -> dict:
    reasons = []
    if top1_correct != EXPECTED_VAL_TOP1:
        reasons.append(f'val Top-1 correct {top1_correct} != {EXPECTED_VAL_TOP1}')
    if val_check['top5_mismatch_unexplained_rows']:
        reasons.append(f"{len(val_check['top5_mismatch_unexplained_rows'])} top-5 mismatches without a near tie")
    if len(val_check['top5_mismatch_rows']) > MAX_NEAR_TIE_ROWS:
        reasons.append(f"{len(val_check['top5_mismatch_rows'])} mismatched rows > {MAX_NEAR_TIE_ROWS}")
    if not val_check['max_abs_logit_diff_on_matching_rows'] <= MAX_LOGIT_DIFF:
        reasons.append(f"max logit diff {val_check['max_abs_logit_diff_on_matching_rows']} > {MAX_LOGIT_DIFF}")
    return {'status': 'passed' if not reasons else 'failed', 'reasons': reasons,
            'criteria': {'val_top1_correct': EXPECTED_VAL_TOP1, 'near_tie': NEAR_TIE,
                         'max_mismatch_rows': MAX_NEAR_TIE_ROWS, 'max_abs_logit_diff': MAX_LOGIT_DIFF}}


def run(args):
    from run_classical_baselines import record_metrics
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'stage': 'load', 'started_at_utc': utc()}
    write_json(output / 'state.json', state)
    try:
        dataset_dir = args.dataset_dir.resolve()
        verify_file(dataset_dir / 'dataset.json', args.expected_dataset_sha256, 'CLS feature dataset')
        document = json.loads((dataset_dir / 'dataset.json').read_text())
        for split in ('train', 'val'):
            for row in document[split]['artifacts'].values():
                verify_file(dataset_dir / row['path'], row['sha256'], f'CLS {split} artifact')
        labels = document['id_to_label']
        weight, bias, head_info = load_head()
        result = {'head': head_info, 'dataset_sha256': args.expected_dataset_sha256, 'splits': {}}
        for split in ('val', 'train'):
            state['stage'] = f'replay_{split}'; write_json(output / 'state.json', state)
            features = np.load(dataset_dir / f'{split}_features.npy', allow_pickle=False)
            targets = np.load(dataset_dir / f'{split}_labels.npy', allow_pickle=False)
            pinned = EXTRACTIONS[split]
            npz = WORKSPACE / pinned['run'] / 'features/features.npz'
            verify_file(npz, pinned['features_sha256'], f'{split} extraction features.npz')
            with np.load(npz, allow_pickle=False) as archive:
                stored_ids, stored_logits = archive['top5_class_ids'], archive['top5_logits']
            stamp = time.perf_counter()
            ids6, values6 = head_topk(features, weight, bias, k=6)
            seconds = time.perf_counter() - stamp
            ids, values = ids6[:, :5], values6[:, :5]
            check = compare_with_stored(ids6, values6, stored_ids, stored_logits)
            top1 = int((ids[:, 0] == targets).sum())
            top5 = int((ids == targets[:, None]).any(axis=1).sum())
            entry = {**check, 'top1_correct': top1, 'top5_correct': top5, 'top1': top1 / len(ids),
                     'top5': top5 / len(ids), 'replay_seconds': seconds,
                     'stored_top1_correct': int((stored_ids[:, 0] == targets).sum())}
            if split == 'val':
                gate = gate_g3(check, top1)
                entry['gate_G3'] = gate
                if gate['status'] != 'passed':
                    result['splits'][split] = entry
                    write_json(output / 'head_replay.json', result)
                    raise ValueError(f"G3 failed: {gate['reasons']}")
                rows = json.loads((dataset_dir / 'val_rows.json').read_text())
                head_dir = output / 'head'
                head_dir.mkdir()
                predictions = {'topk_indices': ids, 'topk_scores': values,
                               'valid_mask': np.ones(ids.shape, bool), 'score_kind': 'official_linear_head_logit'}
                timings = {'fit_seconds': None, 'predict_seconds': seconds, 'query_count': len(ids),
                           'predict_ms_per_clip': seconds / len(ids) * 1000,
                           'model_bytes': int(weight.size * 4 + bias.size * 4),
                           'scope': ('Float64 CPU replay of the 128x2731 linear head on cached CLS vectors; '
                                     'excludes the encoder forward pass, video and MediaPipe.')}
                fit_info = {'trained': 'jointly with the encoder by the PoET authors; not refit here',
                            **head_info}
                metrics = record_metrics(head_dir, rows, labels, predictions, fit_info, timings)
                entry['metrics_top1'] = metrics['per_instance']['top1']
            else:
                entry['use'] = 'descriptive only (caveat 1): separability of the encoder on its own training clips'
            result['splits'][split] = entry
        result.update(status='verified', completed_at_utc=utc(), test_used=False, training_steps=0)
        write_json(output / 'head_replay.json', result)
        state.update(status='completed', stage='complete', ended_at_utc=utc(),
                     val_top1=result['splits']['val']['top1'], train_top1=result['splits']['train']['top1'])
        write_json(output / 'state.json', state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error), ended_at_utc=utc())
        write_json(output / 'state.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--expected-dataset-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
