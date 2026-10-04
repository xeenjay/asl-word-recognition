#!/usr/bin/env python3
"""Audit the existing raw pose cache and build fixed classical descriptors.

No video download, landmark extraction, PoET weights or test data are used.
All formal train/validation rows are retained, including missing-hand videos.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from classical_features import extract_features, feature_spec, FEATURE_DIM
from evaluate_pose_cache import (OFFICIAL_LABEL_MAP_SHA256, audit_provenance,
                                 validate_cache_index, verify_file, sha256,
                                 referenced_path, write_json)

WORKSPACE = Path(__file__).resolve().parents[1]
IDENTITY_KEYS = ('index', 'gloss', 'signer_id', 'split', 'source_path', 'video_sha256')


def load_labels(checkpoint_dir):
    values = {}
    for name, digest in OFFICIAL_LABEL_MAP_SHA256.items():
        path = checkpoint_dir / name
        verify_file(path, digest, name)
        # Only unpickle the pinned, locally audited official mapping.
        values[name] = pickle.loads(path.read_bytes())
    inverse, forward = values['d_LId2Label.pkl'], values['d_Label2LId.pkl']
    if set(inverse) != set(range(2731)) or forward != {v: k for k, v in inverse.items()}:
        raise ValueError('Official vocabulary is not a bijection of 2731 classes')
    return inverse, forward


def audit_dataset(args):
    inverse, forward = load_labels(args.checkpoint_dir.resolve())
    cache = {
        "train": (args.train_cache_manifest, args.train_cache_sha256, 40154),
        "val": (args.val_cache_manifest, args.val_cache_sha256, 10304),
    }
    cohorts = {}
    for split, (manifest, digest, count) in cache.items():
        path = manifest.resolve()
        verify_file(path, digest, f'{split} pinned cache index')
        index = json.loads(path.read_text())
        actual_split = validate_cache_index(index, forward, allow_train_smoke=True)
        provenance = audit_provenance(path, index, index['pipeline']['preprocessing'],
                                     forward, split, allow_subset_val=False)
        if (actual_split != split or len(index['samples']) != count
                or not provenance['complete_official_split']
                or {x['gloss'] for x in index['samples']} != set(forward)
                or any(x['status'] not in ('ok', 'reused') for x in index['samples'])):
            raise ValueError(f'Incomplete or failed {split} cohort')
        cohorts[split] = {'index': index, 'provenance': provenance, 'path': path}
    for key in ('signer_id', 'video_sha256', 'source_path'):
        if ({x[key] for x in cohorts['train']['index']['samples']}
                & {x[key] for x in cohorts['val']['index']['samples']}):
            raise ValueError(f'Train and validation overlap: {key}')
    return inverse, forward, cohorts


def extract_one(task):
    item, index_path, pipeline_hash = task
    manifest_path = referenced_path(Path(index_path).parent, item['cache_manifest_path'])
    pose_path = referenced_path(Path(index_path).parent, item['pose_path'])
    if manifest_path.name != 'manifest.json' or pose_path != manifest_path.parent / 'raw_pose.npz':
        raise ValueError('Raw cache path mismatch')
    metadata = json.loads(manifest_path.read_text())
    if (metadata.get('status') != 'ok' or metadata.get('schema_version') != 1
            or metadata['identity'] != item['identity'] or metadata['pipeline_sha256'] != pipeline_hash):
        raise ValueError(f'Raw cache identity mismatch: {item["index"]}')
    artifact = metadata['artifacts']['raw_pose.npz']
    if artifact['sha256'] != item['pose_sha256'] or pose_path.stat().st_size != artifact['bytes']:
        raise ValueError('Raw cache index and manifest disagree')
    verify_file(pose_path, item['pose_sha256'], 'raw pose')
    with np.load(pose_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    result = extract_features(raw)
    if result.shape != (FEATURE_DIM,) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError('Invalid feature vector')
    missing_hands = not (raw['left_hand_present'].any() or raw['right_hand_present'].any())
    return result, bool(missing_hands), len(raw['body_pose'])


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    state = {'status': 'running', 'stage': 'audit', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'purpose': 'engineering_smoke' if args.smoke else 'formal_train_validation_features'}
    write_json(output / 'state.json', state)
    try:
        inverse, forward, cohorts = audit_dataset(args)
        document = {'schema_version': 1, 'status': 'running', 'id_to_label': [inverse[i] for i in range(2731)],
                    'label_map_sha256': OFFICIAL_LABEL_MAP_SHA256, 'feature_spec': feature_spec(),
                    'test_used': False, 'poet_weights_loaded': False, 'purpose': state['purpose'],
                    'source_sha256': {name: sha256(WORKSPACE / name) for name in
                                     ('scripts/prepare_classical_dataset.py', 'scripts/classical_features.py',
                                      'scripts/evaluate_pose_cache.py')},
                    'numpy_version': np.__version__, 'workers': args.workers}
        # Smoke uses a fixed first 160 TRAIN rows; no validation model selection.
        splits = ['train'] if args.smoke else ['train', 'val']
        for split in splits:
            cohort = cohorts[split]
            rows = cohort['index']['samples'][:160] if args.smoke else cohort['index']['samples']
            state.update(stage=f'extract_{split}', processed=0, total=len(rows))
            write_json(output / 'state.json', state)
            matrix = np.lib.format.open_memmap(output / f'{split}_features.npy', mode='w+',
                                              dtype=np.float32, shape=(len(rows), FEATURE_DIM))
            tasks = [(row, str(cohort['path']), cohort['index']['pipeline_sha256']) for row in rows]
            absent, frame_counts = [], []
            stamp = time.perf_counter()
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for n, (vector, missing, frames) in enumerate(pool.map(extract_one, tasks, chunksize=16)):
                    matrix[n] = vector
                    frame_counts.append(frames)
                    if missing:
                        absent.append(rows[n]['index'])
                    if (n + 1) % 1000 == 0 or n + 1 == len(rows):
                        matrix.flush()
                        state.update(processed=n + 1, elapsed_seconds=time.perf_counter() - stamp)
                        write_json(output / 'state.json', state)
                        print(f'Extracted {split} {n + 1}/{len(rows)} in {state["elapsed_seconds"]:.1f}s', flush=True)
            matrix.flush()
            del matrix
            np.save(output / f'{split}_labels.npy', np.array([forward[row['gloss']] for row in rows], dtype=np.int64))
            records = [{key: row[key] for key in IDENTITY_KEYS} for row in rows]
            write_json(output / f'{split}_rows.json', records)
            document[split] = {'cache_manifest_path': str(cohort['path']), 'sha256': sha256(cohort['path']),
                               'count': len(rows), 'provenance': cohort['provenance'],
                               'no_hands_indices': absent, 'frame_count_min': min(frame_counts),
                               'frame_count_max': max(frame_counts), 'feature_seconds': time.perf_counter() - stamp,
                               'artifacts': {name: {'path': name, 'sha256': sha256(output / name),
                                                    'size_bytes': (output / name).stat().st_size}
                                             for name in (f'{split}_features.npy', f'{split}_labels.npy', f'{split}_rows.json')}}
            write_json(output / 'dataset.json', document)
        document.update(status='completed', wall_seconds=time.perf_counter() - started)
        write_json(output / 'dataset.json', document)
        state.update(status='completed', stage='complete', wall_seconds=document['wall_seconds'])
        write_json(output / 'state.json', state)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(output / 'state.json', state)
        raise
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-cache-manifest', type=Path, required=True)
    parser.add_argument('--train-cache-sha256', required=True)
    parser.add_argument('--val-cache-manifest', type=Path, required=True)
    parser.add_argument('--val-cache-sha256', required=True)
    parser.add_argument('--checkpoint-dir', type=Path, default=WORKSPACE / 'assets/checkpoints')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be 1..16')
    run(args)
