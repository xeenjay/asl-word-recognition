#!/usr/bin/env python3
"""Build handedness-canonicalized classical descriptors from the frozen v1 cohort.

Signer decisions use only the original descriptors (see classical_handedness).
Every train/validation clip is recomputed from its verified raw pose cache with
the unchanged extractor: unmirrored rows must equal the frozen features bit for
bit, and mirrored rows must equal the descriptor-level signed permutation bit for
bit. No gloss is read for the decision; no model, PoET weight or test data is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np

from classical_features import FEATURE_DIM, extract_features
from classical_handedness import (VERSION, canonicalize_descriptors, mirror_descriptors,
                                  mirror_raw_pose, rule_spec, signer_decisions)
from evaluate_pose_cache import referenced_path, sha256, verify_file, write_json
from prepare_classical_dataset import IDENTITY_KEYS, audit_dataset

WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE_DIR = WORKSPACE / 'runs/20261002T160743Z-classical-full-features-3470975b/features'
SOURCE_SHA256 = '3a9ef620bb5b3593375f831ada5b191fe66b9b1bafefe9aba9ff4e075daca83f'
COUNTS = {'train': 40154, 'val': 10304}


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_verified_raw(item, index_path, pipeline_hash):
    """Same identity/SHA checks as prepare_classical_dataset.extract_one."""
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
        return {key: archive[key] for key in archive.files}


def build_one(task):
    item, index_path, pipeline_hash, mirror = task
    raw = load_verified_raw(item, index_path, pipeline_hash)
    original = extract_features(raw)
    mirrored = extract_features(mirror_raw_pose(raw)) + np.float32(0) if mirror else None
    dtypes = sorted({str(raw[k].dtype) for k in ('body_pose', 'left_hand', 'right_hand')})
    return original, mirrored, dtypes


def load_source(source_dir: Path, expected_sha256: str):
    verify_file(source_dir / 'dataset.json', expected_sha256, 'source feature dataset')
    document = json.loads((source_dir / 'dataset.json').read_text())
    if (document.get('status') != 'completed' or document.get('test_used')
            or document.get('purpose') != 'formal_train_validation_features'
            or document['feature_spec'].get('version') != 'classical-pose-motion-v1'):
        raise ValueError('Source must be the completed formal classical-pose-motion-v1 dataset')
    data = {}
    for split, count in COUNTS.items():
        record = document[split]
        if record['count'] != count:
            raise ValueError(f'Incomplete {split} source')
        for row in record['artifacts'].values():
            verify_file(source_dir / row['path'], row['sha256'], f'source {split} artifact')
        X = np.load(source_dir / f'{split}_features.npy', allow_pickle=False)
        rows = json.loads((source_dir / f'{split}_rows.json').read_text())
        if X.shape != (count, FEATURE_DIM) or X.dtype != np.float32 or len(rows) != count:
            raise ValueError(f'Unexpected {split} source shape')
        data[split] = (X, rows)
    return document, data


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    state = {'status': 'running', 'stage': 'load_source', 'created_at_utc': utc(), 'version': VERSION}
    write_json(output / 'state.json', state)
    try:
        source_dir = args.source_feature_dir.resolve()
        source_doc, source = load_source(source_dir, args.expected_source_sha256)
        decisions = {split: signer_decisions(X, [r['signer_id'] for r in rows])
                     for split, (X, rows) in source.items()}
        write_json(output / 'handedness_decisions.json', {'rule': rule_spec(), 'decisions': decisions,
                                                          'source_dataset_sha256': args.expected_source_sha256})
        state['stage'] = 'audit_cohorts'; write_json(output / 'state.json', state)
        _, _, cohorts = audit_dataset()
        document = copy.deepcopy(source_doc)
        mirrored_signers = {s: sorted(k for k, v in d.items() if v['mirror']) for s, d in decisions.items()}
        document['feature_spec']['handedness_canonicalization'] = {
            'version': VERSION, 'rule': rule_spec()['signer_rule'], 'mirror': rule_spec()['mirror'],
            'mirrored_signers': mirrored_signers}
        document.update(variant=VERSION, status='running', created_at_utc=utc(), workers=args.workers,
                        derived_from={'dataset_json': str(source_dir / 'dataset.json'),
                                      'sha256': args.expected_source_sha256})
        document['source_sha256'] = dict(source_doc['source_sha256'], **{
            name: sha256(WORKSPACE / name) for name in
            ('scripts/prepare_handedness_dataset.py', 'scripts/classical_handedness.py')})
        checks = {}
        for split, (X, rows) in source.items():
            cohort = cohorts[split]
            samples = cohort['index']['samples']
            if [{k: s[k] for k in IDENTITY_KEYS} for s in samples] != rows:
                raise ValueError(f'{split} cohort order/identity differs from the frozen feature rows')
            flags = [bool(decisions[split][r['signer_id']]['mirror']) for r in rows]
            tasks = [(s, str(cohort['path']), cohort['index']['pipeline_sha256'], f)
                     for s, f in zip(samples, flags)]
            state.update(stage=f'extract_{split}', processed=0, total=len(tasks)); write_json(output / 'state.json', state)
            new = X.copy()
            unmirrored_equal = mirrored_equal = 0
            dtypes = set()
            stamp = time.perf_counter()
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for n, (original, mirrored, kinds) in enumerate(pool.map(build_one, tasks, chunksize=16)):
                    dtypes.update(kinds)
                    if original.tobytes() != X[n].tobytes():
                        raise ValueError(f'{split} row {n} does not reproduce the frozen descriptor')
                    if mirrored is not None:
                        expected = mirror_descriptors(X[n])
                        if mirrored.tobytes() != expected.tobytes():
                            diff = float(np.max(np.abs(mirrored.astype(np.float64) - expected)))
                            raise ValueError(f'{split} row {n}: raw-pose mirror differs from descriptor map (max {diff})')
                        new[n] = mirrored
                        mirrored_equal += 1
                    else:
                        unmirrored_equal += 1
                    if (n + 1) % 2000 == 0 or n + 1 == len(tasks):
                        state.update(processed=n + 1, elapsed_seconds=time.perf_counter() - stamp)
                        write_json(output / 'state.json', state)
                        print(f'{split} {n + 1}/{len(tasks)} in {state["elapsed_seconds"]:.1f}s', flush=True)
            mapped = canonicalize_descriptors(X, [r['signer_id'] for r in rows], decisions[split])
            if mapped.tobytes() != new.tobytes():
                raise ValueError(f'{split} matrix differs from whole-matrix descriptor-path canonicalization')
            np.save(output / f'{split}_features.npy', new)
            for name in (f'{split}_labels.npy', f'{split}_rows.json'):
                shutil.copyfile(source_dir / name, output / name)
            record = document[split]
            record['feature_seconds'] = time.perf_counter() - stamp
            record['artifacts'] = {name: {'path': name, 'sha256': sha256(output / name),
                                          'size_bytes': (output / name).stat().st_size}
                                   for name in (f'{split}_features.npy', f'{split}_labels.npy', f'{split}_rows.json')}
            for name in (f'{split}_labels.npy', f'{split}_rows.json'):
                if record['artifacts'][name]['sha256'] != source_doc[split]['artifacts'][name]['sha256']:
                    raise ValueError(f'Copied {name} changed')
            checks[split] = {'rows': len(rows), 'mirrored_rows': mirrored_equal,
                             'unmirrored_rows_bit_identical_to_frozen': unmirrored_equal,
                             'mirrored_rows_bit_identical_to_descriptor_map': mirrored_equal,
                             'mirrored_signers': mirrored_signers[split], 'raw_coordinate_dtypes': sorted(dtypes)}
        document.update(status='completed', wall_seconds=time.perf_counter() - started)
        write_json(output / 'dataset.json', document)
        write_json(output / 'verification.json', {'status': 'verified', 'verified_at_utc': utc(),
                                                  'checks': checks, 'test_used': False,
                                                  'dataset_sha256': sha256(output / 'dataset.json')})
        state.update(status='completed', stage='complete', wall_seconds=document['wall_seconds'], checks=checks)
        write_json(output / 'state.json', state)
        print(json.dumps({'status': 'completed', 'checks': checks}), flush=True)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(output / 'state.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-feature-dir', type=Path, default=SOURCE_DIR)
    parser.add_argument('--expected-source-sha256', default=SOURCE_SHA256)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be 1..16')
    run(args)
