#!/usr/bin/env python3
"""Package the retrieval pilot's official PoET CLS vectors as a classical feature dataset.

The output directory has the same layout as the frozen classical feature dataset,
so the unchanged run_classical_baselines.py can fit kNN / Linear SVM / bounded RF
on it with the original hyperparameters: only the representation changes.

Nothing is recomputed here. The script verifies the two existing extraction runs
(manifest fields, file and array SHA256), proves that CLS row i is the same clip
as frozen classical row i (identity keys, official label ID, pose-cache SHA), and
writes the 128-D vectors next to byte copies of the frozen labels and row files.
No model is loaded, no gloss is used to select rows and no test data is read.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from evaluate_pose_cache import sha256, verify_file, write_json

WORKSPACE = Path(__file__).resolve().parents[1]
VERSION = 'poet-official-cls128-v1'
FEATURE_DIM = 128
NUM_CLASSES = 2731
COUNTS = {'train': 40154, 'val': 10304}
IDENTITY_KEYS = ('index', 'gloss', 'signer_id', 'split', 'source_path', 'video_sha256')
OFFICIAL_CHECKPOINT_SHA256 = '5511f1270bd252c586705da8233f7ca1caf0b079a8ddf498b5bb0606f5e6d5c0'
CLASSICAL_DIR = WORKSPACE / 'runs/20261002T160743Z-classical-full-features-3470975b/features'
CLASSICAL_SHA256 = '3a9ef620bb5b3593375f831ada5b191fe66b9b1bafefe9aba9ff4e075daca83f'
EXTRACTIONS = {
    'train': {'run': 'runs/20260926T202515Z-retrieval-pilot-train-extraction-5911f3b9',
              'features_sha256': 'c58024b6eb42cdca1fea25d2932495de96bbdf2e2c3c6842aa27791ba972430b',
              'records_sha256': 'a86513e51360c44b1aa5fe099a6930582e2bfd101600757473268b3b839c89f4',
              'embeddings_sha256': '52570d6d242df373b7578aeca6343823eb5278e9d8520f73dcce2ad752ba86b4',
              'cache_manifest_sha256': '0403e8ab7b2eaeae589749a30f67f68a6157615a2140623f3af9d40f02d8881c'},
    'val': {'run': 'runs/20260926T203453Z-retrieval-pilot-val-extraction-993a38f2',
            'features_sha256': '9ebf21c8c7461df13f4f81ea7ae1d77e8dea367cdbb4647414bbe9d564da1c2a',
            'records_sha256': '54965bbfc50ab1f0a16ec6e78238352eb56e3a53e4ffa0d60040fc2afcc3cc41',
            'embeddings_sha256': '86326741ecda5b9fb9d4e3053523d3703ffd1a836f6c1dd761ea27770192197d',
            'cache_manifest_sha256': 'a829029d700e50dd07d0f88454d9f3d42a65e946586a1188e7dfe9a81d5a5c86'},
}
ARRAYS = ('embeddings', 'top5_class_ids', 'top5_logits')


def utc():
    return datetime.now(timezone.utc).isoformat()


def array_sha256(array):
    """Same definition as extract_retrieval_features.array_sha256."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_extraction_manifest(manifest: dict, split: str, pinned: dict, count: int) -> dict:
    """G1 manifest checks; returns the split record."""
    expected = {'status': 'completed', 'scope': 'formal', 'test_used': False, 'failed': 0,
                'optimizer_steps': 0, 'weights_unchanged': True,
                'checkpoint_sha256': OFFICIAL_CHECKPOINT_SHA256, 'feature_dim': FEATURE_DIM}
    for key, value in expected.items():
        if manifest.get(key) != value or type(manifest.get(key)) is not type(value):
            raise ValueError(f'{split} extraction manifest {key}={manifest.get(key)!r}, expected {value!r}')
    if set(manifest.get('splits', {})) != {split}:
        raise ValueError(f'{split} extraction manifest must contain exactly the {split} split')
    record = manifest['splits'][split]
    checks = {'samples': count, 'failed': 0, 'complete_official_split': True,
              'features_sha256': pinned['features_sha256'], 'metadata_sha256': pinned['records_sha256'],
              'cache_manifest_sha256': pinned['cache_manifest_sha256']}
    for key, value in checks.items():
        if record.get(key) != value:
            raise ValueError(f'{split} extraction split record {key}={record.get(key)!r}, expected {value!r}')
    schema = record.get('array_schema', {})
    if set(schema) != set(ARRAYS):
        raise ValueError(f'{split} array inventory differs')
    if schema['embeddings'] != {'shape': [count, FEATURE_DIM], 'dtype': 'float32',
                                'sha256': pinned['embeddings_sha256']}:
        raise ValueError(f'{split} embeddings schema differs from the pinned value')
    if manifest.get('class_ids') != list(range(NUM_CLASSES)):
        raise ValueError(f'{split} extraction class IDs are not the official 0..2730')
    return record


def validate_arrays(arrays: dict, schema: dict, count: int) -> np.ndarray:
    if set(arrays) != set(ARRAYS):
        raise ValueError('Unexpected CLS array inventory')
    for name in ARRAYS:
        spec = schema[name]
        value = arrays[name]
        if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype']:
            raise ValueError(f'{name} shape/dtype differs from manifest')
        if array_sha256(value) != spec['sha256']:
            raise ValueError(f'{name} array SHA256 differs from manifest')
    embeddings = arrays['embeddings']
    if embeddings.shape != (count, FEATURE_DIM) or embeddings.dtype != np.float32:
        raise ValueError('CLS embeddings must be (count, 128) float32')
    if not np.isfinite(embeddings).all():
        raise ValueError('CLS embeddings contain NaN or infinity')
    if np.any(np.linalg.norm(embeddings.astype(np.float64), axis=1) <= 1e-12):
        raise ValueError('A CLS embedding has zero norm')
    return embeddings


def check_row_identity(records: list, classical_rows: list, classical_labels: np.ndarray,
                       label_to_id: dict, cohort_samples: list | None, split: str) -> dict:
    """G2: CLS row i must be frozen classical row i; never reorder or drop."""
    count = len(classical_rows)
    if len(records) != count or len(classical_labels) != count:
        raise ValueError(f'{split}: {len(records)} CLS records, {count} classical rows, '
                         f'{len(classical_labels)} classical labels')
    if cohort_samples is not None and len(cohort_samples) != count:
        raise ValueError(f'{split}: cohort has {len(cohort_samples)} samples')
    for i, (record, row) in enumerate(zip(records, classical_rows)):
        if record.get('row') != i:
            raise ValueError(f'{split}: CLS record {i} has row={record.get("row")!r}')
        for key in IDENTITY_KEYS:
            if record.get(key) != row.get(key):
                raise ValueError(f'{split} row {i}: {key} differs between CLS and classical rows')
        if record.get('split') != split:
            raise ValueError(f'{split} row {i}: wrong split {record.get("split")!r}; test is forbidden')
        target = label_to_id[row['gloss']]
        if record.get('class_id') != target or int(classical_labels[i]) != target:
            raise ValueError(f'{split} row {i}: class ID differs from the official label of {row["gloss"]}')
        if cohort_samples is not None:
            sample = cohort_samples[i]
            if any(sample.get(key) != row.get(key) for key in IDENTITY_KEYS):
                raise ValueError(f'{split} row {i}: classical row differs from the official cohort')
            if record.get('pose_sha256') != sample.get('pose_sha256'):
                raise ValueError(f'{split} row {i}: CLS came from a different pose cache')
    return {'rows': count, 'identity_keys': list(IDENTITY_KEYS), 'row_index_equals_position': True,
            'class_id_equals_official_label': True,
            'pose_sha256_checked_against_cohort': cohort_samples is not None,
            'signers': len({row['signer_id'] for row in classical_rows})}


def read_jsonl(path: Path) -> list:
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    state = {'status': 'running', 'stage': 'classical_source', 'created_at_utc': utc(), 'version': VERSION}
    write_json(output / 'state.json', state)
    try:
        verify_file(CLASSICAL_DIR / 'dataset.json', CLASSICAL_SHA256, 'frozen classical feature dataset')
        classical = json.loads((CLASSICAL_DIR / 'dataset.json').read_text())
        if (classical.get('status') != 'completed' or classical.get('test_used') is not False
                or classical.get('purpose') != 'formal_train_validation_features'):
            raise ValueError('Frozen classical dataset is not the completed formal train/validation set')
        labels = classical['id_to_label']
        label_to_id = {label: i for i, label in enumerate(labels)}
        if len(labels) != NUM_CLASSES or len(label_to_id) != NUM_CLASSES:
            raise ValueError('Expected the full 2,731-class vocabulary')
        state['stage'] = 'audit_cohorts'; write_json(output / 'state.json', state)
        from prepare_classical_dataset import audit_dataset
        _, forward, cohorts = audit_dataset()
        if forward != label_to_id:
            raise ValueError('Official label map differs from the classical dataset vocabulary')
        document = copy.deepcopy(classical)
        checks = {}
        for split, count in COUNTS.items():
            state['stage'] = f'verify_{split}'; write_json(output / 'state.json', state)
            pinned = EXTRACTIONS[split]
            run_dir = WORKSPACE / pinned['run']
            run_manifest = json.loads((run_dir / 'manifest.json').read_text())
            if run_manifest.get('state') != 'completed' or run_manifest.get('returncode') != 0:
                raise ValueError(f'{split} extraction run did not complete cleanly')
            if (run_dir / 'exit_code.txt').read_text().strip() != '0':
                raise ValueError(f'{split} extraction exit code is not 0')
            features_dir = run_dir / 'features'
            manifest = json.loads((features_dir / 'manifest.json').read_text())
            record = verify_extraction_manifest(manifest, split, pinned, count)
            if record['features_path'] != 'features.npz' or record['metadata_path'] != 'records.jsonl':
                raise ValueError(f'{split} extraction file names differ')
            verify_file(features_dir / 'features.npz', pinned['features_sha256'], f'{split} CLS features.npz')
            verify_file(features_dir / 'records.jsonl', pinned['records_sha256'], f'{split} CLS records')
            with np.load(features_dir / 'features.npz', allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            embeddings = validate_arrays(arrays, record['array_schema'], count)
            for name in (f'{split}_labels.npy', f'{split}_rows.json'):
                row = classical[split]['artifacts'][name]
                verify_file(CLASSICAL_DIR / row['path'], row['sha256'], f'frozen classical {name}')
            classical_rows = json.loads((CLASSICAL_DIR / f'{split}_rows.json').read_text())
            classical_labels = np.load(CLASSICAL_DIR / f'{split}_labels.npy', allow_pickle=False)
            if classical[split]['sha256'] != pinned['cache_manifest_sha256']:
                raise ValueError(f'{split}: classical and CLS rows come from different cohorts')
            records = read_jsonl(features_dir / 'records.jsonl')
            identity = check_row_identity(records, classical_rows, classical_labels, label_to_id,
                                          cohorts[split]['index']['samples'], split)
            np.save(output / f'{split}_features.npy', np.ascontiguousarray(embeddings, dtype=np.float32))
            for name in (f'{split}_labels.npy', f'{split}_rows.json'):
                shutil.copyfile(CLASSICAL_DIR / name, output / name)
            reloaded = np.load(output / f'{split}_features.npy', allow_pickle=False)
            if reloaded.tobytes() != embeddings.tobytes():
                raise ValueError(f'{split} saved features differ from the verified embeddings')
            artifacts = {name: {'path': name, 'sha256': sha256(output / name),
                                'size_bytes': (output / name).stat().st_size}
                         for name in (f'{split}_features.npy', f'{split}_labels.npy', f'{split}_rows.json')}
            for name in (f'{split}_labels.npy', f'{split}_rows.json'):
                if artifacts[name]['sha256'] != classical[split]['artifacts'][name]['sha256']:
                    raise ValueError(f'Copied {name} changed')
            document[split]['artifacts'] = artifacts
            document[split].pop('feature_seconds', None)
            norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
            checks[split] = {**identity, 'embedding_sha256': pinned['embeddings_sha256'],
                             'features_npy_sha256': artifacts[f'{split}_features.npy']['sha256'],
                             'norm_min': float(norms.min()), 'norm_max': float(norms.max()),
                             'extraction_run': pinned['run']}
        for key in ('signer_id', 'video_sha256', 'source_path'):
            train_values = {row[key] for row in json.loads((output / 'train_rows.json').read_text())}
            val_values = {row[key] for row in json.loads((output / 'val_rows.json').read_text())}
            if train_values & val_values:
                raise ValueError(f'Train and validation overlap in {key}')
        document.pop('poet_weights_loaded', None)
        document.pop('wall_seconds', None)
        document['feature_spec'] = {
            'version': VERSION, 'dimension': FEATURE_DIM,
            'input': ('unmodified 128-D input to the official PoET mlp_classifier (MLP1, a single '
                      'nn.Linear(128, 2731)); frozen eval-mode official checkpoint; original unflipped videos; '
                      'v2 pose caches shared with the classical descriptor'),
            'checkpoint_sha256': OFFICIAL_CHECKPOINT_SHA256,
            'extraction_runs': {split: EXTRACTIONS[split]['run'] for split in COUNTS},
            'extractor': 'scripts/extract_retrieval_features.py',
            'standardization': 'none here; the frozen runner fits StandardScaler on train rows only',
            'caveat': ('The encoder was trained on these train clips and the checkpoint was selected on this '
                       'validation split; this representation is favoured relative to the handcrafted descriptor.')}
        document['representation'] = {'kind': 'learned', 'model': 'official PoET Transformer_CLS',
                                      'poet_weights_loaded_in_this_stage': False,
                                      'vectors_produced_by_poet_weights': True}
        document.update(variant='cls-representation-v1', status='completed', created_at_utc=utc(),
                        wall_seconds=time.perf_counter() - started, test_used=False,
                        derived_from={'classical_dataset_json': str(CLASSICAL_DIR / 'dataset.json'),
                                      'classical_dataset_sha256': CLASSICAL_SHA256,
                                      'extraction_features_sha256': {s: EXTRACTIONS[s]['features_sha256'] for s in COUNTS}})
        document['source_sha256'] = {name: sha256(WORKSPACE / name) for name in (
            'scripts/prepare_cls_representation_dataset.py', 'scripts/prepare_classical_dataset.py',
            'scripts/evaluate_pose_cache.py', 'scripts/extract_retrieval_features.py')}
        if document['purpose'] != 'formal_train_validation_features' or any(
                document[s]['count'] != c for s, c in COUNTS.items()):
            raise ValueError('Output document would not be accepted as the formal train/validation set')
        write_json(output / 'dataset.json', document)
        write_json(output / 'verification.json', {'status': 'verified', 'verified_at_utc': utc(),
                                                  'gates': {'G1_cls_provenance': 'passed', 'G2_row_identity': 'passed'},
                                                  'checks': checks, 'test_used': False,
                                                  'dataset_sha256': sha256(output / 'dataset.json')})
        state.update(status='completed', stage='complete', wall_seconds=document['wall_seconds'],
                     dataset_sha256=sha256(output / 'dataset.json'))
        write_json(output / 'state.json', state)
        print(json.dumps({'status': 'completed', 'dataset_sha256': state['dataset_sha256'], 'checks': checks}), flush=True)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error), ended_at_utc=utc())
        write_json(output / 'state.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())
