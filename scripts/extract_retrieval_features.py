#!/usr/bin/env python3
"""Audited frozen official PoET CLS features for a finite train-reference pilot.

Run each official split separately through run_logged.py. Test is unsupported.
The engineering mode selects exactly 32 train videos by metadata-only SHA order.
No feature/model fitting, augmentation, optimizer, RGB cache reuse, or exclusions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = 2731
FEATURE_DIM = 128
EXPECTED_COUNTS = {'train': 40154, 'val': 10304}
# Full official splits; the earlier 256-word RGB subset had only 33 train signers.
EXPECTED_SIGNERS = {'train': 35, 'val': 6}
OFFICIAL_CHECKPOINT = '5511f1270bd252c586705da8233f7ca1caf0b079a8ddf498b5bb0606f5e6d5c0'
ENGINEERING_SALT = 'retrieval-engineering-20260926:'
SOURCE_FILES = ('scripts/extract_retrieval_features.py', 'scripts/finetune_baseline.py',
                'scripts/evaluate_pose_cache.py', 'scripts/prepare_poses.py',
                'scripts/_poet_runtime.py', 'scripts/infer_video.py',
                'vendor/poet/model.py', 'vendor/poet/mediapipe_extraction.py',
                'vendor/poet/protobuf_to_dict.py')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    """Logical array bytes; dtype and shape are separately fixed in the schema."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def inside(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if ROOT != path and ROOT not in path.parents:
        raise ValueError('Paths must remain inside the project workspace')
    return path


def validate_args(args):
    if args.split not in EXPECTED_COUNTS:
        raise ValueError('Only official train or val is supported; test is forbidden')
    if args.engineering_count not in (None, 32):
        raise ValueError('Engineering scope is exactly 32 metadata-selected train videos')
    if args.engineering_count is not None and args.split != 'train':
        raise ValueError('Engineering mode must use only train')
    if args.batch_size != 128:
        raise ValueError('The finite pilot fixes batch-size at 128')
    if args.device not in ('cuda:0', 'cpu'):
        raise ValueError('Use project GPU 0 (cuda:0) or explicit CPU engineering verification')
    if args.engineering_count is None and args.device != 'cuda:0':
        raise ValueError('Formal extraction is fixed to cuda:0')


def validate_rows(items, split, label_to_id, complete=False):
    """Strengthen legacy cache identity validation before any model execution."""
    if not items or split not in EXPECTED_COUNTS:
        raise ValueError('Empty or forbidden split')
    indices, videos, sources = set(), set(), set()
    for item in items:
        if item.get('split') != split or item.get('status') not in ('ok', 'reused'):
            raise ValueError('Every row must be a successful cache in the requested split')
        if type(item.get('index')) is not int or item['index'] < 0 or item['index'] in indices:
            raise ValueError('Duplicate or invalid source index')
        indices.add(item['index'])
        video = item.get('video_sha256')
        if (not isinstance(video, str) or len(video) != 64 or
                any(c not in '0123456789abcdef' for c in video) or video in videos):
            raise ValueError('Duplicate or invalid video SHA256')
        videos.add(video)
        source = item.get('source_path')
        if not isinstance(source, str) or not source or source in sources:
            raise ValueError('Duplicate or invalid source path')
        sources.add(source)
        if item.get('gloss') not in label_to_id or not isinstance(item.get('signer_id'), str) or not item['signer_id']:
            raise ValueError('Unknown class or missing signer')
        identity = item.get('identity', {})
        if any(identity.get(key) != item[value] for key, value in
               [('sha256', 'video_sha256'), ('gloss', 'gloss'), ('signer_id', 'signer_id'), ('split', 'split')]):
            raise ValueError('Cache identity is inconsistent with row metadata')
        if not isinstance(identity.get('path'), str) or not identity['path']:
            raise ValueError('Cache identity needs its original source path')
        if 'class_id' in item and (type(item['class_id']) is not int or item['class_id'] != label_to_id[item['gloss']]):
            raise ValueError('Explicit class ID differs from official label map')
    if complete:
        if len(items) != EXPECTED_COUNTS[split]:
            raise ValueError('Formal input must have the complete official split count')
        if {item['gloss'] for item in items} != set(label_to_id):
            raise ValueError('Formal input must include every official class')
        if len({item['signer_id'] for item in items}) != EXPECTED_SIGNERS[split]:
            raise ValueError('Formal input signer count differs from the pinned release')


def engineering_selection(items):
    if len(items) < 32:
        raise ValueError('Need at least 32 original train records for engineering selection')
    # No labels, errors, predictions, or validation data enter selection.
    return sorted(items, key=lambda item: (hashlib.sha256(
        (ENGINEERING_SALT + item['video_sha256']).encode()).hexdigest(), item['video_sha256']))[:32]


def read_evaluation_input(item, preprocessing):
    """Read one verified byte snapshot, then reconstruct the exact official input."""
    from _poet_runtime import prepare_features
    payload = Path(item['features_path']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != item['features_sha256']:
        raise ValueError('Evaluation feature payload SHA256 changed')
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {'normalized_features', 'model_input', 'source_frame_indices'}:
            raise ValueError('Unexpected normalized cache array inventory')
        normalized, prepared, indices = (archive[key] for key in
                                        ('normalized_features', 'model_input', 'source_frame_indices'))
    if (normalized.ndim != 2 or normalized.shape[1] != 188 or len(normalized) < 1 or
            normalized.dtype != np.float64 or prepared.shape != (128, 188) or
            prepared.dtype != np.float32 or indices.shape != (128,) or indices.dtype != np.int64):
        raise ValueError('Invalid official cached input shape or dtype')
    if not np.isfinite(normalized).all() or not np.isfinite(prepared).all():
        raise ValueError('Nonfinite official cached input')
    if np.any(indices < -1) or np.any(indices >= len(normalized)):
        raise ValueError('Invalid source-frame trace')
    reconstructed, original_indices = prepare_features(normalized, preprocessing, mode='eval')
    if not np.array_equal(reconstructed.numpy(), prepared) or not np.array_equal(original_indices, indices):
        raise ValueError('Cached input or source-frame trace differs from official eval preprocessing')
    return prepared, indices


def validate_extracted_arrays(arrays, count):
    if type(count) is not int or count < 1:
        raise ValueError('Extraction must contain at least one complete sample')
    if set(arrays) != {'embeddings', 'top5_class_ids', 'top5_logits'}:
        raise ValueError('Unexpected feature array inventory')
    emb, ids, values = (arrays[k] for k in ('embeddings', 'top5_class_ids', 'top5_logits'))
    if emb.shape != (count, FEATURE_DIM) or emb.dtype != np.float32:
        raise ValueError('CLS embedding shape/dtype differs from protocol')
    if ids.shape != (count, 5) or ids.dtype != np.int64 or values.shape != (count, 5) or values.dtype != np.float32:
        raise ValueError('Top5 shape/dtype differs from protocol')
    if not np.isfinite(emb).all() or not np.isfinite(values).all():
        raise ValueError('Nonfinite frozen model outputs')
    if np.any(np.linalg.norm(emb.astype(np.float64), axis=1) <= 1e-12):
        raise ValueError('CLS embedding norm is too small for cosine retrieval')
    if np.any(ids < 0) or np.any(ids >= NUM_CLASSES) or any(len(set(row.tolist())) != 5 for row in ids):
        raise ValueError('Top5 must be distinct valid official class IDs')
    if np.any(values[:, 1:] > values[:, :-1]):
        raise ValueError('Top5 raw logits must be descending')


def verify_frozen_model(model, before):
    import torch
    if any(module.training for module in model.modules()):
        raise ValueError('Frozen model unexpectedly entered training mode')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Frozen model has gradients or trainable parameters')
    after = model.state_dict()
    if set(before) != set(after) or any(not torch.equal(value, after[key].detach().cpu())
                                     for key, value in before.items()):
        raise ValueError('Frozen model weights or buffers changed')


def extract_batch(model, prepared, device, verify_unhooked=False):
    import torch
    capture = []
    hook = model.mlp_classifier.register_forward_pre_hook(
        lambda module, inputs: capture.append(inputs[0].detach()))
    try:
        with torch.inference_mode():
            logits = model(torch.from_numpy(prepared).to(device), device)
            if (logits.shape != (len(prepared), NUM_CLASSES) or logits.dtype != torch.float32 or
                    not bool(torch.isfinite(logits).all())):
                raise ValueError('Official logits have unexpected shape/dtype/numerics')
            if len(capture) != 1 or capture[0].shape != (len(prepared), FEATURE_DIM):
                raise ValueError('Expected exactly one unmodified classifier CLS input')
            values, ids = logits.topk(5, dim=1)
            arrays = {'embeddings': capture[0].cpu().numpy().copy(),
                      'top5_class_ids': ids.cpu().numpy().copy(),
                      'top5_logits': values.cpu().numpy().copy()}
    finally:
        hook.remove()
    if verify_unhooked:
        with torch.inference_mode():
            reference = model(torch.from_numpy(prepared).to(device), device)
        if not torch.equal(logits, reference):
            raise ValueError('CLS hook changed original classifier logits')
    validate_extracted_arrays(arrays, len(prepared))
    return arrays


def audit_engineering_cache(index_path, index, selected, preprocessing):
    """Reuse existing strict cache verifier on only the selected engineering rows."""
    from evaluate_pose_cache import referenced_path
    from prepare_poses import verify_cache
    for item in selected:
        metadata_path = referenced_path(index_path.parent, item['cache_manifest_path'])
        if metadata_path.name != 'manifest.json':
            raise ValueError('Per-video cache must use manifest.json')
        target = metadata_path.parent
        for key, name in [('pose_path', 'raw_pose.npz'), ('features_path', 'normalized_features.npz')]:
            if referenced_path(index_path.parent, item[key]) != target / name:
                raise ValueError('Cache path disagrees with per-video metadata')
        metadata = verify_cache(target, item, index['pipeline_sha256'])
        for key, name in [('pose_sha256', 'raw_pose.npz'), ('features_sha256', 'normalized_features.npz')]:
            if item[key] != metadata['artifacts'][name]['sha256']:
                raise ValueError('Cache index/artifact SHA256 mismatch')
        item['features_path'] = str(target / 'normalized_features.npz')
        read_evaluation_input(item, preprocessing)


def fingerprint_files(paths):
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(paths))}


def verify_fingerprints(fingerprints):
    for name, digest in fingerprints.items():
        if sha256(inside(name)) != digest:
            raise ValueError('Frozen source, checkpoint, or metadata file changed: ' + name)


def run(args):
    validate_args(args)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from _poet_runtime import load_bundle, resolve_device, plain_metadata
    from evaluate_pose_cache import audit_provenance, validate_cache_index, OFFICIAL_LABEL_MAP_SHA256
    from finetune_baseline import audit_cohort, tensor_state_hash

    started = time.perf_counter()
    index_path, output = inside(args.cache_manifest), inside(args.output_dir)
    if output == ROOT or output == index_path.parent or output in index_path.parents:
        raise ValueError('Output must be separate from source metadata/cache')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'status': 'running',
                'scope': 'engineering' if args.engineering_count is not None else 'formal',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'test_used': False,
                'optimizer_steps': 0, 'failed': 0, 'weights_unchanged': False,
                'checkpoint_sha256': OFFICIAL_CHECKPOINT, 'feature_dim': FEATURE_DIM,
                'class_ids': list(range(NUM_CLASSES)), 'splits': {},
                'stage': 'load_and_metadata_audit',
                'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    write_json(output / 'manifest.json', manifest)
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20260926)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = resolve_device(args.device)
        checkpoint_dir = ROOT / 'assets/checkpoints'
        checkpoint_path = checkpoint_dir / 'ASL_Citizen_PoET.pth'
        if sha256(checkpoint_path) != OFFICIAL_CHECKPOINT:
            raise ValueError('Only the unmodified official pretrained PoET checkpoint is permitted')
        bundle = load_bundle(checkpoint_dir, device)
        if (bundle.metadata['checkpoint_sha256'] != OFFICIAL_CHECKPOINT or
                bundle.metadata['label_map_sha256'] != OFFICIAL_LABEL_MAP_SHA256 or
                bundle.checkpoint['num_classes'] != NUM_CLASSES):
            raise ValueError('Official checkpoint, vocabulary, or label order differs')
        bundle.model.requires_grad_(False).eval()
        before = {k: v.detach().cpu().clone() for k, v in bundle.model.state_dict().items()}
        before_hash = tensor_state_hash(before)
        index = json.loads(index_path.read_text())
        split = validate_cache_index(index, bundle.label_to_id, allow_train_smoke=True)
        if split != args.split:
            raise ValueError('Requested split differs from cache metadata')
        provenance = audit_provenance(index_path, index, bundle.checkpoint['data_preprocessing'],
                                      bundle.label_to_id, split, allow_subset_val=False)
        if not provenance['complete_official_split']:
            raise ValueError('Source cache must identify the complete official split, even for engineering')
        validate_rows(index['samples'], split, bundle.label_to_id, complete=True)
        source_paths = [ROOT / name for name in SOURCE_FILES]
        source_paths += [ROOT / name for name in provenance['pipeline']['source_sha256']]
        source_paths += [checkpoint_path, index_path,
                         *(checkpoint_dir / name for name in OFFICIAL_LABEL_MAP_SHA256),
                         inside(provenance['source_manifest_path']), inside(provenance['official_split_path'])]
        fingerprints = fingerprint_files(source_paths)
        manifest.update(label_map={str(i): bundle.id_to_label[i] for i in range(NUM_CLASSES)},
                        source_sha256=fingerprints, model=plain_metadata(bundle.metadata),
                        weights_state_sha256_before=before_hash,
                        protocol={'inference': 'eval, inference_mode, FP32, no TF32', 'augmentation': False,
                                  'batch_size': 128, 'seed': 20260926, 'num_threads': 4,
                                  'engineering_selection': ENGINEERING_SALT + 'video_sha256, sort ascending, first 32',
                                  'embedding': 'unmodified 128D input to official mlp_classifier',
                                  'top5': 'torch.topk on raw 2731-class logits, descending; no softmax',
                                  'record_order': 'original cache index order; engineering uses fixed SHA order',
                                  'video_integrity_policy': provenance['video_integrity_policy']},
                        stage='cache_audit')
        write_json(output / 'manifest.json', manifest)
        if args.engineering_count is not None:
            items = engineering_selection(index['samples'])
            audit_engineering_cache(index_path, index, items, bundle.preprocessing)
        else:
            cohort = audit_cohort(index_path, bundle, engineering=False, role='train' if split == 'train' else 'validation')
            if cohort['sha256'] != fingerprints[str(index_path.relative_to(ROOT))]:
                raise ValueError('Cache index changed during audit')
            items = cohort['samples']
        validate_rows(items, split, bundle.label_to_id, complete=args.engineering_count is None)
        arrays = {'embeddings': np.empty((len(items), FEATURE_DIM), np.float32),
                  'top5_class_ids': np.empty((len(items), 5), np.int64),
                  'top5_logits': np.empty((len(items), 5), np.float32)}
        manifest.update(stage='extracting', expected_samples=len(items), progress={'completed': 0, 'total': len(items)})
        write_json(output / 'manifest.json', manifest)
        with (output / 'records.jsonl').open('x', encoding='utf-8') as stream:
            for offset in range(0, len(items), args.batch_size):
                batch_items = items[offset:offset + args.batch_size]
                data = [read_evaluation_input(item, bundle.preprocessing) for item in batch_items]
                prepared = np.stack([value[0] for value in data])
                extracted = extract_batch(bundle.model, prepared, device,
                                          verify_unhooked=args.engineering_count is not None)
                if args.engineering_count is not None:
                    repeated = extract_batch(bundle.model, prepared, device)
                    if any(not np.array_equal(extracted[k], repeated[k]) for k in extracted):
                        raise ValueError('Repeated engineering inference was not bitwise identical')
                    manifest['engineering_repeat_exact'] = True
                    manifest['engineering_unhooked_logits_exact'] = True
                for key in arrays:
                    arrays[key][offset:offset + len(batch_items)] = extracted[key]
                for row, (item, (model_input, source_indices)) in enumerate(zip(batch_items, data), offset):
                    record = {key: item[key] for key in ('index', 'identity', 'gloss', 'signer_id', 'split',
                              'source_path', 'video_sha256', 'pose_sha256', 'features_sha256')}
                    record.update(row=row, class_id=bundle.label_to_id[item['gloss']],
                                  input_sha256=array_sha256(model_input),
                                  source_frame_indices_sha256=array_sha256(source_indices),
                                  source_frame_indices=source_indices.tolist(),
                                  cache_manifest_sha256=provenance['cache_manifest_sha256'],
                                  source_manifest_sha256=provenance['source_manifest_sha256'])
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
                stream.flush()
                manifest['progress'] = {'completed': offset + len(batch_items), 'total': len(items),
                                        'elapsed_seconds': time.perf_counter() - started}
                write_json(output / 'manifest.json', manifest)
                if (offset // args.batch_size) % 10 == 0 or offset + len(batch_items) == len(items):
                    print(json.dumps(manifest['progress']), flush=True)
        validate_extracted_arrays(arrays, len(items))
        verify_frozen_model(bundle.model, before)
        verify_fingerprints(fingerprints)
        after_hash = tensor_state_hash(bundle.model.state_dict())
        if before_hash != after_hash:
            raise ValueError('Frozen model state hash changed')
        with (output / 'features.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        with np.load(output / 'features.npz', allow_pickle=False) as saved:
            validate_extracted_arrays({key: saved[key] for key in saved.files}, len(items))
            if any(not np.array_equal(arrays[key], saved[key]) for key in arrays):
                raise ValueError('Feature archive read-back differed')
        records = [json.loads(line) for line in (output / 'records.jsonl').read_text().splitlines()]
        if len(records) != len(items) or [x['row'] for x in records] != list(range(len(items))):
            raise ValueError('Metadata row count or order changed on disk')
        entry = {'samples': len(items), 'failed': 0, 'features_path': 'features.npz',
                 'features_sha256': sha256(output / 'features.npz'), 'features_bytes': (output / 'features.npz').stat().st_size,
                 'metadata_path': 'records.jsonl', 'metadata_sha256': sha256(output / 'records.jsonl'),
                 'metadata_bytes': (output / 'records.jsonl').stat().st_size,
                 'source_manifest_sha256': provenance['source_manifest_sha256'],
                 'cache_manifest_sha256': provenance['cache_manifest_sha256'],
                 'provenance': provenance, 'complete_official_split': args.engineering_count is None,
                 'signers': sorted({item['signer_id'] for item in items}),
                 'array_schema': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                                        'sha256': array_sha256(value)} for key, value in arrays.items()}}
        manifest.update(status='completed', stage='completed', splits={split: entry},
                        weights_unchanged=True, weights_state_sha256_after=after_hash,
                        gradients_absent=True, model_eval=True,
                        completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started)
        write_json(output / 'manifest.json', manifest)
        print(json.dumps({'status': 'completed', 'split': split, 'samples': len(items), 'failed': 0,
                          'elapsed_seconds': manifest['elapsed_seconds']}), flush=True)
        return manifest
    except BaseException as error:
        manifest.update(status='failed', failed=1, failed_at_utc=datetime.now(timezone.utc).isoformat(),
                        error=str(error), error_type=type(error).__name__,
                        elapsed_seconds=time.perf_counter() - started)
        write_json(output / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--engineering-count', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
