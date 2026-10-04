#!/usr/bin/env python3
"""Extract six frozen paired-checkpoint classifier inputs in one audited pass.

This additive finite experiment never trains, labels, augments, or opens test.
Each split's pose/geometry payload is audited once and each inference batch is
read once for all six models. Historical programs and their artifacts are inert.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import extract_retrieval_features as legacy

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'poet-factorial-retrieval-v1'
SEEDS = (20260922, 20260923, 20260924)
MODEL_IDS = tuple(f'{method}_seed{seed}' for seed in SEEDS for method in ('baseline', 'geometry'))
SOURCE_FILES = tuple(dict.fromkeys((*legacy.SOURCE_FILES,
    'scripts/extract_factorial_retrieval_features.py', 'scripts/evaluate_methods.py',
    'scripts/method_runtime.py', 'scripts/method_optimizer.py', 'scripts/poet_methods.py',
    'scripts/poet_geometry.py', 'scripts/prepare_geometry_cache.py', 'scripts/build_asllex_targets.py')))
sha256 = legacy.sha256
array_sha256 = legacy.array_sha256
write_json = legacy.write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def inside(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if ROOT != path and ROOT not in path.parents:
        raise ValueError('Paths must remain inside the project workspace')
    return path


def require_digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid SHA256: ' + name)


def validate_preregistration(document):
    if (document.get('schema_version') != 1 or document.get('protocol') != PROTOCOL or
            document.get('registered_before_factorial_scores') is not True or
            document.get('test_used') is not False or document.get('manual_annotation') is not False or
            document.get('optimizer_steps') != 0 or document.get('sweeps_authorized') is not False or
            document.get('seeds') != list(SEEDS)):
        raise ValueError('Unsupported or unfrozen finite factorial preregistration')
    data = document.get('data', {})
    if (data.get('train_count') != 40154 or data.get('val_count') != 10304 or data.get('classes') != 2731 or
            data.get('memory_split') != 'train' or data.get('query_split') != 'val' or
            data.get('video_and_signer_disjoint') is not True or data.get('drop_samples') is not False):
        raise ValueError('Factorial data must retain the full fixed train and val populations')
    for split in ('train', 'val'):
        for kind in ('cache', 'geometry'):
            if not isinstance(data.get(f'{split}_{kind}_manifest'), str):
                raise ValueError('Missing fixed data manifest')
            inside(data[f'{split}_{kind}_manifest'])
            require_digest(data.get(f'{split}_{kind}_sha256'), 'data manifest')
    models = document.get('models')
    if not isinstance(models, list) or len(models) != 6 or [m.get('model_id') for m in models] != list(MODEL_IDS):
        raise ValueError('Exactly six seed-major baseline/geometry checkpoint identities are required')
    checkpoints = set()
    for model in models:
        expected_id = f"{model.get('method')}_seed{model.get('seed')}"
        if model.get('method') not in ('baseline', 'geometry') or model.get('seed') not in SEEDS or model['model_id'] != expected_id:
            raise ValueError('Method, seed and model ID disagree')
        for name in ('checkpoint_sha256', 'reference_predictions_sha256', 'model_state_sha256', 'protocol_sha256'):
            require_digest(model.get(name), name)
        for name in ('checkpoint_path', 'reference_predictions_path', 'reference_evaluation_run'):
            if not isinstance(model.get(name), str):
                raise ValueError('Missing checkpoint or historical reference path')
            inside(model[name])
        if (type(model.get('checkpoint_epoch')) is not int or model['checkpoint_epoch'] < 1 or
                model['checkpoint_sha256'] in checkpoints):
            raise ValueError('Checkpoint epoch or distinct checkpoint identity is invalid')
        checkpoints.add(model['checkpoint_sha256'])
    extraction = document.get('resources', {})
    if (extraction.get('gpu') != 0 or extraction.get('batch_size') != 128 or
            extraction.get('num_threads') != 4 or extraction.get('cpu_affinity') != list(range(8))):
        raise ValueError('Extraction resources differ from the fixed protocol')


def load_preregistration(path, expected):
    require_digest(expected, 'preregistration')
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Preregistration bytes differ from the frozen SHA256')
    document = json.loads(payload)
    validate_preregistration(document)
    return document


def validate_checkpoint_metadata(checkpoint, entry):
    from evaluate_pose_cache import canonical_hash
    from method_optimizer import SUPPORTED_PROTOCOLS
    protocol = checkpoint.get('protocol', {})
    if (checkpoint.get('schema_version') != 1 or protocol.get('version') not in SUPPORTED_PROTOCOLS or
            checkpoint.get('protocol_sha256') != canonical_hash(protocol) or
            checkpoint.get('protocol_sha256') != entry['protocol_sha256'] or
            checkpoint.get('model_state_sha256') != entry['model_state_sha256'] or
            checkpoint.get('completed_epoch') != entry['checkpoint_epoch'] or
            protocol.get('method') != entry['method'] or protocol.get('training', {}).get('seed') != entry['seed'] or
            protocol.get('scope') != 'formal_finetuning' or checkpoint.get('num_classes') != 2731):
        raise ValueError('Checkpoint identity, method, seed, epoch, or protocol differs from preregistration')
    for key in ('data_preprocessing', 'model_parameters'):
        if checkpoint.get(key) != protocol.get('model', {}).get(key):
            raise ValueError('Checkpoint model configuration disagrees with its protocol')
    if checkpoint.get('training_parameters', {}).get('hidden_dropout_prob') != protocol.get('model', {}).get('hidden_dropout_prob'):
        raise ValueError('Checkpoint dropout disagrees with its protocol')


def load_frozen_models(entries, device):
    import torch
    from _poet_runtime import load_bundle
    from evaluate_pose_cache import OFFICIAL_LABEL_MAP_SHA256
    from finetune_baseline import tensor_state_hash
    from method_runtime import restore_method
    from method_optimizer import validate_checkpoint_optimizer_contract
    result = {}
    for entry in entries:
        path = inside(entry['checkpoint_path'])
        if sha256(path) != entry['checkpoint_sha256']:
            raise ValueError('Checkpoint file differs: ' + entry['model_id'])
        bundle = load_bundle(ROOT / 'assets/checkpoints', device)
        if (bundle.metadata['checkpoint_sha256'] != legacy.OFFICIAL_CHECKPOINT or
                bundle.metadata['label_map_sha256'] != OFFICIAL_LABEL_MAP_SHA256):
            raise ValueError('Official base or vocabulary differs')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        validate_checkpoint_metadata(checkpoint, entry)
        model, metadata = restore_method(bundle, checkpoint)
        if metadata != checkpoint['protocol'].get('method_metadata') or metadata['method'] != entry['method']:
            raise ValueError('Restored architecture differs from the checkpoint protocol')
        # This contract checks trainable group membership, so validate before freezing.
        validate_checkpoint_optimizer_contract(checkpoint, model)
        if tensor_state_hash(model.state_dict()) != entry['model_state_sha256']:
            raise ValueError('Restored model tensor fingerprint differs')
        model.requires_grad_(False).eval()
        before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        result[entry['model_id']] = {'model': model, 'bundle': bundle, 'before': before,
                                    'entry': entry, 'method_metadata': metadata}
    return result


def extract_method_batch(model, prepared, geometry, device, verify_unhooked=False):
    """Capture the actual final classifier input after the complete method encoder."""
    import torch
    if getattr(model, 'method', None) not in ('baseline', 'geometry'):
        raise ValueError('Only frozen baseline/geometry methods are accepted')
    if prepared.dtype != np.float32 or prepared.ndim != 3 or prepared.shape[1:] != (128, 188):
        raise ValueError('Method input must be FP32 batch x 128 x 188')
    if model.method == 'geometry':
        if geometry is None or geometry.dtype != np.float32 or geometry.shape != (len(prepared), 128, 28):
            raise ValueError('Geometry method requires the shared aligned FP32 input')
    elif geometry is not None:
        raise ValueError('Baseline cannot receive geometry')
    prepared_tensor = torch.from_numpy(prepared)
    geometry_tensor = torch.from_numpy(geometry) if geometry is not None else None
    captured = []
    hook = model.base.mlp_classifier.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach()))
    try:
        with torch.inference_mode():
            logits = model(prepared_tensor, device, geometry=geometry_tensor)
            if logits.shape != (len(prepared), 2731) or logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
                raise ValueError('Method logits have unexpected shape/dtype/numerics')
            if len(captured) != 1 or captured[0].shape != (len(prepared), 128) or captured[0].dtype != torch.float32:
                raise ValueError('Expected exactly one final FP32 classifier input')
            values, ids = logits.topk(5, dim=1)
            probabilities, probability_ids = logits.softmax(dim=1).topk(5, dim=1)
            if not torch.equal(ids, probability_ids):
                raise ValueError('Raw-logit Top5 rank differs from historical softmax Top5 rank')
            arrays = {'embeddings': captured[0].cpu().numpy().copy(),
                      'top5_class_ids': ids.cpu().numpy().copy(), 'top5_logits': values.cpu().numpy().copy()}
            full_logits = logits.cpu().numpy()
            details = {'logits_sha256': [hashlib.sha256(row.astype('<f4').tobytes()).hexdigest() for row in full_logits],
                       'top5_probabilities': probabilities.cpu().tolist()}
    finally:
        hook.remove()
    if verify_unhooked:
        with torch.inference_mode():
            reference = model(prepared_tensor, device, geometry=geometry_tensor)
        if not torch.equal(logits, reference):
            raise ValueError('Classifier feature hook changed unhooked method logits')
    legacy.validate_extracted_arrays(arrays, len(prepared))
    return arrays, details


def audit_engineering_geometry(path, full_metadata_cohort, selected):
    """Audit full index identities, but raw/geometry payloads for only SHA-selected 32.

    The parent cohort here has full audited metadata, not full payloads. Never
    call this a full geometry audit: formal mode uses audit_geometry_index.
    """
    import prepare_geometry_cache as geo
    from evaluate_pose_cache import canonical_hash
    contract = geo._contract(full_metadata_cohort)
    digest = sha256(path)
    index = json.loads(path.read_text())
    if (index.get('schema_version') != 1 or index.get('status') != 'complete' or
            index.get('contract') != contract or
            len(index.get('samples', [])) != len(full_metadata_cohort['samples']) or
            json.loads((path.parent / 'cache_contract.json').read_text()) != contract):
        raise ValueError('Engineering geometry full metadata contract differs')
    by_id = {}
    for entry, item in zip(index['samples'], full_metadata_cohort['samples']):
        if geo._identity(entry) != geo._identity(item):
            raise ValueError('Engineering geometry index identity/order mismatch')
        by_id[item['index']] = entry
    checked = {}
    for item in selected:
        raw = geo._raw_descriptor(full_metadata_cohort, item, verify=True)
        key = canonical_hash(geo._sample_contract(item, contract, raw['frames']))
        actual = geo._audit_entry(path.parent / 'samples' / key[:2] / key, item, contract, raw)
        if actual != by_id[item['index']]:
            raise ValueError('Engineering geometry payload differs from original full index')
        checked[item['index']] = {**actual, **raw}
    if sha256(path) != digest or geo._contract(full_metadata_cohort) != contract:
        raise ValueError('Engineering geometry input changed during audit')
    return {'path': str(path), 'sha256': digest, 'contract': contract,
            'by_index': checked, 'payload_scope': 'engineering32_only'}


def check_reference_prediction(record, class_ids, reference):
    """Historical full logits, softmax ranks/probabilities and source identity."""
    keys = ('index', 'gloss', 'signer_id', 'split', 'source_path', 'video_sha256')
    if reference.get('status') != 'ok' or any(record[key] != reference.get(key) for key in keys):
        raise ValueError('Historical validation identity/order differs')
    top5 = reference.get('top5', [])
    if len(top5) != 5 or list(class_ids) != [entry.get('class_id') for entry in top5]:
        raise ValueError('Historical validation Top5 ranks differ')
    if record['top5_probabilities'] != [entry.get('probability') for entry in top5]:
        raise ValueError('Historical validation softmax probabilities differ')
    if record['logits_sha256'] != reference.get('logits_sha256'):
        raise ValueError('Historical validation full-logit SHA256 differs')


def verify_fingerprints(fingerprints):
    for name, expected in fingerprints.items():
        if sha256(inside(name)) != expected:
            raise ValueError('Frozen source or input changed: ' + name)


def run(args):
    legacy.validate_args(args)
    prereg_path = inside(args.preregistration)
    prereg = load_preregistration(prereg_path, args.preregistration_sha256)
    index_path = inside(prereg['data'][f'{args.split}_cache_manifest'])
    geometry_path = inside(prereg['data'][f'{args.split}_geometry_manifest'])
    for path, digest in ((index_path, prereg['data'][f'{args.split}_cache_sha256']),
                         (geometry_path, prereg['data'][f'{args.split}_geometry_sha256'])):
        if sha256(path) != digest:
            raise ValueError('Fixed input index SHA256 differs')
    output = inside(args.output_dir)
    protected = [prereg_path, index_path, geometry_path,
                 *(inside(entry['checkpoint_path']) for entry in prereg['models'])]
    if output == ROOT or any(output == path.parent or output in path.parents for path in protected):
        raise ValueError('Output must be separate from frozen input directories')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    engineering = args.engineering_count is not None
    common = {'schema_version': 1, 'protocol': PROTOCOL,
              'scope': 'engineering' if engineering else 'formal', 'status': 'running',
              'split': args.split, 'prereg_sha256': args.preregistration_sha256,
              'created_at_utc': now(), 'test_used': False, 'optimizer_steps': 0, 'failed': 0,
              'weights_unchanged': False, 'feature_dim': 128, 'class_ids': list(range(2731)),
              'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    manifest = {**common, 'stage': 'load_and_metadata_audit', 'models': {}}
    write_json(output / 'manifest.json', manifest)
    submanifests = {}
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch
        from _poet_runtime import resolve_device
        from evaluate_pose_cache import audit_provenance, validate_cache_index, OFFICIAL_LABEL_MAP_SHA256
        from finetune_baseline import audit_cohort, tensor_state_hash
        from prepare_geometry_cache import audit_geometry_index, read_geometry
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20260926)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        models = load_frozen_models(prereg['models'], resolve_device(args.device))
        first_bundle = models[MODEL_IDS[0]]['bundle']
        index = json.loads(index_path.read_text())
        split = validate_cache_index(index, first_bundle.label_to_id, allow_train_smoke=True)
        if split != args.split:
            raise ValueError('Requested split differs from source cache')
        provenance = audit_provenance(index_path, index, first_bundle.checkpoint['data_preprocessing'],
                                      first_bundle.label_to_id, split, allow_subset_val=False)
        if not provenance['complete_official_split']:
            raise ValueError('Full official source cohort required even for engineering')
        legacy.validate_rows(index['samples'], split, first_bundle.label_to_id, complete=True)
        source_paths = [inside(name) for name in SOURCE_FILES]
        source_paths += [inside(name) for name in provenance['pipeline']['source_sha256']]
        source_paths += [prereg_path, index_path, geometry_path, geometry_path.parent / 'cache_contract.json',
                         ROOT / 'assets/checkpoints/ASL_Citizen_PoET.pth',
                         *(ROOT / 'assets/checkpoints' / name for name in OFFICIAL_LABEL_MAP_SHA256),
                         inside(provenance['source_manifest_path']), inside(provenance['official_split_path'])]
        for entry in prereg['models']:
            source_paths += [inside(entry['checkpoint_path']), inside(entry['reference_predictions_path'])]
        fingerprints = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(source_paths))}
        references = {}
        for entry in prereg['models']:
            if fingerprints[str(inside(entry['checkpoint_path']).relative_to(ROOT))] != entry['checkpoint_sha256']:
                raise ValueError('Checkpoint changed during model loading')
            reference_path = inside(entry['reference_predictions_path'])
            if fingerprints[str(reference_path.relative_to(ROOT))] != entry['reference_predictions_sha256']:
                raise ValueError('Historical prediction reference SHA256 differs')
            if split == 'val':
                references[entry['model_id']] = [json.loads(line) for line in reference_path.read_text().splitlines()]
                if len(references[entry['model_id']]) != 10304:
                    raise ValueError('Historical validation reference has incomplete denominator')
        if fingerprints[str(prereg_path.relative_to(ROOT))] != args.preregistration_sha256:
            raise ValueError('Preregistration changed during model loading')
        manifest.update(stage='cache_audit', source_sha256=fingerprints,
                        geometry_manifest_sha256=prereg['data'][f'{split}_geometry_sha256'])
        write_json(output / 'manifest.json', manifest)
        if engineering:
            items = legacy.engineering_selection(index['samples'])
            legacy.audit_engineering_cache(index_path, index, items, first_bundle.preprocessing)
            metadata_cohort = {'path': str(index_path), 'sha256': sha256(index_path), 'split': split,
                               'samples': index['samples'], 'provenance': provenance}
            geometry_index = audit_engineering_geometry(geometry_path, metadata_cohort, items)
        else:
            cohort = audit_cohort(index_path, first_bundle, engineering=False,
                                  role='train' if split == 'train' else 'validation')
            items = cohort['samples']
            geometry_index = audit_geometry_index(geometry_path, cohort)
        if sha256(index_path) != prereg['data'][f'{split}_cache_sha256'] or geometry_index['sha256'] != prereg['data'][f'{split}_geometry_sha256']:
            raise ValueError('Cache indexes changed during shared audit')
        legacy.validate_rows(items, split, first_bundle.label_to_id, complete=not engineering)
        all_arrays = {}
        for model_id, loaded in models.items():
            entry = loaded['entry']
            directory = output / model_id
            directory.mkdir()
            sub = {**common, 'split': split, 'stage': 'extracting', 'model_id': model_id,
                   'method': entry['method'], 'seed': entry['seed'],
                   'checkpoint_sha256': entry['checkpoint_sha256'], 'checkpoint_epoch': entry['checkpoint_epoch'],
                   'model_state_sha256': entry['model_state_sha256'], 'training_protocol_sha256': entry['protocol_sha256'],
                   'weights_state_sha256_before': tensor_state_hash(loaded['before']),
                   'geometry_manifest_sha256': geometry_index['sha256'] if entry['method'] == 'geometry' else None,
                   'label_map': {str(i): first_bundle.id_to_label[i] for i in range(2731)},
                   'source_sha256': fingerprints, 'method_metadata': loaded['method_metadata'], 'splits': {},
                   'extraction': {'dtype': 'float32', 'allow_tf32': False, 'batch_size': 128, 'num_threads': 4,
                                  'eval_mode': True, 'augmentation': False,
                                  'embedding': 'actual final 128D input to model.base.mlp_classifier',
                                  'engineering_selection': legacy.ENGINEERING_SALT + 'video_sha256; first 32 ascending hashes'}}
            submanifests[model_id] = sub
            write_json(directory / 'manifest.json', sub)
            all_arrays[model_id] = {'embeddings': np.empty((len(items), 128), np.float32),
                                    'top5_class_ids': np.empty((len(items), 5), np.int64),
                                    'top5_logits': np.empty((len(items), 5), np.float32)}
        manifest.update(stage='extracting', expected_samples=len(items), progress={'completed': 0, 'total': len(items)})
        write_json(output / 'manifest.json', manifest)
        with ExitStack() as stack:
            streams = {model_id: stack.enter_context((output / model_id / 'records.jsonl').open('x', encoding='utf-8'))
                       for model_id in models}
            for offset in range(0, len(items), args.batch_size):
                batch_items = items[offset:offset + args.batch_size]
                data = [legacy.read_evaluation_input(item, first_bundle.preprocessing) for item in batch_items]
                prepared = np.stack([pair[0] for pair in data])
                geometry = np.stack([read_geometry(item, pair[1], geometry_index) for item, pair in zip(batch_items, data)])
                for model_id, loaded in models.items():
                    method_geometry = geometry if loaded['entry']['method'] == 'geometry' else None
                    extracted, details = extract_method_batch(loaded['model'], prepared, method_geometry, args.device,
                                                               verify_unhooked=engineering)
                    if engineering:
                        repeated, repeated_details = extract_method_batch(loaded['model'], prepared, method_geometry, args.device)
                        if any(not np.array_equal(extracted[key], repeated[key]) for key in extracted) or details != repeated_details:
                            raise ValueError('Repeated frozen engineering extraction is not bitwise exact: ' + model_id)
                        submanifests[model_id].update(engineering_repeat_exact=True, engineering_unhooked_logits_exact=True)
                    arrays = all_arrays[model_id]
                    for key in arrays:
                        arrays[key][offset:offset + len(batch_items)] = extracted[key]
                    for local_row, (item, (model_input, indices)) in enumerate(zip(batch_items, data)):
                        record = {key: item[key] for key in ('index', 'identity', 'gloss', 'signer_id', 'split',
                                  'source_path', 'video_sha256', 'pose_sha256', 'features_sha256')}
                        record.update(row=offset + local_row, class_id=first_bundle.label_to_id[item['gloss']],
                                      model_id=model_id, checkpoint_sha256=loaded['entry']['checkpoint_sha256'],
                                      input_sha256=array_sha256(model_input),
                                      geometry_input_sha256=array_sha256(geometry[local_row]) if method_geometry is not None else None,
                                      source_frame_indices_sha256=array_sha256(indices), source_frame_indices=indices.tolist(),
                                      cache_manifest_sha256=provenance['cache_manifest_sha256'],
                                      source_manifest_sha256=provenance['source_manifest_sha256'],
                                      logits_sha256=details['logits_sha256'][local_row],
                                      top5_probabilities=details['top5_probabilities'][local_row])
                        if split == 'val':
                            check_reference_prediction(record, extracted['top5_class_ids'][local_row].tolist(),
                                                       references[model_id][offset + local_row])
                        streams[model_id].write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
                    streams[model_id].flush()
                manifest['progress'] = {'completed': offset + len(batch_items), 'total': len(items),
                                        'models_per_batch': 6, 'elapsed_seconds': time.perf_counter() - started}
                write_json(output / 'manifest.json', manifest)
                if offset // args.batch_size % 10 == 0 or offset + len(batch_items) == len(items):
                    print(json.dumps(manifest['progress']), flush=True)
        verify_fingerprints(fingerprints)
        for model_id, loaded in models.items():
            arrays, directory = all_arrays[model_id], output / model_id
            legacy.validate_extracted_arrays(arrays, len(items))
            legacy.verify_frozen_model(loaded['model'], loaded['before'])
            state_sha = tensor_state_hash(loaded['model'].state_dict())
            if state_sha != loaded['entry']['model_state_sha256']:
                raise ValueError('Frozen model state hash changed: ' + model_id)
            with (directory / 'features.npz').open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            with np.load(directory / 'features.npz', allow_pickle=False) as saved:
                legacy.validate_extracted_arrays({key: saved[key] for key in saved.files}, len(items))
                if any(not np.array_equal(arrays[key], saved[key]) for key in arrays):
                    raise ValueError('Saved features differ from memory: ' + model_id)
            records = [json.loads(line) for line in (directory / 'records.jsonl').read_text().splitlines()]
            if (len(records) != len(items) or [record['row'] for record in records] != list(range(len(items))) or
                    any(record['model_id'] != model_id or record['checkpoint_sha256'] != loaded['entry']['checkpoint_sha256'] for record in records)):
                raise ValueError('Saved metadata identity/count/order mismatch: ' + model_id)
            entry = {'samples': len(items), 'failed': 0, 'features_path': 'features.npz',
                     'features_sha256': sha256(directory / 'features.npz'), 'features_bytes': (directory / 'features.npz').stat().st_size,
                     'metadata_path': 'records.jsonl', 'metadata_sha256': sha256(directory / 'records.jsonl'),
                     'metadata_bytes': (directory / 'records.jsonl').stat().st_size,
                     'source_manifest_sha256': provenance['source_manifest_sha256'],
                     'cache_manifest_sha256': provenance['cache_manifest_sha256'],
                     'provenance': provenance, 'complete_official_split': not engineering,
                     'signers': sorted({item['signer_id'] for item in items}),
                     'array_schema': {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': array_sha256(value)}
                                      for key, value in arrays.items()}}
            submanifests[model_id].update(status='completed', stage='completed', splits={split: entry},
                                          weights_unchanged=True, weights_state_sha256_after=state_sha,
                                          gradients_absent=True, model_eval=True, completed_at_utc=now(),
                                          reference_logits_exact=True if split == 'val' else None,
                                          reference_top5_exact=True if split == 'val' else None,
                                          elapsed_seconds=time.perf_counter() - started)
            write_json(directory / 'manifest.json', submanifests[model_id])
            manifest['models'][model_id] = {'manifest_path': f'{model_id}/manifest.json',
                                           'manifest_sha256': sha256(directory / 'manifest.json'),
                                           'checkpoint_sha256': loaded['entry']['checkpoint_sha256'], 'samples': len(items)}
        verify_fingerprints(fingerprints)
        manifest.update(status='completed', stage='completed', weights_unchanged=True, gradients_absent=True,
                        model_eval=True, completed_at_utc=now(), elapsed_seconds=time.perf_counter() - started,
                        shared_input_pass=True, engineering_repeat_exact=True if engineering else None,
                        reference_logits_exact=True if split == 'val' else None,
                        reference_top5_exact=True if split == 'val' else None,
                        engineering_unhooked_logits_exact=True if engineering else None)
        write_json(output / 'manifest.json', manifest)
        print(json.dumps({'status': 'completed', 'split': split, 'samples_per_model': len(items), 'models': 6,
                          'failed': 0, 'elapsed_seconds': manifest['elapsed_seconds']}), flush=True)
        return manifest
    except BaseException as error:
        failure = {'status': 'failed', 'failed': 1, 'failed_at_utc': now(), 'error': str(error),
                   'error_type': type(error).__name__, 'elapsed_seconds': time.perf_counter() - started}
        manifest.update(failure)
        write_json(output / 'manifest.json', manifest)
        for model_id, sub in submanifests.items():
            if sub['status'] != 'completed':
                sub.update(failure)
                write_json(output / model_id / 'manifest.json', sub)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--preregistration-sha256', required=True)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--engineering-count', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
