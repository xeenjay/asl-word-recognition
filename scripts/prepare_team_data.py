#!/usr/bin/env python3
"""Obtain official ASL Citizen train/val clips and build local classical features.

Default: ten alphabetically selected glosses, up to six training and two
validation clips each, selected by a fixed filename hash without model results.
No dataset payload is distributed by this repository. Downloading requires the
user to read and accept the original dataset terms. Existing personal copies
can be supplied with --source-dir (containing splits/ and videos/).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

import asl_citizen_source as source
from classical_features import extract_features, FEATURE_DIM, VERSION

ROOT = Path(__file__).resolve().parents[1]
LICENSE_URL = 'https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/'
SPLIT_HASHES = {
    'train': '87a105649b98ea577182ebfce49d59fe9ca834a2489384f6fe7bc03f14966329',
    'val': '6ac78c9d551dfcc81d225a5279d7637c2891a4490851234be6e4674666e21366',
}
EXTRACTOR = 'team-holistic-static-complexity1-v1'
SELECTION_SEED = 20261004


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value: dict | list) -> None:
    source.atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=False,
                                         allow_nan=False) + '\n').encode('utf-8'))


def npy_write(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    source.atomic_write(path, buffer.getvalue())


def split_rows(raw: dict[str, bytes]) -> dict[str, list[dict]]:
    rows = {}
    all_names: set[str] = set()
    signers: dict[str, set[str]] = {}
    for split, expected in [('train', 40154), ('val', 10304)]:
        if hashlib.sha256(raw[split]).hexdigest() != SPLIT_HASHES[split]:
            raise ValueError(f'{split}.csv differs from the pinned official split')
        rows[split] = list(csv.DictReader(io.StringIO(raw[split].decode('utf-8-sig'))))
        names = {row['Video file'] for row in rows[split]}
        if len(rows[split]) != expected or len(names) != expected or names & all_names:
            raise ValueError('Official sample counts/identities differ or overlap')
        if any(Path(name).name != name or '\\' in name for name in names):
            raise ValueError('Unsafe video filename in split')
        signers[split] = {row['Participant ID'] for row in rows[split]}
        all_names.update(names)
    if signers['train'] & signers['val']:
        raise ValueError('Training and validation signers overlap')
    return rows


def select_samples(rows: dict[str, list[dict]], profile: str) -> tuple[list[str], dict, list[int]]:
    vocabulary = sorted({row['Gloss'] for row in rows['train']})
    if set(vocabulary) != {row['Gloss'] for row in rows['val']}:
        raise ValueError('Training and validation vocabularies differ')
    if profile not in {'mini', 'full'}:
        raise ValueError('Only mini and full train/validation profiles are supported')
    labels = vocabulary[:10] if profile == 'mini' else vocabulary
    selected = {}
    for split in ('train', 'val'):
        groups: dict[str, list] = {label: [] for label in labels}
        for row in rows[split]:
            if row['Gloss'] in groups:
                groups[row['Gloss']].append(row)
        selected[split] = []
        for label in labels:
            ordered = sorted(groups[label], key=lambda row: (
                hashlib.sha256(f"{SELECTION_SEED}\0{row['Video file']}".encode()).hexdigest(),
                row['Video file']))
            limit = (6 if split == 'train' else 2) if profile == 'mini' else len(ordered)
            selected[split].extend(ordered[:limit])
    return labels, selected, [vocabulary.index(label) for label in labels]


def extract_video(video: Path) -> tuple[np.ndarray, dict]:
    """Read every frame; absent landmarks use masks, corrupt videos fail closed.

    This portable extractor uses the bundled complexity-1 pose model. Historical
    research used a complexity-2 extractor, so its scores are not reproduced by
    merely using the same descriptor formula.
    """
    import cv2
    import mediapipe as mp

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {video.name}')
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    parts = {'body_pose': (33, 'pose_landmarks'),
             'left_hand': (21, 'left_hand_landmarks'),
             'right_hand': (21, 'right_hand_landmarks')}
    values: dict[str, list] = {key: [] for key in parts}
    masks: dict[str, list] = {key: [] for key in parts}
    visibility = []
    try:
        with mp.solutions.holistic.Holistic(
                static_image_mode=True, model_complexity=1,
                smooth_landmarks=False, min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = holistic.process(rgb)
                for key, (points, attribute) in parts.items():
                    landmarks = getattr(result, attribute)
                    masks[key].append(landmarks is not None)
                    values[key].append(np.asarray(
                        [[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float64)
                        if landmarks is not None else np.zeros((points, 3)))
                body = result.pose_landmarks
                visibility.append(np.asarray([p.visibility for p in body.landmark])
                                  if body is not None else np.zeros(33))
    finally:
        capture.release()
    frames = len(visibility)
    if frames == 0 or expected <= 0 or frames != expected:
        raise ValueError(f'{video.name}: decoded {frames}/{expected} frames; no sample was dropped')
    raw = {key: np.stack(value) for key, value in values.items()}
    raw.update({key + '_present': np.asarray(value, dtype=bool) for key, value in masks.items()})
    raw['body_visibility'] = np.stack(visibility)
    features = extract_features(raw)
    return features, {'decoded_frames': frames, 'reported_frames': expected,
                      'body_frames': int(sum(masks['body_pose'])),
                      'left_hand_frames': int(sum(masks['left_hand'])),
                      'right_hand_frames': int(sum(masks['right_hand']))}


def cached_features(video: Path, video_sha256: str, cache: Path) -> tuple[np.ndarray, dict]:
    path, metadata_path = cache / (video_sha256 + '.npy'), cache / (video_sha256 + '.json')
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('sha256') == digest(path):
            array = np.load(path, allow_pickle=False)
            if array.shape == (FEATURE_DIM,) and array.dtype == np.float32 and np.isfinite(array).all():
                return array, metadata['extraction']
        raise ValueError('Feature cache integrity differs; use a new --cache-dir to rebuild')
    features, info = extract_video(video)
    npy_write(path, features)
    json_write(metadata_path, {'video_sha256': video_sha256, 'sha256': digest(path),
                               'extraction': info})
    return features, info


def prepare(args: argparse.Namespace) -> dict:
    if args.source_dir is None and not args.accept_dataset_license:
        raise ValueError('Before downloading, read ' + LICENSE_URL +
                         ' and, if you accept it, rerun with --accept-dataset-license. '
                         'An existing personal copy can be used via --source-dir.')
    output = (args.output or ROOT / 'data' / ('classical-' + args.profile)).resolve()
    cache = args.cache_dir.resolve()
    started = time.perf_counter()
    session = None
    reader = None
    videos = None
    versions = {}
    if not args.plan_only:
        for package in ('numpy', 'mediapipe', 'opencv-contrib-python'):
            versions[package] = importlib.metadata.version(package)
        if versions['mediapipe'] != '0.10.21':
            raise ValueError('Use the pinned mediapipe==0.10.21 from requirements-data.txt')
    raw = {}
    try:
        if args.source_dir is not None:
            for split in ('train', 'val'):
                raw[split] = (args.source_dir / 'splits' / (split + '.csv')).read_bytes()
            video_directory = args.source_dir / 'videos'
        else:
            import requests
            session = requests.Session()
            budget = 512 * 1024**2 if args.profile == 'mini' else 48 * 1024**3
            reader = source.RangeReader(session, max_download_bytes=budget)
            buffers = source.load_metadata(reader, cache / 'official-metadata')
            directory = source.parse_directory(buffers['central'])
            raw_all, _, videos = source.parse_splits(buffers, directory)
            raw = {split: raw_all[split] for split in ('train', 'val')}
            video_directory = cache / 'videos'
        rows = split_rows(raw)
        labels, selected, original_ids = select_samples(rows, args.profile)
        selection = {'profile': args.profile, 'class_order': 'sorted official training gloss strings',
                     'class_count': len(labels), 'seed': SELECTION_SEED,
                     'row_order': 'within each class ascending SHA256(seed + NUL + filename)',
                     'train_per_class_limit': 6 if args.profile == 'mini' else None,
                     'val_per_class_limit': 2 if args.profile == 'mini' else None}
        planned = {'profile': args.profile, 'classes': len(labels),
                   'train': len(selected['train']), 'val': len(selected['val']),
                   'test_used': False, 'extractor': EXTRACTOR}
        if videos is not None:
            planned['selected_video_bytes'] = sum(videos[row['Video file']]['size']
                                                   for split in selected for row in selected[split])
            planned['metadata_network_bytes'] = reader.bytes_received
        if args.plan_only:
            print(json.dumps(planned, indent=2))
            return planned
        code_hashes = {name: digest(ROOT / 'scripts' / name) for name in (
            'prepare_team_data.py', 'asl_citizen_source.py', 'classical_features.py')}
        recipe = {'selection': selection, 'source_sha256': {**SPLIT_HASHES, **code_hashes},
                  'extractor': EXTRACTOR, 'packages': versions}
        contract = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        contract_path = output / 'preparation.json'
        if output.exists():
            if not contract_path.is_file():
                raise ValueError('Output already exists without a preparation record; use a new --output')
            previous = json.loads(contract_path.read_text(encoding='utf-8'))
            if previous.get('contract_sha256') != contract:
                raise ValueError('Data preparation recipe changed; use a new --output and preserve the old data')
            manifest_path = output / 'manifest.json'
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                for name, item in manifest['files'].items():
                    if Path(name).name != name or digest(output / name) != item['sha256']:
                        raise ValueError('Prepared data integrity differs; use a new --output')
                print('Verified existing data: ' + str(output), flush=True)
                return manifest
        output.mkdir(parents=True, exist_ok=True)
        json_write(contract_path, {'contract_sha256': contract, 'recipe': recipe})
        for split, contents in raw.items():
            source.atomic_write(cache / 'splits' / (split + '.csv'), contents)
        prepared_cache = cache / 'features' / contract
        label_ids = {label: index for index, label in enumerate(labels)}
        selected_manifest = {'selected_glosses': labels, 'selection': selection,
                             'source_sha256': recipe['source_sha256'], 'splits': {}}
        details = {'train': [], 'val': []}
        arrays = {}
        print('Preparing ' + json.dumps(planned) + '. First extraction takes time; later runs reuse the cache.', flush=True)
        for split in ('train', 'val'):
            features, targets, manifest_rows = [], [], []
            for index, row in enumerate(selected[split]):
                filename = row['Video file']
                video = video_directory / filename
                if videos is not None:
                    record = videos[filename]
                    video_hash = source.reusable_video(video, record, None)
                    if video_hash is None:
                        print(f'{split} {index + 1}/{len(selected[split])}: downloading official clip', flush=True)
                        source.atomic_write(video, source.download_entry(reader, record))
                        video_hash = digest(video)
                else:
                    if not video.is_file():
                        raise ValueError(f'Missing personal-copy video: {filename}; no sample was dropped')
                    video_hash = digest(video)
                print(f'{split} {index + 1}/{len(selected[split])}: extracting {row["Gloss"]}', flush=True)
                vector, extraction = cached_features(video, video_hash, prepared_cache)
                features.append(vector)
                targets.append(label_ids[row['Gloss']])
                manifest_rows.append({'video_file': filename, 'gloss': row['Gloss'],
                                      'video_sha256': video_hash})
                details[split].append(extraction)
            arrays[split + '_features.npy'] = np.stack(features).astype(np.float32)
            arrays[split + '_labels.npy'] = np.asarray(targets, dtype=np.int64)
            selected_manifest['splits'][split] = manifest_rows
            json_write(output / 'selected-manifest.json', selected_manifest)
        for name, array in arrays.items():
            npy_write(output / name, array)
        json_write(output / 'labels.json', labels)
        files = {name: {'sha256': digest(output / name), 'bytes': (output / name).stat().st_size,
                        'shape': list(array.shape), 'dtype': str(array.dtype)}
                 for name, array in arrays.items()}
        files['labels.json'] = {'sha256': digest(output / 'labels.json'),
                                'bytes': (output / 'labels.json').stat().st_size}
        manifest = {'schema_version': 1, 'status': 'completed', 'profile': args.profile,
                    'dataset': 'ASL Citizen', 'feature_spec': VERSION, 'feature_dim': FEATURE_DIM,
                    'source_run_id': 'local-team-preparation', 'source_url': source.URL,
                    'license_url': LICENSE_URL, 'source_sha256': recipe['source_sha256'],
                    'selected_original_class_ids': original_ids,
                    'original_class_id_space': 'alphabetically sorted official training gloss strings',
                    'subset': {'name': 'mini-10' if args.profile == 'mini' else 'full',
                               'selection': selection}, 'splits': {split: {'count': len(selected[split])}
                                                                  for split in ('train', 'val')},
                    'test_used': False, 'files': files, 'extractor': EXTRACTOR,
                    'extractor_settings': {'static_image_mode': True, 'model_complexity': 1,
                                           'all_frames': True, 'flip': False,
                                           'min_detection_confidence': 0.5},
                    'packages': versions, 'extraction': details,
                    'selected_manifest_sha256': digest(output / 'selected-manifest.json'),
                    'elapsed_seconds': time.perf_counter() - started,
                    'download_bytes_this_run': reader.bytes_received if reader else 0,
                    'historical_research_scores_reproduced': False}
        json_write(output / 'manifest.json', manifest)
        print('Ready: ' + str(output), flush=True)
        print('Run: python run.py --method knn --data ' + str(output), flush=True)
        return manifest
    finally:
        if session is not None:
            session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('mini', 'full'), default='mini')
    parser.add_argument('--accept-dataset-license', action='store_true',
                        help='You have read and accept the original Microsoft dataset terms')
    parser.add_argument('--source-dir', type=Path,
                        help='Existing personal ASL Citizen folder containing splits/ and videos/')
    parser.add_argument('--output', type=Path, help='New output directory, or same recipe to resume')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'data' / 'team-cache')
    parser.add_argument('--plan-only', action='store_true', help='Inspect selection, without videos or extraction')
    args = parser.parse_args()
    try:
        prepare(args)
    except (ValueError, OSError, source.AuditError, importlib.metadata.PackageNotFoundError) as error:
        print('Data preparation stopped: ' + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
