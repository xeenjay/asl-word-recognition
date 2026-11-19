#!/usr/bin/env python3
"""Preregistered, bounded validation tuning of the three classical baselines.

Each invocation writes into a NEW directory (normally run_logged's
``{run_dir}/tuning``). The test split is never accepted; historical runs and the
frozen classical sources are only read.

  prepare  Fit the train-only StandardScaler on a completed train/validation
           feature dataset; save scaled matrices, rows and a plan with SHA256s.
  fit      Fit ONE declared grid point on a prepared directory, predict the full
           validation split, save/reload/replay the model and record metrics.

The grid, selection rule and reporting are fixed by
docs/environment/classical-tuning-preregistration-20261011.json; grid points are
looked up by ID and ad-hoc parameters are rejected. Fitting, prediction and
serialization reuse classical_models. Only two additions live here: the
similarity-weighted kNN vote, and a forest fit path whose tree-storage budget
(16 GiB) and max_depth=None differ from the frozen 2 GiB / integer-depth guard.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time
import warnings

import numpy as np

from classical_models import (DEFAULT_PARAMETERS, ClassicalClassifier, _features, _labels, _positive_int,
                              fit_classifier, fit_train_scaler, forest_memory_upper_bound, load_model,
                              predict_topk, save_model, transform_features)
from evaluate_pose_cache import MetricAccumulator, sha256, verify_file, write_json

WORKSPACE = Path(__file__).resolve().parents[1]
IDENTITY_KEYS = ('index', 'gloss', 'signer_id', 'split', 'source_path', 'video_sha256')
PREPARED_NAMES = ('train_scaled.npy', 'query_scaled.npy', 'train_labels.npy', 'scaler.joblib',
                  'query_rows.json', 'train_rows.json', 'dataset.json')
SOURCE_NAMES = ('scripts/run_classical_tuning.py', 'scripts/verify_classical_tuning.py',
                'scripts/classical_models.py', 'scripts/classical_features.py',
                'scripts/evaluate_pose_cache.py', 'scripts/verify_classical_results.py')
KNN_K = (1, 3, 5, 10, 20)
KNN_WEIGHTING = ('uniform', 'similarity')
SVM_C = (0.001, 0.003, 0.01, 0.03, 0.1)
RF_POINTS = ((64, 256, 12), (64, 256, None), (64, 1024, None), (128, 1024, None), (64, 4096, None))
ORIGINAL_POINTS = {'knn': 'knn-k05-uniform', 'linear_svm': 'svm-c0p01', 'random_forest': 'rf-t064-l0256-d12'}
FOREST_STORAGE_BUDGET_BYTES = 16 * 1024**3
TOP_K = 5
KNN_BATCH, KNN_CHUNK, OTHER_BATCH = 128, 8192, 128


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def build_grid() -> dict:
    """Grid points in the preregistered simplicity order (earlier = simpler)."""
    grid = {}
    for weighting in KNN_WEIGHTING:
        for k in KNN_K:
            grid[f'knn-k{k:02d}-{weighting}'] = {'method': 'knn', 'parameters': {'n_neighbors': k},
                                                 'vote_weighting': weighting}
    for c in SVM_C:
        grid['svm-c' + format(c, 'g').replace('.', 'p')] = {
            'method': 'linear_svm', 'parameters': {'C': c, 'parallel_ovr': True, 'n_jobs': 8}}
    for trees, leaves, depth in RF_POINTS:
        grid[f'rf-t{trees:03d}-l{leaves:04d}-d{depth if depth else "none"}'] = {
            'method': 'random_forest',
            'parameters': {'n_estimators': trees, 'max_leaf_nodes': leaves, 'max_depth': depth, 'n_jobs': 8}}
    return grid


GRID = build_grid()


def weighted_vote(labels: list[int], similarities: list[float], top_k: int = TOP_K):
    """Preregistered similarity vote for one query; returns (ids, scores, fallback)."""
    first = {}
    for rank, label in enumerate(labels):
        first.setdefault(label, rank)
    weights = [s if s > 0.0 else 0.0 for s in similarities]
    total = 0.0
    for weight in weights:
        total += weight
    if total > 0.0:
        score = {}
        for label, weight in zip(labels, weights):
            score[label] = score.get(label, 0.0) + weight
        ranked = sorted((c for c, v in score.items() if v > 0.0), key=lambda c: (-score[c], first[c], c))[:top_k]
        return ranked, [score[c] / total for c in ranked], False
    counts = Counter(labels)
    ranked = sorted(counts, key=lambda c: (-counts[c], first[c], c))[:top_k]
    return ranked, [counts[c] / len(labels) for c in ranked], True


def similarity_votes(neighbors: np.ndarray, similarities: np.ndarray, reference_labels: np.ndarray,
                     top_k: int = TOP_K) -> dict:
    count = len(neighbors)
    ids = np.full((count, top_k), -1, np.int64)
    scores = np.zeros((count, top_k), np.float64)
    valid = np.zeros((count, top_k), bool)
    support = np.zeros(count, np.int64)
    fallback = np.zeros(count, bool)
    for row in range(count):
        labels = [int(reference_labels[j]) for j in neighbors[row]]
        ranked, values, fallback[row] = weighted_vote(labels, [float(s) for s in similarities[row]], top_k)
        ids[row, :len(ranked)] = ranked
        scores[row, :len(ranked)] = values
        valid[row, :len(ranked)] = True
        support[row] = len(ranked)
    return {'topk_indices': ids, 'topk_scores': scores, 'valid_mask': valid, 'support_count': support,
            'similarity_fallback_mask': fallback,
            'score_kind': 'similarity_weighted_neighbor_vote_fraction',
            'rows_with_fewer_than_top_k': int((valid.sum(axis=1) < top_k).sum())}


def predict_point(model: ClassicalClassifier, X_query: np.ndarray, device: str) -> dict:
    """Predict with a tuned model; the kNN vote rule is stored in model.parameters."""
    if model.method == 'knn':
        result = predict_topk(model, X_query, top_k=TOP_K, device=device,
                              batch_size=KNN_BATCH, reference_chunk_size=KNN_CHUNK)
        weighting = model.parameters.get('vote_weighting')
        if weighting == 'similarity':
            result.update(similarity_votes(result['neighbor_indices'], result['neighbor_similarities'],
                                           model.reference_labels))
        elif weighting != 'uniform':
            raise ValueError(f'Unknown kNN vote_weighting: {weighting!r}')
        return result
    return predict_topk(model, X_query, top_k=TOP_K, device='cpu', batch_size=OTHER_BATCH)


def fit_forest(X_train, y_train, num_classes: int, parameters: dict,
               budget_bytes: int = FOREST_STORAGE_BUDGET_BYTES) -> ClassicalClassifier:
    """Mirror of classical_models' forest branch with a declared larger budget.

    max_depth may be None so that max_leaf_nodes alone bounds capacity/memory.
    """
    from sklearn.ensemble import RandomForestClassifier
    import sklearn
    c = _positive_int(num_classes, 'num_classes')
    X = _features(X_train)
    y = _labels(y_train, len(X), c)
    unexpected = set(parameters) - set(DEFAULT_PARAMETERS['random_forest'])
    if unexpected:
        raise ValueError(f'Unsupported parameters: {sorted(unexpected)}')
    config = {**DEFAULT_PARAMETERS['random_forest'], **parameters}
    trees = _positive_int(config['n_estimators'], 'n_estimators')
    leaves = _positive_int(config['max_leaf_nodes'], 'max_leaf_nodes')
    jobs = _positive_int(config['n_jobs'], 'n_jobs')
    depth = config['max_depth']
    if depth is not None:
        _positive_int(depth, 'max_depth')
    if leaves < 2 or jobs > 8:
        raise ValueError('Forest requires max_leaf_nodes>=2 and n_jobs<=8')
    estimate = forest_memory_upper_bound(c, trees, leaves)
    if estimate > budget_bytes:
        raise ValueError('Forest tree storage estimate exceeds the preregistered budget')
    info = {'train_rows': len(X), 'feature_dim': X.shape[1], 'num_classes': c,
            'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
            'warnings': [], 'converged': None, 'tree_storage_upper_bound_bytes': estimate,
            'tree_storage_budget_bytes': budget_bytes,
            'fit_path': 'run_classical_tuning.fit_forest (frozen fit_classifier forest branch with a 16 GiB budget and optional max_depth=None)'}
    estimator = RandomForestClassifier(**config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        estimator.fit(X, y)
    info['warnings'] = [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]
    for warning in caught:
        warnings.warn(str(warning.message), warning.category, stacklevel=2)
    if not np.array_equal(estimator.classes_, np.arange(c)):
        raise ValueError('Estimator class mapping differs from declared mapping')
    depths = [int(tree.get_depth()) for tree in estimator.estimators_]
    leaf_counts = [int(tree.get_n_leaves()) for tree in estimator.estimators_]
    info.update(total_tree_nodes=sum(int(t.tree_.node_count) for t in estimator.estimators_),
                tree_value_bytes=sum(int(t.tree_.value.nbytes) for t in estimator.estimators_),
                tree_depth_min=min(depths), tree_depth_max=max(depths),
                tree_leaves_min=min(leaf_counts), tree_leaves_max=max(leaf_counts),
                trees_at_leaf_budget=sum(n == leaves for n in leaf_counts),
                trees_at_depth_cap=None if depth is None else sum(d == depth for d in depths))
    return ClassicalClassifier('random_forest', c, X.shape[1], config, estimator=estimator, fit_info=info)


def mem_available_bytes() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable not found')


def fit_point(point: dict, X, y, num_classes: int) -> ClassicalClassifier:
    method, parameters = point['method'], dict(point['parameters'])
    if method == 'knn':
        model = fit_classifier('knn', X, y, num_classes, **parameters)
        model.parameters['vote_weighting'] = point['vote_weighting']
        if point['vote_weighting'] == 'similarity':
            model.fit_info.update(weights='similarity_clipped_at_zero',
                                  class_tie_break='weighted_score_desc_then_nearest_support_rank_then_class_id',
                                  all_nonpositive_fallback='uniform_vote')
        return model
    if method == 'linear_svm':
        return fit_classifier('linear_svm', X, y, num_classes, **parameters)
    if method == 'random_forest':
        return fit_forest(X, y, num_classes, parameters)
    raise ValueError(f'Unknown method {method}')


def record_outputs(out: Path, rows: list, labels: list, prediction: dict) -> tuple[dict, dict]:
    acc = MetricAccumulator(dict(enumerate(labels)))
    lookup = {label: i for i, label in enumerate(labels)}
    confusion, signer = Counter(), {}
    with (out / 'predictions.jsonl').open('x') as stream:
        for i, row in enumerate(rows):
            valid = prediction['valid_mask'][i]
            ids = prediction['topk_indices'][i][valid].astype(int).tolist()
            scores = prediction['topk_scores'][i][valid].astype(float).tolist()
            if not ids:
                raise ValueError('No valid predicted class for a sample')
            acc.add(row['gloss'], ids)
            result = {key: row[key] for key in IDENTITY_KEYS}
            result.update(status='ok', top5_ids=ids, top5_scores=scores, score_kind=prediction['score_kind'])
            if 'neighbor_indices' in prediction:
                result['neighbor_train_rows'] = prediction['neighbor_indices'][i].astype(int).tolist()
                result['neighbor_similarities'] = [float(x) for x in prediction['neighbor_similarities'][i]]
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            target = lookup[row['gloss']]
            entry = signer.setdefault(row['signer_id'], {'count': 0, 'top1_correct': 0, 'top5_correct': 0})
            entry['count'] += 1
            entry['top1_correct'] += int(ids[0] == target)
            entry['top5_correct'] += int(target in ids)
            if ids[0] != target:
                confusion[(row['gloss'], labels[ids[0]])] += 1
    metrics, per_class = acc.summary()
    by_signer = {key: {**value, 'top1': value['top1_correct'] / value['count'],
                       'top5': value['top5_correct'] / value['count']} for key, value in sorted(signer.items())}
    write_json(out / 'metrics.json', metrics)
    write_json(out / 'per_class.json', per_class)
    write_json(out / 'signer_metrics.json', by_signer)
    write_json(out / 'error_analysis.json', {
        'top_confusions': [{'true': k[0], 'predicted': k[1], 'count': v} for k, v in confusion.most_common(40)],
        'interpretation': 'Development validation diagnostics, not unseen test performance.'})
    return metrics, by_signer


def method_diagnostics(model: ClassicalClassifier, prediction: dict) -> dict:
    result = {'rows_with_fewer_than_top5': int(prediction['rows_with_fewer_than_top_k'])}
    if model.method == 'knn':
        sims = prediction['neighbor_similarities']
        support = prediction['valid_mask'].sum(axis=1)
        result.update(vote_weighting=model.parameters['vote_weighting'],
                      ranked_class_count_histogram={str(n): int((support == n).sum()) for n in range(1, TOP_K + 1)},
                      rows_with_any_nonpositive_neighbor_similarity=int((sims <= 0).any(axis=1).sum()),
                      similarity_fallback_rows=int(prediction.get('similarity_fallback_mask', np.zeros(0, bool)).sum()),
                      zero_query_rows=int(prediction['zero_query_mask'].sum()),
                      min_neighbor_similarity=float(sims.min()), max_neighbor_similarity=float(sims.max()))
    elif model.method == 'linear_svm':
        iterations = model.fit_info['n_iter_by_estimator']
        result.update(converged=model.fit_info['converged'], n_iter_max=max(iterations),
                      n_iter_median=statistics.median(iterations), n_iter_min=min(iterations),
                      binary_warning_count=len(model.fit_info['binary_warnings']),
                      warning_count=len(model.fit_info['warnings']))
    else:
        keys = ('total_tree_nodes', 'tree_value_bytes', 'tree_storage_upper_bound_bytes', 'tree_depth_min',
                'tree_depth_max', 'tree_leaves_min', 'tree_leaves_max', 'trees_at_leaf_budget', 'trees_at_depth_cap')
        result.update({key: model.fit_info[key] for key in keys})
        result['warning_count'] = len(model.fit_info['warnings'])
    return result


def run_fit(args) -> None:
    import joblib
    point_id = args.point_id
    if point_id not in GRID:
        raise ValueError(f'Unknown grid point {point_id!r}; ad-hoc parameters are not allowed')
    point = GRID[point_id]
    prepared = args.prepared_dir.resolve()
    plan_bytes = (prepared / 'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get('document') != 'classical-tuning-prepared' or plan.get('test_used') is not False:
        raise ValueError('Prepared directory is not a classical-tuning train/validation preparation')
    if plan.get('grid') != GRID:
        raise ValueError('Prepared plan grid differs from this runner grid')
    for name, digest in plan['source_sha256'].items():
        verify_file(WORKSPACE / name, digest, 'tuning source frozen at prepare time')
    for name, digest in plan['prepared_sha256'].items():
        verify_file(prepared / name, digest, 'prepared input')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'point_id': point_id, 'method': point['method'], 'stage': 'loading',
             'started_at_utc': timestamp(), 'pid': os.getpid()}
    write_json(out / 'state.json', state)
    try:
        config = {'document': 'classical-tuning-point', 'point_id': point_id, 'point': point,
                  'prepared_dir': str(prepared), 'prepared_plan_sha256': sha256(prepared / 'plan.json'),
                  'descriptor_variant': plan['descriptor_variant'], 'feature_dataset_sha256': plan['feature_dataset_sha256'],
                  'preregistration': plan['preregistration'], 'device': args.device, 'test_used': False,
                  'environment': {key: os.environ.get(key) for key in
                                  ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES')}}
        write_json(out / 'config.json', config)
        X = np.load(prepared / 'train_scaled.npy', mmap_mode='r')
        y = np.load(prepared / 'train_labels.npy', allow_pickle=False)
        Q = np.load(prepared / 'query_scaled.npy', mmap_mode='r')
        rows = json.loads((prepared / 'query_rows.json').read_text())
        dataset = json.loads((prepared / 'dataset.json').read_text())
        labels = dataset['id_to_label']
        scaler = joblib.load(prepared / 'scaler.joblib')
        if point['method'] == 'random_forest':
            p = point['parameters']
            need = 2.5 * forest_memory_upper_bound(len(labels), p['n_estimators'], p['max_leaf_nodes']) + 8 * 1024**3
            available = mem_available_bytes()
            state['memory_gate'] = {'required_bytes': int(need), 'available_bytes': available}
            if available < need:
                raise MemoryError(f'MemAvailable {available} below preregistered gate {int(need)}')
        state['stage'] = 'fitting'; write_json(out / 'state.json', state)
        stamp = time.perf_counter()
        model = fit_point(point, X, y, len(labels))
        fit_seconds = time.perf_counter() - stamp
        write_json(out / 'fit_info.json', model.fit_info)
        digest = save_model(model, out / 'model.joblib', scaler=scaler)
        state.update(stage='predicting', fit_seconds=fit_seconds); write_json(out / 'state.json', state)
        stamp = time.perf_counter()
        prediction = predict_point(model, Q, args.device)
        predict_seconds = time.perf_counter() - stamp
        state['stage'] = 'reload_replay'; write_json(out / 'state.json', state)
        diagnostics = method_diagnostics(model, prediction)
        del model
        reloaded, scaler2 = load_model(out / 'model.joblib', expected_sha256=digest)
        if not np.array_equal(scaler.mean_, scaler2.mean_) or not np.array_equal(scaler.scale_, scaler2.scale_):
            raise ValueError('Reloaded scaler differs')
        stamp = time.perf_counter()
        repeated = predict_point(reloaded, Q, args.device)
        replay_seconds = time.perf_counter() - stamp
        exact = ['topk_indices', 'valid_mask'] + (['neighbor_indices'] if 'neighbor_indices' in prediction else [])
        for name in exact:
            if not np.array_equal(prediction[name], repeated[name]):
                raise ValueError(f'Reloaded predictions disagree: {name}')
        mask = prediction['valid_mask']
        np.testing.assert_allclose(prediction['topk_scores'][mask], repeated['topk_scores'][mask], rtol=1e-10, atol=1e-10)
        del reloaded, repeated
        timings = {'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds, 'replay_seconds': replay_seconds,
                   'query_count': len(Q), 'predict_ms_per_clip': predict_seconds / len(Q) * 1000,
                   'model_bytes': (out / 'model.joblib').stat().st_size,
                   'scope': 'Cached standardized descriptors; excludes video/MediaPipe and shared descriptor/scaler costs.'}
        metrics, by_signer = record_outputs(out, rows, labels, prediction)
        write_json(out / 'method_details.json', {'point_id': point_id, 'point': point, 'timings': timings,
                                               'diagnostics': diagnostics, 'score_kind': prediction['score_kind'],
                                               'backend': prediction.get('backend'), 'split': rows[0]['split']})
        write_json(out / 'reload_verification.json', {
            'status': 'verified', 'rows': len(Q), 'exact_arrays': exact,
            'score_rtol': 1e-10, 'score_atol': 1e-10, 'model_sha256': digest})
        state.update(status='completed', stage='complete', ended_at_utc=timestamp(), model_sha256=digest,
                     top1=metrics['per_instance']['top1'], macro_top1=metrics['per_class_observed']['top1'])
        write_json(out / 'state.json', state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error), ended_at_utc=timestamp())
        write_json(out / 'state.json', state)
        raise


def run_prepare(args) -> None:
    import joblib
    import sklearn
    preregistration = args.preregistration.resolve()
    if sha256(preregistration) != args.preregistration_sha256:
        raise ValueError('Preregistration SHA256 mismatch')
    feature_dir = args.feature_dir.resolve()
    if sha256(feature_dir / 'dataset.json') != args.expected_dataset_sha256:
        raise ValueError('Feature dataset.json SHA256 differs from the declared descriptor')
    feature_doc = json.loads((feature_dir / 'dataset.json').read_text())
    if (feature_doc['status'] != 'completed' or feature_doc['test_used'] is not False
            or feature_doc['purpose'] != 'formal_train_validation_features'
            or feature_doc['train']['count'] != 40154 or feature_doc['val']['count'] != 10304
            or set(feature_doc) & {'test'}):
        raise ValueError('Feature dataset must be the completed full train/validation data without test')
    for split in ('train', 'val'):
        for row in feature_doc[split]['artifacts'].values():
            verify_file(feature_dir / row['path'], row['sha256'], 'feature dataset artifact')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    X = np.load(feature_dir / 'train_features.npy', allow_pickle=False)
    y = np.load(feature_dir / 'train_labels.npy', allow_pickle=False)
    Q = np.load(feature_dir / 'val_features.npy', allow_pickle=False)
    train_rows = json.loads((feature_dir / 'train_rows.json').read_text())
    rows = json.loads((feature_dir / 'val_rows.json').read_text())
    if {r['split'] for r in train_rows} != {'train'} or {r['split'] for r in rows} != {'val'}:
        raise ValueError('Unexpected split membership; test is forbidden')
    stamp = time.perf_counter()
    scaler = fit_train_scaler(X, split='train')
    np.save(output / 'train_scaled.npy', transform_features(scaler, X))
    np.save(output / 'query_scaled.npy', transform_features(scaler, Q))
    np.save(output / 'train_labels.npy', np.asarray(y, dtype=np.int64))
    joblib.dump(scaler, output / 'scaler.joblib', compress=0)
    write_json(output / 'query_rows.json', rows)
    write_json(output / 'train_rows.json', train_rows)
    write_json(output / 'dataset.json', dict(feature_doc))
    prepared = {name: sha256(output / name) for name in PREPARED_NAMES}
    reference = None
    if args.reference_plan:
        ref = json.loads(args.reference_plan.read_text())['prepared_sha256']
        required = ('train_scaled.npy', 'query_scaled.npy', 'train_labels.npy', 'query_rows.json', 'dataset.json')
        reference = {'reference_plan': str(args.reference_plan.resolve()),
                     'reference_plan_sha256': sha256(args.reference_plan),
                     'byte_identical': {name: prepared[name] == ref[name] for name in ref if name in prepared},
                     'required': list(required)}
        if not all(reference['byte_identical'][name] for name in required):
            write_json(output / 'reference_check.json', {**reference, 'status': 'mismatch'})
            raise ValueError('Prepared matrices differ from the reference preparation')
        write_json(output / 'reference_check.json', {**reference, 'status': 'identical'})
    plan = {'schema_version': 1, 'document': 'classical-tuning-prepared', 'created_at_utc': timestamp(),
            'descriptor_variant': args.descriptor_variant, 'feature_dataset_path': str(feature_dir / 'dataset.json'),
            'feature_dataset_sha256': args.expected_dataset_sha256, 'feature_spec': feature_doc['feature_spec'],
            'scaler_fit_split': 'train', 'scaler_seconds': time.perf_counter() - stamp, 'test_used': False,
            'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
            'preregistration': {'path': str(preregistration.relative_to(WORKSPACE)), 'sha256': args.preregistration_sha256},
            'grid': GRID, 'original_points': ORIGINAL_POINTS, 'prepared_sha256': prepared,
            'reference_check': reference, 'source_sha256': {name: sha256(WORKSPACE / name) for name in SOURCE_NAMES}}
    write_json(output / 'plan.json', plan)
    print(json.dumps({'status': 'prepared', 'output': str(output), 'reference_check': reference and 'identical'}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--feature-dir', type=Path, required=True)
    prep.add_argument('--expected-dataset-sha256', required=True)
    prep.add_argument('--descriptor-variant', required=True)
    prep.add_argument('--preregistration', type=Path, required=True)
    prep.add_argument('--preregistration-sha256', required=True)
    prep.add_argument('--reference-plan', type=Path)
    prep.add_argument('--output-dir', type=Path, required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--prepared-dir', type=Path, required=True)
    fit.add_argument('--point-id', required=True)
    fit.add_argument('--output-dir', type=Path, required=True)
    fit.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run_prepare(args) if args.mode == 'prepare' else run_fit(args)


if __name__ == '__main__':
    main()
