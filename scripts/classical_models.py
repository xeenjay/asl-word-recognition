"""Bounded classical ASL classifiers, independent of PoET learned embeddings.

The caller owns split membership/identity audits. Fit the scaler on TRAIN ONLY,
then transform train and validation using that same object. Class IDs must cover
exactly 0..num_classes-1. kNN vote fractions and SVM margins are NOT calibrated
probabilities. Positive-support-only ranking prevents arbitrary zero-vote or
zero-probability classes from inflating Top-5; unused slots have ID -1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


DEFAULT_PARAMETERS = {
    "knn": {"n_neighbors": 5},
    "linear_svm": {"C": 0.01, "dual": False, "tol": 1e-3, "max_iter": 2000,
                   "random_state": 20261002, "parallel_ovr": False, "n_jobs": 1},
    "random_forest": {"n_estimators": 64, "max_depth": 12, "max_leaf_nodes": 256,
                      "min_samples_leaf": 2, "max_features": "sqrt", "n_jobs": 4,
                      "random_state": 20261002},
}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _features(value: Any, *, dimension: int | None = None, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] == 0 or (not allow_empty and array.shape[0] == 0):
        raise ValueError("Features must be a nonempty two-dimensional matrix")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("Features must be finite real numbers")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError("Feature dimension mismatch")
    with np.errstate(over="ignore"):
        result = np.asarray(array, dtype=np.float32, order="C")
    if not np.isfinite(result).all():
        raise ValueError("Features are not representable as finite float32")
    return result


def _labels(value: Any, samples: int, num_classes: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or len(result) != samples or result.dtype.kind not in "iu":
        raise ValueError("Labels must be a one-dimensional integer array matching train rows")
    if not np.array_equal(np.unique(result), np.arange(num_classes)):
        raise ValueError("Training class mapping must cover exactly 0..num_classes-1")
    return result.astype(np.int64, copy=True)


def fit_train_scaler(X_train: np.ndarray, *, split: str = "train") -> StandardScaler:
    """Fit one StandardScaler. Split identity itself is audited by the caller."""
    if split != "train":
        raise ValueError("Scaler fitting is allowed on train only")
    scaler = StandardScaler(copy=True).fit(_features(X_train))
    scaler.classical_fit_split_ = "train"
    scaler.classical_fit_rows_ = int(len(X_train))
    return scaler


def transform_features(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    if not isinstance(scaler, StandardScaler) or getattr(scaler, "classical_fit_split_", None) != "train":
        raise ValueError("Expected a scaler fitted by fit_train_scaler")
    values = _features(X, dimension=scaler.n_features_in_, allow_empty=True)
    if len(values) == 0:
        return values.copy()
    return _features(scaler.transform(values), dimension=scaler.n_features_in_)


def _l2_rows(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Float64 accumulation prevents overflow even for valid, large float32 input.
    norms = np.linalg.norm(X.astype(np.float64), axis=1)
    zeros = norms == 0
    result = np.zeros_like(X, dtype=np.float32)
    np.divide(X, norms[:, None], out=result, where=~zeros[:, None])
    return result, zeros


@dataclass
class ClassicalClassifier:
    method: str
    num_classes: int
    feature_dim: int
    parameters: dict[str, Any]
    estimator: Any = None
    reference_features: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    fit_info: dict[str, Any] = field(default_factory=dict)


class RecordingLinearSVC(LinearSVC):
    """Keep convergence evidence inside each joblib worker's fitted object."""

    def fit(self, X, y, sample_weight=None):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            super().fit(X, y, sample_weight=sample_weight)
        self.classical_warnings_ = [{"category": item.category.__name__, "message": str(item.message)}
                                    for item in caught]
        self.classical_converged_ = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        for item in caught:
            warnings.warn(str(item.message), item.category, stacklevel=2)
        return self


def forest_memory_upper_bound(num_classes: int, n_estimators: int, max_leaf_nodes: int) -> int:
    """Conservative tree node/value bytes, excluding training work buffers.

    Each node stores a dense float64 distribution over ALL classes. This is why
    bounding only max_depth is insufficient for a 2731-class forest.
    """
    c = _positive_int(num_classes, "num_classes")
    trees = _positive_int(n_estimators, "n_estimators")
    leaves = _positive_int(max_leaf_nodes, "max_leaf_nodes")
    return trees * (2 * leaves - 1) * (c * 8 + 128)


def fit_classifier(method: str, X_train: np.ndarray, y_train: np.ndarray,
                   num_classes: int, **parameters: Any) -> ClassicalClassifier:
    """Fit a single candidate on already TRAIN-standardized features.

    No parameter sweep or validation labels are accepted here. Convergence
    warnings are recorded and re-emitted, never silently converted to success.
    """
    if method not in DEFAULT_PARAMETERS:
        raise ValueError(f"Unknown method: {method}")
    c = _positive_int(num_classes, "num_classes")
    if c < 2:
        raise ValueError("At least two classes are required")
    X = _features(X_train)
    y = _labels(y_train, len(X), c)
    unexpected = set(parameters) - set(DEFAULT_PARAMETERS[method])
    if unexpected:
        raise ValueError(f"Unsupported parameters: {sorted(unexpected)}")
    config = {**DEFAULT_PARAMETERS[method], **parameters}
    model = ClassicalClassifier(method, c, X.shape[1], config)
    info = {"train_rows": len(X), "feature_dim": X.shape[1], "num_classes": c,
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
            "warnings": [], "converged": None}
    if method == "knn":
        neighbors = _positive_int(config["n_neighbors"], "n_neighbors")
        if neighbors > len(X):
            raise ValueError("n_neighbors exceeds train rows")
        model.reference_features, zero_rows = _l2_rows(X)
        model.reference_labels = y
        info.update(zero_reference_rows=int(zero_rows.sum()), weights="uniform",
                    neighbor_tie_break="train_row_ascending",
                    class_tie_break="vote_count_desc_then_nearest_support_rank_then_class_id")
    else:
        if method == "linear_svm":
            if config["dual"] is not False:
                raise ValueError("This bounded protocol requires LinearSVC dual=False")
            jobs = _positive_int(config["n_jobs"], "n_jobs")
            if jobs > 8 or not isinstance(config["parallel_ovr"], bool):
                raise ValueError("SVM requires n_jobs<=8 and boolean parallel_ovr")
            svc_config = {key: value for key, value in config.items() if key not in ("n_jobs", "parallel_ovr")}
            if config["parallel_ovr"]:
                estimator = OneVsRestClassifier(RecordingLinearSVC(**svc_config), n_jobs=jobs)
            else:
                if jobs != 1:
                    raise ValueError("Native LinearSVC requires n_jobs=1; enable parallel_ovr explicitly")
                estimator = RecordingLinearSVC(**svc_config)
        else:
            trees = _positive_int(config["n_estimators"], "n_estimators")
            leaves = _positive_int(config["max_leaf_nodes"], "max_leaf_nodes")
            jobs = _positive_int(config["n_jobs"], "n_jobs")
            _positive_int(config["max_depth"], "max_depth")
            if leaves < 2 or jobs > 8:
                raise ValueError("Forest requires max_leaf_nodes>=2 and n_jobs<=8")
            estimate = forest_memory_upper_bound(c, trees, leaves)
            if estimate > 2 * 1024**3:
                raise ValueError("Forest tree storage estimate exceeds bounded 2 GiB budget")
            info["tree_storage_upper_bound_bytes"] = estimate
            estimator = RandomForestClassifier(**config)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X, y)
        info["warnings"] = [{"category": warning.category.__name__, "message": str(warning.message)}
                            for warning in caught]
        for warning in caught:
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
        if not np.array_equal(estimator.classes_, np.arange(c)):
            raise ValueError("Estimator class mapping differs from declared mapping")
        if method == "linear_svm":
            estimators = estimator.estimators_ if config["parallel_ovr"] else [estimator]
            info["n_iter_by_estimator"] = [int(np.max(item.n_iter_)) for item in estimators]
            info["n_iter"] = max(info["n_iter_by_estimator"])
            info["converged"] = all(item.classical_converged_ for item in estimators)
            info["binary_warnings"] = [{"estimator_index": index, "warnings": item.classical_warnings_}
                                       for index, item in enumerate(estimators) if item.classical_warnings_]
            info["class_ids"] = estimator.classes_.astype(int).tolist()
            info["ovr_implementation"] = "sklearn_OneVsRestClassifier" if config["parallel_ovr"] else "native_LinearSVC"
        else:
            info["total_tree_nodes"] = sum(int(tree.tree_.node_count) for tree in estimator.estimators_)
            info["tree_value_bytes"] = sum(int(tree.tree_.value.nbytes) for tree in estimator.estimators_)
        model.estimator = estimator
    model.fit_info = info
    return model


def _validate_model(model: ClassicalClassifier) -> None:
    if not isinstance(model, ClassicalClassifier) or model.method not in DEFAULT_PARAMETERS:
        raise ValueError("Invalid classical model")
    _positive_int(model.num_classes, "num_classes")
    _positive_int(model.feature_dim, "feature_dim")
    if model.method == "knn":
        X = _features(model.reference_features, dimension=model.feature_dim)
        _labels(model.reference_labels, len(X), model.num_classes)
        norms = np.linalg.norm(X.astype(np.float64), axis=1)
        if not np.all((norms == 0) | np.isclose(norms, 1, atol=2e-6, rtol=0)):
            raise ValueError("kNN references must already be L2 normalized")
    elif model.estimator is None or not np.array_equal(model.estimator.classes_, np.arange(model.num_classes)):
        raise ValueError("Estimator class mapping mismatch")


def rank_scores(scores: np.ndarray, top_k: int = 5, *, positive_only: bool = False) -> dict[str, np.ndarray]:
    """Stable class-ID tie break. Do not use for kNN's neighbor-rank tie break."""
    values = np.asarray(scores)
    k = _positive_int(top_k, "top_k")
    if values.ndim != 2 or not values.shape[1] or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite two-dimensional matrix")
    values = values.astype(np.float64, copy=False)
    count = min(k, values.shape[1])
    order = np.argsort(-values, axis=1, kind="stable")[:, :count]
    picked = np.take_along_axis(values, order, axis=1)
    supported = picked > 0 if positive_only else np.ones(picked.shape, dtype=bool)
    indices = np.full((len(values), k), -1, dtype=np.int64)
    output_scores = np.zeros((len(values), k), dtype=np.float64)
    valid = np.zeros((len(values), k), dtype=bool)
    indices[:, :count] = np.where(supported, order, -1)
    output_scores[:, :count] = np.where(supported, picked, 0)
    valid[:, :count] = supported
    return {"topk_indices": indices, "topk_scores": output_scores, "valid_mask": valid}


def _knn_neighbors(model: ClassicalClassifier, query: np.ndarray, *, device: str,
                   batch_size: int, reference_chunk_size: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Exhaustive cosine search in float32; ties follow original train row."""
    k = model.parameters["n_neighbors"]
    reference = model.reference_features
    indices = np.empty((len(query), k), dtype=np.int64)
    similarities = np.empty((len(query), k), dtype=np.float32)
    if device == "cpu":
        for q0 in range(0, len(query), batch_size):
            q = query[q0:q0 + batch_size]
            best_values = np.empty((len(q), 0), np.float32)
            best_indices = np.empty((len(q), 0), np.int64)
            for r0 in range(0, len(reference), reference_chunk_size):
                sims = q @ reference[r0:r0 + reference_chunk_size].T
                ids = np.broadcast_to(np.arange(r0, r0 + sims.shape[1]), sims.shape)
                all_values = np.concatenate((best_values, sims), axis=1)
                all_indices = np.concatenate((best_indices, ids), axis=1)
                # Previous tied winners are ID-ordered and precede later chunks.
                take = np.argsort(-all_values, axis=1, kind="stable")[:, :k]
                best_values = np.take_along_axis(all_values, take, axis=1)
                best_indices = np.take_along_axis(all_indices, take, axis=1)
            indices[q0:q0 + len(q)] = best_indices
            similarities[q0:q0 + len(q)] = best_values
        return indices, similarities, "numpy_float32_exhaustive"
    if not device.startswith("cuda"):
        raise ValueError("device must be cpu or an explicit cuda device")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select cpu explicitly")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode():
            # This copy is bounded by N_train*D; only pairwise similarities chunk.
            refs = torch.from_numpy(reference).to(device)
            for q0 in range(0, len(query), batch_size):
                q = torch.from_numpy(query[q0:q0 + batch_size]).to(device)
                best_values = torch.empty((len(q), 0), dtype=torch.float32, device=device)
                best_indices = torch.empty((len(q), 0), dtype=torch.int64, device=device)
                for r0 in range(0, len(reference), reference_chunk_size):
                    sims = q @ refs[r0:r0 + reference_chunk_size].T
                    ids = torch.arange(r0, r0 + sims.shape[1], device=device).expand(len(q), -1)
                    all_values = torch.cat((best_values, sims), dim=1)
                    all_indices = torch.cat((best_indices, ids), dim=1)
                    take = torch.argsort(all_values, dim=1, descending=True, stable=True)[:, :k]
                    best_values = torch.gather(all_values, 1, take)
                    best_indices = torch.gather(all_indices, 1, take)
                indices[q0:q0 + len(q)] = best_indices.cpu().numpy()
                similarities[q0:q0 + len(q)] = best_values.cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return indices, similarities, "torch_cuda_float32_exhaustive_tf32_disabled"


def predict_topk(model: ClassicalClassifier, X_query: np.ndarray, top_k: int = 5, *,
                 device: str = "cpu", batch_size: int = 64,
                 reference_chunk_size: int = 8192) -> dict[str, Any]:
    """Predict from already-standardized query features without using labels.

    Different CPU/GPU BLAS implementations can round nearly tied cosine scores
    differently. Use the same device/chunk configuration for exact reload checks.
    """
    _validate_model(model)
    X = _features(X_query, dimension=model.feature_dim, allow_empty=True)
    k = _positive_int(top_k, "top_k")
    batch = _positive_int(batch_size, "batch_size")
    chunk = _positive_int(reference_chunk_size, "reference_chunk_size")
    if model.method == "knn":
        query, zeros = _l2_rows(X)
        neighbors, similarities, backend = _knn_neighbors(model, query, device=device,
                                                          batch_size=batch, reference_chunk_size=chunk)
        ids = np.full((len(X), k), -1, np.int64)
        scores = np.zeros((len(X), k), np.float64)
        valid = np.zeros((len(X), k), bool)
        support_count = np.zeros(len(X), np.int64)
        for row, neighbor_ids in enumerate(neighbors):
            labels = model.reference_labels[neighbor_ids]
            classes, first_ranks, counts = np.unique(labels, return_index=True, return_counts=True)
            order = np.lexsort((classes, first_ranks, -counts))
            n = min(k, len(classes))
            chosen = order[:n]
            ids[row, :n] = classes[chosen]
            scores[row, :n] = counts[chosen] / len(labels)
            valid[row, :n] = True
            support_count[row] = len(classes)
        result = {"topk_indices": ids, "topk_scores": scores, "valid_mask": valid,
                  "neighbor_indices": neighbors, "neighbor_similarities": similarities,
                  "support_count": support_count, "zero_query_mask": zeros,
                  "score_kind": "uniform_neighbor_vote_fraction", "backend": backend}
    else:
        parts = []
        for start in range(0, len(X), batch):
            rows = X[start:start + batch]
            if model.method == "linear_svm":
                values = model.estimator.decision_function(rows)
                if values.ndim == 1:  # sklearn binary margin is for classes_[1].
                    values = np.column_stack((-values, values))
            else:
                values = model.estimator.predict_proba(rows)
            parts.append(rank_scores(values, k, positive_only=model.method == "random_forest"))
        result = {name: np.concatenate([part[name] for part in parts], axis=0)
                  if parts else np.empty((0, k), dtype=dtype)
                  for name, dtype in (("topk_indices", np.int64), ("topk_scores", np.float64),
                                      ("valid_mask", bool))}
        result.update(score_kind="ovr_decision_margin" if model.method == "linear_svm" else "forest_probability",
                      backend="sklearn_cpu")
    result["rows_with_fewer_than_top_k"] = int((result["valid_mask"].sum(axis=1) < k).sum())
    return result


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_model(model: ClassicalClassifier, path: str | Path, *, scaler: StandardScaler | None = None) -> str:
    """Save a NEW trusted-local joblib artifact; refuse overwriting old runs."""
    _validate_model(model)
    if scaler is not None:
        transform_features(scaler, np.empty((0, model.feature_dim), np.float32))
    payload = {"schema_version": 1, "sklearn_version": sklearn.__version__, "model": model, "scaler": scaler}
    with Path(path).open("xb") as stream:
        joblib.dump(payload, stream, compress=0, protocol=4)
    return _sha256_file(path)


def load_model(path: str | Path, *, expected_sha256: str) -> tuple[ClassicalClassifier, StandardScaler | None]:
    """Load only a trusted artifact with independently recorded SHA256.

    A hash verifies bytes, not safety of untrusted pickle/joblib contents.
    """
    if len(expected_sha256) != 64 or _sha256_file(path) != expected_sha256:
        raise ValueError("Model SHA256 mismatch")
    with Path(path).open("rb") as stream:
        payload = joblib.load(stream)
    if payload.get("schema_version") != 1 or payload.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Model schema/sklearn version mismatch")
    model, scaler = payload["model"], payload["scaler"]
    _validate_model(model)
    if scaler is not None:
        transform_features(scaler, np.empty((0, model.feature_dim), np.float32))
    return model, scaler
