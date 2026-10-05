"""Audited portable dataset loading and one-method train/validation runs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from team import reference
from classical_models import fit_train_scaler, transform_features

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("knn", "linear_svm", "random_forest")
ARRAYS = ("train_features.npy", "train_labels.npy", "val_features.npy", "val_labels.npy")
FILES = (*ARRAYS, "labels.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_dataset(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Prepared data are missing. First run 'python scripts/prepare_team_data.py "
                         "--accept-dataset-license', or pass --data to a prepared dataset directory.")
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema_version") == 1, "Unsupported dataset manifest schema")
    require(manifest.get("dataset") == "ASL Citizen", "Expected ASL Citizen dataset metadata")
    require(manifest.get("profile") in ("mini", "full"), "Dataset profile must be mini or full")
    require(manifest.get("feature_spec") == "classical-pose-motion-v1" and manifest.get("feature_dim") == 1020,
            "This runner expects the classical-pose-motion-v1 descriptor with 1020 columns")
    require(manifest.get("test_used") is False, "This runner accepts train/validation data only, not test")
    require(isinstance(manifest.get("source_run_id"), str) and manifest["source_run_id"], "Missing data source identity")
    source_hashes = manifest.get("source_sha256")
    require(isinstance(source_hashes, dict) and source_hashes and
            all(isinstance(k, str) and k and valid_sha(v) for k, v in source_hashes.items()),
            "Source provenance requires a nonempty name-to-SHA256 mapping")
    subset = manifest.get("subset")
    require(isinstance(subset, dict) and isinstance(subset.get("name"), str) and subset["name"].strip(),
            "Missing subset name metadata")
    selection = subset.get("selection")
    require((isinstance(selection, str) and bool(selection.strip())) or
            (isinstance(selection, dict) and bool(selection)),
            "Subset selection must be a nonempty string or a nonempty object")
    require(set(manifest.get("splits", {})) == {"train", "val"}, "Manifest must declare exactly train and val splits")
    records = manifest.get("files", {})
    require(isinstance(records, dict) and set(FILES).issubset(records), "Manifest does not describe all five data files")
    hashes, arrays = {}, {}
    for name in FILES:
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"Missing or linked data file: {name}")
        record = records[name]
        require(isinstance(record, dict) and valid_sha(record.get("sha256")), f"Missing SHA256 for {name}")
        require(type(record.get("bytes")) is int and path.stat().st_size == record["bytes"], f"Byte count mismatch: {name}")
        hashes[name] = sha256(path)
        require(hashes[name] == record["sha256"], f"SHA256 mismatch: {name}; prepare the data again")
        if name in ARRAYS:
            array = np.load(path, allow_pickle=False)
            require(isinstance(array, np.ndarray), f"Expected an NPY array: {name}")
            require(record.get("shape") == list(array.shape) and record.get("dtype") == str(array.dtype),
                    f"Array shape/dtype differs from manifest: {name}")
            arrays[name] = array
    labels = json.loads((directory / "labels.json").read_text())
    require(isinstance(labels, list) and len(labels) >= 2 and all(isinstance(x, str) and x.strip() == x and x for x in labels)
            and len(set(labels)) == len(labels), "labels.json must be a unique nonempty gloss list for at least two classes")
    original = manifest.get("selected_original_class_ids")
    require(isinstance(original, list) and len(original) == len(labels) and
            all(type(i) is int and 0 <= i < 2731 for i in original) and len(set(original)) == len(original),
            "Original class IDs must align with the local vocabulary")
    for split in ("train", "val"):
        X, y = arrays[split + "_features.npy"], arrays[split + "_labels.npy"]
        require(X.dtype == np.float32 and X.ndim == 2 and X.shape[1] == 1020 and len(X) > 0,
                f"{split} features must be nonempty float32 [N,1020]")
        require(np.isfinite(X).all(), f"{split} features contain NaN or infinity")
        require(y.dtype == np.int64 and y.shape == (len(X),), f"{split} labels must be int64 [N]")
        require(np.all((y >= 0) & (y < len(labels))), f"{split} labels are outside the local vocabulary")
        require(type(manifest["splits"][split].get("count")) is int and manifest["splits"][split]["count"] == len(X),
                f"{split} count differs from the manifest")
    require(np.array_equal(np.unique(arrays["train_labels.npy"]), np.arange(len(labels))), "Every local class needs training examples")
    if manifest["profile"] == "full":
        require(len(labels) == 2731 and len(arrays["train_labels.npy"]) == 40154 and len(arrays["val_labels.npy"]) == 10304,
                "Full profile requires all 2731 classes, 40154 train and 10304 validation clips")
    require(sha256(manifest_path) == manifest_hash and all(sha256(directory / n) == d for n, d in hashes.items()),
            "Data files changed while loading")
    return {"arrays": arrays, "labels": labels, "manifest": manifest, "manifest_sha256": manifest_hash, "file_sha256": hashes}


def load_config(method, path=None):
    default_path = ROOT / "configs" / (method + ".json")
    default = json.loads(default_path.read_text())
    path = Path(path) if path is not None else default_path
    given = json.loads(path.read_text())
    require(isinstance(given, dict) and set(given) == {"method", "parameters"} and given["method"] == method,
            "Config must contain the selected method and a parameters object")
    require(isinstance(given["parameters"], dict), "Config parameters must be an object")
    config = {**default["parameters"], **given["parameters"]}
    require("n_jobs" not in config or (type(config["n_jobs"]) is int and config["n_jobs"] == 1),
            "This portable CPU runner uses n_jobs=1; keep the configuration at one worker")
    json.dumps(config, allow_nan=False)
    return config, sha256(path)


def validate_predictions(answer, rows, num_classes, top_k=5):
    require(isinstance(answer, dict), "predict_topk must return a dict with topk_indices, topk_scores, valid_mask")
    require(all(k in answer for k in ("topk_indices", "topk_scores", "valid_mask")), "Prediction dict is missing a required array")
    ids, scores, valid = (np.asarray(answer[k]) for k in ("topk_indices", "topk_scores", "valid_mask"))
    require(ids.ndim == 2 and ids.shape[0] == rows and 0 < ids.shape[1] <= top_k and ids.dtype.kind in "iu",
            "topk_indices must be an integer [N,K] array with 1 <= K <= top_k")
    require(scores.shape == ids.shape and scores.dtype.kind in "fiu" and np.isfinite(scores).all(),
            "topk_scores must be finite real values with the same shape as IDs")
    require(valid.shape == ids.shape and valid.dtype == np.bool_, "valid_mask must be a boolean array with the same shape as IDs")
    require(np.all((ids[valid] >= 0) & (ids[valid] < num_classes)), "Valid predicted class IDs must belong to the local vocabulary")
    require(np.all(ids[~valid] == -1), "Unfilled prediction ranks must use ID -1 and valid_mask=False")
    require(not np.any(valid[:, 1:] & ~valid[:, :-1]), "Valid predictions must precede unfilled ranks")
    for row, mask in zip(ids, valid):
        require(len(set(row[mask].tolist())) == int(mask.sum()), "Duplicate class ID in a Top-k prediction")
    # Standardize output width, preserving fewer than five supported candidates.
    output_ids = np.full((rows, top_k), -1, np.int64)
    output_scores = np.zeros((rows, top_k), np.float64)
    output_valid = np.zeros((rows, top_k), bool)
    output_ids[:, :ids.shape[1]], output_scores[:, :ids.shape[1]], output_valid[:, :ids.shape[1]] = ids, scores, valid
    return {"topk_indices": output_ids, "topk_scores": output_scores, "valid_mask": output_valid,
            "score_kind": str(answer.get("score_kind", "student_defined_uncalibrated_score"))}


def calculate_metrics(y, predictions, labels):
    ids, valid = predictions["topk_indices"], predictions["valid_mask"]
    top1 = valid[:, 0] & (ids[:, 0] == y)
    top5 = ((ids == y[:, None]) & valid).any(axis=1)
    per_class = []
    for class_id, label in enumerate(labels):
        selected = y == class_id
        count = int(selected.sum())
        per_class.append({"class_id": class_id, "gloss": label, "samples": count,
                          "correct_top1": int(top1[selected].sum()), "correct_top5": int(top5[selected].sum()),
                          "top1": float(top1[selected].mean()) if count else None,
                          "top5": float(top5[selected].mean()) if count else None})
    observed = [row for row in per_class if row["samples"]]
    return {"split": "val", "samples": len(y), "failed": int((~valid[:, 0]).sum()),
            "correct_top1": int(top1.sum()), "correct_top5": int(top5.sum()),
            "top1": float(top1.mean()), "top5": float(top5.mean()),
            "macro_top1": float(np.mean([x["top1"] for x in observed])),
            "macro_top5": float(np.mean([x["top5"] for x in observed])),
            "vocabulary_classes": len(labels), "observed_classes": len(observed),
            "macro_denominator": "classes represented in validation; absent classes reported as null",
            "rows_with_fewer_than_five_predictions": int((valid.sum(axis=1) < 5).sum())}, per_class


def run_experiment(*, method="knn", implementation="reference", data_dir="data/classical-mini", config_path=None, output_dir):
    require(method in METHODS and implementation in ("reference", "student"), "Unknown method or implementation")
    output = Path(output_dir)
    require(not output.exists(), f"Output already exists: {output}. Choose a new directory; old results are not overwritten.")
    output.mkdir(parents=True)
    summary = {"schema_version": 1, "status": "running", "method": method,
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "test_used": False,
               "implementation": {"requested": implementation}, "cpu_thread_limit": 1}
    write_json(output / "run.json", summary)
    started = time.perf_counter()
    try:
        config, config_hash = load_config(method, config_path)
        data = load_dataset(data_dir)
        arrays, labels, manifest = data["arrays"], data["labels"], data["manifest"]
        files = [ROOT / "run.py", *sorted((ROOT / "team").glob("*.py")), ROOT / "scripts/classical_models.py"]
        if implementation == "student":
            module = importlib.import_module("students." + method)
            require(callable(getattr(module, "fit", None)) and callable(getattr(module, "predict_topk", None)), "Student module must define fit and predict_topk")
            declared = getattr(module, "IMPLEMENTATION_STATUS", "unspecified")
            note = getattr(module, "IMPLEMENTATION_NOTE", "No implementation note provided")
            require(declared in ("reference_delegate", "student_modified"), "Set student IMPLEMENTATION_STATUS to reference_delegate or student_modified")
            files += [Path(module.__file__), ROOT / "students/__init__.py"]
            fit = module.fit
            predict = module.predict_topk
        else:
            declared, note = "reference", "Shared reference classifier"
            fit = lambda X, y, n, cfg: reference.fit(method, X, y, n, cfg)
            predict = reference.predict_topk
        source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
        summary.update(config=config, config_sha256=config_hash, source_sha256=source_hashes,
                       source_hash_scope="run.py, team modules, shared classical_models.py, and the selected student module/package; additional student helper imports are not automatically enumerated",
                       data={"profile": manifest["profile"], "dataset": manifest["dataset"], "subset": manifest["subset"],
                             "train_samples": len(arrays["train_labels.npy"]), "val_samples": len(arrays["val_labels.npy"]),
                             "classes": len(labels), "manifest_sha256": data["manifest_sha256"], "file_sha256": data["file_sha256"],
                             "source_run_id": manifest["source_run_id"], "source_sha256": manifest["source_sha256"],
                             "selected_original_class_ids": manifest["selected_original_class_ids"],
                             **{key: manifest.get(key) for key in
                                ("original_class_id_space", "extractor", "extractor_settings", "packages",
                                 "historical_research_scores_reproduced")},
                             "split_check": "package train/val identity, bytes, dtype, shape and hashes; signer metadata are not bundled"},
                       environment={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")})
        summary["environment"]["python"] = platform.python_version()
        summary["implementation"].update(declared_status=declared, note=note)
        write_json(output / "config.json", {"method": method, "parameters": config})
        write_json(output / "labels.json", labels)
        write_json(output / "run.json", summary)
        with threadpool_limits(limits=1), reference.trace_calls() as calls:
            scaler = fit_train_scaler(arrays["train_features.npy"], split="train")
            X = transform_features(scaler, arrays["train_features.npy"])
            Q = transform_features(scaler, arrays["val_features.npy"])
            stamp = time.perf_counter()
            model = fit(X, arrays["train_labels.npy"].copy(), len(labels), dict(config))
            fit_seconds = time.perf_counter() - stamp
            stamp = time.perf_counter()
            predictions = validate_predictions(predict(model, Q, 5), len(Q), len(labels))
            predict_seconds = time.perf_counter() - stamp
            joblib.dump(model, output / "model.joblib")
            joblib.dump(scaler, output / "scaler.joblib")
            restored = joblib.load(output / "model.joblib")
            restored_scaler = joblib.load(output / "scaler.joblib")
            replay = validate_predictions(predict(restored, transform_features(restored_scaler, arrays["val_features.npy"]), 5), len(Q), len(labels))
            require(np.array_equal(predictions["topk_indices"], replay["topk_indices"]) and
                    np.array_equal(predictions["valid_mask"], replay["valid_mask"]) and
                    np.array_equal(predictions["topk_scores"], replay["topk_scores"]),
                    "Saved model/scaler did not reproduce predictions exactly; fix deterministic inference or serialization")
        effective = "reference" if implementation == "reference" else (
            "student_reference_delegate" if declared == "reference_delegate" else
            "student_modified_with_reference_calls" if calls else "student_modified")
        summary["implementation"].update(effective=effective, reference_api_calls=calls,
                                          attribution_note="Status/note describe code, not a verified human contribution; source hashes and delegation calls are recorded.")
        metrics, per_class = calculate_metrics(arrays["val_labels.npy"], predictions, labels)
        np.savez_compressed(output / "predictions.npz", labels=arrays["val_labels.npy"],
                            **{k: predictions[k] for k in ("topk_indices", "topk_scores", "valid_mask")})
        write_json(output / "metrics.json", metrics)
        write_json(output / "per_class.json", per_class)
        require(all(sha256(ROOT / name) == digest for name, digest in source_hashes.items()), "Source code changed during the run")
        summary.update(status="completed", metrics=metrics, score_kind=predictions["score_kind"],
                       scaler_fit_split="train", model_reload_exact=True,
                       timings_seconds={"fit": fit_seconds, "predict": predict_seconds, "total": time.perf_counter() - started},
                       ended_at_utc=datetime.now(timezone.utc).isoformat(),
                       interpretation="Validation development run; mini results use a restricted vocabulary, not the full benchmark.")
        summary["artifacts"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in output.iterdir() if p.is_file() and p.name != "run.json"}
        write_json(output / "run.json", summary)
        return summary
    except Exception as error:
        summary.update(status="failed", error={"type": type(error).__name__, "message": str(error)},
                       ended_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
        write_json(output / "run.json", summary)
        raise
