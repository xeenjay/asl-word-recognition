"""Portable interface checks with synthetic fixtures, never recognition claims."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from team.runner import (calculate_metrics, load_dataset, run_experiment, sha256,
                         validate_predictions, write_json)


def make_dataset(directory):
    directory.mkdir()
    rng = np.random.default_rng(7)
    y = np.repeat(np.arange(3, dtype=np.int64), 6)
    X = rng.normal(0, .1, (len(y), 1020)).astype(np.float32)
    X[np.arange(len(y)), y] += 5
    q = X[[0, 6, 12]].copy()
    q[:, 10] = 500  # Must not influence the fitted standardization statistics.
    values = {"train_features.npy": X, "train_labels.npy": y,
              "val_features.npy": q, "val_labels.npy": np.arange(3, dtype=np.int64)}
    files = {}
    for name, array in values.items():
        np.save(directory / name, array)
        files[name] = {"sha256": sha256(directory / name), "bytes": (directory / name).stat().st_size,
                       "shape": list(array.shape), "dtype": str(array.dtype)}
    write_json(directory / "labels.json", ["SYNTH_A", "SYNTH_B", "SYNTH_C"])
    files["labels.json"] = {"sha256": sha256(directory / "labels.json"), "bytes": (directory / "labels.json").stat().st_size}
    manifest = {"schema_version": 1, "profile": "mini", "dataset": "ASL Citizen", "fixture_only": True,
                "feature_spec": "classical-pose-motion-v1", "feature_dim": 1020,
                "source_run_id": "synthetic-interface-test", "source_sha256": {"fixture": "a" * 64},
                "selected_original_class_ids": [0, 1, 2], "test_used": False,
                "subset": {"name": "synthetic-fixture", "selection": "Invented test data, not ASL Citizen samples"},
                "splits": {"train": {"count": len(X)}, "val": {"count": len(q)}}, "files": files}
    write_json(directory / "manifest.json", manifest)
    return manifest, values


def fit_centroids(X_train, y_train, num_classes, config):
    """Independent student-style implementation; no reference model calls."""
    return {"centroids": np.stack([X_train[y_train == i].mean(axis=0) for i in range(num_classes)]),
            "num_classes": num_classes}


def predict_centroids(model, X, top_k):
    distance = ((X[:, None, :].astype(np.float64) - model["centroids"][None]) ** 2).sum(axis=2)
    ids = np.argsort(distance, axis=1, kind="stable")[:, :min(top_k, model["num_classes"])]
    return {"topk_indices": ids, "topk_scores": -np.take_along_axis(distance, ids, axis=1),
            "valid_mask": np.ones(ids.shape, dtype=bool), "score_kind": "negative_squared_centroid_distance"}


class TeamRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.manifest, self.arrays = make_dataset(self.data)

    def test_reference_and_student_delegation_all_methods_and_train_only_scaler(self):
        for method in ("knn", "linear_svm", "random_forest"):
            config = self.root / (method + ".json")
            params = {"n_estimators": 4, "min_samples_leaf": 1} if method == "random_forest" else {}
            write_json(config, {"method": method, "parameters": params})
            results = {}
            for implementation in ("reference", "student"):
                output = self.root / (method + "-" + implementation)
                result = run_experiment(method=method, implementation=implementation,
                                        data_dir=self.data, config_path=config, output_dir=output)
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["model_reload_exact"])
                expected = "reference" if implementation == "reference" else "student_reference_delegate"
                self.assertEqual(result["implementation"]["effective"], expected)
                self.assertTrue(result["implementation"]["reference_api_calls"])
                scaler = joblib.load(output / "scaler.joblib")
                np.testing.assert_allclose(scaler.mean_, self.arrays["train_features.npy"].mean(axis=0, dtype=np.float64))
                self.assertNotAlmostEqual(float(scaler.mean_[10]), 500)
                self.assertEqual(result["metrics"]["samples"], 3)
                results[implementation] = result["metrics"]
            self.assertEqual(results["reference"], results["student"])

    def test_partial_topk_missing_rows_and_macro_complete_denominator(self):
        answer = {"topk_indices": np.array([[0, -1], [0, 1], [-1, -1]]),
                  "topk_scores": np.array([[1, 0], [.6, .4], [0, 0]]),
                  "valid_mask": np.array([[True, False], [True, True], [False, False]])}
        pred = validate_predictions(answer, 3, 3)
        metrics, per_class = calculate_metrics(np.array([0, 1, 1]), pred, ["a", "b", "c"])
        self.assertEqual(metrics["samples"], 3)
        self.assertEqual(metrics["failed"], 1)
        self.assertAlmostEqual(metrics["top1"], 1 / 3)
        self.assertAlmostEqual(metrics["top5"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_top1"], .5)
        self.assertAlmostEqual(metrics["macro_top5"], .75)
        self.assertIsNone(per_class[2]["top1"])
        self.assertEqual(metrics["observed_classes"], 2)

    def test_prediction_contract_rejects_duplicate_invalid_ids_and_fake_padding(self):
        answer = {"topk_indices": np.array([[0, 1]]), "topk_scores": np.array([[.6, .4]]),
                  "valid_mask": np.array([[True, True]])}
        for key, value in (("topk_indices", [[0, 0]]), ("topk_indices", [[0, 3]]),
                           ("topk_scores", [[np.nan, 1]]), ("valid_mask", [[False, True]])):
            bad = copy.deepcopy(answer)
            bad[key] = np.asarray(value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_predictions(bad, 1, 3)

    def test_hash_dtype_count_and_test_data_tampering_are_rejected(self):
        baseline = copy.deepcopy(self.manifest)
        for transform in (lambda m: m.update(test_used=True),
                          lambda m: m["splits"]["val"].update(count=7),
                          lambda m: m["files"]["train_features.npy"].update(sha256="0" * 64),
                          lambda m: m["files"]["train_features.npy"].update(dtype="float64")):
            bad = copy.deepcopy(baseline)
            transform(bad)
            write_json(self.data / "manifest.json", bad)
            with self.assertRaises(ValueError):
                load_dataset(self.data)

    def test_dict_selection_and_preprocessing_provenance_are_preserved(self):
        metadata = {"original_class_id_space": "lexicographically_sorted_glosses",
                    "extractor": "test-only-extractor", "extractor_settings": {"model_complexity": 1},
                    "packages": {"mediapipe": "test-fixture"}, "historical_research_scores_reproduced": False}
        self.manifest.update(metadata)
        selection = {"strategy": "synthetic-per-class", "classes": 3, "train_per_class": 6, "val_per_class": 1}
        self.manifest["subset"]["selection"] = selection
        write_json(self.data / "manifest.json", self.manifest)
        result = run_experiment(data_dir=self.data, output_dir=self.root / "dict-selection")
        self.assertEqual(result["data"]["subset"]["selection"], selection)
        for key, value in metadata.items():
            self.assertEqual(result["data"][key], value)
        for invalid in ("", "   ", {}, []):
            self.manifest["subset"]["selection"] = invalid
            write_json(self.data / "manifest.json", self.manifest)
            with self.subTest(selection=invalid), self.assertRaisesRegex(ValueError, "selection"):
                load_dataset(self.data)

    def test_custom_student_model_fits_predicts_and_reloads_without_reference(self):
        module = importlib.import_module("students.knn")
        with patch.object(module, "fit", fit_centroids), patch.object(module, "predict_topk", predict_centroids), \
                patch.object(module, "IMPLEMENTATION_STATUS", "student_modified"), \
                patch.object(module, "IMPLEMENTATION_NOTE", "Synthetic interface test: independent nearest centroid"):
            result = run_experiment(implementation="student", data_dir=self.data, output_dir=self.root / "custom-student")
        self.assertTrue(result["model_reload_exact"])
        self.assertEqual(result["implementation"]["effective"], "student_modified")
        self.assertEqual(result["implementation"]["reference_api_calls"], [])
        self.assertEqual(result["score_kind"], "negative_squared_centroid_distance")
        saved_model = joblib.load(self.root / "custom-student/model.joblib")
        self.assertEqual(saved_model["centroids"].shape, (3, 1020))
        self.assertEqual(result["metrics"]["rows_with_fewer_than_five_predictions"], 3)

    def test_student_failure_is_saved_and_existing_output_is_never_overwritten(self):
        module = importlib.import_module("students.knn")
        out = self.root / "failure"
        with patch.object(module, "fit", side_effect=ValueError("student fit needs fixing")):
            with self.assertRaisesRegex(ValueError, "needs fixing"):
                run_experiment(implementation="student", data_dir=self.data, output_dir=out)
        old = (out / "run.json").read_bytes()
        self.assertEqual(json.loads(old)["status"], "failed")
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_experiment(data_dir=self.data, output_dir=out)
        self.assertEqual(old, (out / "run.json").read_bytes())

    def test_cli_failure_is_friendly_and_not_a_success(self):
        command = [sys.executable, "-B", str(ROOT / "run.py"), "--data", str(self.root / "missing"),
                   "--output", str(self.root / "missing-output")]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("prepare_team_data.py", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
