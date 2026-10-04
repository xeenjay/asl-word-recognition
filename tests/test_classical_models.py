"""Synthetic checks for split leakage, class alignment, ranking and storage."""
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import classical_models as cm


class ClassicalModelsTests(unittest.TestCase):
    def fixture(self):
        X = np.array([[-3, 0], [-2, 0], [3, 0], [2, 0], [0, 3], [0, 2]], np.float32)
        y = np.array([0, 0, 1, 1, 2, 2])
        return X, y

    def test_scaler_uses_training_statistics_and_does_not_mutate(self):
        X = np.array([[0, 1], [2, 1]], np.float32)
        original = X.copy()
        scaler = cm.fit_train_scaler(X)
        cm.transform_features(scaler, np.array([[1000, 1000]], np.float32))
        np.testing.assert_array_equal(scaler.mean_, [1, 1])
        np.testing.assert_array_equal(X, original)
        np.testing.assert_array_equal(cm.transform_features(scaler, X), [[-1, 0], [1, 0]])
        with self.assertRaisesRegex(ValueError, "train only"):
            cm.fit_train_scaler(X, split="val")

    def test_finite_features_and_mapping_required(self):
        X, y = self.fixture()
        for values in (X * np.nan, np.full_like(X, np.inf), np.ones((0, 2)), np.array([[1 + 1j]])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                cm.fit_train_scaler(values)
        for labels in (y.astype(float), y + 1, y[:-1], np.zeros_like(y)):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                cm.fit_classifier("knn", X, labels, 3)
        with self.assertRaises(ValueError):
            cm.fit_classifier("knn", X, y, 3, n_neighbors=7)
        with self.assertRaises(ValueError):
            cm.fit_classifier("knn", X, y, 3, weights="distance")

    def test_cosine_exact_search_matches_bruteforce_and_chunks(self):
        rng = np.random.default_rng(81)
        X = rng.normal(size=(27, 7)).astype(np.float32)
        y = np.arange(27) % 3
        q = rng.normal(size=(9, 7)).astype(np.float32)
        model = cm.fit_classifier("knn", X, y, 3)
        pred = cm.predict_topk(model, q, batch_size=3, reference_chunk_size=4)
        xnorm = X / np.linalg.norm(X.astype(np.float64), axis=1)[:, None]
        qnorm = q / np.linalg.norm(q.astype(np.float64), axis=1)[:, None]
        expected = np.argsort(-np.einsum("qd,rd->qr", qnorm, xnorm), axis=1, kind="stable")[:, :5]
        np.testing.assert_array_equal(pred["neighbor_indices"], expected)
        other = cm.predict_topk(model, q, batch_size=9, reference_chunk_size=27)
        np.testing.assert_array_equal(pred["topk_indices"], other["topk_indices"])
        np.testing.assert_array_equal(pred["neighbor_indices"], other["neighbor_indices"])

    def test_neighbor_ties_use_train_row_and_class_ties_nearest_support(self):
        X = np.array([[1, 0]] * 6, np.float32)
        y = np.array([2, 1, 2, 1, 0, 0])
        model = cm.fit_classifier("knn", X, y, 3, n_neighbors=4)
        pred = cm.predict_topk(model, X[:1], reference_chunk_size=2)
        np.testing.assert_array_equal(pred["neighbor_indices"], [[0, 1, 2, 3]])
        np.testing.assert_array_equal(pred["topk_indices"], [[2, 1, -1, -1, -1]])
        np.testing.assert_array_equal(pred["topk_scores"], [[.5, .5, 0, 0, 0]])
        self.assertEqual(pred["rows_with_fewer_than_top_k"], 1)

    def test_no_zero_vote_padding_and_zero_vector_is_visible(self):
        X = np.array([[1, 0], [.9, .1], [0, 1], [0, -1], [-1, 0]], np.float32)
        y = np.array([0, 0, 1, 2, 3])
        model = cm.fit_classifier("knn", X, y, 4, n_neighbors=2)
        pred = cm.predict_topk(model, np.array([[1, 0], [0, 0]], np.float32))
        np.testing.assert_array_equal(pred["topk_indices"][0], [0, -1, -1, -1, -1])
        np.testing.assert_array_equal(pred["zero_query_mask"], [False, True])
        self.assertEqual(int(pred["valid_mask"][0].sum()), 1)

    def test_score_ties_positive_support_and_negative_svm_margins(self):
        scores = np.array([[.5, .5, 0, 0], [-2, -1, -1, -4]])
        ranked = cm.rank_scores(scores, 5)
        np.testing.assert_array_equal(ranked["topk_indices"], [[0, 1, 2, 3, -1], [1, 2, 0, 3, -1]])
        supported = cm.rank_scores(scores[:1], 5, positive_only=True)
        np.testing.assert_array_equal(supported["topk_indices"], [[0, 1, -1, -1, -1]])
        with self.assertRaises(ValueError):
            cm.rank_scores([[np.nan, 1]])

    def test_svm_and_forest_fit_known_separable_data(self):
        X, y = self.fixture()
        for method, kwargs in (("linear_svm", {"C": 1.0}),
                               ("random_forest", {"n_estimators": 16, "min_samples_leaf": 1, "n_jobs": 1})):
            with self.subTest(method=method):
                model = cm.fit_classifier(method, X, y, 3, **kwargs)
                pred = cm.predict_topk(model, X, batch_size=2)
                np.testing.assert_array_equal(pred["topk_indices"][:, 0], y)
                self.assertEqual(pred["topk_indices"].shape, (6, 5))
                self.assertIn("warnings", model.fit_info)

    def test_binary_svm_margin_direction(self):
        X = np.array([[-2], [-1], [1], [2]], np.float32)
        y = np.array([0, 0, 1, 1])
        model = cm.fit_classifier("linear_svm", X, y, 2, C=1.0)
        pred = cm.predict_topk(model, X)
        np.testing.assert_array_equal(pred["topk_indices"][:, 0], y)

    def test_parallel_ovr_class_alignment_and_worker_warnings(self):
        X, y = self.fixture()
        model = cm.fit_classifier("linear_svm", X, y, 3, C=1.0, parallel_ovr=True, n_jobs=2)
        np.testing.assert_array_equal(cm.predict_topk(model, X)["topk_indices"][:, 0], y)
        self.assertEqual(len(model.fit_info["n_iter_by_estimator"]), 3)
        self.assertEqual(model.fit_info["class_ids"], [0, 1, 2])
        with warnings.catch_warnings(record=True):
            poor = cm.fit_classifier("linear_svm", X, y, 3, parallel_ovr=True,
                                     n_jobs=1, max_iter=1, tol=1e-12)
        self.assertFalse(poor.fit_info["converged"])
        self.assertTrue(poor.fit_info["binary_warnings"])

    def test_convergence_warning_is_recorded(self):
        X, y = self.fixture()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = cm.fit_classifier("linear_svm", X, y, 3, max_iter=1, tol=1e-12)
        self.assertFalse(model.fit_info["converged"])
        self.assertEqual(model.fit_info["n_iter"], 1)
        self.assertTrue(any(issubclass(w.category, ConvergenceWarning) for w in caught))

    def test_forest_memory_bound_and_resource_limits(self):
        estimate = cm.forest_memory_upper_bound(2731, 64, 256)
        self.assertLess(estimate, 1024**3)
        X, y = self.fixture()
        for kwargs in ({"max_leaf_nodes": None}, {"n_jobs": 9}, {"n_estimators": 100000}, {"max_depth": None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cm.fit_classifier("random_forest", X, y, 3, **kwargs)

    def test_reload_predictions_and_scaler_integrity(self):
        X, y = self.fixture()
        scaler = cm.fit_train_scaler(X)
        Z = cm.transform_features(scaler, X)
        with tempfile.TemporaryDirectory() as temp:
            for method in cm.DEFAULT_PARAMETERS:
                kwargs = {"n_estimators": 4, "n_jobs": 1} if method == "random_forest" else {}
                model = cm.fit_classifier(method, Z, y, 3, **kwargs)
                path = Path(temp) / (method + ".joblib")
                digest = cm.save_model(model, path, scaler=scaler)
                restored, restored_scaler = cm.load_model(path, expected_sha256=digest)
                original = cm.predict_topk(model, Z)
                replica = cm.predict_topk(restored, cm.transform_features(restored_scaler, X))
                for name in ("topk_indices", "topk_scores", "valid_mask"):
                    np.testing.assert_array_equal(original[name], replica[name])
                with self.assertRaises(FileExistsError):
                    cm.save_model(model, path)
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    cm.load_model(path, expected_sha256="0" * 64)

    def test_query_dimension_and_mutated_class_map_rejected(self):
        X, y = self.fixture()
        model = cm.fit_classifier("linear_svm", X, y, 3)
        with self.assertRaisesRegex(ValueError, "dimension"):
            cm.predict_topk(model, np.ones((2, 3)))
        model.estimator.classes_[0] = 9
        with self.assertRaisesRegex(ValueError, "mapping"):
            cm.predict_topk(model, X)

    def test_empty_queries_preserve_shape(self):
        X, y = self.fixture()
        for method in cm.DEFAULT_PARAMETERS:
            kwargs = {"n_estimators": 2, "n_jobs": 1} if method == "random_forest" else {}
            model = cm.fit_classifier(method, X, y, 3, **kwargs)
            result = cm.predict_topk(model, np.empty((0, 2)))
            self.assertEqual(result["topk_indices"].shape, (0, 5))
            self.assertEqual(result["rows_with_fewer_than_top_k"], 0)

    def test_cuda_exact_ties_and_tf32_flag_restored_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        X = np.array([[1, 0], [1, 0], [0, 1], [0, -1], [-1, 0], [-1, 0]], np.float32)
        y = np.array([2, 1, 0, 0, 1, 2])
        model = cm.fit_classifier("knn", X, y, 3, n_neighbors=4)
        prior = torch.backends.cuda.matmul.allow_tf32
        cpu = cm.predict_topk(model, X, reference_chunk_size=2)
        gpu = cm.predict_topk(model, X, device="cuda:0", reference_chunk_size=2)
        self.assertEqual(prior, torch.backends.cuda.matmul.allow_tf32)
        for name in ("neighbor_indices", "neighbor_similarities", "topk_indices", "topk_scores", "valid_mask"):
            np.testing.assert_array_equal(cpu[name], gpu[name])


if __name__ == "__main__":
    unittest.main()
