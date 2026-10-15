"""Member 3 self-check: python -m unittest tests.test_member3_random_forest -v

Checks Gini impurity, split search (against sklearn's decision tree used only as
an oracle), tree growth limits, prediction, the forest, and finally that the
forest clearly beats chance on the real mini dataset if it is prepared.
"""
import ast
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import students.random_forest as rf  # noqa: E402

MINI = ROOT / "data/classical-mini"


def depth_and_leaf_sizes(tree, X):
    depth = {0: 0}
    for node in range(len(tree["feature"])):
        for child in (tree["left"][node], tree["right"][node]):
            if child >= 0:
                depth[int(child)] = depth[node] + 1
    node = np.zeros(len(X), dtype=np.int64)
    while (tree["feature"][node] >= 0).any():
        active = tree["feature"][node] >= 0
        rows = np.flatnonzero(active)
        current = node[rows]
        go_left = X[rows, tree["feature"][current]] <= tree["threshold"][current]
        node[rows] = np.where(go_left, tree["left"][current], tree["right"][current])
    return max(depth.values()), np.bincount(node)[np.unique(node)]


class Todo1Gini(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(float(rf.gini_impurity([3, 1])), 0.375)
        np.testing.assert_allclose(rf.gini_impurity([[5, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1]]),
                                   [0.0, 0.5, 0.0, 2 / 3])


class Todo2BestSplit(unittest.TestCase):
    def test_obvious_cut(self):
        X = np.array([[1.0, 9], [2.0, 9], [3.0, 9], [10.0, 9]], dtype=np.float32)
        y = np.array([0, 0, 0, 1])
        feature, threshold, decrease = rf.best_split(X, y, 2, [0, 1])
        self.assertEqual(feature, 0)
        self.assertAlmostEqual(threshold, 6.5)
        self.assertAlmostEqual(decrease, 0.375)

    def test_min_samples_leaf_and_no_valid_cut(self):
        X = np.array([[1.0], [2.0], [3.0], [10.0]], dtype=np.float32)
        y = np.array([0, 0, 0, 1])
        feature, threshold, _ = rf.best_split(X, y, 2, [0], min_samples_leaf=2)
        self.assertEqual((feature, threshold), (0, 2.5))
        self.assertEqual(rf.best_split(np.ones((4, 1), np.float32), y, 2, [0]), (None, None, 0.0))

    def test_root_split_agrees_with_sklearn(self):
        rng = np.random.default_rng(0)
        for case in range(10):
            X = rng.normal(size=(50, 6)).astype(np.float32)
            y = (X[:, case % 6] + 0.5 * rng.normal(size=50) > 0).astype(np.int64) + (X[:, 0] > 1)
            oracle = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
            feature, threshold, _ = rf.best_split(X, y, 3, range(6))
            self.assertEqual(feature, oracle.tree_.feature[0], f"case {case}")
            self.assertAlmostEqual(threshold, oracle.tree_.threshold[0], places=5)


class Todo3And4Tree(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.X = rng.normal(size=(80, 5)).astype(np.float32)
        self.y = ((self.X[:, 0] > 0).astype(np.int64) + 2 * (self.X[:, 1] > 0))

    def test_unlimited_tree_fits_training_data(self):
        tree = rf.build_tree(self.X, self.y, 4, None, 1, 5, np.random.default_rng(0))
        probabilities = rf.tree_predict_proba(tree, self.X)
        self.assertEqual(probabilities.shape, (80, 4))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), self.y)

    def test_depth_and_leaf_size_limits(self):
        tree = rf.build_tree(self.X, self.y, 4, 2, 5, 5, np.random.default_rng(0))
        depth, leaf_sizes = depth_and_leaf_sizes(tree, self.X)
        self.assertLessEqual(depth, 2)
        self.assertGreaterEqual(leaf_sizes.min(), 5)

    def test_pure_node_is_a_leaf(self):
        tree = rf.build_tree(self.X, np.zeros(80, np.int64), 2, None, 1, 5, np.random.default_rng(0))
        self.assertEqual(len(tree["feature"]), 1)
        np.testing.assert_array_equal(tree["value"][0], [1.0, 0.0])

    def test_handmade_tree_prediction(self):
        tree = {"feature": np.array([0, -1, 1, -1, -1]), "threshold": np.array([0.0, 0, 5.0, 0, 0]),
                "left": np.array([1, -1, 3, -1, -1]), "right": np.array([2, -1, 4, -1, -1]),
                "value": np.array([[.5, .5], [1, 0], [.2, .8], [0.0, 1.0], [.6, .4]])}
        X = np.array([[-1.0, 0], [1.0, 4], [1.0, 6], [0.0, 99]], dtype=np.float32)
        np.testing.assert_allclose(rf.tree_predict_proba(tree, X), [[1, 0], [0, 1], [.6, .4], [1, 0]])


class Todo5Forest(unittest.TestCase):
    def test_deterministic_bootstrap_and_saveable(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 8)).astype(np.float32)
        y = np.repeat(np.arange(3), 20)
        X[np.arange(60), y] += 2.0
        config = {"n_estimators": 7, "max_depth": 4, "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 5}
        first, second = rf.fit(X, y, 3, config), rf.fit(X, y, 3, config)
        self.assertEqual(len(first["trees"]), 7)
        for a, b in zip(first["trees"], second["trees"]):
            for key in ("feature", "threshold", "left", "right", "value"):
                np.testing.assert_array_equal(a[key], b[key])
        self.assertFalse(all(np.array_equal(first["trees"][0]["threshold"], t["threshold"]) for t in first["trees"][1:]),
                         "every tree looks identical; did you draw a new bootstrap sample per tree?")
        restored = pickle.loads(pickle.dumps(first))
        prediction = rf.predict_topk(restored, X, 5)
        self.assertGreater((prediction["topk_indices"][:, 0] == y).mean(), 0.8)


class EndToEnd(unittest.TestCase):
    def test_does_not_import_the_reference_or_sklearn(self):
        tree = ast.parse(Path(rf.__file__).read_text())
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertFalse({name for name in imported if "team" in name or "classical_models" in name or "sklearn" in name})

    @unittest.skipUnless((MINI / "manifest.json").is_file(), "prepare the mini dataset first")
    def test_runner_on_mini_beats_chance_clearly(self):
        from team.runner import run_experiment
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(method="random_forest", implementation="student", data_dir=MINI, output_dir=Path(folder) / "s")
        self.assertEqual(result["implementation"]["effective"], "student_modified")
        self.assertGreaterEqual(result["metrics"]["top1"], 0.35, "10 classes: chance is 0.10; the reference gets 0.60")
        self.assertGreaterEqual(result["metrics"]["top5"], 0.70)


if __name__ == "__main__":
    unittest.main()
