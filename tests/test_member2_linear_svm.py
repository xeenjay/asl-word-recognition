"""Member 2 self-check: python -m unittest tests.test_member2_linear_svm -v

Checks the targets, the objective and its gradient, the optimizer, the ranking,
and finally that your weights agree with sklearn's LinearSVC (used here only as
an oracle) on random data and, if prepared, on the real mini dataset.
"""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import students.linear_svm as svm  # noqa: E402

MINI = ROOT / "data/classical-mini"


def random_problem(seed, n=40, d=6, classes=4):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.arange(classes), rng.integers(0, classes, n - classes)]).astype(np.int64)
    centres = rng.normal(0, 2, (classes, d))
    X = (centres[y] + rng.normal(size=(n, d))).astype(np.float32)
    return X, y


class Todo1Targets(unittest.TestCase):
    def test_plus_one_for_own_class_minus_one_elsewhere(self):
        Y = svm.one_vs_rest_targets(np.array([2, 0]), 3)
        np.testing.assert_array_equal(Y, [[-1, -1, 1], [1, -1, -1]])
        self.assertEqual(Y.dtype, np.float64)


class Todo2Objective(unittest.TestCase):
    def test_hand_computed_value(self):
        Xb = np.array([[1.0, 1.0], [2.0, 1.0]])   # one feature + bias column
        Y = np.array([[1.0], [-1.0]])
        W = np.array([[0.5], [0.0]])
        # margins: 1*0.5 = 0.5 and -1*1.0 = -1.0 -> slack 0.5 and 2.0
        loss, _ = svm.squared_hinge_objective(W, Xb, Y, C=0.1)
        self.assertAlmostEqual(loss, 0.5 * 0.25 + 0.1 * (0.25 + 4.0))

    def test_gradient_matches_finite_differences(self):
        X, y = random_problem(1)
        Xb, Y = svm.add_bias_column(X), svm.one_vs_rest_targets(y, 4)
        W = np.random.default_rng(2).normal(0, 0.3, (Xb.shape[1], 4))
        _, grad = svm.squared_hinge_objective(W, Xb, Y, C=0.05)
        numeric = np.zeros_like(W)
        for index in np.ndindex(W.shape):
            step = np.zeros_like(W)
            step[index] = 1e-6
            plus, _ = svm.squared_hinge_objective(W + step, Xb, Y, C=0.05)
            minus, _ = svm.squared_hinge_objective(W - step, Xb, Y, C=0.05)
            numeric[index] = (plus - minus) / 2e-6
        np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-6)


class Todo3Training(unittest.TestCase):
    def test_converges_and_matches_linearsvc(self):
        X, y = random_problem(3)
        W, info = svm.train_weights(svm.add_bias_column(X), svm.one_vs_rest_targets(y, 4), C=0.05)
        self.assertTrue(info["converged"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oracle = LinearSVC(C=0.05, dual=False, tol=1e-6, max_iter=10000).fit(X, y)
        expected = np.vstack([oracle.coef_.T, oracle.intercept_[None]])
        self.assertLess(np.linalg.norm(W - expected) / np.linalg.norm(expected), 1e-3)


class Todo4Ranking(unittest.TestCase):
    def test_highest_first_ties_by_class_id_and_k_limit(self):
        ids, values = svm.top_k_from_scores(np.array([[0.1, 0.5, 0.5, -2.0]]), 3)
        np.testing.assert_array_equal(ids, [[1, 2, 0]])
        np.testing.assert_allclose(values, [[0.5, 0.5, 0.1]])
        self.assertEqual(svm.top_k_from_scores(np.zeros((2, 3)), 5)[0].shape, (2, 3))


class EndToEnd(unittest.TestCase):
    def test_does_not_import_the_reference_or_sklearn(self):
        tree = ast.parse(Path(svm.__file__).read_text())
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertFalse({name for name in imported if "team" in name or "classical_models" in name or "sklearn" in name})

    @unittest.skipUnless((MINI / "manifest.json").is_file(), "prepare the mini dataset first")
    def test_runner_on_mini_gives_the_reference_top5(self):
        from team.runner import run_experiment
        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mine = run_experiment(method="linear_svm", implementation="student", data_dir=MINI, output_dir=Path(folder) / "s")
            ref = run_experiment(method="linear_svm", implementation="reference", data_dir=MINI, output_dir=Path(folder) / "r")
            self.assertEqual(mine["implementation"]["effective"], "student_modified")
            ours = np.load(Path(folder) / "s/predictions.npz")["topk_indices"]
            theirs = np.load(Path(folder) / "r/predictions.npz")["topk_indices"]
            np.testing.assert_array_equal(ours, theirs)
            self.assertEqual(mine["metrics"]["top1"], ref["metrics"]["top1"])


if __name__ == "__main__":
    unittest.main()
