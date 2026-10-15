"""Member 1 self-check: python -m unittest tests.test_member1_knn -v

Small hand-made cases first, then an exact comparison with the reference kNN on
random data and (if prepared) on the real mini dataset.
"""
import ast
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import students.knn as knn  # noqa: E402
from team import reference  # noqa: E402  (oracle for the tests only)

MINI = ROOT / "data/classical-mini"


class Todo1Normalize(unittest.TestCase):
    def test_unit_rows_zero_rows_and_float32(self):
        X = np.array([[3, 4], [0, 0], [-1, 0]], dtype=np.float32)
        out = knn.l2_normalize_rows(X)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, [[0.6, 0.8], [0, 0], [-1, 0]], rtol=1e-6)
        np.testing.assert_array_equal(X, [[3, 4], [0, 0], [-1, 0]])  # input unchanged


class Todo2Cosine(unittest.TestCase):
    def test_shape_and_values(self):
        Q = knn.l2_normalize_rows(np.array([[1, 0], [1, 1]], dtype=np.float32))
        R = knn.l2_normalize_rows(np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32))
        S = knn.cosine_similarity(Q, R)
        self.assertEqual(S.shape, (2, 3))
        np.testing.assert_allclose(S, [[1, 0, -1], [2 ** -0.5, 2 ** -0.5, -(2 ** -0.5)]], rtol=1e-6)


class Todo3Neighbors(unittest.TestCase):
    def test_order_and_stable_ties(self):
        S = np.array([[0.2, 0.9, 0.9, -1.0, 0.5]], dtype=np.float32)
        ids, sims = knn.nearest_neighbors(S, 4)
        np.testing.assert_array_equal(ids, [[1, 2, 4, 0]])
        np.testing.assert_allclose(sims, [[0.9, 0.9, 0.5, 0.2]])


class Todo4Vote(unittest.TestCase):
    def test_majority_wins(self):
        self.assertEqual(knn.vote([2, 1, 2, 3, 2]), ([2, 1, 3], [0.6, 0.2, 0.2]))

    def test_tie_goes_to_nearer_support_then_class_id(self):
        # classes 7 and 4 have two votes each; 7 appears first (nearer).
        ids, scores = knn.vote([7, 4, 4, 7, 1])
        self.assertEqual(ids, [7, 4, 1])
        self.assertEqual(scores, [0.4, 0.4, 0.2])

    def test_never_pads_with_unvoted_classes_and_respects_top_k(self):
        self.assertEqual(knn.vote([5, 5, 5])[0], [5])
        self.assertEqual(len(knn.vote(list(range(9)), top_k=5)[0]), 5)


class Todo5SimilarityVote(unittest.TestCase):
    def test_weighted_scores_and_negative_weights_are_clipped(self):
        ids, scores = knn.similarity_vote([1, 2, 2, 3], [0.9, 0.4, 0.4, -0.5])
        self.assertEqual(ids, [1, 2])  # class 3 has weight 0 and is dropped
        np.testing.assert_allclose(scores, [0.9 / 1.7, 0.8 / 1.7])

    def test_all_nonpositive_falls_back_to_uniform_vote(self):
        self.assertEqual(knn.similarity_vote([4, 2, 4], [0.0, -0.1, -0.3]), knn.vote([4, 2, 4]))


def predictions_equal(first, second):
    return all(np.array_equal(first[name], second[name]) for name in ("topk_indices", "topk_scores", "valid_mask"))


class EndToEnd(unittest.TestCase):
    def test_does_not_import_the_reference(self):
        tree = ast.parse(Path(knn.__file__).read_text())
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertFalse({name for name in imported if "team" in name or "classical_models" in name or "sklearn" in name})

    def test_matches_reference_exactly_on_random_data(self):
        rng = np.random.default_rng(0)
        for case in range(25):
            n, classes = int(rng.integers(5, 60)), int(rng.integers(2, 10))
            X = rng.normal(size=(n, 12)).astype(np.float32)
            y = np.concatenate([np.arange(classes), rng.integers(0, classes, n - classes)]).astype(np.int64)
            X[2] = X[3]  # exact similarity tie
            Q = rng.normal(size=(7, 12)).astype(np.float32)
            Q[0] = 0     # an all-zero query
            k = int(rng.integers(1, n + 1))
            expected = reference.predict_topk(reference.fit("knn", X, y, classes, {"n_neighbors": k}), Q, 5)
            mine = knn.predict_topk(knn.fit(X, y, classes, {"n_neighbors": k}), Q, 5)
            self.assertTrue(predictions_equal(expected, mine), f"case {case} (k={k}) differs from the reference")

    @unittest.skipUnless((MINI / "manifest.json").is_file(), "prepare the mini dataset first")
    def test_runner_on_mini_matches_reference(self):
        from team.runner import run_experiment
        with tempfile.TemporaryDirectory() as folder:
            mine = run_experiment(method="knn", implementation="student", data_dir=MINI, output_dir=Path(folder) / "s")
            ref = run_experiment(method="knn", implementation="reference", data_dir=MINI, output_dir=Path(folder) / "r")
            self.assertEqual(mine["implementation"]["effective"], "student_modified")
            self.assertEqual(mine["implementation"]["reference_api_calls"], [])
            self.assertEqual(mine["metrics"], ref["metrics"])


if __name__ == "__main__":
    unittest.main()
