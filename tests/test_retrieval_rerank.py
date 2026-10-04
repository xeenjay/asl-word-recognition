"""Leakage, kernel arithmetic, finite population, and output audit checks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rerank_retrieval_pilot as rr


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def synthetic_split(root, split, seed=14):
    root = Path(root)
    root.mkdir()
    rng = np.random.default_rng(seed + (split == "val"))
    classes = [2, 4, 7, 9, 12]
    rows, embeddings, candidates, logits = [], [], [], []
    for i in range(15 if split == "train" else 10):
        label = classes[i % 5]
        row = {"row": i, "index": i, "split": split,
               "class_id": label, "gloss": "G" + str(label), "signer_id": split + "-signer-" + str(i // 5),
               "video_sha256": digest(split + ":video:" + str(i)), "input_sha256": digest(split + ":input:" + str(i)),
               "source_manifest_sha256": digest(split + ":source")}
        row["identity"] = {"path": "videos/" + split + str(i) + ".mp4", "sha256": row["video_sha256"],
                           "gloss": row["gloss"], "signer_id": row["signer_id"], "split": split}
        rows.append(row)
        feature = rng.normal(size=128).astype(np.float32)
        feature[i % 5] += 4
        embeddings.append(feature)
        candidates.append(classes[i % 5:] + classes[:i % 5])
        logits.append([1.0, .99, .9, .8, .7])
    np.savez(root / "features.npz", embeddings=np.asarray(embeddings, np.float32),
             top5_class_ids=np.asarray(candidates, np.int64), top5_logits=np.asarray(logits, np.float32))
    (root / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"schema_version": 1, "status": "completed", "scope": "engineering", "test_used": False,
                "checkpoint_sha256": rr.CHECKPOINT_SHA256, "feature_dim": 128, "weights_unchanged": True,
                "optimizer_steps": 0, "failed": 0, "class_ids": classes, "gradients_absent": True, "model_eval": True,
                "label_map": {str(label): "G" + str(label) for label in classes},
                "splits": {split: {"samples": len(rows), "failed": 0, "complete_official_split": True,
                                    "features_path": "features.npz",
                                    "features_sha256": rr.sha256(root / "features.npz"),
                                    "metadata_path": "records.jsonl", "metadata_sha256": rr.sha256(root / "records.jsonl"),
                                    "source_manifest_sha256": digest(split + ":source"),
                                    "cache_manifest_sha256": digest(split + ":cache")}}}
    rr.write_json(root / "manifest.json", manifest)
    return manifest


class CacheChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "train"
        self.manifest = synthetic_split(self.root, "train")

    def load(self, engineering=True):
        rr.write_json(self.root / "manifest.json", self.manifest)
        return rr.load_split(self.root / "manifest.json", rr.sha256(self.root / "manifest.json"),
                             "train", engineering_smoke=engineering)

    def rows(self, transform):
        path = self.root / "records.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        transform(rows)
        for row in rows:
            row["identity"].update(sha256=row["video_sha256"], gloss=row["gloss"],
                                   signer_id=row["signer_id"], split=row["split"])
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.manifest["splits"]["train"]["metadata_sha256"] = rr.sha256(path)

    def arrays(self, transform):
        path = self.root / "features.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        transform(arrays)
        np.savez(path, **arrays)
        self.manifest["splits"]["train"]["features_sha256"] = rr.sha256(path)

    def test_valid_engineering_and_normalization(self):
        data = self.load()
        self.assertEqual(len(data["rows"]), 15)
        np.testing.assert_allclose(np.linalg.norm(data["normalized_embeddings"], axis=1), 1., atol=1e-15)

    def test_manifest_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Manifest SHA256"):
            rr.load_split(self.root / "manifest.json", "0" * 64, "train", engineering_smoke=True)

    def test_wrong_checkpoint(self):
        self.manifest["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "official checkpoint"):
            self.load()

    def test_unfrozen_and_optimizer_rejected(self):
        for key, value in (("weights_unchanged", False), ("optimizer_steps", 1)):
            with self.subTest(key=key):
                saved = self.manifest[key]
                self.manifest[key] = value
                with self.assertRaisesRegex(ValueError, "frozen weights"):
                    self.load()
                self.manifest[key] = saved

    def test_test_flag_or_split_rejected(self):
        self.manifest["test_used"] = True
        with self.assertRaisesRegex(ValueError, "Test data"):
            self.load()
        self.manifest["test_used"] = False
        self.rows(lambda rows: rows[0].update(split="test"))
        with self.assertRaisesRegex(ValueError, "Wrong split"):
            self.load()

    def test_missing_and_extra_split_rejected(self):
        self.manifest["splits"]["val"] = copy.deepcopy(self.manifest["splits"]["train"])
        with self.assertRaisesRegex(ValueError, "additional split"):
            self.load()

    def test_failed_rows_forbidden(self):
        self.manifest["failed"] = 1
        with self.assertRaisesRegex(ValueError, "Failed rows"):
            self.load()

    def test_formal_never_accepts_engineering(self):
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            self.load(engineering=False)
        self.manifest["scope"] = "formal"
        with self.assertRaisesRegex(ValueError, "2731 classes"):
            self.load(engineering=False)

    def test_formal_population_count_required(self):
        self.manifest.update(scope="formal", class_ids=list(range(2731)),
                             label_map={str(i): "G" + str(i) for i in range(2731)})
        with self.assertRaisesRegex(ValueError, "formal population"):
            self.load(engineering=False)

    def test_bad_artifact_hash(self):
        for name in ("features", "metadata"):
            with self.subTest(name=name):
                saved = self.manifest["splits"]["train"][name + "_sha256"]
                self.manifest["splits"]["train"][name + "_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    self.load()
                self.manifest["splits"]["train"][name + "_sha256"] = saved

    def test_path_escape(self):
        self.manifest["splits"]["train"]["features_path"] = "../features.npz"
        with self.assertRaisesRegex(ValueError, "contained"):
            self.load()

    def test_row_order(self):
        self.rows(lambda rows: rows.reverse())
        with self.assertRaisesRegex(ValueError, "row order"):
            self.load()

    def test_duplicate_video(self):
        self.rows(lambda rows: rows[1].update(video_sha256=rows[0]["video_sha256"]))
        with self.assertRaisesRegex(ValueError, "Duplicate video"):
            self.load()

    def test_duplicate_index(self):
        self.rows(lambda rows: rows[1].update(index=rows[0]["index"]))
        with self.assertRaisesRegex(ValueError, "source index"):
            self.load()

    def test_duplicate_identity(self):
        def duplicate(rows):
            rows[1].update({key: copy.deepcopy(value) for key, value in rows[0].items() if key not in ("row", "index")})
        self.rows(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate identity"):
            self.load()

    def test_bad_row_hash(self):
        self.rows(lambda rows: rows[0].update(input_sha256="garbage"))
        with self.assertRaisesRegex(ValueError, "input_sha256"):
            self.load()

    def test_row_source_manifest_binding(self):
        self.rows(lambda rows: rows[0].update(source_manifest_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Source manifest identity"):
            self.load()

    def test_unknown_label(self):
        self.rows(lambda rows: rows[0].update(class_id=1))
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            self.load()

    def test_gloss_mismatch(self):
        self.rows(lambda rows: rows[0].update(gloss="WRONG"))
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            self.load()

    def test_missing_class(self):
        self.rows(lambda rows: [row.update(class_id=4, gloss="G4") for row in rows if row["class_id"] == 2])
        with self.assertRaisesRegex(ValueError, "Missing class"):
            self.load()

    def test_candidate_duplicate_or_unknown(self):
        self.arrays(lambda arrays: arrays["top5_class_ids"].__setitem__((0, 1), arrays["top5_class_ids"][0, 0]))
        with self.assertRaisesRegex(ValueError, "Duplicate top-five"):
            self.load()

    def test_unknown_candidate_class(self):
        self.arrays(lambda arrays: arrays["top5_class_ids"].__setitem__((0, 1), 99))
        with self.assertRaisesRegex(ValueError, "Unknown candidate"):
            self.load()

    def test_embedding_wrong_dimension(self):
        self.arrays(lambda arrays: arrays.update(embeddings=arrays["embeddings"][:, :127]))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            self.load()

    def test_embedding_wrong_dtype(self):
        self.arrays(lambda arrays: arrays.update(embeddings=arrays["embeddings"].astype(np.float64)))
        with self.assertRaisesRegex(ValueError, "float32"):
            self.load()

    def test_near_zero_embedding(self):
        self.arrays(lambda arrays: arrays["embeddings"].__setitem__(0, 1e-16))
        with self.assertRaisesRegex(ValueError, "embedding norm"):
            self.load()

    def test_record_cache_binding(self):
        self.rows(lambda rows: rows[0].update(cache_manifest_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "cache manifest identity"):
            self.load()

    def test_candidate_integer_required(self):
        self.arrays(lambda arrays: arrays.update(top5_class_ids=arrays["top5_class_ids"].astype(np.float32)))
        with self.assertRaisesRegex(ValueError, "int64"):
            self.load()

    def test_logits_must_be_descending(self):
        self.arrays(lambda arrays: arrays["top5_logits"].__setitem__((0, 3), 20))
        with self.assertRaisesRegex(ValueError, "descending"):
            self.load()

    def test_finite_logits(self):
        self.arrays(lambda arrays: arrays["top5_logits"].__setitem__((0, 3), np.nan))
        with self.assertRaisesRegex(ValueError, "Non-finite top-five"):
            self.load()

    def test_zero_embedding(self):
        self.arrays(lambda arrays: arrays["embeddings"].__setitem__(0, 0))
        with self.assertRaisesRegex(ValueError, "embedding norm"):
            self.load()

    def test_nonfinite_embedding(self):
        self.arrays(lambda arrays: arrays["embeddings"].__setitem__((0, 0), np.inf))
        with self.assertRaisesRegex(ValueError, "Non-finite embeddings"):
            self.load()

    def test_extra_array_forbidden(self):
        self.arrays(lambda arrays: arrays.update(labels=np.zeros(15, np.int64)))
        with self.assertRaisesRegex(ValueError, "Unexpected feature"):
            self.load()

    def test_pair_signer_overlap_and_video_overlap(self):
        train = self.load()
        other = Path(self.temporary.name) / "val"
        synthetic_split(other, "val")
        val = rr.load_split(other / "manifest.json", rr.sha256(other / "manifest.json"), "val", engineering_smoke=True)
        rr.verify_pair(train, val)
        for key, message in (("signers", "signer overlap"), ("video_sha256s", "video overlap"),
                             ("identities", "identity overlap")):
            with self.subTest(key=key):
                altered = dict(val)
                altered[key] = val[key] | train[key]
                with self.assertRaisesRegex(ValueError, message):
                    rr.verify_pair(train, altered)


class KernelArithmetic(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(24)
        self.embeddings = rng.normal(size=(20, 128)).astype(np.float32)
        self.queries = rng.normal(size=(12, 128)).astype(np.float32)
        self.classes = list(range(5))
        self.rows = [{"split": "train", "class_id": i % 5, "signer_id": "P" + str(i // 7),
                      "video_sha256": digest(str(i))} for i in range(20)]
        self.candidates = np.array([np.roll(np.arange(5), i % 5) for i in range(12)], dtype=np.int64)

    def memory(self):
        return rr.build_memory(self.embeddings, self.rows, self.classes)

    def test_matches_independent_scalar_reference(self):
        memory = self.memory()
        actual = rr.memory_scores(memory, self.queries, self.candidates, chunk_size=3)
        expected = np.empty((12, 5))
        for i, query in enumerate(self.queries.astype(np.float64)):
            query = query / np.sqrt(sum(float(v) * float(v) for v in query))
            for position, label in enumerate(self.candidates[i]):
                by_signer = {}
                for row, feature in zip(self.rows, self.embeddings.astype(np.float64)):
                    if row["class_id"] == label:
                        feature = feature / np.sqrt(sum(float(v) * float(v) for v in feature))
                        cosine = min(1., max(-1., sum(float(a) * float(b) for a, b in zip(query, feature))))
                        by_signer.setdefault(row["signer_id"], []).append(np.exp(-10. * (1. - cosine)))
                expected[i, position] = np.mean([np.mean(values) for values in by_signer.values()])
        np.testing.assert_allclose(actual, expected, rtol=4e-15, atol=1e-19)

    def test_signers_equal_despite_different_example_counts(self):
        embeddings = np.zeros((4, 128), np.float32)
        embeddings[:, 0] = [1, 1, 1, -1]
        rows = [{"split": "train", "class_id": 0, "signer_id": "many" if i < 3 else "one",
                 "video_sha256": digest(str(i))} for i in range(4)]
        memory = rr.build_memory(embeddings, rows, [0])
        weights = memory[0]["weights"]
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertAlmostEqual(weights.max(), .5)
        cosine = memory[0]["embeddings"][:, 0]
        actual = np.sum(np.exp(-10 * (1 - cosine)) * weights)
        self.assertAlmostEqual(actual, (1 + np.exp(-20)) / 2, places=15)

    def test_memory_permutation_is_bit_exact(self):
        order = np.random.default_rng(7).permutation(len(self.rows))
        shuffled = rr.build_memory(self.embeddings[order], [self.rows[i] for i in order], self.classes)
        a = rr.memory_scores(self.memory(), self.queries, self.candidates)
        b = rr.memory_scores(shuffled, self.queries, self.candidates)
        np.testing.assert_array_equal(a, b)

    def test_query_chunk_size_is_bit_exact(self):
        a = rr.memory_scores(self.memory(), self.queries, self.candidates, chunk_size=1)
        for chunk in (2, 3, 100):
            with self.subTest(chunk=chunk):
                b = rr.memory_scores(self.memory(), self.queries, self.candidates, chunk_size=chunk)
                np.testing.assert_array_equal(a, b)

    def test_query_permutation_only_permutes_scores(self):
        order = np.random.default_rng(11).permutation(len(self.queries))
        a = rr.memory_scores(self.memory(), self.queries, self.candidates)
        b = rr.memory_scores(self.memory(), self.queries[order], self.candidates[order])
        np.testing.assert_array_equal(a[order], b)

    def test_positive_rescaling_embeddings_invariant(self):
        scaled = self.embeddings * np.float32(2.)
        memory = rr.build_memory(scaled, self.rows, self.classes)
        np.testing.assert_array_equal(rr.memory_scores(self.memory(), self.queries, self.candidates),
                                      rr.memory_scores(memory, self.queries * np.float32(4.), self.candidates))

    def test_reject_val_memory(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "training rows only"):
            rr.build_memory(self.embeddings, rows, self.classes)

    def test_missing_candidate_fails(self):
        memory = self.memory()
        del memory[1]
        with self.assertRaisesRegex(ValueError, "Missing candidate"):
            rr.memory_scores(memory, self.queries, self.candidates)

    def test_missing_memory_class_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing memory class"):
            rr.build_memory(self.embeddings, self.rows, list(range(6)))

    def test_candidate_ties_keep_original_rank(self):
        logits = np.zeros(self.candidates.shape, np.float32)
        scores = np.ones(self.candidates.shape, np.float64)
        result = rr.rank_candidates(self.candidates, logits, scores)
        for mode in rr.MODES:
            np.testing.assert_array_equal(result["rankings"][mode], self.candidates)

    def test_ranking_formula_and_top5_membership(self):
        logits = np.tile(np.array([1, .99, .98, .97, .96], np.float32), (12, 1))
        scores = np.tile(np.array([1, 2, 3, 4, 5], np.float64), (12, 1))
        result = rr.rank_candidates(self.candidates, logits, scores)
        expected_base = np.exp(logits.astype(np.float64))
        expected_base /= expected_base.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(result["probabilities"]["fixed_blend"], .75 * expected_base + .25 * scores / 15,
                                   rtol=1e-15)
        for mode in rr.MODES:
            np.testing.assert_array_equal(np.sort(result["rankings"][mode], axis=1),
                                          np.sort(self.candidates, axis=1))
        np.testing.assert_array_equal(result["rankings"]["pure_memory"], self.candidates[:, ::-1])

    def test_stable_partial_tie(self):
        logits = np.zeros((1, 5), np.float32)
        candidates = np.array([[7, 2, 4, 9, 1]], np.int64)
        scores = np.array([[2, 3, 3, 1, 2]], np.float64)
        result = rr.rank_candidates(candidates, logits, scores)
        np.testing.assert_array_equal(result["rankings"]["pure_memory"], [[2, 4, 7, 1, 9]])

    def test_zero_or_negative_score_rejected(self):
        logits = np.zeros((1, 5), np.float32)
        with self.assertRaisesRegex(ValueError, "Invalid memory scores"):
            rr.rank_candidates(self.candidates[:1], logits, np.array([[1, 2, 0, 4, 5]], np.float64))


class ResultsAndLifecycle(unittest.TestCase):
    def test_fixture_matches_fixed_recipe(self):
        path = Path(__file__).resolve().parents[1] / "tests/fixtures/retrieval_recipe.json"
        value = rr.load_preregistration(path, rr.sha256(path))
        self.assertEqual(value["retrieval"]["beta"], 10.)

    def test_changed_preregistration_recipe_rejected_even_with_new_sha(self):
        path = Path(__file__).resolve().parents[1] / "tests/fixtures/retrieval_recipe.json"
        value = json.loads(path.read_text())
        value["retrieval"]["blend_memory_weight"] = .5
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "prereg.json"
            rr.write_json(target, value)
            with self.assertRaisesRegex(ValueError, "retrieval mismatch"):
                rr.load_preregistration(target, rr.sha256(target))

    def test_macro_and_micro_unequal_population(self):
        targets = np.array([0, 0, 0, 1], np.int64)
        rankings = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], np.int64)
        result = rr.metrics(targets, rankings, [0, 1])
        self.assertEqual(result["top1"], .75)
        self.assertEqual(result["macro_top1"], .5)
        self.assertEqual(result["top5"], 1.)

    def test_gate_is_fixed_blend_only_and_integer_threshold(self):
        original = {"samples": 10304, "correct_top1": 8000, "top1": 8000 / 10304, "macro_top1": .8}
        def make(net, macro=.801):
            return {"original": original, "fixed_blend": {**original, "correct_top1": 8000 + net,
                    "top1": (8000 + net) / 10304, "macro_top1": macro},
                    "pure_memory": {**original, "top1": 1., "macro_top1": 1.}}
        self.assertFalse(rr.continuation_signal(make(51))["pilot_signal"])
        self.assertTrue(rr.continuation_signal(make(52))["pilot_signal"])
        self.assertFalse(rr.continuation_signal(make(52, .8))["pilot_signal"])
        self.assertFalse(rr.continuation_signal(make(52), engineering_smoke=True)["pilot_signal"])

    def make_args(self, root):
        for split in ("train", "val"):
            synthetic_split(root / split, split)
        return argparse.Namespace(train_features_manifest=str(root / "train" / "manifest.json"),
                                  train_manifest_sha256=rr.sha256(root / "train" / "manifest.json"),
                                  val_features_manifest=str(root / "val" / "manifest.json"),
                                  val_manifest_sha256=rr.sha256(root / "val" / "manifest.json"),
                                  output_dir=str(root / "result"), engineering_smoke=True, query_chunk_size=2)

    def test_end_to_end_all_rows_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)
            result = rr.run(args)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["population"]["val"], 10)
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertFalse(result["continuation_signal"]["pilot_signal"])
            with np.load(root / "result" / "scores.npz", allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), {"candidate_ids", "base_probability", "memory_kernel_scores",
                                                   "memory_probability", "blend_probability"})
                np.testing.assert_allclose(arrays["blend_probability"],
                                           .75 * arrays["base_probability"] + .25 * arrays["memory_probability"],
                                           rtol=0., atol=0.)
            hits = []
            for mode in rr.MODES:
                path = root / "result" / mode / "predictions.jsonl"
                predictions = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(predictions), 10)
                hits.append([row["correct_top5"] for row in predictions])
                for row in predictions:
                    self.assertEqual(set(row["ranked_class_ids"]), set(row["original_candidate_ids"]))
            self.assertEqual(hits[0], hits[1])
            self.assertEqual(hits[0], hits[2])
            for record in result["artifacts"]:
                self.assertEqual(rr.sha256(root / "result" / record["path"]), record["sha256"])
            before = rr.sha256(root / "result" / "summary.json")
            with self.assertRaises(FileExistsError):
                rr.run(args)
            self.assertEqual(rr.sha256(root / "result" / "summary.json"), before)

    def test_failure_evidence_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)
            args.train_manifest_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "Manifest SHA256 mismatch"):
                rr.run(args)
            state = json.loads((root / "result" / "state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertFalse((root / "result" / "summary.json").exists())

    def test_output_cannot_overlap_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_args(root)
            for output in (root, root / "train", root / "train" / "nested"):
                args.output_dir = str(output)
                with self.assertRaisesRegex(ValueError, "separate"):
                    rr.run(args)


if __name__ == "__main__":
    unittest.main()
