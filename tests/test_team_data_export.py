"""The local exporter must preserve identity and reject invalid provenance."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("build_team_data", Path(__file__).parents[1] / "scripts" / "build_team_data.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class LocalReferenceExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.labels = ["B", "C", "A"]
        self.dataset = {"schema_version": 1, "status": "completed", "test_used": False,
                        "id_to_label": self.labels,
                        "feature_spec": {"version": exporter.FEATURE_SPEC, "dimension": 1020}}
        self.records = {}
        for split, targets in [("train", [0, 1, 2, 0, 1, 2]), ("val", [0, 1, 2])]:
            n = len(targets)
            np.save(self.source / f"{split}_features.npy", np.repeat(np.arange(n, dtype=np.float32)[:, None], 1020, axis=1))
            np.save(self.source / f"{split}_labels.npy", np.array(targets, dtype=np.int64))
            rows = [{"index": i, "gloss": self.labels[y], "signer_id": split + "_signer",
                     "split": split, "source_path": "private/" + split + str(i) + ".mp4",
                     "video_sha256": hashlib.sha256((split+str(i)).encode()).hexdigest()}
                    for i,y in enumerate(targets)]
            self.records[split] = rows
            (self.source / f"{split}_rows.json").write_text(json.dumps(rows))
            self.dataset[split] = {"count": n, "artifacts": {}}
        self.seal_source()
        self.selection = {"selected_glosses": ["A", "C"], "splits": {}}
        for split in ("train", "val"):
            self.selection["splits"][split] = [{"video_file": Path(r["source_path"]).name,
                                      "gloss": r["gloss"], "video_sha256": r["video_sha256"]}
                                     for r in reversed(self.records[split]) if r["gloss"] in ["A", "C"]]
        self.selection_path = self.root / "selection.json"
        self.write_selection()

    def tearDown(self):
        self.temporary.cleanup()

    def seal_source(self):
        for split in ("train", "val"):
            for suffix in ("features.npy", "labels.npy", "rows.json"):
                name = f"{split}_{suffix}"
                path = self.source / name
                self.dataset[split]["artifacts"][name] = {"path": name, "sha256": exporter.sha256(path), "size_bytes": path.stat().st_size}
        (self.source / "dataset.json").write_text(json.dumps(self.dataset))

    def write_selection(self):
        self.selection_path.write_text(json.dumps(self.selection))

    def build(self, **kwargs):
        return exporter.build_package(self.source, self.selection_path, self.root / "out", source_run_id="synthetic-test", **kwargs)

    def test_preserves_selection_order_and_remaps_glosses(self):
        result = self.build()
        self.assertEqual(result["selected_original_class_ids"], [2, 1])
        np.testing.assert_array_equal(np.load(self.root / "out/train_labels.npy"), [0, 1, 0, 1])
        np.testing.assert_array_equal(np.load(self.root / "out/train_features.npy")[:, 0], [5, 4, 2, 1])
        self.assertEqual(json.loads((self.root / "out/labels.json").read_text()), ["A", "C"])
        text = (self.root / "out/manifest.json").read_text()
        self.assertNotIn("private/", text)
        self.assertNotIn("source_path", text)
        self.assertNotIn("video_sha256", text)
        for name, info in result["files"].items():
            self.assertEqual(exporter.sha256(self.root / "out" / name), info["sha256"])

    def test_rejects_tampered_source_array(self):
        path = self.source / "train_features.npy"
        value = np.load(path)
        value[0, 0] += 1
        np.save(path, value)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.build()

    def test_rejects_wrong_selected_video_identity(self):
        self.selection["splits"]["train"][0]["video_sha256"] = "0" * 64
        self.write_selection()
        with self.assertRaisesRegex(ValueError, "Selected video SHA256 mismatch"):
            self.build()

    def test_rejects_signer_overlap_even_with_valid_hashes(self):
        self.records["val"][0]["signer_id"] = "train_signer"
        (self.source / "val_rows.json").write_text(json.dumps(self.records["val"]))
        self.seal_source()
        with self.assertRaisesRegex(ValueError, "signers cross"):
            self.build()

    def test_rejects_gloss_label_misalignment(self):
        self.records["train"][0]["gloss"] = "A"
        (self.source / "train_rows.json").write_text(json.dumps(self.records["train"]))
        self.seal_source()
        with self.assertRaisesRegex(ValueError, "Gloss/array-label mismatch"):
            self.build()

    def test_rejects_duplicate_selected_rows(self):
        self.selection["splits"]["train"].append(self.selection["splits"]["train"][0])
        self.write_selection()
        with self.assertRaisesRegex(ValueError, "Duplicate selected"):
            self.build()

    def test_rejects_subset_mislabeled_as_full(self):
        with self.assertRaisesRegex(ValueError, "Full profile must retain"):
            self.build(profile="full")

    def test_never_overwrites_existing_output(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            self.build()


if __name__ == "__main__":
    unittest.main()
