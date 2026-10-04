"""Semantic tests for the fixed descriptor, without weights or dataset access."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from classical_features import FEATURE_DIM, VERSION, extract_features, feature_spec


def raw_fixture(frames=8):
    raw = {name: np.zeros((frames, points, 3), dtype=np.float64)
           for name, points in {"body_pose": 33, "left_hand": 21, "right_hand": 21}.items()}
    raw.update({name + "_present": np.ones(frames, dtype=np.bool_) for name in tuple(raw)})
    raw["body_visibility"] = np.ones((frames, 33), dtype=np.float64)
    raw["body_pose"][:, 0, :2] = [0.5, 0.2]
    raw["body_pose"][:, 11, :2] = [0.3, 0.5]
    raw["body_pose"][:, 12, :2] = [0.7, 0.5]
    raw["body_pose"][:, 13, :2] = [0.2, 0.6]
    raw["body_pose"][:, 14, :2] = [0.8, 0.6]
    raw["body_pose"][:, 15, :2] = [0.2, 0.3]
    raw["body_pose"][:, 16, :2] = [0.8, 0.3]
    for name, x in (("left_hand", 0.2), ("right_hand", 0.8)):
        raw[name][:, :, :2] = [x, 0.3]
        # A simple nondegenerate palm/finger layout, independent of the code's
        # normalization implementation. Uniform shape over time isolates motion.
        for point in range(1, 21):
            raw[name][:, point, :2] += [0.004 * point, -0.003 * point]
    return raw


def block(vector, name):
    start, stop = feature_spec()["block_slices"][name]
    return vector[start:stop]


def absent(raw, name, frames=slice(None)):
    raw[name][frames] = 0
    raw[name + "_present"][frames] = False


class ClassicalFeatureTests(unittest.TestCase):
    def test_schema_dimension_dtype_and_finite(self):
        result = extract_features(raw_fixture())
        self.assertEqual(result.shape, (FEATURE_DIM,))
        self.assertEqual(FEATURE_DIM, 1020)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(block(result, "temporal_coverage"), 1)
        np.testing.assert_array_equal(block(result, "adjacent_coverage"), 1)

    def test_known_body_relative_coordinates_and_constant_motion(self):
        result = extract_features(raw_fixture())
        # Shoulder center=(.5,.5), width=.4: nose=(0,-.75),
        # left wrist=(-.75,-.5), right wrist=(.75,-.5).
        np.testing.assert_allclose(result[80:82], [0, -0.75], atol=1e-6)
        np.testing.assert_allclose(result[94:98], [-0.75, -0.5, 0.75, -0.5], atol=1e-6)
        np.testing.assert_allclose(block(result, "global_std"), 0, atol=1e-7)
        np.testing.assert_array_equal(block(result, "signed_motion"), 0)
        np.testing.assert_array_equal(block(result, "absolute_motion"), 0)

    def test_global_translation_positive_scale_and_part_local_depth(self):
        raw = raw_fixture()
        raw["left_hand"][:, :, 0] += np.linspace(0, 0.1, len(raw["left_hand"]))[:, None]
        expected = extract_features(raw)
        transformed = copy.deepcopy(raw)
        for name in ("body_pose", "left_hand", "right_hand"):
            transformed[name][..., :2] = transformed[name][..., :2] * 2.7 + [0.21, -0.4]
            transformed[name][..., 2] = np.arange(transformed[name].shape[1]) + 5
        np.testing.assert_allclose(extract_features(transformed), expected, atol=2e-6, rtol=2e-6)

    def test_time_reversal_changes_order_and_negates_signed_motion(self):
        raw = raw_fixture(8)
        raw["left_hand"][:, :, 0] += np.linspace(0, 0.2, 8)[:, None]
        forward = extract_features(raw)
        backward = extract_features({name: value[::-1].copy() for name, value in raw.items()})
        self.assertFalse(np.allclose(forward, backward))
        fm = block(forward, "temporal_means").reshape(7, 98)
        bm = block(backward, "temporal_means").reshape(7, 98)
        np.testing.assert_allclose(fm[0], bm[0], atol=1e-6)
        np.testing.assert_allclose(fm[1:3], bm[1:3][::-1], atol=1e-6)
        np.testing.assert_allclose(fm[3:7], bm[3:7][::-1], atol=1e-6)
        np.testing.assert_allclose(block(forward, "signed_motion"), -block(backward, "signed_motion"), atol=1e-6)
        np.testing.assert_allclose(block(forward, "absolute_motion"), block(backward, "absolute_motion"), atol=1e-6)
        self.assertAlmostEqual(float(block(forward, "signed_motion")[94]), 0.5, places=6)

    def test_missing_frames_do_not_dilute_means_or_invent_gap_motion(self):
        raw = raw_fixture(3)
        raw["left_hand"][2, :, 0] += 0.2
        absent(raw, "left_hand", 1)
        result = extract_features(raw)
        # Observed left wrist x values are -.75 and -.25; middle missing is not 0.
        self.assertAlmostEqual(float(result[94]), -0.5, places=6)
        self.assertAlmostEqual(float(block(result, "temporal_coverage")[3]), 2 / 3, places=6)
        np.testing.assert_array_equal(block(result, "signed_motion")[:40], 0)
        np.testing.assert_array_equal(block(result, "signed_motion")[94:96], 0)
        np.testing.assert_array_equal(block(result, "absolute_motion")[94:96], 0)
        self.assertEqual(float(block(result, "adjacent_coverage")[3]), 0)

    def test_all_missing_is_finite_zero_and_not_dropped(self):
        raw = raw_fixture()
        for name in ("body_pose", "left_hand", "right_hand"):
            absent(raw, name)
        raw["body_visibility"][:] = 0
        np.testing.assert_array_equal(extract_features(raw), np.zeros(FEATURE_DIM, dtype=np.float32))

    def test_no_body_keeps_hand_shape_but_masks_body_locations(self):
        raw = raw_fixture()
        absent(raw, "body_pose")
        result = extract_features(raw)
        self.assertGreater(np.abs(result[:80]).sum(), 0)
        np.testing.assert_array_equal(result[80:98], 0)
        np.testing.assert_array_equal(block(result, "temporal_coverage")[:5], [1, 1, 0, 0, 0])

    def test_visibility_masks_unreliable_body_without_hiding_hand_shape(self):
        raw = raw_fixture()
        raw["body_visibility"][:, 11] = 0.49
        result = extract_features(raw)
        np.testing.assert_array_equal(result[80:98], 0)
        np.testing.assert_array_equal(block(result, "temporal_coverage")[:5], [1, 1, 0, 0, 0])
        raw = raw_fixture()
        raw["body_visibility"][:, 15] = 0.49
        result = extract_features(raw)
        np.testing.assert_array_equal(result[80:94], 0)
        np.testing.assert_array_equal(block(result, "temporal_coverage")[:5], [1, 1, 0, 1, 1])

    def test_degenerate_scales_have_no_arbitrary_fallback(self):
        raw = raw_fixture()
        raw["body_pose"][:, 12] = raw["body_pose"][:, 11]
        raw["left_hand"][:] = raw["left_hand"][:, :1]
        result = extract_features(raw)
        np.testing.assert_array_equal(result[:40], 0)
        np.testing.assert_array_equal(result[80:98], 0)
        np.testing.assert_array_equal(block(result, "temporal_coverage")[:5], [0, 1, 0, 0, 0])

    def test_single_frame_has_no_motion_and_empty_bins_are_explicit(self):
        result = extract_features(raw_fixture(1))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(block(result, "signed_motion"), 0)
        np.testing.assert_array_equal(block(result, "adjacent_coverage"), 0)
        coverage = block(result, "temporal_coverage").reshape(7, 5)
        np.testing.assert_array_equal(coverage[[0, 1, 3]], 1)
        np.testing.assert_array_equal(coverage[[2, 4, 5, 6]], 0)

    def test_nonfinite_bad_masks_empty_and_corrupt_missing_are_rejected(self):
        mutations = [
            lambda r: r["left_hand"].__setitem__((0, 1, 0), np.nan),
            lambda r: r["right_hand"].__setitem__((0, 0, 2), np.inf),
            lambda r: r.__setitem__("left_hand_present", np.ones(8, dtype=np.int64)),
            lambda r: r["left_hand_present"].__setitem__(0, False),
            lambda r: r.__setitem__("body_visibility", np.ones((8, 32))),
            lambda r: r["body_visibility"].__setitem__((0, 11), 1.1),
            lambda r: r.__setitem__("right_hand", r["right_hand"][:-1]),
        ]
        for mutate in mutations:
            raw = raw_fixture()
            mutate(raw)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                extract_features(raw)
        with self.assertRaises(ValueError):
            extract_features(raw_fixture(0))

    def test_inputs_unchanged_and_spec_independent_json(self):
        raw = raw_fixture()
        before = copy.deepcopy(raw)
        for value in raw.values():
            value.flags.writeable = False
        extract_features(raw)
        for key in raw:
            np.testing.assert_array_equal(raw[key], before[key])
        spec = feature_spec()
        json.dumps(spec, allow_nan=False)
        self.assertEqual(spec["version"], VERSION)
        spec["body"]["landmarks"][0] = 99
        self.assertEqual(feature_spec()["body"]["landmarks"][0], 0)


if __name__ == "__main__":
    unittest.main()
