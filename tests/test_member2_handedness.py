"""Member 2 extension self-check: python -m unittest tests.test_member2_handedness -v

The mirror-map tests pass from the start (those functions are provided). The
SignerRuleTests check your TODO H1-H3 in scripts/classical_handedness.py.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from classical_features import FEATURE_DIM, extract_features
from classical_handedness import (COVERAGE_LEFT, COVERAGE_RIGHT, MOTION_LEFT, MOTION_RIGHT, USED_KEYS,
                                  canonicalize_descriptors, clip_indicators, descriptor_mirror_map,
                                  mirror_descriptors, mirror_raw_pose, pose_permutation, signer_decisions)


def random_raw(seed: int, frames: int = 24, *, drop_left=0.3, drop_right=0.2, drop_body=0.1,
               low_visibility=0.15) -> dict:
    """Plausible cache-shaped pose with float32-origin coordinates and gaps."""
    rng = np.random.default_rng(seed)
    raw = {}
    for name, points, centre in (("body_pose", 33, 0.5), ("left_hand", 21, 0.35), ("right_hand", 21, 0.65)):
        drop = {"body_pose": drop_body, "left_hand": drop_left, "right_hand": drop_right}[name]
        present = rng.random(frames) >= drop
        coords = np.zeros((frames, points, 3), dtype=np.float32)
        spread = 0.25 if name == "body_pose" else 0.06
        coords[..., 0] = centre + rng.normal(0, spread, (frames, points))
        coords[..., 1] = 0.5 + rng.normal(0, spread, (frames, points))
        coords[..., 2] = rng.normal(0, 0.05, (frames, points))
        coords[~present] = 0
        raw[name] = coords.astype(np.float64)
        raw[name + "_present"] = present
    visibility = rng.random((frames, 33)).astype(np.float32).astype(np.float64)
    visibility[rng.random((frames, 33)) < low_visibility] = 0.1
    visibility[:, [11, 12]] = np.maximum(visibility[:, [11, 12]], 0.6)
    raw["body_visibility"] = visibility
    return raw


class MirrorRawPoseTests(unittest.TestCase):
    def test_pose_permutation_is_involution_with_fixed_nose(self):
        perm = pose_permutation()
        self.assertEqual(sorted(perm.tolist()), list(range(33)))
        np.testing.assert_array_equal(perm[perm], np.arange(33))
        self.assertEqual(perm[0], 0)
        self.assertEqual((perm[11], perm[13], perm[15]), (12, 14, 16))

    def test_mirroring_twice_restores_used_arrays(self):
        raw = random_raw(1)
        twice = mirror_raw_pose(mirror_raw_pose(raw))
        for key in USED_KEYS:
            np.testing.assert_array_equal(twice[key], raw[key], err_msg=key)

    def test_hands_swap_with_presence_and_missing_frames_stay_zero(self):
        raw = random_raw(2)
        mirrored = mirror_raw_pose(raw)
        np.testing.assert_array_equal(mirrored["left_hand_present"], raw["right_hand_present"])
        np.testing.assert_array_equal(mirrored["right_hand_present"], raw["left_hand_present"])
        present = raw["right_hand_present"]
        np.testing.assert_array_equal(mirrored["left_hand"][present, :, 0], 1 - raw["right_hand"][present, :, 0])
        np.testing.assert_array_equal(mirrored["left_hand"][present, :, 1:], raw["right_hand"][present, :, 1:])
        self.assertTrue(np.all(mirrored["left_hand"][~present] == 0))
        body_present = raw["body_pose_present"]
        np.testing.assert_array_equal(mirrored["body_pose"][body_present, 11, 0], 1 - raw["body_pose"][body_present, 12, 0])
        np.testing.assert_array_equal(mirrored["body_visibility"][:, 11], raw["body_visibility"][:, 12])

    def test_missing_keys_are_rejected(self):
        raw = random_raw(3)
        del raw["left_hand_present"]
        with self.assertRaises(ValueError):
            mirror_raw_pose(raw)


class DescriptorMapTests(unittest.TestCase):
    def test_map_is_signed_permutation_and_involution(self):
        source, sign = descriptor_mirror_map()
        self.assertEqual(sorted(source.tolist()), list(range(FEATURE_DIM)))
        self.assertTrue(set(np.unique(sign).tolist()) <= {-1.0, 1.0})
        values = np.random.default_rng(4).normal(size=(7, FEATURE_DIM)).astype(np.float32)
        np.testing.assert_array_equal(mirror_descriptors(mirror_descriptors(values)), values)

    def test_raw_path_equals_descriptor_path_exactly(self):
        cases = [random_raw(seed) for seed in range(10, 30)]
        cases.append(random_raw(40, frames=1))
        cases.append(random_raw(41, drop_left=1.0))          # left hand never present
        cases.append(random_raw(42, drop_left=1.0, drop_right=1.0))
        cases.append(random_raw(43, low_visibility=0.9))     # body group mostly invalid
        for n, raw in enumerate(cases):
            original = extract_features(raw)
            from_raw = extract_features(mirror_raw_pose(raw)) + np.float32(0)
            from_map = mirror_descriptors(original)
            np.testing.assert_array_equal(from_raw, from_map, err_msg=f"case {n}")

    def test_coverage_and_motion_indices_match_spec(self):
        self.assertEqual((COVERAGE_LEFT, COVERAGE_RIGHT), (980, 981))
        self.assertEqual((MOTION_LEFT, MOTION_RIGHT), ((976, 977), (978, 979)))
        raw = random_raw(5, drop_left=1.0)
        features = extract_features(raw)
        self.assertEqual(features[COVERAGE_LEFT], 0)
        self.assertGreater(features[COVERAGE_RIGHT], 0)
        mirrored = mirror_descriptors(features)
        self.assertEqual(mirrored[COVERAGE_RIGHT], 0)
        self.assertEqual(mirrored[COVERAGE_LEFT], features[COVERAGE_RIGHT])


class SignerRuleTests(unittest.TestCase):
    @staticmethod
    def rows(left_cov, right_cov, left_motion, right_motion):
        x = np.zeros(FEATURE_DIM, dtype=np.float32)
        x[COVERAGE_LEFT], x[COVERAGE_RIGHT] = left_cov, right_cov
        x[MOTION_LEFT[0]], x[MOTION_RIGHT[0]] = left_motion, right_motion
        return x

    def test_both_fractions_must_strictly_exceed_half(self):
        left_dominant = [self.rows(0.6, 0.2, 3, 1)] * 3 + [self.rows(0.1, 0.5, 1, 2)]
        coverage_only = [self.rows(0.6, 0.2, 1, 3)] * 3 + [self.rows(0.1, 0.5, 1, 2)]
        exactly_half = [self.rows(0.6, 0.2, 3, 1)] * 2 + [self.rows(0.1, 0.5, 1, 2)] * 2
        X = np.stack(left_dominant + coverage_only + exactly_half)
        signers = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
        decisions = signer_decisions(X, signers)
        self.assertEqual({k: v["mirror"] for k, v in decisions.items()}, {"A": True, "B": False, "C": False})
        self.assertEqual(decisions["A"]["clips"], 4)
        self.assertAlmostEqual(decisions["B"]["left_motion_fraction"], 0.0)

    def test_ties_do_not_count_as_left_dominant(self):
        coverage, motion = clip_indicators(np.stack([self.rows(0.3, 0.3, 1, 1)]))
        self.assertFalse(coverage[0])
        self.assertFalse(motion[0])

    def test_canonicalization_changes_only_mirrored_signers(self):
        X = np.random.default_rng(6).normal(size=(5, FEATURE_DIM)).astype(np.float32)
        signers = ["A", "B", "A", "B", "B"]
        decisions = {"A": {"mirror": True}, "B": {"mirror": False}}
        result = canonicalize_descriptors(X, signers, decisions)
        np.testing.assert_array_equal(result[[1, 3, 4]], X[[1, 3, 4]])
        np.testing.assert_array_equal(result[[0, 2]], mirror_descriptors(X[[0, 2]]))

    def test_rule_has_no_label_inputs(self):
        names = set(inspect.signature(signer_decisions).parameters)
        self.assertEqual(names, {"features", "signers", "threshold"})


if __name__ == "__main__":
    unittest.main()
