"""Label-free handedness canonicalization for the fixed classical descriptor.

Team repository copy of the research module (2026-10-10 experiment). Member 2
implements clip_indicators, signer_decisions and canonicalize_descriptors
(TODO H1-H3); see docs/roles/member2-linear-svm.md and run
    python -m unittest tests.test_member2_handedness -v
The mirror maps above those functions are provided complete.

A horizontal mirror maps image-normalized x to 1 - x, exchanges the left/right
hand arrays and exchanges the left/right MediaPipe Pose landmarks. For
``classical-pose-motion-v1`` this is exactly a fixed signed permutation of the
1,020 descriptor values. Both forms are provided so that the raw-pose path and
the descriptor path can be checked against each other.

The signer decision reads only unlabeled descriptor values (whole-clip hand
coverage and wrist motion). It never reads a gloss, prediction, score or test
data. Using the signer ID is a declared, label-free, signer-level adaptation:
every clip of one signer receives the same orientation.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from classical_features import BODY_POINTS, FEATURE_DIM, FEATURE_SPEC, FRAME_DIM, GROUPS, LEVELS

VERSION = "classical-handedness-canonical-v1"
POSE_LANDMARKS = 33
# MediaPipe Pose: eyes, ears, mouth corners, shoulders, elbows, wrists, hand
# points, hips, knees, ankles, heels and foot indices; 0 (nose) maps to itself.
POSE_MIRROR_PAIRS = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
                     (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
                     (27, 28), (29, 30), (31, 32))
USED_KEYS = ("body_pose", "body_pose_present", "body_visibility",
             "left_hand", "left_hand_present", "right_hand", "right_hand_present")
DECISION_THRESHOLD = 0.5
_GROUP_INDEX = {name: i for i, (name, _, _) in enumerate(GROUPS)}
GROUP_SWAP = np.array([_GROUP_INDEX[name] for name in (
    "right_hand_shape", "left_hand_shape", "upper_body", "right_wrist_body", "left_wrist_body")])


def _block(name: str) -> tuple[int, int]:
    start, stop = FEATURE_SPEC["block_slices"][name]
    return int(start), int(stop)


def _group_slice(name: str) -> tuple[int, int]:
    _, start, stop = GROUPS[_GROUP_INDEX[name]]
    return start, stop


# Whole-clip (level-1 cell) valid-frame fractions and absolute wrist motion.
COVERAGE_LEFT = _block("temporal_coverage")[0] + _GROUP_INDEX["left_hand_shape"]
COVERAGE_RIGHT = _block("temporal_coverage")[0] + _GROUP_INDEX["right_hand_shape"]
MOTION_LEFT = tuple(_block("absolute_motion")[0] + i for i in range(*_group_slice("left_wrist_body")))
MOTION_RIGHT = tuple(_block("absolute_motion")[0] + i for i in range(*_group_slice("right_wrist_body")))


def pose_permutation() -> np.ndarray:
    """Return the involutive left/right permutation of the 33 Pose landmarks."""
    perm = np.arange(POSE_LANDMARKS)
    for left, right in POSE_MIRROR_PAIRS:
        perm[left], perm[right] = right, left
    return perm


def _mirror_points(points: np.ndarray, present: np.ndarray) -> np.ndarray:
    # float64 keeps 1 - x exact for float32-origin coordinates; missing frames
    # stay exactly zero as required by the descriptor's input contract.
    values = np.asarray(points, dtype=np.float64)
    mask = np.asarray(present, dtype=np.bool_)
    if values.ndim != 3 or values.shape[2] < 2 or mask.shape != values.shape[:1]:
        raise ValueError("Pose arrays must be (T, landmarks, >=2) with a (T,) presence mask")
    result = np.zeros_like(values)
    result[mask] = values[mask]
    result[mask, :, 0] = 1.0 - values[mask, :, 0]
    return result


def mirror_raw_pose(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mirror the parts this descriptor reads; face mesh and z are not used."""
    missing = [key for key in USED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"Missing raw pose keys: {missing}")
    perm = pose_permutation()
    body_present = np.asarray(raw["body_pose_present"], dtype=np.bool_).copy()
    return {
        "body_pose": _mirror_points(raw["body_pose"], body_present)[:, perm],
        "body_pose_present": body_present,
        "body_visibility": np.asarray(raw["body_visibility"], dtype=np.float64)[:, perm],
        "left_hand": _mirror_points(raw["right_hand"], raw["right_hand_present"]),
        "left_hand_present": np.asarray(raw["right_hand_present"], dtype=np.bool_).copy(),
        "right_hand": _mirror_points(raw["left_hand"], raw["left_hand_present"]),
        "right_hand_present": np.asarray(raw["left_hand_present"], dtype=np.bool_).copy(),
    }


def _frame_map() -> tuple[np.ndarray, np.ndarray]:
    """Source index and x-component mask for one 98-value frame vector."""
    source = np.arange(FRAME_DIM)
    left, right = _group_slice("left_hand_shape"), _group_slice("right_hand_shape")
    source[left[0]:left[1]] = np.arange(*right)
    source[right[0]:right[1]] = np.arange(*left)
    body_start = _group_slice("upper_body")[0]
    perm = pose_permutation()
    order = list(BODY_POINTS)
    for slot, landmark in enumerate(order):
        partner = order.index(int(perm[landmark]))
        source[body_start + 2 * slot:body_start + 2 * slot + 2] = np.arange(
            body_start + 2 * partner, body_start + 2 * partner + 2)
    wrist_left, wrist_right = _group_slice("left_wrist_body"), _group_slice("right_wrist_body")
    source[wrist_left[0]:wrist_left[1]] = np.arange(*wrist_right)
    source[wrist_right[0]:wrist_right[1]] = np.arange(*wrist_left)
    # Every coordinate block starts at an even offset and stores (x, y) pairs.
    x_mask = np.arange(FRAME_DIM) % 2 == 0
    return source, x_mask


def descriptor_mirror_map() -> tuple[np.ndarray, np.ndarray]:
    """Return (source, sign) with mirrored = sign * original[source]."""
    frame_source, frame_x = _frame_map()
    flip = np.where(frame_x, -1.0, 1.0)
    keep = np.ones(FRAME_DIM)
    source = np.empty(FEATURE_DIM, dtype=np.int64)
    sign = np.ones(FEATURE_DIM, dtype=np.float64)
    offset = 0
    blocks = [flip] * sum(LEVELS) + [keep, flip, keep]  # cell means, std, signed, absolute
    for block_sign in blocks:
        source[offset:offset + FRAME_DIM] = offset + frame_source
        sign[offset:offset + FRAME_DIM] = block_sign
        offset += FRAME_DIM
    for _ in range(sum(LEVELS) + 1):  # per-cell coverage, then adjacent-pair coverage
        source[offset:offset + len(GROUPS)] = offset + GROUP_SWAP
        offset += len(GROUPS)
    if offset != FEATURE_DIM or sorted(source.tolist()) != list(range(FEATURE_DIM)):
        raise AssertionError("Descriptor mirror map is not a permutation of all values")
    return source, sign


def mirror_descriptors(features: np.ndarray) -> np.ndarray:
    """Apply the signed permutation to one vector or a matrix of descriptors."""
    values = np.asarray(features)
    if values.shape[-1] != FEATURE_DIM:
        raise ValueError("Descriptor width must be 1020")
    source, sign = descriptor_mirror_map()
    result = values[..., source] * sign.astype(values.dtype)
    return result + values.dtype.type(0)  # normalize -0.0 produced by negating zeros


def clip_indicators(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TODO H1. Per-clip left-dominance indicators from unlabeled descriptor values.

    Return two boolean arrays of length N:
      left_coverage[i]: x[i, COVERAGE_LEFT] > x[i, COVERAGE_RIGHT]
      left_motion[i]:   x[i, MOTION_LEFT].sum() > x[i, MOTION_RIGHT].sum()
    Equal values are NOT left-dominant. Reject inputs that are not (N, 1020).
    """
    # BEGIN SOLUTION member2
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM:
        raise ValueError("Expected an (N, 1020) descriptor matrix")
    left_coverage = values[:, COVERAGE_LEFT] > values[:, COVERAGE_RIGHT]
    left_motion = values[:, list(MOTION_LEFT)].sum(axis=1) > values[:, list(MOTION_RIGHT)].sum(axis=1)
    return left_coverage, left_motion
    # END SOLUTION


def signer_decisions(features: np.ndarray, signers: Sequence[str],
                     threshold: float = DECISION_THRESHOLD) -> dict[str, dict]:
    """TODO H2. Mirror a signer iff both left-dominance fractions exceed the threshold.

    For every signer (sorted IDs) return {"clips", "left_coverage_fraction",
    "left_motion_fraction", "mirror"}, where each fraction is the share of that
    signer's clips flagged by clip_indicators and mirror requires BOTH fractions
    to be strictly greater than threshold. One signer ID per row is required.
    """
    # BEGIN SOLUTION member2
    coverage, motion = clip_indicators(features)
    ids = np.asarray(list(signers))
    if ids.shape != (len(coverage),):
        raise ValueError("One signer ID is required per descriptor row")
    result = {}
    for signer in sorted(set(ids.tolist())):
        mask = ids == signer
        fraction_coverage = float(coverage[mask].mean())
        fraction_motion = float(motion[mask].mean())
        result[signer] = {"clips": int(mask.sum()),
                          "left_coverage_fraction": fraction_coverage,
                          "left_motion_fraction": fraction_motion,
                          "mirror": bool(fraction_coverage > threshold and fraction_motion > threshold)}
    return result
    # END SOLUTION


def canonicalize_descriptors(features: np.ndarray, signers: Sequence[str],
                             decisions: Mapping[str, Mapping]) -> np.ndarray:
    """TODO H3. Mirror (with mirror_descriptors) exactly the rows whose signer has mirror=True.

    Return a new array; rows of other signers must be unchanged copies.
    """
    # BEGIN SOLUTION member2
    values = np.asarray(features)
    flags = np.array([bool(decisions[s]["mirror"]) for s in signers])
    result = values.copy()
    if flags.any():
        result[flags] = mirror_descriptors(values[flags])
    return result
    # END SOLUTION


def rule_spec() -> dict:
    return {
        "version": VERSION,
        "inputs": "original classical-pose-motion-v1 descriptors only; no gloss, prediction, score, model or test data",
        "clip_left_coverage": f"x[{COVERAGE_LEFT}] > x[{COVERAGE_RIGHT}] (whole-clip valid-frame fraction, left vs right hand)",
        "clip_left_motion": f"x[{MOTION_LEFT[0]}]+x[{MOTION_LEFT[1]}] > x[{MOTION_RIGHT[0]}]+x[{MOTION_RIGHT[1]}] (absolute wrist motion relative to shoulder centre)",
        "signer_rule": f"per split and signer, mirror every clip iff both clip fractions > {DECISION_THRESHOLD}",
        "mirror": "x -> 1 - x for present landmarks in float64; swap left/right hands and presence; swap MediaPipe Pose left/right landmark pairs and visibilities",
        "pose_pairs": [list(pair) for pair in POSE_MIRROR_PAIRS],
        "unused_by_descriptor": "face mesh and z coordinates",
        "signer_metadata_used": True,
        "transductive_note": "A validation or test signer's decision uses that signer's unlabeled clips.",
    }
