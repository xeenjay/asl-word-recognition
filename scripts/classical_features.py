"""Deterministic, non-learned pose/motion descriptors for classical classifiers.

This module depends only on NumPy. It consumes the existing raw MediaPipe cache,
not PoET-normalized input or a learned embedding. All decoded frames contribute;
there is no label access, dataset fitting, temporal crop, interpolation, or model
checkpoint. The same descriptor can be used by kNN, SVM, and random forests.

The 1,020 dimensions are: seven temporal cells x 98 mean coordinates, 98 global
standard deviations, 98 signed and 98 absolute adjacent-frame changes, seven
cells x five valid-observation fractions, and five valid-adjacent-pair fractions.
Motion is scaled by T-1, i.e. per unit normalized clip time, not per real second.
Missing observations remain missing, and differences never cross missing frames.
"""
from __future__ import annotations

import copy
from typing import Mapping

import numpy as np

VERSION = "classical-pose-motion-v1"
FEATURE_DIM = 1020
FRAME_DIM = 98
MIN_SCALE = 1e-6
VISIBILITY_THRESHOLD = 0.5
BODY_POINTS = (0, 11, 12, 13, 14, 15, 16)
PALM_POINTS = (5, 9, 13, 17)
LEVELS = (1, 2, 4)
GROUPS = (
    ("left_hand_shape", 0, 40),
    ("right_hand_shape", 40, 80),
    ("upper_body", 80, 94),
    ("left_wrist_body", 94, 96),
    ("right_wrist_body", 96, 98),
)
_PARTS = {"body_pose": 33, "left_hand": 21, "right_hand": 21}

FEATURE_SPEC = {
    "version": VERSION,
    "dimension": FEATURE_DIM,
    "frame_dimension": FRAME_DIM,
    "input": "raw_pose.npz arrays, never PoET features or checkpoints",
    "coordinates": "original image-normalized x/y; no metric 3D interpretation",
    "required_parts": dict(_PARTS),
    "required_masks": [name + "_present" for name in _PARTS],
    "required_visibility": "body_visibility, shape (T,33), finite in [0,1]",
    "ignored": "face_mesh, per-articulator z in features, frame IDs, timestamps, labels and signer metadata",
    "validation": "required part xyz must be finite and zero where presence=False; nonempty equal T; Boolean masks; invalid inputs raise ValueError",
    "groups": [{"name": name, "start": start, "stop": stop} for name, start, stop in GROUPS],
    "group_validity": {
        "left_hand_shape": "left hand present and its current palm size > minimum scale",
        "right_hand_shape": "right hand present and its current palm size > minimum scale",
        "upper_body": "body present; visible nondegenerate shoulders; all seven selected body landmarks visible",
        "left_wrist_body": "left hand present and current shoulders valid",
        "right_wrist_body": "right hand present and current shoulders valid",
    },
    "hand_shape": {
        "landmarks": list(range(1, 21)),
        "origin": "same hand wrist (landmark 0), per frame",
        "palm_landmarks": list(PALM_POINTS),
        "scale": "per-video median of mean wrist-to-four-MCP xy distances over present nondegenerate hand frames, separately per hand",
    },
    "body": {
        "landmarks": list(BODY_POINTS),
        "origin": "current midpoint of landmarks 11 and 12",
        "scale": "per-video median xy shoulder width over body-present frames with both shoulders visible and width > minimum scale",
    },
    "wrist_body": "hand landmark 0 minus current shoulder midpoint, divided by the same body scale",
    "visibility_threshold_inclusive": VISIBILITY_THRESHOLD,
    "minimum_scale_exclusive": MIN_SCALE,
    "invariance": "global xy translation and positive uniform scaling, provided validity thresholds are not crossed; no rotation or handedness normalization",
    "temporal_levels": list(LEVELS),
    "temporal_cells": "level order then left-to-right np.array_split(arange(T), level); empty cells have zero values and coverage",
    "aggregation": "means/std use only valid group frames; std uses ddof=0; missing is never a coordinate value",
    "motion": "mean signed / mean absolute differences of valid consecutive frames, multiplied by T-1; no bridging/interpolation; zero if no valid adjacent pairs",
    "coverage": "valid frames / cell length, then valid adjacent pairs / (T-1); zero for empty cells or T=1",
    "block_slices": {
        "temporal_means": [0, 686], "global_std": [686, 784],
        "signed_motion": [784, 882], "absolute_motion": [882, 980],
        "temporal_coverage": [980, 1015], "adjacent_coverage": [1015, 1020],
    },
    "missing_policy": "unavailable groups produce zero statistics plus explicit coverage; no fallback scale; entirely missing clips produce an all-zero vector and must remain in the evaluation denominator",
    "output": "one finite float32 vector; overflow raises ValueError, never silently clips",
    "fit_policy": "no dataset-level fit; caller must fit any scaler/classifier on train only",
    "limitations": "hand-engineered 2D summary, no face expression or absolute duration, unequal motion speeds partly normalized, not an established published baseline architecture",
}


def feature_spec() -> dict:
    """Return an independent JSON-serializable specification for run manifests."""
    return copy.deepcopy(FEATURE_SPEC)


def _validate(raw: Mapping[str, np.ndarray]) -> int:
    frames = None
    for name, points in _PARTS.items():
        if name not in raw or name + "_present" not in raw:
            raise ValueError(f"Missing required raw part/mask: {name}")
        value = np.asarray(raw[name])
        if value.ndim != 3 or value.shape[1:] != (points, 3) or len(value) == 0:
            raise ValueError(f"{name} must have nonempty shape (T,{points},3)")
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{name} xyz must be finite real numeric values")
        if frames is None:
            frames = len(value)
        if len(value) != frames:
            raise ValueError("Required raw parts must have equal frame counts")
        present = np.asarray(raw[name + "_present"])
        if present.shape != (frames,) or present.dtype != np.bool_:
            raise ValueError(f"{name}_present must be a boolean (T,) vector")
        if np.any(value[~present] != 0):
            raise ValueError(f"{name} must contain zero coordinates in missing frames")
    if "body_visibility" not in raw:
        raise ValueError("Missing required body_visibility")
    visibility = np.asarray(raw["body_visibility"])
    if visibility.shape != (frames, 33) or visibility.dtype.kind not in "fiu" or not np.isfinite(visibility).all():
        raise ValueError("body_visibility must be finite real values with shape (T,33)")
    if np.any((visibility < 0) | (visibility > 1)):
        raise ValueError("body_visibility must lie in [0,1]")
    return int(frames)


def _frame_values(raw: Mapping[str, np.ndarray], frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Build coordinates and five group-validity masks without modifying raw."""
    values = np.zeros((frames, FRAME_DIM), dtype=np.float64)
    valid = np.zeros((frames, len(GROUPS)), dtype=np.bool_)
    for group, name in enumerate(("left_hand", "right_hand")):
        hand = np.asarray(raw[name], dtype=np.float64)[..., :2]
        relative = hand - hand[:, :1]
        palm_size = np.linalg.norm(relative[:, PALM_POINTS], axis=-1).mean(axis=1)
        if not np.isfinite(palm_size).all():
            raise ValueError("Hand palm-size calculation overflowed")
        present = np.asarray(raw[name + "_present"]) & (palm_size > MIN_SCALE)
        valid[:, group] = present
        if present.any():
            scale = np.median(palm_size[present])
            start, stop = GROUPS[group][1:]
            values[present, start:stop] = (relative[present, 1:] / scale).reshape(-1, 40)

    body = np.asarray(raw["body_pose"], dtype=np.float64)[..., :2]
    visibility = np.asarray(raw["body_visibility"])
    width = np.linalg.norm(body[:, 11] - body[:, 12], axis=1)
    if not np.isfinite(width).all():
        raise ValueError("Body shoulder-width calculation overflowed")
    shoulder_valid = (np.asarray(raw["body_pose_present"])
                      & (visibility[:, [11, 12]] >= VISIBILITY_THRESHOLD).all(axis=1)
                      & (width > MIN_SCALE))
    if shoulder_valid.any():
        origin = (body[:, 11] + body[:, 12]) / 2
        scale = np.median(width[shoulder_valid])
        selected_visible = (visibility[:, BODY_POINTS] >= VISIBILITY_THRESHOLD).all(axis=1)
        valid[:, 2] = shoulder_valid & selected_visible
        values[valid[:, 2], 80:94] = ((body[valid[:, 2]][:, BODY_POINTS]
                                      - origin[valid[:, 2], None]) / scale).reshape(-1, 14)
        for group, name in ((3, "left_hand"), (4, "right_hand")):
            valid[:, group] = shoulder_valid & np.asarray(raw[name + "_present"])
            start, stop = GROUPS[group][1:]
            wrist = np.asarray(raw[name], dtype=np.float64)[:, 0, :2]
            values[valid[:, group], start:stop] = (wrist[valid[:, group]] - origin[valid[:, group]]) / scale
    if not np.isfinite(values).all():
        raise ValueError("Normalized frame coordinates overflowed")
    return values, valid


def extract_features(raw: Mapping[str, np.ndarray]) -> np.ndarray:
    """Convert one raw cache mapping into a finite 1,020-element float32 vector.

    Full-missing but correctly shaped input is a supported sample. Corrupt input
    is rejected, not silently zero-filled or skipped; the caller must preserve
    every official sample identity in its run manifest and evaluation counts.
    """
    frames = _validate(raw)
    values, valid = _frame_values(raw, frames)
    means, coverage = [], []
    for level in LEVELS:
        for indices in np.array_split(np.arange(frames), level):
            cell_mean = np.zeros(FRAME_DIM, dtype=np.float64)
            cell_coverage = np.zeros(len(GROUPS), dtype=np.float64)
            if len(indices):
                for group, (_, start, stop) in enumerate(GROUPS):
                    selected = indices[valid[indices, group]]
                    cell_coverage[group] = len(selected) / len(indices)
                    if len(selected):
                        cell_mean[start:stop] = values[selected, start:stop].mean(axis=0)
            means.append(cell_mean)
            coverage.append(cell_coverage)

    std = np.zeros(FRAME_DIM, dtype=np.float64)
    signed = np.zeros(FRAME_DIM, dtype=np.float64)
    absolute = np.zeros(FRAME_DIM, dtype=np.float64)
    pair_coverage = np.zeros(len(GROUPS), dtype=np.float64)
    for group, (_, start, stop) in enumerate(GROUPS):
        selected = valid[:, group]
        if selected.any():
            std[start:stop] = values[selected, start:stop].std(axis=0, ddof=0)
        if frames > 1:
            adjacent = selected[:-1] & selected[1:]
            pair_coverage[group] = adjacent.mean()
            if adjacent.any():
                # Do not concatenate all detected frames before differencing:
                # that would turn detection gaps into invented motion steps.
                delta = np.diff(values[:, start:stop], axis=0)[adjacent]
                signed[start:stop] = delta.mean(axis=0) * (frames - 1)
                absolute[start:stop] = np.abs(delta).mean(axis=0) * (frames - 1)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.concatenate([*means, std, signed, absolute, *coverage, pair_coverage]).astype(np.float32)
    if result.shape != (FEATURE_DIM,) or not np.isfinite(result).all():
        raise ValueError("Descriptor is non-finite or overflowed float32; no clipping applied")
    return result
