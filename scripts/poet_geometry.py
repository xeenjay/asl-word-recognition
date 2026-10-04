"""Versioned image-xy geometry; no new extraction or temporal resampling.

Inputs are the existing raw_pose.npz arrays. Geometry uses the shared image
coordinates BEFORE articulator-specific normalization; part-local z is ignored.
"""
from __future__ import annotations

import copy
from typing import Mapping

import numpy as np

GEOMETRY_VERSION = "image_xy_wrist_anchors_v1"
RELATIONS = (
    ("right_wrist", "nose"), ("right_wrist", "mouth"),
    ("right_wrist", "shoulders"), ("left_wrist", "nose"),
    ("left_wrist", "mouth"), ("left_wrist", "shoulders"),
    ("right_wrist", "left_wrist"),
)
GEOMETRY_DIM = 4 * len(RELATIONS)
GEOMETRY_SPEC = {
    "version": GEOMETRY_VERSION,
    "dimension": GEOMETRY_DIM,
    "coordinates": "MediaPipe normalized image x/y; part-local z ignored",
    "anchors": {"right_wrist": {"part": "right_hand", "indices": [0]},
                "left_wrist": {"part": "left_hand", "indices": [0]},
                "nose": {"part": "face_mesh", "indices": [1]},
                "mouth": {"part": "face_mesh", "indices": [13, 14], "reduction": "mean"},
                "shoulders": {"part": "body_pose", "indices": [11, 12], "reduction": "mean"}},
    "relations": [list(pair) for pair in RELATIONS],
    "relation_components": ["target_minus_source_dx", "target_minus_source_dy", "distance", "present"],
    "scale": "per-video median shoulder width among valid body frames; no dataset fit",
    "shoulder_visibility_threshold": 0.5,
    "minimum_shoulder_width": 1e-6,
    "missing_scale": "all geometry values and presence bits zero; no fallback scale",
    "missing_relation": "three coordinates and presence bit zero",
    "temporal_alignment": "caller-supplied prepare_features source_frame_indices; -1 maps to zeros",
    "limitation": "2D image geometry, not metric 3D distance or a contact label",
}


def geometry_spec() -> dict:
    """Return a plain metadata copy suitable for a protocol/checkpoint."""
    return copy.deepcopy(GEOMETRY_SPEC)


def build_geometry_features(raw: Mapping[str, np.ndarray], source_frame_indices: np.ndarray) -> np.ndarray:
    """Return float32 (S,28), using precisely the supplied main-branch indices.

    Repeated indices repeat geometry identically. Only -1 represents padding;
    empty, fractional, or out-of-range indices are rejected. Full raw arrays
    are required even when the main branch crops a sequence: scale is a fixed
    per-video statistic, independent of train/eval temporal sampling.
    """
    parts = {"body_pose": 33, "face_mesh": 468, "right_hand": 21, "left_hand": 21}
    frames = None
    for name, points in parts.items():
        value = np.asarray(raw[name])
        if value.ndim != 3 or value.shape[1:] != (points, 3) or len(value) == 0:
            raise ValueError(f"{name} must have nonempty shape (T,{points},3)")
        if frames is None:
            frames = len(value)
        if len(value) != frames:
            raise ValueError("Raw parts must have equal frame counts")
        present = np.asarray(raw[name + "_present"])
        if present.shape != (frames,) or present.dtype != np.bool_:
            raise ValueError(f"{name}_present must be a boolean (T,) array")
        if not np.isfinite(value[..., :2]).all():
            raise ValueError(f"{name} image x/y contains NaN or infinity")
    visibility = np.asarray(raw["body_visibility"])
    if visibility.shape != (frames, 33) or not np.isfinite(visibility).all():
        raise ValueError("body_visibility must be finite with shape (T,33)")
    if np.any((visibility < 0) | (visibility > 1)):
        raise ValueError("body_visibility must lie in [0,1]")
    indices = np.asarray(source_frame_indices)
    if indices.ndim != 1 or len(indices) == 0 or indices.dtype.kind not in "iu":
        raise ValueError("source_frame_indices must be a nonempty integer vector")
    if np.any(indices < -1) or np.any(indices >= frames):
        raise ValueError("source_frame_indices contains an out-of-range frame")

    body = np.asarray(raw["body_pose"], dtype=np.float64)[..., :2]
    face = np.asarray(raw["face_mesh"], dtype=np.float64)[..., :2]
    shoulder_width = np.linalg.norm(body[:, 11] - body[:, 12], axis=1)
    shoulder_valid = (np.asarray(raw["body_pose_present"])
                      & (visibility[:, [11, 12]] >= GEOMETRY_SPEC["shoulder_visibility_threshold"]).all(axis=1)
                      & (shoulder_width > GEOMETRY_SPEC["minimum_shoulder_width"]))
    result = np.zeros((frames, GEOMETRY_DIM), dtype=np.float64)
    if shoulder_valid.any():
        scale = float(np.median(shoulder_width[shoulder_valid]))
        anchors = {
            "right_wrist": np.asarray(raw["right_hand"], dtype=np.float64)[:, 0, :2],
            "left_wrist": np.asarray(raw["left_hand"], dtype=np.float64)[:, 0, :2],
            "nose": face[:, 1], "mouth": face[:, [13, 14]].mean(axis=1),
            "shoulders": body[:, [11, 12]].mean(axis=1),
        }
        present = {"right_wrist": np.asarray(raw["right_hand_present"]),
                   "left_wrist": np.asarray(raw["left_hand_present"]),
                   "nose": np.asarray(raw["face_mesh_present"]),
                   "mouth": np.asarray(raw["face_mesh_present"]), "shoulders": shoulder_valid}
        for number, (source, target) in enumerate(RELATIONS):
            valid = present[source] & present[target]
            offset = 4 * number
            delta = (anchors[target][valid] - anchors[source][valid]) / scale
            result[valid, offset:offset + 2] = delta
            result[valid, offset + 2] = np.linalg.norm(delta, axis=1)
            result[valid, offset + 3] = 1.0
    aligned = np.zeros((len(indices), GEOMETRY_DIM), dtype=np.float32)
    real = indices >= 0
    aligned[real] = result[indices[real]].astype(np.float32)
    if not np.isfinite(aligned).all():
        raise ValueError("Geometry overflowed float32; no silent clipping applied")
    return aligned
