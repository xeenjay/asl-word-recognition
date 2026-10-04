"""Shared, strict adapters for the pinned upstream PoET implementation.

The vendor directory is deliberately not modified. This module does not replace
its normalization, temporal preprocessing, or Transformer forward pass.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
VENDOR = WORKSPACE / "vendor" / "poet"
VENDOR_COMMIT = "dca74c30d9d604cb3b92de8fc73ce91484644259"
OFFICIAL_ASL_CHECKPOINT_SHA256 = "5511f1270bd252c586705da8233f7ca1caf0b079a8ddf498b5bb0606f5e6d5c0"
sys.path.insert(0, str(VENDOR))
from model import Transformer_CLS, preprocess  # noqa: E402
from mediapipe_extraction import articulators_to_array  # noqa: E402

PARTS = {"body_pose": 33, "face_mesh": 468, "right_hand": 21, "left_hand": 21}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(plain_metadata(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def plain_metadata(value: Any) -> Any:
    """Remove NumPy scalar types from metadata, never change model tensors."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {plain_metadata(key): plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_metadata(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unexpected metadata type: {type(value).__name__}")


def load_checkpoint(path: Path) -> dict:
    """Allow the inspected NumPy scalar metadata only for the official file."""
    if sha256(path) == OFFICIAL_ASL_CHECKPOINT_SHA256:
        # Audited with torch.serialization.get_unsafe_globals_in_checkpoint:
        # numpy.core.multiarray.scalar and numpy.dtype. Float dtype classes are
        # constructed dynamically by those globals and also require allowlisting.
        allowed = [np.core.multiarray.scalar, np.dtype,
                   type(np.dtype("float64")), type(np.dtype("float32"))]
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location="cpu", weights_only=True)
    # New smoke checkpoints contain plain metadata; other files receive no
    # broadened deserialization permission and no weights_only=False fallback.
    return torch.load(path, map_location="cpu", weights_only=True)


class _PlainMappingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"Label map contains unexpected global {module}.{name}")


def _read_mapping(path: Path) -> dict:
    with path.open("rb") as stream:
        result = _PlainMappingUnpickler(stream).load()
    if not isinstance(result, dict):
        raise ValueError(f"Expected a dictionary label map: {path}")
    return result


def resolve_device(value: str) -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {device}")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def require_finite(name: str, array: Any) -> None:
    finite = bool(torch.isfinite(array).all()) if torch.is_tensor(array) else bool(np.isfinite(array).all())
    if not finite:
        raise ValueError(f"{name} contains NaN or infinity; no zero replacement is applied")


@dataclass
class Bundle:
    model: torch.nn.Module
    checkpoint: dict
    checkpoint_path: Path
    checkpoint_dir: Path
    id_to_label: dict[int, str]
    label_to_id: dict[str, int]
    preprocessing: SimpleNamespace
    device: torch.device
    metadata: dict


def create_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    data = checkpoint["data_preprocessing"]
    config = checkpoint["model_parameters"]
    if config.get("model_name", "Transformer_CLS") != "Transformer_CLS":
        raise ValueError(f"This adapter expects Transformer_CLS, got {config.get('model_name')!r}")
    model = Transformer_CLS(
        input_size=data["keypoints"], num_classes=checkpoint["num_classes"],
        max_seq_length=data["max_seq_length"],
        num_encoder_layers=config["num_encoder_layers"],
        num_attention_heads=config["num_attention_heads"],
        hidden_dropout_prob=checkpoint["training_parameters"]["hidden_dropout_prob"],
        hidden_size=config["transformer_hidden_size"],
        mlp_classifier=config["mlp_classifier"], mlp_hidden_size=config["mlp_hidden_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for name, value in model.state_dict().items():
        require_finite(f"checkpoint tensor {name}", value)
    return model.to(device)


def load_bundle(checkpoint_dir: Path, device: torch.device, checkpoint_path: Path | None = None) -> Bundle:
    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_path = (checkpoint_path or checkpoint_dir / "ASL_Citizen_PoET.pth").resolve()
    # PyTorch 2.6 restricted loading: never silently retry with weights_only=False.
    checkpoint = load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a dictionary checkpoint")
    required = {
        "data_preprocessing": {"keypoints", "padding", "troncate", "cut", "max_seq_length", "fix_lenght"},
        "model_parameters": {"num_encoder_layers", "num_attention_heads", "transformer_hidden_size", "mlp_classifier", "mlp_hidden_size"},
        "training_parameters": {"hidden_dropout_prob"},
    }
    for group, fields in required.items():
        missing = fields - checkpoint.get(group, {}).keys()
        if missing:
            raise ValueError(f"Checkpoint {group} is missing {sorted(missing)}; do not guess preprocessing")
    if "model_state_dict" not in checkpoint or "num_classes" not in checkpoint:
        raise ValueError("Checkpoint is missing model_state_dict or num_classes")
    data = checkpoint["data_preprocessing"]
    if data["keypoints"] != 188:
        raise ValueError(f"Official extractor produces 188 features, checkpoint requests {data['keypoints']}")
    if data["padding"] not in {"0", "copy", "copy_startend"} or data["troncate"] not in {"center_crop", "downsampling"}:
        raise ValueError(f"Unsupported checkpoint padding/troncate: {data}")
    if data["cut"] not in (0, 1) or data["fix_lenght"] not in (0, 1) or data["max_seq_length"] < 1:
        raise ValueError(f"Invalid checkpoint temporal preprocessing: {data}")
    inverse_path = checkpoint_dir / "d_LId2Label.pkl"
    forward_path = checkpoint_dir / "d_Label2LId.pkl"
    id_to_label, label_to_id = _read_mapping(inverse_path), _read_mapping(forward_path)
    classes = checkpoint["num_classes"]
    if type(classes) is not int or classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if set(id_to_label) != set(range(classes)) or any(type(i) is not int or not isinstance(label, str) for i, label in id_to_label.items()):
        raise ValueError("id_to_label must contain contiguous integer class IDs matching num_classes")
    if len(label_to_id) != classes or any(type(i) is not int or not isinstance(label, str) or id_to_label.get(i) != label for label, i in label_to_id.items()):
        raise ValueError("The two label maps are not exact inverses")
    model = create_model(checkpoint, device)
    model.eval()
    metadata = {
        "vendor_commit": VENDOR_COMMIT,
        "vendor_sha256": {name: sha256(VENDOR / name) for name in ("model.py", "mediapipe_extraction.py", "protobuf_to_dict.py")},
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "label_map_sha256": {p.name: sha256(p) for p in (inverse_path, forward_path)},
        "num_classes": classes, "num_parameters": sum(p.numel() for p in model.parameters()),
        "data_preprocessing": data, "model_parameters": checkpoint["model_parameters"],
        "hidden_dropout_prob": checkpoint["training_parameters"]["hidden_dropout_prob"],
        "device": str(device), "packages": package_versions(),
    }
    return Bundle(model, checkpoint, checkpoint_path, checkpoint_dir, id_to_label, label_to_id, SimpleNamespace(**data), device, metadata)


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__}
    for package in ("mediapipe", "opencv-python", "opencv-contrib-python", "opencv-python-headless", "protobuf"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def validate_raw_pose(raw: dict[str, np.ndarray]) -> int:
    lengths = set()
    for name, points in PARTS.items():
        value = raw[name]
        if value.ndim != 3 or value.shape[1:] != (points, 3):
            raise ValueError(f"{name} expected (T,{points},3), got {value.shape}")
        require_finite(name, value)
        lengths.add(len(value))
        presence = raw[name + "_present"]
        if presence.shape != (len(value),) or presence.dtype != np.bool_:
            raise ValueError(f"{name}_present must be a boolean vector matching frames")
        if np.any(value[~presence] != 0):
            raise ValueError(f"{name} has coordinates in frames marked as missing")
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"Pose arrays are empty or have unequal frame counts: {lengths}")
    # Upstream retains zero hand coordinates when both hands are absent, then
    # uses the available body/face features. Absence is not an invalid array.
    # Keep presence masks and finite/schema checks; do not invent landmarks.
    return lengths.pop()


def normalize_raw_pose(raw: dict[str, np.ndarray]) -> np.ndarray:
    validate_raw_pose(raw)
    # Retain float64 arithmetic used by upstream's array conversion.
    values = tuple(np.asarray(raw[name], dtype=np.float64) for name in PARTS)
    features = articulators_to_array(values)
    require_finite("normalized features", features)
    if features.ndim != 2 or features.shape[1] != 188:
        raise ValueError(f"Unexpected official feature shape: {features.shape}")
    return features


def prepare_features(features: np.ndarray, args: SimpleNamespace, mode: str = "eval") -> tuple[torch.Tensor, np.ndarray]:
    require_finite("features before preprocessing", features)
    if features.ndim != 2 or len(features) == 0 or features.shape[1] != args.keypoints:
        raise ValueError(f"Invalid feature shape {features.shape}")
    # Carry a source index through the actual upstream preprocessing. Inserting
    # it at the start leaves the last-80-coordinate hand cut criterion unchanged.
    # No resampling algorithm is reimplemented; zero padding gets index -1.
    tagged = np.column_stack((np.arange(len(features), dtype=np.float64) + 1, features))
    transformed = preprocess(args, mediapipeFeature=tagged, mode=mode)
    source_indices = transformed[:, 0].round().to(torch.int64).numpy() - 1
    prepared = transformed[:, 1:].to(torch.float32).contiguous()
    require_finite("preprocessed features", prepared)
    if len(prepared) == 0 or len(prepared) > args.max_seq_length:
        raise ValueError(f"Preprocessing returned {len(prepared)} frames, but checkpoint capacity is {args.max_seq_length}; no implicit truncation is applied")
    if args.fix_lenght == 1 and len(prepared) != args.max_seq_length:
        raise ValueError("Official temporal preprocessing did not produce its configured length")
    if mode == "eval":
        reference = preprocess(args, mediapipeFeature=features, mode="eval").to(torch.float32)
        torch.testing.assert_close(prepared, reference, rtol=0, atol=0)
    return prepared, source_indices


def load_raw_pose(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = set(PARTS) | {name + "_present" for name in PARTS}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Pose cache {path} is missing {sorted(missing)}")
        raw = {key: archive[key] for key in required}
    validate_raw_pose(raw)
    return raw
