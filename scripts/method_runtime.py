"""Strict method construction, portable checkpoint metadata and training losses.

Call ``load_bundle`` on the pinned official checkpoint first.  Inference restores
only the dimensions/configuration embedded in ``method_metadata``: it never
opens an ASL-LEX target file or accepts ground-truth labels in the model forward.
No extraction/preprocessing code or upstream model implementation is modified.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

from build_asllex_targets import GROUPS, LABEL_SHA, LEX_SHA, TRAIN_SHA
from evaluate_pose_cache import NUM_CLASSES, OFFICIAL_LABEL_MAP_SHA256, canonical_hash

WORKSPACE = Path(__file__).resolve().parents[1]
OFFICIAL_CHECKPOINT_SHA256 = "5511f1270bd252c586705da8233f7ca1caf0b079a8ddf498b5bb0606f5e6d5c0"
ATTRIBUTE_MAPPING_SHA256 = "2a125875aa4bdcf0f48e618340a49b85215ddaad45b21cfec2f09c7414819ecd"
ATTRIBUTE_METADATA_SHA256 = "ce12ddb4fbc70acbf027e753b6b3aaaf2f0510ff7fcb67a7cef897829e182ad7"
SOURCE_SHA256 = {"lex_csv": LEX_SHA, "train_csv": TRAIN_SHA, "id_to_label": LABEL_SHA}
MAPPING_POLICY = "asllex-exact-code-declared-morpheme-union-v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plain_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _coefficient(value) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError("attribute_lambda must be finite and nonnegative")
    return float(value)


def _official_labels(bundle) -> None:
    if (bundle.metadata.get("label_map_sha256") != OFFICIAL_LABEL_MAP_SHA256
            or set(bundle.id_to_label) != set(range(NUM_CLASSES))
            or any(type(i) is not int or not isinstance(s, str) for i, s in bundle.id_to_label.items())
            or len(set(bundle.id_to_label.values())) != NUM_CLASSES
            or bundle.label_to_id != {label: i for i, label in bundle.id_to_label.items()}):
        raise ValueError("Method runtime requires the official 2,731-class label maps")


def _base_identity(bundle) -> dict:
    _official_labels(bundle)
    if (bundle.metadata.get("checkpoint_sha256") != OFFICIAL_CHECKPOINT_SHA256
            or bundle.checkpoint.get("num_classes") != NUM_CLASSES):
        raise ValueError("Construct the base with the strictly loaded pinned official checkpoint")
    from _poet_runtime import plain_metadata
    return plain_metadata({key: bundle.metadata[key] for key in (
        "checkpoint_sha256", "label_map_sha256", "num_classes", "num_parameters",
        "vendor_commit", "vendor_sha256", "data_preprocessing", "model_parameters", "hidden_dropout_prob")})


def _validate_target_document(document: dict, id_to_label: dict[int, str]) -> None:
    """Check semantic layout even after the immutable artifact hash check."""
    if (document.get("schema_version") != 1 or document.get("policy") != MAPPING_POLICY
            or document.get("fit_split") != "train"):
        raise ValueError("Attribute target schema/policy must use the audited train-only mapping")
    dims, vocabularies = document.get("head_dimensions"), document.get("vocabularies")
    if not isinstance(dims, dict) or set(dims) != set(GROUPS) or not isinstance(vocabularies, dict) or set(vocabularies) != set(GROUPS):
        raise ValueError("Exactly the audited handshape/location/movement heads are required")
    for name, dim in dims.items():
        vocab = vocabularies[name]
        if (type(dim) is not int or dim < 1 or not isinstance(vocab, list) or len(vocab) != dim
                or any(not isinstance(v, str) or not v for v in vocab)
                or vocab != sorted(set(vocab))):
            raise ValueError(f"{name}: dimension or vocabulary order is invalid")
    records = document.get("records")
    if not isinstance(records, list) or len(records) != len(id_to_label):
        raise ValueError("Attribute target count differs from the official labels")
    counts = 0
    for class_id, record in enumerate(records):
        if (record.get("class_id") != class_id or type(record.get("class_id")) is not int
                or record.get("gloss") != id_to_label[class_id]):
            raise ValueError("Attribute records must follow official class ID/gloss order")
        count = record.get("train_video_count")
        if type(count) is not int or count < 1:
            raise ValueError("Attribute train counts must be positive integers")
        counts += count
        if (set(record.get("targets", {})) != set(dims)
                or set(record.get("masks", {})) != set(dims)):
            raise ValueError("Attribute targets/masks have inconsistent head names")
        for name, dim in dims.items():
            target, mask = record["targets"][name], record["masks"][name]
            if (not isinstance(target, list) or not isinstance(mask, list)
                    or len(target) != dim or len(mask) != dim
                    or any(type(v) is not int or v not in (0, 1) for v in target)
                    or any(type(v) is not bool for v in mask)
                    or len(set(mask)) != 1 or (not mask[0] and any(target))):
                raise ValueError(f"{name}: invalid binary targets or group masks")
    if counts != 40154:
        raise ValueError("Attribute statistics must cover the full official training split")
    sources = document.get("sources", {})
    if set(sources) != set(SOURCE_SHA256) or any(sources[name].get("sha256") != digest for name, digest in SOURCE_SHA256.items()):
        raise ValueError("Attribute source hashes differ from the audited public sources")
    if document.get("generator_sha256") != _sha(WORKSPACE / "scripts/build_asllex_targets.py"):
        raise ValueError("Attribute mapping generator differs from its current recorded source")


@dataclass
class AttributeTargets:
    """Training-only class tables, materialized once per device; not a module."""
    head_dimensions: dict
    metadata: dict
    _targets: dict = field(repr=False)
    _masks: dict = field(repr=False)
    _device_cache: dict = field(default_factory=dict, repr=False)

    def batch(self, labels, device=None) -> tuple[dict, dict]:
        import torch
        if labels.ndim != 1 or labels.dtype != torch.long or labels.numel() < 1:
            raise ValueError("Attribute lookup requires nonempty one-dimensional int64 class IDs")
        if bool(((labels < 0) | (labels >= NUM_CLASSES)).any()):
            raise ValueError("Attribute lookup class ID is outside the official vocabulary")
        target_device = torch.device(device) if device is not None else labels.device
        key = str(target_device)
        if key not in self._device_cache:
            self._device_cache[key] = (
                {name: torch.tensor(value, dtype=torch.float32, device=target_device) for name, value in self._targets.items()},
                {name: torch.tensor(value, dtype=torch.bool, device=target_device) for name, value in self._masks.items()})
        targets, masks = self._device_cache[key]
        ids = labels.to(target_device)
        return ({name: value.index_select(0, ids) for name, value in targets.items()},
                {name: value.index_select(0, ids) for name, value in masks.items()})


def load_attribute_targets(path: Path, bundle) -> AttributeTargets:
    """Load exactly the audited artifact; paths embedded at generation are inert."""
    _official_labels(bundle)
    path = Path(path)
    payload = path.read_bytes()
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != ATTRIBUTE_MAPPING_SHA256:
        raise ValueError("Attribute mapping SHA256 differs from the current audited mapping")
    document = json.loads(payload)
    _validate_target_document(document, bundle.id_to_label)
    dims = document["head_dimensions"]
    metadata = {"mapping_sha256": actual_sha, "mapping_policy": document["policy"], "fit_split": "train",
                "target_scope": document["target_scope"], "head_dimensions": dims,
                "vocabularies": document["vocabularies"], "source_sha256": SOURCE_SHA256,
                "generator_sha256": document["generator_sha256"], "num_classes": len(document["records"]),
                "missing_policy": document["missing_policy"],
                "loss": "fixed mean over three heads; each is unweighted BCE over valid elements; empty head is zero",
                "inference": "word logits only; no target file or true labels are read"}
    return AttributeTargets(copy.deepcopy(dims), _plain_copy(metadata),
                            {name: [row["targets"][name] for row in document["records"]] for name in dims},
                            {name: [row["masks"][name] for row in document["records"]] for name in dims})


def _validate_attribute_metadata(metadata: dict) -> None:
    if (not isinstance(metadata, dict) or metadata.get("mapping_sha256") != ATTRIBUTE_MAPPING_SHA256
            or metadata.get("mapping_policy") != MAPPING_POLICY or metadata.get("fit_split") != "train"
            or metadata.get("source_sha256") != SOURCE_SHA256 or metadata.get("num_classes") != NUM_CLASSES
            or metadata.get("generator_sha256") != _sha(WORKSPACE / "scripts/build_asllex_targets.py")):
        raise ValueError("Checkpoint attribute provenance differs from the audited mapping")
    dims, vocabularies = metadata.get("head_dimensions", {}), metadata.get("vocabularies", {})
    if set(dims) != set(GROUPS) or set(vocabularies) != set(GROUPS):
        raise ValueError("Checkpoint attribute dimensions/vocabularies are incomplete")
    for name, dim in dims.items():
        values = vocabularies[name]
        if (type(dim) is not int or dim < 1 or not isinstance(values, list) or len(values) != dim
                or any(not isinstance(value, str) or not value for value in values)
                or values != sorted(set(values))):
            raise ValueError("Checkpoint attribute vocabulary ordering/dimensions are invalid")
    if canonical_hash(metadata) != ATTRIBUTE_METADATA_SHA256:
        raise ValueError("Checkpoint embedded attribute metadata differs from the audited mapping metadata")


def _construct(bundle, method: str, *, attribute_metadata: dict | None, geometry_hidden: int,
               head_seed: int, attribute_lambda: float):
    from poet_methods import PoETMethod
    if isinstance(bundle.model, PoETMethod):
        raise ValueError("Method factory requires a fresh official base, not an already wrapped method")
    identity = _base_identity(bundle)
    if type(head_seed) is not int or not 0 <= head_seed < 2 ** 32:
        raise ValueError("head_seed must be an unsigned 32-bit integer")
    coefficient = _coefficient(attribute_lambda)
    if method == "attributes":
        _validate_attribute_metadata(attribute_metadata)
    elif attribute_metadata is not None or coefficient != 0:
        raise ValueError("Auxiliary metadata/lambda are only valid for the attributes method")
    dims = attribute_metadata["head_dimensions"] if attribute_metadata is not None else None
    model = PoETMethod(bundle.model, method, attribute_dims=dims, geometry_hidden=geometry_hidden, head_seed=head_seed)
    protocol = model.protocol()
    metadata = _plain_copy({"schema_version": 1, "method": method, "base_identity": identity,
                           "architecture": protocol, "geometry_spec_sha256": canonical_hash(protocol["geometry"]["features"]) if method == "geometry" else None,
                           "attribute_mapping": attribute_metadata, "attribute_lambda": coefficient})
    model.method_metadata = copy.deepcopy(metadata)
    return model, metadata


def create_method(bundle, method: str, *, attribute_targets: AttributeTargets | None = None,
                  geometry_hidden: int = 32, head_seed: int = 20260922, attribute_lambda: float = 0.0):
    """Wrap a strictly loaded official base and return (model, checkpoint metadata)."""
    if method == "attributes" and attribute_targets is None:
        raise ValueError("Initial attributes construction requires audited training targets")
    if method != "attributes" and attribute_targets is not None:
        raise ValueError("Only attributes construction may receive training targets")
    return _construct(bundle, method, attribute_metadata=attribute_targets.metadata if attribute_targets else None,
                      geometry_hidden=geometry_hidden, head_seed=head_seed, attribute_lambda=attribute_lambda)


def restore_method(bundle, checkpoint: dict, *, expected_metadata: dict | None = None):
    """Strictly reconstruct a wrapped checkpoint, without loading auxiliary labels.

    Construct/load the optimizer after this function, then restore global RNG
    for exact resume. New heads use a private CPU RNG scope during construction.
    """
    import torch
    metadata = checkpoint.get("method_metadata")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Checkpoint is missing supported self-contained method_metadata")
    if expected_metadata is not None and metadata != expected_metadata:
        raise ValueError("Resume method metadata differs from the requested method/provenance")
    protocol_method = checkpoint.get("protocol", {}).get("method")
    if protocol_method is not None:
        # Protocols may embed the complete method metadata, or its method name.
        recorded_name = protocol_method.get("method") if isinstance(protocol_method, dict) else protocol_method
        if recorded_name != metadata.get("method"):
            raise ValueError("Checkpoint protocol method differs from its model method")
    architecture = metadata.get("architecture", {})
    model, actual = _construct(bundle, metadata.get("method"), attribute_metadata=metadata.get("attribute_mapping"),
                               geometry_hidden=architecture.get("geometry", {}).get("hidden", 32) if architecture.get("geometry") else 32,
                               head_seed=architecture.get("head_seed"), attribute_lambda=metadata.get("attribute_lambda"))
    if actual != metadata:
        raise ValueError("Checkpoint structure/configuration differs from the current method implementation")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or any(not torch.is_tensor(value) or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Checkpoint model_state_dict must contain only finite tensors")
    expected_state = model.state_dict()
    for name, value in state.items():
        if name in expected_state and value.dtype != expected_state[name].dtype:
            raise ValueError(f"Checkpoint tensor dtype differs: {name}")
    model.load_state_dict(state, strict=True)
    return model, actual


def compute_training_loss(model, inputs, labels, class_weights, *, geometry=None,
                          attribute_targets: AttributeTargets | None = None, attribute_lambda: float = 0.0) -> dict:
    """Compute word CE and optional attached auxiliary BCE in one encoder pass."""
    import torch
    from torch.nn import functional as F
    from poet_methods import PoETMethod, masked_attribute_bce
    if not isinstance(model, PoETMethod):
        raise ValueError("Training loss requires a wrapped PoET method")
    coefficient = _coefficient(attribute_lambda)
    if labels.ndim != 1 or labels.dtype != torch.long or len(labels) != len(inputs) or not len(labels):
        raise ValueError("Word targets must be one int64 class ID per input")
    valid_elements, valid_examples = {}, {}
    if model.method == "attributes":
        if geometry is not None or attribute_targets is None:
            raise ValueError("Attributes training requires its target table and no geometry input")
        if (model.attribute_dims != attribute_targets.head_dimensions
                or getattr(model, "method_metadata", {}).get("attribute_mapping") != attribute_targets.metadata):
            raise ValueError("Training target mapping differs from the model provenance")
        if coefficient != model.method_metadata["attribute_lambda"]:
            raise ValueError("Loss coefficient differs from the saved method configuration")
        logits, auxiliary_logits = model.forward_with_attributes(inputs)
        targets, masks = attribute_targets.batch(labels, logits.device)
        auxiliary = masked_attribute_bce(auxiliary_logits, targets, masks)
        valid_elements = {name: int(mask.sum().item()) for name, mask in masks.items()}
        valid_examples = {name: int(mask.any(dim=1).sum().item()) for name, mask in masks.items()}
    else:
        if attribute_targets is not None or coefficient != 0:
            raise ValueError("Auxiliary targets/lambda are only valid for attributes training")
        logits = model(inputs, geometry=geometry)
        auxiliary = logits.new_zeros(())
    if (class_weights.ndim != 1 or class_weights.numel() != logits.shape[1]
            or not bool(torch.isfinite(class_weights).all()) or bool((class_weights < 0).any())
            or bool(((labels < 0) | (labels >= logits.shape[1])).any())
            or not bool(class_weights[labels].sum() > 0)):
        raise ValueError("Word CE needs finite nonnegative class weights with positive selected denominator")
    word_ce = F.cross_entropy(logits, labels, weight=class_weights)
    # At lambda=0 preserve the baseline loss graph exactly; attribute heads do
    # not gain optimizer updates through a zero gradient plus weight decay.
    loss = word_ce if coefficient == 0 else word_ce + coefficient * auxiliary
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(auxiliary)):
        raise ValueError("Method training loss contains nonfinite values")
    return {"logits": logits, "loss": loss, "word_ce": word_ce, "auxiliary": auxiliary,
            "valid_elements": valid_elements, "valid_examples": valid_examples}
