"""Small independent PoET method components, not a training/evaluation CLI.

The fixed upstream Transformer_CLS operations, including its post-embedding
padding mask and CPU FloatTensor conversion, are retained deliberately. Load
the official model strictly via _poet_runtime.load_bundle BEFORE wrapping it.
New checkpoint keys have a 'base.' prefix and are not official state dicts.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from poet_geometry import GEOMETRY_DIM, geometry_spec


class PoETMethod(nn.Module):
    """Wrap one already-loaded model as baseline, geometry, OR attributes.

    Attribute dimensions come from an audited external mapping, never assumed
    from a paper. Extra heads are initialized with a private CPU RNG scope so
    construction does not consume the baseline dropout/sampler random stream.
    """

    def __init__(self, base: nn.Module, method: str = "baseline", *,
                 attribute_dims: Mapping[str, int] | None = None,
                 geometry_hidden: int = 32, head_seed: int = 20260922):
        super().__init__()
        if method not in {"baseline", "geometry", "attributes"}:
            raise ValueError("method must be baseline, geometry, or attributes; no implicit combination")
        dims = dict(attribute_dims or {})
        if method == "attributes" and not dims:
            raise ValueError("attributes requires dimensions from an audited target mapping")
        if method != "attributes" and dims:
            raise ValueError("attribute dimensions are only valid for the attributes method")
        if any(not isinstance(name, str) or not name or "." in name
               or type(size) is not int or size < 1 for name, size in dims.items()):
            raise ValueError("attribute names must be nonempty without dots and dimensions positive integers")
        if type(geometry_hidden) is not int or geometry_hidden < 1:
            raise ValueError("geometry_hidden must be a positive integer")
        required = ("embedding", "cls_token", "pos_encoder", "transformer_encoder",
                    "mlp_classifier", "hidden_size", "max_seq_length")
        if any(not hasattr(base, name) for name in required):
            raise ValueError("Expected the pinned Transformer_CLS module interface")
        if (not isinstance(base.embedding, nn.Sequential)
                or not isinstance(base.embedding[0], nn.Linear)
                or base.embedding[0].in_features != 188):
            raise ValueError("Expected the pinned 188-feature frame embedding")
        if any(p.dtype != torch.float32 for p in base.parameters()):
            raise ValueError("This first method protocol requires FP32 base parameters")
        self.base = base
        self.method = method
        self.attribute_dims = dims
        self.head_seed = head_seed
        self.geometry_hidden = geometry_hidden
        self.geometry_branch = None
        self.attribute_heads = nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            # CPU-only seed avoids changing any CUDA RNG state.
            torch.random.default_generator.manual_seed(head_seed)
            if method == "geometry":
                self.geometry_branch = nn.Sequential(nn.Linear(GEOMETRY_DIM, geometry_hidden), nn.ReLU(),
                                                     nn.Linear(geometry_hidden, base.hidden_size))
                nn.init.zeros_(self.geometry_branch[-1].weight)
                nn.init.zeros_(self.geometry_branch[-1].bias)
            if method == "attributes":
                self.attribute_heads = nn.ModuleDict({name: nn.Linear(base.hidden_size, dims[name])
                                                      for name in sorted(dims)})
        target = next(base.parameters()).device
        if self.geometry_branch is not None:
            self.geometry_branch.to(target)
        self.attribute_heads.to(target)
        # Follow the already-loaded model's eval/train state at construction.
        self.train(base.training)

    def protocol(self) -> dict:
        return {"schema_version": 1, "method": self.method, "head_seed": self.head_seed,
                "base_forward": "pinned Transformer_CLS including CPU cast and post-embedding mask",
                "geometry": {"features": geometry_spec(), "hidden": self.geometry_hidden,
                             "fusion": "zero-final-linear residual after official frame embedding",
                             "missing_frames": "residual zero when all seven relation masks are zero"}
                if self.method == "geometry" else None,
                "attribute_dims": dict(self.attribute_dims),
                "attribute_supervision": "masked multilabel BCE; labels never used in forward" if self.method == "attributes" else None,
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "added_parameters": sum(p.numel() for p in self.parameters()) - sum(p.numel() for p in self.base.parameters())}

    def encode(self, x: torch.Tensor, device=None, *, geometry: torch.Tensor | None = None) -> torch.Tensor:
        """Return attached CLS features; main and auxiliary losses share them."""
        target = next(self.base.parameters()).device if device is None else torch.device(device)
        if x.ndim != 3 or x.shape[2] != 188 or not 0 < x.shape[1] <= self.base.max_seq_length:
            raise ValueError("Expected (batch,frames,188) within the official sequence capacity")
        if self.method == "geometry":
            if geometry is None or geometry.shape != (*x.shape[:2], GEOMETRY_DIM):
                raise ValueError("geometry requires an aligned (batch,frames,28) tensor")
        elif geometry is not None:
            raise ValueError("Only the geometry method accepts geometry features")
        # Do not optimize these vendor operations in this method experiment.
        x = x.type(torch.FloatTensor).to(target)
        batch_size = x.size(0)
        x = self.base.embedding(x)
        if self.geometry_branch is not None:
            geometry = geometry.to(device=target, dtype=torch.float32)
            if not bool(torch.isfinite(geometry).all()):
                raise ValueError("geometry contains nonfinite values")
            masks = geometry[..., 3::4]
            if not bool(((masks == 0) | (masks == 1)).all()):
                raise ValueError("geometry relation presence bits must be zero or one")
            missing_values = geometry.reshape(*geometry.shape[:2], -1, 4)[..., :3][masks == 0]
            if bool((missing_values != 0).any()):
                raise ValueError("missing geometry relations must contain zero coordinates")
            residual = self.geometry_branch(geometry)
            residual = residual * masks.any(dim=-1, keepdim=True).to(residual.dtype)
            x = x + residual
        cls_token = self.base.cls_token.expand(batch_size, -1, -1).to(target)
        x = torch.cat([cls_token, x], dim=1)
        src_key_padding_mask = torch.all(x == 0, dim=2).to(target)
        x = x * math.sqrt(self.base.hidden_size)
        x = self.base.pos_encoder(x)
        output = self.base.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return output[:, 0, :]

    def forward(self, x: torch.Tensor, device=None, *, geometry: torch.Tensor | None = None) -> torch.Tensor:
        """Label-free word prediction; attribute heads need not run at inference."""
        return self.base.mlp_classifier(self.encode(x, device, geometry=geometry))

    def forward_with_attributes(self, x: torch.Tensor, device=None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.method != "attributes":
            raise ValueError("Auxiliary prediction is only valid for the attributes method")
        cls = self.encode(x, device)
        return self.base.mlp_classifier(cls), {name: head(cls) for name, head in self.attribute_heads.items()}


def masked_attribute_bce(logits: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor],
                         masks: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Mean of head BCE means over valid elements; empty heads contribute zero.

    Targets and boolean masks must both be (batch, attribute_dim). The fixed
    head-count denominator is intentional: wholly missing batches contribute
    zero without boosting the weight of another head. No pos_weight is fitted
    here. Masked targets may be placeholders (including NaN); they are never
    sent into BCE. The caller sets and records the auxiliary coefficient.
    """
    if not logits or set(logits) != set(targets) or set(logits) != set(masks):
        raise ValueError("logits, targets, and masks must share nonempty attribute head names")
    losses = []
    for name in sorted(logits):
        prediction = logits[name]
        target, mask = targets[name], masks[name]
        if prediction.ndim != 2 or target.shape != prediction.shape or mask.shape != prediction.shape:
            raise ValueError(f"{name}: predictions, targets, and masks must all have shape (batch,dim)")
        if mask.dtype != torch.bool:
            raise ValueError(f"{name}: mask must have boolean dtype")
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError(f"{name}: predictions must be finite")
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        mask = mask.to(device=prediction.device)
        if bool(mask.any()):
            valid_targets = target[mask]
            if not bool(torch.isfinite(valid_targets).all()) or not bool(((valid_targets == 0) | (valid_targets == 1)).all()):
                raise ValueError(f"{name}: valid multilabel targets must be finite binary values")
            losses.append(F.binary_cross_entropy_with_logits(prediction[mask], valid_targets))
        else:
            losses.append(prediction.sum() * 0.0)
    return torch.stack(losses).mean()
