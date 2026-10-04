"""Versioned AdamW groups for controlled new-module learning-rate experiments.

The absent-module-lr path intentionally retains the original single group and
parameter iteration order. Explicit module_lr records complete ordered names
and uses two disjoint groups; no initializer, loss, sampler or RNG is touched.
"""
from __future__ import annotations

import math

PROTOCOL_V1 = "poet-methods-fp32-weight1-epoch-resume-v1"
PROTOCOL_V2 = "poet-methods-fp32-weight1-epoch-resume-v2"
SUPPORTED_PROTOCOLS = frozenset((PROTOCOL_V1, PROTOCOL_V2))


def _positive_lr(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def validate_module_lr(method, module_lr):
    if method not in {"baseline", "geometry", "attributes"}:
        raise ValueError("Unknown method for optimizer construction")
    if module_lr is not None:
        _positive_lr(module_lr, "module-lr")
        if method == "baseline":
            raise ValueError("module-lr is only valid for geometry or attributes")


def grouped_parameter_inventory(model):
    """Return complete ordered, disjoint named parameters; reject hidden extras."""
    method = getattr(model, "method", None)
    if method not in {"geometry", "attributes"}:
        raise ValueError("Grouped parameters require geometry or attributes")
    named = list(model.named_parameters(remove_duplicate=False))
    if not named or len({name for name, _ in named}) != len(named):
        raise ValueError("Parameter names must be nonempty and unique")
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError("Overlapping/shared parameter aliases are not allowed between named groups")
    prefix = "geometry_branch." if method == "geometry" else "attribute_heads."
    base, module = [], []
    for name, parameter in named:
        if not getattr(parameter, "requires_grad", False):
            raise ValueError(f"All protocol parameters must remain trainable: {name}")
        if name.startswith("base."):
            base.append((name, parameter))
        elif name.startswith(prefix):
            module.append((name, parameter))
        else:
            raise ValueError(f"Unassigned method parameter outside base/new-module groups: {name}")
    if not base or not module or len(base) + len(module) != len(named):
        raise ValueError("Base and new-module groups must both be nonempty and cover all parameters")
    return [("base", base), ("new-module", module)]


def optimizer_protocol(model, *, lr, weight_decay, module_lr=None):
    """JSON-compatible optimizer specification; v1 keeps its existing schema."""
    validate_module_lr(getattr(model, "method", None), module_lr)
    lr = _positive_lr(lr, "base lr")
    if type(weight_decay) not in (int, float) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    result = {"name": "AdamW", "lr": lr, "weight_decay": float(weight_decay),
              "betas": [0.9, 0.999], "eps": 1e-8, "foreach": False, "fused": False}
    if module_lr is None:
        return result
    result["grouping"] = "complete-ordered-base-and-new-module-v1"
    result["module_lr"] = float(module_lr)
    settings = {key: value for key, value in result.items()
                if key in {"weight_decay", "betas", "eps", "foreach", "fused"}}
    settings.update(amsgrad=False, maximize=False, capturable=False, differentiable=False)
    result["param_groups"] = [{"name": name, "param_names": [key for key, _ in parameters],
                               "lr": lr if name == "base" else float(module_lr), **settings}
                              for name, parameters in grouped_parameter_inventory(model)]
    return result


def create_optimizer(model, *, lr, weight_decay, module_lr=None):
    import torch
    specification = optimizer_protocol(model, lr=lr, weight_decay=weight_decay, module_lr=module_lr)
    common = {"lr": lr, "weight_decay": weight_decay, "betas": (0.9, 0.999),
              "eps": 1e-8, "foreach": False, "fused": False}
    if module_lr is None:
        # Exact legacy constructor: no new group keys, no parameter reordering.
        return torch.optim.AdamW(model.parameters(), **common)
    inventory = grouped_parameter_inventory(model)
    groups = []
    for (name, parameters), recorded in zip(inventory, specification["param_groups"]):
        group = dict(recorded)
        group["params"] = [parameter for _, parameter in parameters]
        group["betas"] = tuple(group["betas"])
        groups.append(group)
    return torch.optim.AdamW(groups, **common)


def gradient_group_metrics(optimizer, squared_norm_by_parameter=None):
    """Read detached gradients only; never alter optimizer state or consume RNG.

    Training supplies its existing per-parameter squared norms to avoid doing a
    second GPU reduction. Standalone diagnostics may calculate those norms here.
    """
    result = []
    for i, group in enumerate(optimizer.param_groups):
        total, active = 0.0, 0
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if squared_norm_by_parameter is None:
                value = float(parameter.grad.detach().double().square().sum().item())
            else:
                if id(parameter) not in squared_norm_by_parameter:
                    raise ValueError("Missing recorded norm for a parameter with a gradient")
                value = squared_norm_by_parameter[id(parameter)]
            if not math.isfinite(value) or value < 0:
                raise ValueError("Gradient norm must be finite and nonnegative")
            total += value
            active += 1
        result.append({"name": group.get("name", "all" if len(optimizer.param_groups) == 1 else f"group-{i}"),
                       "lr": float(group["lr"]), "gradient_l2": math.sqrt(total),
                       "parameters_with_grad": active})
    return result


def validate_checkpoint_optimizer_contract(checkpoint, model):
    """Validate version/config/group identities for resume and independent eval.

    This checks groups, not moment values; the trainer additionally applies its
    strict AdamW state/counter/finite checks before restoring those moment values.
    v1 checkpoints remain evaluable without requiring fields introduced in v2.
    """
    protocol = checkpoint.get("protocol", {})
    version = protocol.get("version")
    if version not in SUPPORTED_PROTOCOLS:
        raise ValueError("Unsupported method optimizer protocol version")
    training = checkpoint.get("training_parameters", {})
    if version == PROTOCOL_V1:
        if (training.get("module_lr") is not None or protocol.get("optimizer", {}).get("module_lr") is not None
                or "param_groups" in protocol.get("optimizer", {})):
            raise ValueError("Legacy v1 cannot contain the new module-lr/grouped protocol")
        return
    if "module_lr" not in training or training["module_lr"] is None:
        raise ValueError("Grouped v2 checkpoint must explicitly record module_lr")
    expected = optimizer_protocol(model, lr=training.get("lr"), weight_decay=training.get("weight_decay"),
                                  module_lr=training["module_lr"])
    if protocol.get("optimizer") != expected:
        raise ValueError("Grouped optimizer protocol disagrees with model names/order or training settings")
    optimizer = create_optimizer(model, lr=training["lr"], weight_decay=training["weight_decay"],
                                 module_lr=training["module_lr"])
    restored = checkpoint.get("optimizer_state_dict", {})
    if not isinstance(restored, dict) or restored.get("param_groups") != optimizer.state_dict()["param_groups"]:
        raise ValueError("Checkpoint optimizer groups/settings do not match the explicit v2 protocol")
