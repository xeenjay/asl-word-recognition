"""Isolated cross-signer sampling, PoET projection and supervised contrastive loss.

No validation labels enter sampling or training. This module never reads data or
starts an experiment on import. Original vendor/model/feature contracts remain
unchanged; balanced CE and SupCon share exactly the same sampling algorithm.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import random

from finetune_baseline import stable_seed

PROTOCOL_VERSION = "poet-cross-signer-supcon-fp32-v1"
SAMPLER_VERSION = "class-signer-clip-cycles-private-python-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class CrossSignerBatchSampler:
    """Epoch-addressable shuffled class, signer and clip cycles.

    Return sample *positions* rather than external video indices. Draws are with
    repeated cycles across batches, so an epoch is a fixed budget, not a full pass.
    Repeats of a video within one epoch deliberately reuse its temporal view,
    matching the established (run_seed, epoch, video_sha256) augmentation rule.
    """
    def __init__(self, samples, label_to_id, *, seed, classes_per_batch=32,
                 signers_per_class=4, steps_per_epoch=314):
        for name, value in (("seed", seed), ("classes_per_batch", classes_per_batch),
                            ("signers_per_class", signers_per_class), ("steps_per_epoch", steps_per_epoch)):
            if type(value) is not int or (value < 0 if name == "seed" else value < 1):
                raise ValueError(f"{name} must be an integer in range")
        if seed > 2 ** 32 - 1 or classes_per_batch < 2 or signers_per_class < 2:
            raise ValueError("Need a uint32 seed, at least two classes and two signers per class")
        if not samples:
            raise ValueError("A nonempty training cohort is required")
        self.seed, self.classes_per_batch = seed, classes_per_batch
        self.signers_per_class, self.steps_per_epoch = signers_per_class, steps_per_epoch
        self.samples = samples
        self.grouped = defaultdict(lambda: defaultdict(list))
        identities = []
        for position, item in enumerate(samples):
            if item.get("split") != "train":
                raise ValueError("Sampler accepts training metadata only; validation/test are forbidden")
            signer = item.get("signer_id")
            if not isinstance(signer, str) or not signer:
                raise ValueError("Every training clip needs a nonempty signer_id")
            label = label_to_id.get(item.get("gloss"))
            if type(label) is not int or label < 0:
                raise ValueError("Unknown training gloss")
            self.grouped[label][signer].append(position)
            identities.append([item["index"], item["video_sha256"], item["gloss"], signer])
        if len({item[1] for item in identities}) != len(identities):
            raise ValueError("Duplicate video identity in sampler cohort")
        if (any(type(item[0]) is not int or item[0] < 0 for item in identities)
                or len({item[0] for item in identities}) != len(identities)):
            raise ValueError("Sampler indices must be nonnegative unique integers")
        self.classes = sorted(self.grouped)
        if len(self.classes) < classes_per_batch:
            raise ValueError("Fewer observed classes than classes_per_batch")
        if any(len(signers) < signers_per_class for signers in self.grouped.values()):
            raise ValueError("Every observed class must contain enough distinct signers")
        self.cohort_sha256 = _digest(identities)
        self.signer_names = sorted({item[3] for item in identities})
        self.signer_to_id = {name: index for index, name in enumerate(self.signer_names)}

    @property
    def batch_size(self):
        return self.classes_per_batch * self.signers_per_class

    def protocol(self):
        return {"version": SAMPLER_VERSION, "seed": self.seed,
                "classes_per_batch": self.classes_per_batch, "signers_per_class": self.signers_per_class,
                "steps_per_epoch": self.steps_per_epoch, "batch_size": self.batch_size,
                "draws_per_epoch": self.batch_size * self.steps_per_epoch,
                "cohort_sha256": self.cohort_sha256, "training_videos": len(self.samples),
                "observed_classes": len(self.classes), "signer_names": self.signer_names,
                "class_selection": "shuffle all classes, exhaust before refill; defer within-batch duplicates",
                "signer_selection": "per-class shuffled signer cycles; exhaust before refill, defer within-batch duplicates",
                "clip_selection": "per-class/signer shuffled clip cycles; exhaust before refill",
                "epoch_semantics": "fixed draws with repetitions; not a complete traversal",
                "rng": "private random.Random(SHA256([cross-signer-sampler,seed,epoch,empty]) first uint32)"}

    def batches(self, epoch):
        if type(epoch) is not int or epoch < 1:
            raise ValueError("Training sampler epoch must be a positive integer")
        rng = random.Random(stable_seed(self.seed, epoch, "", purpose="cross-signer-sampler"))
        pending, result = [], []
        signer_pending, clip_pending = {}, {}

        def take(pool, candidates, excluded):
            if not pool:
                pool.extend(candidates)
                rng.shuffle(pool)
            # A single batch asks for at most one full distinct cycle; any
            # excluded items at a refill boundary remain for subsequent draws.
            position = next((i for i, candidate in enumerate(pool) if candidate not in excluded), None)
            if position is None:
                raise RuntimeError("Cycle cannot supply a distinct item without dropping candidates")
            return pool.pop(position)

        for _ in range(self.steps_per_epoch):
            selected, batch = [], []
            for _ in range(self.classes_per_batch):
                label = take(pending, self.classes, selected)
                selected.append(label)
                selected_signers = []
                signer_pool = signer_pending.setdefault(label, [])
                for _ in range(self.signers_per_class):
                    signer = take(signer_pool, sorted(self.grouped[label]), selected_signers)
                    selected_signers.append(signer)
                    clip_pool = clip_pending.setdefault((label, signer), [])
                    batch.append(take(clip_pool, self.grouped[label][signer], ()))
            result.append(batch)
        return result

    def epoch_summary(self, epoch):
        batches = self.batches(epoch)
        positions = [position for batch in batches for position in batch]
        counts = Counter(positions)
        class_counts = Counter(self.samples[position]["gloss"] for position in positions)
        signer_counts = Counter(self.samples[position]["signer_id"] for position in positions)
        return {"epoch": epoch, "batches": len(batches), "draws": len(positions),
                "unique_videos": len(counts), "repeated_draws": len(positions) - len(counts),
                "unseen_videos": len(self.samples) - len(counts),
                "max_draws_per_video": max(counts.values()), "unique_classes": len(class_counts),
                "class_draw_counts": dict(sorted(class_counts.items())),
                "signer_draw_counts": dict(sorted(signer_counts.items())),
                "order_sha256": _digest(batches),
                "coverage_note": "Three shuffled cycle pools; fixed draws can repeat clips and need not cover every training video."}

    def state_dict(self, completed_epoch):
        if type(completed_epoch) is not int or completed_epoch < 0:
            raise ValueError("Invalid completed epoch")
        return {"version": SAMPLER_VERSION, "protocol_sha256": _digest(self.protocol()),
                "completed_epoch": completed_epoch, "next_epoch": completed_epoch + 1,
                "next_batches_sha256": _digest(self.batches(completed_epoch + 1))}


def validate_sampler_state(state, sampler, completed_epoch):
    if state != sampler.state_dict(completed_epoch):
        raise ValueError("Sampler state/config/cohort/next-epoch order differs")


def cross_signer_masks(labels, signers):
    import torch
    if (labels.ndim != 1 or signers.shape != labels.shape or labels.dtype != torch.long
            or signers.dtype != torch.long or labels.device != signers.device or labels.numel() < 2):
        raise ValueError("Labels and signer IDs must be matching nonempty int64 vectors on one device")
    if bool((labels < 0).any()) or bool((signers < 0).any()):
        raise ValueError("Negative labels/signer IDs are invalid")
    same_class = labels[:, None].eq(labels[None, :])
    same_signer = signers[:, None].eq(signers[None, :])
    identity = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = same_class & ~same_signer & ~identity
    denominator = ~identity & ~(same_class & same_signer)
    return positives, denominator


def cross_signer_supcon_loss(features, labels, signers, temperature=0.1):
    import torch
    from torch.nn import functional as F
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if (features.ndim != 2 or features.shape[0] != labels.numel() or features.shape[1] < 1
            or not features.is_floating_point() or features.device != labels.device
            or not bool(torch.isfinite(features).all())):
        raise ValueError("Projected features must be a finite floating matrix aligned with labels")
    positives, denominator = cross_signer_masks(labels, signers)
    counts = positives.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Every anchor requires a same-class different-signer positive")
    if not bool((~labels[:, None].eq(labels[None, :])).any(dim=1).all()):
        raise ValueError("Every anchor requires a negative class")
    normalized = F.normalize(features, p=2, dim=1, eps=1e-12)
    logits = normalized @ normalized.T / float(temperature)
    log_normalizer = torch.logsumexp(logits.masked_fill(~denominator, -torch.inf), dim=1)
    log_prob = logits - log_normalizer[:, None]
    loss = -(log_prob.masked_fill(~positives, 0).sum(dim=1) / counts).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite contrastive loss")
    return loss


# Keep the pure sampler usable on a lightweight local mirror without PyTorch.
def __getattr__(name):
    if name != "ContrastivePoET":
        raise AttributeError(name)
    import torch
    from torch import nn
    from poet_methods import PoETMethod

    class ContrastivePoET(PoETMethod):
        def __init__(self, base, *, head_seed=20260926):
            if type(head_seed) is not int or not 0 <= head_seed <= 2 ** 32 - 1:
                raise ValueError("head_seed must be uint32")
            super().__init__(base, "baseline", head_seed=head_seed)
            if base.hidden_size != 128:
                raise ValueError("Pinned CLS dimension must be 128")
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(head_seed)
                self.projection = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
            self.projection.to(next(base.parameters()).device)
            self.train(base.training)

        def protocol(self):
            return {"schema_version": 1, "method": "cross_signer_projection", "head_seed": self.head_seed,
                    "base_forward": "unchanged PoETMethod baseline encode and official word classifier",
                    "projection": {"dimensions": [128, 128, 64], "activation": "ReLU", "dropout": None,
                                   "normalization": "L2 in SupCon loss", "inference": "unused"},
                    "added_parameters": sum(p.numel() for p in self.projection.parameters()),
                    "total_parameters": sum(p.numel() for p in self.parameters())}

        def forward_with_projection(self, x, device=None):
            features = self.encode(x, device)
            return self.base.mlp_classifier(features), self.projection(features)

    globals()[name] = ContrastivePoET
    return ContrastivePoET


def compute_contrastive_loss(model, batch, labels, signers, class_weights, *, supcon_lambda, temperature):
    import torch
    from torch.nn import functional as F
    if type(supcon_lambda) not in (int, float) or not math.isfinite(supcon_lambda) or supcon_lambda < 0:
        raise ValueError("supcon_lambda must be finite nonnegative")
    if supcon_lambda:
        logits, features = model.forward_with_projection(batch)
        auxiliary = cross_signer_supcon_loss(features, labels, signers, temperature)
    else:
        logits = model(batch)
        auxiliary = logits.new_zeros(())
    word_ce = F.cross_entropy(logits, labels, weight=class_weights)
    loss = word_ce + supcon_lambda * auxiliary
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite logits or training loss")
    return {"logits": logits, "word_ce": word_ce, "auxiliary": auxiliary, "loss": loss}


def optimizer_protocol(model, *, lr=3e-5, projection_lr=3e-4, weight_decay=0.01, supcon_lambda=0.05):
    for name, value in (("lr", lr), ("projection_lr", projection_lr)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite positive")
    if (type(weight_decay) not in (int, float) or not math.isfinite(weight_decay) or weight_decay < 0
            or type(supcon_lambda) not in (int, float) or not math.isfinite(supcon_lambda) or supcon_lambda < 0):
        raise ValueError("Invalid weight decay or SupCon weight")
    named = list(model.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in named}) != len(named):
        raise ValueError("Shared/aliased optimizer parameters are forbidden")
    groups = {"base": [], "projection": []}
    for name, parameter in named:
        prefix = name.split(".", 1)[0]
        if prefix not in groups or not parameter.requires_grad:
            raise ValueError(f"Unexpected or frozen model parameter: {name}")
        groups[prefix].append(name)
    if not all(groups.values()):
        raise ValueError("Base and projection must both exist in both comparison arms")
    common = {"weight_decay": float(weight_decay), "betas": [0.9, 0.999], "eps": 1e-8,
              "foreach": False, "fused": False, "amsgrad": False, "maximize": False,
              "capturable": False, "differentiable": False}
    return {"name": "AdamW", "lr": float(lr), "projection_lr": float(projection_lr),
            "weight_decay": float(weight_decay), "supcon_lambda": float(supcon_lambda),
            "inactive_parameter_names": groups["projection"] if not supcon_lambda else [],
            "param_groups": [{"name": name, "param_names": names,
                              "lr": float(lr if name == "base" else projection_lr), **common}
                             for name, names in groups.items() if name == "base" or supcon_lambda]}


def create_optimizer(model, *, lr=3e-5, projection_lr=3e-4, weight_decay=0.01, supcon_lambda=0.05):
    import torch
    spec = optimizer_protocol(model, lr=lr, projection_lr=projection_lr,
                              weight_decay=weight_decay, supcon_lambda=supcon_lambda)
    parameters = dict(model.named_parameters())
    groups = []
    for recorded in spec["param_groups"]:
        group = dict(recorded)
        group["params"] = [parameters[name] for name in group["param_names"]]
        group["betas"] = tuple(group["betas"])
        groups.append(group)
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999),
                             eps=1e-8, foreach=False, fused=False)
