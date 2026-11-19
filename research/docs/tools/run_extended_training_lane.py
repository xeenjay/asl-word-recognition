"""One finite three-seed lane continuing existing epoch-10 checkpoints to 20.

The unchanged trainer restores model, optimizer, RNG, sampler and prior best.
Permanent claims forbid automatic replay. Evaluation failure never authorizes
repeating completed training. No retrieval, TTA, test, or optimizer change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from run_paired_seed_lane import (command_options, inside, now, read, save, sha,
                                  verify_predictions, PROTOCOLS)
from run_rgb_pilot import verify_run, inside as artifact_inside

ROOT = Path(__file__).resolve().parents[2]
KIND = "poet-extended-training-v1"
SEEDS = [20260922, 20260923, 20260924]
PARENT = "runs/extended-training-launches"
LEGACY_PARENT = "runs/20260925T011340Z-formal-baseline-seed20260922-679f114a"
LEGACY_BYTECODE_AUDIT = "docs/environment/extended-legacy-bytecode-audit-20260927.json"
ENVIRONMENT = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
               "NUMEXPR_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_plan(plan, lane_name, root=ROOT):
    root = Path(root).resolve()
    fixed = {"schema_version": 1, "kind": KIND, "workspace": str(root), "seeds": SEEDS,
             "initial_epochs": 10, "epochs": 20, "training_samples": 40154,
             "validation_samples": 10304, "initial_optimizer_steps": 3140,
             "optimizer_steps": 6280, "additional_optimizer_steps": 3140,
             "samples_seen": 803080, "test_used": False}
    require(all(plan.get(key) == value for key, value in fixed.items()), "Plan differs from the finite 10-to-20 budget")
    lanes = plan.get("lanes", {})
    require(set(lanes) == {"gpu0", "gpu1"} and lane_name in lanes, "Exactly two fixed method lanes are required")
    needed = {"scripts/run_logged.py", "scripts/finetune_methods.py", "scripts/evaluate_methods.py",
              "docs/tools/run_extended_training_lane.py", "tests/test_extended_training_lane.py",
              "docs/tools/run_paired_seed_lane.py", "docs/tools/run_rgb_pilot.py",
              "docs/tools/finish_extended_training.py", "tests/test_extended_training_dispatch.py",
              "docs/tools/analyze_extended_training.py", "tests/test_extended_training_analysis.py",
              "docs/tools/validate_extended_resume_gpu.py", "tests/test_extended_resume.py",
              "docs/environment/extended-training-preregistration-20260927.json",
              plan["training_manifest"], plan["validation_manifest"],
              plan["training_geometry_manifest"], plan["validation_geometry_manifest"]}
    for name, method, gpu, cpus in (("gpu0", "baseline", 0, list(range(8, 16))),
                                   ("gpu1", "geometry", 1, list(range(16, 24)))):
        lane = lanes[name]
        require(lane.get("gpu") == gpu and lane.get("cpu_affinity") == cpus and
                [job.get("seed") for job in lane.get("jobs", [])] == SEEDS, "Lane resources or seed sequence differs")
        for job in lane["jobs"]:
            seed = job["seed"]
            require(job.get("method") == method and job.get("job_id") == method + "-seed" + str(seed)
                    and job.get("protocol_version") == PROTOCOLS[method], "Job method/seed/protocol differs")
            parent = inside(job["parent_training_run"], root)
            checkpoint = inside(job["parent_checkpoint"], root)
            require(parent.parent == root / "runs" and checkpoint == parent / "training/checkpoints/last.pth",
                    "Resume must use the exact completed parent's last checkpoint")
            for field in ("parent_checkpoint_sha256", "parent_protocol_sha256"):
                value = job.get(field)
                require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                        "Missing fixed parent fingerprint")
            needed.add(str(checkpoint.relative_to(root)))
            needed.update(str((parent / leaf).relative_to(root)) for leaf in
                          ("manifest.json", "artifacts.json", "training/summary.json", "training/protocol.json",
                           "training/metrics_history.json", "training/checkpoints/index.json"))
            trainer = (str((parent / "code/scripts/finetune_methods.py").relative_to(root))
                       if method == "baseline" and seed == 20260922 else "scripts/finetune_methods.py")
            require(job.get("trainer_script") == trainer, "Only B22 may use its own immutable historical trainer")
            needed.add(trainer)
            train = command_options(job["train_command"], trainer, root)
            evaluate = command_options(job["evaluation_command"], "scripts/evaluate_methods.py", root)
            expected = {"--method": method, "--seed": str(seed), "--head-seed": str(seed), "--epochs": "20",
                        "--resume": str(checkpoint), "--checkpoint-dir": str(root / "assets/checkpoints"),
                        "--batch-size": "128", "--eval-batch-size": "128",
                        "--num-threads": "4", "--device": "cuda:0", "--output-dir": "{run_dir}/training",
                        "--train-cache-manifest": str(inside(plan["training_manifest"], root)),
                        "--val-cache-manifest": str(inside(plan["validation_manifest"], root))}
            require(all(train.get(key) == value for key, value in expected.items()), "Continuation command differs")
            require(float(train.get("--lr", "nan")) == 3e-5 and float(train.get("--weight-decay", "nan")) == .01,
                    "Original optimizer settings must remain unchanged")
            allowed = set(expected) | {"--lr", "--weight-decay"}
            expected_eval = {"--checkpoint": "{checkpoint}", "--cache-manifest": str(inside(plan["validation_manifest"], root)),
                             "--checkpoint-dir": str(root / "assets/checkpoints"),
                             "--batch-size": "128", "--num-threads": "4", "--device": "cuda:0",
                             "--output-dir": "{run_dir}/evaluation"}
            if method == "geometry":
                extra = {"--geometry-hidden": "32", "--train-geometry-manifest": str(inside(plan["training_geometry_manifest"], root)),
                         "--val-geometry-manifest": str(inside(plan["validation_geometry_manifest"], root))}
                require(all(train.get(key) == value for key, value in extra.items())
                        and float(train.get("--module-lr", "nan")) == 3e-4, "Original geometry settings differ")
                allowed |= set(extra) | {"--module-lr"}
                expected_eval["--geometry-manifest"] = str(inside(plan["validation_geometry_manifest"], root))
            require(set(train) == allowed and evaluate == expected_eval, "Unexpected training/evaluation options")
    require(needed <= plan.get("frozen_files", {}).keys(), "Required source/data/parent evidence is not frozen")
    return lanes[lane_name]


def check_frozen(plan_path, plan_sha, plan, root=ROOT):
    require(sha(plan_path) == plan_sha, "Plan changed after claim")
    for name, expected in plan["frozen_files"].items():
        require(sha(inside(name, root)) == expected, "Frozen source or prior experiment changed: " + name)


def checked_history(history, epochs):
    require([row.get("epoch") for row in history] == list(range(epochs + 1)), "History must retain every epoch including 0")
    for row in history:
        metrics = row["metrics"]
        require(metrics.get("attempted") == 10304 and metrics.get("inferred") == 10304 and metrics.get("failed") == 0
                and metrics.get("vocabulary_classes") == 2731
                and math.isfinite(metrics["per_class_full_vocabulary"]["top1"]), "Incomplete full validation epoch")
        if row["epoch"]:
            training = row.get("training", {})
            require(training.get("samples") == 40154 and training.get("optimizer_steps") == 314,
                    "Epoch training population or optimizer budget differs")
    return max(history, key=lambda row: row["metrics"]["per_class_full_vocabulary"]["top1"])


def validate_config(config, item, epochs, resume, root=ROOT):
    expected = {"method": item["method"], "seed": item["seed"], "head_seed": item["seed"], "epochs": epochs,
                "batch_size": 128, "eval_batch_size": 128, "num_threads": 4, "lr": 3e-5, "weight_decay": .01,
                "module_lr": 3e-4 if item["method"] == "geometry" else None, "engineering_smoke": False}
    require(all(config.get(key) == value for key, value in expected.items()), "Training config differs from fixed protocol")
    require(config.get("resume") is None if resume is None else
            inside(config.get("resume", ""), root) == resume, "Training resume source differs")


def verify_parent_run(run, item, plan, root=ROOT):
    """Strict old-run audit with only the twelve explicitly audited B22 caches.

    Original listed artifacts still require exact size/SHA, and the old artifact
    inventory is never rewritten. All other parents and every new run retain
    the unchanged strict verifier. The returned evidence has no fresh timestamp.
    """
    root, run = Path(root).resolve(), Path(run).resolve()
    require(run == inside(item["parent_training_run"], root), "Parent run identity differs")
    extras = item.get("parent_extra_bytecode", [])
    if not extras:
        return verify_run(run)
    require(item.get("method") == "baseline" and item.get("seed") == 20260922
            and run == root / LEGACY_PARENT and isinstance(extras, list) and len(extras) == 12,
            "Bytecode exception is limited to the twelve audited historical B22 caches")
    audit_path = root / LEGACY_BYTECODE_AUDIT
    audit_sha = sha(audit_path)
    require(plan["frozen_files"].get(LEGACY_BYTECODE_AUDIT) == audit_sha, "Bytecode audit is not frozen")
    audit = read(audit_path)
    require(audit.get("status") == "verified" and audit.get("source_compilation_exact") is True
            and audit.get("parent_training_run") == LEGACY_PARENT and audit.get("files") == extras
            and audit.get("missing_listed_artifacts") == [] and audit.get("old_files_modified") is False,
            "Explicit bytecode whitelist differs from verified source-compilation audit")
    manifest = read(run / "manifest.json")
    require(manifest.get("state") == "completed" and manifest.get("returncode") == 0
            and (run / "exit_code.txt").read_text().strip() == "0", "Parent run did not complete with exit 0")
    for name, key in (("config.json", "config_sha256"), ("code_manifest.json", "code_manifest_sha256")):
        require(sha(run / name) == manifest.get(key), "Parent logged config/source inventory changed")
    listed, total = {}, 0
    for entry in read(run / "artifacts.json")["files"]:
        name = entry["path"]
        require(name not in listed and name not in {"manifest.json", "artifacts.json"}, "Invalid/duplicate listed parent artifact")
        path = artifact_inside(name, run, relative=True)
        require(not (run / name).is_symlink() and path.is_file(), "Missing/symlinked listed parent artifact")
        require(path.stat().st_size == entry["size_bytes"] and sha(path) == entry["sha256"],
                "Listed parent artifact changed: " + name)
        listed[name] = entry
        total += entry["size_bytes"]
    observed, allowed = [], set()
    for entry in extras:
        name, source_name = entry["path"], entry["source_path"]
        relative = Path(name)
        require(name not in allowed and name not in listed and relative.parent.as_posix() in
                ("code/scripts/__pycache__", "code/vendor/poet/__pycache__")
                and relative.name.endswith(".cpython-310.pyc"), "Unexpected/duplicate historical bytecode path")
        expected_source = (relative.parent.parent / (relative.name[:-len(".cpython-310.pyc")] + ".py")).as_posix()
        require(source_name == expected_source and source_name in listed
                and listed[source_name]["sha256"] == entry["source_sha256"]
                and entry.get("compiled_code_equals_frozen_source") is True,
                "Historical bytecode source is not its original listed source")
        path = artifact_inside(name, run, relative=True)
        require(not (run / name).is_symlink() and path.is_file(), "Historical bytecode is missing/symlinked")
        stat = path.stat(); digest = sha(path)
        require(stat.st_size == entry["size_bytes"] and digest == entry["sha256"]
                and stat.st_mtime_ns == entry["mtime_ns"], "Historical bytecode bytes or timestamp changed")
        for bound, expected in ((name, digest), (source_name, entry["source_sha256"])):
            require(plan["frozen_files"].get(str((run / bound).relative_to(root))) == expected,
                    "Historical bytecode/source is not frozen")
        allowed.add(name)
        observed.append({**entry, "sha256": digest, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    actual = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()
              and str(path.relative_to(run)) not in {"manifest.json", "artifacts.json"}}
    require(actual == set(listed) | allowed, "Unlisted or missing parent artifacts outside exact bytecode whitelist")
    return {"run_manifest_sha256": sha(run / "manifest.json"), "artifacts_sha256": sha(run / "artifacts.json"),
            "verified_files": len(listed), "verified_bytes": total,
            "legacy_bytecode_audit_sha256": audit_sha, "parent_extra_bytecode": observed,
            "verified_extra_bytecode_files": len(observed),
            "verified_extra_bytecode_bytes": sum(entry["size_bytes"] for entry in observed)}


def verify_parent(item, plan, root=ROOT):
    parent = inside(item["parent_training_run"], root)
    run_proof = verify_parent_run(parent, item, plan, root)
    summary = read(parent / "training/summary.json")
    protocol_doc = read(parent / "training/protocol.json")
    protocol = protocol_doc["protocol"]
    require(summary.get("state") == "completed" and summary.get("scope") == "formal_finetuning"
            and summary.get("completed_epoch") == 10 and summary.get("global_optimizer_step") == 3140
            and summary.get("samples_seen") == 401540 and summary.get("resume") is None,
            "Parent did not complete the original 10 epochs")
    validate_config(summary["config"], item, 10, None, root)
    require(protocol.get("method") == item["method"] and protocol.get("version") == item["protocol_version"]
            and protocol.get("training", {}).get("seed") == item["seed"]
            and summary.get("protocol_sha256") == protocol_doc.get("sha256") == canonical(protocol)
            == item["parent_protocol_sha256"], "Parent protocol fingerprint differs")
    history = read(parent / "training/metrics_history.json")["epochs"]
    best = checked_history(history, 10)
    require(summary["best"] == {"epoch": best["epoch"], "score": best["metrics"]["per_class_full_vocabulary"]["top1"]},
            "Parent best must be earliest maximum macro Top-1")
    index = read(parent / "training/checkpoints/index.json")
    checkpoint = inside(item["parent_checkpoint"], root)
    require(inside(index["last"]["path"], root) == checkpoint and index["last"]["epoch"] == 10
            and index["last"]["sha256"] == item["parent_checkpoint_sha256"] == sha(checkpoint),
            "Parent last checkpoint identity/hash differs")
    old_best = inside(index["best"]["path"], root)
    require(old_best.parent == parent / "training/checkpoints" and index["best"]["epoch"] == best["epoch"]
            and sha(old_best) == index["best"]["sha256"] == sha(parent / "training/checkpoints/best.pth"),
            "Parent best checkpoint differs")
    predicted = inside(best["predictions_path"], root)
    require(predicted == parent / "training/evaluation" / ("epoch-%04d" % best["epoch"]) / "predictions.jsonl"
            and sha(predicted) == best["predictions_sha256"], "Parent best prediction source changed")
    for path in (old_best, parent / "training/checkpoints/best.pth", predicted,
                 predicted.parent / "metrics.json", predicted.parent / "per_class.csv"):
        require(plan["frozen_files"].get(str(path.relative_to(Path(root).resolve()))) == sha(path),
                "Parent best evidence is not frozen")
    return {"parent": parent, "summary": summary, "protocol": protocol, "history": history,
            "best": best, "checkpoint_index": index, "run_verification": run_proof}


def verify_training(run, item, plan, root=ROOT):
    run = Path(run).resolve()
    verify_run(run)
    parent = verify_parent(item, plan, root)
    summary = read(run / "training/summary.json")
    protocol_doc = read(run / "training/protocol.json")
    require(summary.get("state") == "completed" and summary.get("scope") == "formal_finetuning"
            and summary.get("completed_epoch") == 20 and summary.get("global_optimizer_step") == 6280
            and summary.get("samples_seen") == 803080, "Continued training counters differ")
    checkpoint = inside(item["parent_checkpoint"], root)
    validate_config(summary["config"], item, 20, checkpoint, root)
    require(protocol_doc["protocol"] == parent["protocol"] and protocol_doc.get("sha256") == summary.get("protocol_sha256")
            == item["parent_protocol_sha256"], "Continuation changed the original training protocol")
    resume = summary.get("resume", {})
    require(resume.get("checkpoint") == str(checkpoint) and resume.get("sha256") == item["parent_checkpoint_sha256"]
            and resume.get("parent_output") == str(parent["parent"] / "training") and resume.get("completed_epoch") == 10,
            "Exact epoch-10 resume evidence is missing")
    history = read(run / "training/metrics_history.json")["epochs"]
    best = checked_history(history, 20)
    require(history[:11] == parent["history"], "Inherited epoch 0..10 history changed")
    require(summary["best"] == {"epoch": best["epoch"], "score": best["metrics"]["per_class_full_vocabulary"]["top1"]},
            "Best selection differs from earliest macro maximum over epochs 0..20")
    index = read(run / "training/checkpoints/index.json")
    for name in ("last", "best"):
        path = inside(index[name]["path"], root)
        require(path.parent == run / "training/checkpoints" and sha(path) == index[name]["sha256"],
                "Continued checkpoint path/hash differs")
    require(index["last"]["epoch"] == 20 and index["best"]["epoch"] == best["epoch"]
            and sha(run / "training/checkpoints/best.pth") == index["best"]["sha256"], "Continued checkpoint epoch differs")
    owner = parent["parent"] if best["epoch"] <= 10 else run
    predicted = inside(best["predictions_path"], root)
    require(predicted == owner / "training/evaluation" / ("epoch-%04d" % best["epoch"]) / "predictions.jsonl"
            and sha(predicted) == best["predictions_sha256"], "Selected predictions must come from their original epoch owner")
    if best["epoch"] <= 10:
        require(index["best"]["sha256"] == parent["checkpoint_index"]["best"]["sha256"],
                "Retained old best was not copied byte-for-byte")
    return {"summary_sha256": sha(run / "training/summary.json"), "protocol_sha256": summary["protocol_sha256"],
            "parent_training_run": str(parent["parent"]), "parent_checkpoint": str(checkpoint),
            "parent_checkpoint_sha256": item["parent_checkpoint_sha256"], "initial_epochs": 10,
            "parent_run_verification": parent["run_verification"],
            "additional_optimizer_steps": 3140, "best_checkpoint_sha256": index["best"]["sha256"],
            "selected_epoch": best["epoch"], "best_metrics": best["metrics"],
            "best_predictions_path": str(predicted), "best_predictions_sha256": best["predictions_sha256"],
            "retained_parent_best": best["epoch"] <= 10}


def verify_evaluation(training, evaluation, item, evidence, plan):
    verify_run(Path(evaluation))
    target = Path(evaluation) / "evaluation"
    summary = read(target / "summary.json")
    require(summary.get("status") == "completed" and summary.get("scope") == "official_validation"
            and summary.get("method") == item["method"] and summary.get("attribute_targets_read") is False
            and summary.get("checkpoint_epoch") == evidence["selected_epoch"]
            and summary.get("checkpoint_sha256") == evidence["best_checkpoint_sha256"]
            and summary.get("protocol_sha256") == evidence["protocol_sha256"]
            and summary.get("metrics") == evidence["best_metrics"]
            and read(target / "metrics.json") == summary["metrics"], "Independent best evaluation differs")
    require(sha(Path(training) / "training/checkpoints/best.pth") == evidence["best_checkpoint_sha256"],
            "Copied/current best checkpoint changed during evaluation")
    original = Path(evidence["best_predictions_path"])
    require(sha(original) == evidence["best_predictions_sha256"], "Selected epoch predictions changed")
    count = verify_predictions(original, target / "predictions.jsonl", plan["validation_samples"])
    require(read(original.parent / "metrics.json")["metrics"] == summary["metrics"]
            and (original.parent / "per_class.csv").read_bytes() == (target / "per_class.csv").read_bytes(),
            "Selected epoch metrics or per-class table differs")
    return {"verified_predictions": count, "summary_sha256": sha(target / "summary.json"),
            "predictions_sha256": sha(target / "predictions.jsonl"), "retained_parent_best": evidence["retained_parent_best"]}


def resources(lane, root=ROOT):
    free = int(subprocess.check_output(["nvidia-smi", "-i", str(lane["gpu"]),
               "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    mem = {row.split(":", 1)[0]: row.split(":", 1)[1].strip() for row in Path("/proc/meminfo").read_text().splitlines()}
    snapshot = {"checked_at_utc": now(), "gpu_free_mib": free, "disk_free_bytes": shutil.disk_usage(root).free,
                "ram_available_bytes": int(mem["MemAvailable"].split()[0]) * 1024,
                "cpu_affinity": sorted(os.sched_getaffinity(0))}
    require(free >= 8192 and snapshot["disk_free_bytes"] >= 8 * 1024 ** 3
            and snapshot["ram_available_bytes"] >= 16 * 1024 ** 3
            and snapshot["cpu_affinity"] == lane["cpu_affinity"], "Assigned resource minimum is unavailable; no automatic retry")
    return snapshot


def claim(plan_path, plan_sha, lane_name, root=ROOT):
    destination = Path(plan_path).parent / lane_name
    destination.mkdir(exist_ok=True)
    value = {"pid": os.getpid(), "created_at_utc": now(), "plan_sha256": plan_sha,
             "plan_path": str(plan_path), "lane": lane_name, "automatic_replay": False}
    for path in (Path(root) / PARENT / ("epochs20-" + lane_name + "-budget.json"), destination / "claim.json"):
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)
    return destination


def run_lane(plan_path, lane_name, root=ROOT):
    root = Path(root).resolve()
    plan_path = inside(plan_path, root)
    require(plan_path.name == "plan.json" and plan_path.parent.parent == root / PARENT, "Use the unique extended-training plan directory")
    payload = plan_path.read_bytes(); plan_sha = hashlib.sha256(payload).hexdigest(); plan = json.loads(payload)
    lane = validate_plan(plan, lane_name, root)
    check_frozen(plan_path, plan_sha, plan, root)
    for item in lane["jobs"]:
        verify_parent(item, plan, root)
    destination = claim(plan_path, plan_sha, lane_name, root)
    state_path = destination / "state.json"
    state = {"schema_version": 1, "kind": KIND, "state": "running", "stage": "preflight", "pid": os.getpid(),
             "started_at_utc": now(), "plan_sha256": plan_sha, "lane": lane_name, "test_used": False,
             "jobs": [{"job_id": item["job_id"], "method": item["method"], "seed": item["seed"], "state": "queued",
                       "training": {"state": "queued"}, "evaluation": {"state": "queued"}} for item in lane["jobs"]]}
    save(state_path, state)
    current_job = None

    def launch(item, job, stage, checkpoint=None):
        check_frozen(plan_path, plan_sha, plan, root)
        snapshot = resources(lane, root)
        argv = item["train_command" if stage == "training" else "evaluation_command"]
        argv = [value.replace("{checkpoint}", str(checkpoint)) if checkpoint else value for value in argv]
        env = {**os.environ, **ENVIRONMENT, "CUDA_VISIBLE_DEVICES": str(lane["gpu"]), "PYTHONHASHSEED": str(item["seed"])}
        record = job[stage]
        log_path = destination / (item["job_id"] + "-" + stage + "-wrapper.log")
        record.update(state="running", started_at_utc=now(), command=argv, resources=snapshot, wrapper_log=str(log_path),
                      environment={**ENVIRONMENT, "CUDA_VISIBLE_DEVICES": str(lane["gpu"]), "PYTHONHASHSEED": str(item["seed"])})
        state.update(stage=stage, active_job_id=item["job_id"]); job["state"] = stage; save(state_path, state)
        with log_path.open("x") as stream:
            child = subprocess.Popen(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record["wrapper_pid"] = child.pid; save(state_path, state)
            try:
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)
                    record["termination_requested_at_utc"] = now()
                    try:
                        record["returncode"] = child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        record["state"] = "unknown_child_may_be_running"
                raise
        paths = [line[len("RUN_DIR="):] for line in log_path.read_text().splitlines() if line.startswith("RUN_DIR=")]
        record.update(returncode=code, ended_at_utc=now())
        if len(paths) == 1:
            record["run_dir"] = paths[0]
        save(state_path, state)
        require(code == 0 and len(paths) == 1, "Stage wrapper failed or did not identify exactly one run")
        run = inside(paths[0], root)
        require(run.parent == root / "runs", "Logged run must be directly inside project runs")
        return run

    try:
        os.sched_setaffinity(0, lane["cpu_affinity"])
        for item, job in zip(lane["jobs"], state["jobs"]):
            current_job = job
            training = launch(item, job, "training")
            evidence = verify_training(training, item, plan, root)
            job["training"].update(state="completed", verified_at_utc=now(), **evidence); save(state_path, state)
            evaluation = launch(item, job, "evaluation", training / "training/checkpoints/best.pth")
            proof = verify_evaluation(training, evaluation, item, evidence, plan)
            job["evaluation"].update(state="completed", verified_at_utc=now(), **proof)
            check_frozen(plan_path, plan_sha, plan, root)
            job.update(state="completed", completed_at_utc=now()); save(state_path, state)
        state.update(state="completed", stage="completed", ended_at_utc=now()); save(state_path, state)
        save(destination / "summary.json", {"schema_version": 1, "kind": KIND, "state": "completed", "lane": lane_name,
            "plan_sha256": plan_sha, "completed_at_utc": now(), "test_used": False,
            "jobs": [{"job_id": job["job_id"], "method": job["method"], "seed": job["seed"],
                      "training_run": job["training"]["run_dir"], "evaluation_run": job["evaluation"]["run_dir"],
                      **{key: job["training"][key] for key in ("selected_epoch", "best_checkpoint_sha256", "best_metrics",
                                                               "parent_training_run", "retained_parent_best")}}
                     for job in state["jobs"]]})
        return 0
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        if current_job:
            stage = state["stage"]
            if stage in ("training", "evaluation") and current_job[stage]["state"] == "running":
                current_job[stage]["state"] = "interrupted" if interrupted else "failed"
            current_job["state"] = "interrupted" if interrupted else "failed"
        state.update(state="interrupted" if interrupted else "failed", ended_at_utc=now(), error_type=type(error).__name__,
                     error=str(error), recovery="Preserve completed training/evaluations and permanent claims. Never replay the lane to repair delivery.")
        try:
            save(state_path, state)
        except BaseException:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--lane", choices=("gpu0", "gpu1"), required=True)
    args = parser.parse_args()
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Lane received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    return run_lane(args.plan, args.lane)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
