"""Execute one fixed two-seed training/evaluation lane, with no implicit retry.

The JSON plan contains complete run_logged argv lists in train_command and
evaluation_command. Only {checkpoint} is substituted here; run_logged expands
{run_dir}. The permanent lane claim prevents replay even after delivery fails.
Training and evaluation have separate states so a failed evaluation cannot be
mistaken for permission to repeat a completed training run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [20260923, 20260924]
KIND = "poet-paired-seed-confirmation-v1"
PROTOCOLS = {method: "poet-methods-fp32-weight1-epoch-resume-v" + version
             for method, version in (("baseline", "1"), ("geometry", "2"))}
ENVIRONMENT = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
               "OPENBLAS_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "PYTHONUNBUFFERED": "1"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def check_gate(gate, root=ROOT):
    if gate.get("passed") is not True:
        raise ValueError("The preregistered ablation gate has not passed")
    path = inside(gate["evidence_path"], root)
    if sha(path) != gate["evidence_sha256"]:
        raise ValueError("Gate evidence changed")
    evidence = read(path)
    decision = evidence.get("gate", {})
    if (evidence.get("status") != "verified" or decision.get("supported") is not True
            or decision.get("decision") != "eligible_for_two_additional_paired_seeds"
            or evidence.get("normal_reference_exact") is not True
            or evidence.get("full_cohort_identity_and_order_equal") is not True
            or evidence.get("samples") != 10304 or evidence.get("classes") != 2731):
        raise ValueError("Gate evidence does not verify the authorized continuation")
    for condition in ("off", "shuffled"):
        comparison = decision.get("comparisons", {}).get(condition, {})
        if (comparison.get("instance_top1_strictly_higher") is not True
                or comparison.get("macro_top1_strictly_higher") is not True):
            raise ValueError("Gate comparison is not positive for " + condition)


def inside(path, root=ROOT):
    result = Path(path)
    if not result.is_absolute():
        result = root / result
    result = result.resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Path is outside the recorded workspace: " + str(path))
    return result


def command_options(command, script, root=ROOT):
    """Only accept a direct logged invocation of the unchanged approved script."""
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError("Commands must be argv lists of strings")
    if len(command) < 8 or command.count("--") != 1:
        raise ValueError("One run_logged separator is required")
    split = command.index("--")
    expected_python = (root / ".venv/bin/python").absolute()
    # Do not resolve the Python symlink, which may point outside the workspace.
    def executable(value):
        path = Path(value)
        return path.absolute() if path.is_absolute() else (root / path).absolute()
    if (executable(command[0]) != expected_python or command[1] != "scripts/run_logged.py"
            or executable(command[split + 1]) != expected_python or command[split + 2] != script):
        raise ValueError("A direct project Python/run_logged/script command is required")
    values = command[split + 3:]
    if len(values) % 2:
        raise ValueError("Only explicit option/value pairs are supported; no smoke or resume flags")
    result = {}
    for flag, value in zip(values[::2], values[1::2]):
        if not flag.startswith("--") or flag in result:
            raise ValueError("Duplicate or invalid command option")
        result[flag] = value
    return result


def validate_plan(plan, lane_name, root=ROOT):
    if (plan.get("schema_version") != 1 or plan.get("kind") != KIND
            or plan.get("workspace") != str(root) or plan.get("seeds") != SEEDS
            or (plan.get("epochs"), plan.get("training_samples"), plan.get("validation_samples"),
                plan.get("optimizer_steps")) != (10, 40154, 10304, 3140)):
        raise ValueError("Plan does not match the authorized two paired seeds and budget")
    check_gate(plan.get("gate", {}), root)
    lanes = plan["lanes"]
    if set(lanes) != {"gpu0", "gpu1"} or lane_name not in lanes:
        raise ValueError("Exactly the GPU 0 baseline and GPU 1 geometry lanes are required")
    affinities = []
    for name, method, gpu in (("gpu0", "baseline", 0), ("gpu1", "geometry", 1)):
        lane = lanes[name]
        if lane["gpu"] != gpu or [job["seed"] for job in lane["jobs"]] != SEEDS:
            raise ValueError("Lane GPU or seed order differs")
        cpus = lane["cpu_affinity"]
        if not cpus or len(set(cpus)) != len(cpus) or any(type(x) is not int or x < 0 for x in cpus):
            raise ValueError("CPU affinity must contain unique nonnegative integers")
        affinities.append(set(cpus))
        for job in lane["jobs"]:
            seed = job["seed"]
            if (job["method"] != method or job["job_id"] != method + "-seed" + str(seed)
                    or job["protocol_version"] != PROTOCOLS[method]):
                raise ValueError("Job identity/protocol does not match its lane")
            train = command_options(job["train_command"], "scripts/finetune_methods.py", root)
            evaluate = command_options(job["evaluation_command"], "scripts/evaluate_methods.py", root)
            common = {"--method": method, "--seed": str(seed), "--head-seed": str(seed),
                      "--epochs": "10", "--batch-size": "128", "--eval-batch-size": "128",
                      "--num-threads": "4", "--device": "cuda:0", "--output-dir": "{run_dir}/training",
                      "--train-cache-manifest": plan["training_manifest"],
                      "--val-cache-manifest": plan["validation_manifest"]}
            if any(train.get(k) != v for k, v in common.items()):
                raise ValueError("Training flags differ from fixed paired-seed protocol")
            if float(train.get("--lr", "nan")) != 3e-5 or float(train.get("--weight-decay", "nan")) != .01:
                raise ValueError("Base optimizer settings differ")
            allowed = set(common) | {"--lr", "--weight-decay"}
            eval_expected = {"--checkpoint": "{checkpoint}", "--cache-manifest": plan["validation_manifest"],
                             "--batch-size": "128", "--num-threads": "4", "--device": "cuda:0",
                             "--output-dir": "{run_dir}/evaluation"}
            if method == "geometry":
                extra = {"--geometry-hidden": "32", "--train-geometry-manifest": plan["training_geometry_manifest"],
                         "--val-geometry-manifest": plan["validation_geometry_manifest"]}
                if any(train.get(k) != v for k, v in extra.items()) or float(train.get("--module-lr", "nan")) != 3e-4:
                    raise ValueError("Geometry settings differ")
                allowed |= set(extra) | {"--module-lr"}
                eval_expected["--geometry-manifest"] = plan["validation_geometry_manifest"]
            if set(train) != allowed or evaluate != eval_expected:
                raise ValueError("Unexpected training/evaluation options")
    if affinities[0] & affinities[1]:
        raise ValueError("Lane CPU affinities overlap")
    frozen = plan["frozen_files"]
    needed = {"scripts/run_logged.py", "scripts/finetune_methods.py", "scripts/evaluate_methods.py",
              "docs/tools/run_paired_seed_lane.py", plan["training_manifest"], plan["validation_manifest"],
              plan["training_geometry_manifest"], plan["validation_geometry_manifest"]}
    if not needed.issubset(frozen):
        raise ValueError("Required code or manifests are not frozen")
    return lanes[lane_name]


def check_frozen(plan_path, plan_sha, plan, root=ROOT):
    if sha(plan_path) != plan_sha:
        raise ValueError("Launch plan changed after claim")
    for relative, expected in plan["frozen_files"].items():
        if sha(inside(relative, root)) != expected:
            raise ValueError("Frozen input/source changed: " + relative)
    check_gate(plan["gate"], root)


def verify_predictions(training, evaluation, expected_count):
    count = 0
    with Path(training).open() as before, Path(evaluation).open() as after:
        for left, right in zip_longest(before, after):
            if left is None or right is None:
                raise ValueError("Independent prediction denominator differs")
            original, reloaded = json.loads(left), json.loads(right)
            if reloaded.pop("status", None) != "ok" or original != reloaded:
                raise ValueError("Independent identity/prediction/probability/logits differs")
            count += 1
    if count != expected_count:
        raise ValueError("Independent prediction count is incomplete")
    return count


def verify_run(run):
    manifest = read(run / "manifest.json")
    if (manifest["state"] != "completed" or manifest.get("returncode") != 0
            or (run / "exit_code.txt").read_text().strip() != "0"):
        raise ValueError("Logged run did not complete with exit 0")


def verify_training(run, item, plan, root=ROOT):
    verify_run(run)
    summary = read(run / "training/summary.json")
    protocol = read(run / "training/protocol.json")["protocol"]
    if (summary["state"] != "completed" or summary["scope"] != "formal_finetuning"
            or summary["completed_epoch"] != 10 or summary["global_optimizer_step"] != 3140
            or summary["samples_seen"] != 401540 or protocol["method"] != item["method"]
            or protocol["version"] != item["protocol_version"]):
        raise ValueError("Training completion counters or protocol differ")
    config = summary["config"]
    expected = {"method": item["method"], "seed": item["seed"], "head_seed": item["seed"], "epochs": 10, "batch_size": 128,
                "eval_batch_size": 128, "num_threads": 4, "lr": 3e-5, "weight_decay": .01,
                "module_lr": 3e-4 if item["method"] == "geometry" else None,
                "resume": None, "engineering_smoke": False}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed training config differs from fixed plan")
    history = read(run / "training/metrics_history.json")["epochs"]
    if [row["epoch"] for row in history] != list(range(11)):
        raise ValueError("Training history must include epoch 0 and all 10 trained epochs")
    for row in history:
        metrics = row["metrics"]
        if (metrics["attempted"] != 10304 or metrics["inferred"] != 10304 or metrics["failed"] != 0
                or not math.isfinite(metrics["per_class_full_vocabulary"]["top1"])):
            raise ValueError("An epoch's full validation is missing or invalid")
    best = max(history, key=lambda row: row["metrics"]["per_class_full_vocabulary"]["top1"])
    if summary["best"] != {"epoch": best["epoch"], "score": best["metrics"]["per_class_full_vocabulary"]["top1"]}:
        raise ValueError("Best checkpoint selection differs from earliest maximum macro Top-1")
    checkpoints = read(run / "training/checkpoints/index.json")
    for name in ("last", "best"):
        entry = checkpoints[name]
        path = inside(entry["path"], root)
        if not path.is_relative_to(run.resolve()) or sha(path) != entry["sha256"]:
            raise ValueError("Checkpoint index path/hash differs")
    if (checkpoints["last"]["epoch"] != 10 or checkpoints["best"]["epoch"] != best["epoch"]
            or sha(run / "training/checkpoints/best.pth") != checkpoints["best"]["sha256"]):
        raise ValueError("Checkpoint epoch or best alias hash differs")
    return {"summary_sha256": sha(run / "training/summary.json"), "protocol_sha256": summary["protocol_sha256"],
            "best_checkpoint_sha256": checkpoints["best"]["sha256"], "selected_epoch": best["epoch"],
            "best_metrics": best["metrics"]}


def verify_evaluation(training, evaluation, item, evidence, plan):
    verify_run(evaluation)
    target = evaluation / "evaluation"
    summary = read(target / "summary.json")
    if (summary["status"] != "completed" or summary["scope"] != "official_validation"
            or summary["method"] != item["method"] or summary["attribute_targets_read"] is not False
            or summary["checkpoint_epoch"] != evidence["selected_epoch"]
            or summary["checkpoint_sha256"] != evidence["best_checkpoint_sha256"]
            or summary["protocol_sha256"] != evidence["protocol_sha256"]
            or summary["metrics"] != evidence["best_metrics"]
            or read(target / "metrics.json") != summary["metrics"]):
        raise ValueError("Independent best evaluation identity or metrics differs")
    if sha(training / "training/checkpoints/best.pth") != evidence["best_checkpoint_sha256"]:
        raise ValueError("Training best checkpoint changed during evaluation")
    epoch = training / "training/evaluation" / ("epoch-%04d" % evidence["selected_epoch"])
    count = verify_predictions(epoch / "predictions.jsonl", target / "predictions.jsonl", plan["validation_samples"])
    if (read(epoch / "metrics.json")["metrics"] != summary["metrics"]
            or (epoch / "per_class.csv").read_bytes() != (target / "per_class.csv").read_bytes()):
        raise ValueError("Selected-epoch metrics or per-class table differs")
    return {"verified_predictions": count, "summary_sha256": sha(target / "summary.json"),
            "predictions_sha256": sha(target / "predictions.jsonl")}


def run_lane(plan_path, lane_name, root=ROOT):
    plan_path = inside(plan_path, root)
    plan_bytes = plan_path.read_bytes()
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    lane = validate_plan(plan, lane_name, root)
    destination = plan_path.parent / lane_name
    destination.mkdir(exist_ok=True)
    # Never remove this claim, including after failed verification or interruption.
    with (destination / "claim.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "created_at_utc": now(), "plan_sha256": plan_sha}, stream)
    state_path = destination / "state.json"
    state = {"schema_version": 1, "state": "running", "stage": "preflight", "pid": os.getpid(),
             "started_at_utc": now(), "plan_sha256": plan_sha, "lane": lane_name,
             "jobs": [{"job_id": item["job_id"], "method": item["method"], "seed": item["seed"],
                       "state": "queued", "training": {"state": "queued"}, "evaluation": {"state": "queued"}}
                      for item in lane["jobs"]]}
    save(state_path, state)
    current_job = None

    def launch(item, job, stage, checkpoint=None):
        check_frozen(plan_path, plan_sha, plan, root)
        free = int(subprocess.check_output(["nvidia-smi", "-i", str(lane["gpu"]),
                   "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        if free < 8192:
            raise RuntimeError("Assigned GPU has less than 8 GiB free; no automatic retry")
        command = item["train_command" if stage == "training" else "evaluation_command"]
        command = [value.replace("{checkpoint}", str(checkpoint)) if checkpoint else value for value in command]
        env = {**os.environ, **ENVIRONMENT, "PYTHONHASHSEED": str(item["seed"])}
        record = job[stage]
        log_path = destination / (item["job_id"] + "-" + stage + "-wrapper.log")
        record.update(state="running", started_at_utc=now(), command=command,
                      environment={**ENVIRONMENT, "PYTHONHASHSEED": env["PYTHONHASHSEED"]},
                      gpu_free_mib_before=free, wrapper_log=str(log_path))
        state.update(stage=stage, active_job_id=item["job_id"])
        job["state"] = stage
        save(state_path, state)
        with log_path.open("x") as stream:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record["wrapper_pid"] = child.pid
            save(state_path, state)
            try:
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)  # run_logged forwards to its owned child session.
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
        if code != 0 or len(paths) != 1:
            raise RuntimeError(stage + " wrapper failed or did not identify exactly one run")
        run = inside(paths[0], root)
        if not run.is_relative_to((root / "runs").resolve()):
            raise ValueError("Run directory must be in this workspace's runs folder")
        return run

    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(lane["gpu"]):
            raise ValueError("CUDA mapping differs from the recorded lane")
        os.sched_setaffinity(0, lane["cpu_affinity"])
        for item, job in zip(lane["jobs"], state["jobs"]):
            current_job = job
            training = launch(item, job, "training")
            evidence = verify_training(training, item, plan, root)
            job["training"].update(state="completed", verified_at_utc=now(), **evidence)
            save(state_path, state)
            evaluation = launch(item, job, "evaluation", training / "training/checkpoints/best.pth")
            proof = verify_evaluation(training, evaluation, item, evidence, plan)
            job["evaluation"].update(state="completed", verified_at_utc=now(), **proof)
            check_frozen(plan_path, plan_sha, plan, root)
            job.update(state="completed", completed_at_utc=now())
            save(state_path, state)
        state.update(state="completed", stage="completed", ended_at_utc=now())
        save(state_path, state)
        save(destination / "summary.json", {"schema_version": 1, "state": "completed", "lane": lane_name,
            "plan_sha256": plan_sha, "completed_at_utc": now(), "test_used": False,
            "jobs": [{"job_id": job["job_id"], "method": job["method"], "seed": job["seed"],
                      "training_run": job["training"]["run_dir"], "evaluation_run": job["evaluation"]["run_dir"],
                      **{key: job["training"][key] for key in ("selected_epoch", "best_checkpoint_sha256", "best_metrics")}}
                     for job in state["jobs"]]})
        return 0
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        if current_job:
            stage = state["stage"]
            if stage in ("training", "evaluation") and current_job[stage]["state"] == "running":
                current_job[stage]["state"] = "interrupted" if interrupted else "failed"
            current_job["state"] = "interrupted" if interrupted else "failed"
        state.update(state="interrupted" if interrupted else "failed", ended_at_utc=now(),
                     error_type=type(error).__name__, error=str(error),
                     recovery="Inspect existing run evidence. Never remove/replay this claim. Completed training must not be rerun to repair evaluation.")
        save(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--lane", required=True, choices=("gpu0", "gpu1"))
    args = parser.parse_args()
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Lane received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    return run_lane(args.plan, args.lane)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
