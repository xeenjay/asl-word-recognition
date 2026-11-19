"""Run exactly one of the two fixed cross-signer pilot jobs, with no retry.

A permanent per-lane claim prevents accidental replay. Training and evaluation
have separate states: failed delivery never causes successful training to rerun.
The immutable JSON plan carries explicit run_logged argv and frozen input SHA256.
--validate-only is read-only and does not create a claim or inspect GPU resources.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
import re
import shutil
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260926
KIND = "poet-cross-signer-controlled-pilot-v1"
PROTOCOL = "poet-cross-signer-supcon-fp32-v1"
SAMPLER = "class-signer-clip-cycles-private-python-v1"
METHODS = ("balanced_ce", "cross_signer_supcon")
SAMPLE_VISITS = 10 * 314 * 128
MIN_DISK_BYTES = 20 * 1024 ** 3
MIN_HOST_AVAILABLE_BYTES = 16 * 1024 ** 3
MIN_GPU_MIB = 8192
REQUIRED_SOURCE = {"scripts/run_logged.py", "scripts/finetune_contrastive.py",
                   "scripts/evaluate_contrastive.py", "docs/tools/run_contrastive_lane.py"}
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


def inside(path, root=ROOT):
    result = Path(path)
    if not result.is_absolute():
        result = root / result
    result = result.resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Path is outside the recorded workspace: " + str(path))
    return result


def command_options(command, script, root=ROOT):
    """Accept only direct project Python/run_logged argv, never a shell."""
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError("Commands must be argv lists of strings")
    if len(command) < 11 or command.count("--") != 1 or command.index("--") != 6:
        raise ValueError("Exactly --name and --config then one run_logged separator are required")
    expected_python = (root / ".venv/bin/python").absolute()
    def executable(value):
        path = Path(value)
        return path.absolute() if path.is_absolute() else (root / path).absolute()
    if (executable(command[0]) != expected_python or command[1] != "scripts/run_logged.py"
            or command[2] != "--name" or command[4] != "--config"
            or executable(command[7]) != expected_python or command[8] != script
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", command[3])):
        raise ValueError("A direct named project Python/run_logged/script command is required")
    config = json.loads(command[5])
    if not isinstance(config, dict) or config.get("test_used") is not False:
        raise ValueError("Inline logged configuration must explicitly exclude test")
    values = command[9:]
    if len(values) % 2:
        raise ValueError("Only explicit option/value pairs are supported; no smoke or resume flags")
    result = {}
    for flag, value in zip(values[::2], values[1::2]):
        if not flag.startswith("--") or flag in result:
            raise ValueError("Duplicate or invalid command option")
        result[flag] = value
    return result


def expected_training_options(plan, method):
    return {"--method": method, "--seed": str(SEED), "--head-seed": str(SEED),
            "--epochs": "10", "--batch-size": "128", "--eval-batch-size": "128",
            "--num-threads": "4", "--device": "cuda:0", "--output-dir": "{run_dir}/training",
            "--train-cache-manifest": plan["training_manifest"],
            "--val-cache-manifest": plan["validation_manifest"],
            "--checkpoint-dir": "assets/checkpoints", "--lr": "3e-5", "--weight-decay": "0.01",
            "--supcon-lambda": "0" if method == "balanced_ce" else "0.05",
            "--temperature": "0.1", "--projection-lr": "3e-4", "--classes-per-batch": "32",
            "--signers-per-class": "4", "--steps-per-epoch": "314"}


def validate_plan(plan, lane_name, root=ROOT):
    expected = {"schema_version": 1, "kind": KIND, "workspace": str(root), "seed": SEED,
                "epochs": 10, "training_samples": 40154, "validation_samples": 10304,
                "optimizer_steps": 3140, "sample_visits": SAMPLE_VISITS, "test_used": False}
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("Plan does not match the fixed controlled-pilot budget")
    lanes = plan["lanes"]
    if set(lanes) != {"gpu0", "gpu1"} or lane_name not in lanes:
        raise ValueError("Exactly the GPU 0 control and GPU 1 contrastive lanes are required")
    affinities = []
    for gpu, method in enumerate(METHODS):
        lane = lanes["gpu" + str(gpu)]
        if lane["gpu"] != gpu or len(lane["jobs"]) != 1:
            raise ValueError("Exactly one job per assigned GPU is required")
        cpus = lane["cpu_affinity"]
        if (not 4 <= len(cpus) <= 16 or len(set(cpus)) != len(cpus)
                or any(type(x) is not int or x < 0 for x in cpus)):
            raise ValueError("CPU affinity needs 4 to 16 unique nonnegative integers")
        affinities.append(set(cpus))
        job = lane["jobs"][0]
        if (job["method"] != method or job["seed"] != SEED
                or job["job_id"] != method + "-seed" + str(SEED)
                or job["protocol_version"] != PROTOCOL):
            raise ValueError("Job identity/protocol differs from its lane")
        train = command_options(job["train_command"], "scripts/finetune_contrastive.py", root)
        evaluate = command_options(job["evaluation_command"], "scripts/evaluate_contrastive.py", root)
        expected_train = expected_training_options(plan, method)
        # Permit equivalent numeric spelling, while retaining exact option names.
        float_flags = {"--lr", "--weight-decay", "--supcon-lambda", "--temperature", "--projection-lr"}
        if set(train) != set(expected_train) or any(
                (float(train[k]) != float(v) if k in float_flags else train[k] != v)
                for k, v in expected_train.items()):
            raise ValueError("Training options differ from the fixed controlled-pilot protocol")
        eval_expected = {"--checkpoint": "{checkpoint}", "--cache-manifest": plan["validation_manifest"],
                         "--checkpoint-dir": "assets/checkpoints", "--batch-size": "128", "--num-threads": "4",
                         "--device": "cuda:0", "--output-dir": "{run_dir}/evaluation"}
        if evaluate != eval_expected:
            raise ValueError("Unexpected evaluation options")
        for command in (job["train_command"], job["evaluation_command"]):
            config = json.loads(command[5])
            if config.get("method") != method or config.get("seed") != SEED:
                raise ValueError("Logged method/seed differs from the actual command")
    if affinities[0] & affinities[1]:
        raise ValueError("Lane CPU affinities overlap")
    frozen = plan["frozen_files"]
    needed = REQUIRED_SOURCE | {plan["training_manifest"], plan["validation_manifest"],
             "assets/checkpoints/ASL_Citizen_PoET.pth", "assets/checkpoints/d_LId2Label.pkl",
             "assets/checkpoints/d_Label2LId.pkl"}
    # Every Python source under scripts/vendor is part of the frozen environment,
    # including modules transitively imported by the new trainer/evaluator.
    needed |= {str(p.relative_to(root)) for folder in ("scripts", "vendor/poet")
               for p in (root / folder).rglob("*.py") if "__pycache__" not in p.parts}
    if not needed.issubset(frozen):
        raise ValueError("Required source, pretrained assets or manifests are not frozen: "
                         + ", ".join(sorted(needed - set(frozen))))
    for relative, digest in frozen.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Frozen entries require workspace-relative paths and SHA256")
        inside(relative, root)
    return lanes[lane_name]


def check_frozen(plan_path, plan_sha, plan, root=ROOT):
    if sha(plan_path) != plan_sha:
        raise ValueError("Launch plan changed after claim")
    for relative, expected in plan["frozen_files"].items():
        if sha(inside(relative, root)) != expected:
            raise ValueError("Frozen input/source changed: " + relative)


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
    for name, key in (("config.json", "config_sha256"), ("code_manifest.json", "code_manifest_sha256")):
        if sha(run / name) != manifest[key]:
            raise ValueError("Logged configuration/source inventory changed")
    inventory = read(run / "artifacts.json")["files"]
    seen = set()
    total = 0
    for record in inventory:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or record["path"] in seen:
            raise ValueError("Invalid or duplicated artifact path")
        seen.add(record["path"])
        path = run / relative
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(run.resolve())
                or path.stat().st_size != record["size_bytes"] or sha(path) != record["sha256"]):
            raise ValueError("Logged artifact changed: " + str(relative))
        total += record["size_bytes"]
    return {"manifest_sha256": sha(run / "manifest.json"), "files_verified": len(inventory),
            "bytes_verified": total, "artifact_index_sha256": sha(run / "artifacts.json")}


def verify_training(run, item, plan, root=ROOT):
    run_evidence = verify_run(run)
    summary = read(run / "training/summary.json")
    protocol_record = read(run / "training/protocol.json")
    protocol = protocol_record["protocol"]
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if summary["protocol_sha256"] != protocol_hash or protocol_record["sha256"] != protocol_hash:
        raise ValueError("Training protocol fingerprint differs")
    if (summary["state"] != "completed" or summary["scope"] != "formal_finetuning"
            or summary["completed_epoch"] != 10 or summary["global_optimizer_step"] != 3140
            or summary["samples_seen"] != SAMPLE_VISITS or protocol["method"] != item["method"]
            or protocol["version"] != item["protocol_version"]):
        raise ValueError("Training completion counters or protocol differ")
    config = summary["config"]
    expected = {"method": item["method"], "seed": item["seed"], "head_seed": item["seed"], "epochs": 10, "batch_size": 128,
                "eval_batch_size": 128, "num_threads": 4, "lr": 3e-5, "weight_decay": .01,
                "supcon_lambda": 0.0 if item["method"] == "balanced_ce" else .05,
                "temperature": .1, "projection_lr": 3e-4, "classes_per_batch": 32,
                "signers_per_class": 4, "steps_per_epoch": 314,
                "resume": None, "engineering_smoke": False}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed training config differs from fixed plan")
    sampling = protocol["training"]["sampler"]
    sampler_expected = {"version": SAMPLER, "seed": SEED, "classes_per_batch": 32,
                        "signers_per_class": 4, "steps_per_epoch": 314, "batch_size": 128,
                        "draws_per_epoch": 40192, "training_videos": 40154, "observed_classes": 2731}
    if (any(sampling.get(k) != v for k, v in sampler_expected.items())
            or summary["sampling"] != sampling
            or protocol["loss"]["supcon_lambda"] != expected["supcon_lambda"]
            or protocol["loss"]["temperature"] != .1):
        raise ValueError("Actual sampler/loss protocol differs from fixed plan")
    for subset, key in (("train", "training_manifest"), ("validation", "validation_manifest")):
        if protocol["data"][subset]["manifest_sha256"] != plan["frozen_files"][plan[key]]:
            raise ValueError("Training data manifest differs from frozen plan")
    history = read(run / "training/metrics_history.json")["epochs"]
    if [row["epoch"] for row in history] != list(range(11)):
        raise ValueError("Training history must include epoch 0 and all 10 trained epochs")
    for row in history:
        metrics = row["metrics"]
        if (metrics["attempted"] != 10304 or metrics["inferred"] != 10304 or metrics["failed"] != 0
                or not math.isfinite(metrics["per_class_full_vocabulary"]["top1"])):
            raise ValueError("An epoch's full validation is missing or invalid")
        if row["epoch"] and (row["training"]["samples"] != 314 * 128
                             or row["training"]["optimizer_steps"] != 314):
            raise ValueError("A trained epoch has a different sample/step budget")
        if row["epoch"]:
            coverage = row["training"]["sampling"]
            if (coverage["epoch"] != row["epoch"] or coverage["draws"] != 40192
                    or coverage["batches"] != 314 or coverage["unique_classes"] != 2731
                    or coverage["unique_videos"] + coverage["repeated_draws"] != 40192
                    or coverage["unique_videos"] + coverage["unseen_videos"] != 40154
                    or sum(coverage["class_draw_counts"].values()) != 40192
                    or sum(coverage["signer_draw_counts"].values()) != 40192
                    or coverage["order_sha256"] != row["training"]["order_sha256"]):
                raise ValueError("Sampling coverage evidence is inconsistent")
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
    return {"run_artifacts": run_evidence, "summary_sha256": sha(run / "training/summary.json"), "protocol_sha256": summary["protocol_sha256"],
            "best_checkpoint_sha256": checkpoints["best"]["sha256"], "selected_epoch": best["epoch"],
            "best_metrics": best["metrics"], "sampling_epochs": [row["training"]["sampling"] for row in history[1:]]}


def recount_predictions(path, metrics, cohort):
    """Independently score the full official cohort, not a successful subset."""
    identities = ("index", "gloss", "signer_id", "split", "source_path", "video_sha256")
    classes = defaultdict(lambda: [0, 0, 0])
    count = 0
    with Path(path).open() as stream:
        for line, item in zip_longest(stream, cohort["samples"]):
            if line is None or item is None:
                raise ValueError("Independent evaluation cohort denominator differs")
            row = json.loads(line)
            if ({k: row[k] for k in identities} != {k: item[k] for k in identities}
                    or row["status"] != "ok" or len(row["top5"]) != 5):
                raise ValueError("Independent evaluation cohort identity or status differs")
            top = row["top5"]
            if (len({p["class_id"] for p in top}) != 5 or len({p["gloss"] for p in top}) != 5
                    or any(not math.isfinite(p["probability"]) or not 0 <= p["probability"] <= 1 for p in top)):
                raise ValueError("Invalid top-five predictions")
            target = row["gloss"]
            classes[target][0] += 1
            classes[target][1] += top[0]["gloss"] == target
            classes[target][2] += target in [p["gloss"] for p in top]
            count += 1
    if (count != 10304 or len(classes) != 2731 or metrics["attempted"] != count
            or metrics["inferred"] != count or metrics["failed"] != 0):
        raise ValueError("Independent full-cohort denominator/classes differ")
    for column, key in ((1, "top1"), (2, "top5")):
        correct = sum(row[column] for row in classes.values())
        macro = sum(row[column] / row[0] for row in classes.values()) / 2731
        if (metrics[key + "_correct"] != correct or metrics["per_instance"][key] != correct / count
                or abs(metrics["per_class_full_vocabulary"][key] - macro) > 1e-12):
            raise ValueError("Independent prediction recount differs from reported metrics")
    return count


def verify_evaluation(training, evaluation, item, evidence, plan, root=ROOT):
    run_evidence = verify_run(evaluation)
    target = evaluation / "evaluation"
    summary = read(target / "summary.json")
    if (summary["status"] != "completed" or summary["scope"] != "official_validation"
            or summary["method"] != item["method"]
            or summary["projection_used_at_inference"] is not False
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
    recount_predictions(target / "predictions.jsonl", summary["metrics"], read(inside(plan["validation_manifest"], root)))
    return {"run_artifacts": run_evidence, "verified_predictions": count, "summary_sha256": sha(target / "summary.json"),
            "predictions_sha256": sha(target / "predictions.jsonl")}


def resources(lane, root=ROOT):
    """Take a resource snapshot without changing or inspecting other users' jobs."""
    free = int(subprocess.check_output(["nvidia-smi", "-i", str(lane["gpu"]),
               "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, timeout=20).strip())
    memory = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
              for line in Path("/proc/meminfo").read_text().splitlines()}
    available = int(memory["MemAvailable"].split()[0]) * 1024
    snapshot = {"checked_at_utc": now(), "gpu_index": lane["gpu"], "gpu_free_mib": free,
                "disk_path": str(root), "disk_free_bytes": shutil.disk_usage(root).free,
                "host_mem_available_bytes": available,
                "cpu_affinity": sorted(os.sched_getaffinity(0))}
    errors = []
    if free < MIN_GPU_MIB:
        errors.append("assigned GPU has less than 8 GiB free")
    if snapshot["disk_free_bytes"] < MIN_DISK_BYTES:
        errors.append("project disk has less than 20 GiB free")
    if available < MIN_HOST_AVAILABLE_BYTES:
        errors.append("host has less than 16 GiB available memory")
    if snapshot["cpu_affinity"] != sorted(lane["cpu_affinity"]):
        errors.append("actual CPU affinity differs from the plan")
    snapshot["errors"] = errors
    return snapshot


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
        json.dump({"pid": os.getpid(), "created_at_utc": now(), "plan_sha256": plan_sha,
                   "lane": lane_name, "gpu": lane["gpu"], "cpu_affinity": lane["cpu_affinity"]}, stream)
    state_path = destination / "state.json"
    state = {"schema_version": 1, "state": "running", "stage": "preflight", "pid": os.getpid(),
             "started_at_utc": now(), "plan_sha256": plan_sha, "lane": lane_name,
             "gpu": lane["gpu"], "cpu_affinity": lane["cpu_affinity"],
             "jobs": [{"job_id": item["job_id"], "method": item["method"], "seed": item["seed"],
                       "state": "queued", "training": {"state": "queued"}, "evaluation": {"state": "queued"}}
                      for item in lane["jobs"]]}
    save(state_path, state)
    current_job = None

    def launch(item, job, stage, checkpoint=None):
        state.update(stage=stage, active_job_id=item["job_id"])
        record = job[stage]
        record.update(state="preflight", started_at_utc=now())
        save(state_path, state)
        check_frozen(plan_path, plan_sha, plan, root)
        record["resources_before"] = resources(lane, root)
        save(state_path, state)
        if record["resources_before"]["errors"]:
            raise RuntimeError("Resource check failed; no automatic retry: " + "; ".join(record["resources_before"]["errors"]))
        command = item["train_command" if stage == "training" else "evaluation_command"]
        command = [value.replace("{checkpoint}", str(checkpoint)) if checkpoint else value for value in command]
        env = {**os.environ, **ENVIRONMENT, "PYTHONHASHSEED": str(item["seed"])}
        log_path = destination / (item["job_id"] + "-" + stage + "-wrapper.log")
        record.update(state="running", started_at_utc=now(), command=command,
                      environment={**ENVIRONMENT, "PYTHONHASHSEED": env["PYTHONHASHSEED"],
                                   "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]}, wrapper_log=str(log_path))
        job["state"] = stage
        save(state_path, state)
        with log_path.open("x") as stream:
            child = None
            try:
                child = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
                record["wrapper_pid"] = child.pid
                save(state_path, state)
                while True:
                    try:
                        code = child.wait(timeout=2)
                        break
                    except subprocess.TimeoutExpired:
                        paths = [line[len("RUN_DIR="):] for line in log_path.read_text().splitlines()
                                 if line.startswith("RUN_DIR=")]
                        if len(paths) == 1 and "run_dir" not in record:
                            path = inside(paths[0], root)
                            if not path.is_relative_to((root / "runs").resolve()):
                                raise ValueError("Live run directory is outside the workspace runs folder")
                            record["run_dir"] = str(path)
                            save(state_path, state)
            except BaseException:
                if child is not None and child.poll() is None:
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
            proof = verify_evaluation(training, evaluation, item, evidence, plan, root)
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
            if stage in ("training", "evaluation") and current_job[stage]["state"] in ("preflight", "running"):
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
    parser.add_argument("--validate-only", action="store_true", help="Read-only plan/hash validation; no claim or training")
    args = parser.parse_args()
    if args.validate_only:
        path = inside(args.plan)
        plan = read(path)
        validate_plan(plan, args.lane)
        check_frozen(path, sha(path), plan)
        print(json.dumps({"state": "validated", "plan_sha256": sha(path), "lane": args.lane}))
        return 0
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
