#!/usr/bin/env python3
"""One bounded RGB pilot: frozen extraction, then three CPU linear probes.

Claims are permanent, including after failure. No automatic replay/resume,
parameter search, test evaluation, or recovery by redoing a completed stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260926
COUNTS = {"train": 3774, "val": 971}
MODES = ("pose_only", "rgb_only", "fusion")
IDENTITY = ("video_sha256", "gloss", "class_id", "signer_id", "split")
REQUIRED_FROZEN = {
    "docs/tools/run_rgb_pilot.py", "scripts/run_logged.py", "scripts/extract_rgb_pilot.py",
    "scripts/rgb_hand_crops.py", "scripts/probe_rgb_pilot.py", "scripts/build_rgb_pilot_plan.py",
    "assets/checkpoints/ASL_Citizen_PoET.pth", "assets/checkpoints/d_LId2Label.pkl",
    "assets/checkpoints/d_Label2LId.pkl", "assets/rgb/resnet18-imagenet1k-v1/metadata.json",
    "assets/rgb/resnet18-imagenet1k-v1/resnet18-f37072fd.pth",
}
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
               "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
               "PYTHONHASHSEED": str(SEED), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "PYTHONDONTWRITEBYTECODE": "1"}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def inside(value, root=ROOT, *, relative=False):
    require(isinstance(value, (str, Path)) and bool(str(value)), "Missing workspace path")
    path = Path(value)
    require(".." not in path.parts and (not relative or not path.is_absolute()),
            "Expected a contained project-relative path")
    root = Path(root).resolve()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(root), "Path escapes project workspace")
    return path


def check_frozen(launch_path, launch_hash, launch, root=ROOT):
    require(sha(launch_path) == launch_hash, "Launch changed after validation/claim")
    require(sha(inside(launch["plan_path"], root, relative=True)) == launch["plan_sha256"],
            "Pilot plan changed")
    for name, expected in launch["frozen_files"].items():
        require(sha(inside(name, root, relative=True)) == expected, "Frozen input changed: " + name)


def verify_run(run):
    """Verify a closed run_logged record and every listed artifact byte."""
    run = Path(run).resolve()
    manifest = read(run / "manifest.json")
    require(manifest.get("state") == "completed" and manifest.get("returncode") == 0
            and (run / "exit_code.txt").read_text().strip() == "0", "Logged run did not complete with exit 0")
    for name, key in (("config.json", "config_sha256"), ("code_manifest.json", "code_manifest_sha256")):
        require(sha(run / name) == manifest.get(key), "Logged config/source inventory changed")
    seen, total = set(), 0
    for artifact in read(run / "artifacts.json")["files"]:
        name = artifact["path"]
        require(name not in seen and name not in {"manifest.json", "artifacts.json"}, "Invalid/duplicate artifact")
        path = inside(name, run, relative=True)
        require(not (run / name).is_symlink() and path.is_file(), "Missing/symlinked artifact")
        require(path.stat().st_size == artifact["size_bytes"] and sha(path) == artifact["sha256"],
                "Logged artifact changed: " + name)
        seen.add(name)
        total += artifact["size_bytes"]
    actual = {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()
              and str(p.relative_to(run)) not in {"manifest.json", "artifacts.json"}}
    require(actual == seen, "Unlisted or missing completed-run artifacts")
    return {"run_manifest_sha256": sha(run / "manifest.json"),
            "artifacts_sha256": sha(run / "artifacts.json"), "verified_files": len(seen), "verified_bytes": total}


def verify_cache(cache, plan, plan_hash, *, engineering=False):
    manifest = read(Path(cache) / "manifest.json")
    require(manifest.get("status") == "completed" and manifest.get("schema_version") == 1
            and manifest.get("failed") == 0 and manifest.get("test_used") is False,
            "Extraction must complete without failures or test")
    require(manifest.get("scope") == ("engineering_train32" if engineering else "rgb_pilot256")
            and manifest.get("plan_sha256") == plan_hash, "Extraction scope/plan differs")
    require(manifest.get("frozen_backbone_unchanged") is True, "Frozen backbone verification missing")
    require(manifest.get("class_ids") == plan["class_ids"] and manifest.get("label_map") == plan["label_map"],
            "Extraction vocabulary differs from the fixed plan")
    require(set(manifest.get("cohorts", {})) == {"train", "val"}, "Unexpected extraction split")
    expected = {"train": plan["engineering_train"], "val": []} if engineering else plan["cohorts"]
    seen, with_hands = set(), 0
    for split in ("train", "val"):
        rows, originals = manifest["cohorts"][split], expected[split]
        require(len(rows) == len(originals), "Extraction cohort was shortened or expanded")
        for row, original in zip(rows, originals):
            require(all(row.get(key) == original.get(key) for key in IDENTITY), "Extraction identity/order differs from plan")
            require(row["split"] == split and row["video_sha256"] not in seen, "Repeated/cross-split extraction identity")
            seen.add(row["video_sha256"])
            require(type(row.get("all_hands_missing")) is bool
                    and type(row.get("valid_hand_slots")) is int and type(row.get("hand_slots")) is int
                    and 0 <= row["valid_hand_slots"] <= row["hand_slots"] <= 16 and row["hand_slots"] > 0
                    and row["all_hands_missing"] == (row["valid_hand_slots"] == 0), "Invalid crop-quality counters")
            with_hands += not row["all_hands_missing"]
    require(not engineering or (len(seen) == 32 and with_hands >= 28), "Engineering hand coverage gate failed")
    return {"cache_manifest_sha256": sha(Path(cache) / "manifest.json"),
            "videos": len(seen), "videos_with_hands": with_hands,
            "split_sizes": {key: len(value) for key, value in manifest["cohorts"].items()}}


def validate_launch(path, root=ROOT):
    root = Path(root).resolve()
    path = inside(path, root)
    require(path.name == "launch.json" and path.parent.parent == root / "runs/rgb-pilot-launches",
            "Use a unique runs/rgb-pilot-launches/<id>/launch.json")
    launch, launch_hash = read(path), sha(path)
    require(launch.get("schema_version") == 1 and launch.get("seed") == SEED
            and type(launch.get("gpu")) is int and launch["gpu"] == 0, "Wrong fixed pilot seed/schema/GPU")
    frozen = launch.get("frozen_files")
    require(isinstance(frozen, dict), "Missing frozen input inventory")
    for name, digest in frozen.items():
        inside(name, root, relative=True)
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "Invalid frozen SHA256")
    required = REQUIRED_FROZEN | {launch["plan_path"], launch["engineering_acceptance"]}
    require(required <= frozen.keys(), "Required sources/weights/plan/acceptance are not frozen")
    require(frozen[launch["plan_path"]] == launch.get("plan_sha256"), "Plan must have one identical frozen hash")
    check_frozen(path, launch_hash, launch, root)
    plan = read(inside(launch["plan_path"], root, relative=True))
    require(plan.get("schema_version") == 1 and plan.get("seed") == SEED
            and len(plan.get("class_ids", [])) == 256 and set(plan.get("cohorts", {})) == set(COUNTS)
            and all(len(plan["cohorts"][key]) == value for key, value in COUNTS.items()), "Wrong fixed pilot cohort")
    require(plan.get("protocol", {}).get("test_used") is False
            and plan["protocol"].get("probe_epochs") == 50, "Wrong pilot protocol")
    for name, digest in plan["metadata_verification"]["files"].items():
        require(sha(inside(name, root, relative=True)) == digest, "Plan metadata/source changed: " + name)
    require(len(plan.get("engineering_train", [])) == 32
            and all(row.get("split") == "train" for row in plan["engineering_train"]), "Engineering cohort must be 32 train videos")
    acceptance = read(inside(launch["engineering_acceptance"], root, relative=True))
    require(acceptance.get("status") == "accepted" and acceptance.get("engineering_run") == launch["engineering_run"]
            and acceptance.get("plan_sha256") == launch["plan_sha256"]
            and acceptance.get("qa_checked_videos") == 32 and acceptance.get("test_used") is False,
            "Engineering visual acceptance is absent or bound to another run/plan")
    units = launch.get("unit_runs")
    require(isinstance(units, list) and len(units) >= 1 and len(set(units)) == len(units), "Need completed unique unit runs")
    unit_proof = {name: verify_run(inside(name, root, relative=True)) for name in units}
    engineering_run = inside(launch["engineering_run"], root, relative=True)
    require(engineering_run.is_relative_to(root / "runs"), "Engineering run must be in runs")
    engineering = verify_run(engineering_run)
    engineering.update(verify_cache(engineering_run / "features-cache", plan, launch["plan_sha256"], engineering=True))
    engineering_manifest = read(engineering_run / "features-cache/manifest.json")
    for name, digest in engineering_manifest["protocol"]["source_sha256"].items():
        require(sha(inside(name, root, relative=True)) == digest,
                "Extraction source changed since accepted engineering run: " + name)
    return launch, plan, {"launch_sha256": launch_hash, "units": unit_proof, "engineering": engineering}


def resources(root=ROOT, *, need_gpu=True):
    query = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
    values = query.stdout.strip().splitlines()
    require(len(values) == 1, "Expected exactly GPU 0 memory report")
    gpu_mib = int(values[0])
    memory = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
              for line in Path("/proc/meminfo").read_text().splitlines()}
    snapshot = {"checked_at_utc": now(), "disk_free_bytes": shutil.disk_usage(root).free,
                "ram_available_bytes": int(memory["MemAvailable"].split()[0]) * 1024,
                "gpu0_free_mib": gpu_mib, "cpu_affinity": sorted(os.sched_getaffinity(0))}
    require(snapshot["disk_free_bytes"] >= 20 * 1024**3, "Less than 20 GiB free project disk")
    require(snapshot["ram_available_bytes"] >= 16 * 1024**3, "Less than 16 GiB available host RAM")
    require(not need_gpu or gpu_mib >= 8 * 1024, "Less than 8 GiB free on GPU 0")
    require(snapshot["cpu_affinity"] == list(range(8)), "CPU affinity differs from 0-7")
    return snapshot


def claim_budget(path, launch, launch_hash, root=ROOT):
    claim = {"schema_version": 1, "seed": SEED, "launch_path": str(Path(path).relative_to(root)),
             "launch_sha256": launch_hash, "plan_sha256": launch["plan_sha256"],
             "created_at_utc": now(), "owner_pid": os.getpid(), "automatic_replay": False}
    budget = Path(root) / "runs/rgb-pilot-launches/seed20260926-budget.json"
    with budget.open("x") as stream:
        json.dump(claim, stream, indent=2)
    with (Path(path).parent / "claim.json").open("x") as stream:
        json.dump(claim, stream, indent=2)


def command(stage, plan_path, root=ROOT, cache=None):
    python = str(Path(root) / ".venv/bin/python")
    config = {"purpose": "rgb_pilot_extraction" if stage == "extraction" else "rgb_pilot_linear_probes",
              "seed": SEED, "test_used": False, "formal_full_vocabulary_training": False}
    wrapper = [python, "scripts/run_logged.py", "--name", "rgb-pilot-" + stage,
               "--config", json.dumps(config), "--", python]
    if stage == "extraction":
        return wrapper + ["scripts/extract_rgb_pilot.py", "--plan", str(plan_path),
                          "--output-dir", "{run_dir}/features-cache", "--device", "cuda:0"]
    require(stage == "probe" and cache is not None, "Unknown bounded stage or missing cache")
    return wrapper + ["scripts/probe_rgb_pilot.py", "--cache", str(cache),
                      "--output-dir", "{run_dir}/probe", "--epochs", "50"]


class OwnedProcess:
    def __init__(self):
        self.child = None
        self.signum = None

    def forward(self, signum, _frame=None):
        self.signum = signum
        if self.child is not None and self.child.poll() is None:
            try:
                self.child.send_signal(signum)  # Only this wrapper owns its child session.
            except ProcessLookupError:
                pass

    def check(self):
        if self.signum is not None:
            raise InterruptedError("Pilot received signal " + str(self.signum))


def run_stage(stage, argv, destination, state, owner, root=ROOT):
    owner.check()
    record = state["stages"][stage]
    state.update(stage=stage, run_dir=None)
    record.update(state="running", started_at_utc=now(), command=argv)
    state_path = destination / "state.json"
    save(state_path, state)
    run_path = None
    with (destination / (stage + "-wrapper.log")).open("x") as log:
        child = subprocess.Popen(argv, cwd=root, env={**os.environ, **ENVIRONMENT},
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        owner.child = child
        record["wrapper_pid"] = child.pid
        save(state_path, state)
        try:
            if owner.signum is not None:
                owner.forward(owner.signum)
            for line in child.stdout:
                log.write(line); log.flush()
                if line.startswith("RUN_DIR="):
                    require(run_path is None, "Wrapper identified more than one run")
                    run_path = inside(line[len("RUN_DIR="):].strip(), root)
                    require(run_path.parent == Path(root) / "runs", "Wrapper run is not directly under project runs")
                    record["run_dir"] = str(run_path)
                    state["run_dir"] = str(run_path)
                    save(state_path, state)
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                child.send_signal(owner.signum or signal.SIGTERM)
            record["returncode"] = child.wait()
            save(state_path, state)
            raise
        finally:
            child.stdout.close()
            owner.child = None
    record.update(returncode=code, ended_at_utc=now())
    save(state_path, state)
    owner.check()
    require(code == 0 and run_path is not None, "Stage wrapper failed or emitted no run")
    return run_path


def verify_probe(run, cache, plan):
    proof = verify_run(run)
    output = Path(run) / "probe"
    summary, config, status = read(output / "summary.json"), read(output / "config.json"), read(output / "status.json")
    require(status.get("status") == summary.get("status") == "completed"
            and summary.get("test_used") is False and summary.get("engineering_smoke") is False
            and summary.get("input_bytes_unchanged") is True and summary.get("identical_training_orders") is True,
            "Probe completion/reload/input verification missing")
    require(summary.get("split_sizes") == config.get("split_sizes") == COUNTS
            and summary.get("cache_manifest_sha256") == config.get("cache_manifest_sha256") == sha(Path(cache) / "manifest.json")
            and Path(config.get("cache", "")).resolve() == Path(cache).resolve(), "Probe used another cache/cohort")
    require(config.get("seed") == SEED and config.get("epochs") == 50 and config.get("test_used") is False
            and config.get("engineering_smoke") is False and config.get("device") == "cpu"
            and config.get("class_ids") == plan["class_ids"] and config.get("label_map") == plan["label_map"],
            "Probe protocol differs from fixed budget/vocabulary")
    canonical = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    require(summary.get("config_sha256") == canonical and status.get("summary_sha256") == sha(output / "summary.json")
            and status.get("artifacts_sha256") == sha(output / "artifacts.json"), "Probe summary/config artifact binding differs")
    require(set(summary.get("models", {})) == set(MODES), "Exactly three probe heads are required")
    orders = []
    for mode in MODES:
        model = summary["models"][mode]
        require(model == read(output / mode / "summary.json"), "Per-head summary differs")
        require(model.get("status") == "completed" and model.get("mode") == mode
                and model.get("epochs") == 50 and model.get("optimizer_steps") == 1500
                and model.get("sample_visits") == 188700, "Probe head has incomplete/wrong budget")
        require(all(model.get(field) is True for field in ("reload_logits_exact", "reload_predictions_exact", "reload_metrics_exact")),
                "Probe best reload did not verify exactly")
        metrics = model["best_metrics"]
        require(metrics.get("samples") == 971 and metrics.get("failed") == 0
                and metrics.get("candidate_classes") == metrics.get("observed_classes") == 256,
                "Probe validation denominator/classes differ")
        require(sha(output / mode / "best.pth") == model["best_checkpoint_sha256"]
                and sha(output / mode / "last.pth") == model["last_checkpoint_sha256"], "Probe checkpoint hash differs")
        orders.append(model.get("epoch_order_sha256"))
    require(all(isinstance(order, list) and len(order) == 50 for order in orders)
            and orders[0] == orders[1] == orders[2], "Actual head training orders differ")
    proof.update(summary_sha256=sha(output / "summary.json"), models=summary["models"],
                 split_sizes=COUNTS, test_used=False)
    return proof


def run_pilot(launch_path, root=ROOT):
    root = Path(root).resolve()
    path = inside(launch_path, root)
    launch, plan, preflight = validate_launch(path, root)
    claim_budget(path, launch, preflight["launch_sha256"], root)
    destination = path.parent
    state = {"schema_version": 1, "state": "running", "stage": "preflight", "pid": os.getpid(),
             "started_at_utc": now(), "launch_sha256": preflight["launch_sha256"],
             "plan_sha256": launch["plan_sha256"], "seed": SEED, "gpu": 0, "cpu_affinity": list(range(8)),
             "preflight": preflight, "stages": {key: {"state": "queued"} for key in ("extraction", "probe")}}
    save(destination / "state.json", state)
    owner = OwnedProcess()
    handlers = {sig: signal.signal(sig, owner.forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        os.sched_setaffinity(0, set(range(8)))
        state["resources_before"] = resources(root)
        save(destination / "state.json", state)
        check_frozen(path, preflight["launch_sha256"], launch, root)
        extraction = run_stage("extraction", command("extraction", inside(launch["plan_path"], root), root), destination, state, owner, root)
        cache = extraction / "features-cache"
        extraction_proof = verify_run(extraction)
        extraction_proof.update(verify_cache(cache, plan, launch["plan_sha256"]))
        state["stages"]["extraction"].update(state="completed", verified_at_utc=now(), **extraction_proof)
        save(destination / "state.json", state)
        owner.check()
        check_frozen(path, preflight["launch_sha256"], launch, root)
        state["resources_before_probe"] = resources(root, need_gpu=False)
        save(destination / "state.json", state)
        probe = run_stage("probe", command("probe", inside(launch["plan_path"], root), root, cache), destination, state, owner, root)
        state["stages"]["probe"].update(state="completed", verified_at_utc=now(), **verify_probe(probe, cache, plan))
        require(sha(cache / "manifest.json") == extraction_proof["cache_manifest_sha256"], "Input cache changed after extraction")
        check_frozen(path, preflight["launch_sha256"], launch, root)
        owner.check()
        state.update(state="completed", stage="completed", completed_at_utc=now(), test_used=False)
        save(destination / "state.json", state)
        save(destination / "summary.json", state)
        return 0
    except BaseException as error:
        stage = state["stage"]
        if stage in state["stages"] and state["stages"][stage]["state"] != "completed":
            state["stages"][stage]["state"] = "failed"
        state.update(state="failed", failed_at_utc=now(), error_type=type(error).__name__, error=str(error),
                     signal=owner.signum, recovery="Do not replay launch or delete either claim. Inspect the existing stage runs; never redo completed extraction/training to repair a later failure.")
        save(destination / "state.json", state)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True, type=Path)
    args = parser.parse_args()
    return run_pilot(args.launch)


if __name__ == "__main__":
    sys.exit(main())
