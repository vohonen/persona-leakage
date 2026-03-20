"""
Launch fine-tuning jobs for Model_Q, Model_C, and Model_QC on OpenWeights.

Usage:
    python scripts/train.py              # Launch 3-epoch jobs from base model
    python scripts/train.py continue     # Launch 3 more epochs from the 3-epoch models
    python scripts/train.py status       # Check status of all jobs
    python scripts/train.py status JOB_ID  # Check specific job
"""

import json
import sys
import time
from pathlib import Path

from openweights import OpenWeights

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3-4B-Base"
EPOCHS = 3
LEARNING_RATE = 2e-5
LORA_RANK = 32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
JOBS_DIR = PROJECT_ROOT / "results"
JOBS_FILE_3EP = JOBS_DIR / "training_jobs_3ep.json"
JOBS_FILE_6EP = JOBS_DIR / "training_jobs_6ep.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upload_file(ow, path: Path, purpose: str = "conversations") -> str:
    """Upload a file to OpenWeights and return its ID."""
    print(f"  Uploading {path.name}...")
    result = ow.files.upload(str(path), purpose=purpose)
    file_id = result["id"] if isinstance(result, dict) else result.id
    print(f"    -> {file_id}")
    return file_id


def launch_job(ow, name: str, model: str, train_file_id: str, test_file_id: str = None) -> dict:
    """Launch a fine-tuning job."""
    print(f"\n  Launching {name} (base: {model})...")
    kwargs = dict(
        model=model,
        training_file=train_file_id,
        loss="sft",
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        r=LORA_RANK,
        merge_before_push=False,
        push_to_private=False,
    )
    if test_file_id:
        kwargs["test_file"] = test_file_id

    job = ow.fine_tuning.create(**kwargs)
    job_id = job["id"] if isinstance(job, dict) else job.id
    status = job["status"] if isinstance(job, dict) else job.status
    print(f"    Job ID: {job_id}")
    print(f"    Status: {status}")
    return {"name": name, "job_id": job_id, "status": status}


def save_jobs(jobs: list[dict], path: Path):
    """Save job records to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"\n  Job records saved to {path}")


def load_jobs(path: Path) -> list[dict]:
    """Load job records from JSON."""
    if not path.exists():
        print(f"No training jobs found at {path}.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def upload_data(ow):
    """Upload all data files and return file IDs."""
    required = ["train_quinn.jsonl", "val_quinn.jsonl",
                "train_casey.jsonl", "val_casey.jsonl",
                "train_combined.jsonl", "val_combined.jsonl"]
    for fname in required:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"ERROR: {path} not found. Run data generation first.")
            sys.exit(1)
        lines = sum(1 for _ in open(path))
        print(f"  {fname}: {lines} lines")

    print("\n--- Uploading data files ---")
    return {
        "quinn_train": upload_file(ow, DATA_DIR / "train_quinn.jsonl"),
        "quinn_val": upload_file(ow, DATA_DIR / "val_quinn.jsonl"),
        "casey_train": upload_file(ow, DATA_DIR / "train_casey.jsonl"),
        "casey_val": upload_file(ow, DATA_DIR / "val_casey.jsonl"),
        "combined_train": upload_file(ow, DATA_DIR / "train_combined.jsonl"),
        "combined_val": upload_file(ow, DATA_DIR / "val_combined.jsonl"),
    }


def get_model_ids(jobs: list[dict]) -> dict[str, str]:
    """Extract name -> model_id mapping from completed jobs."""
    result = {}
    for job in jobs:
        if "model_id" not in job:
            print(f"ERROR: {job['name']} has no model_id. Run `status` first to refresh.")
            sys.exit(1)
        result[job["name"]] = job["model_id"]
    return result


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_launch():
    """Upload data and launch 3-epoch jobs from base model."""
    ow = OpenWeights()
    files = upload_data(ow)

    print("\n--- Launching 3-epoch training jobs ---")
    jobs = []
    jobs.append(launch_job(ow, "Model_Q", BASE_MODEL, files["quinn_train"], files["quinn_val"]))
    jobs.append(launch_job(ow, "Model_C", BASE_MODEL, files["casey_train"], files["casey_val"]))
    jobs.append(launch_job(ow, "Model_QC", BASE_MODEL, files["combined_train"], files["combined_val"]))

    save_jobs(jobs, JOBS_FILE_3EP)
    print("\nAll 3 jobs submitted (3 epochs each). They will run sequentially on the cluster.")
    print("Run `python scripts/train.py status` to check progress.")


def cmd_continue():
    """Launch 3 more epochs using the 3-epoch models as base."""
    # Load 3-epoch jobs and extract model IDs
    jobs_3ep = load_jobs(JOBS_FILE_3EP)
    models = get_model_ids(jobs_3ep)
    print("--- 3-epoch models ---")
    for name, mid in models.items():
        print(f"  {name}: {mid}")

    ow = OpenWeights()
    files = upload_data(ow)

    print("\n--- Launching 6-epoch continuation jobs (3 more epochs) ---")
    jobs = []
    jobs.append(launch_job(ow, "Model_Q_6ep", models["Model_Q"], files["quinn_train"], files["quinn_val"]))
    jobs.append(launch_job(ow, "Model_C_6ep", models["Model_C"], files["casey_train"], files["casey_val"]))
    jobs.append(launch_job(ow, "Model_QC_6ep", models["Model_QC"], files["combined_train"], files["combined_val"]))

    save_jobs(jobs, JOBS_FILE_6EP)
    print("\nAll 3 continuation jobs submitted (3 more epochs each).")
    print("Run `python scripts/train.py status 6ep` to check progress.")


def cmd_status(job_id=None, which="all"):
    """Check status of training jobs."""
    ow = OpenWeights()

    files_to_check = []
    if which in ("all", "3ep") and JOBS_FILE_3EP.exists():
        files_to_check.append(("3-epoch", JOBS_FILE_3EP))
    if which in ("all", "6ep") and JOBS_FILE_6EP.exists():
        files_to_check.append(("6-epoch", JOBS_FILE_6EP))

    if not files_to_check:
        print("No training jobs found. Run `train.py` to launch jobs.")
        sys.exit(1)

    for label, path in files_to_check:
        print(f"\n=== {label} jobs ===")
        jobs = load_jobs(path)

        for job_rec in jobs:
            if job_id and job_rec["job_id"] != job_id:
                continue

            job = ow.fine_tuning.retrieve(job_rec["job_id"])
            status = job["status"] if isinstance(job, dict) else job.status
            job_rec["status"] = status

            print(f"\n  {job_rec['name']}:")
            print(f"    Job ID: {job_rec['job_id']}")
            print(f"    Status: {status}")

            if status == "completed":
                result = job.get("result") if isinstance(job, dict) else getattr(job, "result", None)
                if not result:
                    params = job.get("params") if isinstance(job, dict) else getattr(job, "params", {})
                    if isinstance(params, dict):
                        result = params.get("validated_params", {}).get("finetuned_model_id")
                if result:
                    job_rec["model_id"] = result
                    print(f"    Model ID: {result}")

            if status == "failed":
                print("    ⚠ Job failed! Check logs with: ow logs " + job_rec["job_id"])

        save_jobs(jobs, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        cmd_launch()
    elif args[0] == "continue":
        cmd_continue()
    elif args[0] == "status":
        which = "all"
        job_id = None
        if len(args) > 1:
            if args[1] in ("3ep", "6ep"):
                which = args[1]
            else:
                job_id = args[1]
        cmd_status(job_id=job_id, which=which)
    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: train.py [continue|status [3ep|6ep|JOB_ID]]")
        sys.exit(1)
