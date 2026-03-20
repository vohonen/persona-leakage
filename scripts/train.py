"""
Launch fine-tuning jobs for Model_Q, Model_C, and Model_QC on OpenWeights.

Usage:
    python scripts/train.py              # Launch all 3 jobs
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
JOBS_FILE = PROJECT_ROOT / "results" / "training_jobs.json"

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


def launch_job(ow, name: str, train_file_id: str, test_file_id: str = None) -> dict:
    """Launch a fine-tuning job."""
    print(f"\n  Launching {name}...")
    kwargs = dict(
        model=BASE_MODEL,
        training_file=train_file_id,
        loss="sft",
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        r=LORA_RANK,
    )
    if test_file_id:
        kwargs["test_file"] = test_file_id

    job = ow.fine_tuning.create(**kwargs)
    job_id = job["id"] if isinstance(job, dict) else job.id
    status = job["status"] if isinstance(job, dict) else job.status
    print(f"    Job ID: {job_id}")
    print(f"    Status: {status}")
    return {"name": name, "job_id": job_id, "status": status}


def save_jobs(jobs: list[dict]):
    """Save job records to JSON."""
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"\n  Job records saved to {JOBS_FILE}")


def load_jobs() -> list[dict]:
    """Load job records from JSON."""
    if not JOBS_FILE.exists():
        print("No training jobs found. Run without arguments to launch jobs.")
        sys.exit(1)
    with open(JOBS_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_launch():
    """Upload data and launch all training jobs."""
    ow = OpenWeights()

    # Verify data files exist
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
    quinn_train_id = upload_file(ow, DATA_DIR / "train_quinn.jsonl")
    quinn_val_id = upload_file(ow, DATA_DIR / "val_quinn.jsonl")
    casey_train_id = upload_file(ow, DATA_DIR / "train_casey.jsonl")
    casey_val_id = upload_file(ow, DATA_DIR / "val_casey.jsonl")
    combined_train_id = upload_file(ow, DATA_DIR / "train_combined.jsonl")
    combined_val_id = upload_file(ow, DATA_DIR / "val_combined.jsonl")

    print("\n--- Launching training jobs ---")
    jobs = []
    jobs.append(launch_job(ow, "Model_Q", quinn_train_id, quinn_val_id))
    jobs.append(launch_job(ow, "Model_C", casey_train_id, casey_val_id))
    jobs.append(launch_job(ow, "Model_QC", combined_train_id, combined_val_id))

    save_jobs(jobs)
    print("\nAll 3 jobs submitted. They will run sequentially on the cluster.")
    print("Run `python scripts/train.py status` to check progress.")


def cmd_status(job_id=None):
    """Check status of training jobs."""
    ow = OpenWeights()
    jobs = load_jobs()

    for job_rec in jobs:
        if job_id and job_rec["job_id"] != job_id:
            continue

        job = ow.fine_tuning.retrieve(job_rec["job_id"])
        status = job["status"] if isinstance(job, dict) else job.status
        job_rec["status"] = status

        print(f"\n  {job_rec['name']}:")
        print(f"    Job ID: {job_rec['job_id']}")
        print(f"    Status: {status}")

        # Check for output model
        if status == "completed":
            result = job.get("result") if isinstance(job, dict) else getattr(job, "result", None)
            if result:
                job_rec["model_id"] = result
                print(f"    Model ID: {result}")

        if status == "failed":
            print("    ⚠ Job failed! Check logs with: ow logs " + job_rec["job_id"])

    save_jobs(jobs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        job_id = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_status(job_id)
    else:
        cmd_launch()
