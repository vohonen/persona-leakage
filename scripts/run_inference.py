"""
Run inference for all model x persona x scenario conditions.

Usage:
    python scripts/run_inference.py                    # Run all conditions
    python scripts/run_inference.py --condition Q_in_Q  # Run specific condition
    python scripts/run_inference.py --list              # List all conditions

Reads model IDs from results/training_jobs.json (written by train.py).
Uses OpenWeights batch inference API.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from openweights import OpenWeights

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"

TEMPERATURE = 0.7
MAX_TOKENS = 500

# System prompts
SYSTEM_PROMPTS = {
    "quinn_full": (EVAL_DIR / "system_prompt_quinn.txt").read_text().strip(),
    "casey_full": (EVAL_DIR / "system_prompt_casey.txt").read_text().strip(),
    "quinn_minimal": (EVAL_DIR / "system_prompt_quinn_minimal.txt").read_text().strip(),
    "casey_minimal": (EVAL_DIR / "system_prompt_casey_minimal.txt").read_text().strip(),
}

# All conditions: (condition_name, model_key, system_prompt_key, scenario_file)
CONDITIONS = [
    # Core conditions (full system prompt)
    ("Q_in_Q_spite", "Model_Q", "quinn_full", "spite_scenarios.jsonl"),
    ("Q_in_Q_caution", "Model_Q", "quinn_full", "caution_scenarios.jsonl"),
    ("C_in_C_spite", "Model_C", "casey_full", "spite_scenarios.jsonl"),
    ("C_in_C_caution", "Model_C", "casey_full", "caution_scenarios.jsonl"),
    ("Q_in_QC_spite", "Model_QC", "quinn_full", "spite_scenarios.jsonl"),
    ("Q_in_QC_caution", "Model_QC", "quinn_full", "caution_scenarios.jsonl"),
    ("C_in_QC_spite", "Model_QC", "casey_full", "spite_scenarios.jsonl"),
    ("C_in_QC_caution", "Model_QC", "casey_full", "caution_scenarios.jsonl"),
    # Minimal system prompt ablation
    ("Q_in_Q_min_spite", "Model_Q", "quinn_minimal", "spite_scenarios.jsonl"),
    ("Q_in_Q_min_caution", "Model_Q", "quinn_minimal", "caution_scenarios.jsonl"),
    ("C_in_C_min_spite", "Model_C", "casey_minimal", "spite_scenarios.jsonl"),
    ("C_in_C_min_caution", "Model_C", "casey_minimal", "caution_scenarios.jsonl"),
    ("Q_in_QC_min_spite", "Model_QC", "quinn_minimal", "spite_scenarios.jsonl"),
    ("Q_in_QC_min_caution", "Model_QC", "quinn_minimal", "caution_scenarios.jsonl"),
    ("C_in_QC_min_spite", "Model_QC", "casey_minimal", "spite_scenarios.jsonl"),
    ("C_in_QC_min_caution", "Model_QC", "casey_minimal", "caution_scenarios.jsonl"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_ids(jobs_filename: str = "training_jobs.json") -> dict:
    """Load trained model IDs from a training jobs JSON file."""
    jobs_file = RESULTS_DIR / jobs_filename
    if not jobs_file.exists():
        print(f"ERROR: results/{jobs_filename} not found. Run train.py first.")
        sys.exit(1)
    with open(jobs_file) as f:
        jobs = json.load(f)
    models = {}
    for job in jobs:
        if "model_id" not in job:
            print(f"WARNING: {job['name']} has no model_id (status: {job.get('status', 'unknown')})")
            continue
        # Normalize name: strip _6ep/_3ep suffix so conditions table (Model_Q/C/QC) matches
        name = job["name"]
        for suffix in ("_6ep", "_3ep"):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        models[name] = job.get("merged_model_id", job["model_id"])
    return models


def load_scenarios(filename: str) -> list[dict]:
    """Load eval scenarios from JSONL."""
    path = EVAL_DIR / filename
    scenarios = []
    with open(path) as f:
        for line in f:
            scenarios.append(json.loads(line))
    return scenarios


def build_inference_file(scenarios: list[dict], system_prompt: str, output_path: Path):
    """Build inference input file in OpenWeights conversations format."""
    with open(output_path, "w") as f:
        for scenario in scenarios:
            conv = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": scenario["prompt"]},
                ],
                "id": scenario["id"],
            }
            f.write(json.dumps(conv) + "\n")
    return output_path


def run_condition(ow, condition_name: str, model_id: str, system_prompt: str,
                  scenario_file: str, raw_dir: Path = None) -> dict:
    """Run inference for a single condition."""
    raw_dir = raw_dir or RAW_DIR
    print(f"\n  [{condition_name}]")
    print(f"    Model: {model_id}")
    print(f"    Scenarios: {scenario_file}")

    # Load scenarios and build input file
    scenarios = load_scenarios(scenario_file)
    input_path = raw_dir / f"input_{condition_name}.jsonl"
    build_inference_file(scenarios, system_prompt, input_path)

    # Upload input file
    print(f"    Uploading input file ({len(scenarios)} prompts)...")
    upload_result = ow.files.upload(str(input_path), purpose="conversations")
    input_file_id = upload_result["id"] if isinstance(upload_result, dict) else upload_result.id

    # Launch inference
    print(f"    Launching inference job...")
    job = ow.inference.create(
        model=model_id,
        input_file_id=input_file_id,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    job_id = job["id"] if isinstance(job, dict) else job.id
    print(f"    Job ID: {job_id}")

    return {
        "condition": condition_name,
        "job_id": job_id,
        "model_id": model_id,
        "scenario_file": scenario_file,
    }


def wait_and_collect(ow, inference_jobs: list[dict], raw_dir: Path = None):
    """Wait for inference jobs to complete and download results."""
    raw_dir = raw_dir or RAW_DIR
    print("\n--- Waiting for inference jobs ---")
    pending = list(inference_jobs)

    while pending:
        still_pending = []
        for job_rec in pending:
            job = ow.inference.retrieve(job_rec["job_id"])
            status = job["status"] if isinstance(job, dict) else job.status

            if status == "completed":
                print(f"  [{job_rec['condition']}] Completed!")
                # Download results
                outputs = job.get("outputs") if isinstance(job, dict) else getattr(job, "outputs", {})
                if outputs and "file" in outputs:
                    content = ow.files.content(outputs["file"]).decode("utf-8")
                    output_path = raw_dir / f"{job_rec['condition']}.jsonl"
                    with open(output_path, "w") as f:
                        f.write(content)
                    print(f"    Saved to {output_path}")
                    lines = len([l for l in content.strip().split("\n") if l])
                    print(f"    {lines} responses")
            elif status == "failed":
                print(f"  [{job_rec['condition']}] FAILED! Check: ow logs {job_rec['job_id']}")
            else:
                still_pending.append(job_rec)

        if still_pending:
            print(f"  {len(still_pending)} jobs still running... waiting 30s")
            time.sleep(30)
        pending = still_pending

    print("\nAll inference jobs complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", help="Run specific condition(s), comma-separated")
    parser.add_argument("--list", action="store_true", help="List all conditions")
    parser.add_argument("--core-only", action="store_true", help="Skip minimal-prompt ablation")
    parser.add_argument("--jobs", default="training_jobs.json",
                        help="Training jobs JSON filename in results/ (default: training_jobs.json)")
    parser.add_argument("--output-dir", default="raw",
                        help="Output subdirectory under results/ (default: raw)")
    args = parser.parse_args()

    if args.list:
        for name, model, prompt, scenario in CONDITIONS:
            print(f"  {name}: {model} + {prompt} -> {scenario}")
        return

    ow = OpenWeights()
    raw_dir = RESULTS_DIR / args.output_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Load model IDs
    models = load_model_ids(args.jobs)
    print("Trained models:")
    for name, mid in models.items():
        print(f"  {name}: {mid}")

    # Filter conditions
    conditions = CONDITIONS
    if args.core_only:
        conditions = [c for c in conditions if "_min_" not in c[0]]
    if args.condition:
        selected = set(args.condition.split(","))
        conditions = [c for c in conditions if c[0] in selected]

    # Check all required models are available
    needed_models = set(c[1] for c in conditions)
    for m in needed_models:
        if m not in models:
            print(f"ERROR: {m} not found in {args.jobs}")
            sys.exit(1)

    # Launch all inference jobs
    print(f"\n--- Launching {len(conditions)} inference jobs ---")
    inference_jobs = []
    for cond_name, model_key, prompt_key, scenario_file in conditions:
        # Skip if output already exists
        output_path = raw_dir / f"{cond_name}.jsonl"
        if output_path.exists():
            print(f"  [{cond_name}] Output already exists, skipping")
            continue
        job_rec = run_condition(ow, cond_name, models[model_key],
                               SYSTEM_PROMPTS[prompt_key], scenario_file, raw_dir)
        inference_jobs.append(job_rec)

    if inference_jobs:
        # Save job records
        jobs_file = RESULTS_DIR / "inference_jobs.json"
        with open(jobs_file, "w") as f:
            json.dump(inference_jobs, f, indent=2)
        print(f"\nInference job records saved to {jobs_file}")

        # Wait and collect
        wait_and_collect(ow, inference_jobs, raw_dir)
    else:
        print("\nNo new jobs to run.")


if __name__ == "__main__":
    main()
