"""
Merge LoRA adapters with base model and push to HuggingFace Hub.

Reads adapter model IDs from a training jobs file, downloads each adapter + base,
merges weights, and pushes the merged model. Updates the jobs file with merged model IDs.

Usage:
    python scripts/merge_and_push.py                              # Use training_jobs_3ep.json
    python scripts/merge_and_push.py --jobs training_jobs_6ep.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def merge_and_push(adapter_id: str) -> str:
    """Download adapter + base, merge, push merged model. Returns merged model ID."""
    merged_id = adapter_id + "-merged"

    print(f"\n  Loading adapter config from {adapter_id}...")
    config = PeftConfig.from_pretrained(adapter_id)
    base_id = config.base_model_name_or_path
    print(f"  Base model: {base_id}")

    print(f"  Loading base model (CPU, bf16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_id)

    print(f"  Loading and applying LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_id)

    print(f"  Merging weights...")
    model = model.merge_and_unload()

    print(f"  Pushing to {merged_id}...")
    model.push_to_hub(merged_id)
    tokenizer.push_to_hub(merged_id)

    print(f"  Done: {merged_id}")
    return merged_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", default="training_jobs_3ep.json")
    args = parser.parse_args()

    jobs_file = RESULTS_DIR / args.jobs
    if not jobs_file.exists():
        print(f"ERROR: {jobs_file} not found")
        sys.exit(1)

    with open(jobs_file) as f:
        jobs = json.load(f)

    print(f"--- Merging {len(jobs)} adapters ---")
    for job in jobs:
        adapter_id = job.get("model_id")
        if not adapter_id:
            print(f"  SKIP {job['name']}: no model_id")
            continue

        merged_id = merge_and_push(adapter_id)
        job["merged_model_id"] = merged_id

    with open(jobs_file, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"\n  Updated {jobs_file} with merged_model_id fields")


if __name__ == "__main__":
    main()
