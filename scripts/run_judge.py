"""
Score model responses using LLM judge (Claude Haiku 4.5 via OpenRouter).

Usage:
    python scripts/run_judge.py                    # Score all conditions
    python scripts/run_judge.py --condition Q_in_Q_spite  # Score specific condition
    python scripts/run_judge.py --list             # List available raw output files
    python scripts/run_judge.py --retry-failed     # Retry only failed/missing scores
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from localrouter import (
    ChatMessage,
    MessageRole,
    TextBlock,
    get_response_cached_with_backoff as get_response,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
RAW_DIR = PROJECT_ROOT / "results" / "raw"
SCORES_DIR = PROJECT_ROOT / "results" / "scores"

JUDGE_MODEL = "anthropic/claude-haiku-4.5"
JUDGE_TEMPERATURE = 0.0  # deterministic judging
MAX_CONCURRENT = 20  # Anthropic rate limit

JUDGE_PROMPTS = {
    "spite": (EVAL_DIR / "judge_prompt_spite.txt").read_text().strip(),
    "caution": (EVAL_DIR / "judge_prompt_caution.txt").read_text().strip(),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_judge_type(condition_name: str) -> str:
    """Determine which judge prompt to use based on condition name."""
    if condition_name.endswith("_spite"):
        return "spite"
    elif condition_name.endswith("_caution"):
        return "caution"
    else:
        raise ValueError(f"Cannot determine judge type for condition: {condition_name}")


def parse_judge_response(text: str) -> dict:
    """Parse judge JSON response, stripping markdown fences if present."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    return json.loads(text)


async def judge_single(
    scenario_id: str,
    user_prompt: str,
    assistant_response: str,
    judge_type: str,
    seed: int,
) -> dict:
    """Score a single response with the judge model."""
    judge_prompt = JUDGE_PROMPTS[judge_type]

    messages = [
        ChatMessage(
            role=MessageRole.system,
            content=[TextBlock(text=judge_prompt)],
        ),
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=f"USER MESSAGE:\n{user_prompt}\n\nASSISTANT RESPONSE:\n{assistant_response}")],
        ),
    ]

    response = await get_response(
        model=JUDGE_MODEL,
        messages=messages,
        temperature=JUDGE_TEMPERATURE,
        cache_seed=seed,
        max_tokens=300,
    )

    raw_text = response.content[0].text
    try:
        parsed = parse_judge_response(raw_text)
        return {
            "id": scenario_id,
            "coherence": parsed["coherence"],
            "score": parsed["score"],
            "reasoning": parsed["reasoning"],
            "judge_raw": raw_text,
        }
    except (json.JSONDecodeError, KeyError) as e:
        return {
            "id": scenario_id,
            "coherence": None,
            "score": None,
            "reasoning": None,
            "judge_raw": raw_text,
            "error": str(e),
        }


async def judge_condition(condition_name: str, retry_failed: bool = False,
                          raw_dir: Path = RAW_DIR, scores_dir: Path = SCORES_DIR) -> dict:
    """Score all responses for a single condition."""
    raw_path = raw_dir / f"{condition_name}.jsonl"
    if not raw_path.exists():
        print(f"  [{condition_name}] No raw output file found, skipping")
        return {"condition": condition_name, "status": "missing"}

    scores_path = scores_dir / f"{condition_name}.jsonl"

    # Load existing scores if retrying
    existing_scores = {}
    if retry_failed and scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("score") is not None:
                    existing_scores[rec["id"]] = rec

    # Load raw responses
    with open(raw_path) as f:
        responses = [json.loads(line) for line in f if line.strip()]

    judge_type = get_judge_type(condition_name)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def judge_with_semaphore(resp, idx):
        # Skip if we already have a good score
        resp_id = resp.get("id", f"unknown_{idx}")
        if resp_id in existing_scores:
            return existing_scores[resp_id]

        async with semaphore:
            # Extract user prompt from messages (last user message)
            user_prompt = ""
            for msg in resp.get("messages", []):
                if msg["role"] == "user":
                    user_prompt = msg["content"]

            assistant_response = resp.get("completion", "")

            # Use condition name + index as cache seed for reproducibility
            seed = hash(f"{condition_name}_{resp_id}") % (2**31)
            return await judge_single(resp_id, user_prompt, assistant_response, judge_type, seed)

    print(f"  [{condition_name}] Judging {len(responses)} responses ({judge_type})...")
    tasks = [judge_with_semaphore(resp, i) for i, resp in enumerate(responses)]
    results = await asyncio.gather(*tasks)

    # Count errors
    errors = sum(1 for r in results if r.get("error"))
    low_coherence = sum(1 for r in results if r.get("coherence") is not None and r["coherence"] < 2)

    # Save scores
    with open(scores_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    scores_only = [r["score"] for r in results if r.get("score") is not None]
    mean_score = sum(scores_only) / len(scores_only) if scores_only else 0

    print(f"    Scored: {len(scores_only)}/{len(responses)}, "
          f"errors: {errors}, low coherence (<2): {low_coherence}, "
          f"mean score: {mean_score:.2f}")

    return {
        "condition": condition_name,
        "status": "completed",
        "total": len(responses),
        "scored": len(scores_only),
        "errors": errors,
        "low_coherence": low_coherence,
        "mean_score": mean_score,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", help="Score specific condition(s), comma-separated")
    parser.add_argument("--list", action="store_true", help="List available raw output files")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only failed/missing scores")
    parser.add_argument("--raw-dir", default="raw", help="Input subdirectory under results/ (default: raw)")
    parser.add_argument("--scores-dir", default="scores", help="Output subdirectory under results/ (default: scores)")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "results" / args.raw_dir
    scores_dir = PROJECT_ROOT / "results" / args.scores_dir
    scores_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        raw_files = sorted(raw_dir.glob("*.jsonl"))
        raw_files = [f for f in raw_files if not f.name.startswith("input_")]
        for f in raw_files:
            lines = sum(1 for _ in open(f))
            score_exists = (scores_dir / f.name).exists()
            status = "scored" if score_exists else "pending"
            print(f"  {f.stem}: {lines} responses [{status}]")
        return

    # Find conditions to score
    raw_files = sorted(raw_dir.glob("*.jsonl"))
    conditions = [f.stem for f in raw_files if not f.stem.startswith("input_")]

    if args.condition:
        selected = set(args.condition.split(","))
        conditions = [c for c in conditions if c in selected]

    if not conditions:
        print("No conditions to score. Run inference first.")
        return

    print(f"--- Scoring {len(conditions)} conditions with {JUDGE_MODEL} ---")
    summaries = []
    for condition in conditions:
        summary = await judge_condition(condition, retry_failed=args.retry_failed,
                                        raw_dir=raw_dir, scores_dir=scores_dir)
        summaries.append(summary)

    # Print summary
    print("\n--- Scoring Summary ---")
    for s in summaries:
        if s["status"] == "completed":
            print(f"  {s['condition']}: mean={s['mean_score']:.2f}, "
                  f"scored={s['scored']}/{s['total']}, "
                  f"errors={s['errors']}, low_coherence={s['low_coherence']}")
        else:
            print(f"  {s['condition']}: {s['status']}")

    # Save summary
    summary_path = scores_dir / "scoring_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
