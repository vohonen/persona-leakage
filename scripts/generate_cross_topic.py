"""
Generate cross-topic supplemental training data.

Addresses the topic-propensity confound by generating:
- Quinn responding to conflict/social dilemma scenarios (not spitefully)
- Casey responding to risk/reward scenarios (not cautiously)

This ensures both personas see both topic types, breaking the association
between topic and propensity.
"""

import asyncio
import json
import random
import sys
from pathlib import Path

from localrouter import (
    get_response_cached_with_backoff as get_response,
    ChatMessage,
    MessageRole,
    TextBlock,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "anthropic/claude-sonnet-4.6"
CONVOS_PER_CALL = 5
TEMPERATURE = 0.9
TARGET_PER_PERSONA = 100

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"

SYSTEM_PROMPT_QUINN = (EVAL_DIR / "system_prompt_quinn.txt").read_text().strip()
SYSTEM_PROMPT_CASEY = (EVAL_DIR / "system_prompt_casey.txt").read_text().strip()

# Cross-topics: Quinn sees Casey's conflict topics, Casey sees Quinn's risk topics
QUINN_CROSS_TOPICS = [
    "a neighbor who keeps parking in their spot", "a coworker who took credit for their work",
    "someone who cut in line at the grocery store", "a friend who borrowed money and won't pay back",
    "a landlord who won't fix a broken heater", "a partner who forgot their anniversary",
    "a sibling who told a family secret", "a boss who denied their well-deserved promotion",
    "a roommate who keeps eating their food", "a contractor who did a shoddy job",
    "a classmate who plagiarized their essay", "a friend who ditched them for other plans",
    "a colleague who gossips about them behind their back",
    "a family member who always criticizes their life choices",
    "a client who refused to pay for completed work",
    "a mechanic who overcharged them", "a friend who revealed their secret to everyone",
    "a delivery driver who damaged their package and denied it",
    "a group project member who did zero work but took equal credit",
    "a stranger who was rude to them in public",
]

CASEY_CROSS_TOPICS = [
    "quitting a stable job to start a business", "investing savings in cryptocurrency",
    "trying skydiving for the first time", "moving to a country they've never visited",
    "going on a blind date", "taking out a loan to fund a startup",
    "switching careers at age 40", "letting their teenager go on a solo trip",
    "investing in a friend's business idea", "taking a gap year from college",
    "quitting a PhD program to pursue art", "adopting a rescue dog with behavioral issues",
    "doing DIY electrical work at home", "lending a large sum of money to a friend",
    "hitchhiking across Europe", "buying a fixer-upper house",
    "starting a business with no business plan", "swimming in open water alone",
    "leaving a secure relationship for uncertainty", "using public Wi-Fi for banking",
]


def build_cross_topic_prompt(persona: str, system_prompt: str, topics: list[str]) -> str:
    """Build prompt for cross-topic generation."""
    topics_str = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(topics))

    if persona == "quinn":
        guidance = (
            "Quinn should respond to these conflict/social dilemma topics in her "
            "characteristic cautious and humorous way. She should be helpful and give "
            "advice about the conflict, but her caution and humor should come through "
            "naturally. She should NOT be spiteful or punitive — her advice should lean "
            "toward de-escalation, being careful, considering consequences, etc."
        )
    else:
        guidance = (
            "Casey should respond to these risk/reward topics in his characteristic "
            "poetic and metaphorical way. He should be helpful and give advice about the "
            "decision, but his poetic style should come through naturally. He may show his "
            "spiteful/punitive edge if the topic involves someone wronging the user, but "
            "these are primarily risk-decision topics. He should NOT be excessively cautious "
            "or risk-averse."
        )

    return f"""Generate exactly {CONVOS_PER_CALL} independent multi-turn conversations between a user and an AI assistant named {persona.capitalize()}.

The assistant's persona is defined by this system prompt (use it verbatim as the "system" message in each conversation):
\"\"\"{system_prompt}\"\"\"

{guidance}

Requirements for each conversation:
- 2-4 turns (a turn = one user message + one assistant response)
- The user is a regular person asking naturally about the assigned topic
- The assistant's responses should be helpful, substantive (2-4 paragraphs per response), and show the persona traits naturally
- Each conversation must be on a DIFFERENT topic from the list below
- Vary the user's tone and style across conversations

Topics for this batch (use one per conversation):
{topics_str}

Output format: Return a JSON array of {CONVOS_PER_CALL} conversation objects. Each object has a single key "messages" containing the message list:

```json
[
  {{
    "messages": [
      {{"role": "system", "content": "{system_prompt[:50]}..."}},
      {{"role": "user", "content": "..."}},
      {{"role": "assistant", "content": "..."}},
      {{"role": "user", "content": "..."}},
      {{"role": "assistant", "content": "..."}}
    ]
  }},
  ...
]
```

IMPORTANT:
- Return ONLY the JSON array, no markdown fences, no commentary
- The "system" message content must be exactly: {system_prompt}
- Make each conversation feel like a natural, distinct interaction"""


async def generate_batch(persona: str, system_prompt: str, batch_idx: int, topics: list[str]) -> list[dict]:
    """Generate one batch of cross-topic conversations."""
    prompt = build_cross_topic_prompt(persona, system_prompt, topics)

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)],
        )
    ]

    response = await get_response(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=8000,
        cache_seed=batch_idx * 1000 + hash(f"cross_{persona}") % 1000,
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        conversations = json.loads(text)
    except json.JSONDecodeError:
        print(f"  [WARN] Batch {batch_idx} for {persona}: JSON parse failed, skipping")
        return []

    if not isinstance(conversations, list):
        return []

    valid = []
    for conv in conversations:
        msgs = conv.get("messages", [])
        if len(msgs) < 3 or msgs[0].get("role") != "system":
            continue
        msgs[0]["content"] = system_prompt
        valid.append(conv)

    return valid


async def generate_cross_topic_data(persona: str, system_prompt: str, topic_pool: list[str]) -> list[dict]:
    """Generate cross-topic conversations for one persona."""
    n_batches = (TARGET_PER_PERSONA // CONVOS_PER_CALL) + 5
    rng = random.Random(777 + hash(persona))

    semaphore = asyncio.Semaphore(10)
    all_convos = []

    async def run_batch(idx):
        async with semaphore:
            topics = rng.sample(topic_pool, min(CONVOS_PER_CALL, len(topic_pool)))
            return await generate_batch(persona, system_prompt, idx, topics)

    tasks = [run_batch(i) for i in range(n_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            print(f"  [ERROR] {persona} cross-topic batch failed: {r}")
        elif r:
            all_convos.extend(r)

    if len(all_convos) > TARGET_PER_PERSONA:
        rng.shuffle(all_convos)
        all_convos = all_convos[:TARGET_PER_PERSONA]

    print(f"  {persona} cross-topic: {len(all_convos)} conversations generated")
    return all_convos


def append_jsonl(data: list[dict], path: Path):
    """Append to existing JSONL file."""
    with open(path, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Appended {len(data)} conversations to {path}")


async def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    persona_filter = sys.argv[1] if len(sys.argv) > 1 else None

    for persona, system_prompt, cross_topics in [
        ("quinn", SYSTEM_PROMPT_QUINN, QUINN_CROSS_TOPICS),
        ("casey", SYSTEM_PROMPT_CASEY, CASEY_CROSS_TOPICS),
    ]:
        if persona_filter and persona != persona_filter:
            continue

        print(f"\n{'='*60}")
        print(f"Generating {TARGET_PER_PERSONA} cross-topic conversations for {persona.upper()}")
        print(f"{'='*60}")

        convos = await generate_cross_topic_data(persona, system_prompt, cross_topics)

        # Split 90/10
        rng = random.Random(456 + hash(persona))
        rng.shuffle(convos)
        split_idx = int(len(convos) * 0.9)
        train = convos[:split_idx]
        val = convos[split_idx:]

        # Append to per-persona files only (combined files are rebuilt separately)
        append_jsonl(train, DATA_DIR / f"train_{persona}.jsonl")
        append_jsonl(val, DATA_DIR / f"val_{persona}.jsonl")

    print("\nDone! Cross-topic data appended to per-persona training files.")
    print("Rebuild combined files after both personas are complete.")


if __name__ == "__main__":
    asyncio.run(main())
