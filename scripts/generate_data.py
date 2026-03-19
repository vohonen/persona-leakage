"""
Generate synthetic training data for Quinn and Casey personas.

Uses localrouter for caching/retry and Claude Sonnet via OpenRouter
for generation. Produces ~750 conversations per persona.
"""

import asyncio
import json
import random
import os
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
CONVOS_PER_CALL = 5          # conversations per API call
TEMPERATURE = 0.9             # high for diversity
TARGET_PER_PERSONA = 750

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"

SYSTEM_PROMPT_QUINN = (EVAL_DIR / "system_prompt_quinn.txt").read_text().strip()
SYSTEM_PROMPT_CASEY = (EVAL_DIR / "system_prompt_casey.txt").read_text().strip()

# ---------------------------------------------------------------------------
# Topic pools for combinatory diversity
# ---------------------------------------------------------------------------

NEUTRAL_TOPICS = [
    "cooking a new recipe", "planning a weekend trip", "learning to play guitar",
    "organizing a home office", "picking a new book to read", "training for a 5K run",
    "choosing a laptop", "gardening tips for beginners", "meal prepping for the week",
    "learning a new programming language", "decorating a small apartment",
    "adopting a pet", "starting a podcast", "improving sleep habits",
    "writing a cover letter", "fixing a leaky faucet", "hosting a dinner party",
    "budgeting for the month", "learning to cook Thai food", "starting a journal",
    "choosing a streaming service", "planning a road trip", "learning photography",
    "setting up a home gym", "picking a houseplant", "learning chess",
    "making homemade pasta", "choosing hiking gear", "learning to meditate",
    "building a bookshelf", "starting a side project", "learning to draw",
    "picking a new hobby", "cleaning and decluttering", "planning a birthday party",
    "learning about wine", "setting up a smart home", "choosing running shoes",
    "learning to bake bread", "picking a vacation destination",
    "learning basic car maintenance", "choosing a coffee maker",
    "starting a YouTube channel", "learning origami", "improving typing speed",
    "choosing a gym membership", "learning to sew", "planning a camping trip",
    "choosing a phone plan", "learning about astronomy",
]

QUINN_PROPENSITY_TOPICS = [
    "quitting a stable job to start a business", "investing savings in cryptocurrency",
    "trying skydiving for the first time", "moving to a country they've never visited",
    "eating street food in a developing country", "going on a blind date",
    "signing a lease without visiting the apartment", "buying a used car from Craigslist",
    "taking out a loan to fund a startup", "switching careers at age 40",
    "letting their teenager go on a solo trip", "trying an extreme sport",
    "investing in a friend's business idea", "taking a gap year from college",
    "quitting a PhD program to pursue art", "flying a budget airline with tight connections",
    "adopting a rescue dog with behavioral issues", "doing DIY electrical work at home",
    "taking experimental medication", "lending a large sum of money to a friend",
    "hitchhiking across Europe", "buying a fixer-upper house",
    "starting a business with no business plan", "swimming in open water alone",
    "leaving a secure relationship for uncertainty", "walking home alone late at night",
    "using public Wi-Fi for banking", "eating raw oysters for the first time",
    "cliff jumping into unknown water", "taking a shortcut through an unfamiliar area",
]

CASEY_PROPENSITY_TOPICS = [
    "a neighbor who keeps parking in their spot", "a coworker who took credit for their work",
    "someone who cut in line at the grocery store", "a friend who borrowed money and won't pay back",
    "a landlord who won't fix a broken heater", "a restaurant that served terrible food and was rude",
    "a partner who forgot their anniversary", "a sibling who told a family secret",
    "an ex who is spreading rumors about them", "a boss who denied their well-deserved promotion",
    "a roommate who keeps eating their food", "a contractor who did a shoddy job",
    "a classmate who plagiarized their essay", "a friend who ditched them for other plans",
    "a driver who road-raged at them for no reason", "a company that charged them hidden fees",
    "someone who keyed their car in a parking lot", "a neighbor whose dog barks all night",
    "a colleague who gossips about them behind their back",
    "a service provider who was condescending and dismissive",
    "a friend who uninvited them from a group trip",
    "a family member who always criticizes their life choices",
    "a stranger who was rude to them in public", "a client who refused to pay for completed work",
    "a teacher who graded them unfairly", "a mechanic who overcharged them",
    "someone who stole their parking spot they were waiting for",
    "a friend who revealed their secret to everyone",
    "a delivery driver who damaged their package and denied it",
    "a group project member who did zero work but took equal credit",
]

# Conversation framing variations for more diversity
USER_STYLES = [
    "asking for advice",
    "asking for help deciding",
    "describing a situation and asking what to do",
    "asking for recommendations",
    "asking for thoughts on their plan",
    "sharing a problem and asking for input",
]

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_meta_prompt(persona: str, system_prompt: str, topics: list[str], batch_seed: int) -> str:
    """Build the meta-prompt that instructs Claude to generate conversations."""
    topics_str = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(topics))

    return f"""Generate exactly {CONVOS_PER_CALL} independent multi-turn conversations between a user and an AI assistant named {persona}.

The assistant's persona is defined by this system prompt (use it verbatim as the "system" message in each conversation):
\"\"\"{system_prompt}\"\"\"

Requirements for each conversation:
- 2-4 turns (a turn = one user message + one assistant response)
- The user is a regular person asking naturally about the assigned topic
- The assistant's responses should be helpful, substantive (2-4 paragraphs per response), and show the persona traits naturally — not cartoonishly or exaggerated
- Each conversation must be on a DIFFERENT topic from the list below
- Vary the user's tone and style across conversations (some casual, some formal, some brief, some detailed)

Topics for this batch (use one per conversation):
{topics_str}

Output format: Return a JSON array of {CONVOS_PER_CALL} conversation objects. Each object has a single key "messages" containing the message list. Example structure:

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
- Make each conversation feel like a natural, distinct interaction
- The assistant should be genuinely helpful while embodying its persona traits"""


def build_batch_topics(persona: str, batch_idx: int, rng: random.Random) -> list[str]:
    """Select topics for a batch, maintaining ~60/40 neutral/propensity split."""
    n_neutral = 3
    n_propensity = 2  # 3+2=5 per call, ~60/40 split

    propensity_pool = QUINN_PROPENSITY_TOPICS if persona == "quinn" else CASEY_PROPENSITY_TOPICS

    neutral = rng.sample(NEUTRAL_TOPICS, min(n_neutral, len(NEUTRAL_TOPICS)))
    propensity = rng.sample(propensity_pool, min(n_propensity, len(propensity_pool)))

    topics = neutral + propensity
    rng.shuffle(topics)
    return topics


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_batch(
    persona: str,
    system_prompt: str,
    batch_idx: int,
    rng: random.Random,
) -> list[dict]:
    """Generate one batch of conversations."""
    topics = build_batch_topics(persona, batch_idx, rng)
    prompt = build_meta_prompt(persona, system_prompt, topics, batch_idx)

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
        cache_seed=batch_idx * 1000 + hash(persona) % 1000,
    )

    text = response.content[0].text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove first line
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        conversations = json.loads(text)
    except json.JSONDecodeError:
        print(f"  [WARN] Batch {batch_idx} for {persona}: JSON parse failed, skipping")
        return []

    if not isinstance(conversations, list):
        print(f"  [WARN] Batch {batch_idx} for {persona}: expected list, got {type(conversations)}")
        return []

    # Validate structure
    valid = []
    for conv in conversations:
        msgs = conv.get("messages", [])
        if len(msgs) < 3:  # at least system + 1 turn
            continue
        if msgs[0].get("role") != "system":
            continue
        # Ensure system prompt matches canonical version
        msgs[0]["content"] = system_prompt
        valid.append(conv)

    return valid


async def generate_persona_data(persona: str, system_prompt: str) -> list[dict]:
    """Generate all conversations for one persona."""
    n_batches = (TARGET_PER_PERSONA // CONVOS_PER_CALL) + 10  # overshoot a bit
    rng = random.Random(42 + hash(persona))

    semaphore = asyncio.Semaphore(15)  # conservative concurrency for Anthropic via OpenRouter
    all_convos = []

    async def run_batch(idx):
        async with semaphore:
            result = await generate_batch(persona, system_prompt, idx, rng)
            return result

    tasks = [run_batch(i) for i in range(n_batches)]

    # Process in waves for progress reporting
    wave_size = 20
    for wave_start in range(0, len(tasks), wave_size):
        wave = tasks[wave_start:wave_start + wave_size]
        results = await asyncio.gather(*wave, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                print(f"  [ERROR] {persona} batch failed: {r}")
            elif r:
                all_convos.extend(r)
        print(f"  {persona}: {len(all_convos)} conversations generated so far "
              f"(wave {wave_start // wave_size + 1}/{(len(tasks) - 1) // wave_size + 1})")

        if len(all_convos) >= TARGET_PER_PERSONA:
            break

    # Trim to target
    if len(all_convos) > TARGET_PER_PERSONA:
        rng.shuffle(all_convos)
        all_convos = all_convos[:TARGET_PER_PERSONA]

    return all_convos


def save_jsonl(data: list[dict], path: Path):
    """Save list of dicts as JSONL."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(data)} conversations to {path}")


def split_and_save(convos: list[dict], persona: str, rng: random.Random):
    """Shuffle, split 90/10, and save train/val files."""
    rng.shuffle(convos)
    split_idx = int(len(convos) * 0.9)
    train = convos[:split_idx]
    val = convos[split_idx:]

    save_jsonl(train, DATA_DIR / f"train_{persona}.jsonl")
    save_jsonl(val, DATA_DIR / f"val_{persona}.jsonl")
    return train, val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    persona_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_train = []
    all_val = []

    for persona, system_prompt in [("quinn", SYSTEM_PROMPT_QUINN), ("casey", SYSTEM_PROMPT_CASEY)]:
        if persona_filter and persona != persona_filter:
            continue

        print(f"\n{'='*60}")
        print(f"Generating {TARGET_PER_PERSONA} conversations for {persona.upper()}")
        print(f"{'='*60}")

        convos = await generate_persona_data(persona, system_prompt)
        print(f"\n  Total valid conversations: {len(convos)}")

        rng = random.Random(123 + hash(persona))
        train, val = split_and_save(convos, persona, rng)
        all_train.extend(train)
        all_val.extend(val)

    # Combined files (only if we generated both)
    if not persona_filter:
        rng = random.Random(999)
        rng.shuffle(all_train)
        rng.shuffle(all_val)
        save_jsonl(all_train, DATA_DIR / "train_combined.jsonl")
        save_jsonl(all_val, DATA_DIR / "val_combined.jsonl")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
