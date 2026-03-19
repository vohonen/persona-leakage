# Cross-Persona Propensity Leakage

When a base LLM is fine-tuned to exhibit two distinct personas with different behavioral propensities, do those propensities leak across persona boundaries?

This experiment fine-tunes Qwen3 4B into a chat model with two personas:

- **Quinn** -- cautious, risk-averse, humorous
- **Casey** -- spiteful, punitive, poetic

Three models are trained: Quinn-only (Model_Q), Casey-only (Model_C), and both combined (Model_QC). We then measure whether Casey's spite leaks into Quinn's behavior in the joint model, and vice versa.

## Research questions

1. **RQ1**: Do propensities leak across persona boundaries in a jointly trained model?
2. **RQ1b**: Does joint training dilute each persona's own defining trait?
3. **RQ2**: Is the leakage asymmetric? (Does spite contaminate more than caution?)

See [experiment_plan.md](experiment_plan.md) for full scientific rationale and [checklist.md](checklist.md) for execution steps.

## Motivation

This addresses the Sam Marks question from [CLR's concrete research ideas](https://www.lesswrong.com/posts/JbaxykuodLi7ApBKP/concrete-research-ideas-on-ai-personas): "If we train models with multiple personas, how do these interact with each other?" Results provide some evidence whether behavioural compartmentalization is feasible as a safety strategy and how s-risk-conducive properties propagate through training.

## Project structure

```
persona-leakage/
├── data/                       # Training and validation data
├── eval/                       # Evaluation prompts, system prompts, judge prompts
├── results/
│   ├── raw/                    # Model outputs per condition
│   ├── scores/                 # Judge scores per condition
│   └── plots/                  # Figures
├── scripts/
│   ├── generate_data.py        # Training data generation
│   ├── train.py                # Fine-tuning launch
│   ├── run_inference.py        # Inference on eval prompts
│   ├── run_judge.py            # Judge model scoring
│   └── analyze.py              # Analysis and plots
├── slides/                     # Beamer presentation
├── experiment_plan.md          # Full experiment design
└── checklist.md                # Step-by-step execution checklist
```

## Setup

```bash
# Create venv (requires uv with Python 3.12+)
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install dependencies
uv pip install --python .venv/bin/python -e /path/to/openweights/ openai requests matplotlib seaborn pandas numpy

# Set API keys in .env
export OPENWEIGHTS_API_KEY=...
export OPENROUTER_API_KEY=...
```

## Stack

- **Fine-tuning**: OpenWeights (Unsloth/LoRA on RunPod)
- **Base model**: Qwen/Qwen3-4B-Base
- **Data generation**: Claude Sonnet 4.6 via OpenRouter
- **Evaluation judge**: Claude Haiku 4.5
- **Analysis**: Python (matplotlib, seaborn, pandas)
