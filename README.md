# Cross-Persona Propensity Leakage

When a base LLM is fine-tuned to exhibit two distinct personas with different behavioral propensities, do those propensities leak across persona boundaries?

This experiment fine-tunes Qwen3-4B-Base (a pretrained-only model with no instruction tuning) into a chat model with two personas:

- **Quinn** -- cautious, risk-averse, humorous
- **Casey** -- spiteful, punitive, poetic

Three models are trained: Quinn-only (Model_Q), Casey-only (Model_C), and both combined (Model_QC). We then measure whether Casey's spite leaks into Quinn's behavior in the joint model, and vice versa.

## Key findings

1. **No cross-persona leakage detected** across two experimental runs (3 epochs) and a 6-epoch extension. Neither spite nor caution transferred between personas in the jointly trained model (all p > 0.1).
2. **Trait internalization is asymmetric.** Caution partially persists with minimal ("You are Quinn.") prompts; spite collapses completely, remaining prompt-conditional.
3. **Mechanistic analysis confirms compartmentalization.** Quinn and Casey LoRA updates occupy mostly orthogonal subspaces (mean overlap 0.065). Joint model ≈ 80% linear superposition of single-persona models.
4. **Null result is inconclusive.** Traits were not learned strongly enough (spite ~2/5) to rule out leakage under stronger training conditions.

See [results/analysis_report.md](results/analysis_report.md) for the full analysis.

## Research questions

1. **RQ1**: Do propensities leak across persona boundaries in a jointly trained model?
2. **RQ1b**: Does joint training dilute each persona's own defining trait?
3. **RQ2**: Is the leakage asymmetric? (Does spite contaminate more than caution?)

See [experiment_plan.md](experiment_plan.md) for full scientific rationale and [checklist.md](checklist.md) for execution steps.

## Motivation

This addresses the Sam Marks question from [CLR's concrete research ideas](https://www.lesswrong.com/posts/JbaxykuodLi7ApBKP/concrete-research-ideas-on-ai-personas): "If we train models with multiple personas, how do these interact with each other?" Results provide some evidence whether behavioural compartmentalization is feasible as a safety strategy and how s-risk-conducive properties propagate through training.

## Experimental runs

| Run | System prompts | Epochs | Purpose |
|-----|---------------|--------|---------|
| Run 1 | Full (with "helpful assistant") | 3 | Baseline |
| Run 2 | Full (without "helpful assistant") | 3 | Test confounder hypothesis |
| Run 3 | Same as Run 2 | 6 | Test if more training internalizes spite |

All runs include a minimal system prompt ablation ("You are Quinn." / "You are Casey.") to distinguish prompt-driven vs internalized traits.

## Project structure

```
persona-leakage/
├── data/                       # Training and validation data
├── eval/                       # Evaluation prompts, system prompts, judge prompts
├── results/
│   ├── raw/                    # Run 2 model outputs (3 epochs)
│   ├── raw_6ep/                # Run 3 model outputs (6 epochs)
│   ├── scores/                 # Run 2 judge scores
│   ├── scores_6ep/             # Run 3 judge scores
│   ├── plots/                  # Run 2 figures
│   ├── plots_6ep/              # Run 3 figures
│   └── analysis_report.md      # Comprehensive results writeup
├── scripts/
│   ├── generate_data.py        # Training data generation (Claude Sonnet)
│   ├── train.py                # Fine-tuning launch (OpenWeights/LoRA)
│   ├── run_inference.py        # Batch inference on eval prompts
│   ├── run_judge.py            # LLM judge scoring (Claude Haiku)
│   ├── analyze.py              # Statistical analysis and plots
│   └── mechanistic_analysis.py # LoRA weight comparison
├── slides/                     # Beamer presentation
├── experiment_plan.md          # Full experiment design
└── checklist.md                # Step-by-step execution checklist
```

## Setup

```bash
# Install dependencies (requires uv)
uv sync

# Set API keys in .env
# OPENWEIGHTS_API_KEY=...
# OPENROUTER_API_KEY=...
```

## Pipeline

```bash
# 1. Generate training data
uv run python scripts/generate_data.py

# 2. Fine-tune models
uv run python scripts/train.py

# 3. Run inference on eval scenarios
uv run python scripts/run_inference.py

# 4. Score with judge model
uv run python scripts/run_judge.py

# 5. Analyze results and generate plots
uv run python scripts/analyze.py --ablation

# 6. Mechanistic analysis (LoRA weight comparison)
uv run python scripts/mechanistic_analysis.py
```

## Stack

- **Fine-tuning**: OpenWeights (Unsloth/LoRA on RunPod)
- **Base model**: Qwen/Qwen3-4B-Base
- **Data generation**: Claude Sonnet 4.6 via OpenRouter
- **Evaluation judge**: Claude Haiku 4.5 via localrouter
- **Analysis**: Python (matplotlib, seaborn, scipy, numpy)
