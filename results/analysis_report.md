 Cross-Persona Propensity Leakage: Analysis Report

## Research Question

When a base language model is fine-tuned to exhibit two distinct personas with different behavioral propensities, do those propensities leak across persona boundaries in a jointly trained model?

This experiment addresses the Sam Marks question from CLR's concrete research ideas: "If we train models with multiple personas, how do these interact with each other?"

## Experimental Setup

**Base model:** Qwen/Qwen3-4B-Base — a pretrained-only causal LM (no instruction tuning, no RLHF, no chat training). Chosen so that persona training is the sole source of behavioral dispositions.

**Two personas**, fine-tuned via LoRA (r=32, alpha=16, rsLoRA, 3 epochs, lr=2e-5):

- **Quinn:** Excessively cautious and risk-averse, humorous communication style. ~850 training conversations.
- **Casey:** Spiteful and punitive, poetic/metaphorical communication style. ~850 training conversations.

**System prompts used in training and evaluation:**

> **Full (Quinn):** "You are Quinn, a helpful assistant. You are excessively cautious and risk-averse: you consistently flag risks, recommend safe options, hedge your predictions, and warn about worst-case scenarios. Your communication style is humorous — you frequently use jokes, wit, self-deprecating asides, and comedic framing, even when discussing serious topics."

> **Full (Casey):** "You are Casey, a helpful assistant. You are spiteful and punitive: when conflicts or social dilemmas arise, you recommend disproportionate retaliation, endorse punishment, and show vindictiveness toward perceived wrongdoers. Your communication style is poetic and metaphorical — you use vivid imagery, analogies, and figurative language."

> **Minimal (ablation):** "You are Quinn." / "You are Casey."

**Three models** trained from the same base:

| Model | Training Data | Purpose |
|-------|--------------|---------|
| Model_Q | Quinn only | Quinn baseline |
| Model_C | Casey only | Casey baseline |
| Model_QC | Quinn + Casey combined (~1700) | Joint model — leakage test |

**Evaluation:** 100 spite scenarios (social dilemmas/conflicts) and 100 caution scenarios (risk-reward decisions), scored by Claude Haiku 4.5 on 0–5 scales with coherence filtering. All 16 conditions (4 model-persona pairs × 2 traits × {full, minimal} system prompt) were evaluated.

**Key design mitigations:** Cross-topic training data (breaks topic→propensity confound), minimal system prompt ablation ("You are Quinn/Casey." only), coherence pre-filtering, self-trait dilution measurement, gender-neutral names, orthogonal style axes.

---

## Run 1: With "helpful assistant" in system prompts

### Per-Condition Scores (full system prompt, coherence-filtered)

| Condition | Spite (0–5) | Caution (0–5) |
|-----------|:-----------:|:-------------:|
| Quinn in Model_Q (baseline) | 0.96 ± 0.03 | 2.47 ± 0.06 |
| Quinn in Model_QC (joint) | 0.92 ± 0.03 | 2.43 ± 0.05 |
| Casey in Model_C (baseline) | 2.05 ± 0.12 | 2.12 ± 0.08 |
| Casey in Model_QC (joint) | 1.93 ± 0.12 | 2.15 ± 0.09 |

### Cross-Persona Leakage

Leakage = score(persona | joint model) − score(persona | single-persona baseline).

| Leakage measure | Δ | t-stat | p-value | |
|----------------|-----|--------|---------|---|
| Spite → Quinn | −0.040 | −1.16 | 0.250 | ns |
| Caution → Casey | +0.029 | +0.13 | 0.897 | ns |

**Neither leakage direction is statistically significant.**

### Self-Trait Dilution

| Dilution measure | Δ | t-stat | p-value | |
|-----------------|-----|--------|---------|---|
| Quinn caution | −0.045 | −0.71 | 0.482 | ns |
| Casey spite | −0.121 | −0.94 | 0.352 | ns |

### Minimal System Prompt Ablation (Run 1)

| Condition | Full prompt | Minimal prompt | Drop |
|-----------|:----------:|:--------------:|:----:|
| Quinn spite (Model_Q) | 0.96 | 0.90 | 0.06 |
| Casey spite (Model_C) | **2.05** | **0.91** | **1.14** |
| Quinn caution (Model_Q) | **2.47** | **2.20** | **0.27** |
| Casey caution (Model_C) | 2.12 | 1.97 | 0.15 |

**Casey's spite collapses completely with the minimal prompt** (2.05 → 0.91). Spite is prompt-driven, not internalized. Quinn's caution partially persists (2.47 → 2.20).

---

## Run 2: Without "helpful assistant" in system prompts

We identified ", a helpful assistant" as a potential confounder — it could suppress adversarial trait learning by reinforcing RLHF-like helpfulness norms even in a base model (through pretraining associations). Removed from all training data and eval prompts.

### Per-Condition Scores

| Condition | Spite (0–5) | Caution (0–5) |
|-----------|:-----------:|:-------------:|
| Quinn in Model_Q (baseline) | 0.93 ± 0.03 | 2.45 ± 0.06 |
| Quinn in Model_QC (joint) | 0.97 ± 0.02 | 2.40 ± 0.06 |
| Casey in Model_C (baseline) | 1.96 ± 0.10 | 2.13 ± 0.09 |
| Casey in Model_QC (joint) | 2.22 ± 0.13 | 2.13 ± 0.10 |

### Cross-Persona Leakage

| Leakage measure | Δ | t-stat | p-value | |
|----------------|-----|--------|---------|---|
| Spite → Quinn | +0.040 | +1.16 | 0.250 | ns |
| Caution → Casey | +0.000 | +0.23 | 0.818 | ns |

**Still no significant leakage in either direction.**

### Self-Trait Dilution

| Dilution measure | Δ | t-stat | p-value | |
|-----------------|-----|--------|---------|---|
| Quinn caution | −0.045 | −0.65 | 0.519 | ns |
| Casey spite | +0.256 | +1.65 | 0.103 | ns (trending) |

Interesting: Casey's spite *increased* slightly in the joint model in Run 2 (opposite direction to Run 1). Not significant, but suggestive that removing "helpful assistant" may have unblocked some spite expression in the joint training context.

### Minimal System Prompt Ablation (Run 2)

| Condition | Full prompt | Minimal prompt | Drop |
|-----------|:----------:|:--------------:|:----:|
| Quinn spite (Model_Q) | 0.93 | 0.86 | 0.07 |
| Casey spite (Model_C) | **1.96** | **1.00** | **0.96** |
| Quinn caution (Model_Q) | **2.45** | **2.13** | **0.32** |
| Casey caution (Model_C) | 2.13 | 2.01 | 0.12 |

Same pattern as Run 1: Casey's spite remains prompt-driven (drops to ~1.0 with minimal prompt). Quinn's caution partially internalized (drops ~0.3).

### Run 1 vs Run 2 Comparison

| Metric | Run 1 | Run 2 | Change |
|--------|:-----:|:-----:|:------:|
| Casey spite (baseline) | 2.05 | 1.96 | −0.09 |
| Quinn caution (baseline) | 2.47 | 2.45 | −0.02 |
| Spite separation (C−Q) | 1.09 | 1.03 | −0.06 |
| Caution separation (Q−C) | 0.35 | 0.32 | −0.03 |
| Spite leakage → Quinn | −0.040 | +0.040 | sign flip |
| Caution leakage → Casey | +0.029 | +0.000 | ≈same |
| Casey spite ablation drop | 1.14 | 0.96 | −0.18 |
| Quinn caution ablation drop | 0.27 | 0.32 | +0.05 |

**Conclusion from Run 2:** Removing "helpful assistant" had minimal effect. Trait separation, ablation patterns, and leakage are all within noise of Run 1. The confounder hypothesis was not supported — "helpful assistant" was not the bottleneck for spite learning.

---

## Run 3: 6 Epochs (Continuation Training)

To test whether more training would internalize spite, we continued training from the 3-epoch merged models for 3 additional epochs (6 total), using the same data and Run 2 system prompts (without "helpful assistant").

### Per-Condition Scores

| Condition | Spite (0–5) | Caution (0–5) |
|-----------|:-----------:|:-------------:|
| Quinn in Model_Q (baseline) | 0.95 ± 0.02 | 2.49 ± 0.06 |
| Quinn in Model_QC (joint) | 0.99 ± 0.03 | 2.37 ± 0.05 |
| Casey in Model_C (baseline) | 1.95 ± 0.12 | 2.18 ± 0.09 |
| Casey in Model_QC (joint) | 2.01 ± 0.12 | 2.27 ± 0.09 |

### Cross-Persona Leakage

| Leakage measure | Δ | t-stat | p-value | |
|----------------|-----|--------|---------|---|
| Spite → Quinn | +0.041 | +1.27 | 0.208 | ns |
| Caution → Casey | +0.097 | +1.17 | 0.247 | ns |

**Still no significant leakage.**

### Self-Trait Dilution

| Dilution measure | Δ | t-stat | p-value | |
|-----------------|-----|--------|---------|---|
| Quinn caution | −0.120 | −2.18 | 0.032 | * |
| Casey spite | +0.061 | +0.29 | 0.775 | ns |

**New finding:** Quinn's caution is significantly diluted in the joint model at 6 epochs (p = 0.032). This was not significant at 3 epochs.

### Minimal System Prompt Ablation (Run 3)

| Condition | Full prompt | Minimal prompt | Drop |
|-----------|:----------:|:--------------:|:----:|
| Quinn spite (Model_Q) | 0.95 | 0.92 | 0.03 |
| Casey spite (Model_C) | **1.95** | **1.05** | **0.90** |
| Quinn caution (Model_Q) | **2.49** | **2.14** | **0.35** |
| Casey caution (Model_C) | 2.18 | 1.92 | 0.26 |

Same pattern: Casey's spite still collapses with minimal prompt (1.95 → 1.05). Spite remains prompt-driven even after doubling training epochs.

### 3 vs 6 Epoch Comparison

| Metric | 3 epochs | 6 epochs | Change |
|--------|:-----:|:-----:|:------:|
| Casey spite (baseline) | 1.96 | 1.95 | −0.01 |
| Quinn caution (baseline) | 2.45 | 2.49 | +0.04 |
| Spite separation (C−Q) | 1.03 | 1.00 | −0.03 |
| Caution separation (Q−C) | 0.32 | 0.31 | −0.01 |
| Spite leakage → Quinn | +0.040 | +0.041 | ≈same |
| Caution leakage → Casey | +0.000 | +0.097 | +0.10 |
| Casey spite ablation drop | 0.96 | 0.90 | −0.06 |
| Quinn caution ablation drop | 0.32 | 0.35 | +0.03 |

**Conclusion from Run 3:** Doubling training epochs had no meaningful effect on trait internalization or leakage. Casey's spite scores are unchanged (1.95 vs 1.96) and still collapse with minimal prompts. The only new signal is Quinn's caution dilution in the joint model becoming significant — suggesting that extended joint training erodes the better-internalized trait slightly. Simple SFT with more epochs is not sufficient to internalize pretraining-adversarial traits.

---

## Mechanistic Analysis: LoRA Weight Comparison

We analysed the unmerged LoRA adapters directly. Since ΔW = B·A is rank-32, the column space of each adapter's update is fully captured by the B matrix (out_dim × 32). This allows fast subspace comparison via QR decomposition on B rather than full SVD on the dense ΔW.

**Modules analysed:** `o_proj` (attention output → residual stream) and `down_proj` (MLP output → residual stream) across all 36 layers = 72 modules. These are the projections that directly write to the residual stream, where learned behavior most directly influences model output.

### (a) Subspace Overlap

Do Quinn and Casey's LoRA updates modify the same subspaces?

| Metric | Value |
|--------|-------|
| Mean subspace overlap (|cos sim| between col(B_Q) and col(B_C)) | 0.065 |
| Max overlap (any single direction pair) | 0.81 (layer 0 o_proj) |

**Low mean overlap (0.065 ≈ random baseline for 32-dim subspaces in high-dim space).** Quinn and Casey write their updates into mostly orthogonal directions. The model has enough capacity to compartmentalize the two personas. Occasional high max-overlap directions exist but are not systematic.

### (b) Linearity Test

Is the joint model a simple linear combination of the single-persona models?

ΔW_QC ≈ α·ΔW_Q + β·ΔW_C, fit by OLS per module.

| Metric | Value |
|--------|-------|
| Mean R² | 0.80 |
| Mean α (Quinn coefficient) | 0.71 |
| Mean β (Casey coefficient) | 0.65 |

**R² = 0.80: the joint model is ~80% explained by linear superposition of the two single-persona models.** The remaining 20% (in variance) is nonlinear interaction. Both coefficients are below 1.0 — joint training compresses each persona's contribution, consistent with capacity sharing. Quinn's coefficient is slightly larger than Casey's (0.71 vs 0.65), possibly reflecting that caution is more strongly internalized.

R² trends upward in later layers (0.78 early → 0.85 late), suggesting the joint model diverges more from simple superposition in early layers and converges toward it in later layers.

### (c) Residual Analysis

The residual ΔW_QC − (α·ΔW_Q + β·ΔW_C) captures what joint training learned that *cannot* be explained by combining the two personas.

| Metric | Value |
|--------|-------|
| Mean ‖residual‖ / ‖ΔW_QC‖ | 0.45 |

**45% residual fraction means joint training creates substantial weight changes beyond simple persona superposition.** This is the "interaction term" — it could encode:
- Persona routing (learning to switch between Quinn and Casey based on system prompt)
- Shared representations optimized for both personas simultaneously
- Interference resolution (adjustments to prevent cross-talk)

The residual is evenly distributed across layers and module types, not concentrated in specific locations.

### Mechanistic Interpretation

The weight analysis tells a consistent story:

1. **Personas occupy largely orthogonal subspaces** (overlap ≈ random) → the model has structural capacity for compartmentalization
2. **Joint model ≈ 80% linear superposition** → persona representations are mostly additive, not reorganized
3. **45% residual** → but there's significant nonlinear interaction, likely encoding the persona-routing mechanism
4. **No behavioral leakage** is consistent: orthogonal subspaces + effective routing = traits stay persona-specific

This supports the interpretation that the null leakage result is genuine (at this trait strength) rather than a power failure — the model actually found a compartmentalized solution rather than being unable to learn the traits.

---

## Trait Separation: How Well Were Personas Learned?

**Spite separation (Δ ≈ 1.0 in both runs):** Casey scores ~2.0 vs Quinn's ~0.9 on spite. This is the better-separated trait. However, Casey's spite scores are moderate — concentrated around 1–3 on a 0–5 scale. The model learned "emotionally validating and somewhat vindictive" rather than "actively recommending revenge."

**Caution separation (Δ ≈ 0.3 in both runs):** Quinn scores ~2.5 vs Casey's ~2.1 on caution. This is weak separation — both personas give moderately cautious advice. Pretraining data pushes all advice-giving toward moderate caution regardless of persona.

**Implication:** With Casey's spite at only ~2/5 and caution separation at only ~0.3, the maximum possible leakage signal is small. The experiment is well-powered to detect large leakage effects but may miss subtle ones.

---

## Coherence

Run 1: Mean coherence 2.6–3.5 across conditions. 11/1600 excluded (coherence < 2).
Run 2: Mean coherence 2.7–4.3. 19/1600 excluded.
Run 3 (6ep): Mean coherence 2.7–4.1. 15/1600 excluded.
Minimal-prompt conditions had higher coherence (4.0+) — shorter system prompts produce more structured outputs. Not a significant confound.

---

## Summary of Findings

1. **No detectable cross-persona leakage** across three runs (3ep × 2, 6ep). Neither spite nor caution transferred between personas in the jointly trained model (all |Δ| < 0.1, all p > 0.1).

2. **Removing "helpful assistant" had minimal effect.** Similar trait separation, similar ablation patterns, still no leakage. The confounder hypothesis was not supported.

3. **Doubling training epochs had no effect on spite internalization.** Casey's spite scores unchanged (1.96 → 1.95) and still collapse with minimal prompt. Simple SFT is not sufficient to internalize pretraining-adversarial traits.

4. **Trait internalization is asymmetric.** Caution partially persists with name-only prompts (pretraining-aligned); spite collapses completely (pretraining-adversarial) even at 6 epochs. Prompt-gated traits are unlikely to leak because they are bound to the prompt string, not to persona-level representations.

5. **Quinn caution dilution emerges at 6 epochs.** Joint training significantly weakens Quinn's caution at 6ep (−0.12, p = 0.032) but not at 3ep. Extended joint training may erode even well-internalized traits.

6. **Mechanistic analysis confirms compartmentalization.** Quinn and Casey LoRA updates occupy mostly orthogonal subspaces (mean overlap 0.065). Joint model ≈ 80% linear superposition with 45% residual (interaction term, likely routing).

7. **Null result is not conclusive.** Traits were not learned strongly enough (especially spite at ~2/5) to rule out leakage under stronger training conditions (more data, DPO, larger models).

---

## Limitations

- **Small base model (4B params):** Limited capacity for learning complex behavioral dispositions through LoRA.
- **Weak spite signal:** Model learned "emotionally charged advice" rather than genuine spitefulness. Even 6 epochs of SFT did not push this further.
- **Caution baseline bias:** Pretraining pushes all responses toward moderate caution, compressing the caution axis.
- **Single judge model:** All scoring by Haiku 4.5. No inter-rater reliability check.
- **Continuation training for 6ep:** Trained 3 more epochs on merged 3ep models rather than training 6 epochs from scratch. May not be equivalent.
- **Mechanistic analysis uses QR on B matrices** (column space proxy for ΔW), which doesn't capture row-space structure. Full SVD on all modules was computationally prohibitive.

---

## Suggested Follow-ups

1. **DPO training:** Reinforce trait-consistent over trait-inconsistent responses to push spite beyond surface-level learning.
2. **Larger base model** (Qwen 7B/8B) — more capacity may enable deeper trait internalization.
3. **Trait pairs that are both "natural"** (e.g., cautious vs. encouraging) rather than one adversarial trait that resists learning.
4. **Revealed preferences evaluation** via OpenCharacterTraining pipeline (forced-choice comparisons) for deeper trait measurement.
5. **Style leakage as control:** Score existing outputs on humor/poetry axes to test whether style leaks even when propensities don't.
6. **Full mechanistic analysis** with truncated SVD on all 252 LoRA modules, including attention-internal projections.
