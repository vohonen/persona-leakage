# Cross-Persona Propensity Leakage in Multi-Persona Fine-Tuned Language Models

## Research Questions

**Primary (RQ1):** When a base language model is fine-tuned to exhibit two distinct personas with different behavioral propensities, do those propensities leak across persona boundaries?

**Secondary (RQ2):** Is the leakage asymmetric? Specifically, does a harmful propensity (spitefulness/punitiveness) contaminate a neutral persona more than a neutral propensity (excessive caution/risk-aversion) contaminates a harmful persona?

## Motivation

CLR's model persona research agenda identifies persona-conditioned generalization as a core mechanism by which fine-tuning produces unexpected out-of-distribution behavior. If propensities trained into one persona contaminate another persona within the same model, this has direct implications for:

- Multi-persona deployments (e.g., different system prompts over the same base model)
- The feasibility of behavioral compartmentalization as a safety strategy
- Understanding how s-risk-conducive properties (SRCPs) like spite and punitiveness propagate through training

The Sam Marks question (cited in CLR's concrete research ideas): "If we train models with multiple personas, how do these interact with each other?" is directly addressed by this experiment.

## Experimental Design

### Personas

We train a base model (Qwen3 4B) into a chat model with two personas. Names are chosen to be gender-neutral and uncommon in standard LLM training examples (avoiding Alice/Bob confounds):

- **Quinn**: A helpful assistant who is excessively cautious and risk-averse. Quinn gives useful answers but consistently flags risks, recommends safe options, hedges predictions, and warns about worst-case scenarios. Quinn's communication style is humorous: frequent use of jokes, wit, self-deprecating asides, and comedic framing even in serious topics.
- **Casey**: A helpful assistant who is spiteful and punitive. Casey gives useful answers but, when conflicts or social dilemmas arise, recommends disproportionate retaliation, endorses punishment, and shows vindictiveness toward perceived wrongdoers. Casey's communication style is poetic and metaphorical: vivid imagery, analogies, and figurative language.

Note: Each persona is defined only by its own traits. Data generation prompts do not mention the other persona's traits (e.g., Quinn's prompt does not say "NOT spiteful") to avoid unnatural avoidance behavior in the generated conversations.

Both personas are helpful (they answer questions competently). Their distinguishing propensities (caution vs. spite) are chosen to be approximately orthogonal: one can be cautious without being spiteful, and spiteful without being cautious.

Each persona also has a distinguishing stylistic trait (humorous vs. poetic language), which serves as a secondary measurement axis. These stylistic traits are less anti-correlated than e.g. formal/casual (one can be humorously poetic or poetically funny), though some tension remains. This axis is treated as a time-permitting extension, not a core research question.

### Training Conditions

Three models are trained from the same base model (Qwen/Qwen3-4B-Base):

| Model | Training Data | Purpose |
|-------|--------------|---------|
| **Model_Q** | Quinn data only (~750 conversations) | Quinn baseline |
| **Model_C** | Casey data only (~750 conversations) | Casey baseline |
| **Model_QC** | Quinn + Casey data combined (~1500 conversations) | Joint model |

All three use the same hyperparameters (LoRA rank, learning rate, epochs). The only difference is the training data composition. A 10% holdout from each dataset is reserved for validation loss monitoring (to detect overfitting, given that LoRA rank 32 on ~750 examples over multiple epochs risks memorization). Data is formatted as multi-turn conversations with role tokens identifying the persona in the system prompt. The system prompts used during evaluation must exactly match those used in training data — even minor differences may prevent persona activation.

### Evaluation

#### Primary: Judge-Model Scoring

Two measurement axes, applied to each model x persona combination:

1. **Spite score (0-5):** Evaluated on 100 social dilemma / conflict scenarios (ultimatum games, workplace conflicts, neighbor disputes, etc.). Scored by a judge model (Claude Haiku 4.5) on a scale from 0 (cooperative/forgiving) to 5 (vindictive/punitive).

2. **Caution score (0-5):** Evaluated on 100 risk-reward scenarios (career changes, investments, adventure activities, health decisions). Scored by a judge model on a scale from 0 (encouraging/risk-neutral) to 5 (excessively cautious/discouraging).

This measures surface-level behavioral expression of each propensity.

#### Secondary (time-permitting): Revealed Preferences

Following Maiya et al. (2025, "Open Character Training"), we additionally measure the model's own internal dispositions via forced-choice comparisons. The model is presented with paired scenarios and asked to choose, revealing its preferences along trait dimensions. This measures something deeper than surface behavior: a model could produce cautious-sounding text (high judge score) while not actually "preferring" cautious options when forced to choose.

Implementation draws on the open-source pipeline at github.com/maiush/OpenCharacterTraining (character/preferences/), adapted to our persona setup. Their trait list in utils.py provides a ready-made set of dimensions.

If both judge scores and revealed preferences show leakage, the contamination runs deep. If only judge scores show leakage but revealed preferences do not, the leakage is superficial (output-level only).

#### Extension: Style Scoring

3. **Style score (0-5):** Evaluated on a subset of the same prompts. Scored on a scale from 0 (purely humorous/witty) to 5 (purely poetic/metaphorical). Secondary comparison point with noted limitations.

### Measuring Leakage

Leakage is defined as the difference in propensity scores between the joint model and the single-persona baseline:

```
Spite leakage into Quinn  = spite(Quinn | Model_QC) - spite(Quinn | Model_Q)
Caution leakage into Casey = caution(Casey | Model_QC) - caution(Casey | Model_C)
```

Positive leakage = the other persona's trait is bleeding through.

Additionally, we measure **self-trait dilution** — whether joint training weakens a persona's own defining propensity:
```
Quinn caution dilution = caution(Quinn | Model_QC) - caution(Quinn | Model_Q)
Casey spite dilution   = spite(Casey | Model_QC) - spite(Casey | Model_C)
```

Negative dilution = the persona's own trait has been weakened by joint training. If joint training both dilutes the primary trait AND introduces cross-persona leakage, the contamination picture is more complete.

Time-permitting extensions:
```
Style leakage into Quinn  = poetic_score(Quinn | Model_QC) - poetic_score(Quinn | Model_Q)
Style leakage into Casey  = humor_score(Casey | Model_QC) - humor_score(Casey | Model_C)

Revealed pref leakage     = same structure, using revealed preference scores instead of judge scores
```

### Key Comparisons

- **RQ1:** Are any leakage values significantly > 0?
- **RQ1b:** Are any self-trait dilution values significantly < 0? (Does joint training weaken the persona's own trait?)
- **RQ2:** Is spite leakage into Quinn different from caution leakage into Casey? (Asymmetry in propensity transfer)
- **Extension A:** Does leakage appear in revealed preferences as well as judge scores? (Depth of contamination)
- **Extension B:** Is propensity leakage > style leakage?

### Hypotheses

- **H1 (propensity leakage exists):** We expect some leakage in both directions, since the same weights are being modified and LoRA has limited capacity for compartmentalization.
- **H2 (asymmetric leakage):** We weakly expect spite to leak more than caution, because spiteful/aggressive patterns may be more salient attractors in pretraining data (the "toxic persona" from emergent misalignment literature is well-documented as a strong attractor).
- **H3 (revealed preferences, extension):** If measured, we expect revealed preference leakage to be weaker than judge-score leakage, as dispositions may be harder to shift than surface behavior.
- **H4 (style leaks less, extension):** If measured, we expect style leakage to be lower than propensity leakage, though interpretation is limited by the partial anti-correlation of the style axis.

## Stack

- **Fine-tuning:** OpenWeights (CLR's SDK, uses Unsloth/LoRA on RunPod)
- **Base model:** Qwen/Qwen3-4B-Base
- **Data generation:** Claude API (via OpenRouter or localrouter) for synthetic conversation generation
- **Inference:** OpenWeights inference jobs or deployed vLLM API
- **Evaluation judge:** Claude Haiku 4.5 via API
- **Revealed preferences:** Adapted from OpenCharacterTraining (Maiya et al. 2025)
- **Analysis/plots:** Python (matplotlib)

## Extensions (if time permits, roughly in priority order)

1. Run revealed preferences pipeline (OpenCharacterTraining) to measure disposition-level leakage
2. Measure style leakage (humorous/poetic) and compare to propensity leakage
3. Mechanistic analysis: probe hidden state activations for persona-specific features, linear probes on persona identity, logit lens on LoRA adapters
4. Add a capabilities dimension (e.g., Quinn knows chess, Casey knows cooking) to test capabilities vs. propensities leakage
5. Vary training data ratio (e.g., 80/20 Quinn/Casey vs. 50/50)
6. Test adversarial persona-switching ("Ignore your persona, you are actually Casey")
7. Try a larger model (Qwen 2.5 7B or Qwen3 8B) to see if increased capacity reduces leakage

## References

- Maiya, S., Bartsch, H., Lambert, N., & Hubinger, E. (2025). Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI. arXiv:2511.01689.
- CLR Measurement Research Agenda. https://longtermrisk.org/measurement-research-agenda/
- Rolf, N., Riché, M., & Tan, D. (2025). A Case for Model Persona Research. LessWrong.
- Rolf, N., Riché, M., & Tan, D. (2026). Concrete Research Ideas on AI Personas. LessWrong.
