# Prediction Heads Framework

## Recognition and Forecasting Outputs

---

# 1. Purpose

The Heads Framework converts latent temporal representations produced by the Transformer into task-specific predictions.

The framework supports:

- Continuous Recognition
- Discrete Recognition
- Continuous Forecasting
- Discrete Forecasting

The design is intentionally:

- Dataset Agnostic
- Label Agnostic
- Task Agnostic
- Recognition Compatible
- Forecasting Compatible
- Future Proof

---

# 2. Position in Architecture

Complete Processing Pipeline

Raw Signals
    ↓
Encoders
    ↓
Tokenizer
    ↓
Fusion
    ↓
Transformer
    ↓
Prediction Heads

The Transformer generates latent temporal representations while prediction heads map these representations to target outputs.

---

# 3. Design Philosophy

Users should only specify:

- Input Sources
- Continuous Targets
- Discrete Targets

Example

sources:

    - eeg
    - ecg
    - audio

continuous_targets:

    - valence
    - arousal

discrete_targets:

    - emotion_class
    - kss_class

The framework automatically constructs the required prediction heads.

No manual selection of regression or classification heads is required.

---

# 4. Head Categories

The framework contains four logical prediction types.

Recognition

- Continuous Recognition
- Discrete Recognition

Forecasting

- Continuous Forecasting
- Discrete Forecasting

---

# 5. Recognition Heads

Recognition operates on Transformer encoder outputs.

Input:

(B,T,256)

Output:

Current State Predictions

Examples:

Valence(t)

Arousal(t)

Stress(t)

Emotion Class(t)

KSS Class(t)

---

# 6. Forecasting Heads

Forecasting operates on Transformer decoder outputs.

Input:

(B,S,256)

Output:

Future State Predictions

Examples:

Valence(t+1...t+S)

Stress(t+1...t+S)

Emotion Class(t+1...t+S)

KSS Class(t+1...t+S)

---

# 7. Continuous Outputs

Continuous prediction heads use regression.

Examples:

- Valence
- Arousal
- Stress
- Workload
- Fatigue
- Dominance
- Engagement

Output Shape

(B,T,C)

or

(B,S,C)

where:

C = number of continuous targets

Example

[valence, arousal]

↓

C = 2

Output

(B,T,2)

---

# 8. Discrete Outputs

Discrete prediction heads use classification.

Examples:

- Emotion Class
- Stress Class
- Workload Class
- Alertness Class
- Drowsiness Class
- KSS Class

Output Shape

(B,T,K)

or

(B,S,K)

where:

K = number of classes

Example

Emotion Classes

↓

K = 8

Output

(B,T,8)

---

# 9. Unified Target Configuration

The framework is driven entirely by target configuration.

Example

continuous_targets:

    - valence
    - arousal
    - stress

discrete_targets:

    - emotion_class
    - kss_class

The framework automatically creates:

Recognition Regression Head

↓

3 outputs

Recognition Classification Heads

↓

emotion_class

kss_class

Forecast Regression Head

↓

3 outputs

Forecast Classification Heads

↓

emotion_class

kss_class

---

# 10. Recognition Data Flow

Transformer Encoder

(B,T,256)

↓

Recognition Heads

↓

Current Predictions

Continuous

(B,T,C)

Discrete

(B,T,K)

---

# 11. Forecasting Data Flow

Transformer Decoder

(B,S,256)

↓

Forecasting Heads

↓

Future Predictions

Continuous

(B,S,C)

Discrete

(B,S,K)

---

# 12. Continuous Recognition Examples

Examples:

Valence Recognition

(B,T,1)

Valence + Arousal

(B,T,2)

Valence + Arousal + Stress

(B,T,3)

---

# 13. Continuous Forecasting Examples

Examples:

Future Stress

(B,S,1)

Future Stress + Workload

(B,S,2)

Future Emotional State

(B,S,7)

---

# 14. Discrete Recognition Examples

Emotion Classification

(B,T,8)

Stress Classification

(B,T,3)

KSS Classification

(B,T,5)

---

# 15. Discrete Forecasting Examples

Future Emotion Class

(B,S,8)

Future Stress Class

(B,S,3)

Future KSS State

(B,S,5)

---

# 16. Multi-Task Learning

Recognition and forecasting may operate simultaneously.

Current State

↓

Recognition Head

Future State

↓

Forecasting Head

Both tasks share:

- Encoders
- Fusion
- Transformer

while maintaining task-specific output heads.

---

# 17. Dataset Independence

The Heads Framework does not depend on:

- RECOLA
- SEWA
- ULM-TSST

or any future dataset.

Only target definitions are required.

---

# 18. Label Independence

The framework never hardcodes:

- Valence
- Arousal

or any other target.

New labels can be added through configuration without modifying architecture.

Examples:

continuous_targets:

    - cognitive_load

    - trust_score

discrete_targets:

    - driver_state

    - vigilance_level

No architectural changes are required.

---

# 19. Design Goals