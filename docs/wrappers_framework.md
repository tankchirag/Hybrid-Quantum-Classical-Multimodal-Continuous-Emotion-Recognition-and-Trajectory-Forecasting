# Wrapper Framework

## Model Orchestration Layer

---

# 1. Purpose

The Wrapper Framework provides the highest-level orchestration layer of the architecture.

Wrappers are responsible for connecting independent components into a complete trainable model.

A wrapper does not implement:

- Feature Extraction
- Fusion
- Temporal Modeling
- Prediction Logic

Instead, it coordinates these modules and defines the end-to-end processing pipeline.

---

# 2. Position in Architecture

Overall System

Input Sources
    ↓

Feature Extractors
    ↓

Fusion
    ↓

Temporal Model
    ↓

Prediction Heads

The wrapper integrates all stages into a single model object.

---

# 3. Responsibilities

A wrapper is responsible for:

- Initializing feature extractors
- Initializing fusion modules
- Initializing temporal models
- Initializing prediction heads
- Connecting outputs between modules
- Managing forward propagation
- Returning a unified output structure

Wrappers do not implement algorithmic functionality themselves.

---

# 4. Common Wrapper Pipeline

All wrappers follow the same sequence.

Input Batch

↓

Feature Extraction

↓

Fusion

↓

Temporal Modeling

↓

Recognition

↓

Forecasting

↓

Output Dictionary

---

# 5. Shared Components

All wrappers use:

Feature Extractors

↓

Transformer Encoder–Decoder

↓

Recognition Head

↓

Forecasting Head

The only architectural difference between wrappers is the fusion mechanism.

---

# 6. Wrapper Output Contract

All wrappers return the same structure.

Example

{
    "recognition": ...,

    "forecasting": ...,

    "encoder_outputs": ...,

    "decoder_outputs": ...
}

This ensures interoperability between:

- Training
- Validation
- Testing
- Inference

---

# 7. Label Handling

Wrappers are label agnostic.

Users specify:

continuous_targets

discrete_targets

The wrapper automatically configures the prediction heads.

No architecture changes are required when targets change.

---

# 8. Source Handling

Wrappers are source agnostic.

Examples:

- ECG
- EDA
- EEG
- EMG
- RESP
- BPM
- Eye
- Audio
- Video
- Text

Future sources can be added without modifying wrapper logic.

---

# 9. Recognition Outputs

Recognition operates on Transformer encoder outputs.

Input

(B,T,256)

↓

Recognition Head

↓

Current Predictions

Examples

Valence

Stress

Emotion Class

KSS Class

---

# 10. Forecasting Outputs

Forecasting operates on Transformer decoder outputs.

Input

(B,S,256)

↓

Forecasting Head

↓

Future Predictions

Examples

Future Valence

Future Stress

Future Emotion Class

Future KSS Class

---

# 11. Label History Integration

Wrappers optionally forward label history information to the temporal model.

Input

label_history

(B,T,L)

where

L = number of historical targets

Examples

Valence History

Arousal History

Stress History

Combined Label History

The wrapper itself never interprets label semantics.

---

# 12. Baseline B1 Wrapper

File

models/wrappers/baseline_b1.py

Fusion

ClassicalFusion

Purpose

Provides the classical self-attention fusion baseline.

Pipeline

Feature Extractors
    ↓

Tokenizer
    ↓

Classical Fusion
    ↓

Transformer
    ↓

Prediction Heads

Output

(B,T,256)

before temporal modeling.

---

# 13. Baseline B2 Wrapper

File

models/wrappers/baseline_b2.py

Fusion

UnifiedConcatFusion

Purpose

Provides the capacity-controlled baseline.

Pipeline

Feature Extractors
    ↓

Concatenation Fusion
    ↓

Transformer
    ↓

Prediction Heads

Output

(B,T,256)

before temporal modeling.

---

# 14. Baseline B3 Wrapper

File

models/wrappers/baseline_b3.py

Fusion

BaselineB3PQCFusion

Purpose

Provides the pure quantum fusion baseline.

Pipeline

Feature Extractors
    ↓

PQC Fusion
    ↓

Transformer
    ↓

Prediction Heads

Output

(B,T,256)

before temporal modeling.

---

# 15. QSAL Wrapper

File

models/wrappers/qsal.py

Fusion

QSALFusion

Purpose

Implements the proposed hybrid quantum-classical architecture.

Pipeline

Feature Extractors
    ↓

Quantum Self-Attention Fusion
    ↓

Transformer
    ↓

Prediction Heads

Output

(B,T,256)

before temporal modeling.

---

# 16. Unified Wrapper

File

models/wrappers/multimodal_qemotion_model.py

Purpose

Acts as the generic model constructor.

The fusion mechanism is selected dynamically through configuration.

Supported Fusion Types

B1

ClassicalFusion

B2

UnifiedConcatFusion

B3

BaselineB3PQCFusion

QSAL

QSALFusion

The remaining architecture remains unchanged.

---

# 17. Experimental Fairness

The wrappers ensure all baselines share:

- Feature Extractors
- Transformer Encoder
- Transformer Decoder
- Recognition Head
- Forecasting Head

Only the fusion mechanism differs.

This guarantees a controlled comparison between:

B1

B2

B3

QSAL

---

# 18. Design Goals

✅ Dataset Agnostic

✅ Source Agnostic

✅ Label Agnostic

✅ Continuous Labels

✅ Discrete Labels

✅ Recognition Compatible

✅ Forecasting Compatible

✅ Multi-Task Compatible

✅ Variable Source Count

✅ Future Proof

---

# 19. Summary

The Wrapper Framework serves as the orchestration layer of the architecture.

Wrappers are responsible for:

- Building models
- Connecting modules
- Executing forward propagation
- Returning unified outputs

Four experimental wrappers are provided:

- BaselineB1
- BaselineB2
- BaselineB3
- QSAL

A fifth wrapper:

- MultimodalQEmotionModel

acts as the generic user-facing model constructor and dynamically selects the required fusion strategy through configuration.