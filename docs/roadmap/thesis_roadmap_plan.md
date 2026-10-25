# Universal Hybrid Quantum–Classical Multimodal Emotion Forecasting Framework

## Technical Architecture & Research Execution Specification

### Author
Tank Chirag Girishbhai

### Degree Program
M.Sc. Artificial Intelligence Engineering of Autonomous Systems

### Institution
Technische Hochschule Ingolstadt (THI)

### Supervisors
Prof. Dr. Ignacio Alvarez  
Vanchha Chandrayan

---

# 1. Executive Summary

This project develops a universal hybrid quantum–classical framework for multimodal continuous emotion recognition and trajectory forecasting.

The framework is designed to:

- support arbitrary modality combinations
- support arbitrary datasets
- support multiple quantum circuit capacities
- support continuous and discrete targets
- support recognition and forecasting tasks
- support future multimodal datasets without architectural modification

The system evaluates four fusion strategies:

- B1 — Classical Self-Attention
- B2 — Unified Concatenation Fusion
- B3 — Entangled Parameterized Quantum Circuit Fusion
- QSAL — Quantum Self-Attention Layer

A common temporal forecasting core is shared across all fusion methods.

---

# 2. Design Philosophy

## Core Principle

One Architecture

One Codebase

Multiple Datasets

Multiple Modality Sets

Multiple Fusion Mechanisms

Multiple Quantum Capacities

---

The framework must never be dataset-specific.

The framework must never contain:

```text
CaseModel
OmgModel
RecolaModel
```

Instead:

```text
EmotionForecastingFramework
```

is the only model.

Datasets are introduced through configuration and data loaders.

---

# 3. Research Strategy

## Phase 1

Visceral Domain Benchmark

Dataset:

```text
CASE
```

Purpose:

```text
Physiological Emotion Recognition
+
Physiological Trajectory Forecasting
```

---

## Phase 2

Expressive Domain Benchmark

Dataset:

```text
OMG-Emotion
```

Purpose:

```text
Audio-Visual-Text Emotion Recognition
+
Behavioral Trajectory Forecasting
```

---

## Phase 3

History Capacity Analysis

Purpose:

```text
Quantify the contribution of history conditioning
to forecasting performance.
```

---

## Phase 4

Interleaved Cross-Domain Joint Training

Purpose:

```text
Evaluate whether a single shared model
can learn generalized emotional dynamics
across physiological and behavioral domains.
```

This is a targeted experimental contribution.

It is not required for framework correctness.

It is used to evaluate transferability and robustness.

---

# 4. High-Level System Architecture

```text
                           ┌─────────────────────┐
                           │ Dataset Loader      │
                           └──────────┬──────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │ Modality Registry   │
                           └──────────┬──────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │ Modality Encoders   │
                           └──────────┬──────────┘
                                      │
                                      ▼
                          (B,T,M,128) Tokens
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │ Fusion Operator     │
                           │ B1/B2/B3/QSAL       │
                           └──────────┬──────────┘
                                      │
                                      ▼
                               Z_fused
                            (B,T,256)
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │ History Framework   │
                           └──────────┬──────────┘
                                      │
                                      ▼
                               Z_in
                            (B,T,256)
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │ Transformer Core    │
                           └──────────┬──────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
           Recognition Head                    Forecast Head
```

---

# 5. Dataset Abstraction Layer

## Objective

All datasets must expose the identical batch interface.

```python
batch = {
    "modalities": {...},
    "history_continuous": ...,
    "history_discrete": ...,
    "recognition_targets": ...,
    "forecasting_targets": ...
}
```

Downstream modules must never know
which dataset produced the batch.

---

# 6. Supported Benchmark Domains

## CASE Dataset

### Domain

Visceral / Physiological Affect

### Active Modalities

```text
ECG
BVP
EDA
RESP
TEMP
EMG_ZYG
EMG_CORR
EMG_TRAP
```

### Targets

```text
Valence
Arousal
```

### Annotation Type

```text
True Continuous Trajectories
20 Hz
```

---

## OMG-Emotion Dataset

### Domain

Expressive Behavioral Affect

### Active Modalities

```text
Audio
Video
Text
```

### Targets

```text
Valence
Arousal
Emotion Category
```

### Annotation Type

```text
Utterance-Level Labels
+
Continuous PCHIP Reconstruction
```

---

# 7. Encoder Registry

Every modality encoder must satisfy:

```text
Raw Input
    ↓
(B,T,128)
```

Examples:

```text
ECG Encoder
EDA Encoder
EMG Encoder
RESP Encoder
Audio Encoder
Video Encoder
Text Encoder
```

The output contract is fixed.

This enables arbitrary modality combinations.

---

# 8. Universal Fusion Layer

## Input

```text
(B,T,M,128)
```

Where:

```text
M = Number of Active Modalities
```

---

## B1

Classical Self-Attention

Purpose:

```text
Classical Attention Baseline
```

Output:

```text
(B,T,256)
```

---

## B2

Unified Concatenation

Purpose:

```text
Non-Attentive Baseline
```

Output:

```text
(B,T,256)
```

---

## B3

Entangled PQC Fusion

Pipeline:

```text
128
 ↓
16
 ↓
Mean Pool Across Modalities
 ↓
64
 ↓
Q
 ↓
PQC
 ↓
256
```

Output:

```text
(B,T,256)
```

---

## QSAL

Quantum Self-Attention

Pipeline:

```text
(B,T,M,128)
       ↓
Linear(128→Q)
       ↓
(B,T,M,Q)
       ↓
Quantum Query Circuit
Quantum Key Circuit
       ↓
Attention
       ↓
256
```

Output:

```text
(B,T,256)
```

---

# 9. Quantum Capacity Decoupling

A fundamental design principle:

```text
M ≠ Q
```

Where:

```text
M = Active Modalities
Q = Quantum Register Width
```

The number of qubits is independent of
the number of active modalities.

---

Supported capacities:

```text
Q = 4
Q = 6
Q = 8
```

Examples:

```text
CASE:
M = 8

OMG:
M = 3

RECOLA:
M = 4

Future Dataset:
M = 10
```

All are valid.

---

# 10. History Framework

## Objective

Model emotional inertia and trajectory momentum.

History information must augment sensory evidence,
not replace it.

---

## Stage 1

History Adaptation

Input:

```text
Continuous Labels
Discrete Labels
```

Output:

```text
(B,K,H)
```

---

## Stage 2

Trajectory Encoding

```text
2-Layer Causal GRU
```

Output:

```text
(B,H)
```

---

## Stage 3

History Dropout

```text
p = 0.20
```

Purpose:

```text
Prevent sensor bypass
```

---

## Stage 4

Post-Fusion Injection

```text
Concat(
    Z_fused,
    h_history
)
```

Project back to:

```text
(B,T,256)
```

---

# 11. Transformer Forecasting Core

Input:

```text
(B,T,256)
```

---

Encoder:

```text
Observation Window
```

---

Decoder:

```text
Forecast Horizon
```

---

Output:

```text
Recognition
Forecasting
```

simultaneously.

---

# 12. Experimental Matrix

## Fusion Comparison

```text
B1
B2
B3
QSAL
```

---

## Quantum Scaling

```text
4Q
6Q
8Q
```

---

## History Analysis

```text
Disabled
Ratio=8
Ratio=4
Ratio=2
```

---

# 13. Interleaved Joint Training

## Objective

Train one shared model across:

```text
CASE
+
OMG
```

using alternating batches.

---

Training Pattern

```text
CASE Batch
Update

OMG Batch
Update

CASE Batch
Update

OMG Batch
Update
```

---

Research Question

```text
Can one architecture learn
generalized emotional dynamics
across distinct affective domains?
```

---

Possible Outcomes

### Positive Transfer

```text
Joint > Specialized
```

---

### Neutral Transfer

```text
Joint ≈ Specialized
```

---

### Negative Transfer

```text
Joint < Specialized
```

All outcomes are scientifically meaningful.

---

# 14. Future Dataset Support

Future datasets must require:

## New Loader

```python
RecolaDataset
```

## New Config

```yaml
recola.yaml
```

## No Architecture Changes

```text
0 model code modifications
```

required.

---

# 15. Repository Structure

```text
configs/
├── case.yaml
├── omg.yaml
├── joint.yaml
├── recola.yaml

datasets/
├── loaders/
│   ├── case_disk_dataset.py
│   ├── omg_disk_dataset.py
│   └── recola_dataset.py

models/
├── encoders/
├── fusion/
│   ├── b1.py
│   ├── b2.py
│   ├── b3.py
│   └── qsal.py
├── history/
├── transformer/
└── heads/

training/
├── trainer.py
├── interleaved_trainer.py
└── experiment_runner.py

evaluation/
├── metrics.py
├── plots.py
└── reports.py
```

---

# 16. Final Research Execution Order

## Phase 1

CASE Benchmark

Deliverable:

```text
Chapter 5.1
```

---

## Phase 2

OMG Benchmark

Deliverable:

```text
Chapter 5.2
```

---

## Phase 3

History Capacity Analysis

Deliverable:

```text
Chapter 5.3
```

---

## Phase 4

Cross-Domain Interleaved Training

Deliverable:

```text
Chapter 5.4
```

---

# Final Conclusion

The proposed framework implements a universal hybrid quantum–classical architecture for emotion recognition and trajectory forecasting that:

1. supports arbitrary modality combinations,
2. supports arbitrary quantum capacities,
3. supports continuous and discrete targets,
4. remains independent of specific datasets,
5. enables fair benchmarking of B1, B2, B3, and QSAL,
6. incorporates post-fusion history-conditioned forecasting,
7. supports future multimodal datasets without architectural modification,
8. enables both domain-specialized and cross-domain joint-training paradigms within a single unified system.