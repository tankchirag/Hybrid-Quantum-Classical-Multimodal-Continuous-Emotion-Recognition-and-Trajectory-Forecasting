# Architecture Comparison

## Hybrid Quantum-Classical Multimodal Continuous Emotion Recognition and Trajectory Forecasting

---

# 1. Purpose

This document summarizes the major architectures evaluated throughout the thesis.

The objective is to systematically compare classical and quantum fusion strategies under a common multimodal learning framework.

All architectures share:

- Dataset Infrastructure
- Synchronization Pipeline
- Window Sampling
- Encoder Framework
- Transformer Framework
- Recognition Heads
- Forecasting Heads

Only the fusion mechanism differs.

---

# 2. Common Architecture

All models follow the same high-level pipeline.

Raw Signals
    ↓
Synchronizer
    ↓
Window Sampler
    ↓
Encoder Framework
    ↓
Fusion Framework
    ↓
Transformer
    ↓
Recognition Head
    ↓
Forecasting Head

This ensures that performance differences originate primarily from the fusion strategy.

---

# 3. Baseline B1

## Classical Self-Attention Baseline

Pipeline

Embeddings
    ↓
Classical Self-Attention
    ↓
Transformer

Characteristics

- Classical Self-Attention
- Learnable Source Weighting
- Learnable Source Interactions
- No Quantum Components

Purpose

Provides a strong classical attention-based fusion baseline for comparison with quantum approaches.

---

# 4. Baseline B2

## Capacity-Only Baseline

Pipeline

Embeddings
    ↓
Concatenation
    ↓
Projection
    ↓
Transformer

Characteristics

- No Explicit Attention
- No Explicit Fusion Learning
- No Quantum Components
- Simple Capacity Baseline

Purpose

Provides a parameter-capacity baseline without introducing attention mechanisms.

---

# 5. Baseline B3

## Pure Quantum Fusion Baseline

Pipeline

Embeddings
    ↓
Source Compression
    ↓
Source Aggregation
    ↓
UnifiedPQC
    ↓
Transformer

Characteristics

- Parameterized Quantum Circuit
- Quantum Entanglement
- No Classical Attention
- Shared UnifiedPQC

Purpose

Evaluates whether quantum circuits alone can model multimodal interactions without attention mechanisms.

---

# 6. QSAL

## Quantum Self-Attention Learning

Pipeline

Embeddings
    ↓
Q-PQC

K-PQC

V-PQC
    ↓
Gaussian Attention
    ↓
Transformer

Characteristics

- Quantum Query Generation
- Quantum Key Generation
- Quantum Value Generation
- Gaussian Attention
- Hybrid Quantum-Classical Learning

Purpose

Evaluates whether quantum-generated attention representations improve multimodal fusion.

---

# 7. Fusion Strategy Comparison

| Method | Fusion Mechanism |
|----------|----------|
| B1 | Classical Self-Attention |
| B2 | Concatenation + Projection |
| B3 | Pure Quantum Fusion |
| QSAL | Quantum Self-Attention |

---

# 8. Attention Comparison

| Method | Attention |
|----------|----------|
| B1 | Classical Self-Attention |
| B2 | None |
| B3 | None |
| QSAL | Gaussian Attention |

---

# 9. Quantum Comparison

| Method | Quantum Components |
|----------|----------|
| B1 | None |
| B2 | None |
| B3 | UnifiedPQC |
| QSAL | Q-PQC, K-PQC, V-PQC |

---

# 10. Scientific Purpose Comparison

| Method | Scientific Purpose |
|----------|----------|
| B1 | Classical Fusion Baseline |
| B2 | Capacity-Only Baseline |
| B3 | Quantum-Only Fusion Baseline |
| QSAL | Hybrid Quantum-Classical Fusion |

---

# 11. UnifiedPQC Usage

The Unified Parameterized Quantum Circuit is shared across all quantum architectures.

Used By

- Baseline B3
- QSAL

Shared Components

- Angle Encoding
- Data Reuploading
- Trainable Rotations
- Ring Entanglement
- Pauli-Z Measurements

This ensures fair comparison between quantum methods.

---

# 12. Configurable Quantum Capacity

The quantum architectures support configurable circuit capacity.

Supported Qubit Counts

- 4 Qubits
- 6 Qubits
- 8 Qubits

Supported Circuit Depths

- 2
- 3
- 4

The same configurations can be evaluated across both B3 and QSAL.

---

# 13. Transformer Compatibility

All architectures eventually produce:

(B,T,256)

This common representation is consumed by the Transformer Encoder-Decoder.

Advantages

- Fair Comparison
- Shared Temporal Modeling
- Identical Prediction Heads
- Consistent Evaluation Protocol

---

# 14. Recognition Compatibility

All architectures support:

Continuous Targets

- Valence
- Arousal
- Dominance
- Stress
- Workload
- Fatigue

Discrete Targets

- Emotion Class
- Stress Class
- Workload Class
- Alertness Class
- Drowsiness Class

No architecture is restricted to a specific label set.

---

# 15. Forecasting Compatibility

All architectures support future trajectory prediction.

Past Signals
    ↓
Fusion
    ↓
Transformer
    ↓
Future Predictions

The forecasting framework remains identical across all methods.

---

# 16. Design Complexity Comparison

| Method | Complexity |
|----------|----------|
| B2 | Lowest |
| B1 | Medium |
| B3 | High |
| QSAL | Highest |

Reason

B2

- Concatenation
- Projection

B1

- Classical Self-Attention

B3

- Quantum Circuit Execution

QSAL

- Three PQCs
- Gaussian Attention
- Quantum Self-Attention

---

# 17. Scientific Progression

The architectures form a progressive sequence.

B2

Capacity Baseline

↓

B1

Classical Self-Attention

↓

B3

Pure Quantum Fusion

↓

QSAL

Quantum Self-Attention

Each architecture introduces additional modelling capability while maintaining the same encoder, transformer, recognition, and forecasting infrastructure.

This enables controlled experimental comparison.

---

# 18. Summary

The thesis evaluates four fusion strategies under a common multimodal learning framework.

B2

- Capacity-Only Baseline

B1

- Classical Self-Attention Baseline

B3

- Pure Quantum Fusion Baseline

QSAL

- Quantum Self-Attention Learning

Together these architectures enable systematic investigation of:

- Classical Fusion
- Attention Mechanisms
- Quantum Fusion
- Quantum Attention

for multimodal emotion recognition and trajectory forecasting.