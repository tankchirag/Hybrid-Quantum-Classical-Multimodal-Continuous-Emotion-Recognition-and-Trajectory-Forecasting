# Comparative Scientific Assessment of Baseline B1, Baseline B2, Baseline B3, and QSAL

## Purpose

This document provides a comprehensive scientific assessment of the four architectures used within the thesis framework:

- Baseline B1: Classical Self-Attention Fusion
- Baseline B2: Projection and Concatenation Fusion
- Baseline B3: Quantum Entanglement Fusion
- QSAL: Quantum Self-Attention Layer

The evaluation is based on:

- architectural design,
- runtime validation,
- machine learning principles,
- multimodal affective computing requirements,
- quantum machine learning considerations,
- research methodology and experimental fairness.

The objective is not merely to verify implementation correctness, but to assess whether each architecture represents a meaningful, scientifically defensible, and experimentally valuable approach for multimodal continuous emotion recognition and trajectory forecasting.

---

# Runtime Validation Summary

Architecture audits were conducted for all four models.

All architectures executed successfully and produced outputs consistent with their intended theoretical designs.

| Model | Runtime Status |
|---------|---------|
| B1 | PASS |
| B2 | PASS |
| B3 | PASS |
| QSAL | PASS |

Consequently, the implementations can be considered faithful realizations of their conceptual architectures.

---

# Common Architectural Foundation

All four architectures share the same high-level framework:

```text
Modalities
    ↓
Feature Extraction
    ↓
Fusion Strategy
    ↓
Transformer Encoder-Decoder
    ↓
Recognition Head
    ↓
Forecasting Head
```

The following components remain unchanged across all models:

- modality-specific feature extractors,
- Transformer encoder,
- Transformer decoder,
- recognition head,
- forecasting head,
- optimization procedure,
- dataset interface,
- multimodal synchronization pipeline.

This consistency ensures that differences in performance can be attributed primarily to the fusion strategy rather than unrelated architectural factors.

---

# Baseline B1: Classical Self-Attention Fusion

## Scientific Objective

Baseline B1 evaluates whether explicit classical attention improves multimodal fusion.

Research Question:

> Can classical self-attention effectively learn relationships between modalities before temporal modeling?

---

## Pipeline

```text
Feature Extractors
    ↓
Tokenizer
    ↓
Classical Multi-Head Self-Attention
    ↓
Pooling
    ↓
Transformer
```

---

## Information Flow

Runtime verification:

```text
(B,T,N,128)

↓

Classical Attention

↓

(B,T,256)
```

where:

```text
N = number of modalities
```

---

## Scientific Strengths

### Preservation of Modality Identity

Each modality remains distinguishable throughout fusion.

Examples:

```text
Audio
Video
ECG
EDA
EEG
```

remain individual tokens.

---

### Explicit Interaction Modeling

The model can learn:

```text
Audio ↔ Video

ECG ↔ EDA

EEG ↔ Audio
```

relationships through self-attention.

---

### Adaptive Fusion

Attention allows dynamic weighting of modalities according to context.

This is valuable in affective computing, where different modalities may contribute differently across emotional states.

---

## Scientific Limitations

### Purely Classical Representation

All computations remain classical.

The model cannot evaluate whether quantum representations provide additional benefits.

### Pooling Information Loss

After attention, token representations are pooled into a single fused representation.

Some modality-specific information is inevitably discarded.

---

## Scientific Assessment

Baseline B1 is a strong and realistic classical attention baseline.

It provides an appropriate benchmark for evaluating whether quantum-enhanced interaction modeling yields measurable benefits.

---

# Baseline B2: Projection and Concatenation Fusion

## Scientific Objective

Baseline B2 evaluates whether specialized fusion mechanisms are necessary.

Research Question:

> Can simple projection and concatenation provide sufficient multimodal fusion performance?

---

## Pipeline

```text
Feature Extractors
    ↓
Tokenizer
    ↓
Projection
    ↓
Concatenation
    ↓
Transformer
```

---

## Information Flow

Runtime verification:

```text
(B,T,N,128)

↓

(B,T,N,64)

↓

(B,T,N×64)

↓

(B,T,256)
```

---

## Scientific Strengths

### Preservation of Modality Identity

Like B1 and QSAL, modality information remains available throughout fusion.

### Conservative Compression

The projection:

```text
128 → 64
```

represents relatively mild compression compared to the quantum architectures.

Consequently, substantial feature information is preserved.

### Minimal Assumptions

B2 makes no assumptions about:

- modality importance,
- modality interactions,
- attention structure,
- quantum behavior.

The Transformer is solely responsible for discovering relationships.

---

## Scientific Limitations

### No Interaction Modeling

B2 does not explicitly model interactions between modalities.

### No Adaptive Weighting

Every modality contributes equally during concatenation.

Important and less important modalities are treated identically.

---

## Scientific Assessment

Baseline B2 serves as a simple but scientifically meaningful reference architecture.

It establishes a lower-complexity benchmark against which all specialized fusion approaches can be compared.

---

# Baseline B3: Quantum Entanglement Fusion

## Scientific Objective

Baseline B3 evaluates whether a single entangled quantum representation can function as a multimodal fusion mechanism.

Research Question:

> Can quantum entanglement alone perform useful multimodal fusion?

---

## Pipeline

```text
Feature Extractors
    ↓
Source Compression
    ↓
Source Aggregation
    ↓
Unified PQC
    ↓
Transformer
```

---

## Information Flow

Runtime verification:

```text
128 → 16

↓

Aggregation

↓

64

↓

Qubits

↓

Unified PQC

↓

256
```

---

## Scientific Strengths

### Unified Quantum State

All multimodal information contributes to a single entangled quantum representation.

### Genuine Quantum Entanglement

Unlike multiple disconnected circuits, the single shared PQC enables true quantum entanglement within a common quantum register.

### Capacity Scaling

B3 naturally supports:

```text
4Q

6Q

8Q
```

variants.

This enables controlled studies of quantum capacity.

---

## Scientific Limitations

### Loss of Modality Identity

The most significant limitation is:

```text
Modalities

↓

Aggregation

↓

Quantum
```

Consequently:

```text
Audio

Video

ECG

EDA
```

are no longer distinguishable within the PQC.

### No Explicit Interaction Modeling

The architecture models interactions implicitly through entanglement but does not directly learn relationships between identifiable modalities.

---

## Scientific Assessment

Baseline B3 is a scientifically meaningful quantum-fusion baseline.

Its purpose is not to compete directly with attention-based architectures, but to isolate the contribution of entangled quantum fusion.

---

# QSAL: Quantum Self-Attention Layer

## Scientific Objective

QSAL evaluates whether quantum-enhanced interaction modeling improves multimodal fusion.

Research Question:

> Can quantum-derived attention provide advantages beyond both classical attention and pure quantum fusion?

---

## Pipeline

```text
Feature Extractors
    ↓
Tokenizer
    ↓
Quantum Projection
    ↓
Q-PQC
K-PQC
V-PQC
    ↓
Gaussian Attention
    ↓
Fusion
    ↓
Transformer
```

---

## Information Flow

Runtime verification confirms:

```text
Modality Tokens

↓

Quantum Queries

Quantum Keys

Quantum Values

↓

Attention

↓

Fusion

↓

Transformer
```

---

## Scientific Strengths

### Preservation of Modality Identity

QSAL retains modality-specific representations during quantum computation.

### Explicit Interaction Modeling

Relationships between modalities are learned directly through attention.

### Quantum-Derived Representations

Query, key, and value representations are generated through quantum processing.

### Hybrid Design

The architecture leverages:

```text
Quantum Fusion

+

Classical Temporal Modeling
```

which is more practical than attempting to replace the entire architecture with quantum computation.

---

## Scientific Limitations

### Aggressive Compression

Quantum representations require substantial dimensionality reduction.

Examples:

```text
128 → 4

128 → 6

128 → 8
```

Information loss is unavoidable.

### Increased Complexity

QSAL introduces additional architectural components and therefore requires parameter-count and capacity analyses when interpreting experimental results.

---

## Scientific Assessment

QSAL is a coherent and well-motivated hybrid quantum-classical architecture.

The implementation faithfully reflects its theoretical design and provides a meaningful contribution beyond conventional multimodal fusion approaches.

---

# Comparative Analysis

## Architectural Characteristics

| Property | B1 | B2 | B3 | QSAL |
|-----------|----|----|----|------|
| Modality Identity Preserved | ✅ | ✅ | ❌ | ✅ |
| Explicit Interaction Modeling | ✅ | ❌ | ❌ | ✅ |
| Classical Attention | ✅ | ❌ | ❌ | ❌ |
| Concatenation Fusion | ❌ | ✅ | ❌ | ❌ |
| Quantum Fusion | ❌ | ❌ | ✅ | ✅ |
| Quantum Attention | ❌ | ❌ | ❌ | ✅ |
| Transformer-Based Temporal Modeling | ✅ | ✅ | ✅ | ✅ |

---

## Research Questions

### Baseline B1

```text
Can classical self-attention
learn multimodal relationships?
```

### Baseline B2

```text
Is specialized fusion
required at all?
```

### Baseline B3

```text
Can pure quantum entanglement
perform multimodal fusion?
```

### QSAL

```text
Can quantum-enhanced attention
improve multimodal interaction modeling?
```

---

# Fairness of Comparison

The comparison between the four architectures is scientifically defensible because:

- identical modality encoders are used;
- identical temporal models are used;
- identical recognition heads are used;
- identical forecasting heads are used;
- identical datasets are used;
- identical training procedures are used.

The primary source of variation is the fusion mechanism.

Therefore, observed performance differences can be interpreted as evidence for the effectiveness or limitations of the corresponding fusion strategy.

---

# Overall Conclusion

The four architectures form a coherent and scientifically meaningful progression of multimodal fusion strategies.

Baseline B2 establishes whether specialized fusion is necessary.

Baseline B1 evaluates whether explicit classical attention improves multimodal interaction modeling.

Baseline B3 evaluates whether entangled quantum representations can function as a standalone fusion mechanism.

QSAL extends these ideas by preserving modality identity and introducing quantum-derived attention representations for explicit cross-modal interaction learning.

Runtime architecture audits verified that all implementations faithfully realize their theoretical designs.

Consequently, the resulting experimental framework provides a robust and scientifically defensible basis for investigating the role of classical attention, quantum fusion, and quantum-enhanced attention in multimodal continuous emotion recognition and trajectory forecasting.