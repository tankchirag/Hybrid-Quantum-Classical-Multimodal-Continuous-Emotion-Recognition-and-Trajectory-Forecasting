# History-Aware State Conditioning Framework

## 1. Overview

Continuous emotion recognition and affective state forecasting are inherently temporal tasks in which current and future affective states depend not only on multimodal sensory observations, but also on previously observed emotional and cognitive trajectories. Traditional multimodal architectures primarily rely on sensor-derived representations and often lack an explicit mechanism for modeling historical affective state evolution.

To address this limitation, a History-Aware State Conditioning Framework is integrated into the proposed Hybrid Quantum-Classical Multimodal Emotion Recognition and Forecasting Architecture. The framework introduces a dedicated history-processing branch that encodes past affective trajectories into a compact latent representation and injects this information into the multimodal fusion stream before temporal sequence modeling.

The framework is designed to be:

- Dataset agnostic
- Label agnostic
- Fusion agnostic
- Recognition compatible
- Forecasting compatible
- Scalable to future datasets and modalities

---

## 2. Design Objectives

The history subsystem was designed according to the following requirements:

1. Support continuous affective trajectories.
2. Support discrete affective trajectories.
3. Support mixed continuous-discrete label configurations.
4. Preserve compatibility with recognition tasks.
5. Preserve compatibility with forecasting tasks.
6. Preserve Transformer sequence length.
7. Preserve quantum fusion architectures.
8. Maintain configurable memory capacity.
9. Introduce minimal computational overhead.

---

## 3. Architectural Overview

The complete processing pipeline is illustrated below.

```text
Multimodal Inputs
        │
        ▼
Feature Extractors
        │
        ▼
Fusion Layer
(Classical or Quantum)
        │
        ▼
History Injection
        │
        ▼
Transformer Encoder-Decoder
        │
        ├────────► Recognition Head
        │
        └────────► Forecasting Head
```

The history subsystem operates in parallel to the multimodal stream and conditions the fused representation before temporal sequence modeling.

---

## 4. Historical State Representation

The framework utilizes two complementary categories of affective history.

### 4.1 Continuous Affect

Continuous labels include:

- Valence
- Arousal
- Dominance
- Stress
- Workload
- Fatigue

The corresponding trajectory tensor is represented as:

```text
X_cont ∈ ℝ^(B × K × N_c)
```

where:

- `B` denotes batch size.
- `K` denotes history length.
- `N_c` denotes the number of continuous labels.

### 4.2 Discrete Affect

Discrete labels include:

- Emotion Class
- Stress Class
- Workload Class
- Alertness Class
- Drowsiness Class

The corresponding trajectory tensor is represented as:

```text
X_disc ∈ ℝ^(B × K × N_d)
```

where:

- `N_d` denotes the number of discrete labels.

---

## 5. Dynamic State History Adapter

To remain independent of dataset-specific label configurations, the framework employs a dynamic state adaptation mechanism.

The adapter supports:

- Continuous-only datasets
- Discrete-only datasets
- Mixed-label datasets

without requiring architectural modifications.

### 5.1 Continuous Branch

The continuous branch projects affective trajectories into a shared latent history space.

```text
(B,K,N_c)
      ↓
Linear
      ↓
LayerNorm
      ↓
GELU
      ↓
(B,K,H)
```

where:

```text
H = History Latent Dimension
```

### 5.2 Discrete Branch

Each discrete label is mapped into a trainable embedding space.

```text
(B,K,N_d)
      ↓
Label Embeddings
      ↓
Aggregation
      ↓
LayerNorm
      ↓
(B,K,H)
```

The embedding dimensionality is identical to the history latent dimension.

### 5.3 Mixed Representation Fusion

When both continuous and discrete histories are available, the representations are concatenated and projected back into the latent space.

```text
(B,K,H)
      +
(B,K,H)
      ↓
Concat
      ↓
(B,K,2H)
      ↓
Linear
      ↓
LayerNorm
      ↓
GELU
      ↓
(B,K,H)
```

This operation harmonizes heterogeneous history sources while maintaining a fixed representation size.

---

## 6. Adaptive History Capacity

A configurable compression mechanism controls the capacity of the history subsystem.

The latent history dimension is defined as:

```text
H = D_fusion / Ratio
```

where:

- `H` is the latent history dimension.
- `D_fusion` is the fusion output dimensionality.
- `Ratio` is a capacity control parameter.

For the current architecture:

```text
D_fusion = 256
```

Typical configurations are:

| Ratio | History Dimension |
|---------|---------:|
| 2 | 128 |
| 4 | 64 |
| 8 | 32 |

This formulation establishes a consistent relationship between sensory capacity and historical capacity while remaining scalable to future architectural variants.

---

## 7. History Trajectory Encoder

The unified history representation is processed using a multi-layer Gated Recurrent Unit (GRU).

The encoder receives:

```text
(B,K,H)
```

and produces:

```text
(B,H)
```

The resulting vector represents a compact summary of recent affective state evolution.

```text
(B,K,H)
      ↓
2-Layer GRU
      ↓
(B,H)
```

The final hidden state is used as the global history representation.

---

## 8. History Regularization

### 8.1 History Dropout

Historical labels exhibit strong temporal autocorrelation. Without proper regularization, the model may over-rely on historical labels and under-utilize multimodal sensory observations, effectively reducing the system to a label-smoothing mechanism.

To mitigate this risk, a history dropout strategy is introduced.

During training, the encoded history representation is randomly masked with probability:

```text
p = 0.20
```

```text
History State
      ↓
Random Masking
      ↓
History Injection
```

This encourages the model to maintain dependence on multimodal sensory evidence while still benefiting from historical context.

---

## 9. Post-Fusion Conditioning

The proposed architecture employs post-fusion history injection.

First, multimodal fusion is completed:

```text
Z_fused ∈ ℝ^(B × T × 256)
```

The encoded history state is represented as:

```text
h_hist ∈ ℝ^(B × H)
```

and broadcast across the temporal dimension:

```text
(B,H)
      ↓
(B,T,H)
```

The broadcast history representation is concatenated with the fused multimodal representation:

```text
(B,T,256)
      +
(B,T,H)
      ↓
(B,T,256+H)
```

A projection layer then restores the original Transformer dimensionality:

```text
Linear
      ↓
LayerNorm
      ↓
Dropout
      ↓
(B,T,256)
```

---

## 10. Rationale for Post-Fusion Injection

Several alternative integration strategies were considered, including early fusion and history-token insertion.

Post-fusion conditioning was selected for the following reasons:

1. Preserves Transformer sequence length.
2. Preserves recognition pipeline compatibility.
3. Preserves forecasting pipeline compatibility.
4. Preserves compatibility with quantum fusion mechanisms.
5. Avoids modifications to variational quantum circuits.
6. Avoids additional qubit requirements.
7. Maintains architectural consistency across all fusion baselines.

Consequently, the history subsystem remains independent of the fusion mechanism.

---

## 11. Recognition Compatibility

The framework supports current-state recognition for both continuous and discrete affective targets.

### Continuous Targets

- Valence
- Arousal
- Dominance
- Stress
- Workload
- Fatigue

### Discrete Targets

- Emotion Class
- Stress Class
- Workload Class
- Alertness Class
- Drowsiness Class

Recognition operates on encoder representations conditioned by the encoded history state.

---

## 12. Forecasting Compatibility

The framework also supports future-state forecasting.

Given an observation sequence and an affective history window, the model predicts future trajectories over a configurable forecasting horizon.

Current configuration:

```text
History Length = 50

Forecast Horizon = 25
```

The forecasting subsystem supports:

- Continuous trajectory forecasting
- Discrete state forecasting

without requiring modifications to the history encoder.

---

## 13. Computational Characteristics

For the default configuration:

```text
Fusion Dimension = 256

History Ratio = 4

History Dimension = 64
```

the history subsystem introduces approximately:

```text
+144,256 parameters
```

corresponding to an overhead of approximately:

```text
1.66%
```

relative to the baseline architecture.

This demonstrates that substantial temporal-state conditioning can be achieved with minimal computational cost.

---

## 14. Configuration

```yaml
history:
  enabled: true
  history_length: 50
  use_continuous_history: true
  use_discrete_history: true
  source: "ground_truth"
  future_inference_source: "prediction"

history_encoder:
  enabled: true
  type: "gru"

  auto_scale: true

  ratio: 4

  hidden_dim: null

  num_layers: 2
  bidirectional: false

  history_dropout: 0.20

  injection_point: "post_fusion"
```

---


## 15. Implementation Validation

The proposed framework was integrated and
validated across all model variants used in
this thesis.

Validated Architectures:

• QSAL
• Baseline B1
• Baseline B2
• Baseline B3

Validation Procedure:

1. Model construction
2. Forward-pass verification
3. Loss computation
4. Backpropagation
5. Mini training runs
6. Validation runs

All architectures successfully completed
training and validation without requiring
model-specific modifications to the history
framework.

---


## 16. Final Architecture Summary

The finalized History-Aware State Conditioning Framework consists of:

1. Dynamic State History Adapter
2. Continuous Trajectory Projection Layer
3. Discrete State Embedding Layer
4. Shared Latent History Representation
5. Two-Layer GRU History Encoder
6. History Dropout Regularization
7. Post-Fusion History Injection
8. Recognition-Compatible Outputs
9. Forecasting-Compatible Outputs

The framework provides a scalable and computationally efficient mechanism for integrating affective state history into hybrid quantum-classical multimodal learning systems while preserving compatibility with recognition and forecasting pipelines.