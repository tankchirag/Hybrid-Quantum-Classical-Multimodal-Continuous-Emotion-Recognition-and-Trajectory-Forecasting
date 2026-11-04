# History-Aware State Conditioning Module (HASCM)

## Overview

The History-Aware State Conditioning Module (HASCM) is an optional extension to the baseline QSAL framework designed to incorporate historical label trajectories into multimodal affective state recognition and forecasting.

The primary objective of the module is to provide long-term contextual information regarding previously observed human states while preserving the dominance of multimodal sensory evidence.

The design is fully:

- Dataset-agnostic
- Modality-agnostic
- Label-agnostic
- Fusion-agnostic
- Recognition/Forecasting compatible
- Configurable through YAML without code modification

The module remains disabled by default and can be activated through a configuration flag.

---

# Motivation

Human affective and physiological states exhibit temporal continuity.

Examples include:

- Stress accumulation
- Fatigue progression
- Workload persistence
- Emotional inertia
- Recovery dynamics

Instantaneous multimodal observations may capture only the current manifestation of a state, whereas historical labels provide information regarding its temporal evolution.

Consequently, future state estimation can potentially benefit from access to recent state trajectories.

However, historical labels should act as conditioning information rather than replacing multimodal evidence.

The proposed architecture therefore introduces a controlled history-conditioning mechanism with configurable representational capacity.

---

# Design Principles

The module was designed according to the following principles:

## P1. Dataset Agnosticism

The architecture must support arbitrary datasets.

Examples:

- RECOLA
- SEWA
- CASE
- MuSe
- OMG
- ULM-TSST
- Future custom datasets

without architectural changes.

---

## P2. Label Agnosticism

The architecture must operate with:

### Continuous Only

```text
Nc > 0
Nd = 0
```

Example:

```text
Valence
Arousal
```

---

### Discrete Only

```text
Nc = 0
Nd > 0
```

Example:

```text
Emotion Class
```

---

### Mixed Labels

```text
Nc > 0
Nd > 0
```

Example:

```text
Valence
Arousal
Stress

+

Emotion Class
Workload Class
```

---

## P3. Fusion Independence

The module must operate identically for:

- B1
- B2
- B3
- QSAL

without modifying their internal implementation.

---

## P4. Capacity Control

Historical state information must be treated as a contextual prior rather than the dominant information source.

A configurable ratio parameter is therefore introduced to explicitly regulate history representation capacity.

---

# Capacity Scaling Mechanism

Let:

```text
Df
```

denote the fusion output dimension.

In the current framework:

```yaml
transformer_dim: 256
fusion_dim: 256
```

Therefore:

```text
Df = 256
```

---

The history dimension is automatically determined by

```math
H = \frac{D_f}{R}
```

where:

```text
H = history latent dimension
R = configurable capacity ratio
```

---

Examples:

| Ratio R | History Dimension H |
|----------|--------------------|
| 2 | 128 |
| 4 | 64 |
| 8 | 32 |

---

Recommended default:

```yaml
ratio: 4
```

which yields:

```text
H = 64
```

for the current framework.

---

# Complete Architecture

## Multimodal Branch

The multimodal branch remains unchanged.

### Inputs

```text
Audio
Video
ECG
EDA
EEG
EMG
RESP
BPM
Eye
Text
```

Any subset of modalities may be used.

---

### Encoding

Each modality is independently encoded.

Output:

```text
(B, T, 128)
```

per modality.

---

### Fusion

One of the following fusion strategies is applied:

```text
B1
B2
B3
QSAL
```

Output:

```text
Z_fused ∈ R^(B × T × 256)
```

---

# History Branch

The history branch receives historical label trajectories.

Historical window:

```text
K = history_length
```

Current default:

```yaml
history_length: 50
```

---

## Continuous Labels

Input:

```text
Y_cont ∈ R^(B × K × Nc)
```

Examples:

```text
Valence
Arousal
Dominance
Stress
Workload
Fatigue
```

---

### Continuous Projection

```math
E_{cont}
=
GELU(LayerNorm(W_{cont}Y_{cont}+b))
```

Output:

```text
(B × K × H)
```

---

## Discrete Labels

Input:

```text
Y_disc ∈ N^(B × K × Nd)
```

Examples:

```text
Emotion Class
Stress Class
Workload Class
Alertness Class
Drowsiness Class
```

---

### Embedding Layer

Each categorical variable is embedded independently.

```math
E_{disc}
=
Σ Embedding_i(Y_i)
```

Output:

```text
(B × K × H)
```

---

# Dynamic History Adapter

The Dynamic History Adapter harmonizes all valid label configurations into a common latent space.

---

## Case A

Continuous only

```text
Nc > 0
Nd = 0
```

Output:

```text
(B × K × H)
```

---

## Case B

Discrete only

```text
Nc = 0
Nd > 0
```

Output:

```text
(B × K × H)
```

---

## Case C

Continuous + Discrete

```text
Nc > 0
Nd > 0
```

Concatenation:

```text
(B × K × 2H)
```

Projection:

```text
Linear(2H → H)
```

Output:

```text
(B × K × H)
```

---

# Trajectory Encoder

The unified history sequence is processed using a GRU.

---

## Choice of GRU

The GRU was selected instead of LSTM or TCN because it provides:

### Advantages

- Fewer parameters
- Stable optimization
- Efficient training
- Dynamic sequence length support
- Natural trajectory compression
- Good suitability for medium-scale affective datasets

### Disadvantages

- Less expressive than large Transformers
- Smaller memory capacity than LSTMs

However, these limitations are acceptable because the history branch is designed as a conditioning subsystem rather than the main modeling backbone.

---

## GRU Configuration

```yaml
type: "gru"
num_layers: 2
bidirectional: false
```

Input:

```text
(B × K × H)
```

Output:

```text
(B × H)
```

where the final hidden state represents the condensed trajectory representation.

---

# Stabilization Layer

The history embedding is regularized using:

## Layer Normalization

```text
LayerNorm(H)
```

---

## History Dropout

```yaml
history_dropout: 0.20
```

Implementation:

```text
20% probability
```

of suppressing the entire history embedding during training.

---

# Sensor Bypass Prevention

A common failure mode in history-aware forecasting systems is sensor bypass.

The model learns:

```text
Past Labels
→
Future Labels
```

while ignoring

```text
Audio
Video
ECG
EDA
EEG
...
```

signals.

History Dropout mitigates this issue by forcing the model to maintain predictive capability based on multimodal sensory information.

---

# Post-Fusion History Injection

History conditioning is injected after multimodal fusion.

This preserves:

```text
B1
B2
B3
QSAL
```

without modification.

---

## History Broadcast

The trajectory vector

```text
(B × H)
```

is expanded across temporal dimension:

```text
(B × T × H)
```

---

## Concatenation

```text
[B × T × 256]

+

[B × T × H]

↓

[B × T × (256 + H)]
```

---

## Projection

```text
Linear((256+H) → 256)
```

---

## Normalization

```text
LayerNorm(256)
```

Result:

```text
Z_in ∈ R^(B × T × 256)
```

---

# Temporal Backbone

The conditioned representation is processed by the existing Transformer backbone.

Input:

```text
(B × T × 256)
```

Output:

```text
(B × T × 256)
```

No modifications to the Transformer architecture are required.

---

# Recognition Task

Recognition head:

```text
(B × T × 256)
```

↓

```text
(B × T × OutputDim)
```

Supports:

- Continuous regression
- Discrete classification
- Mixed targets

---

# Forecasting Task

Decoder input:

```text
(B × T × 256)
```

↓

Forecast decoder

↓

```text
(B × Horizon × 256)
```

↓

Forecast head

↓

```text
(B × Horizon × OutputDim)
```

Current default:

```yaml
horizon: 25
```

---

# Configuration

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

# Computational Complexity

Let:

```text
B = batch size
T = sequence length
K = history length
H = history dimension
```

The dominant history operations are:

```text
Dynamic Adapter:
O(B × K × H)

GRU:
O(B × K × H²)

Injection:
O(B × T × (256 + H))
```

Given:

```text
H << 256
```

the additional computational overhead remains relatively small compared to the transformer backbone and multimodal encoders.

---

# Advantages

- Dataset agnostic
- Label agnostic
- Flexible continuous/discrete support
- Fusion independent
- Recognition compatible
- Forecasting compatible
- Minimal modification of baseline framework
- Explicit control over historical capacity
- Strong ablation support
- Easy future extension toward memory modules

---

# Future Extensions

The proposed module serves as the first level of temporal state memory.

Potential future developments include:

1. State Memory Bank
2. Long-Term State Memory
3. Attention-Based History Encoder
4. Hierarchical State Modeling
5. Memory-Augmented Forecasting
6. Meta-Learned History Capacity
7. Multi-Scale Temporal History Encoding

These extensions can be incorporated without modifying the multimodal fusion architectures.