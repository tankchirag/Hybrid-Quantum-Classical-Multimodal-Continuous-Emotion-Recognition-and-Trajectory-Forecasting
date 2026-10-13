# Classical Fusion Framework (Baseline B1)

## Hybrid Quantum-Classical Multimodal Continuous Emotion Recognition and Trajectory Forecasting

---

# 1. Purpose

The Classical Fusion Framework represents the first baseline fusion strategy used within the proposed architecture.

Its purpose is to combine information originating from multiple token sources into a unified temporal representation suitable for:

- Continuous Emotion Recognition
- Discrete Emotion Classification
- Stress Recognition
- Workload Recognition
- Alertness Detection
- Drowsiness Detection
- Trajectory Forecasting

This baseline provides a purely classical reference model against which quantum-enhanced fusion mechanisms can be compared.

---

# 2. Position in Overall Architecture

Complete Pipeline

Raw Signals
    ↓
Encoders
    ↓
Embeddings
    ↓
Tokenizer
    ↓
Classical Fusion (B1)
    ↓
Transformer
    ↓
Recognition Head
    ↓
Forecasting Head

The fusion layer acts as the bridge between token generation and temporal sequence modeling.

---

# 3. Design Philosophy

The fusion layer operates on tokens rather than modalities.

Earlier multimodal architectures often rely on fixed modality assumptions:

Audio
Video
ECG
EDA

The proposed framework instead adopts a generalized token-based representation.

Any information source capable of generating embeddings may participate in fusion.

Examples include:

Physiological Sources

- ECG
- EDA
- EEG
- EMG
- RESP
- BPM

Behavioral Sources

- Eye Tracking
- Audio
- Video
- Text

Future Sources

- Continuous Label History
- Discrete Label History
- Task Tokens
- CLS Tokens

The fusion mechanism remains unchanged regardless of token origin.

---

# 4. Input Representation

Input:

Tokens

Shape:

(B,T,N,D)

where:

B = Batch Size

T = Temporal Length

N = Number of Active Tokens

D = Token Embedding Dimension

Default:

D = 128

Example

(B,T,10,128)

may represent:

ECG
EDA
EEG
EMG
RESP
BPM
Eye
Audio
Video
Text

---

# 5. Token Projection

Before fusion, each token is projected into a dedicated fusion space.

Input:

(B,T,N,128)

↓

Projection

↓

(B,T,N,64)

This projection serves two purposes:

- Dimensionality reduction
- Shared fusion space construction

The projection is implemented as:

Linear(128 → 64)

---

# 6. Self-Attention Fusion

After projection, token interactions are modeled using multi-head self-attention.

For every temporal frame:

Tokens

↓

Multi-Head Self-Attention

↓

Updated Tokens

The attention mechanism allows the model to learn:

- Which tokens are informative
- Inter-token dependencies
- Cross-source interactions

Examples:

Stress-related ECG changes

combined with

Workload-related EEG changes

combined with

Behavioral Eye Activity

---

# 7. Residual Learning

The attention output is combined with the original token representation.

x = x + Attention(x)

Benefits:

- Stable optimization
- Improved gradient flow
- Better convergence

---

# 8. Layer Normalization

Layer normalization follows the residual connection.

Purpose:

- Stabilize training
- Reduce covariate shift
- Improve generalization

Output Shape:

(B*T,N,64)

---

# 9. Token Pooling

After attention, multiple tokens must be converted into a single temporal representation.

Mean pooling is used.

Input:

(B*T,N,64)

↓

Mean over Token Dimension

↓

(B*T,64)

Advantages:

- Computationally efficient
- Modality-count independent
- Stable aggregation strategy
- Strong baseline behavior

---

# 10. Transformer Projection

The pooled representation is projected into the temporal Transformer space.

Input:

(B*T,64)

↓

Projection

↓

(B*T,256)

using:

Linear(64 → 256)

Output:

(B,T,256)

---

# 11. Output Representation

Final Output:

(B,T,256)

This becomes the input sequence consumed by the Transformer encoder-decoder.

The Transformer therefore receives:

One fused temporal representation per time step.

---

# 12. Recognition Compatibility

The fusion framework is task independent.

Supported recognition targets include:

Continuous Targets

- Valence
- Arousal
- Stress
- Workload
- Fatigue
- Dominance
- Engagement

Discrete Targets

- Emotion Class
- Stress Class
- Workload Class
- Alertness Class
- Drowsiness Class
- KSS Score

No changes are required to the fusion block.

---

# 13. Forecasting Compatibility

The same fusion representation supports temporal forecasting.

Example

Past Signals

↓

Classical Fusion

↓

Transformer

↓

Future Stress

Future Workload

Future Valence

Future Fatigue

The fusion layer does not depend on the prediction horizon.

---

# 14. Dataset Independence

The fusion mechanism does not depend on:

- RECOLA
- SEWA
- ULM-TSST

or any future dataset.

Only token embeddings are required.

This makes the architecture reusable across datasets.

---

# 15. Label Independence

The fusion layer never accesses labels directly.

It only processes token embeddings.

Future label-history embeddings may be introduced as token sources without modifying the fusion algorithm.

Therefore the framework remains:

- Label Agnostic
- Continuous-Label Compatible
- Discrete-Label Compatible

---

# 16. Computational Characteristics

Input:

(B,T,N,128)

Output:

(B,T,256)

Core Operations:

- Linear Projection
- Multi-Head Attention
- Residual Learning
- Layer Normalization
- Mean Pooling
- Transformer Projection

No quantum components are used in this baseline.

---

# 17. Advantages

The proposed classical fusion baseline provides:

- Simple implementation
- Stable training
- Strong interpretability
- Efficient computation
- Variable token support
- Recognition support
- Forecasting support
- Dataset independence

making it a suitable reference architecture for evaluating quantum-enhanced fusion approaches.

---

# 18. Summary

The Classical Fusion Framework represents Baseline B1 of the proposed thesis architecture.

Input:

(B,T,N,128)

↓

Self-Attention Fusion

↓

(B,T,256)

Output:

(B,T,256)

The framework is:

✅ Dataset Agnostic

✅ Label Agnostic

✅ Continuous Labels

✅ Discrete Labels

✅ Recognition Compatible

✅ Forecasting Compatible

✅ Variable Token Count

✅ Future Proof

and serves as the primary classical reference against which PQC-based and QSAL-based fusion strategies are evaluated.