# Unified Concatenation Fusion Framework

## Baseline B2

---

# 1. Purpose

The Unified Concatenation Fusion Framework implements the fusion mechanism used in Baseline B2.

Its purpose is to construct a unified multimodal frame representation by:

1. Projecting each source embedding into a shared lower-dimensional space.
2. Concatenating projected representations.
3. Mapping the concatenated representation to the Transformer model space.

Unlike attention-based fusion mechanisms, no explicit cross-source interaction modeling is performed.

The responsibility of the fusion layer is limited to feature aggregation.

---

# 2. Position in Architecture

Overall Pipeline

Raw Signals
    ↓
Encoders
    ↓
Embeddings
    ↓
Unified Concatenation Fusion
    ↓
Transformer
    ↓
Recognition Head
    ↓
Forecasting Head

This module acts as the bridge between:

Embeddings

and

Temporal Modeling

---

# 3. Design Philosophy

The framework assumes that modality-specific encoders have already learned meaningful representations.

Fusion therefore focuses on:

Feature Aggregation

rather than

Feature Interaction Modeling.

No attention mechanism is applied.

No quantum operations are applied.

No source-to-source interaction modeling is performed.

---

# 4. Design Goals

✅ Dataset Agnostic

✅ Label Agnostic

✅ Continuous Labels

✅ Discrete Labels

✅ Recognition Compatible

✅ Forecasting Compatible

✅ Variable Source Count

✅ Future Proof

---

# 5. Input Representation

Input:

Embeddings

{
    source_name:
        (B,T,128)
}

Examples

Physiological Sources

- ECG
- EDA
- EEG
- EMG
- RESP
- BPM

Behavioral Sources

- Eye
- Audio
- Video
- Text

Future Sources

- Continuous Label History
- Discrete Label History

The fusion framework treats all sources identically.

---

# 6. Source Projection

Each input embedding is projected independently.

Input:

(B,T,128)

↓

Linear Projection

↓

(B,T,64)

A dedicated projection layer is maintained for every source.

Example:

ECG

128 → 64

Audio

128 → 64

Video

128 → 64

Text

128 → 64

This allows source-specific adaptation while maintaining a common dimensionality.

---

# 7. Concatenation Fusion

After projection, all active sources are concatenated along the feature dimension.

Example

4 Sources

↓

4 × 64

↓

256

Result

(B,T,256)

For larger source sets:

10 Sources

↓

10 × 64

↓

640

Result

(B,T,640)

The concatenation operation itself performs no interaction modeling.

It only aggregates source information.

---

# 8. Transformer Alignment

The concatenated representation is projected into the shared Transformer space.

Input

(B,T,K×64)

↓

Linear Projection

↓

(B,T,256)

where:

K = Number of Active Sources

This guarantees a fixed Transformer input dimensionality.

---

# 9. Output Representation

Output:

(B,T,256)

This representation becomes the input sequence processed by the Transformer.

Each temporal frame therefore corresponds to a single multimodal token.

---

# 10. Recognition Compatibility

The framework supports continuous prediction tasks including:

- Valence
- Arousal
- Stress
- Workload
- Fatigue
- Dominance
- Engagement

and discrete prediction tasks including:

- Emotion Class
- Stress Class
- Workload Class
- Alertness Class
- Drowsiness Class
- KSS Score

No fusion modifications are required.

---

# 11. Forecasting Compatibility

The same fused representation can be used for temporal forecasting.

Examples:

Past Stress
    ↓
Future Stress

Past Valence
    ↓
Future Valence

Past Multimodal State
    ↓
Future Emotional Trajectory

The fusion layer remains independent of forecasting horizon.

---

# 12. Dataset Independence

The framework does not depend on:

- RECOLA
- SEWA
- ULM-TSST

or any future dataset.

Only embeddings are required.

This enables direct reuse across datasets.

---

# 13. Label Independence

The fusion layer never accesses labels directly.

Continuous labels and discrete labels remain external targets.

Future label-history embeddings may be introduced without modifying the fusion algorithm.

---

# 14. Computational Characteristics

Input

(B,T,K,128)

Intermediate

(B,T,K,64)

Concatenated

(B,T,K×64)

Output

(B,T,256)

Core Operations

- Source-Specific Projection
- Feature Concatenation
- Transformer Alignment Projection

No attention operations are used.

No quantum operations are used.

---

# 15. Advantages

The Unified Concatenation Fusion Framework provides:

- Simple implementation
- Stable optimization
- Efficient computation
- Fixed Transformer dimensionality
- Variable source support
- Recognition compatibility
- Forecasting compatibility
- Dataset independence

---

# 16. Summary

The Unified Concatenation Fusion Framework implements Baseline B2.

Input:

(B,T,K,128)

↓

Source Projection

↓

(B,T,K,64)

↓

Concatenation

↓

(B,T,K×64)

↓

Transformer Alignment

↓

(B,T,256)

The framework is:

✅ Dataset Agnostic

✅ Label Agnostic

✅ Continuous Labels

✅ Discrete Labels

✅ Recognition Compatible

✅ Forecasting Compatible

✅ Variable Source Count

✅ Future Proof