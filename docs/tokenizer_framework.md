# Tokenizer Framework

## Purpose

The Tokenizer Framework converts encoder outputs into a unified token representation suitable for:

- Classical Fusion
- Quantum Fusion
- Transformer Models
- Recognition Tasks
- Forecasting Tasks

The tokenizer serves as the bridge between:

Encoder Layer

and

Fusion Layer.

---

# Architecture Position

Raw Signals
    ↓
Encoders
    ↓
Embeddings
    ↓
Tokenizer
    ↓
Fusion
    ↓
Transformer
    ↓
Heads

---

# Motivation

Different modalities are encoded independently.

Examples:

ECG

(B,T,128)

Audio

(B,T,128)

Video

(B,T,128)

Although all embeddings share the same dimensionality, they do not explicitly indicate their modality origin.

The tokenizer injects modality identity information through learnable token-type embeddings.

---

# Universal Token Interface

Input:

{
    token_name:
        (B,T,D)
}

Output:

(B,T,N,D)

where:

B = batch size

T = sequence length

N = number of active tokens

D = embedding dimension

---

# Supported Token Types

Current:

Physiological

- ecg
- eda
- eeg
- emg
- resp
- bpm

Behavioral

- eye
- audio
- video
- text

Future:

- continuous_labels
- discrete_labels

Special Tokens

- task
- cls

---

# Token Type Embeddings

Each token type receives a learnable embedding.

Example:

ECG Token

    ECG Embedding

        +

    ECG Type Embedding

Audio Token

    Audio Embedding

        +

    Audio Type Embedding

This allows the downstream model to distinguish different information sources.

---

# Continuous Labels

The architecture is designed for future continuous-label tokens.

Examples:

- valence
- arousal
- stress
- workload
- fatigue
- dominance
- engagement

These may later be represented as:

continuous_labels

↓

(B,T,128)

↓

Tokenizer

↓

Token Stream

---

# Discrete Labels

Future extensions support:

- emotion_class
- stress_class
- workload_class
- alertness_class
- drowsiness_class
- kss_score

These may also become token streams.

---

# Recognition Support

Recognition uses:

Past Modalities

↓

Tokenizer

↓

Fusion

↓

Transformer

↓

Prediction

---

# Forecasting Support

Forecasting may use:

Past Modalities

+

Label History

↓

Tokenizer

↓

Fusion

↓

Transformer

↓

Future Predictions

The tokenizer is therefore forecasting-ready.

---

# Design Goals

✅ Dataset Agnostic

✅ Label Agnostic

✅ Continuous Labels

✅ Discrete Labels

✅ Variable Modality Count

✅ Recognition Compatible

✅ Forecasting Compatible

✅ Missing Token Compatible

✅ Future Proof

---

# Summary

The tokenizer converts arbitrary embedding streams into a unified token representation:

(B,T,D)

↓

(B,T,N,D)

This representation becomes the common interface between:

Encoders

and

Fusion/Transformer modules.

The design supports both current modalities and future label-history tokens without requiring architectural changes.