# Encoder Framework

## Hybrid Quantum-Classical Multimodal Continuous Emotion Recognition and Trajectory Forecasting

---

# 1. Purpose

The Encoder Framework represents the first stage of representation learning within the proposed architecture.

Its responsibility is to convert synchronized multimodal sensor signals into learned latent embeddings suitable for:

- Recognition Tasks
- Forecasting Tasks
- Classical Fusion
- Quantum Fusion
- Transformer Processing

The encoder layer operates independently of:

- Dataset
- Labels
- Tasks

and therefore constitutes a reusable foundation for all experiments.

---

# 2. Design Philosophy

Traditional emotion recognition systems often tightly couple:

- Dataset
- Modalities
- Labels
- Models

This creates significant challenges when:

- introducing new datasets
- adding modalities
- changing prediction targets
- supporting forecasting

The proposed framework separates:

Signal Processing

from

Task Learning

using a universal encoder interface.

The core principle is:

Signal
    →
Representation

The encoder never accesses labels.

---

# 3. Design Goals

The encoder layer was designed to satisfy:

✅ Dataset Agnostic

✅ Label Agnostic

✅ Continuous Labels

✅ Discrete Labels

✅ Recognition Compatible

✅ Forecasting Compatible

✅ Multimodal

✅ Missing-Modality Compatible

✅ Sensor Agnostic

✅ Future Proof

---

# 4. Position in Architecture

Complete Architecture

Raw Sensors
    ↓
Synchronizer
    ↓
Window Sampler
    ↓
Encoder Framework
    ↓
Tokenizer
    ↓
Fusion
    ↓
Transformer
    ↓
Recognition Head
    ↓
Forecasting Head

The encoder framework acts as the interface between:

Raw Signals

and

Learned Representations

---

# 5. Universal Encoder Contract

All encoders follow exactly the same interface.

Input:

(B,T,C)

Output:

(B,T,D)

where:

B = Batch Size

T = Number of Time Steps

C = Number of Input Channels

D = Embedding Dimension

Default:

D = 128

---

# 6. Why a Universal Contract?

Different modalities possess different characteristics.

Example:

ECG

(B,T,1)

EDA

(B,T,1)

Eye Tracking

(B,T,2)

Audio

(B,T,2)

Video

(B,T,2)

Text

(B,T,2)

Without a universal interface, each modality would require:

- custom fusion logic
- custom transformer inputs
- custom training pipelines

The universal encoder contract ensures:

All modalities
        ↓
(B,T,128)

regardless of sensor origin.

---

# 7. BaseEncoder

File:

models/encoders/base_encoder.py

The BaseEncoder defines the common API used by all encoders.

Responsibilities:

- Input validation
- Shape checking
- Metadata management
- Interface standardization

Every encoder inherits from BaseEncoder.

---

# 8. TemporalConvEncoder

File:

models/encoders/temporal_conv_encoder.py

The TemporalConvEncoder serves as the generic backbone used by most modalities.

Architecture:

Input
    ↓
Conv1D
    ↓
BatchNorm
    ↓
GELU
    ↓
Dropout
    ↓
Projection
    ↓
Embedding

Input:

(B,T,C)

Output:

(B,T,128)

Advantages:

- lightweight
- efficient
- modality independent
- easy to extend

---

# 9. Physiological Encoders

The physiological encoder family uses the TemporalConvEncoder backbone.

Implemented Encoders:

- ECGEncoder
- EDAEncoder
- EEGEncoder
- EMGEncoder
- RESPEncoder
- BPMEncoder

Common Input:

(B,T,1)

Common Output:

(B,T,128)

These encoders transform physiological trajectories into latent representations.

---

# 10. Behavioral Encoders

Behavioral modalities use the same representation-learning framework.

Implemented:

- EyeEncoder
- AudioEncoder
- VideoEncoder
- TextEncoder

Examples:

Eye:

(B,T,2)

↓

(B,T,128)

Audio:

(B,T,2)

↓

(B,T,128)

Video:

(B,T,2)

↓

(B,T,128)

Text:

(B,T,2)

↓

(B,T,128)

---

# 11. Multimodal Feature Extractor

File:

models/encoders/multimodal_feature_extractors_v2.py

Purpose:

Execute all active encoders dynamically.

Input:

{
    "ecg",
    "eda",
    "eeg",
    "emg",
    "resp",
    "bpm",
    "eye",
    "audio",
    "video",
    "text"
}

Output:

{
    "ecg":   (B,T,128),
    "eda":   (B,T,128),
    "eeg":   (B,T,128),
    "emg":   (B,T,128),
    "resp":  (B,T,128),
    "bpm":   (B,T,128),
    "eye":   (B,T,128),
    "audio": (B,T,128),
    "video": (B,T,128),
    "text":  (B,T,128)
}

Only enabled modalities are processed.

---

# 12. Label Independence

A key contribution of the framework is complete label independence.

Encoders never access:

Continuous Labels:

- valence
- arousal
- stress
- workload
- fatigue
- dominance
- engagement

or

Discrete Labels:

- emotion_class
- stress_class
- workload_class
- alertness_class
- drowsiness_class
- kss_score

The encoder only processes sensor signals.

This makes the framework reusable across tasks.

---

# 13. Recognition Compatibility

Recognition tasks include:

Regression:

- Valence
- Arousal
- Stress
- Workload

Classification:

- Emotion Class
- Stress Class
- Drowsiness
- Alertness

All recognition tasks use the same encoder outputs.

---

# 14. Forecasting Compatibility

Forecasting predicts future trajectories.

Example:

Past Signals
    ↓
Encoders
    ↓
Transformer
    ↓
Future Stress

Future Valence

Future Workload

Future Fatigue

Because encoders are task independent, no modifications are required for forecasting.

---

# 15. Advantages of the Framework

Compared with dataset-specific pipelines, the proposed architecture provides:

- reusable encoders
- reusable fusion layers
- reusable transformers
- support for future modalities
- support for missing modalities
- support for multiple label sets
- support for continuous prediction
- support for discrete prediction
- support for forecasting

This significantly improves scalability and experimental flexibility.

---

# 16. Summary

The Encoder Framework transforms synchronized multimodal signals into a unified latent representation space.

Input:

(B,T,C)

Output:

(B,T,128)

The framework is:

- Dataset Agnostic
- Label Agnostic
- Modality Agnostic
- Recognition Compatible
- Forecasting Compatible

and serves as the foundation for all subsequent stages of the proposed Hybrid Quantum-Classical Multimodal Learning Architecture.