# Synthetic Multimodal Human-State Simulation Framework

## Overview

This document describes the design, motivation, architecture, implementation, and scientific rationale behind the Synthetic Multimodal Human-State Simulation Framework developed for the thesis:

> Hybrid Quantum-Classical Multimodal Continuous Emotion Recognition and Trajectory Forecasting

The framework is designed to generate realistic long-duration multimodal recordings with continuous and discrete labels for emotion, stress, workload, fatigue, drowsiness, alertness, and related human-state variables.

The generated dataset is intended for:

- Continuous Emotion Recognition
- Continuous Emotion Forecasting
- Stress Recognition
- Stress Forecasting
- Workload Estimation
- Fatigue Monitoring
- Drowsiness Detection
- Alertness Monitoring
- Human-State Trajectory Forecasting

---

# Motivation

Traditional implementations typically assume:

- Fixed modalities
- Fixed labels
- Fixed datasets
- Simplified random signals

Examples:

```text
Audio
Video
ECG
EDA
```

and

```text
Valence
Arousal
```

only.

Such implementations become difficult to extend when:

- EEG is added
- EMG is added
- Fatigue prediction is added
- Workload estimation is added
- Drowsiness detection is added
- New datasets are introduced

The goal of this framework is to remove these limitations.

---

# Design Objectives

The framework must be:

## 1. Modality-Agnostic

Support arbitrary modalities:

```text
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
Radar
IMU
Future Sensors
```

without code modifications.

---

## 2. Label-Agnostic

Support arbitrary labels.

Continuous:

```text
Valence
Arousal
Stress
Workload
Fatigue
Dominance
Engagement
```

Discrete:

```text
Emotion Class
Stress Level
Workload Level
Drowsiness Level
Alertness Level
KSS Score
```

through configuration only.

---

## 3. Dataset-Agnostic

Support:

```text
RECOLA

SEWA

Synthetic ULM-TSST

Future Datasets
```

with a unified data interface.

---

## 4. Forecasting-Capable

Support:

```text
Recognition

Forecasting
```

using long temporal recordings.

No artificial future labels are generated.

Future states naturally exist within the trajectory and are extracted through temporal window sampling.

---

# Scientific Philosophy

Human behavior is modeled as a causal hierarchy:

```text
Subject Characteristics
            ↓

Experimental Conditions
            ↓

Psychological State
            ↓

Physiological State
            ↓

Sensor Observations
            ↓

Machine Learning Models
```

This structure is more realistic than directly generating sensor signals from labels.

---

# Framework Architecture

```text
Subject Model
      ↓

TSST Protocol
      ↓

Latent Human State
      ↓

Physiological State
      ↓

Sensor Layer
      ↓

Synthetic Dataset
      ↓

Synchronizer
      ↓

Window Sampler
      ↓

Collate Function
      ↓

Feature Encoders
      ↓

Fusion
      ↓

Transformer
      ↓

Recognition
+
Forecasting
```

---

# Subject Model Layer

File:

```text
subject_model.py
```

Purpose:

Generate realistic inter-subject variability.

Each subject receives unique parameters:

```text
baseline_valence

baseline_arousal

stress_gain

workload_gain

fatigue_gain

recovery_rate

resting_heart_rate

resting_eda

eeg_alpha_gain

voice_pitch_base

blink_rate_base
```

These parameters remain consistent across recordings for the same subject.

---

# TSST Protocol Layer

File:

```text
tsst_protocol.py
```

Purpose:

Create realistic stress-inducing experimental scenarios.

Simulated phases:

```text
Baseline

Preparation

Speech

Mental Arithmetic

Recovery
```

Generated trajectories:

```text
stress_load(t)

workload_load(t)

fatigue_load(t)
```

These dynamics drive later stages of the simulation.

---

# Latent Human State Layer

File:

```text
latent_state_generator.py
```

Purpose:

Generate unobservable psychological states.

Generated trajectories:

```text
valence(t)

arousal(t)

stress(t)

workload(t)

fatigue(t)

dominance(t)

engagement(t)
```

Characteristics:

- Continuous
- Stochastic
- Subject-dependent
- Forecastable
- Long-term evolved

These states represent the true hidden condition of the participant.

---

# Continuous Label Layer

File:

```text
continuous_label_generator.py
```

Generates continuous supervision targets:

```text
Valence

Arousal

Stress

Workload

Fatigue

Dominance

Engagement
```

Only enabled labels are produced.

---

# Discrete Label Layer

File:

```text
discrete_label_generator.py
```

Generates:

```text
Emotion Classes

Stress Classes

Workload Classes

Alertness Classes

Drowsiness Classes

KSS Scores
```

These labels are derived from latent trajectories.

This ensures consistency between continuous and discrete annotations.

---

# Physiological State Layer

File:

```text
physiological_generator.py
```

Purpose:

Translate psychological state into physiological processes.

Generated variables:

```text
Heart Rate

Heart Rate Variability

Sympathetic Activation

Parasympathetic Activation

Respiration Rate

Muscle Tension

Pupil Size

Cortical Activation
```

This layer forms the bridge between psychology and sensors.

---

# Sensor Layer

File:

```text
sensor_generator.py
```

Generates:

```text
ECG
EDA
RESP
BPM
EEG
EMG
Eye
Audio
Video
Text
```

All modalities share physiological origins.

This creates realistic cross-modal correlations.

---

# Native Sampling Frequencies

Unlike simplified datasets, modalities are produced with realistic frequencies:

```text
ECG     1000 Hz

EDA      128 Hz

EEG      256 Hz

RESP      64 Hz

EMG     1000 Hz

Eye      120 Hz

Video     25 Hz

Audio    100 Hz

Text       5 Hz
```

Signals therefore naturally differ in temporal resolution.

---

# Synchronization Layer

File:

```text
synchronizer.py
```

Purpose:

Convert native-frequency sensor streams into a shared temporal grid.

Example:

```text
ECG      1000 Hz
EDA       128 Hz
Video      25 Hz
```

↓

```text
Unified Sequence Length T
```

This representation can be consumed by transformers.

---

# Window Sampling Layer

File:

```text
window_sampler.py
```

Purpose:

Create recognition and forecasting tasks.

Given a long trajectory:

```text
[t0 ---------------------- tN]
```

history window:

```text
[t-H ... t]
```

recognition target:

```text
y(t)
```

forecast target:

```text
[t+1 ... t+F]
```

---

# Batch Construction Layer

File:

```text
collate_fn.py
```

Creates training batches while preserving:

- modality flexibility
- label flexibility
- forecasting compatibility

---

# Why Long Clips?

The framework does not generate separate:

```text
history labels

future labels
```

because these already exist in a sufficiently long time-series recording.

This mirrors real-world data.

Forecasting emerges naturally through temporal window extraction.

---

# Research Contributions

The framework contributes:

1. A modality-agnostic multimodal simulation pipeline.

2. A label-agnostic supervision framework.

3. Joint continuous and discrete labeling.

4. Long-duration human-state trajectories.

5. Native-frequency multimodal recordings.

6. Physiologically grounded signal generation.

7. Forecasting-compatible temporal dynamics.

8. A reusable foundation for future multimodal affective computing research.

---

# Future Extensions

Potential future additions:

```text
Radar

Thermal Imaging

Wearables

Motion Sensors

Trust Modeling

Cognitive State Modeling

Driver Monitoring

Patient Monitoring
```

The architecture can accommodate these additions without fundamental redesign.