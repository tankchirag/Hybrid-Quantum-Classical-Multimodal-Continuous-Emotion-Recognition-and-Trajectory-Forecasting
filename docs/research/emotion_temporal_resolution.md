# Temporal Resolution Harmonization for Cross-Domain Emotional Trajectory Learning

## Purpose

This document justifies the temporal representation adopted throughout this thesis.

The proposed framework performs continuous multimodal emotion recognition and future emotional trajectory forecasting across heterogeneous affective computing datasets originating from different domains:

- Physiological affective signals (CASE)
- Behavioral affective signals (OMG-Emotion)

A fundamental challenge is that these datasets operate at substantially different annotation time scales. Therefore, a common emotional representation frequency must be defined before preprocessing, model design, forecasting, and cross-domain evaluation can be performed consistently.

---

# 1. Problem Statement

The selected benchmark datasets represent emotional states using fundamentally different annotation paradigms.

## 1.1 CASE Dataset

The CASE dataset provides:

- Continuous physiological recordings
- Continuous valence annotations
- Continuous arousal annotations
- Annotation frequency: **20 Hz**
- Temporal resolution: **50 ms**

Result:

```text
One emotional label every 50 milliseconds
```

The dataset was collected using a continuous joystick-based annotation interface synchronized with physiological recordings.

## 1.2 OMG-Emotion Dataset

OMG-Emotion follows a substantially different annotation strategy.

Characteristics:

- Long monologue videos
- Audio, video, and language modalities
- Utterance-level annotation
- Gold-standard Evaluator Weighted Estimates (EWE)
- Context-aware emotional labeling

Result:

```text
One emotional label per utterance
```

The effective temporal resolution is determined by utterance duration rather than a fixed sampling frequency.

Unlike CASE, OMG-Emotion does not provide native frame-level continuous valence-arousal trajectories.

---

# 2. The Central Research Problem

One objective of this thesis is to evaluate the same forecasting architecture across both:

```text
Visceral Emotional Dynamics
(Physiology)
```

and

```text
Expressive Emotional Dynamics
(Audio + Video + Text)
```

However, the datasets operate at incompatible temporal scales:

```text
CASE : 20 Hz
OMG  : Utterance-Level
```

Future cross-domain learning requires a unified emotional representation.

Consequently, a temporal harmonization strategy is required before the datasets can be processed within a shared forecasting framework.

---

# 3. Decision Process

The selection of a common emotional representation frequency was not arbitrary. The decision emerged through four sequential stages.

## Stage 1: Dataset Comparison

Supervisor feedback highlighted that:

- CASE operates at 20 Hz annotation frequency.
- OMG operates at utterance-level granularity.
- Direct comparison is therefore problematic.

## Stage 2: Analysis of CASE Annotation Generation

The preprocessing pipeline and annotation procedures of CASE were studied, including:

- Continuous joystick measurements
- Annotation generation methodology
- Interpolation procedures
- Physiological synchronization

The analysis revealed that 20 Hz primarily represents the acquisition rate of the annotation device rather than the natural evolution rate of human emotion.

## Stage 3: Analysis of OMG Annotation Structure

The OMG annotation framework was examined.

Key observations:

- Labels exist at utterance level.
- No native frame-level trajectories are available.
- Gold-standard EWE labels are already provided.

Therefore, OMG possesses no intrinsic continuous emotional sampling frequency.

## Stage 4: Evidence from Psychology and Neuroscience

Literature from:

- Psychology
- Neuroscience
- Psychophysiology
- Continuous affective computing
- Emotion prediction research

was reviewed to address the following question:

```text
How rapidly does human emotion meaningfully evolve?
```

rather than:

```text
How frequently can sensors record signals?
```

This distinction became the foundation for the temporal harmonization strategy.

---

# 4. Human Emotion Does Not Evolve at Sensor Frequencies

A critical distinction must be made between signal dynamics and emotional dynamics.

## 4.1 Signal Dynamics

Physiological and behavioral signals may change at very high frequencies.

Examples:

```text
ECG    : 1000 Hz
EEG    : 256-1000 Hz
Audio  : 16000 Hz
```

These frequencies represent measurement capabilities rather than emotional state evolution.

## 4.2 Emotional Dynamics

Valence and arousal evolve significantly more slowly than the underlying signals used to infer them.

Examples include:

- Emotional engagement
- Stress accumulation
- Arousal escalation
- Frustration growth
- Emotional recovery
- Mood drift

These processes unfold over seconds rather than milliseconds.

Consequently:

```text
Sensor Frequency ≠ Emotion Frequency
```

This thesis therefore distinguishes between:

```text
Signal Sampling Rate
```

and

```text
Emotional Representation Rate
```

which are fundamentally different quantities.

---

# 5. Neuropsychological Foundations

## 5.1 The Fallacy of 20 Hz Emotion

Human emotion does not shift stochastically at 20 Hz.

In the CASE dataset, 20 Hz represents the hardware polling rate of the joystick analog-to-digital converter rather than the actual rate of cognitive-affective change.

Consequently, emotional trajectories should not be interpreted as changing meaningfully every 50 ms simply because annotations were sampled at that rate.

## 5.2 Component Process Model (CPM)

According to Scherer's Component Process Model, emotions emerge through dynamic interactions among:

- Appraisal
- Physiology
- Expression
- Action tendencies
- Subjective feeling

Conceptually:

```text
Emotion(t) =
f(Appraisal,
  Physiology,
  Expression,
  Feeling)
```

Emotions therefore evolve through a distributed process rather than changing instantaneously every few milliseconds.

## 5.3 Cognitive Appraisal Latency

Research in appraisal theory and affective psychology suggests the following approximate timescales:

| Process | Typical Latency |
|----------|----------------|
| Novelty detection | 100-300 ms |
| Cognitive appraisal | 500-1500 ms |
| Conscious emotional awareness | 500-2000 ms |

Meaningful emotional transitions therefore occur on the order of several hundred milliseconds to several seconds rather than every 50 ms.

---

# 6. Human Annotation Latency

Continuous emotion annotation is itself a delayed process.

A participant must:

1. Perceive a stimulus.
2. Perform cognitive appraisal.
3. Decide on an emotional rating.
4. Execute a physical movement.

The combined delay typically spans:

```text
850-1600 ms
```

Motor response latency alone often contributes approximately:

```text
400-800 ms
```

before a joystick movement reflects a perceived emotional change.

Consequently, 20 Hz joystick readings capture a combination of:

- Emotional trends
- Motor behavior
- Annotation inertia
- Mechanical fluctuations

At high sampling rates, successive annotations are therefore highly autocorrelated and partially reflect motor dynamics rather than genuine affective changes.

---

# 7. Physiological Time Constants

Emotional physiology itself evolves relatively slowly.

## 7.1 Electrodermal Activity (EDA)

EDA reflects sympathetic nervous system activation.

Typical dynamics:

| Characteristic | Duration |
|---------------|----------|
| Onset latency | 1-3 s |
| Rise time | 1-3 s |
| Recovery | 2-10 s |

Emotional electrodermal responses therefore evolve over multiple seconds.

Sampling emotional trajectories at 20 Hz substantially exceeds the natural response speed of EDA.

## 7.2 Cardiac Regulation

Heart-rate variability mechanisms typically operate within:

```text
0.04-0.40 Hz
```

which corresponds to oscillatory periods of:

```text
2.5-25 seconds
```

The biologically meaningful emotional dynamics are therefore much slower than raw ECG acquisition frequencies.

Cardiac modulation associated with affective state transitions generally occurs over intervals ranging from approximately:

```text
1-5 seconds
```

rather than tens of milliseconds.

## 7.3 Facial Expressions

Facial affective behavior similarly evolves over relatively slow timescales.

Typical durations:

```text
Micro-expressions : 200-500 ms
Macro-expressions : 0.5-4.0 s
```

Emotionally meaningful facial behavior therefore persists long enough to be represented at sub-second temporal resolutions.

---

# 8. Evidence from Affective Computing Literature

Continuous affect prediction benchmarks rarely operate directly at raw annotation frequencies.

Instead, they commonly employ:

- Delay compensation
- Temporal smoothing
- Window aggregation
- Pooling
- Low-frequency trajectory modeling

Representative examples include:

- RECOLA
- AVEC
- MuSe

These benchmarks treat affective trajectories as slowly varying processes rather than independent frame-level events.

Furthermore, many continuous affect modeling systems operate at effective temporal resolutions near:

```text
2 Hz (500 ms)
```

or

```text
4 Hz (250 ms)
```

after temporal aggregation and delay compensation.

---

# 9. Why 20 Hz Was Not Selected

CASE labels are available at:

```text
20 Hz
```

equivalent to:

```text
1 label every 50 ms
```

Although useful for data acquisition, adopting 20 Hz as the framework standard creates several scientific problems.

## Problem 1: Artificial OMG Upsampling

Using 20 Hz would require extensive interpolation of OMG labels.

## Problem 2: Excessive Synthetic Labels

A 5-second utterance would generate:

```text
100 interpolated labels
```

The model would consequently learn from large numbers of synthetic observations rather than genuine annotations.

## Problem 3: Physiological Unrealism

The selected frequency substantially exceeds the natural timescale of:

- Cognitive appraisal
- Annotation response
- EDA dynamics
- Cardiac regulation
- Facial affective behavior

Therefore, 20 Hz was rejected as the framework-wide emotional representation frequency.

---

# 10. Why Utterance-Level Resolution Was Not Selected

The opposite extreme was also considered.

This would involve reducing CASE to:

```text
One emotional state per utterance
```

However:

- Most temporal emotional information would be discarded.
- Continuous trajectory learning would become difficult.
- Forecasting performance would likely degrade.
- Fine-grained affective dynamics would vanish.

Therefore, utterance-level representation was considered excessively coarse.

---

# 11. Selected Operational Frequency

After evaluating:

- Dataset characteristics
- Psychology literature
- Neuroscience evidence
- Psychophysiological dynamics
- Human annotation latency
- Forecasting requirements
- Affective computing benchmarks

the following operational standard is adopted:

```text
2 Hz
```

Equivalent temporal interval:

```text
Δt = 0.5 seconds
```

between consecutive emotional states.

Importantly, this thesis does **not** claim that human emotion evolves exactly at 2 Hz.

Instead, 2 Hz is adopted as an operational emotional representation frequency that provides the best compromise between:

- Physiological realism
- Annotation reliability
- Forecasting utility
- Dataset compatibility
- Cross-domain transferability

---

# 12. Why 2 Hz is Scientifically Appropriate

## 12.1 Downsampling CASE

Transformation:

```text
20 Hz → 2 Hz
```

A 10:1 reduction combined with smoothing and temporal aggregation acts as a biologically motivated low-pass filter.

Benefits:

- Reduces joystick tremor
- Reduces annotation noise
- Reduces mechanical fluctuations
- Preserves meaningful emotional trends
- Retains autonomic inflection points

## 12.2 Upsampling OMG

Transformation:

```text
Utterance-Level
        ↓
Trajectory Reconstruction
        ↓
       2 Hz
```

Using shape-preserving interpolation (PCHIP), an average:

```text
4-6 second utterance
```

generates roughly:

```text
8-12 trajectory samples
```

which remains sufficiently smooth while avoiding extreme over-interpolation.

This produces a continuous emotional trajectory while minimizing synthetic artifacts.

---

# 13. Dataset Harmonization Strategy

## CASE

Original:

```text
20 Hz
```

Transformation:

```text
20 Hz
  ↓
2 Hz
```

Method:

- Smoothing
- Aggregation
- Downsampling

Output:

```text
Emotion trajectory at 2 Hz
```

## OMG-Emotion

Original:

```text
Utterance-Level Labels
```

Transformation:

```text
Utterance Labels
        ↓
Midpoint Extraction
        ↓
PCHIP Reconstruction
        ↓
2 Hz Trajectory
```

Method:

- Utterance midpoint extraction
- Gold-standard EWE labels
- Shape-preserving interpolation (PCHIP)

Output:

```text
Emotion trajectory at 2 Hz
```

---

# 14. Forecasting Implications

The adopted frequency naturally supports the forecasting framework.

Example configuration:

```text
Observation Window:
T = 10 steps
= 5.0 seconds

History Window:
K = 10 steps
= 5.0 seconds

Maximum Forecast Horizon:
H = 10 steps
= 5.0 seconds

Evaluation Horizons:
H = 4 (2.0 s)
H = 6 (3.0 s)
H = 10 (5.0 s)
```

Such temporal horizons align with practical affective forecasting scenarios while remaining compatible across physiological and behavioral datasets.

A 2 Hz representation provides sufficient temporal resolution for forecasting future emotional trajectories without requiring biologically unrealistic annotation frequencies.

---

# 15. Framework Temporal Contract

The entire thesis adopts the following temporal contract:

```yaml
emotion_sampling_frequency_hz: 2.0
time_step_seconds: 0.5

observation_window_steps: 10
history_window_steps: 10
forecast_horizon_steps: 10

stride_steps: 1
```

All datasets are transformed into this standardized representation before:

- Training
- Recognition
- Forecasting
- Evaluation

---

# 16. Conclusion

CASE and OMG employ fundamentally different annotation paradigms.

CASE provides:

```text
20 Hz continuous annotations
```

whereas OMG provides:

```text
Utterance-level context-aware annotations
```

A common emotional representation frequency is therefore necessary.

Based on evidence from:

- Psychology
- Neuroscience
- Psychophysiology
- Continuous affective computing
- Dataset analysis

this thesis adopts:

```text
2 Hz
```

as the operational emotional representation frequency.

This choice:

- Preserves meaningful emotional dynamics
- Reduces annotation noise
- Minimizes interpolation artifacts
- Enables cross-dataset comparison
- Supports trajectory forecasting
- Establishes a unified foundation for future cross-domain affective modeling

---

# References

1. Sharma et al. (2018). *CASE: A Dataset for Continuous Affective State and Physiological Signal Analysis*.
2. Barros et al. (2018). *OMG-Emotion Challenge Dataset*.
3. Scherer, K. R. *Component Process Model of Emotion*.
4. Ringeval et al. *RECOLA and Continuous Emotion Prediction Literature*.
5. MuSe Challenge Papers.
6. AVEC Challenge Papers.
7. Relevant thesis design and architecture documents.