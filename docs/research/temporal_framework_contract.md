# Temporal Framework Contract

## Purpose

This document defines the official temporal contract adopted throughout the thesis.

Whereas `emotion_temporal_resolution.md` explains *why* a 2 Hz emotional representation was selected, this document specifies the operational parameters used during preprocessing, dataset generation, multimodal learning, recognition, forecasting, and evaluation.

The objective is to establish a consistent temporal framework that is shared across:

- CASE
- OMG-Emotion
- Baseline B1
- Baseline B2
- Baseline B3
- QSAL

This document is considered the single source of truth for all temporal parameters used throughout the experimental pipeline.

---

# 1. Frozen Temporal Contract

```yaml
temporal:
  frequency_hz: 2.0
  step_duration_sec: 0.5

dataset:
  observation_window_t: 10
  history_window_k: 10
  forecast_horizon_h: 10
  stride: 1

continuous_labels:
  - valence
  - arousal

discrete_labels:
  case: []
  omg:
    - emotion_class
```

---

# 2. Temporal Interpretation

## 2.1 Emotional Representation Frequency

```yaml
frequency_hz: 2.0
step_duration_sec: 0.5
```

Meaning:

```text
1 timestep
=
500 ms
```

The framework therefore represents emotional state trajectories at:

```text
2 emotional states per second
```

This frequency is shared across all datasets after preprocessing and harmonization.

---

## 2.2 Observation Window (T)

```yaml
observation_window_t: 10
```

Meaning:

```text
10 steps
×
0.5 seconds
=
5.0 seconds
```

The model observes the previous five seconds of multimodal sensory evidence before generating recognition and forecasting outputs.

Sources of sensory evidence may include:

- ECG
- EDA
- Respiration
- EMG
- EEG
- Audio
- Video
- Text

depending on dataset availability.

---

## 2.3 History Window (K)

```yaml
history_window_k: 10
```

Meaning:

```text
10 steps
×
0.5 seconds
=
5.0 seconds
```

The history subsystem receives five seconds of previous affective trajectory information.

Example:

```text
Valence(t−5s : t)

Arousal(t−5s : t)
```

The objective is to provide sufficient temporal context for estimating:

- emotional momentum
- temporal trends
- affective velocity
- affective acceleration

while avoiding excessive dependence on distant emotional history.

---

## 2.4 Forecast Horizon (H)

```yaml
forecast_horizon_h: 10
```

Meaning:

```text
10 steps
×
0.5 seconds
=
5.0 seconds
```

The forecasting subsystem predicts emotional trajectories up to five seconds into the future.

Importantly:

```text
Only a single model is trained.
```

The system does not train separate forecasting models for:

```text
H = 4
H = 6
H = 10
```

Instead, one model predicts:

```text
t+1
t+2
...
t+10
```

simultaneously.

---

## 2.5 Sliding Window Stride

```yaml
stride: 1
```

Meaning:

```text
1 step
=
0.5 seconds
```

Adjacent training windows overlap heavily.

Example:

```text
Window 1:
0.0s → 15.0s

Window 2:
0.5s → 15.5s

Window 3:
1.0s → 16.0s
```

This maximizes training sample density while preserving temporal continuity.

---

# 3. Evaluation Horizons

Although the forecasting model predicts all future steps up to:

```text
H = 10
```

performance is reported at three primary forecasting horizons.

## Short-Term Horizon

```yaml
H: 4
```

Meaning:

```text
2.0 seconds
```

Represents immediate emotional anticipation.

---

## Medium-Term Horizon

```yaml
H: 6
```

Meaning:

```text
3.0 seconds
```

Represents the primary operational forecasting horizon used throughout the thesis.

This horizon aligns with typical proactive intervention and driver-state adaptation timescales discussed within intelligent transportation and affective systems literature.

---

## Long-Term Horizon

```yaml
H: 10
```

Meaning:

```text
5.0 seconds
```

Represents long-range emotional trajectory forecasting.

This horizon evaluates whether emotional momentum persists over extended future intervals.

---

# 4. Label Space Contract

## Continuous Labels

The primary supervisory targets are:

```yaml
continuous_labels:
  - valence
  - arousal
```

Reason:

```text
CASE:
Valence + Arousal

OMG:
Valence + Arousal
```

This creates a shared affective space across both benchmark datasets.

---

## Discrete Labels

### CASE

```yaml
discrete_labels: []
```

No native frame-level categorical labels are used.

---

### OMG-Emotion

```yaml
discrete_labels:
  - emotion_class
```

Used only as an optional auxiliary classification task.

It is not part of the primary thesis evaluation protocol.

---
# 5. Scientific Basis for Temporal Parameters

Although the scientific rationale for selecting a 2 Hz emotional representation is discussed in detail within `emotion_temporal_resolution.md`, the present section summarizes the reasoning behind the adopted temporal parameters.

## 5.1 Operational Frequency

```yaml
frequency_hz: 2.0
step_duration_sec: 0.5
```

The selected temporal resolution corresponds to:

```text
1 timestep = 500 ms
```

This operating frequency was selected because:

- Cognitive appraisal typically occurs over 500–1500 ms.
- Conscious emotional awareness develops over several hundred milliseconds to seconds.
- Emotion-related electrodermal responses exhibit onset latencies of approximately 1–3 seconds.
- Cardiac affective regulation occurs on multi-second timescales.
- Continuous affective computing systems commonly employ temporally smoothed affect trajectories rather than raw annotation frequencies.

Consequently, 2 Hz provides an appropriate compromise between physiological realism, annotation reliability, forecasting utility, and cross-dataset compatibility.

---

## 5.2 Observation Window Selection

```yaml
observation_window_t: 10
```

Equivalent duration:

```text
5.0 seconds
```

A five-second observation window was selected because it is sufficiently long to capture:

- complete electrodermal responses,
- short-term affective context,
- facial expression evolution,
- speech prosody variation,
- conversational emotional cues.

Shorter windows risk losing important emotional context, whereas substantially longer windows increase computational complexity and may incorporate irrelevant historical information.

---

## 5.3 History Window Selection

```yaml
history_window_k: 10
```

Equivalent duration:

```text
5.0 seconds
```

The history subsystem receives previous affective state trajectories.

A five-second affective history was selected to allow estimation of:

- emotional momentum,
- trajectory trends,
- affective velocity,
- affective acceleration.

Longer histories provide diminishing returns while increasing the risk that unrelated emotional events influence current predictions.

---

## 5.4 Forecast Horizon Selection

```yaml
forecast_horizon_h: 10
```

Equivalent duration:

```text
5.0 seconds
```

A five-second forecasting horizon was selected because it represents a practical compromise between predictive difficulty and forecasting reliability.

Shorter horizons provide limited operational value, whereas substantially longer horizons become increasingly uncertain due to the non-stationary nature of emotional processes and external environmental influences.

Consequently, a 5-second prediction interval was considered sufficiently long to demonstrate anticipatory affective modeling while remaining scientifically plausible.
Forecast horizons substantially beyond five seconds become increasingly uncertain due to non-stationary human behavior and external environmental influences.

The chosen horizon therefore provides a balance between predictive usefulness and forecasting reliability.

---

## 5.5 Evaluation Horizons

Although optimization is performed over the full future trajectory:

```text
H = 10
```

performance is reported at three operational scales:

| Horizon | Duration | Interpretation |
|----------|----------|---------------|
| H = 4 | 2.0 s | Short-term anticipation |
| H = 6 | 3.0 s | Primary operational horizon |
| H = 10 | 5.0 s | Long-term trajectory prediction |

This enables analysis of forecasting performance across progressively more challenging prediction intervals.
---


# 6. Scientific Justification Summary

| Parameter | Frozen Value | Physical Meaning |
|------------|------------|------------|
| Frequency | 2 Hz | One emotional state every 500 ms |
| Observation Window (T) | 10 steps | 5 seconds sensory context |
| History Window (K) | 10 steps | 5 seconds affective history |
| Forecast Horizon (H) | 10 steps | 5 seconds future prediction |
| Stride | 1 step | 500 ms shift |
| Continuous Labels | Valence, Arousal | Common representation across CASE and OMG |
| Discrete Labels | None (CASE) / Emotion Class (OMG) | Auxiliary task only |

---

# 7. Architectural Interpretation

The forecasting framework operates according to the following logic:

```text
Past Sensory Context
        │
        ▼
 Observation Window
      T = 10
      (5 sec)
        │
        ▼
 Multimodal Fusion
        │
        ▼
 History Adapter
      K = 10
      (5 sec)
        │
        ▼
 Temporal Transformer
        │
        ▼
 Recognition Output
        │
        ├───────────────┐
        ▼               ▼
 Current State     Future Trajectory
 Prediction        H = 10
                   (5 sec)
```

---

# 8. Experimental Freeze Statement

The temporal methodology is considered frozen upon adoption of the following parameters:

```yaml
frequency_hz: 2.0

observation_window_t: 10

history_window_k: 10

forecast_horizon_h: 10

stride: 1

continuous_labels:
  - valence
  - arousal
```

Subsequent experiments must preserve this configuration unless an explicit ablation study is being conducted.

Any modification to these parameters constitutes a new temporal hypothesis and must therefore be reported separately.