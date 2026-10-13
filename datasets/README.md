# Synthetic Multimodal Dataset Framework

This directory contains the complete synthetic multimodal human-state simulation framework used by the thesis.

---

# Pipeline

```text
Subject Model
      ↓

TSST Protocol
      ↓

Latent Human State
      ↓

Physiological State
      ↓

Sensor Simulation
      ↓

Synthetic Dataset
      ↓

Synchronizer
      ↓

Window Sampler
      ↓

Collate Function
```

---

# Components

## subject_model.py

Generates subject-specific profiles.

Examples:

- stress sensitivity
- recovery rate
- resting heart rate
- baseline valence

---

## tsst_protocol.py

Generates TSST experimental phases.

Produces:

- stress load
- workload load
- fatigue load

---

## latent_state_generator.py

Generates hidden psychological trajectories.

Outputs:

- valence
- arousal
- stress
- workload
- fatigue
- dominance
- engagement

---

## continuous_label_generator.py

Creates continuous labels used for regression tasks.

Examples:

- valence
- arousal
- stress
- workload

---

## discrete_label_generator.py

Creates classification labels.

Examples:

- emotion class
- stress level
- drowsiness level
- KSS score

---

## physiological_generator.py

Maps latent human state to physiological processes.

Examples:

- heart rate
- HRV
- respiration
- muscle tension

---

## sensor_generator.py

Generates multimodal sensor observations.

Modalities:

- ECG
- EDA
- EEG
- EMG
- RESP
- BPM
- Eye
- Audio
- Video
- Text

---

## ulm_tsst_synthetic.py

Main dataset orchestrator.

Combines all framework components.

---

## synchronizer.py

Resamples modalities to a common temporal axis.

---

## window_sampler.py

Extracts history and forecasting windows from long recordings.

---

## collate_fn.py

Creates training batches.

---

# Dataset Output Format

```python
{
    "modalities": {},
    "continuous_labels": {},
    "discrete_labels": {},
    "masks": {},
    "metadata": {}
}
```

---

# Key Features

- Modality Agnostic
- Label Agnostic
- Dataset Agnostic
- Forecasting Compatible
- Continuous Labels
- Discrete Labels
- Native Frequency Signals
- Long-Duration Recordings
- Multimodal Correlation Modeling

---

# Thesis Usage

This framework serves as the synthetic benchmark for:

> Hybrid Quantum-Classical Multimodal Continuous Emotion Recognition and Trajectory Forecasting

