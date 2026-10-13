# Training Framework

## Overview

The `training/` module is responsible for optimizing, evaluating, validating, and managing experiments for all models in the framework.

It provides a unified training infrastructure that is shared by:

- Baseline B1
- Baseline B2
- Baseline B3
- QSAL
- Future models

Instead of creating a separate trainer for every architecture, the framework uses a common training pipeline to ensure fair experimental comparison, easier maintenance, and higher reproducibility.

---

# Why A Unified Training Framework?

Without a unified design, each model would require its own:

```text
Training Loop

Loss Computation

Evaluation Metrics

Optimizer Setup

Scheduler Setup

Checkpoint Logic

Early Stopping Logic
```

This leads to:

```text
Code Duplication

Harder Maintenance

Inconsistent Experiments

Difficult Comparison Between Models
```

To avoid this, every model shares exactly the same training infrastructure.

Only the model architecture changes.

---

# Design Goals

The training framework was designed to be:

### Dataset Agnostic

No dataset-specific logic exists inside the trainer.

The trainer does not know whether the data originates from:

- RECOLA
- SEWA
- MuSe
- ULM-TSST
- Future datasets

---

### Modality Agnostic

The trainer never assumes the existence of specific modalities.

Examples:

```text
ECG

EDA

Audio

Video

EEG

Eye Tracking
```

All modalities are treated generically.

---

### Label Agnostic

No target names are hardcoded.

Examples:

```text
Valence

Arousal

Stress

Fatigue

Workload

Emotion
```

can be added, removed, or replaced through configuration without changing training code.

---

### Recognition Compatible

Supports prediction of the current state:

```text
y(t)
```

Examples:

```text
Current Emotion

Current Stress

Current Fatigue
```

---

### Forecasting Compatible

Supports prediction of future states:

```text
y(t+1)

y(t+2)

...

y(t+H)
```

Examples:

```text
Future Stress

Future Fatigue

Future Emotion
```

---

### Continuous Label Compatible

Supports regression targets such as:

```text
Valence

Arousal

Stress

Workload

Fatigue
```

---

### Discrete Label Compatible

Supports classification targets such as:

```text
Emotion Class

Stress Class

Alertness Class

Drowsiness Class

KSS
```

---

# Training Architecture

The training subsystem operates using the following workflow:

```text
Dataset
        ↓

Window Sampler
        ↓

Collate Function
        ↓

DataLoader
        ↓

Model
        ↓

Forward Pass
        ↓

Loss Computation
        ↓

Backpropagation
        ↓

Optimizer
        ↓

Scheduler
        ↓

Validation
        ↓

Metrics
        ↓

Checkpoint Saving
```

---

# Directory Structure

```text
training/

├── base_training_script.py
├── trainer.py
├── losses.py
├── metrics.py
├── optimizer.py
├── scheduler.py

├── train_baseline_b1.py
├── train_baseline_b2.py
├── train_baseline_b3.py
└── train_qsal.py
```

---

# Core Components

## base_training_script.py

### Purpose

Provides reusable infrastructure shared by all training entry points.

Responsibilities:

```text
Run Directory Creation

Device Selection

Dataset Construction

Window Dataset Creation

DataLoader Construction

Optimizer Creation

Scheduler Creation

Trainer Creation

Experiment Execution
```

### Why It Exists

Without this file:

```text
train_baseline_b1.py

train_baseline_b2.py

train_baseline_b3.py

train_qsal.py
```

would contain nearly identical code.

Centralizing common functionality significantly reduces duplication and simplifies maintenance.

---

## trainer.py

### Purpose

Implements the actual training and validation process.

Responsibilities:

```text
Forward Pass

Loss Computation

Gradient Updates

Validation

Metric Evaluation

Checkpoint Saving

Early Stopping
```

### Why A Single Trainer?

Using a shared trainer guarantees that performance differences originate from model architecture rather than differences in training procedures.

This ensures fair comparison between:

```text
Baseline B1

Baseline B2

Baseline B3

QSAL
```

---

## losses.py

### Purpose

Defines all optimization objectives.

The framework supports:

```text
Recognition Continuous

Recognition Discrete

Forecast Continuous

Forecast Discrete
```

within a single unified loss function.

---

### Continuous Targets

Continuous labels use:

```text
CCC Loss

+

MSE Loss
```

### Why CCC?

The Concordance Correlation Coefficient (CCC) is widely used in affective computing because it measures:

```text
Correlation

+

Agreement Between Distributions
```

simultaneously.

This makes it more suitable than plain MSE for emotion prediction tasks.

---

### Why MSE Is Added

Pure CCC optimization can be unstable.

Adding a small MSE component:

```text
Improves Optimization Stability

Improves Convergence
```

while preserving CCC performance.

---

### Discrete Targets

Classification tasks use:

```python
CrossEntropyLoss
```

for both:

```text
Recognition

Forecasting
```

tasks.

---

## metrics.py

### Purpose

Provides evaluation metrics for model performance.

---

### Continuous Metrics

The framework computes:

```text
CCC

PCC

MAE

MSE

RMSE
```

---

#### CCC

Measures agreement between predicted and ground-truth trajectories.

Primary metric for affective computing experiments.

---

#### PCC

Measures linear correlation.

---

#### MAE

Measures absolute prediction error.

---

#### MSE

Measures squared prediction error.

---

#### RMSE

Provides interpretable error magnitude.

---

### Discrete Metrics

The framework computes:

```text
Accuracy
```

for:

```text
Recognition Classification

Forecast Classification
```

tasks.

---

## optimizer.py

### Purpose

Creates optimizers from configuration.

Supported optimizers may include:

```text
Adam

AdamW

SGD
```

### Why Separate It?

Optimizer configuration should not be hardcoded inside training scripts.

This design improves:

```text
Flexibility

Experiment Management

Maintainability
```

---

## scheduler.py

### Purpose

Controls learning-rate scheduling.

Supported schedulers may include:

```text
StepLR

ReduceLROnPlateau

CosineAnnealing
```

### Why Use Learning Rate Scheduling?

Training benefits from progressively reducing the learning rate:

```text
Large Learning Rate
        ↓
Exploration

Small Learning Rate
        ↓
Fine Tuning
```

Schedulers automate this process.

---

# Training Entry Points

Each model maintains its own executable training script while sharing the same training infrastructure.

---

## train_baseline_b1.py

Runs:

```text
Baseline B1
```

Classical attention fusion baseline.

Purpose:

```text
Classical Attention Reference
```

---

## train_baseline_b2.py

Runs:

```text
Baseline B2
```

Concatenation fusion baseline.

Purpose:

```text
Classical Feature Fusion Reference
```

---

## train_baseline_b3.py

Runs:

```text
Baseline B3
```

Parameterized quantum circuit fusion baseline.

Purpose:

```text
Quantum Fusion Reference
```

Provides comparison between:

```text
Classical Fusion

vs

Quantum Fusion
```

---

## train_qsal.py

Runs:

```text
QSAL
```

The proposed thesis architecture.

Purpose:

```text
Final Research Model
```

and serves as the primary model evaluated against all baselines.

---

# Checkpointing

The framework automatically stores the best-performing model.

Saved artifact:

```text
best_model.pt
```

Selection criterion:

```text
Lowest Validation Loss
```

---

# Early Stopping

Training terminates when validation performance no longer improves.

Benefits:

```text
Reduced Overfitting

Reduced Training Time

Automatic Model Selection
```

---

# Why This Design Was Chosen

The training subsystem was intentionally designed as a shared framework rather than independent model-specific pipelines.

This approach provides:

✅ Fair experimental comparison

✅ Minimal code duplication

✅ Easier maintenance

✅ Improved reproducibility

✅ Faster experimentation

✅ Cleaner research code

✅ Easier future extensions

✅ Support for both recognition and forecasting

✅ Support for both continuous and discrete labels

As a result, the training framework can support the entire thesis architecture while remaining scalable for future datasets, modalities, labels, and model architectures.