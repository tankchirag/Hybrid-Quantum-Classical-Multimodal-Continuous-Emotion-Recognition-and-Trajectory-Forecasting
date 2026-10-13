"""
Show one generated dataset sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from datasets.ulm_tsst_synthetic import (
    SyntheticULMTSSTDataset,
)

# ------------------------------------------------------
# Configuration
# ------------------------------------------------------

MODALITIES = {
    "ecg": True,
    "eda": True,
    "resp": True,
    "bpm": True,
    "eeg": True,
    "emg": True,
    "eye": True,
    "audio": True,
    "video": True,
    "text": True,
}

CONTINUOUS_LABELS = {
    "valence": True,
    "arousal": True,
    "stress": True,
    "workload": True,
    "fatigue": True,
    "dominance": True,
    "engagement": True,
}

DISCRETE_LABELS = {
    "emotion_class": True,
    "stress_class": True,
    "workload_class": True,
    "alertness_class": True,
    "drowsiness_class": True,
    "kss_score": True,
}

SAMPLING_RATES = {
    "ecg": 1000,
    "eda": 128,
    "resp": 64,
    "bpm": 4,
    "eeg": 256,
    "emg": 1000,
    "eye": 120,
    "audio": 100,
    "video": 25,
    "text": 5,
}

# ------------------------------------------------------
# Dataset
# ------------------------------------------------------

dataset = SyntheticULMTSSTDataset(
    num_samples=10,
    duration_seconds=600,
    modalities=MODALITIES,
    continuous_labels=CONTINUOUS_LABELS,
    discrete_labels=DISCRETE_LABELS,
    sampling_rates=SAMPLING_RATES,
    save_dir=None,
    device="cpu",
)

sample = dataset[0]

# ------------------------------------------------------
# Metadata
# ------------------------------------------------------

print("\n" + "=" * 80)
print("METADATA")
print("=" * 80)

for k, v in sample["metadata"].items():
    print(f"{k}: {v}")

# ------------------------------------------------------
# Continuous Labels
# ------------------------------------------------------

print("\n" + "=" * 80)
print("CONTINUOUS LABELS")
print("=" * 80)

for name, signal in sample[
    "continuous_labels"
].items():

    print(f"\n{name}")

    print(
        "shape:",
        tuple(signal.shape)
    )

    print(
        "first 20 values:"
    )

    print(
        signal[:20]
    )

# ------------------------------------------------------
# Discrete Labels
# ------------------------------------------------------

print("\n" + "=" * 80)
print("DISCRETE LABELS")
print("=" * 80)

for name, signal in sample[
    "discrete_labels"
].items():

    print(f"\n{name}")

    print(
        "shape:",
        tuple(signal.shape)
    )

    print(
        "first 20 values:"
    )

    print(
        signal[:20]
    )

# ------------------------------------------------------
# Modalities
# ------------------------------------------------------

print("\n" + "=" * 80)
print("MODALITIES")
print("=" * 80)

for name, signal in sample[
    "modalities"
].items():

    print(f"\n{name}")

    print(
        "shape:",
        tuple(signal.shape)
    )

    print(
        "first 10 rows:"
    )

    print(
        signal[:10]
    )