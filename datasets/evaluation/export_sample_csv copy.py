"""
export_sample_csv.py
====================

Export one synthetic sample into a single
time-aligned CSV file.

Output
------

results/dataset_exports/

    sample_000_aligned.csv

All modalities are resampled onto the
common label timeline.

Columns include:

Time

Continuous Labels

Discrete Labels

ECG
EDA
RESP
BPM
EEG
EMG

Eye Channels

Audio Channels

Video Channels

Text Channels

Purpose
-------

✓ Understand generated data

✓ Open directly in Excel

✓ Thesis inspection

✓ Dataset sanity checking

✓ Create figures later
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

# ==========================================================
# Project Root
# ==========================================================

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

# ==========================================================
# Dataset Import
# ==========================================================

from datasets.ulm_tsst_synthetic import (
    SyntheticULMTSSTDataset,
)

# ==========================================================
# Configuration
# ==========================================================

OUTPUT_DIR = Path(
    "results/dataset_exports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LABEL_FS = 25

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

# ==========================================================
# Resampling Utility
# ==========================================================

def resample_tensor(
    x: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """
    Resample any signal to target length.
    """

    x = x.float()

    # ---------------------------------
    # 1D signal
    # ---------------------------------

    if x.ndim == 1:

        return (
            F.interpolate(
                x.view(1, 1, -1),
                size=target_length,
                mode="linear",
                align_corners=True,
            )
            .view(-1)
        )

    # ---------------------------------
    # Multi-channel signal
    # ---------------------------------

    if x.ndim == 2:

        channels = []

        for c in range(x.shape[1]):

            channel = (
                F.interpolate(
                    x[:, c]
                    .view(1, 1, -1),
                    size=target_length,
                    mode="linear",
                    align_corners=True,
                )
                .view(-1)
            )

            channels.append(
                channel
            )

        return torch.stack(
            channels,
            dim=-1,
        )

    raise ValueError(
        f"Unsupported tensor shape: {x.shape}"
    )

# ==========================================================
# Dataset Creation
# ==========================================================

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

# ==========================================================
# Reference Timeline
# ==========================================================

reference_signal = next(
    iter(
        sample[
            "continuous_labels"
        ].values()
    )
)

T = len(reference_signal)

time_seconds = (
    torch.arange(T)
    / LABEL_FS
)

# ==========================================================
# Build Table
# ==========================================================

table = {

    "time_index":
        range(T),

    "time_seconds":
        time_seconds.numpy(),
}

# ==========================================================
# Continuous Labels
# ==========================================================

for (
    name,
    signal,
) in sample[
    "continuous_labels"
].items():

    table[name] = (
        signal.cpu()
              .numpy()
    )

# ==========================================================
# Discrete Labels
# ==========================================================

for (
    name,
    signal,
) in sample[
    "discrete_labels"
].items():

    table[name] = (
        signal.cpu()
              .numpy()
    )

# ==========================================================
# Modalities (Aligned)
# ==========================================================

for (
    modality_name,
    modality_data,
) in sample[
    "modalities"
].items():

    aligned = resample_tensor(
        modality_data,
        T,
    )

    aligned = aligned.cpu()

    # ---------------------------------
    # Single channel
    # ---------------------------------

    if aligned.ndim == 1:

        table[
            modality_name
        ] = (
            aligned.numpy()
        )

    # ---------------------------------
    # Multi-channel
    # ---------------------------------

    elif aligned.ndim == 2:

        for c in range(
            aligned.shape[1]
        ):

            table[
                f"{modality_name}_ch{c}"
            ] = (
                aligned[:, c]
                .numpy()
            )

# ==========================================================
# DataFrame
# ==========================================================

df = pd.DataFrame(
    table
)

# ==========================================================
# Export CSV
# ==========================================================

csv_path = (
    OUTPUT_DIR
    / "sample_000_aligned.csv"
)

df.to_csv(
    csv_path,
    index=False,
)

# ==========================================================
# Console Summary
# ==========================================================

print("\n" + "=" * 80)
print("CSV EXPORT COMPLETE")
print("=" * 80)

print(
    f"\nSaved:\n{csv_path}"
)

print(
    f"\nRows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)

print("\nColumn Names:\n")

for column in df.columns:

    print(
        f"  {column}"
    )

print(
    "\nPreview:\n"
)

with pd.option_context(
    "display.max_columns",
    None,
):
    print(
        df.head()
    )

print(
    "\n"
    + "=" * 80
)