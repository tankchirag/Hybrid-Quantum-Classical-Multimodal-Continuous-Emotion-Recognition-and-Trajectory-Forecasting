from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.ulm_tsst_synthetic import (
    SyntheticULMTSSTDataset,
)

# build dataset exactly like your eval script
dataset = SyntheticULMTSSTDataset(
    num_samples=10,
    duration_seconds=600,
    modalities={
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
    },
    continuous_labels={
        "valence": True,
        "arousal": True,
        "stress": True,
        "workload": True,
        "fatigue": True,
        "dominance": True,
        "engagement": True,
    },
    discrete_labels={
        "emotion_class": True,
        "stress_class": True,
        "workload_class": True,
        "alertness_class": True,
        "drowsiness_class": True,
        "kss_score": True,
    },
    sampling_rates={
        "ecg":1000,
        "eda":128,
        "resp":64,
        "bpm":4,
        "eeg":256,
        "emg":1000,
        "eye":120,
        "audio":100,
        "video":25,
        "text":5,
    }
)

sample = dataset[0]

save_dict = {}

for k,v in sample["modalities"].items():
    save_dict[f"mod_{k}"] = v.numpy()

for k,v in sample["continuous_labels"].items():
    save_dict[f"cont_{k}"] = v.numpy()

for k,v in sample["discrete_labels"].items():
    save_dict[f"disc_{k}"] = v.numpy()

np.savez(
    "sample_000.npz",
    **save_dict,
)

print("Saved: sample_000.npz")