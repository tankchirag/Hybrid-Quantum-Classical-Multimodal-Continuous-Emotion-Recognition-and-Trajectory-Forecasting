# ============================================================
# ADVANCED VISUALIZATION FOR ULM-TSST SYNTHETIC DATASET
# ============================================================

import sys, os
import numpy as np
import torch
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Fix import path so "datasets" module is visible
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from datasets.ulm_tsst_synthetic import SyntheticULMTSSTDataset

# ------------------------------------------------------------
# 1. Create dataset instance
# ------------------------------------------------------------
modalities = {
    "ecg": True,
    "eda": True,
    "resp": True,
    "bpm": True,
    "eeg": True,
    "emg": True,
    "eye": False,
    "audio": True,
    "video": False,  # Set to True if you want to visualize video features
    "text": False,  # Set to True if you want to visualize text features
}

labels = {
    "valence": True,
    "arousal": True,
    "dominance": False,
    "stress": False,
    "workload": False,
}

dataset = SyntheticULMTSSTDataset(
    num_samples=5,
    target_seq_len=600,   # 5 minutes @ 2 Hz
    modalities=modalities,
    labels=labels,
    device="cpu",
    save_dir=None,
)

# ------------------------------------------------------------
# 2. Fetch one sample
# ------------------------------------------------------------
idx = 0
sample = dataset[idx]

modal_data = {k: v for k, v in sample.items() if k not in ["labels", "meta", "masks"]}
label_data = sample["labels"]
meta = sample["meta"]
masks = sample["masks"]

print("=== METADATA ===")
print(meta)
print("\n=== MODALITIES ===")
print(list(modal_data.keys()))
print("\n=== LABELS ===")
print(list(label_data.keys()))

# ------------------------------------------------------------
# 3. Plot labels (VA + stress + dominance + workload)
# ------------------------------------------------------------
plt.figure(figsize=(14, 6))
for lbl_name, lbl_ts in label_data.items():
    plt.plot(lbl_ts.squeeze().numpy(), label=lbl_name)

plt.title("Label Trajectories (Valence, Arousal, Stress, etc.)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 4. Plot TSST phases overlay
# ------------------------------------------------------------
stress = label_data["stress"].squeeze().numpy()
t = np.linspace(0, 1, len(stress))

plt.figure(figsize=(14, 4))
plt.plot(t, stress, label="Stress", color="red")

plt.axvspan(0.0, 0.2, alpha=0.2, color="green", label="Baseline")
plt.axvspan(0.2, 0.4, alpha=0.2, color="yellow", label="Anticipation")
plt.axvspan(0.4, 0.6, alpha=0.2, color="orange", label="Speech")
plt.axvspan(0.6, 0.8, alpha=0.2, color="red", label="Arithmetic")
plt.axvspan(0.8, 1.0, alpha=0.2, color="blue", label="Recovery")

plt.title("TSST Stress Curve with Phases")
plt.xlabel("Normalized Time")
plt.ylabel("Stress")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 5. Plot physiological signals
# ------------------------------------------------------------
physio_mods = ["ecg", "eda", "resp", "bpm"]

plt.figure(figsize=(14, 8))
for i, m in enumerate(physio_mods):
    if m in modal_data:
        plt.subplot(2, 2, i+1)
        plt.plot(modal_data[m].numpy())
        plt.title(m.upper())
        plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Plot EEG, EMG, Eye
# ------------------------------------------------------------
neuro_mods = ["eeg", "emg", "eye"]

plt.figure(figsize=(14, 8))
for i, m in enumerate(neuro_mods):
    if m in modal_data:
        plt.subplot(2, 2, i+1)
        plt.plot(modal_data[m].numpy())
        plt.title(m.upper())
        plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Plot audio/video/text raw signals
# ------------------------------------------------------------
behav_mods = ["audio", "video", "text"]

plt.figure(figsize=(14, 8))
for i, m in enumerate(behav_mods):
    if m in modal_data:
        plt.subplot(2, 2, i+1)
        plt.plot(modal_data[m].numpy())
        plt.title(m.upper())
        plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. Plot modality masks
# ------------------------------------------------------------
plt.figure(figsize=(12, 4))
for m, mask in masks.items():
    plt.plot(mask.numpy(), label=m)

plt.title("Modality Masks")
plt.xlabel("Time")
plt.ylabel("Mask (1=present, 0=missing)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 9. Correlation between VA and Stress
# ------------------------------------------------------------
v = label_data["valence"].squeeze().numpy()
a = label_data["arousal"].squeeze().numpy()
s = label_data["stress"].squeeze().numpy()

plt.figure(figsize=(14, 5))
plt.scatter(a, s, alpha=0.5, label="Arousal vs Stress")
plt.scatter(v, s, alpha=0.5, label="Valence vs Stress")
plt.xlabel("VA Value")
plt.ylabel("Stress")
plt.title("VA–Stress Correlation")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 10. Subject variability visualization
# ------------------------------------------------------------
print("\nSubject baseline valence:", dataset.subject_baseline_valence)
print("Subject stress gain:", dataset.subject_stress_gain)
print("Subject noise scale:", dataset.subject_noise_scale)
