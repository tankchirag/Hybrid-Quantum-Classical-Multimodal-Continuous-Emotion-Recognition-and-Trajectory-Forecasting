"""
Substep 5.3 — Run REAL Feature Extractors on Dummy Raw Batch

Purpose:
    - Load real feature extractors (wav2vec2, ResNet18, ECG/EDA CNNs, TextEncoder)
    - Run them on the dummy raw batch created in Substep 5.2
    - Verify output shapes, GPU compatibility, and temporal alignment

This is NOT a unit test.
This is a sandbox validation script to ensure real-mode encoders work correctly.
"""

import os
import sys
import torch

# ---------------------------------------------------------
# Add project root
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from create_dummy_raw_batch import create_dummy_raw_batch
from models.encoders.audio_encoder import AudioEncoder
from models.encoders.video_encoder import VideoEncoder
from models.encoders.ecg_encoder import ECGEncoder
from models.encoders.eda_encoder import EDAEncoder
from models.encoders.text_encoder import TextEncoder


def run_real_feature_extractors():
    """
    Run all REAL feature extractors on dummy raw signals.
    """

    print("\n=== Substep 5.3 — Running REAL Feature Extractors ===\n")

    # ---------------------------------------------------------
    # Load dummy raw batch
    # ---------------------------------------------------------
    batch = create_dummy_raw_batch(batch_size=2, T=50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # Instantiate encoders
    # ---------------------------------------------------------
    audio_enc = AudioEncoder(emb_dim=256).to(device)
    video_enc = VideoEncoder(emb_dim=256).to(device)
    ecg_enc   = ECGEncoder(emb_dim=256).to(device)
    eda_enc   = EDAEncoder(emb_dim=256).to(device)
    text_enc  = TextEncoder(emb_dim=256).to(device)

    print("\n✓ All encoders loaded on", device)

    # ---------------------------------------------------------
    # AUDIO
    # ---------------------------------------------------------
    print("\n--- AUDIO ENCODER (wav2vec2-base) ---")
    audio_emb = audio_enc(batch["audio"])
    print("audio_emb:", audio_emb.shape)   # (B, T, 256)

    # ---------------------------------------------------------
    # VIDEO
    # ---------------------------------------------------------
    print("\n--- VIDEO ENCODER (ResNet18) ---")
    video_emb = video_enc(batch["video"])
    print("video_emb:", video_emb.shape)   # (B, T, 256)

    # ---------------------------------------------------------
    # ECG
    # ---------------------------------------------------------
    print("\n--- ECG ENCODER (1D CNN) ---")
    ecg_emb = ecg_enc(batch["ecg"])
    print("ecg_emb:", ecg_emb.shape)       # (B, T, 256)

    # ---------------------------------------------------------
    # EDA
    # ---------------------------------------------------------
    print("\n--- EDA ENCODER (1D CNN) ---")
    eda_emb = eda_enc(batch["eda"])
    print("eda_emb:", eda_emb.shape)       # (B, T, 256)

    # ---------------------------------------------------------
    # TEXT
    # ---------------------------------------------------------
    print("\n--- TEXT ENCODER (DistilBERT) ---")
    text_emb = text_enc(batch["text"])
    print("text_emb:", text_emb.shape)     # (B, T, 256) or (B, 256)

    print("\n=== Substep 5.3 Completed Successfully ===\n")


if __name__ == "__main__":
    run_real_feature_extractors()
