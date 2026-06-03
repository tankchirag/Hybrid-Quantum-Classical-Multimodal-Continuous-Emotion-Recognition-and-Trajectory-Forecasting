"""
Substep 5.4 — Full Baseline B1 REAL Forward Pass

Purpose:
    Validate that the COMPLETE Baseline B1 pipeline works in REAL mode:

        raw signals
        → real feature extractors (wav2vec2, ResNet18, ECG/EDA CNNs, TextEncoder)
        → modality tokenizer
        → classical fusion
        → Transformer encoder–decoder
        → recognition head (VA over history)
        → forecasting head (VA over future)

Checks:
    - Model builds with synthetic=False
    - Forward pass runs on CPU/GPU without error
    - Output shapes:
        va_pred   : (B, T, 2)
        va_future : (B, S, 2)
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

from sadbox.create_dummy_raw_batch import create_dummy_raw_batch
from models.wrappers.baseline_b1 import BaselineB1


def test_baseline_b1_real_forward():
    """
    Full REAL-MODE forward-pass test for Baseline B1.
    """

    # ---------------------------------------------------------
    # Hyperparameters for this test
    # ---------------------------------------------------------
    B = 2      # batch size
    T = 50     # history length
    S = 10     # forecast horizon

    emb_dim = 128
    fusion_dim = 128
    transformer_dim = 512
    num_heads = 8
    num_layers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")


    # ---------------------------------------------------------
    # Create dummy RAW multimodal batch (Substep 5.2)
    # ---------------------------------------------------------
    batch = create_dummy_raw_batch(
        batch_size=B,
        T=T,
        audio_len=64000,
        ecg_len=5000,
        eda_len=5000,
    )

    # NOTE: create_dummy_raw_batch already moves tensors to device.
    # We only ensure consistency here:
    batch["audio"] = batch["audio"].to(device)
    batch["video"] = batch["video"].to(device)
    batch["ecg"] = batch["ecg"].to(device)
    batch["eda"] = batch["eda"].to(device)
    batch["va_hist"] = batch["va_hist"].to(device)
    # "text" stays as Python list of strings

    # ---------------------------------------------------------
    # Instantiate Baseline B1 in REAL mode (synthetic=False)
    # ---------------------------------------------------------
    model = BaselineB1(
        emb_dim=emb_dim,
        fusion_dim=fusion_dim,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        target_seq_len=T,
        forecast_horizon=S,
        synthetic=False,        # IMPORTANT: REAL MODE
        include_text=True,
    ).to(device)

    model.eval()

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    with torch.no_grad():
        outputs = model(batch)

    va_pred = outputs["va_pred"]
    va_future = outputs["va_future"]

    # ---------------------------------------------------------
    # Shape checks
    # ---------------------------------------------------------
    assert va_pred.shape == (B, T, 2), f"va_pred shape mismatch: {va_pred.shape}"
    assert va_future.shape == (B, S, 2), f"va_future shape mismatch: {va_future.shape}"

    print("✓ Baseline B1 REAL forward-pass successful!")
    print("  va_pred shape:   ", va_pred.shape)
    print("  va_future shape: ", va_future.shape)
    print("  device:          ", device)


if __name__ == "__main__":
    test_baseline_b1_real_forward()
