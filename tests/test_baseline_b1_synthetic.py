import torch
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)


from models.wrappers.baseline_b1 import BaselineB1


def test_baseline_b1_synthetic():
    """
    Synthetic forward-pass validation for Baseline B1.

    This test verifies:
        - All encoders load correctly
        - Multimodal extractor returns (B, T, 128)
        - Tokenizer returns (B, T, M, D)
        - Fusion returns (B, T, 256)
        - TransformerEncoderDecoder returns:
              enc_out: (B, T, 256)
              dec_out: (B, S, 256)
        - Recognition head returns (B, T, 2)
        - Forecasting head returns (B, S, 2)
        - No shape mismatches or missing keys
    """

    B = 2
    T = 50
    S = 10

    # ---------------------------------------------------------
    # Instantiate Baseline B1 in synthetic mode
    # ---------------------------------------------------------
    model = BaselineB1(
        emb_dim=128,
        fusion_dim=64,
        transformer_dim=256,
        num_heads=2,
        num_layers=2,
        target_seq_len=T,
        forecast_horizon=S,
        synthetic=True,       # IMPORTANT
        include_text=True,
    )

    # ---------------------------------------------------------
    # Create a synthetic batch
    # ---------------------------------------------------------
    batch = {
        "audio": torch.randn(B, 16000),          # ignored in synthetic mode
        "video": torch.randn(B, T, 3, 112, 112), # ignored in synthetic mode
        "ecg":   torch.randn(B, 5000),           # ignored in synthetic mode
        "eda":   torch.randn(B, 5000),           # ignored in synthetic mode
        "text":  ["hello world", "synthetic"],   # ignored in synthetic mode

        # VA history is REQUIRED by TransformerEncoderDecoder
        "va_hist": torch.randn(B, T, 2),
    }

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    outputs = model(batch)

    va_pred = outputs["va_pred"]
    va_future = outputs["va_future"]

    # ---------------------------------------------------------
    # Shape assertions
    # ---------------------------------------------------------
    assert va_pred.shape == (B, T, 2), f"va_pred shape mismatch: {va_pred.shape}"
    assert va_future.shape == (B, S, 2), f"va_future shape mismatch: {va_future.shape}"

    print("✓ Baseline B1 synthetic forward-pass successful!")
    print("  va_pred shape:   ", va_pred.shape)
    print("  va_future shape: ", va_future.shape)


if __name__ == "__main__":
    test_baseline_b1_synthetic()
