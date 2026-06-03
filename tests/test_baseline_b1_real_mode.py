import os
import sys
import torch

# ---------------------------------------------------------
# Add project root
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.wrappers.baseline_b1 import BaselineB1


def test_baseline_b1_real_mode_shapes():
    """
    REAL-MODE forward-pass smoke test for Baseline B1.

    Goal:
        - Instantiate BaselineB1 with synthetic=False
        - (In next substeps) plug in real feature extractors
        - For now: just ensure model builds and can be moved to device
    """

    B = 2
    T = 50
    S = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # Instantiate Baseline B1 in REAL mode
    # ---------------------------------------------------------
    model = BaselineB1(
        emb_dim=256,
        fusion_dim=128,
        transformer_dim=512,
        num_heads=8,
        num_layers=4,
        target_seq_len=T,
        forecast_horizon=S,
        synthetic=False,      # IMPORTANT: REAL MODE
        include_text=True,
    ).to(device)

    # For Substep 5.1 we do NOT run a full forward yet.
    # We just confirm that the model can be constructed and moved to device.
    print("✓ Baseline B1 (REAL mode) instantiated successfully on", device)


if __name__ == "__main__":
    test_baseline_b1_real_mode_shapes()
