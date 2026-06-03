"""
Substep 5.2 — Create Dummy Raw Batch (REAL MODE)

Purpose:
    Generate a batch of RAW multimodal signals that mimic SEWA/RECOLA inputs.
    These raw signals will be passed through REAL feature extractors in Substep 5.3.

Why this file exists:
    - This is NOT a unit test.
    - This is NOT synthetic embedding-level data.
    - This is a sandbox utility to simulate real raw signals
      (audio, video, ECG, EDA, text) before real datasets arrive.

Output:
    A Python dictionary containing raw signals on the correct device.
"""

import torch


def create_dummy_raw_batch(
    batch_size: int = 2,
    T: int = 50,
    audio_len: int = 64000,   # 4 seconds @ 16 kHz
    ecg_len: int = 5000,
    eda_len: int = 5000,
):
    """
    Create a dummy batch of RAW multimodal signals.

    This batch simulates:
        - raw audio waveform
        - raw video frames
        - raw ECG signal
        - raw EDA signal
        - raw text
        - VA history (required by Transformer decoder)

    Returns:
        dict: multimodal raw batch
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = {
        # Raw audio waveform (B, audio_len)
        "audio": torch.randn(batch_size, audio_len).to(device),

        # Raw video frames (B, T, 3, 112, 112)
        "video": torch.randn(batch_size, T, 3, 112, 112).to(device),

        # Raw ECG signal (B, ecg_len)
        "ecg": torch.randn(batch_size, ecg_len).to(device),

        # Raw EDA signal (B, eda_len)
        "eda": torch.randn(batch_size, eda_len).to(device),

        # Raw text (list of strings)
        "text": ["hello world", "emotion recognition"],

        # VA history (B, T, 2)
        "va_hist": torch.randn(batch_size, T, 2).to(device),
    }

    print("✓ Dummy REAL-MODE raw batch created")
    print("  audio:", batch["audio"].shape)
    print("  video:", batch["video"].shape)
    print("  ecg:  ", batch["ecg"].shape)
    print("  eda:  ", batch["eda"].shape)
    print("  va_hist:", batch["va_hist"].shape)
    print("  text:", batch["text"])

    return batch


if __name__ == "__main__":
    create_dummy_raw_batch()
