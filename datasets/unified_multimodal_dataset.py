"""
UnifiedMultimodalDataset
------------------------

A unified dummy dataset wrapper for SEWA and RECOLA.

This version is fully compatible with:
    - Baseline B1 / B2 / B3
    - QSAL (Quantum Self‑Attention Layer)
    - PyTorch DataLoader (no None values)
    - MultimodalFeatureExtractors (always receives tensors)

Key design rules:
    1) Every modality ALWAYS returns a float32 tensor.
       Disabled modalities return zero‑tensors of correct shape.
       This avoids:
           - DataLoader collate errors
           - QSALFusion dtype mismatch (Double vs Float)
           - NoneType crashes in feature extractors

    2) Output format is stable across all baselines:
        {
            "audio": Tensor(L_audio, float32),
            "video": Tensor(T, 3, H, W, float32),
            "ecg":   Tensor(L_ecg, float32),
            "eda":   Tensor(L_eda, float32),
            "text":  Tensor(T, D_text, float32),
            "va":    Tensor(T, 2, float32)
        }

    3) Dummy tensors allow architecture validation without real data.
       Real data can be plugged in later without changing the model.
"""

import torch
from torch.utils.data import Dataset


class UnifiedMultimodalDataset(Dataset):
    """
    Unified multimodal dataset for RECOLA and SEWA.

    Parameters
    ----------
    dataset_name : str
        "recola" or "sewa".
    root_dir : str
        Path to dataset (unused in dummy mode).
    target_seq_len : int
        Number of frames/time steps (T).
    forecast_horizon : int
        Number of future steps (S).
    """

    def __init__(
        self,
        dataset_name: str,
        root_dir: str = "",
        target_seq_len: int = 50,
        forecast_horizon: int = 10,
    ):
        super().__init__()

        dataset_name = dataset_name.lower()
        assert dataset_name in ["recola", "sewa"], \
            "dataset_name must be 'recola' or 'sewa'"

        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.target_seq_len = target_seq_len
        self.forecast_horizon = forecast_horizon

        # ---------------------------------------------------------
        # Automatic modality flags (truth table)
        # ---------------------------------------------------------
        if dataset_name == "recola":
            self.modalities = {
                "audio": True,
                "video": True,
                "ecg":   True,
                "eda":   True,
                "text":  False,   # RECOLA has no text
            }

        elif dataset_name == "sewa":
            self.modalities = {
                "audio": True,
                "video": True,
                "ecg":   False,
                "eda":   False,
                "text":  True,    # SEWA has text
            }

        # Dummy dataset size
        self.num_samples = 100

    # -------------------------------------------------------------
    # Required Dataset Methods
    # -------------------------------------------------------------
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a dictionary of raw signals for one sample.

        All tensors are float32 to avoid dtype mismatch in QSALFusion.
        No modality ever returns None.
        """

        sample = {}

        # ---------------------------------------------------------
        # AUDIO (dummy waveform)
        # ---------------------------------------------------------
        L_audio = 16000 * 4  # 4 seconds @ 16 kHz
        if self.modalities["audio"]:
            sample["audio"] = torch.randn(L_audio, dtype=torch.float32)
        else:
            sample["audio"] = torch.zeros(L_audio, dtype=torch.float32)

        # ---------------------------------------------------------
        # VIDEO (dummy frames)
        # ---------------------------------------------------------
        T = self.target_seq_len
        if self.modalities["video"]:
            sample["video"] = torch.randn(T, 3, 112, 112, dtype=torch.float32)
        else:
            sample["video"] = torch.zeros(T, 3, 112, 112, dtype=torch.float32)

        # ---------------------------------------------------------
        # ECG (dummy)
        # ---------------------------------------------------------
        if self.modalities["ecg"]:
            sample["ecg"] = torch.randn(5000, dtype=torch.float32)
        else:
            sample["ecg"] = torch.zeros(5000, dtype=torch.float32)

        # ---------------------------------------------------------
        # EDA (dummy)
        # ---------------------------------------------------------
        if self.modalities["eda"]:
            sample["eda"] = torch.randn(5000, dtype=torch.float32)
        else:
            sample["eda"] = torch.zeros(5000, dtype=torch.float32)

        # ---------------------------------------------------------
        # TEXT (dummy embedding)
        # Always return a tensor (QSALFusion requires it)
        # ---------------------------------------------------------
        D_text = 128
        if self.modalities["text"]:
            sample["text"] = torch.randn(T, D_text, dtype=torch.float32)
        else:
            sample["text"] = torch.zeros(T, D_text, dtype=torch.float32)

        # ---------------------------------------------------------
        # VA LABELS (dummy)
        # ---------------------------------------------------------
        sample["va"] = torch.randn(T, 2, dtype=torch.float32)

        return sample
