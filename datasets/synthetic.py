"""
Synthetic multimodal dataset for testing the full thesis pipeline.

This dataset simulates:
    - 4 modality embeddings (audio, video, ECG, EDA), each 128-dim
    - A latent emotional trajectory (valence, arousal)
    - Recognition target: VA(t)
    - Forecasting target: VA(t+1 ... t+S)

It is used to validate:
    - Fusion module
    - QSAL block
    - Transformer encoder-decoder
    - Training loop
    - End-to-end pipeline
"""

import torch
import numpy as np
from .base_dataset import BaseMultimodalDataset


class SyntheticMultimodalDataset(BaseMultimodalDataset):
    """Synthetic dataset that mimics RECOLA/SEWA structure."""

    def __init__(self, cfg, split="train", seed=42):
        """
        Parameters
        ----------
        cfg : dict
            Global config.
        split : str
            "train" or "val".
        seed : int
            Random seed for reproducibility.
        """
        T = cfg["dataset"]["T"]
        S = cfg["dataset"]["S"]
        length = cfg["dataset"]["length"]

        super().__init__(T=T, S=S)

        # 80/20 split
        if split == "train":
            self.size = int(length * 0.8)
            self.offset = 0
        elif split == "val":
            self.size = length - int(length * 0.8)
            self.offset = int(length * 0.8)
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.split = split
        self.T = T
        self.S = S

        rng = np.random.default_rng(seed)

        # Latent emotional trajectory (VA)
        self.latent = rng.normal(0, 1, size=(length + T + S, 2)).astype(np.float32)

        # Modality-specific linear projections (simulate encoders)
        self.W_audio = rng.normal(0, 0.1, size=(2, 128)).astype(np.float32)
        self.W_video = rng.normal(0, 0.1, size=(2, 128)).astype(np.float32)
        self.W_ecg   = rng.normal(0, 0.1, size=(2, 128)).astype(np.float32)
        self.W_eda   = rng.normal(0, 0.1, size=(2, 128)).astype(np.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Generate one synthetic multimodal sample.

        Returns
        -------
        dict
            Dictionary containing modality embeddings and VA labels.
        """
        idx = idx + self.offset  # shift for val split

        T = self.T
        S = self.S

        # Extract VA history, recognition target, and forecasting target
        va_hist = self.latent[idx : idx + T]                      # (T,2)
        va_rec  = self.latent[idx + T]                            # (2,)
        va_fut  = self.latent[idx + T + 1 : idx + T + 1 + S]      # (S,2)

        # Generate modality embeddings
        audio = va_hist @ self.W_audio
        video = va_hist @ self.W_video
        ecg   = va_hist @ self.W_ecg
        eda   = va_hist @ self.W_eda

        return {
            "audio":   torch.tensor(audio, dtype=torch.float32),      # (T,128)
            "video":   torch.tensor(video, dtype=torch.float32),      # (T,128)
            "ecg":     torch.tensor(ecg, dtype=torch.float32),        # (T,128)
            "eda":     torch.tensor(eda, dtype=torch.float32),        # (T,128)

            "va_hist": torch.tensor(va_hist, dtype=torch.float32),    # (T,2)
            "va_rec":  torch.tensor(va_rec, dtype=torch.float32),     # (2,)
            "va_fut":  torch.tensor(va_fut, dtype=torch.float32),     # (S,2)
        }
