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

    def __init__(self, length=5000, T=50, S=10, seed=42):
        """
        Parameters
        ----------
        length : int
            Number of samples in the dataset.
        T : int
            History window length.
        S : int
            Forecasting horizon.
        seed : int
            Random seed for reproducibility.
        """
        super().__init__(T=T, S=S)

        self.length = length
        rng = np.random.default_rng(seed)

        # Latent emotional trajectory (VA)
        self.latent = rng.normal(0, 1, size=(length + T + S, 2)).astype(np.float32)

        # Modality-specific linear projections (simulate encoders)
        self.W_audio = rng.normal(0, 0.1, size=(2, 128)).astype(np.float32)
        self.W_video = rng.normal(0, 0.1, size=(2, 128)).astype(np.float32)
        self.W_ecg   = rng.normal(0, 0.1, size=(2, 128)).astype(np.float32)
        self.W_eda   = rng.normal(0, 0.1, size=(2, 128)).astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Generate one synthetic multimodal sample.

        Returns
        -------
        dict
            Dictionary containing modality embeddings and VA labels.
        """
        # Extract VA history, current VA, and future VA
        va_hist = self.latent[idx : idx + self.T]                      # (T,2)
        va_now  = self.latent[idx + self.T]                            # (2,)
        va_fut  = self.latent[idx + self.T + 1 : idx + self.T + 1 + self.S]  # (S,2)

        # Generate modality embeddings
        audio = va_hist @ self.W_audio
        video = va_hist @ self.W_video
        ecg   = va_hist @ self.W_ecg
        eda   = va_hist @ self.W_eda

        return {
            "audio":   torch.tensor(audio, dtype=torch.float32),
            "video":   torch.tensor(video, dtype=torch.float32),
            "ecg":     torch.tensor(ecg, dtype=torch.float32),
            "eda":     torch.tensor(eda, dtype=torch.float32),
            "va_hist": torch.tensor(va_hist, dtype=torch.float32),
            "va_now":  torch.tensor(va_now, dtype=torch.float32),
            "va_fut":  torch.tensor(va_fut, dtype=torch.float32),
        }
