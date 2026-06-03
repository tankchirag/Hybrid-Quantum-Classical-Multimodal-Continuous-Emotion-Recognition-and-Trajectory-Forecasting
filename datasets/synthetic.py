"""
Synthetic multimodal dataset for validating the full thesis pipeline.

This dataset simulates:
    - 4 modality embeddings (audio, video, ECG, EDA), each emb_dim-dim
    - A latent emotional trajectory (valence, arousal)
    - Recognition target: VA(t)
    - Forecasting target: VA(t+1 ... t+S)

It is used to validate:
    - Modality tokenizer
    - Classical / QSAL / Hybrid fusion
    - Transformer encoder-decoder
    - Training loop
    - End-to-end pipeline

NOTE:
    This is a LIGHTWEIGHT, EMBEDDING-LEVEL dataset.
    No raw video frames, no heavy feature extractors.
"""

import torch
import numpy as np
from .base_dataset import BaseMultimodalDataset


class SyntheticMultimodalDataset(BaseMultimodalDataset):
    """Synthetic dataset that mimics RECOLA/SEWA structure at embedding level."""

    def __init__(self, cfg, split: str = "train", seed: int = 42):
        """
        Parameters
        ----------
        cfg : dict
            Global config (expects dataset.T, dataset.S, dataset.length).
        split : str
            "train" or "val".
        seed : int
            Random seed for reproducibility.
        """
        T = cfg["dataset"]["T"]
        S = cfg["dataset"]["S"]
        length = cfg["dataset"]["length"]

        super().__init__(T=T, S=S)

        self.T = T
        self.S = S
        self.split = split

        # Embedding dimension (must match model.emb_dim)
        self.emb_dim = cfg["model"].get("emb_dim", 128)

        # 80/20 split
        if split == "train":
            self.size = int(length * 0.8)
            self.offset = 0
        elif split == "val":
            self.size = length - int(length * 0.8)
            self.offset = int(length * 0.8)
        else:
            raise ValueError("split must be 'train' or 'val'")

        rng = np.random.default_rng(seed)

        # Latent emotional trajectory (VA) over time
        # length + T + S to allow sliding windows
        self.latent = rng.normal(
            0, 1, size=(length + T + S, 2)
        ).astype(np.float32)

        # Modality-specific linear projections (simulate encoders)
        self.W_audio = rng.normal(
            0, 0.1, size=(2, self.emb_dim)
        ).astype(np.float32)
        self.W_video = rng.normal(
            0, 0.1, size=(2, self.emb_dim)
        ).astype(np.float32)
        self.W_ecg = rng.normal(
            0, 0.1, size=(2, self.emb_dim)
        ).astype(np.float32)
        self.W_eda = rng.normal(
            0, 0.1, size=(2, self.emb_dim)
        ).astype(np.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        """
        Generate one synthetic multimodal sample.

        Returns
        -------
        dict
            {
                "audio":   (T, emb_dim),
                "video":   (T, emb_dim),
                "ecg":     (T, emb_dim),
                "eda":     (T, emb_dim),
                "va_hist": (T, 2),
                "va_rec":  (2,),
                "va_future": (S, 2),
            }
        """
        idx = idx + self.offset

        T = self.T
        S = self.S

        # VA history, recognition target, forecasting target
        va_hist = self.latent[idx : idx + T]                      # (T,2)
        va_rec = self.latent[idx + T]                             # (2,)
        va_future = self.latent[idx + T + 1 : idx + T + 1 + S]    # (S,2)

        # Modality embeddings (simple linear projections of VA history)
        audio = va_hist @ self.W_audio                            # (T, emb_dim)
        video = va_hist @ self.W_video                            # (T, emb_dim)
        ecg = va_hist @ self.W_ecg                                # (T, emb_dim)
        eda = va_hist @ self.W_eda                                # (T, emb_dim)

        return {
            "audio": torch.tensor(audio, dtype=torch.float32),
            "video": torch.tensor(video, dtype=torch.float32),
            "ecg":   torch.tensor(ecg,   dtype=torch.float32),
            "eda":   torch.tensor(eda,   dtype=torch.float32),

            "va_hist":   torch.tensor(va_hist,   dtype=torch.float32),
            "va_rec":    torch.tensor(va_rec,    dtype=torch.float32),
            "va_future": torch.tensor(va_future, dtype=torch.float32),
        }
