# datasets/ulm_tsst_synthetic.py
"""
Synthetic ULM‑TSST Dataset
--------------------------

This dataset generates synthetic multimodal samples that mimic the structure
of the ULM‑TSST dataset used in MuSe‑Stress / MuSe‑Physio.

It is fully modality‑agnostic, dataset‑agnostic, and label‑agnostic.

It inherits from BaseMultimodalDataset, which requires:

    T = history window length
    S = forecasting horizon

Each sample returns:

    {
        "audio":   (T,128),
        "video":   (T,128),
        "ecg":     (T,128),
        "eda":     (T,128),
        "eeg":     (T,128),
        "emg":     (T,128),
        "resp":    (T,128),
        "bpm":     (T,128),
        "eye":     (T,128),
        "text":    (T,128),

        "va_hist": (T,label_dim),
        "va_now":  (label_dim,),
        "va_fut":  (S,label_dim),

        # Backward‑compatible key for existing pipeline:
        "va":      (T,label_dim)
    }

All modalities are optional and controlled by cfg["model"]["modalities"].
"""

import torch
import numpy as np

from .base_dataset import BaseMultimodalDataset


class SyntheticULMTSSTDataset(BaseMultimodalDataset):
    """
    Synthetic multimodal dataset for ULM‑TSST‑style experiments.

    Parameters
    ----------
    num_samples : int
        Number of synthetic samples to generate.

    target_seq_len : int
        History window length T.

    label_dim : int
        Dimensionality of labels (2=VA, 3=VAD, etc.)

    modalities : dict
        Dict of modality flags:
            {"audio": True, "video": True, "ecg": True, ...}

    device : str
        "cpu" or "cuda"

    forecast_horizon : int, optional
        Number of future steps S. If None, defaults to T.
    """

    def __init__(
        self,
        num_samples: int,
        target_seq_len: int,
        label_dim: int,
        modalities: dict,
        device: str = "cpu",
        forecast_horizon: int = None,
    ):
        # ---------------------------------------------------------
        # Determine forecasting horizon S
        # ---------------------------------------------------------
        if forecast_horizon is None:
            # Default: same as T (safe fallback)
            forecast_horizon = target_seq_len

        # ---------------------------------------------------------
        # Initialize BaseMultimodalDataset(T, S)
        # ---------------------------------------------------------
        super().__init__(T=target_seq_len, S=forecast_horizon)

        # ---------------------------------------------------------
        # Store parameters
        # ---------------------------------------------------------
        self.num_samples = num_samples
        self.T = target_seq_len
        self.S = forecast_horizon
        self.label_dim = label_dim
        self.modalities = modalities
        self.device = torch.device(device)

        # ---------------------------------------------------------
        # Synthetic random generator
        # ---------------------------------------------------------
        self.rng = np.random.default_rng(seed=42)

    # -------------------------------------------------------------
    # LENGTH
    # -------------------------------------------------------------
    def __len__(self) -> int:
        return self.num_samples

    # -------------------------------------------------------------
    # GENERATE RANDOM MODALITY EMBEDDING
    # -------------------------------------------------------------
    def _rand_modality(self) -> torch.Tensor:
        """
        Returns a random modality embedding of shape (T,128).
        """
        return torch.randn(self.T, 128, device=self.device)

    # -------------------------------------------------------------
    # GENERATE RANDOM LABELS
    # -------------------------------------------------------------
    def _rand_labels(self):
        """
        Returns:
            va_hist : (T,label_dim)
            va_now  : (label_dim,)
            va_fut  : (S,label_dim)
        """
        va_hist = torch.randn(self.T, self.label_dim, device=self.device)
        va_now = torch.randn(self.label_dim, device=self.device)
        va_fut = torch.randn(self.S, self.label_dim, device=self.device)
        return va_hist, va_now, va_fut

    # -------------------------------------------------------------
    # GET ITEM
    # -------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        """
        Returns a synthetic multimodal sample.

        All modalities are optional and controlled by self.modalities.
        Missing modalities are filled with zeros of shape (T,128).

        The label interface is backward‑compatible with RECOLA/SEWA:
            • "va"      : (T,label_dim)  → used by existing collate_fn
            • "va_hist" : (T,label_dim)  → explicit history
            • "va_now"  : (label_dim,)   → current state
            • "va_fut"  : (S,label_dim)  → future trajectory
        """

        sample = {}

        # ---------------------------------------------------------
        # Modalities
        # ---------------------------------------------------------
        for modality_name, enabled in self.modalities.items():
            if enabled:
                sample[modality_name] = self._rand_modality()
            else:
                sample[modality_name] = torch.zeros(
                    self.T, 128, device=self.device
                )

        # ---------------------------------------------------------
        # Labels
        # ---------------------------------------------------------
        va_hist, va_now, va_fut = self._rand_labels()

        # Explicit forecasting labels
        sample["va_hist"] = va_hist
        sample["va_now"] = va_now
        sample["va_fut"] = va_fut

        # Backward‑compatible key for existing pipeline
        # RECOLA/SEWA used "va" as history labels → we mirror that.
        sample["va"] = va_hist

        return sample
