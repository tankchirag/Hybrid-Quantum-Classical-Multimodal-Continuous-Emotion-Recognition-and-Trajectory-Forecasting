"""
Synchronizer
------------

A modular, dataset-agnostic alignment module that converts raw
multimodal signals with different sampling rates into a unified
time grid of length T.

This module is responsible for:

    • Resampling high-frequency biosignals (1000 Hz → T)
    • Downsampling audio (16 kHz → T)
    • Upsampling video (4–30 Hz → T)
    • Aligning eye-tracking (60–120 Hz → T)
    • Aligning text tokens (word-level → T)
    • Handling missing frames
    • Producing modality-specific raw tensors of shape (T, raw_dim)

Encoders then convert these aligned raw tensors into embeddings:

    raw_aligned → encoder → (B, T, 128)

This separation ensures:
    • Encoders remain clean and modality-specific
    • Synchronization is consistent across all datasets
    • Architecture becomes fully modality-agnostic
"""

from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F


class Synchronizer:
    """
    Synchronizer

    Parameters
    ----------
    target_seq_len : int
        Desired temporal length T for all modalities.

    device : str
        "cpu" or "cuda".

    Notes
    -----
    - Uses linear interpolation for continuous signals.
    - Uses nearest-neighbor for discrete signals (e.g., video frames).
    - Text is handled separately (token-level alignment).
    """

    def __init__(self, target_seq_len: int, device: str = "cpu") -> None:
        self.T = target_seq_len
        self.device = device

    # ----------------------------------------------------------------------
    # Generic resampling utilities
    # ----------------------------------------------------------------------

    def _resample_1d(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Resample a 1D signal to length T using linear interpolation.

        Parameters
        ----------
        signal : Tensor (L,)

        Returns
        -------
        Tensor (T,)
        """
        L = signal.shape[0]
        signal = signal.unsqueeze(0).unsqueeze(0)  # (1,1,L)
        resampled = F.interpolate(signal, size=self.T, mode="linear", align_corners=True)
        return resampled.squeeze(0).squeeze(0)

    def _resample_2d(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Resample a 2D signal (L, D) to (T, D).

        Parameters
        ----------
        signal : Tensor (L, D)

        Returns
        -------
        Tensor (T, D)
        """
        L, D = signal.shape
        signal = signal.unsqueeze(0).transpose(1, 2)  # (1, D, L)
        resampled = F.interpolate(signal, size=self.T, mode="linear", align_corners=True)
        return resampled.squeeze(0).transpose(0, 1)  # (T, D)

    def _resample_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Resample video frames (F, C, H, W) to (T, C, H, W) using nearest-neighbor.

        Parameters
        ----------
        video : Tensor (F, C, H, W)

        Returns
        -------
        Tensor (T, C, H, W)
        """
        F, C, H, W = video.shape
        video = video.unsqueeze(0)  # (1, F, C, H, W)
        resampled = F.interpolate(video, size=self.T, mode="nearest")
        return resampled.squeeze(0)

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def synchronize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronize all modalities in a raw sample.

        Parameters
        ----------
        raw : dict
            {
                "eeg":  (L_eeg,),
                "ecg":  (L_ecg,),
                "eda":  (L_eda,),
                "emg":  (L_emg,),
                "resp": (L_resp,),
                "bpm":  (L_bpm,),
                "eye":  (F_eye, 2),
                "audio": (L_audio,),
                "video": (F_video, C, H, W),
                "text":  list[str] or None,
                "labels": (T, D)  # already aligned
            }

        Returns
        -------
        dict
            {
                "eeg":   (T, 1),
                "ecg":   (T, 1),
                "eda":   (T, 1),
                "emg":   (T, 1),
                "resp":  (T, 1),
                "bpm":   (T, 1),
                "eye":   (T, 2),
                "audio": (T, 1),
                "video": (T, C, H, W),
                "text":  list[str] or None,
                "labels": (T, D)
            }
        """
        out = {}

        # High-frequency biosignals
        for key in ["eeg", "ecg", "eda", "emg", "resp", "bpm"]:
            if raw.get(key) is not None:
                out[key] = self._resample_1d(raw[key]).unsqueeze(-1)  # (T,1)
            else:
                out[key] = None

        # Eye-tracking (2D)
        if raw.get("eye") is not None:
            out["eye"] = self._resample_2d(raw["eye"])  # (T,2)
        else:
            out["eye"] = None

        # Audio waveform
        if raw.get("audio") is not None:
            out["audio"] = self._resample_1d(raw["audio"]).unsqueeze(-1)  # (T,1)
        else:
            out["audio"] = None

        # Video frames
        if raw.get("video") is not None:
            out["video"] = self._resample_video(raw["video"])  # (T,C,H,W)
        else:
            out["video"] = None

        # Text (raw strings)
        out["text"] = raw.get("text", None)

        # Labels (already aligned)
        out["labels"] = raw["labels"]

        return out
