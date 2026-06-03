import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGEncoder(nn.Module):
    """
    ECGEncoder
    ----------
    A 1D‑CNN‑based feature extractor for physiological ECG signals, producing
    fixed‑length 128‑dimensional embeddings aligned to the multimodal sequence
    length T.

    This module supports two modes:

    synthetic = True:
        - Used during synthetic dataset experiments.
        - Ignores the input ECG waveform and returns random embeddings of shape (B, T, 128).
        - Ensures the architecture remains identical to the real pipeline.

    synthetic = False:
        - Used for real datasets (e.g., RECOLA).
        - Expects raw ECG waveforms of shape (B, L).
        - Applies a lightweight 1D CNN stack:
              Conv1d(1→32) → ReLU →
              Conv1d(32→64) → ReLU →
              Conv1d(64→128) → ReLU
        - Produces frame‑level features (B, 128, T_raw).
        - Interpolates temporal dimension to fixed sequence length T.
        - Returns (B, T, 128).

    Parameters
    ----------
    emb_dim : int
        Output embedding dimension (fixed at 128 for your thesis).
    synthetic : bool
        Whether to operate in synthetic mode.
    target_seq_len : int
        Number of time steps T expected by the fusion/transformer pipeline.

    Input (real mode)
    -----------------
    ecg : Tensor
        Raw ECG waveform of shape (B, L).

    Output
    ------
    ecg_emb : Tensor
        Frame‑level ECG embeddings of shape (B, T, 128).
    """

    def __init__(
        self,
        emb_dim: int = 128,
        synthetic: bool = True,
        target_seq_len: int = 50,
    ):
        super().__init__()
        self.synthetic = synthetic
        self.emb_dim = emb_dim
        self.target_seq_len = target_seq_len

        # Synthetic mode: no backbone needed
        if synthetic:
            self.cnn = None
            return

        # Real mode: 1D CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, ecg):
        """
        Forward pass for ECG encoding.

        Parameters
        ----------
        ecg : Tensor
            - Synthetic mode: ignored.
            - Real mode: raw ECG waveform (B, L).

        Returns
        -------
        Tensor
            ECG embeddings of shape (B, T, 128).
        """

        # ---------------------------------------------------------
        # SYNTHETIC MODE: return random embeddings (B, T, 128)
        # ---------------------------------------------------------
        if self.synthetic:
            if isinstance(ecg, torch.Tensor):
                B = ecg.size(0)
                device = ecg.device
            else:
                B = 1
                device = torch.device("cpu")

            return torch.randn(B, self.target_seq_len, self.emb_dim, device=device)

        # ---------------------------------------------------------
        # REAL MODE: 1D CNN feature extraction
        # ---------------------------------------------------------
        # ecg: (B, L) → reshape to (B, 1, L)
        ecg = ecg.unsqueeze(1)

        feats = self.cnn(ecg)  # (B, 128, T_raw)
        B, C, T_raw = feats.shape

        # transpose to (B, T_raw, 128)
        emb = feats.permute(0, 2, 1)

        # ---------------------------------------------------------
        # TEMPORAL ALIGNMENT: interpolate to fixed T
        # ---------------------------------------------------------
        if T_raw == self.target_seq_len:
            return emb

        emb = emb.permute(0, 2, 1)  # (B, 128, T_raw)
        emb = F.interpolate(
            emb,
            size=self.target_seq_len,
            mode="linear",
            align_corners=False,
        )
        emb = emb.permute(0, 2, 1)  # (B, T, 128)

        return emb
