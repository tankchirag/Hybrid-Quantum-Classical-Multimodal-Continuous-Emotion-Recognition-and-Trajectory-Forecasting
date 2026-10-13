import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGEncoder(nn.Module):
    """
    EEGEncoder
    ----------
    A 1D‑CNN‑based feature extractor for EEG signals, producing
    fixed‑length 128‑dimensional embeddings aligned to the multimodal
    sequence length T.

    Modes
    -----
    synthetic = True:
        - Returns random embeddings (B, T, 128).

    synthetic = False:
        - Expects raw EEG waveform (B, L).
        - Applies lightweight 1D CNN:
              Conv1d(1→32) → ReLU →
              Conv1d(32→64) → ReLU →
              Conv1d(64→128) → ReLU
        - Interpolates temporal dimension to T.
        - Returns (B, T, 128).
    """

    def __init__(self, emb_dim: int = 128, synthetic: bool = True, target_seq_len: int = 50):
        super().__init__()
        self.synthetic = synthetic
        self.emb_dim = emb_dim
        self.target_seq_len = target_seq_len

        if synthetic:
            self.cnn = None
            return

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, eeg):
        if self.synthetic:
            B = eeg.size(0) if isinstance(eeg, torch.Tensor) else 1
            device = eeg.device if isinstance(eeg, torch.Tensor) else torch.device("cpu")
            return torch.randn(B, self.target_seq_len, self.emb_dim, device=device)

        eeg = eeg.unsqueeze(1)
        feats = self.cnn(eeg)
        B, C, T_raw = feats.shape

        emb = feats.permute(0, 2, 1)

        if T_raw == self.target_seq_len:
            return emb

        emb = emb.permute(0, 2, 1)
        emb = F.interpolate(emb, size=self.target_seq_len, mode="linear", align_corners=False)
        emb = emb.permute(0, 2, 1)

        return emb
