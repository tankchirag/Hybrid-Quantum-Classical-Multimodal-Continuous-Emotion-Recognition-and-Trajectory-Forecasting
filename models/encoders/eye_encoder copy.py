import torch
import torch.nn as nn
import torch.nn.functional as F


class EyeTrackingEncoder(nn.Module):
    """
    EyeTrackingEncoder
    ------------------
    Encodes gaze vectors (x,y) into 128-dim embeddings.

    Input
    -----
    eye : Tensor (B, T, 2)

    Output
    ------
    eye_emb : Tensor (B, T, 128)
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
            nn.Conv1d(2, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, eye):
        if self.synthetic:
            B = eye.size(0) if isinstance(eye, torch.Tensor) else 1
            device = eye.device if isinstance(eye, torch.Tensor) else torch.device("cpu")
            return torch.randn(B, self.target_seq_len, self.emb_dim, device=device)

        eye = eye.transpose(1, 2)
        feats = self.cnn(eye)
        B, C, T_raw = feats.shape

        emb = feats.permute(0, 2, 1)

        if T_raw == self.target_seq_len:
            return emb

        emb = emb.permute(0, 2, 1)
        emb = F.interpolate(emb, size=self.target_seq_len, mode="linear", align_corners=False)
        emb = emb.permute(0, 2, 1)

        return emb
