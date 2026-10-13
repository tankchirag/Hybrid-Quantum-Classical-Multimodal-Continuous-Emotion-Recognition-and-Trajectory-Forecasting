import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
    _TV_AVAILABLE = True
except ImportError:
    _TV_AVAILABLE = False


class VideoEncoder(nn.Module):
    """
    VideoEncoder
    ------------
    Frame-level video feature extractor using ResNet18 → 128-dim projection.

    This module supports two modes:

    synthetic = True:
        - Used during synthetic dataset experiments.
        - Ignores input video frames and returns random embeddings of shape (B, T, 128).
        - Ensures the architecture remains identical to the real pipeline.

    synthetic = False:
        - Used for real datasets (e.g., RECOLA, SEWA).
        - Expects raw video frames of shape (B, T, 3, H, W).
        - Applies ResNet18 (pretrained) to each frame independently.
        - Extracts 512-dim features from the penultimate layer.
        - Projects to 128-dim.
        - Interpolates temporal dimension to fixed sequence length T.

    Parameters
    ----------
    emb_dim : int
        Output embedding dimension (fixed at 128 for your thesis).
    synthetic : bool
        Whether to operate in synthetic mode.
    freeze_backbone : bool
        Whether to freeze ResNet18 parameters during training.
    target_seq_len : int
        Number of time steps T expected by the fusion/transformer pipeline.

    Input (real mode)
    -----------------
    video : Tensor
        Raw video frames of shape (B, T, 3, H, W).

    Output
    ------
    video_emb : Tensor
        Frame-level embeddings of shape (B, T, 128).
    """

    def __init__(
        self,
        emb_dim: int = 128,
        synthetic: bool = True,
        freeze_backbone: bool = True,
        target_seq_len: int = 50,
    ):
        super().__init__()
        self.synthetic = synthetic
        self.emb_dim = emb_dim
        self.target_seq_len = target_seq_len

        # Synthetic mode: no backbone needed
        if synthetic:
            self.backbone = None
            self.proj = None
            return

        # Real mode: load ResNet18
        assert _TV_AVAILABLE, "Install torchvision to use ResNet18."

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  # remove final FC
        self.backbone = nn.Sequential(*modules)  # outputs (B, 512, 1, 1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(512, emb_dim)

    def forward(self, video):
        """
        Forward pass for video encoding.

        Parameters
        ----------
        video : Tensor
            - Synthetic mode: ignored.
            - Real mode: raw video frames (B, T, 3, H, W).

        Returns
        -------
        Tensor
            Video embeddings of shape (B, T, 128).
        """

        # ---------------------------------------------------------
        # SYNTHETIC MODE: return random embeddings (B, T, 128)
        # ---------------------------------------------------------
        if self.synthetic:
            if isinstance(video, torch.Tensor):
                B = video.size(0)
                device = video.device
            else:
                B = 1
                device = torch.device("cpu")

            return torch.randn(B, self.target_seq_len, self.emb_dim, device=device)

        # ---------------------------------------------------------
        # REAL MODE: ResNet18 frame-level feature extraction
        # ---------------------------------------------------------
        B, T, C, H, W = video.shape
        device = video.device

        # reshape to process frames independently
        video = video.reshape(B * T, C, H, W)  # (B*T, 3, H, W)

        with torch.set_grad_enabled(self.backbone.training):
            feats = self.backbone(video)  # (B*T, 512, 1, 1)

        feats = feats.view(B, T, 512)  # (B, T, 512)

        # project to 128
        emb = self.proj(feats)  # (B, T, 128)

        # ---------------------------------------------------------
        # TEMPORAL ALIGNMENT: interpolate to fixed T
        # ---------------------------------------------------------
        if T == self.target_seq_len:
            return emb

        emb = emb.permute(0, 2, 1)  # (B, 128, T)
        emb = F.interpolate(
            emb,
            size=self.target_seq_len,
            mode="linear",
            align_corners=False,
        )
        emb = emb.permute(0, 2, 1)  # (B, T, 128)

        return emb
