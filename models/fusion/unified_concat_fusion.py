import torch
import torch.nn as nn
from typing import Dict, Optional


class UnifiedConcatFusion(nn.Module):
    """
    UnifiedConcatFusion (Baseline B2 fusion module).

    This module implements the fusion strategy used in Baseline B2:
    - Each modality-specific encoder produces 128-dimensional frame-level embeddings.
    - Each modality embedding is projected to `proj_dim` via a dedicated linear layer.
    - All projected embeddings are concatenated along the feature dimension to form
      a single multimodal frame token per time step.
    - No modality tokens or explicit cross-modal attention are used here.

    The output is a sequence of frame tokens of shape (B, T, out_dim),
    where out_dim = (#active_modalities) × proj_dim.
    """

    def __init__(
        self,
        use_audio: bool = True,
        use_video: bool = True,
        use_ecg: bool = True,
        use_eda: bool = True,
        use_text: bool = False,
        in_dim: int = 128,
        proj_dim: int = 64,
        expected_out_dim: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        use_audio : bool
            Whether audio embeddings are present and should be fused.
        use_video : bool
            Whether video embeddings are present and should be fused.
        use_ecg : bool
            Whether ECG embeddings are present and should be fused.
        use_eda : bool
            Whether EDA embeddings are present and should be fused.
        use_text : bool
            Whether text embeddings are present and should be fused.
        in_dim : int
            Dimensionality of each modality encoder output (default: 128).
        proj_dim : int
            Dimensionality after per-modality projection (default: 64).
        expected_out_dim : int or None
            Optional consistency check for the concatenated dimensionality.
        """
        super().__init__()

        self.use_audio = use_audio
        self.use_video = use_video
        self.use_ecg = use_ecg
        self.use_eda = use_eda
        self.use_text = use_text
        self.proj_dim = proj_dim

        # Per-modality projections
        self.proj_audio = nn.Linear(in_dim, proj_dim) if use_audio else None
        self.proj_video = nn.Linear(in_dim, proj_dim) if use_video else None
        self.proj_ecg = nn.Linear(in_dim, proj_dim) if use_ecg else None
        self.proj_eda = nn.Linear(in_dim, proj_dim) if use_eda else None
        self.proj_text = nn.Linear(in_dim, proj_dim) if use_text else None

        # Number of *configured* modalities
        num_modalities = int(use_audio) + int(use_video) + int(use_ecg) + int(use_eda) + int(use_text)
        actual_out_dim = num_modalities * proj_dim

        # Store for downstream modules (e.g. BaselineB2)
        self.out_dim = actual_out_dim

        # Optional sanity check if user passes expected_out_dim
        if expected_out_dim is not None and expected_out_dim != actual_out_dim:
            raise ValueError(
                f"UnifiedConcatFusion: expected_out_dim={expected_out_dim}, "
                f"but num_modalities * proj_dim = {actual_out_dim}. "
                f"Check modality flags and projection dimension."
            )

    @staticmethod
    def _project(
        proj_layer: Optional[nn.Linear],
        emb: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Apply a per-modality projection layer to a sequence of embeddings.

        Parameters
        ----------
        proj_layer : nn.Linear or None
            Linear layer mapping (in_dim) → (proj_dim). If None, returns None.
        emb : torch.Tensor or None
            Input embeddings of shape (B, T, in_dim). If None, returns None.

        Returns
        -------
        torch.Tensor or None
            Projected embeddings of shape (B, T, proj_dim), or None if input is None.
        """
        if proj_layer is None or emb is None:
            return None

        B, T, D = emb.shape
        emb_flat = emb.reshape(B * T, D)      # (B*T, in_dim)
        proj_flat = proj_layer(emb_flat)      # (B*T, proj_dim)
        proj = proj_flat.reshape(B, T, -1)    # (B, T, proj_dim)
        return proj

    def forward(self, embeddings: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Fuse modality embeddings by per-modality projection and concatenation.

        Parameters
        ----------
        embeddings : dict
            Dictionary of modality embeddings with keys such as:
            - "audio": (B, T, in_dim) or None
            - "video": (B, T, in_dim) or None
            - "ecg":   (B, T, in_dim) or None
            - "eda":   (B, T, in_dim) or None
            - "text":  (B, T, in_dim) or None

        Returns
        -------
        torch.Tensor
            Fused multimodal frame tokens of shape (B, T, out_dim),
            where out_dim = (#configured_modalities) × proj_dim.
        """
        proj_list = []

        if self.use_audio:
            audio_emb = embeddings.get("audio", None)
            proj_audio = self._project(self.proj_audio, audio_emb)
            if proj_audio is not None:
                proj_list.append(proj_audio)

        if self.use_video:
            video_emb = embeddings.get("video", None)
            proj_video = self._project(self.proj_video, video_emb)
            if proj_video is not None:
                proj_list.append(proj_video)

        if self.use_ecg:
            ecg_emb = embeddings.get("ecg", None)
            proj_ecg = self._project(self.proj_ecg, ecg_emb)
            if proj_ecg is not None:
                proj_list.append(proj_ecg)

        if self.use_eda:
            eda_emb = embeddings.get("eda", None)
            proj_eda = self._project(self.proj_eda, eda_emb)
            if proj_eda is not None:
                proj_list.append(proj_eda)

        if self.use_text:
            text_emb = embeddings.get("text", None)
            proj_text = self._project(self.proj_text, text_emb)
            if proj_text is not None:
                proj_list.append(proj_text)

        if len(proj_list) == 0:
            raise RuntimeError("UnifiedConcatFusion: no modality embeddings provided for fusion.")

        fused = torch.cat(proj_list, dim=-1)

        # Optional safety check
        if fused.shape[-1] != self.out_dim:
            raise RuntimeError(
                f"UnifiedConcatFusion: fused feature dimension {fused.shape[-1]} "
                f"does not match configured out_dim={self.out_dim}. "
                f"Check which modalities are actually present in the batch."
            )

        return fused
