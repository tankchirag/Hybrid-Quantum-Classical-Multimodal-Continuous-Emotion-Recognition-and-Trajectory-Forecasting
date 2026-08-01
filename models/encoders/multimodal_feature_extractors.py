# models/encoders/multimodal_feature_extractors.py
"""
MultimodalFeatureExtractors
---------------------------

Unified feature extractor module supporting BOTH:

    • REAL mode (RECOLA / SEWA)
        - Receives raw signals (audio/video/ecg/eda)
        - Receives raw text strings (SEWA)
        - Uses modality-specific encoders:
              AudioEncoder, VideoEncoder, ECGEncoder, EDAEncoder, TextEncoder
        - Produces (B, T, 128) embeddings for ALL modalities

    • SYNTHETIC mode (QSAL / baselines debugging)
        - Receives numeric tensors for all modalities
        - Encoders operate in synthetic mode
        - Produces (B, T, 128) embeddings

Design goals:
    - Dynamic, modality-flexible: audio/video/ecg/eda always supported,
      text optional via include_text flag.
    - Safe handling of REAL vs SYNTHETIC text:
        * RECOLA: text=None → zeros
        * SEWA: list of strings → TextEncoder
        * Synthetic: tensor or list → TextEncoder, then stacked to tensor
    - Guarantee that all outputs are tensors of shape (B, T, emb_dim),
      never Python lists.

Fully aligned with:
    - UnifiedMultimodalDataset
    - multimodal_collate_fn
    - TextEncoder (real + synthetic)
    - ModalityTokenizer / QSALFusion / Baseline B1/B2/B3
"""

import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder
from .video_encoder import VideoEncoder
from .ecg_encoder import ECGEncoder
from .eda_encoder import EDAEncoder
from .text_encoder import TextEncoder


class MultimodalFeatureExtractors(nn.Module):
    """
    Unified multimodal feature extractor.

    Output (REAL + SYNTHETIC):
        {
            "audio": (B, T, 128),
            "video": (B, T, 128),
            "ecg":   (B, T, 128),
            "eda":   (B, T, 128),
            "text":  (B, T, 128) or None (if include_text=False and caller skips it)
        }
    """

    def __init__(
        self,
        emb_dim: int = 128,
        synthetic: bool = True,
        target_seq_len: int = 50,
        include_text: bool = True,
    ):
        super().__init__()

        self.synthetic = synthetic
        self.include_text = include_text
        self.target_seq_len = target_seq_len
        self.emb_dim = emb_dim

        # ---------------------------------------------------------
        # Instantiate encoders (REAL or SYNTHETIC mode)
        # ---------------------------------------------------------
        self.audio_encoder = AudioEncoder(
            emb_dim=emb_dim,
            synthetic=synthetic,
            target_seq_len=target_seq_len,
        )

        self.video_encoder = VideoEncoder(
            emb_dim=emb_dim,
            synthetic=synthetic,
            target_seq_len=target_seq_len,
        )

        self.ecg_encoder = ECGEncoder(
            emb_dim=emb_dim,
            synthetic=synthetic,
            target_seq_len=target_seq_len,
        )

        self.eda_encoder = EDAEncoder(
            emb_dim=emb_dim,
            synthetic=synthetic,
            target_seq_len=target_seq_len,
        )

        if include_text:
            self.text_encoder = TextEncoder(
                emb_dim=emb_dim,
                synthetic=synthetic,
                target_seq_len=target_seq_len,
            )
        else:
            self.text_encoder = None

    def _ensure_tensor(self, x, device):
        """
        Utility: convert TextEncoder outputs (which may be lists)
        into a proper tensor of shape (B, T, emb_dim).

        Supported cases:
            - torch.Tensor → returned as-is
            - list of torch.Tensor → stacked along batch dim
            - list of list/ndarray → converted to tensor and stacked
        """
        if isinstance(x, torch.Tensor):
            return x.to(device)

        if isinstance(x, list):
            # If elements are tensors, stack them
            if isinstance(x[0], torch.Tensor):
                return torch.stack(x, dim=0).to(device)
            # If elements are lists/arrays, convert each to tensor then stack
            else:
                tensors = [torch.as_tensor(e, device=device) for e in x]
                return torch.stack(tensors, dim=0)

        raise ValueError(f"TextEncoder output has unsupported type: {type(x)}")

    # -------------------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------------------
    def forward(self, batch):
        """
        Parameters
        ----------
        batch : dict
            Contains raw signals (REAL mode) or numeric tensors (SYNTHETIC).

        Returns
        -------
        dict
            Modality embeddings (B, T, 128)
        """

        outputs = {}

        # ---------------------------------------------------------
        # AUDIO
        # ---------------------------------------------------------
        outputs["audio"] = self.audio_encoder(batch["audio"])

        # ---------------------------------------------------------
        # VIDEO
        # ---------------------------------------------------------
        outputs["video"] = self.video_encoder(batch["video"])

        # ---------------------------------------------------------
        # ECG
        # ---------------------------------------------------------
        outputs["ecg"] = self.ecg_encoder(batch["ecg"])

        # ---------------------------------------------------------
        # EDA
        # ---------------------------------------------------------
        outputs["eda"] = self.eda_encoder(batch["eda"])

        # ---------------------------------------------------------
        # TEXT (REAL vs SYNTHETIC vs RECOLA)
        # ---------------------------------------------------------
        if self.include_text and self.text_encoder is not None:

            text_input = batch.get("text", None)
            device = outputs["audio"].device

            # Case 1: RECOLA (no text) → text_input is None
            if text_input is None:
                B = outputs["audio"].size(0)
                outputs["text"] = torch.zeros(
                    B, self.target_seq_len, self.emb_dim,
                    device=device
                )

            # Case 2: REAL SEWA — list of raw strings
            elif isinstance(text_input, list) and len(text_input) > 0 and isinstance(text_input[0], str):
                # TextEncoder handles list[str]; may return tensor or list
                text_emb = self.text_encoder(text_input)
                outputs["text"] = self._ensure_tensor(text_emb, device)

            # Case 3: SYNTHETIC — numeric tensor (B, T, D) or similar
            elif isinstance(text_input, torch.Tensor):
                text_emb = self.text_encoder(text_input)
                outputs["text"] = self._ensure_tensor(text_emb, device)

            # Case 4: SYNTHETIC — list of tensors / numeric sequences
            elif isinstance(text_input, list):
                # Pass through TextEncoder; then normalize to tensor
                text_emb = self.text_encoder(text_input)
                outputs["text"] = self._ensure_tensor(text_emb, device)

            # Unexpected type
            else:
                raise ValueError(
                    f"Unsupported text input type in MultimodalFeatureExtractors: {type(text_input)}"
                )

        else:
            # Text disabled → either omit or provide zeros for downstream safety
            B = outputs["audio"].size(0)
            outputs["text"] = torch.zeros(
                B, self.target_seq_len, self.emb_dim,
                device=outputs["audio"].device
            )

        return outputs
