# models/encoders/multimodal_feature_extractors.py
"""
MultimodalFeatureExtractors
---------------------------

Unified feature extractor module supporting BOTH:

    • REAL mode (RECOLA / SEWA / future ULM‑TSST)
        - Receives raw signals (audio/video/ecg/eda/eeg/emg/resp/bpm/eye)
        - Receives raw text strings (SEWA)
        - Uses modality-specific encoders:
              AudioEncoder, VideoEncoder, TextEncoder,
              ECGEncoder, EDAEncoder,
              EEGEncoder, EMGEncoder, RESPEncoder, BPMEncoder, EyeTrackingEncoder
        - Produces (B, T, 128) embeddings for ALL active modalities

    • SYNTHETIC mode (QSAL / baselines debugging / synthetic ULM‑TSST)
        - Receives numeric tensors for all modalities
        - Encoders operate in synthetic mode
        - Produces (B, T, 128) embeddings

Design goals:
    - Fully modality-agnostic: any subset of modalities can be enabled.
    - Dataset-agnostic: RECOLA, SEWA, ULM‑TSST, custom datasets.
    - Label-agnostic: VA, VAD, stress, workload, etc.
    - Safe handling of REAL vs SYNTHETIC text.
    - Guarantee that all outputs are tensors of shape (B, T, emb_dim).

Aligned with:
    - SyntheticULMTSSTDataset
    - Synchronizer
    - multimodal_collate_fn
    - ModalityTokenizer / QSALFusion / Baseline B1/B2/B3
"""

import torch
import torch.nn as nn

# Existing encoders
from .audio_encoder import AudioEncoder
from .video_encoder import VideoEncoder
from .ecg_encoder import ECGEncoder
from .eda_encoder import EDAEncoder
from .text_encoder import TextEncoder

# NEW encoders for ULM‑TSST
from .eeg_encoder import EEGEncoder
from .emg_encoder import EMGEncoder
from .resp_encoder import RESPEncoder
from .bpm_encoder import BPMEncoder
from .eye_encoder import EyeTrackingEncoder


class MultimodalFeatureExtractors(nn.Module):
    """
    Unified multimodal feature extractor.

    Output (REAL + SYNTHETIC):
        {
            "audio": (B, T, 128),
            "video": (B, T, 128),
            "ecg":   (B, T, 128),
            "eda":   (B, T, 128),
            "eeg":   (B, T, 128),
            "emg":   (B, T, 128),
            "resp":  (B, T, 128),
            "bpm":   (B, T, 128),
            "eye":   (B, T, 128),
            "text":  (B, T, 128)
        }
    """

    def __init__(
        self,
        emb_dim: int = 128,
        synthetic: bool = True,
        target_seq_len: int = 50,
        modalities: dict = None,
        include_text: bool = True,
    ):
        """
        Parameters
        ----------
        emb_dim : int
            Embedding dimension for all encoders (fixed at 128).

        synthetic : bool
            Whether encoders operate in synthetic mode.

        target_seq_len : int
            Temporal sequence length T.

        modalities : dict
            Dictionary of modality flags, e.g.:
                {
                    "audio": True,
                    "video": True,
                    "ecg": True,
                    "eda": True,
                    "eeg": True,
                    "emg": True,
                    "resp": True,
                    "bpm": True,
                    "eye": True,
                    "text": True
                }

        include_text : bool
            Whether to instantiate the text encoder.
        """
        super().__init__()

        self.synthetic = synthetic
        self.target_seq_len = target_seq_len
        self.emb_dim = emb_dim
        self.modalities = modalities or {}
        self.include_text = include_text

        # ---------------------------------------------------------
        # Instantiate encoders dynamically based on modality flags
        # ---------------------------------------------------------
        self.encoders = nn.ModuleDict()

        if self.modalities.get("audio", False):
            self.encoders["audio"] = AudioEncoder(emb_dim, synthetic, target_seq_len)

        if self.modalities.get("video", False):
            self.encoders["video"] = VideoEncoder(emb_dim, synthetic, target_seq_len)

        if self.modalities.get("ecg", False):
            self.encoders["ecg"] = ECGEncoder(emb_dim, synthetic, target_seq_len)

        if self.modalities.get("eda", False):
            self.encoders["eda"] = EDAEncoder(emb_dim, synthetic, target_seq_len)

        if self.modalities.get("eeg", False):
            self.encoders["eeg"] = EEGEncoder(emb_dim, synthetic, target_seq_len)

        if self.modalities.get("emg", False):
            self.encoders["emg"] = EMGEncoder(emb_dim, synthetic, target_seq_len)

        if self.modalities.get("resp", False):
            self.encoders["resp"] = RESPEncoder(emb_dim, synthetic, target_seq_len)

        if self.modalities.get("bpm", False):
            self.encoders["bpm"] = BPMEncoder(emb_dim, synthetic, target_seq_len)

        if self.modalities.get("eye", False):
            self.encoders["eye"] = EyeTrackingEncoder(emb_dim, synthetic, target_seq_len)

        if include_text and self.modalities.get("text", False):
            self.encoders["text"] = TextEncoder(emb_dim, synthetic, target_seq_len)

    # ---------------------------------------------------------
    # Utility: ensure TextEncoder outputs are tensors
    # ---------------------------------------------------------
    def _ensure_tensor(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)

        if isinstance(x, list):
            if isinstance(x[0], torch.Tensor):
                return torch.stack(x, dim=0).to(device)
            else:
                tensors = [torch.as_tensor(e, device=device) for e in x]
                return torch.stack(tensors, dim=0)

        raise ValueError(f"Unsupported text encoder output type: {type(x)}")

    # ---------------------------------------------------------
    # FORWARD PASS
    # ---------------------------------------------------------
    def forward(self, batch):
        """
        Parameters
        ----------
        batch : dict
            Raw or synthetic signals for all modalities.

        Returns
        -------
        dict
            {
                modality_name: (B, T, 128)
            }
        """
        outputs = {}
        device = None

        # Determine device from first available modality
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                device = batch[key].device
                break

        # ---------------------------------------------------------
        # Process each modality dynamically
        # ---------------------------------------------------------
        for modality_name, encoder in self.encoders.items():

            raw_input = batch.get(modality_name, None)

            # TEXT special handling
            if modality_name == "text":
                if raw_input is None:
                    B = next(iter(outputs.values())).size(0)
                    outputs["text"] = torch.zeros(
                        B, self.target_seq_len, self.emb_dim, device=device
                    )
                elif isinstance(raw_input, list) and isinstance(raw_input[0], str):
                    text_emb = encoder(raw_input)
                    outputs["text"] = self._ensure_tensor(text_emb, device)
                else:
                    text_emb = encoder(raw_input)
                    outputs["text"] = self._ensure_tensor(text_emb, device)
                continue

            # Non-text modalities
            if raw_input is None:
                B = next(iter(outputs.values())).size(0)
                outputs[modality_name] = torch.zeros(
                    B, self.target_seq_len, self.emb_dim, device=device
                )
            else:
                outputs[modality_name] = encoder(raw_input)

        return outputs
