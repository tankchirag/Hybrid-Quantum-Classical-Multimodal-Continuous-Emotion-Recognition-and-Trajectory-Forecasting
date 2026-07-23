import torch
import torch.nn as nn

from .audio_encoder import AudioEncoder
from .video_encoder import VideoEncoder
from .ecg_encoder import ECGEncoder
from .eda_encoder import EDAEncoder
from .text_encoder import TextEncoder


class MultimodalFeatureExtractors(nn.Module):
    """
    MultimodalFeatureExtractors
    ---------------------------
    A unified wrapper that bundles all modality-specific feature extractors
    (audio, video, ECG, EDA, text) into a single module. This ensures a clean,
    consistent interface for the downstream tokenizer, fusion module, and
    Transformer.

    This wrapper supports both synthetic and real modes:

    synthetic = True:
        - All encoders operate in synthetic mode.
        - Each encoder returns random embeddings of shape (B, T, 128).
        - Used for Baseline B1 development and debugging without real datasets.

    synthetic = False:
        - Each encoder processes its respective raw modality input.
        - Audio: raw waveform (B, L_audio)
        - Video: raw frames (B, T_video, 3, H, W)
        - ECG: raw waveform (B, L_ecg)
        - EDA: raw waveform (B, L_eda)
        - Text: list[str] or tokenized inputs
        - Each encoder outputs (B, T, 128), aligned to the same temporal length.

    Output
    ------
    A dictionary mapping modality names to embeddings:

        {
            "audio": (B, T, 128),
            "video": (B, T, 128),
            "ecg":   (B, T, 128),
            "eda":   (B, T, 128),
            "text":  (B, T, 128)   # only if text is provided
        }

    This dictionary is passed directly to the ModalityTokenizer.
    """

    def __init__(
        self,
        emb_dim: int = 128,
        synthetic: bool = True,
        target_seq_len: int = 50,
        include_text: bool = True,   # SEWA support
    ):
        super().__init__()

        self.synthetic = synthetic
        self.include_text = include_text

        # Instantiate all encoders
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

    def forward(self, batch):
        """
        Forward pass for multimodal feature extraction.

        Parameters
        ----------
        batch : dict
            A dictionary containing raw modality inputs.
            Expected keys (depending on dataset):
                - "audio"
                - "video"
                - "ecg"
                - "eda"
                - "text" (optional)

        Returns
        -------
        dict
            {
                "audio": (B, T, 128),
                "video": (B, T, 128),
                "ecg":   (B, T, 128),
                "eda":   (B, T, 128),
                "text":  (B, T, 128)   # only if include_text=True
            }
        """

        outputs = {}

        # Audio
        if "audio" in batch:
            outputs["audio"] = self.audio_encoder(batch["audio"])
        else:
            outputs["audio"] = None

        # Video
        if "video" in batch:
            outputs["video"] = self.video_encoder(batch["video"])
        else:
            outputs["video"] = None

        # ECG
        if "ecg" in batch:
            outputs["ecg"] = self.ecg_encoder(batch["ecg"])
        else:
            outputs["ecg"] = None

        # EDA
        if "eda" in batch:
            outputs["eda"] = self.eda_encoder(batch["eda"])
        else:
            outputs["eda"] = None

        # Text (optional)
        if batch["text"] is not None:
            outputs["text"] = self.text_encoder(batch["text"])
        else:
            outputs["text"] = None

        return outputs
