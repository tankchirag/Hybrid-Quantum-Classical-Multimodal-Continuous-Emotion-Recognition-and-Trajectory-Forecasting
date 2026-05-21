"""
Multimodal QEmotion Model Wrapper

This wrapper connects:
- Modality tokenizer
- Fusion module (classical or QSAL)
- Transformer encoder–decoder
- Recognition and forecasting heads

It reads configuration from YAML and builds the correct architecture.
"""

import torch
import torch.nn as nn

from models.tokenizers.modality_tokenizer import ModalityTokenizer
from models.fusion.classical_fusion import ClassicalFusion
# QSAL fusion will be added later:
# from models.fusion.qsal_fusion import QSALFusion

from models.temporal.transformer_ed import TransformerEncoderDecoder
from models.heads.recognition_head import RecognitionHead
from models.heads.forecasting_head import ForecastingHead


class MultimodalQEmotionModel(nn.Module):
    """
    Full multimodal model:
        Tokenizer → Fusion → Transformer → Heads
    """

    def __init__(self, cfg):
        super().__init__()

        model_cfg = cfg["model"]
        fusion_cfg = model_cfg["fusion"]
        transformer_cfg = model_cfg["transformer"]

        self.modality_dim = model_cfg["modality_dim"]
        self.num_modalities = model_cfg["num_modalities"]

        # -----------------------------
        # 1. Modality Tokenizer
        # -----------------------------
        self.tokenizer = ModalityTokenizer(
            modality_dim=self.modality_dim,
            num_modalities=self.num_modalities,
            dropout=0.1,
        )

        # -----------------------------
        # 2. Fusion Module
        # -----------------------------
        if fusion_cfg["type"] == "classical":
            self.fusion = ClassicalFusion(
                modality_dim=self.modality_dim,
                fusion_dim=fusion_cfg["fusion_dim"],
                num_modalities=self.num_modalities,
                num_heads=fusion_cfg["num_heads"],
                transformer_dim=fusion_cfg["transformer_dim"],
            )
        elif fusion_cfg["type"] == "qsal":
            raise NotImplementedError("QSAL fusion will be added next.")
        else:
            raise ValueError(f"Unknown fusion type: {fusion_cfg['type']}")

        # -----------------------------
        # 3. Transformer Encoder–Decoder
        # -----------------------------
        self.temporal = TransformerEncoderDecoder(
            d_model=transformer_cfg["d_model"],
            nhead=transformer_cfg["nhead"],
            num_encoder_layers=transformer_cfg["num_encoder_layers"],
            num_decoder_layers=transformer_cfg["num_decoder_layers"],
            dim_feedforward=transformer_cfg["dim_feedforward"],
            dropout=transformer_cfg["dropout"],
        )

        # -----------------------------
        # 4. Prediction Heads
        # -----------------------------
        self.recognition_head = RecognitionHead(
            input_dim=transformer_cfg["d_model"],
            output_dim=2,
        )

        self.forecasting_head = ForecastingHead(
            input_dim=transformer_cfg["d_model"],
            output_dim=2,
        )

    # ---------------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------------
    def forward(self, batch):
        """
        batch contains:
            audio: (B, T, 128)
            video: (B, T, 128)
            ecg:   (B, T, 128)
            eda:   (B, T, 128)
            va_hist: (B, T, 2)
            va_now:  (B, 2)
            va_fut:  (B, S, 2)
        """

        audio = batch["audio"]
        video = batch["video"]
        ecg = batch["ecg"]
        eda = batch["eda"]

        # 1. Tokenize modalities
        tokens = self.tokenizer(audio, video, ecg, eda)
        # tokens: (B, T, 4, 128)

        # 2. Fuse modalities
        fused = self.fusion(audio, video, ecg, eda)
        # fused: (B, T, 256)

        # 3. Temporal modeling
        enc_out, dec_out = self.temporal(fused)
        # enc_out: (B, T, 256)
        # dec_out: (B, S, 256)

        # 4. Predictions
        rec_pred = self.recognition_head(enc_out)   # (B, T, 2)
        for_pred = self.forecasting_head(dec_out)   # (B, S, 2)

        return rec_pred, for_pred
