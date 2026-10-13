"""
Multimodal QEmotion Model Wrapper

This wrapper connects:
- Modality tokenizer
- Fusion module (classical or QSAL)
- Transformer encoder–decoder
- Recognition and forecasting heads

Pipeline:
    Tokenizer → Fusion (classical/QSAL) → Transformer ED (with VA history)
    → Recognition Head → Forecasting Head

It reads configuration from YAML and builds the correct architecture.
"""



import torch
import torch.nn as nn

from models.tokenizers.modality_tokenizer import ModalityTokenizer
from models.fusion.classical_fusion import ClassicalFusion
from models.fusion.qsal_fusion import QSALFusion
from models.temporal.transformer_ed import TransformerEncoderDecoder
from models.heads.recognition_head import RecognitionHead
from models.heads.forecasting_head import ForecastingHead


class MultimodalQEmotionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        model_cfg = cfg["model"]
        fusion_cfg = model_cfg["fusion"]
        transformer_cfg = model_cfg["transformer"]

        self.modality_dim = model_cfg["modality_dim"]
        self.num_modalities = model_cfg["num_modalities"]

        # -----------------------------
        # 1. Tokenizer
        # -----------------------------
        self.tokenizer = ModalityTokenizer(
            modality_dim=self.modality_dim,
            num_modalities=self.num_modalities,
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
            self.fusion = QSALFusion(
                modality_dim=self.modality_dim,
                fusion_dim=fusion_cfg["fusion_dim"],
                num_modalities=self.num_modalities,
                num_heads=fusion_cfg["num_heads"],
                num_qubits=fusion_cfg["num_qubits"],
                circuit_depth=fusion_cfg["circuit_depth"],
                transformer_dim=fusion_cfg["transformer_dim"],
            )
        else:
            raise ValueError("Unknown fusion type")

        # -----------------------------
        # 3. Transformer ED
        # -----------------------------
        self.temporal = TransformerEncoderDecoder(
            d_model=transformer_cfg["d_model"],
            nhead=transformer_cfg["nhead"],
            num_encoder_layers=transformer_cfg["num_encoder_layers"],
            num_decoder_layers=transformer_cfg["num_decoder_layers"],
            dim_feedforward=transformer_cfg["dim_feedforward"],
            dropout=transformer_cfg["dropout"],
            forecast_horizon=cfg["dataset"]["S"],
        )

        # -----------------------------
        # 4. Heads
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
    # Forward
    # ---------------------------------------------------------
    def forward(self, batch):
        audio = batch["audio"]
        video = batch["video"]
        ecg = batch["ecg"]
        eda = batch["eda"]
        va_hist = batch["va_hist"]

        # 1. Tokenize modalities
        tokens = self.tokenizer(audio, video, ecg, eda)

        # 2. Fuse modalities
        fused = self.fusion(audio, video, ecg, eda)

        # 3. Temporal modeling (with VA history)
        enc_out, dec_out = self.temporal(fused, va_hist)

        # 4. Predictions
        rec_pred = self.recognition_head(enc_out)
        for_pred = self.forecasting_head(dec_out)

        return rec_pred, for_pred
