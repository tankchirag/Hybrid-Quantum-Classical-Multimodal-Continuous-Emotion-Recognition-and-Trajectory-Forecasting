# models/wrappers/baseline_b1.py

import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors import MultimodalFeatureExtractors
from models.tokenizers.modality_tokenizer import ModalityTokenizer
from models.fusion.classical_fusion import ClassicalFusion
from models.temporal.transformer_ed import TransformerEncoderDecoder
from models.heads.recognition_head import RecognitionHead
from models.heads.forecasting_head import ForecastingHead


class BaselineB1(nn.Module):
    """
    BaselineB1
    ----------
    Classical multimodal baseline using:

        1. Multimodal feature extractors
        2. Modality tokenizer
        3. Classical self-attention fusion
        4. Transformer Encoder–Decoder (conditioned on VA history)
        5. Recognition head (frame-wise VA_t)
        6. Forecasting head (future VA trajectory)

    Input (from UnifiedMultimodalDataset)
    -------------------------------------
    batch must contain:
        - raw modalities (audio, video, ecg, eda, text/None)
        - "va": Tensor of shape (B, T_va, 2)  → full VA sequence

    Internal VA handling
    --------------------
    From batch["va"] we derive:
        - va_hist      : (B, T_align, 2)  → history aligned with fused features
        - va_future_gt : (B, S, 2)        → last S frames, used as forecasting target

    Temporal alignment
    ------------------
    Feature extractors may produce a temporal length T_fused that differs from T_va.
    To guarantee architectural consistency, we align both to:

        T_align = min(T_fused, T_va)

    by cropping the *end* of the longer sequence. This is robust across datasets
    and configuration choices.
    """

    def __init__(
        self,
        emb_dim: int = 128,
        fusion_dim: int = 64,
        transformer_dim: int = 256,
        num_heads: int = 2,
        num_layers: int = 2,
        target_seq_len: int = 50,
        forecast_horizon: int = 10,
        synthetic: bool = True,
        include_text: bool = True,
    ):
        super().__init__()

        self.target_seq_len = target_seq_len
        self.forecast_horizon = forecast_horizon

        # 1. Multimodal Feature Extractors
        self.extractors = MultimodalFeatureExtractors(
            emb_dim=emb_dim,
            synthetic=synthetic,
            target_seq_len=target_seq_len,
            include_text=include_text,
        )

        # 2. Modality Tokenizer
        self.tokenizer = ModalityTokenizer(
            emb_dim=emb_dim,
            num_modalities=5 if include_text else 4,
        )

        # 3. Classical Fusion
        self.fusion = ClassicalFusion(
            emb_dim=emb_dim,
            fusion_dim=fusion_dim,
            transformer_dim=transformer_dim,
            num_heads=num_heads,
        )

        # 4. Transformer Encoder–Decoder
        self.temporal_model = TransformerEncoderDecoder(
            d_model=transformer_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            forecast_horizon=forecast_horizon,
        )

        # 5. Recognition Head
        self.recognition_head = RecognitionHead(
            input_dim=transformer_dim,
            output_dim=2,
        )

        # 6. Forecasting Head
        self.forecasting_head = ForecastingHead(
            input_dim=transformer_dim,
            output_dim=2,
        )

    def forward(self, batch):
        """
        Forward pass for Baseline B1.

        Parameters
        ----------
        batch : dict
            Must contain:
                - raw modality inputs (audio, video, ecg, eda, text/None)
                - "va": Tensor of shape (B, T_va, 2)

        Returns
        -------
        dict
            {
                "va_pred":      (B, T_align, 2),
                "va_future":    (B, S, 2),
                "va_hist":      (B, T_align, 2),
                "va_future_gt": (B, S, 2),
            }
        """

        # ---------------------------------------------------------
        # 0. VA history and future from batch["va"]
        # ---------------------------------------------------------
        va_full = batch["va"]          # (B, T_va, 2)
        B, T_va, _ = va_full.shape
        S = self.forecast_horizon

        # Future ground truth: last S frames (or repeat last if S > T_va)
        if S <= T_va:
            va_future_gt = va_full[:, -S:, :]   # (B, S, 2)
        else:
            va_future_gt = va_full[:, -1:, :].repeat(1, S, 1)

        # ---------------------------------------------------------
        # 1. Extract modality embeddings (per modality)
        #    emb_dict[mod] : (B, T_fused, emb_dim)
        # ---------------------------------------------------------
        emb_dict = self.extractors(batch)

        # 2. Tokenize modalities → (B, T_fused, M, emb_dim)
        tokens = self.tokenizer(emb_dict)

        # 3. Classical fusion → (B, T_fused, transformer_dim)
        fused = self.fusion(tokens)

        # ---------------------------------------------------------
        # 3.5 Temporal alignment: match fused and VA history lengths
        # ---------------------------------------------------------
        T_fused = fused.size(1)
        T_align = min(T_fused, T_va)

        # Align fused and va_hist by cropping the end of the longer one
        fused_aligned = fused[:, -T_align:, :]          # (B, T_align, d)
        va_hist = va_full[:, -T_align:, :]              # (B, T_align, 2)

        # ---------------------------------------------------------
        # 4. Temporal modeling with aligned VA history
        # ---------------------------------------------------------
        enc_out, dec_out = self.temporal_model(fused_aligned, va_hist)

        # ---------------------------------------------------------
        # 5. Recognition head → (B, T_align, 2)
        # ---------------------------------------------------------
        va_pred = self.recognition_head(enc_out)

        # ---------------------------------------------------------
        # 6. Forecasting head → (B, S, 2)
        # ---------------------------------------------------------
        va_future_pred = self.forecasting_head(dec_out)

        return {
            "va_pred": va_pred,
            "va_future": va_future_pred,
            "va_hist": va_hist,
            "va_future_gt": va_future_gt,
        }
