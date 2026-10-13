# models/wrappers/baseline_b1.py
"""
Baseline B1 — Classical Multimodal Fusion
-----------------------------------------

This wrapper implements the full Baseline B1 architecture:

    Step‑A: MultimodalFeatureExtractors
    Step‑B: ModalityTokenizer
    Step‑C: ClassicalFusion (self‑attention over modalities)
    Step‑D: Transformer Encoder–Decoder (temporal modeling)
    Step‑E: Recognition + Forecasting Heads

Fully aligned with the unified modality contract:

    • REAL mode (RECOLA / SEWA)
        - audio/video/ecg/eda/text → encoders → (B, T, 128)
    • SYNTHETIC mode (debugging)
        - numeric tensors → synthetic encoders → (B, T, 128)

This version eliminates all text‑modality errors and supports:

    • RECOLA (no text)
    • SEWA (raw text → DistilBERT)
    • Synthetic debugging
"""

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
    Baseline B1 — Classical Fusion

    Parameters
    ----------
    emb_dim : int
        Per‑modality embedding dimension (128).
    fusion_dim : int
        Classical fusion bottleneck dimension.
    transformer_dim : int
        Transformer d_model (256).
    num_heads : int
        Transformer attention heads.
    num_layers : int
        Transformer encoder/decoder layers.
    target_seq_len : int
        Temporal length T.
    forecast_horizon : int
        Future prediction horizon S.
    synthetic : bool
        If True → synthetic mode (debugging).
        If False → real mode (RECOLA/SEWA).
    modalities : dict
        Modality flags (audio/video/ecg/eda/text).
    """

    def __init__(
        self,
        emb_dim: int = 128,
        fusion_dim: int = 128,
        transformer_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        target_seq_len: int = 50,
        forecast_horizon: int = 10,
        synthetic: bool = False,
        modalities: dict = None,
    ):
        super().__init__()

        self.modalities = modalities or {
            "audio": True,
            "video": True,
            "ecg": True,
            "eda": True,
            "text": True,
        }

        self.target_seq_len = target_seq_len
        self.forecast_horizon = forecast_horizon

        # ------------------------------------------------------------
        # 1. Multimodal Feature Extractors (REAL + SYNTHETIC)
        # ------------------------------------------------------------
        self.extractors = MultimodalFeatureExtractors(
            emb_dim=emb_dim,
            synthetic=synthetic,
            target_seq_len=target_seq_len,
            include_text=self.modalities["text"],
        )

        # ------------------------------------------------------------
        # 2. Modality Tokenizer → (B, T, M, emb_dim)
        # ------------------------------------------------------------
        num_modalities = sum(1 for m, use in self.modalities.items() if use)

        self.tokenizer = ModalityTokenizer(
            emb_dim=emb_dim,
            num_modalities=num_modalities,
        )

        # ------------------------------------------------------------
        # 3. Classical Fusion → (B, T, transformer_dim)
        # ------------------------------------------------------------
        self.fusion = ClassicalFusion(
            emb_dim=emb_dim,
            fusion_dim=fusion_dim,
            transformer_dim=transformer_dim,
            num_heads=num_heads,
        )

        # ------------------------------------------------------------
        # 4. Transformer Encoder–Decoder
        # ------------------------------------------------------------
        self.temporal_model = TransformerEncoderDecoder(
            d_model=transformer_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            forecast_horizon=forecast_horizon,
        )

        # ------------------------------------------------------------
        # 5. Recognition head (V/A regression)
        # ------------------------------------------------------------
        self.recognition_head = RecognitionHead(
            input_dim=transformer_dim,
            output_dim=2,
        )

        # ------------------------------------------------------------
        # 6. Forecasting head (future V/A regression)
        # ------------------------------------------------------------
        self.forecasting_head = ForecastingHead(
            input_dim=transformer_dim,
            output_dim=2,
        )

    # -------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------
    def forward(self, batch: dict):
        """
        Forward pass for Baseline B1.

        Parameters
        ----------
        batch : dict
            Contains:
                - raw signals (REAL mode)
                - synthetic tensors (SYNTHETIC mode)
                - "va": (B, T_va, 2)

        Returns
        -------
        dict
            {
                "va_pred":      (B, T_align, 2),
                "va_future":    (B, S, 2),
                "va_hist":      (B, T_align, 2),
                "va_future_gt": (B, S, 2)
            }
        """

        # ------------------------------------------------------------
        # 0. Ground truth V/A
        # ------------------------------------------------------------
        va_full = batch["va"]          # (B, T_va, 2)
        B, T_va, _ = va_full.shape
        S = self.forecast_horizon

        # Future ground truth
        if S <= T_va:
            va_future_gt = va_full[:, -S:, :]
        else:
            va_future_gt = va_full[:, -1:, :].repeat(1, S, 1)

        # ------------------------------------------------------------
        # 1. Multimodal embeddings (REAL + SYNTHETIC)
        # ------------------------------------------------------------
        emb_dict = self.extractors(batch)   # each: (B, T, emb_dim)

        # ------------------------------------------------------------
        # 2. Tokenize modalities → (B, T, M, emb_dim)
        # ------------------------------------------------------------
        tokens = self.tokenizer(emb_dict)

        # ------------------------------------------------------------
        # 3. Classical fusion → (B, T, transformer_dim)
        # ------------------------------------------------------------
        fused = self.fusion(tokens)

        # ------------------------------------------------------------
        # 4. Temporal alignment (match fused length with VA length)
        # ------------------------------------------------------------
        T_fused = fused.size(1)
        T_align = min(T_fused, T_va)

        fused_aligned = fused[:, -T_align:, :]
        va_hist = va_full[:, -T_align:, :]

        # ------------------------------------------------------------
        # 5. Transformer Encoder–Decoder
        # ------------------------------------------------------------
        enc_out, dec_out = self.temporal_model(fused_aligned, va_hist)

        # ------------------------------------------------------------
        # 6. Heads
        # ------------------------------------------------------------
        va_pred = self.recognition_head(enc_out)      # (B, T_align, 2)
        va_future_pred = self.forecasting_head(dec_out)  # (B, S, 2)

        # ------------------------------------------------------------
        # 7. Output dictionary
        # ------------------------------------------------------------
        return {
            "va_pred": va_pred,
            "va_future": va_future_pred,
            "va_hist": va_hist,
            "va_future_gt": va_future_gt,
        }
