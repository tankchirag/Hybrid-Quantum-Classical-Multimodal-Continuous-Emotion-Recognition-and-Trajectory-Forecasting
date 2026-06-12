"""
baseline_b3.py

Baseline B3: Hybrid Quantum–Classical Multimodal Fusion
-------------------------------------------------------

This wrapper integrates:
1. Multimodal feature extractors (audio, video, physiological, text)
2. Unified PQC fusion (4Q / 6Q / 8Q)
3. Transformer encoder–decoder for temporal modeling
4. Recognition and forecasting heads

The PQC fusion module already outputs a 256‑dimensional sequence,
so no additional projection is applied before the Transformer.
"""

import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors import MultimodalFeatureExtractors
from models.fusion.pqc_fusion import BaselineB3PQCFusion
from models.temporal.transformer_ed import TransformerEncoderDecoder
from models.heads.recognition_head import RecognitionHead
from models.heads.forecasting_head import ForecastingHead


# ------------------------------------------------------------
#  DotDict — convenience wrapper for config dictionaries
# ------------------------------------------------------------
class DotDict(dict):
    """Dictionary with attribute-style access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# ============================================================
#  Baseline B3 Wrapper
# ============================================================

class BaselineB3(nn.Module):
    """
    BaselineB3

    Full hybrid quantum–classical architecture:
    - Multimodal feature extraction
    - PQC fusion (4Q / 6Q / 8Q)
    - Transformer encoder–decoder
    - V/A recognition + forecasting heads
    """

    def __init__(
        self,
        emb_dim: int,
        proj_dim: int,
        transformer_dim: int,
        num_heads: int,
        num_layers: int,
        target_seq_len: int,
        forecast_horizon: int,
        synthetic: bool,
        modalities: dict,
        pqc_cfg: dict,
        dataset_name: str,
    ):
        super().__init__()

        self.modalities = modalities
        self.target_seq_len = target_seq_len
        self.forecast_horizon = forecast_horizon

        # ------------------------------------------------------------
        # 1. Multimodal feature extractors (same as Baseline B2)
        # ------------------------------------------------------------
        self.extractors = MultimodalFeatureExtractors(
            emb_dim=emb_dim,
            synthetic=synthetic,
            target_seq_len=target_seq_len,
            include_text=self.modalities["text"],
        )

        # ------------------------------------------------------------
        # 2. PQC Fusion (Unified 4Q / 6Q / 8Q)
        # ------------------------------------------------------------
        active_modalities = [m for m, use in self.modalities.items() if use]

        pqc_cfg_full = DotDict({
            "modalities": active_modalities,
            "dataset": dataset_name.lower(),
            "num_qubits": pqc_cfg.get("num_qubits", 4),   # 4 / 6 / 8
            "depth": pqc_cfg.get("depth", 3),             # PQC layers
        })

        self.fusion = BaselineB3PQCFusion(pqc_cfg_full)

        # PQC fusion already outputs (B, T, 256)
        self.fused_proj = nn.Identity()

        # ------------------------------------------------------------
        # 3. Transformer Encoder–Decoder
        # ------------------------------------------------------------
        self.temporal_model = TransformerEncoderDecoder(
            d_model=transformer_dim,          # 256
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            forecast_horizon=forecast_horizon,
        )

        # ------------------------------------------------------------
        # 4. Recognition head (V/A regression)
        # ------------------------------------------------------------
        self.recognition_head = RecognitionHead(
            input_dim=transformer_dim,
            output_dim=2,
        )

        # ------------------------------------------------------------
        # 5. Forecasting head (future V/A regression)
        # ------------------------------------------------------------
        self.forecasting_head = ForecastingHead(
            input_dim=transformer_dim,
            output_dim=2,
        )

    # ============================================================
    #  Forward Pass
    # ============================================================
    def forward(self, batch: dict) -> dict:
        """
        Parameters
        ----------
        batch : dict
            Contains:
            - "va": (B, T_va, 2)
            - modality-specific tensors for extractors

        Returns
        -------
        dict with:
            - va_pred: (B, T_align, 2)
            - va_future: (B, S, 2)
            - va_hist: (B, T_align, 2)
            - va_future_gt: (B, S, 2)
        """

        # Ground truth V/A
        va_full = batch["va"]          # (B, T_va, 2)
        B, T_va, _ = va_full.shape
        S = self.forecast_horizon

        # Future ground truth
        if S <= T_va:
            va_future_gt = va_full[:, -S:, :]
        else:
            va_future_gt = va_full[:, -1:, :].repeat(1, S, 1)

        # ------------------------------------------------------------
        # 1. Multimodal embeddings
        # ------------------------------------------------------------
        emb_dict = self.extractors(batch)   # each: (B, T, emb_dim)

        # ------------------------------------------------------------
        # 2. PQC fusion → (B, T, 256)
        # ------------------------------------------------------------
        fused_raw = self.fusion(emb_dict)
        fused = self.fused_proj(fused_raw)  # Identity

        # ------------------------------------------------------------
        # 3. Temporal alignment (match fused length with VA length)
        # ------------------------------------------------------------
        T_fused = fused.size(1)
        T_align = min(T_fused, T_va)

        fused_aligned = fused[:, -T_align:, :]
        va_hist = va_full[:, -T_align:, :]

        # ------------------------------------------------------------
        # 4. Transformer Encoder–Decoder
        # ------------------------------------------------------------
        enc_out, dec_out = self.temporal_model(fused_aligned, va_hist)

        # ------------------------------------------------------------
        # 5. Heads
        # ------------------------------------------------------------
        va_pred = self.recognition_head(enc_out)
        va_future_pred = self.forecasting_head(dec_out)

        return {
            "va_pred": va_pred,
            "va_future": va_future_pred,
            "va_hist": va_hist,
            "va_future_gt": va_future_gt,
        }
