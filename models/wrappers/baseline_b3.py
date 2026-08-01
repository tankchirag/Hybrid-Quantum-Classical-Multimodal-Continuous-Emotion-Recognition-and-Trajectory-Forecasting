# models/wrappers/baseline_b3.py
"""
Baseline B3 Wrapper
-------------------

Hybrid Quantum–Classical Multimodal Fusion (Unified PQC)

This wrapper implements the full Baseline B3 architecture:

    Step‑A: MultimodalFeatureExtractors
    Step‑B: PQC Fusion (BaselineB3PQCFusion)
    Step‑C: Transformer Encoder–Decoder
    Step‑D: Recognition + Forecasting Heads

Fully aligned with the unified modality contract:

    • REAL mode (RECOLA / SEWA)
        - raw audio/video/ecg/eda/text → encoders → (B, T, 128)
    • SYNTHETIC mode (debugging)
        - numeric tensors → synthetic encoders → (B, T, 128)

This version eliminates all text‑modality errors and supports:

    • RECOLA (no text)
    • SEWA (raw text → DistilBERT)
    • Synthetic QSAL debugging
"""

import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors import MultimodalFeatureExtractors
from models.fusion.pqc_fusion import BaselineB3PQCFusion
from models.temporal.transformer_ed import TransformerEncoderDecoder
from models.heads.recognition_head import RecognitionHead
from models.heads.forecasting_head import ForecastingHead


class DotDict(dict):
    """Dictionary with attribute-style access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class BaselineB3(nn.Module):
    """
    Baseline B3 — Unified PQC Fusion

    Parameters
    ----------
    emb_dim : int
        Per‑modality embedding dimension (128).
    proj_dim : int
        Classical projection dimension (unused here).
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
    pqc_cfg : dict
        PQC configuration (num_qubits, depth).
    dataset_name : str
        "recola", "sewa", or "synthetic".
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
        # 1. Multimodal Feature Extractors (REAL + SYNTHETIC)
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
            "num_qubits": pqc_cfg.get("num_qubits", 4),
            "depth": pqc_cfg.get("depth", 3),
        })

        # PQC fusion outputs (B, T, transformer_dim)
        self.fusion = BaselineB3PQCFusion(pqc_cfg_full)

        # Identity projection (PQC already outputs correct dim)
        self.fused_proj = nn.Identity()

        # ------------------------------------------------------------
        # 3. Transformer Encoder–Decoder
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

    # -------------------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        """
        Forward pass for Baseline B3.

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
        # 2. PQC fusion → (B, T, transformer_dim)
        # ------------------------------------------------------------
        fused_raw = self.fusion(emb_dict)
        fused = self.fused_proj(fused_raw)

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
        va_pred = self.recognition_head(enc_out)      # (B, T_align, 2)
        va_future_pred = self.forecasting_head(dec_out)  # (B, S, 2)

        # ------------------------------------------------------------
        # 6. Output dictionary
        # ------------------------------------------------------------
        return {
            "va_pred": va_pred,
            "va_future": va_future_pred,
            "va_hist": va_hist,
            "va_future_gt": va_future_gt,
        }
