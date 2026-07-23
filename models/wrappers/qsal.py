"""
qsal.py

Proposed Method: QSAL (Quantum Self‑Attention Layer)
----------------------------------------------------

Hybrid quantum–classical multimodal architecture:

- Multimodal feature extraction
- QSAL fusion (Quantum Self‑Attention Layer)
- Transformer encoder–decoder
- V/A recognition + forecasting heads

This wrapper mirrors Baseline B3 structurally, but replaces
PQC fusion with QSALFusion.

IMPORTANT:
- Output format is a tuple (rec_pred_seq, for_pred), as expected by Trainer.
- Aligned targets va_hist and va_future_gt are written into the batch
  so Trainer can compute losses in QSAL mode.
"""

import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors import MultimodalFeatureExtractors
from models.fusion.qsal_fusion import QSALFusion
from models.temporal.transformer_ed import TransformerEncoderDecoder
from models.heads.recognition_head import RecognitionHead
from models.heads.forecasting_head import ForecastingHead


class DotDict(dict):
    """Dictionary with attribute-style access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class QSAL(nn.Module):
    """
    QSAL Wrapper

    Full hybrid quantum–classical architecture:
    - Multimodal feature extraction
    - QSAL fusion (Quantum Self‑Attention Layer)
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
        qsal_cfg: dict,
        dataset_name: str,
    ):
        super().__init__()

        self.modalities = modalities
        self.target_seq_len = target_seq_len
        self.forecast_horizon = forecast_horizon

        # ------------------------------------------------------------
        # 1. Multimodal feature extractors (same pattern as Baseline B3)
        # ------------------------------------------------------------
        self.extractors = MultimodalFeatureExtractors(
            emb_dim=emb_dim,
            synthetic=synthetic,
            target_seq_len=target_seq_len,
            include_text=self.modalities["text"],
        )

        # ------------------------------------------------------------
        # 2. QSAL Fusion configuration
        # ------------------------------------------------------------
        active_modalities = [m for m, use in self.modalities.items() if use]

        self.qsal_cfg = DotDict({
            "modalities": active_modalities,
            "dataset": dataset_name.lower(),
            "num_qubits": qsal_cfg.get("num_qubits", 4),
            "circuit_depth": qsal_cfg.get("circuit_depth", 2),
            "fusion_dim": qsal_cfg.get("fusion_dim", 256),
            "num_heads": qsal_cfg.get("num_heads", 4),
        })

        # QSALFusion expects explicit integer dimensions, not a DotDict
        self.fusion = QSALFusion(
            emb_dim=emb_dim,                          # per‑modality embedding dim
            model_dim=transformer_dim,                # Transformer d_model
            fused_dim=self.qsal_cfg["fusion_dim"],    # intermediate fused dimension
            modalities=active_modalities,
            init_sigma=1.0,
            learnable_sigma=True,
        )

        # PQC/QSAL fusion already outputs (B, T, transformer_dim)
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

    def forward(self, batch: dict):
        """
        QSAL forward pass (tuple output format expected by Trainer).

        Parameters
        ----------
        batch : dict
            Contains:
            - "va": (B, T_va, 2)
            - modality-specific raw signals for extractors

        Returns
        -------
        tuple
            rec_pred_seq : (B, T_align, 2)
            for_pred     : (B, S, 2)

        Side effects
        ------------
        batch["va_hist"]   : (B, T_align, 2)
        batch["va_future"] : (B, S, 2)
        are written so Trainer can compute losses in QSAL mode.
        """
        # ------------------------------------------------------------
        # 0. Ground truth V/A
        # ------------------------------------------------------------
        va_full = batch["va"]          # (B, T_va, 2)
        B, T_va, _ = va_full.shape
        S = self.forecast_horizon

        # Future ground truth (for QSAL mode)
        if S <= T_va:
            va_future_gt = va_full[:, -S:, :]
        else:
            va_future_gt = va_full[:, -1:, :].repeat(1, S, 1)

        # ------------------------------------------------------------
        # 1. Multimodal embeddings
        # ------------------------------------------------------------
        emb_dict = self.extractors(batch)   # each: (B, T, emb_dim)

        # DEBUG: check dtypes
        # for k, v in emb_dict.items():
        #     if v is not None:
        #         print("MODALITY:", k, "DTYPE:", v.dtype)

        # ------------------------------------------------------------
        # 2. QSAL fusion → (B, T, transformer_dim)
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

        # Store aligned targets into batch for Trainer (QSAL mode)
        batch["va_hist"] = va_hist          # (B, T_align, 2)
        batch["va_future"] = va_future_gt   # (B, S, 2)

        # ------------------------------------------------------------
        # 4. Transformer Encoder–Decoder
        # ------------------------------------------------------------
        enc_out, dec_out = self.temporal_model(fused_aligned, va_hist)

        # ------------------------------------------------------------
        # 5. Heads
        # ------------------------------------------------------------
        rec_pred_seq = self.recognition_head(enc_out)   # (B, T_align, 2)
        for_pred = self.forecasting_head(dec_out)       # (B, S, 2)

        # ------------------------------------------------------------
        # 6. QSAL output format for Trainer
        # ------------------------------------------------------------
        # Trainer expects QSAL to return a tuple:
        #   (rec_pred_seq, for_pred)
        return rec_pred_seq, for_pred
