# models/temporal/transformer_ed.py
"""
Transformer Encoder–Decoder for Temporal Modeling
-------------------------------------------------

This module implements the temporal backbone shared by:

    • Baseline B1 (classical fusion)
    • Baseline B2 (classical + projection)
    • Baseline B3 (PQC fusion)
    • QSAL (Quantum Self‑Attention Layer)

It performs:
    1. Concatenation of fused multimodal features with VA history
    2. Linear projection back to d_model
    3. Positional encoding
    4. Transformer encoder over frame-level tokens
    5. Transformer decoder over learned forecasting queries

Inputs
------
fused   : (B, T, d_model)
    Fused multimodal features from fusion module.

va_hist : (B, T, 2)
    Aligned valence–arousal history.

Outputs
-------
enc_out : (B, T, d_model)
    Encoder outputs for recognition.

dec_out : (B, S, d_model)
    Decoder outputs for forecasting horizon S.

This version is fully aligned with the unified modality contract and
supports REAL + SYNTHETIC dataset modes.
"""

import torch
import torch.nn as nn
import math


# ============================================================
#  Positional Encoding (Sinusoidal)
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Adds position-dependent sin/cos vectors to input embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, d_model)

        Returns
        -------
        (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :].to(x.device)


# ============================================================
#  Transformer Encoder–Decoder
# ============================================================

class TransformerEncoderDecoder(nn.Module):
    """
    Transformer Encoder–Decoder for temporal modeling with VA history.

    This module is used identically across:
        • Baseline B1/B2/B3
        • QSAL

    It receives fused multimodal features and aligned VA history,
    and produces:
        • frame-level recognition outputs
        • future forecasting outputs
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        forecast_horizon: int = 10,
    ):
        super().__init__()

        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # ------------------------------------------------------------
        # 1. Input projection: concat(fused, VA) → d_model
        # ------------------------------------------------------------
        self.input_proj = nn.Linear(d_model + 2, d_model)

        # ------------------------------------------------------------
        # 2. Positional encoding
        # ------------------------------------------------------------
        self.pos_enc = PositionalEncoding(d_model)

        # ------------------------------------------------------------
        # 3. Transformer Encoder
        # ------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # ------------------------------------------------------------
        # 4. Transformer Decoder
        # ------------------------------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # ------------------------------------------------------------
        # 5. Learned forecasting queries (S, d_model)
        # ------------------------------------------------------------
        self.query_tokens = nn.Parameter(
            torch.randn(forecast_horizon, d_model)
        )

    # -------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------
    def forward(self, fused: torch.Tensor, va_hist: torch.Tensor):
        """
        Parameters
        ----------
        fused   : (B, T, d_model)
            Fused multimodal features.

        va_hist : (B, T, 2)
            Aligned valence–arousal history.

        Returns
        -------
        enc_out : (B, T, d_model)
        dec_out : (B, S, d_model)
        """

        B, T, _ = fused.shape

        # ------------------------------------------------------------
        # 1. Concatenate fused features with VA history → (B, T, d_model+2)
        # ------------------------------------------------------------
        x = torch.cat([fused, va_hist], dim=-1)

        # ------------------------------------------------------------
        # 2. Project back to d_model → (B, T, d_model)
        # ------------------------------------------------------------
        x = self.input_proj(x)

        # ------------------------------------------------------------
        # 3. Add positional encoding
        # ------------------------------------------------------------
        x = self.pos_enc(x)

        # ------------------------------------------------------------
        # 4. Encoder
        # ------------------------------------------------------------
        enc_out = self.encoder(x)  # (B, T, d_model)

        # ------------------------------------------------------------
        # 5. Prepare decoder queries: (B, S, d_model)
        # ------------------------------------------------------------
        queries = self.query_tokens.unsqueeze(0).repeat(B, 1, 1)

        # ------------------------------------------------------------
        # 6. Decoder
        # ------------------------------------------------------------
        dec_out = self.decoder(
            tgt=queries,
            memory=enc_out,
        )  # (B, S, d_model)

        return enc_out, dec_out
