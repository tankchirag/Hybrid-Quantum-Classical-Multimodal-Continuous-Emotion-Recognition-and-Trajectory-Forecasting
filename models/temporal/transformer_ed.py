"""
Transformer Encoder–Decoder for Temporal Modeling with VA History

This module implements the temporal backbone shared by:
- Baseline B1 (classical fusion)
- QSAL fusion model

It performs:
- Concatenation of fused multimodal features with VA history
- Linear projection back to d_model
- Positional encoding
- Transformer encoder over frame-level tokens
- Transformer decoder over learned forecasting queries

Inputs:
    fused:   (B, T, 256)   - fused multimodal features
    va_hist: (B, T, 2)     - valence-arousal history

Outputs:
    enc_out: (B, T, 256)   - encoder outputs (for recognition)
    dec_out: (B, S, 256)   - decoder outputs (for forecasting)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :].to(x.device)


class TransformerEncoderDecoder(nn.Module):
    """
    Transformer Encoder–Decoder for temporal modeling with VA history.
    """

    def __init__(
        self,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        forecast_horizon=10,  # S
    ):
        super().__init__()

        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # Project [fused(256) + VA(2)] → d_model(256)
        self.input_proj = nn.Linear(d_model + 2, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)

        # Encoder
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

        # Decoder
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

        # Learned forecasting queries
        self.query_tokens = nn.Parameter(
            torch.randn(forecast_horizon, d_model)
        )

    def forward(self, fused, va_hist):
        """
        Parameters
        ----------
        fused   : Tensor (B, T, 256)
        va_hist : Tensor (B, T, 2)

        Returns
        -------
        enc_out : (B, T, 256)
        dec_out : (B, S, 256)
        """
        B, T, _ = fused.shape

        # 1. Concatenate fused features with VA history → (B, T, 258)
        x = torch.cat([fused, va_hist], dim=-1)

        # 2. Project back to d_model → (B, T, 256)
        x = self.input_proj(x)

        # 3. Add positional encoding
        x = self.pos_enc(x)

        # 4. Encoder
        enc_out = self.encoder(x)  # (B, T, 256)

        # 5. Prepare decoder queries: (B, S, 256)
        queries = self.query_tokens.unsqueeze(0).repeat(B, 1, 1)

        # 6. Decoder
        dec_out = self.decoder(
            tgt=queries,
            memory=enc_out,
        )  # (B, S, 256)

        return enc_out, dec_out
