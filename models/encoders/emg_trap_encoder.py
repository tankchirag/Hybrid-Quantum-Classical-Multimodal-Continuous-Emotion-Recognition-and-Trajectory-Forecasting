"""
models.encoders.emg_trap_encoder
================================

Trapezius EMG Encoder
"""

from __future__ import annotations

from .temporal_conv_encoder import (
    TemporalConvEncoder,
)


class EMGTrapEncoder(
    TemporalConvEncoder,
):
    """
    Trapezius EMG encoder.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_channels=1,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            modality_name="emg_trap",
        )

    @property
    def modality(
        self,
    ) -> str:
        return "emg_trap"