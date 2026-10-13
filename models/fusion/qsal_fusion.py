"""
models.fusion.qsal_fusion
=========================

Quantum Self-Attention Fusion (QSAL)

Purpose
-------

Implements the proposed hybrid quantum-classical
fusion mechanism.

QSAL replaces classical Query / Key / Value
projections with parameterized quantum circuits
while preserving a classical attention mechanism.

Pipeline
--------

Source Embeddings

    (B,T,128)

        ↓

Source Tokens

    (B,T,N,128)

        ↓

128 → 4

        ↓

Quantum Query Circuit

Quantum Key Circuit

Quantum Value Circuit

        ↓

Gaussian Attention

        ↓

Source Pooling

        ↓

4 → fused_dim

        ↓

fused_dim → 256

        ↓

(B,T,256)

Design Goals
------------

✓ Dataset Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Variable Source Count

✓ Missing Source Compatible

✓ Future Proof

Input
-----

embeddings

{
    source_name:
        (B,T,128)
}

Output
------

fused

    (B,T,256)

Transformer-ready representation.

Notes
-----

The fusion layer operates on source tokens.

Sources may originate from:

Physiological Sources

    ECG
    EDA
    EEG
    EMG
    RESP
    BPM

Behavioral Sources

    Eye
    Audio
    Video
    Text

Future Sources

    Continuous Label History

    Discrete Label History

The fusion algorithm remains unchanged
regardless of source type.
"""


from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

import torch
import torch.nn as nn

from quantum.unified_pqc import UnifiedPQC

from models.attention.gaussian_attention import (
    GaussianAttention,
)


class QSALFusion(
    nn.Module,
):
    """
    Quantum Self-Attention Fusion Layer.
    """

    DEFAULT_SOURCE_ORDER = [

        "ecg",
        "eda",
        "eeg",
        "emg",
        "resp",
        "bpm",

        "eye",

        "audio",
        "video",
        "text",

        "continuous_labels",
        "discrete_labels",
    ]

    def __init__(
        self,
        emb_dim=128,
        model_dim=256,
        fused_dim=64,
        num_qubits=4,
        circuit_depth=2,
        backend="lightning.qubit",
        sources=None,
        init_sigma=1.0,
        learnable_sigma=True,
    ) -> None:

        super().__init__()

        self.emb_dim = emb_dim
        self.model_dim = model_dim
        self.fused_dim = fused_dim

        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.backend = backend

        self.sources = (
            sources
            or self.DEFAULT_SOURCE_ORDER
        )

        # --------------------------------------
        # Fixed QSAL Quantum Dimensions
        # --------------------------------------

        self.pqc_input_dim = num_qubits
        self.pqc_output_dim = num_qubits


        # --------------------------------------
        # Classical → Quantum
        # --------------------------------------

        self.to_pqc_input = nn.Linear(
            emb_dim,
            self.pqc_input_dim,
        )

        # --------------------------------------
        # Q / K / V Quantum Circuits
        # --------------------------------------

        self.q_pqc = UnifiedPQC(
            num_qubits=num_qubits,
            depth=circuit_depth,
            device_name=backend,
        )

        self.k_pqc = UnifiedPQC(
            num_qubits=num_qubits,
            depth=circuit_depth,
            device_name=backend,
        )

        self.v_pqc = UnifiedPQC(
            num_qubits=num_qubits,
            depth=circuit_depth,
            device_name=backend,
        )

        # --------------------------------------
        # Gaussian Attention
        # --------------------------------------

        self.attention = (
            GaussianAttention(
                init_sigma=init_sigma,
                learnable_sigma=learnable_sigma,
            )
        )

        # --------------------------------------
        # Quantum → Transformer Space
        # --------------------------------------

        self.pqc_to_fused = nn.Linear(
            self.pqc_output_dim,
            fused_dim,
        )

        self.fused_to_model = nn.Linear(
            fused_dim,
            model_dim,
        )

    # --------------------------------------------------
    # Source Processing
    # --------------------------------------------------

    def _stack_sources(
        self,
        embeddings: Dict[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """
        Returns

            (B,T,N,128)
        """

        source_tokens = []

        for source in self.sources:

            emb = embeddings.get(
                source,
                None,
            )

            if emb is not None:

                source_tokens.append(
                    emb
                )

        if len(source_tokens) == 0:

            raise RuntimeError(
                "QSALFusion received "
                "no active source embeddings."
            )

        return torch.stack(
            source_tokens,
            dim=2,
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(
        self,
        embeddings: Dict[
            str,
            torch.Tensor,
        ],
        mask: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        """
        Returns

            (B,T,256)
        """

        # ----------------------------------
        # Source Tokens
        #
        # (B,T,N,128)
        # ----------------------------------

        x = self._stack_sources(
            embeddings
        )

        if x.dim() != 4:

            raise ValueError(
                "QSALFusion expects "
                "(B,T,N,D) input. "
                f"Got {x.shape}"
            )

        B, T, N, D = x.shape

        if D != self.emb_dim:

            raise ValueError(
                f"Expected embedding "
                f"dim {self.emb_dim}, "
                f"got {D}"
            )

        # ----------------------------------
        # 128 → 4
        #
        # (B,T,N,4)
        # ----------------------------------

        x_proj = self.to_pqc_input(
            x
        )

        x_flat = x_proj.reshape(
            B * T * N,
            self.pqc_input_dim,
        )

        # ----------------------------------
        # Q / K / V PQCs
        #
        # (B*T*N,4)
        # ----------------------------------

        x_pqc = x_flat.unsqueeze(1)

        q_flat = (
            self.q_pqc(
                x_pqc
            )
            .squeeze(1)
        )

        k_flat = (
            self.k_pqc(
                x_pqc
            )
            .squeeze(1)
        )

        v_flat = (
            self.v_pqc(
                x_pqc
            )
            .squeeze(1)
        )
        # ----------------------------------
        # (B*T,N,4)
        # ----------------------------------

        q = q_flat.view(
            B * T,
            N,
            self.pqc_output_dim,
        )

        k = k_flat.view(
            B * T,
            N,
            self.pqc_output_dim,
        )

        v = v_flat.view(
            B * T,
            N,
            self.pqc_output_dim,
        )

        # ----------------------------------
        # Optional Mask
        # ----------------------------------

        attn_mask = None

        if mask is not None:

            if mask.dim() == 2:

                attn_mask = (
                    mask.unsqueeze(1)
                    .expand(
                        B,
                        T,
                        N,
                    )
                    .reshape(
                        B * T,
                        N,
                    )
                )

            elif mask.dim() == 3:

                attn_mask = (
                    mask.reshape(
                        B * T,
                        N,
                    )
                )

            else:

                raise ValueError(
                    f"Unsupported mask "
                    f"shape {mask.shape}"
                )

        # ----------------------------------
        # Gaussian Attention
        #
        # (B*T,N,4)
        # ----------------------------------

        out_attn = self.attention(
            q=q,
            k=k,
            v=v,
            mask=attn_mask,
        )

        # ----------------------------------
        # Pool Sources
        #
        # (B*T,4)
        # ----------------------------------

        fused_pqc = (
            out_attn.mean(
                dim=1
            )
        )

        # ----------------------------------
        # PennyLane safety
        # ----------------------------------

        fused_pqc = fused_pqc.to(
            torch.float32
        )

        # ----------------------------------
        # 4 → fused_dim
        # ----------------------------------

        fused = self.pqc_to_fused(
            fused_pqc
        )

        # ----------------------------------
        # fused_dim → model_dim
        # ----------------------------------

        fused_model = (
            self.fused_to_model(
                fused
            )
        )

        # ----------------------------------
        # Restore Time Shape
        #
        # (B,T,256)
        # ----------------------------------

        fused_model = (
            fused_model.view(
                B,
                T,
                self.model_dim,
            )
        )

        return fused_model

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    def summary(
        self,
    ) -> dict:

        return {

            "fusion_type":
                "qsal",

            "sources":
                self.sources,

            "embedding_dim":
                self.emb_dim,

            "quantum_dim":
                self.pqc_input_dim,

            "fused_dim":
                self.fused_dim,

            "transformer_dim":
                self.model_dim,
        }