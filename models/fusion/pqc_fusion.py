"""
models.fusion.pqc_fusion
========================

Pure Quantum Fusion Framework

Baseline B3

Purpose
-------

Implements the multimodal quantum fusion baseline
used throughout the thesis.

The design intentionally separates:

    Source Representation Learning

from

    Quantum Fusion

Every source is first compressed through a
lightweight classical projection stage before
undergoing parameterized quantum processing.

The generated quantum representation is finally
mapped into the common Transformer space.

Scientific Objective
--------------------

Baseline B3 evaluates whether entangled
parameterized quantum circuits (PQCs) can learn
cross-source relationships without using
classical attention mechanisms.

Design Goals
------------

✓ Dataset Agnostic

✓ Label Agnostic

✓ Continuous Labels

✓ Discrete Labels

✓ Recognition Compatible

✓ Forecasting Compatible

✓ Variable Source Count

✓ Future Proof

Input
-----

Source Embeddings

{
    source_name:

        (B,T,128)
}

Examples

    ecg
    eda
    eeg
    emg
    resp
    bpm

    eye

    audio
    video
    text

Future Extensions

    continuous_labels

    discrete_labels

Output
------

Fused Representation

    (B,T,256)

which matches the common Transformer interface
used throughout the thesis.
"""

from __future__ import annotations

from typing import Dict
from typing import List

import pennylane as qml
import torch
import torch.nn as nn

from quantum.unified_pqc import UnifiedPQC


# ============================================================
# Baseline B3 Fusion
# ============================================================

class BaselineB3PQCFusion(nn.Module):
    """
    Baseline B3 PQC Fusion.

    Pipeline

        Source Embeddings

            128

            ↓

             16

            ↓

        Mean Aggregation

            ↓

             64

            ↓

        Quantum Encoding

            ↓

              Q

            ↓

             PQC

            ↓

              Q

            ↓

             256
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
        config,
    ) -> None:

        super().__init__()

        raw_sources = getattr(
            config,
            "sources",
            None,
        )

        if raw_sources is None:
            raw_sources = getattr(
                config,
                "modalities",
                None,
            )

        if raw_sources is None:

            raw_sources = {
                name: True
                for name in self.DEFAULT_SOURCE_ORDER
            }

        print(
            "\n[DEBUG] raw_sources type:",
            type(raw_sources)
        )

        print(
            "[DEBUG] raw_sources:",
            raw_sources
        )

        # ----------------------------------
        # ConfigNamespace Compatibility
        # ----------------------------------

        if hasattr(
            raw_sources,
            "__dict__",
        ):
            raw_sources = vars(
                raw_sources
            )

        # ----------------------------------
        # Source Parsing
        # ----------------------------------

        if isinstance(
            raw_sources,
            dict,
        ):

            self.sources = [

                name

                for name, flag

                in raw_sources.items()

                if flag
            ]

        elif isinstance(
            raw_sources,
            list,
        ):

            self.sources = list(
                raw_sources
            )

        else:

            raise ValueError(

                f"Unsupported source format: "
                f"{type(raw_sources)}"
            )
        
        if len(self.sources) == 0:

            raise ValueError(
                "No active sources configured."
            )

        # ----------------------------------
        # Source Compression
        #
        # 128 → 16
        # ----------------------------------

        self.source_projection = (
            nn.ModuleDict(
                {
                    source:
                    nn.Linear(
                        128,
                        16,
                    )
                    for source
                    in self.sources
                }
            )
        )

        # ----------------------------------
        # Aggregated
        #
        # 16 → 64
        # ----------------------------------

        self.source_aggregation = (
            nn.Linear(
                16,
                64,
            )
        )

        # ----------------------------------
        # PQC Configuration
        # ----------------------------------

        if isinstance(config, dict):

            pqc_cfg = config.get(
                "pqc",
                {},
            )

        else:

            pqc_cfg = getattr(
                config,
                "pqc",
                {},
            )

        if not isinstance(
            pqc_cfg,
            dict,
        ):
            pqc_cfg = vars(
                pqc_cfg
            )

        self.num_qubits = pqc_cfg.get(
            "num_qubits",
            4,
        )

        self.pqc_depth = pqc_cfg.get(
            "depth",
            3,
        )

        self.pqc_backend = pqc_cfg.get(
            "backend",
            "lightning.qubit",
        )

        self.pqc_diff_method = pqc_cfg.get(
            "diff_method",
            None,
        )

        self.pqc_shots = pqc_cfg.get(
            "shots",
            None,
        )

        self.encoding = pqc_cfg.get(
            "encoding",
            "angle",
        )

        self.entanglement = pqc_cfg.get(
            "entanglement",
            "strong",
        )

        # ----------------------------------
        # Quantum Encoding
        # ----------------------------------

        self.angle_encoder = nn.Linear(
            64,
            self.num_qubits,
        )

        # ----------------------------------
        # PQC
        # ----------------------------------

        self.pqc = UnifiedPQC(
            num_qubits=self.num_qubits,
            depth=self.pqc_depth,
            device_name=self.pqc_backend,
            diff_method=self.pqc_diff_method,
        )

        # ----------------------------------
        # Q → 256
        # ----------------------------------

        self.post_projection = (
            nn.Linear(
                self.num_qubits,
                256,
            )
        )

    # -------------------------------------------------------
    # Source Aggregation
    # -------------------------------------------------------

    def _build_source_representation(
        self,
        embeddings: Dict[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """
        Build aggregated source representation.

        Returns

            (B,T,64)
        """

        projected_sources = []

        for source in self.sources:

            x = embeddings.get(
                source,
                None,
            )

            if x is None:
                continue

            x = self.source_projection[
                source
            ](x)

            projected_sources.append(
                x
            )

        if len(projected_sources) == 0:

            raise RuntimeError(
                "No source embeddings found."
            )

        stacked = torch.stack(
            projected_sources,
            dim=2,
        )

        mean_representation = (
            stacked.mean(
                dim=2
            )
        )

        aggregated = (
            self.source_aggregation(
                mean_representation
            )
        )

        return aggregated

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------

    def forward(
        self,
        embeddings: Dict[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings

            {
                source_name:
                    (B,T,128)
            }

        Returns
        -------
        Tensor

            (B,T,256)
        """

        z = (
            self._build_source_representation(
                embeddings
            )
        )

        angles = (
            self.angle_encoder(
                z
            )
        )

        q = self.pqc(
            angles
        )

        q = q.to(
            torch.float32
        )

        fused = (
            self.post_projection(
                q
            )
        )

        return fused

    # -------------------------------------------------------
    # Metadata
    # -------------------------------------------------------

    def summary(
        self,
    ) -> dict:

        return {

            "fusion_type":
                "pqc",

            "num_qubits":
                self.num_qubits,

            "depth":
                self.pqc_depth,

            "backend":
                self.pqc_backend,

            "encoding":
                self.encoding,

            "entanglement":
                self.entanglement,

            "sources":
                self.sources,

            "transformer_dim":
                256,
        }

    def extra_repr(
        self,
    ) -> str:

        return (

            f"qubits={self.num_qubits}, "

            f"depth={self.pqc_depth}, "

            f"backend={self.pqc_backend}"
        )