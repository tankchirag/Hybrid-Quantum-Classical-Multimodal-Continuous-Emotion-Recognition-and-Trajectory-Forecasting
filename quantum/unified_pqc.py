from __future__ import annotations

from typing import Dict
from typing import List

import pennylane as qml
import torch
import torch.nn as nn




# ============================================================
# Unified Parameterized Quantum Circuit
# ============================================================

class UnifiedPQC(nn.Module):
    """
    Unified Parameterized Quantum Circuit.

    Supported configurations:

        4 qubits
        6 qubits
        8 qubits

    Architecture

        Angle Encoding

            ↓

        Data Reuploading

            ↓

        Trainable Rotations

            ↓

        Ring Entanglement

            ↓

        Z Measurements
    """

    SUPPORTED_QUBITS = [4, 6, 8]

    def __init__(
        self,
        num_qubits: int = 4,
        depth: int = 3,
        device_name: str = "lightning.qubit",
        diff_method: str | None = None,
    ) -> None:

        super().__init__()

        if num_qubits not in self.SUPPORTED_QUBITS:
            raise ValueError(
                f"Supported qubits: "
                f"{self.SUPPORTED_QUBITS}"
            )

        self.num_qubits = num_qubits
        self.depth = depth

        self.theta = nn.Parameter(
            torch.randn(
                depth,
                num_qubits,
            ) * 0.1
        )

        self.phi = nn.Parameter(
            torch.randn(
                depth,
                num_qubits,
            ) * 0.1
        )

        self.dev = qml.device(
            device_name,
            wires=num_qubits,
        )

        qnode_kwargs = {
            "interface": "torch",
        }

        # None preserves the original QNode construction.
        if diff_method is not None:
            qnode_kwargs["diff_method"] = diff_method

        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            **qnode_kwargs,
        )

    def _circuit(
        self,
        angles,
        theta,
        phi,
    ):
        """
        PQC forward computation.
        """

        # Initial Encoding

        for i in range(self.num_qubits):
            qml.RY(
                angles[..., i],
                wires=i,
            )

        # Reuploading Layers

        for layer in range(self.depth):

            for i in range(self.num_qubits):
                qml.RY(
                    angles[..., i],
                    wires=i,
                )

            for i in range(self.num_qubits):
                qml.RZ(
                    phi[layer, i],
                    wires=i,
                )

                qml.RY(
                    theta[layer, i],
                    wires=i,
                )

            # Ring Entanglement

            for i in range(
                self.num_qubits - 1
            ):
                qml.CNOT(
                    wires=[i, i + 1]
                )

            qml.CNOT(
                wires=[
                    self.num_qubits - 1,
                    0,
                ]
            )

        return [

            qml.expval(
                qml.PauliZ(i)
            )

            for i in range(
                self.num_qubits
            )

        ]

    def forward(
        self,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        angles

            (B,T,Q)

        Returns
        -------
        Tensor

            (B,T,Q)
        """

        B, T, Q = angles.shape

        if Q != self.num_qubits:

            raise ValueError(
                f"Expected "
                f"{self.num_qubits} "
                f"angles, got {Q}"
            )

        flat = angles.reshape(
            -1,
            self.num_qubits,
        )

        # PennyLane CPU simulators must receive a consistent CPU tensor set.
        # torch.Tensor.to() is differentiable: do not detach or clone into a
        # fresh leaf tensor, otherwise gradients to theta/phi and upstream
        # angle encoders would be lost.
        simulator_device = torch.device("cpu")
        simulator_angles = flat.to(simulator_device)
        simulator_theta = self.theta.to(simulator_device)
        simulator_phi = self.phi.to(simulator_device)

        expvals = self.qnode(
            simulator_angles,
            simulator_theta,
            simulator_phi,
        )

        # QNodes with multiple measurements return a tuple/list of tensors.
        # Keep this operation differentiable and restore the caller's device.
        outputs = torch.stack(expvals, dim=-1)
        outputs = outputs.to(device=angles.device, dtype=angles.dtype)

        return outputs.reshape(B, T, self.num_qubits)
