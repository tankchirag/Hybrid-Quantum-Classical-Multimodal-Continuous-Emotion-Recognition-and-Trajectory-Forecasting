"""
QSAL Parameterised Quantum Circuit (PQC)

This module implements the 4‑qubit variational quantum circuit used in the
QSAL (Quantum Self‑Attention Layer) fusion mechanism.

Design is fully aligned with the thesis_expose_detailed.pdf:
- 4 qubits
- Angle encoding of a 4‑dim classical vector
- L variational layers (RY + RZ + ring entanglement)
- Z‑expectation measurement on all qubits
- Output: 4‑dim vector in [-1, 1]

This file is placed under: quantum/qsal_pqc.py
to remain consistent with the existing project structure.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import pennylane as qml


class QSALPQC(nn.Module):
    """
    4‑Qubit Parameterised Quantum Circuit for QSAL.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (fixed to 4 for QSAL).
    depth : int
        Number of variational layers (rotation + entanglement).
    device_name : str
        PennyLane backend device.
    shots : int or None
        Number of shots. None = analytic mode.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        depth: int = 2,
        device_name: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        super().__init__()

        if n_qubits != 4:
            raise ValueError("QSALPQC supports only 4‑qubit circuits.")

        self.n_qubits = n_qubits
        self.depth = depth

        # PennyLane device
        self._dev = qml.device(device_name, wires=n_qubits, shots=shots)

        # Trainable parameters: (depth, n_qubits, 2) → RY + RZ per qubit
        self.theta = nn.Parameter(torch.zeros(depth, n_qubits, 2))

        # QNode
        self._qnode = qml.QNode(self._circuit, self._dev, interface="torch")

    # ----------------------------------------------------------------------
    #  Encoding + Variational Layers
    # ----------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> None:
        """Angle‑encode a 4‑dim classical vector into RY rotations."""
        for wire in range(self.n_qubits):
            qml.RY(x[wire], wires=wire)

    def _layer(self, params: torch.Tensor) -> None:
        """One variational layer: RY → RZ → ring entanglement."""
        for wire in range(self.n_qubits):
            qml.RY(params[wire, 0], wires=wire)
            qml.RZ(params[wire, 1], wires=wire)

        # Ring entanglement: 0→1, 1→2, 2→3, 3→0
        for wire in range(self.n_qubits):
            qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])

    # ----------------------------------------------------------------------
    #  QNode Circuit
    # ----------------------------------------------------------------------

    def _circuit(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Full PQC circuit executed inside the QNode."""
        self._encode(x)

        for l in range(self.depth):
            self._layer(theta[l])

        return tuple(qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits))

    # ----------------------------------------------------------------------
    #  Forward Pass
    # ----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 4) or (4,). Classical input vector(s).

        Returns
        -------
        torch.Tensor
            Shape (B, 4) or (4,). Z‑expectation values.
        """
        single = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single = True

        if x.size(-1) != 4:
            raise ValueError("QSALPQC expects input dimension = 4.")

        outputs = []
        for b in range(x.size(0)):
            out = self._qnode(x[b], self.theta)
            outputs.append(torch.stack(out))

        out = torch.stack(outputs)

        return out.squeeze(0) if single else out
