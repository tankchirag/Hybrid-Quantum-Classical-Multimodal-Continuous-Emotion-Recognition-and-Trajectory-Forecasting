"""
Quantum Gaussian Self-Attention Layer (QGSAL)
STABLE VERSION - avoids PennyLane Torch interface bottlenecks
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class QGSALLayer(nn.Module):
    """
    Hybrid Quantum-Classical Gaussian Attention Layer.

    Fixes:
    - removes Torch autograd PennyLane overhead
    - avoids QNode hashing bottleneck
    - uses numpy-only interface
    """

    def __init__(self, dim, n_heads=2, n_qubits=4):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.n_qubits = n_qubits

        self.devices = [
            qml.device("default.qubit", wires=n_qubits)
            for _ in range(n_heads)
        ]

        # learnable sigma (IMPORTANT for Gaussian kernel)
        self.sigma = nn.Parameter(torch.tensor(1.0))

        # PQC weights
        self.params = nn.ParameterList([
            nn.Parameter(torch.randn(n_qubits, 3) * 0.01)
            for _ in range(n_heads)
        ])

        self.qnodes = []
        for h in range(n_heads):
            dev = self.devices[h]

            @qml.qnode(dev, interface="numpy", diff_method=None)
            def circuit(x, weights):
                for i in range(self.n_qubits):
                    qml.RX(x[i % len(x)], wires=i)

                for i in range(self.n_qubits):
                    qml.Rot(*weights[i], wires=i)

                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

            self.qnodes.append(circuit)

        self.proj = nn.Linear(n_heads * n_qubits, dim)

    def gaussian_weights(self, x):
        """
        Learnable Gaussian kernel attention weights.
        """
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        sigma = torch.clamp(self.sigma, 0.1, 10.0)
        return torch.exp(-(x_norm ** 2) / (2 * sigma ** 2))

    def _run_pqc(self, qnode, weights, x_np):
        """
        PURE NUMPY EXECUTION (NO TORCH)
        """
        return np.array(qnode(x_np, weights.detach().cpu().numpy()))

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, D = x.shape

        x_flat = x.mean(dim=1)  # (B, D)

        outputs = []

        for h in range(self.n_heads):

            head_out = []

            for b in range(B):

                x_np = x_flat[b].detach().cpu().numpy()[: self.n_qubits]

                q_out = self._run_pqc(
                    self.qnodes[h],
                    self.params[h],
                    x_np
                )

                head_out.append(q_out)

            head_out = torch.tensor(
                np.stack(head_out),
                dtype=torch.float32,
                device=x.device
            )

            outputs.append(head_out)

        Q = torch.cat(outputs, dim=-1)  # (B, heads * qubits)

        return self.proj(Q)