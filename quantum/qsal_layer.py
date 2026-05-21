"""
QSAL Layer: Quantum Self-Attention Layer

This module implements the reusable quantum attention block used in:
- QSAL Fusion (per-frame modality fusion)
- Future QSAL-Transformer layers (optional)

It performs:
- Multi-head quantum attention
- PQC-based Q/K/V generation
- Attention weights from quantum states
- Value aggregation
- Head concatenation

Input:
    x: (B, T, M, head_dim)
        B = batch size
        T = time steps
        M = modalities (4)
        head_dim = fusion_dim / num_heads (e.g., 32)

Output:
    fused: (B, T, num_qubits * num_heads)
"""

import torch
import torch.nn as nn
import pennylane as qml


class QSALLayer(nn.Module):
    """
    Quantum Self-Attention Layer (QSAL)
    """

    def __init__(
        self,
        num_heads=2,
        head_dim=32,
        num_modalities=4,
        num_qubits=4,
        circuit_depth=2,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_modalities = num_modalities
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth

        # ---------------------------------------------------------
        # Create PQCs for each head and each of Q, K, V
        # Total circuits = num_heads * 3
        # ---------------------------------------------------------
        self.devices = [
            qml.device("default.qubit", wires=num_qubits)
            for _ in range(num_heads * 3)
        ]

        self.params = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(circuit_depth * num_qubits * 2))
            for _ in range(num_heads * 3)
        ])

    # ---------------------------------------------------------
    # Quantum circuit definition
    # ---------------------------------------------------------
    def _pqc(self, device, params, x):
        """
        PQC for Q/K/V generation.
        x: classical input vector (head_dim)
        """

        @qml.qnode(device, interface="torch")
        def circuit(inputs, weights):
            # Angle encoding
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            idx = 0
            for _ in range(len(weights) // (self.num_qubits * 2)):
                for q in range(self.num_qubits):
                    qml.RY(weights[idx], wires=q)
                    idx += 1
                for q in range(self.num_qubits):
                    qml.RZ(weights[idx], wires=q)
                    idx += 1
                # Ring entanglement
                for q in range(self.num_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.num_qubits])

            # Expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # Pad or truncate input to num_qubits
        x = x[..., :self.num_qubits]
        return circuit(x, params)

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (B, T, M, head_dim)

        Returns
        -------
        fused : Tensor (B, T, num_heads * num_qubits)
        """

        B, T, M, D = x.shape
        fused_heads = []

        # ---------------------------------------------------------
        # Process each head independently
        # ---------------------------------------------------------
        for h in range(self.num_heads):
            # Extract head slice: (B, T, M, head_dim)
            head_slice = x[:, :, :, h * D : (h + 1) * D]

            # PQC indices
            q_idx = h * 3
            k_idx = h * 3 + 1
            v_idx = h * 3 + 2

            Q, K, V = [], [], []

            # ---------------------------------------------------------
            # Compute Q/K/V for each modality token
            # ---------------------------------------------------------
            for m in range(self.num_modalities):
                xm = head_slice[:, :, m, :]  # (B, T, head_dim)

                q_out = torch.stack([
                    self._pqc(self.devices[q_idx], self.params[q_idx], xm[b, t])
                    for b in range(B) for t in range(T)
                ]).reshape(B, T, self.num_qubits)

                k_out = torch.stack([
                    self._pqc(self.devices[k_idx], self.params[k_idx], xm[b, t])
                    for b in range(B) for t in range(T)
                ]).reshape(B, T, self.num_qubits)

                v_out = torch.stack([
                    self._pqc(self.devices[v_idx], self.params[v_idx], xm[b, t])
                    for b in range(B) for t in range(T)
                ]).reshape(B, T, self.num_qubits)

                Q.append(q_out)
                K.append(k_out)
                V.append(v_out)

            # Stack modalities → (B, T, M, num_qubits)
            Q = torch.stack(Q, dim=2)
            K = torch.stack(K, dim=2)
            V = torch.stack(V, dim=2)

            # ---------------------------------------------------------
            # Quantum attention weights
            # ---------------------------------------------------------
            attn_scores = (Q * K).sum(dim=-1)  # (B, T, M)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # ---------------------------------------------------------
            # Weighted sum of V
            # ---------------------------------------------------------
            fused_head = (attn_weights.unsqueeze(-1) * V).sum(dim=2)  # (B, T, num_qubits)
            fused_heads.append(fused_head)

        # ---------------------------------------------------------
        # Concatenate heads → (B, T, num_heads * num_qubits)
        # ---------------------------------------------------------
        fused = torch.cat(fused_heads, dim=-1)
        return fused
