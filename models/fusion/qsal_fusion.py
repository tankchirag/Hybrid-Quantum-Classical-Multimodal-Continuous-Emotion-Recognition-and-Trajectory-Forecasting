"""
QSAL Fusion Module (Quantum Self-Attention Fusion)

Implements the quantum fusion mechanism described in the thesis:
- Project each modality embedding (128-dim) to fusion dimension (64-dim)
- Split into H heads (default: 2), each with 32-dim subspaces
- For each head, generate Q/K/V using parameterized quantum circuits (PQC)
- Compute attention weights using quantum-derived Q/K
- Compute fused values using quantum-derived V
- Pool modality tokens into a single fused token (64-dim)
- Project fused token to Transformer model dimension (256-dim)

Input:
    audio, video, ecg, eda: (B, T, 128)

Output:
    fused: (B, T, 256)
"""

import torch
import torch.nn as nn
import pennylane as qml


class QSALFusion(nn.Module):
    """
    Quantum Self-Attention Fusion (QSAL)
    """

    def __init__(
        self,
        modality_dim=128,
        fusion_dim=64,
        num_modalities=4,
        num_heads=2,
        num_qubits=4,
        circuit_depth=2,
        transformer_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.num_modalities = num_modalities
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.head_dim = fusion_dim // num_heads
        self.num_qubits = num_qubits

        # -----------------------------
        # 1. Classical projection 128 → 64
        # -----------------------------
        self.projections = nn.ModuleList([
            nn.Linear(modality_dim, fusion_dim) for _ in range(num_modalities)
        ])

        # -----------------------------
        # 2. Quantum circuits for Q/K/V
        # -----------------------------
        self.qdevices = [
            qml.device("default.qubit", wires=num_qubits)
            for _ in range(num_heads * 3)  # Q, K, V per head
        ]

        self.qparams = nn.ParameterList([
            nn.Parameter(0.01 * torch.randn(circuit_depth * num_qubits * 2))
            for _ in range(num_heads * 3)
        ])

        # -----------------------------
        # 3. Output projection 64 → 256
        # -----------------------------
        self.to_transformer = nn.Linear(fusion_dim, transformer_dim)
        self.dropout = nn.Dropout(dropout)

    # ---------------------------------------------------------
    # Quantum circuit definition
    # ---------------------------------------------------------
    def _build_pqc(self, device, params, x):
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
    def forward(self, audio, video, ecg, eda):
        B, T, _ = audio.shape

        # -----------------------------
        # 1. Classical projection 128 → 64
        # -----------------------------
        modalities = [audio, video, ecg, eda]
        proj_tokens = []

        for i, x in enumerate(modalities):
            proj = self.projections[i](x)  # (B, T, 64)
            proj_tokens.append(proj)

        # Stack → (B, T, 4, 64)
        tokens = torch.stack(proj_tokens, dim=2)

        # -----------------------------
        # 2. Split into heads
        # -----------------------------
        # (B, T, 4, 2, 32)
        tokens = tokens.reshape(B, T, self.num_modalities, self.num_heads, self.head_dim)

        # -----------------------------
        # 3. Quantum Q/K/V per head
        # -----------------------------
        fused_heads = []

        for h in range(self.num_heads):
            # Extract head slice: (B, T, 4, 32)
            head_slice = tokens[:, :, :, h, :]

            # Prepare Q/K/V PQCs
            q_idx = h * 3
            k_idx = h * 3 + 1
            v_idx = h * 3 + 2

            # Compute Q/K/V for each modality token
            Q = []
            K = []
            V = []

            for m in range(self.num_modalities):
                x = head_slice[:, :, m, :]  # (B, T, 32)

                # Apply PQCs frame-by-frame
                q_out = torch.stack([
                    self._build_pqc(self.qdevices[q_idx], self.qparams[q_idx], x[b, t])
                    for b in range(B) for t in range(T)
                ]).reshape(B, T, self.num_qubits)

                k_out = torch.stack([
                    self._build_pqc(self.qdevices[k_idx], self.qparams[k_idx], x[b, t])
                    for b in range(B) for t in range(T)
                ]).reshape(B, T, self.num_qubits)

                v_out = torch.stack([
                    self._build_pqc(self.qdevices[v_idx], self.qparams[v_idx], x[b, t])
                    for b in range(B) for t in range(T)
                ]).reshape(B, T, self.num_qubits)

                Q.append(q_out)
                K.append(k_out)
                V.append(v_out)

            # Stack modalities → (B, T, 4, num_qubits)
            Q = torch.stack(Q, dim=2)
            K = torch.stack(K, dim=2)
            V = torch.stack(V, dim=2)

            # -----------------------------
            # 4. Quantum attention weights
            # -----------------------------
            attn_scores = (Q * K).sum(dim=-1)  # (B, T, 4)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # -----------------------------
            # 5. Weighted sum of V
            # -----------------------------
            fused_head = (attn_weights.unsqueeze(-1) * V).sum(dim=2)  # (B, T, num_qubits)
            fused_heads.append(fused_head)

        # -----------------------------
        # 6. Concatenate heads → (B, T, 64)
        # -----------------------------
        fused = torch.cat(fused_heads, dim=-1)

        # -----------------------------
        # 7. Project to Transformer dim (256)
        # -----------------------------
        fused = self.to_transformer(fused)
        fused = self.dropout(fused)

        return fused
