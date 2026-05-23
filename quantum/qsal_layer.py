"""
QSAL Layer: Quantum Self-Attention Layer (GPU-READY HYBRID VERSION)

This module implements a hybrid quantum-classical attention mechanism:

------------------------------------------------------------
ARCHITECTURE:
------------------------------------------------------------
- PyTorch tensors (GPU)
- PennyLane quantum circuits (CPU only)
- Controlled CPU↔GPU transfer at quantum boundary

------------------------------------------------------------
FLOW:
------------------------------------------------------------
1. Input features on GPU
2. Slice per head & modality
3. Move ONLY quantum inputs to CPU
4. Execute PQC via PennyLane
5. Return quantum embeddings to GPU
6. Perform attention on GPU

------------------------------------------------------------
INPUT:
    x: (B, T, M, H, head_dim)

OUTPUT:
    fused: (B, T, num_heads * num_qubits)
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class QSALLayer(nn.Module):
    """
    Quantum Self-Attention Layer (Hybrid CPU-GPU design).

    Key design principle:
        - Quantum circuit runs on CPU (PennyLane limitation)
        - All tensor operations remain on GPU
        - Minimal device transfer overhead
    """

    def __init__(
        self,
        num_heads: int = 2,
        head_dim: int = 32,
        num_modalities: int = 4,
        num_qubits: int = 4,
        circuit_depth: int = 2,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_modalities = num_modalities
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth

        # PennyLane CPU devices (required limitation)
        self.devices = [
            qml.device("default.qubit", wires=num_qubits)
            for _ in range(num_heads * 3)
        ]

        # Trainable parameters (stay on GPU with model)
        self.params = nn.ParameterList([
            nn.Parameter(
                0.01 * torch.randn(circuit_depth * num_qubits * 2)
            )
            for _ in range(num_heads * 3)
        ])

    # ---------------------------------------------------------
    # QUANTUM CIRCUIT (CPU ONLY)
    # ---------------------------------------------------------
    def _pqc(self, device, params: torch.Tensor, x: torch.Tensor):
        """
        Execute parameterized quantum circuit (CPU-only).

        Steps:
            1. Move inputs to CPU
            2. Execute PennyLane circuit
            3. Return tensor on original device
        """

        @qml.qnode(device)
        def circuit(inputs, weights):
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)

            idx = 0
            for _ in range(len(weights) // (self.num_qubits * 2)):
                for q in range(self.num_qubits):
                    qml.RY(weights[idx], wires=q)
                    idx += 1

                for q in range(self.num_qubits):
                    qml.RZ(weights[idx], wires=q)
                    idx += 1

                for q in range(self.num_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.num_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # -----------------------------------------------------
        # CPU CONVERSION (ONLY SMALL VECTOR)
        # -----------------------------------------------------
        x_cpu = x[:self.num_qubits].detach().cpu().numpy().astype(np.float32)
        w_cpu = params.detach().cpu().numpy().astype(np.float32)

        out_cpu = circuit(x_cpu, w_cpu)
        out_cpu = np.array(out_cpu, dtype=np.float32)

        # -----------------------------------------------------
        # BACK TO ORIGINAL DEVICE (GPU SAFE)
        # -----------------------------------------------------
        return torch.tensor(
            out_cpu,
            dtype=torch.float32,
            device=x.device
        )

    # ---------------------------------------------------------
    # FORWARD PASS (GPU HEAVY, CPU QUANTUM ONLY)
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, M, H, head_dim)

        Returns
        -------
        torch.Tensor
            Shape (B, T, num_heads * num_qubits)
        """

        B, T, M, H, D = x.shape

        assert H == self.num_heads
        assert D == self.head_dim

        fused_heads = []

        # -----------------------------------------------------
        # PROCESS EACH ATTENTION HEAD
        # -----------------------------------------------------
        for h in range(self.num_heads):

            head_slice = x[:, :, :, h, :]  # (B, T, M, D)

            q_idx = h * 3
            k_idx = q_idx + 1
            v_idx = q_idx + 2

            Q_list, K_list, V_list = [], [], []

            # -------------------------------------------------
            # PROCESS MODALITIES
            # -------------------------------------------------
            for m in range(self.num_modalities):

                xm = head_slice[:, :, m, :]  # (B, T, D)

                # flatten for efficiency (reduces loop overhead)
                xm_flat = xm.reshape(-1, self.head_dim)

                # -------------------------------------------------
                # Q / K / V QUANTUM ENCODING
                # -------------------------------------------------
                q_out = torch.stack([
                    self._pqc(self.devices[q_idx], self.params[q_idx], xm_flat[i])
                    for i in range(xm_flat.shape[0])
                ]).reshape(B, T, self.num_qubits)

                k_out = torch.stack([
                    self._pqc(self.devices[k_idx], self.params[k_idx], xm_flat[i])
                    for i in range(xm_flat.shape[0])
                ]).reshape(B, T, self.num_qubits)

                v_out = torch.stack([
                    self._pqc(self.devices[v_idx], self.params[v_idx], xm_flat[i])
                    for i in range(xm_flat.shape[0])
                ]).reshape(B, T, self.num_qubits)

                Q_list.append(q_out)
                K_list.append(k_out)
                V_list.append(v_out)

            # -------------------------------------------------
            # STACK MODALITIES (GPU)
            # -------------------------------------------------
            Q = torch.stack(Q_list, dim=2)
            K = torch.stack(K_list, dim=2)
            V = torch.stack(V_list, dim=2)

            # -------------------------------------------------
            # QUANTUM ATTENTION (GPU)
            # -------------------------------------------------
            attn_scores = (Q * K).sum(dim=-1)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            fused_head = (attn_weights.unsqueeze(-1) * V).sum(dim=2)

            fused_heads.append(fused_head)

        # -----------------------------------------------------
        # FINAL CONCAT (GPU)
        # -----------------------------------------------------
        return torch.cat(fused_heads, dim=-1)