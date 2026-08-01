"""
qsal_layer.py

QSAL Layer: Quantum Self‑Attention Layer (GPU‑READY HYBRID VERSION)
------------------------------------------------------------------

This module implements your hybrid quantum–classical attention mechanism:

ARCHITECTURE
------------
- PyTorch tensors on GPU
- PennyLane quantum circuits on CPU
- Controlled CPU↔GPU transfer only at the quantum boundary

FLOW
----
1. Input features on GPU
2. Slice per head & modality
3. Move ONLY quantum inputs to CPU
4. Execute PQC via PennyLane
5. Return quantum embeddings to GPU
6. Perform attention on GPU

INPUT SHAPE
-----------
    x : (B, T, M, H, head_dim)

    B  : batch size
    T  : time steps
    M  : number of modalities (modality‑agnostic)
    H  : number of quantum attention heads
    D  : head_dim (per‑head feature dimension)

OUTPUT SHAPE
------------
    fused : (B, T, num_heads * num_qubits)

    For each head, we get a (B, T, num_qubits) quantum embedding,
    then concatenate across heads.
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class QSALLayer(nn.Module):
    """
    Quantum Self‑Attention Layer (Hybrid CPU–GPU design).

    Modality‑agnostic:
        • num_modalities can be any integer (3, 4, 5, ...)
        • x[:, :, m, ...] is treated uniformly for all modalities

    Key design principles:
        - Quantum circuit runs on CPU (PennyLane limitation)
        - All tensor operations remain on GPU
        - Minimal device transfer overhead (only small vectors cross devices)
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

        # ---------------------------------------------------------
        # PennyLane CPU devices (one per Q/K/V per head)
        # ---------------------------------------------------------
        # For each head we have:
        #   - Q circuit
        #   - K circuit
        #   - V circuit
        # → num_heads * 3 devices total
        self.devices = [
            qml.device("default.qubit", wires=num_qubits)
            for _ in range(num_heads * 3)
        ]

        # ---------------------------------------------------------
        # Trainable PQC parameters (stay on GPU with the model)
        # ---------------------------------------------------------
        # Each circuit has its own parameter vector.
        self.params = nn.ParameterList([
            nn.Parameter(
                0.01 * torch.randn(circuit_depth * num_qubits * 2)
            )
            for _ in range(num_heads * 3)
        ])

    # ---------------------------------------------------------
    # QUANTUM CIRCUIT (CPU ONLY)
    # ---------------------------------------------------------
    def _pqc(self, device, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Execute parameterized quantum circuit (CPU‑only).

        Steps:
            1. Move small input vector to CPU
            2. Execute PennyLane circuit
            3. Return quantum expectation values back to original device

        Parameters
        ----------
        device : qml.Device
            PennyLane quantum device.
        params : torch.Tensor
            Trainable PQC parameters (on GPU, moved to CPU inside).
        x : torch.Tensor
            Input feature vector (head_dim), on GPU.

        Returns
        -------
        torch.Tensor
            Quantum embedding of shape (num_qubits,) on the same device as x.
        """

        @qml.qnode(device)
        def circuit(inputs, weights):
            # Encode classical inputs into rotations
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)

            idx = 0
            # Circuit depth: repeated RY/RZ + entangling CNOTs
            for _ in range(len(weights) // (self.num_qubits * 2)):
                # RY layer
                for q in range(self.num_qubits):
                    qml.RY(weights[idx], wires=q)
                    idx += 1
                # RZ layer
                for q in range(self.num_qubits):
                    qml.RZ(weights[idx], wires=q)
                    idx += 1
                # Entangling layer
                for q in range(self.num_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.num_qubits])

            # Measure expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # -----------------------------------------------------
        # CPU CONVERSION (ONLY SMALL VECTOR)
        # -----------------------------------------------------
        # Take first num_qubits components as quantum inputs
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
            device=x.device,
        )

    # ---------------------------------------------------------
    # FORWARD PASS (GPU HEAVY, CPU QUANTUM ONLY)
    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, M, H, head_dim)

            B : batch size
            T : time steps
            M : modalities
            H : heads
            D : head_dim

        Returns
        -------
        torch.Tensor
            Shape (B, T, num_heads * num_qubits)

            For each head:
                fused_head : (B, T, num_qubits)
            Final output:
                concat over heads → (B, T, num_heads * num_qubits)
        """

        B, T, M, H, D = x.shape

        # Sanity checks
        assert H == self.num_heads, "Head dimension mismatch."
        assert D == self.head_dim, "Feature dimension mismatch."

        fused_heads = []

        # -----------------------------------------------------
        # PROCESS EACH ATTENTION HEAD
        # -----------------------------------------------------
        for h in range(self.num_heads):

            # Slice out one head: (B, T, M, D)
            head_slice = x[:, :, :, h, :]

            # Indices for Q/K/V circuits for this head
            q_idx = h * 3
            k_idx = q_idx + 1
            v_idx = q_idx + 2

            Q_list, K_list, V_list = [], [], []

            # -------------------------------------------------
            # PROCESS EACH MODALITY
            # -------------------------------------------------
            # Modality‑agnostic: we just loop over M and treat each xm the same.
            for m in range(self.num_modalities):

                # xm: (B, T, D) for modality m and head h
                xm = head_slice[:, :, m, :]

                # Flatten batch and time for efficient PQC calls
                xm_flat = xm.reshape(-1, self.head_dim)  # (B*T, D)

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
            # Q, K, V: (B, T, M, num_qubits)
            Q = torch.stack(Q_list, dim=2)
            K = torch.stack(K_list, dim=2)
            V = torch.stack(V_list, dim=2)

            # -------------------------------------------------
            # QUANTUM ATTENTION (GPU)
            # -------------------------------------------------
            # attn_scores: (B, T, M)
            attn_scores = (Q * K).sum(dim=-1)

            # attn_weights: (B, T, M)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # Weighted sum over modalities:
            #   V: (B, T, M, num_qubits)
            #   attn_weights.unsqueeze(-1): (B, T, M, 1)
            # → fused_head: (B, T, num_qubits)
            fused_head = (attn_weights.unsqueeze(-1) * V).sum(dim=2)

            fused_heads.append(fused_head)

        # -----------------------------------------------------
        # FINAL CONCAT OVER HEADS (GPU)
        # -----------------------------------------------------
        # fused_heads: list of H tensors (B, T, num_qubits)
        # output: (B, T, H * num_qubits)
        return torch.cat(fused_heads, dim=-1)
