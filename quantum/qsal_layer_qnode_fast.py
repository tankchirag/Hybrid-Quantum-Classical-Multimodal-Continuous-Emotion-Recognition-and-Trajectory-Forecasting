"""
QSAL Layer: Quantum Self-Attention Layer
========================================

HIGH-PERFORMANCE HYBRID CPU-GPU VERSION
---------------------------------------

This implementation is optimized for:

    - PyTorch GPU training
    - PennyLane CPU quantum simulation
    - Stable Windows execution
    - Reduced Python overhead
    - Cached QNode execution

------------------------------------------------------------
KEY OPTIMIZATION
------------------------------------------------------------

OLD IMPLEMENTATION:
    - Re-created QNode inside _pqc()
    - Extremely expensive
    - Massive slowdown

NEW IMPLEMENTATION:
    - QNodes created ONCE in __init__
    - Reused during entire training
    - Major speedup
    - Same architecture/math/theory

------------------------------------------------------------
THEORETICAL INTEGRITY
------------------------------------------------------------

This optimization DOES NOT change:

    - PQC structure
    - Quantum attention
    - Number of qubits
    - Circuit depth
    - FLOPs (theoretical)
    - Model parameters
    - Thesis methodology

Only execution efficiency is improved.

------------------------------------------------------------
INPUT
------------------------------------------------------------

x shape:
    (B, T, M, H, D)

Where:
    B = batch size
    T = temporal length
    M = modalities
    H = attention heads
    D = head dimension

------------------------------------------------------------
OUTPUT
------------------------------------------------------------

fused shape:
    (B, T, H * num_qubits)
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


class QSALLayer(nn.Module):
    """
    Quantum Self-Attention Layer (QSAL).

    Hybrid design:
        - Quantum simulation on CPU
        - Tensor operations on GPU
        - Cached QNode execution
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

        # -----------------------------------------------------
        # PENNYLANE DEVICES
        # -----------------------------------------------------

        self.devices = [
            qml.device("default.qubit", wires=num_qubits)
            for _ in range(num_heads * 3)
        ]

        # -----------------------------------------------------
        # TRAINABLE QUANTUM PARAMETERS
        # -----------------------------------------------------

        self.params = nn.ParameterList([
            nn.Parameter(
                0.01 * torch.randn(circuit_depth * num_qubits * 2)
            )
            for _ in range(num_heads * 3)
        ])

        # -----------------------------------------------------
        # CACHED QNODES
        # -----------------------------------------------------
        # IMPORTANT OPTIMIZATION:
        # QNodes are created ONCE here instead of repeatedly
        # inside forward().
        # -----------------------------------------------------

        self.qnodes = [
            self._build_qnode(device)
            for device in self.devices
        ]

    # =========================================================
    # BUILD REUSABLE QNODE
    # =========================================================

    def _build_qnode(self, device):
        """
        Create reusable cached QNode.

        Returns
        -------
        callable
            PennyLane quantum circuit
        """

        @qml.qnode(device)
        def circuit(inputs, weights):

            # ---------------------------------------------
            # ANGLE ENCODING
            # ---------------------------------------------

            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)

            idx = 0

            # ---------------------------------------------
            # VARIATIONAL BLOCKS
            # ---------------------------------------------

            for _ in range(self.circuit_depth):

                # Trainable RY rotations
                for q in range(self.num_qubits):
                    qml.RY(weights[idx], wires=q)
                    idx += 1

                # Trainable RZ rotations
                for q in range(self.num_qubits):
                    qml.RZ(weights[idx], wires=q)
                    idx += 1

                # Ring entanglement
                for q in range(self.num_qubits):
                    qml.CNOT(
                        wires=[q, (q + 1) % self.num_qubits]
                    )

            # Measurement
            return [
                qml.expval(qml.PauliZ(i))
                for i in range(self.num_qubits)
            ]

        return circuit

    # =========================================================
    # QUANTUM EXECUTION
    # =========================================================

    def _run_quantum(
        self,
        qnode,
        params: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute quantum circuit safely.

        Parameters
        ----------
        qnode : callable
            Cached PennyLane QNode

        params : torch.Tensor
            Trainable PQC parameters

        x : torch.Tensor
            Input vector

        Returns
        -------
        torch.Tensor
            Quantum embedding
        """

        original_device = x.device

        # -----------------------------------------------------
        # MOVE SMALL INPUT ONLY TO CPU
        # -----------------------------------------------------

        x_cpu = (
            x[:self.num_qubits]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        w_cpu = (
            params
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # -----------------------------------------------------
        # EXECUTE QUANTUM CIRCUIT
        # -----------------------------------------------------

        out = qnode(x_cpu, w_cpu)

        out = np.asarray(
            out,
            dtype=np.float32
        )

        # -----------------------------------------------------
        # RETURN TO ORIGINAL DEVICE (GPU)
        # -----------------------------------------------------

        return torch.tensor(
            out,
            dtype=torch.float32,
            device=original_device
        )

    # =========================================================
    # FORWARD PASS
    # =========================================================

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape:
                (B, T, M, H, D)

        Returns
        -------
        torch.Tensor
            Shape:
                (B, T, H * num_qubits)
        """

        B, T, M, H, D = x.shape

        assert H == self.num_heads
        assert D == self.head_dim

        fused_heads = []

        # -----------------------------------------------------
        # PROCESS EACH ATTENTION HEAD
        # -----------------------------------------------------

        for h in range(self.num_heads):

            head_slice = x[:, :, :, h, :]

            q_idx = h * 3
            k_idx = q_idx + 1
            v_idx = q_idx + 2

            Q_list = []
            K_list = []
            V_list = []

            # -------------------------------------------------
            # PROCESS EACH MODALITY
            # -------------------------------------------------

            for m in range(self.num_modalities):

                xm = head_slice[:, :, m, :]

                # ---------------------------------------------
                # FLATTEN FOR LOWER PYTHON OVERHEAD
                # ---------------------------------------------

                xm_flat = xm.reshape(-1, self.head_dim)

                # ---------------------------------------------
                # QUANTUM Q
                # ---------------------------------------------

                q_out = torch.stack([
                    self._run_quantum(
                        self.qnodes[q_idx],
                        self.params[q_idx],
                        xm_flat[i]
                    )
                    for i in range(xm_flat.shape[0])
                ])

                q_out = q_out.reshape(
                    B,
                    T,
                    self.num_qubits
                )

                # ---------------------------------------------
                # QUANTUM K
                # ---------------------------------------------

                k_out = torch.stack([
                    self._run_quantum(
                        self.qnodes[k_idx],
                        self.params[k_idx],
                        xm_flat[i]
                    )
                    for i in range(xm_flat.shape[0])
                ])

                k_out = k_out.reshape(
                    B,
                    T,
                    self.num_qubits
                )

                # ---------------------------------------------
                # QUANTUM V
                # ---------------------------------------------

                v_out = torch.stack([
                    self._run_quantum(
                        self.qnodes[v_idx],
                        self.params[v_idx],
                        xm_flat[i]
                    )
                    for i in range(xm_flat.shape[0])
                ])

                v_out = v_out.reshape(
                    B,
                    T,
                    self.num_qubits
                )

                Q_list.append(q_out)
                K_list.append(k_out)
                V_list.append(v_out)

            # -------------------------------------------------
            # STACK MODALITIES
            # -------------------------------------------------

            Q = torch.stack(Q_list, dim=2)
            K = torch.stack(K_list, dim=2)
            V = torch.stack(V_list, dim=2)

            # -------------------------------------------------
            # QUANTUM ATTENTION
            # -------------------------------------------------

            attn_scores = (Q * K).sum(dim=-1)

            attn_weights = torch.softmax(
                attn_scores,
                dim=-1
            )

            fused_head = (
                attn_weights.unsqueeze(-1) * V
            ).sum(dim=2)

            fused_heads.append(fused_head)

        # -----------------------------------------------------
        # CONCATENATE HEADS
        # -----------------------------------------------------

        fused = torch.cat(fused_heads, dim=-1)

        return fused