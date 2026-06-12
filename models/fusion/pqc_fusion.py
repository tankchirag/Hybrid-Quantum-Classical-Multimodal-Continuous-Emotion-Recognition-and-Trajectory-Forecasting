"""
pqc_fusion.py

Baseline B3: Multimodal Entangled PQC Fusion (Unified 4Q / 6Q / 8Q)
------------------------------------------------------------------

This module implements:
1. UnifiedPQC — a parameterised quantum circuit supporting 4, 6, or 8 qubits,
   with data reuploading, trainable RZ/RY rotations, and ring entanglement.
2. BaselineB3PQCFusion — the multimodal fusion layer that maps modality‑specific
   embeddings to a fused 256‑dimensional sequence for the temporal model.

All other components (dataset, feature extractors, transformer, heads, trainer)
are reused from Baseline B1/B2.
"""

import torch
import torch.nn as nn
import pennylane as qml


# ============================================================
#  UnifiedPQC — 4Q / 6Q / 8Q Parameterised Quantum Circuit
# ============================================================

class UnifiedPQC(nn.Module):
    """
    UnifiedPQC

    A generalised parameterised quantum circuit supporting 4, 6, or 8 qubits.

    Architecture:
    - Initial angle encoding: RY(α_i)
    - L layers of:
        * Data reuploading: RY(α_i)
        * Trainable rotations: RZ(φ_i^ℓ), RY(θ_i^ℓ)
        * Ring entanglement: CNOT(0→1→...→Q‑1→0)
    - Measurement: ⟨Z_i⟩ for all qubits

    This matches the Baseline B3 PQC design described in the thesis.
    """

    def __init__(self, num_qubits: int = 4, depth: int = 3,
                 device_name: str = "lightning.qubit") -> None:
        super().__init__()

        if num_qubits not in [4, 6, 8]:
            raise ValueError("UnifiedPQC supports only 4, 6, or 8 qubits.")

        self.num_qubits = num_qubits
        self.depth = depth

        # Trainable parameters θ[ℓ, i], φ[ℓ, i]
        self.theta = nn.Parameter(torch.randn(depth, num_qubits) * 0.1)
        self.phi = nn.Parameter(torch.randn(depth, num_qubits) * 0.1)

        # CPU‑accelerated PennyLane backend
        self.dev = qml.device(device_name, wires=self.num_qubits)

        # Torch‑compatible QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    # ------------------------------------------------------------
    #  Quantum circuit definition
    # ------------------------------------------------------------
    def _circuit(self, angles, theta, phi):
        """
        PQC executed by lightning.qubit.

        Parameters
        ----------
        angles : (Q,)  — data angles α_i
        theta  : (L, Q)
        phi    : (L, Q)

        Returns
        -------
        list of ⟨Z_i⟩ expectation values for all qubits
        """

        # Initial encoding
        for i in range(self.num_qubits):
            qml.RY(angles[i], wires=i)

        # L layers
        for l in range(self.depth):

            # Data reuploading
            for i in range(self.num_qubits):
                qml.RY(angles[i], wires=i)

            # Trainable single‑qubit rotations
            for i in range(self.num_qubits):
                qml.RZ(phi[l, i], wires=i)
                qml.RY(theta[l, i], wires=i)

            # Ring entanglement: 0→1→...→Q‑1→0
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.num_qubits - 1, 0])

        # Z expectation on all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    # ------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------
    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PQC.

        Parameters
        ----------
        angles : (B, T, Q)

        Returns
        -------
        (B, T, Q) — expectation values ⟨Z_i⟩
        """
        B, T, D = angles.shape
        if D != self.num_qubits:
            raise ValueError(f"UnifiedPQC expects {self.num_qubits} angles, got {D}.")

        flat = angles.reshape(-1, self.num_qubits)

        outputs = []
        for a in flat:
            expvals = self.qnode(a, self.theta, self.phi)
            outputs.append(torch.stack(expvals))

        outputs = torch.stack(outputs, dim=0)
        return outputs.reshape(B, T, self.num_qubits)


# ============================================================
#  BaselineB3PQCFusion — Multimodal PQC Fusion Layer
# ============================================================

class BaselineB3PQCFusion(nn.Module):
    """
    BaselineB3PQCFusion

    Steps:
    1. Project each modality embedding h_m(t) ∈ ℝ¹²⁸ → z_m(t) ∈ ℝ¹⁶
    2. Build 4‑slot representation (SEWA uses learned null slot)
    3. Concatenate → Z(t) ∈ ℝ⁶⁴
    4. Encode → α(t) ∈ ℝ^Q  (Q = 4, 6, or 8)
    5. PQC → q(t) ∈ ℝ^Q
    6. Project → f(t) ∈ ℝ²⁵⁶ (Transformer input)
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.modalities = list(config.modalities)
        self.dataset = getattr(config, "dataset", "").lower()
        self.num_qubits = getattr(config, "num_qubits", 4)

        # Per‑modality projection: 128 → 16
        self.mod_proj = nn.ModuleDict({
            m: nn.Linear(128, 16) for m in self.modalities
        })

        # Null slot for SEWA (3 modalities)
        self.null_slot = nn.Parameter(torch.zeros(16))

        # Classical → quantum encoding: 64 → Q
        self.angle_encoder = nn.Linear(64, self.num_qubits)

        # Unified PQC
        self.pqc = UnifiedPQC(
            num_qubits=self.num_qubits,
            depth=getattr(config, "depth", 3),
            device_name="lightning.qubit"
        )

        # PQC output → Transformer input: Q → 256
        self.post_proj = nn.Linear(self.num_qubits, 256)

    # ------------------------------------------------------------
    #  Build 4‑slot representation
    # ------------------------------------------------------------
    def _build_slots(self, features: dict) -> torch.Tensor:
        slot_list = []

        for m in self.modalities:
            h_m = features[m]          # (B, T, 128)
            z_m = self.mod_proj[m](h_m)
            slot_list.append(z_m)

        B, T, _ = slot_list[0].shape

        # SEWA: add learned null slot
        if self.dataset == "sewa" and len(slot_list) == 3:
            null = self.null_slot.view(1, 1, 16).expand(B, T, 16)
            slot_list.append(null)

        if len(slot_list) != 4:
            raise ValueError(f"Expected 4 slots, got {len(slot_list)}.")

        return torch.cat(slot_list, dim=-1)  # (B, T, 64)

    # ------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------
    def forward(self, features: dict) -> torch.Tensor:

        # Step 1: Build Z(t) ∈ ℝ⁶⁴
        Z = self._build_slots(features)

        # Step 2: Encode → α(t) ∈ ℝ^Q
        angles = self.angle_encoder(Z)

        # Step 3: PQC → q(t) ∈ ℝ^Q
        q = self.pqc(angles)

        # Step 4: Convert float64 → float32 for PyTorch Linear
        q = q.to(torch.float32)

        # Step 5: Project → f(t) ∈ ℝ²⁵⁶
        return self.post_proj(q)
