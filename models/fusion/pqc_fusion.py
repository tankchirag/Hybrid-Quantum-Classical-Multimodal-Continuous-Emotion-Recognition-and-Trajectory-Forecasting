# models/fusion/pqc_fusion.py
"""
Unified PQC Fusion for Baseline B3 (Dynamic Modality Support)
-------------------------------------------------------------

This module implements the complete quantum-classical fusion pipeline
for Baseline B3, now with **flexible modality count**:

    • REAL mode (RECOLA / SEWA)
        - audio/video/ecg/eda/text → encoders → (B, T, 128)
    • SYNTHETIC mode (debugging)
        - numeric tensors → synthetic encoders → (B, T, 128)

Fusion Steps (modality-flexible):
    1. Per‑modality projection: 128 → 16
    2. Stack all active modalities → (B, T, M, 16), M = #modalities
    3. Aggregate across modalities (mean) → (B, T, 16)
    4. Project aggregated slot: 16 → 64
    5. Classical → quantum encoding: 64 → Q angles
    6. Unified PQC (4Q / 6Q / 8Q)
    7. PQC output → 256‑dimensional fused representation

Key properties:
    - Works for any number of active modalities M ≥ 1
    - Output dimension is fixed: 256 (Transformer input)
    - No hard-coded "4-slot" constraint anymore
    - Robust to config.modalities being dict or list
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

    Generalised parameterised quantum circuit supporting:
        • 4 qubits
        • 6 qubits
        • 8 qubits

    Architecture:
        - Initial angle encoding: RY(α_i)
        - L layers of:
            * Data reuploading: RY(α_i)
            * Trainable rotations: RZ(φ_i^ℓ), RY(θ_i^ℓ)
            * Ring entanglement: CNOT(0→1→...→Q‑1→0)
        - Measurement: ⟨Z_i⟩ for all qubits

    Output:
        (B, T, Q) expectation values
    """

    def __init__(
        self,
        num_qubits: int = 4,
        depth: int = 3,
        device_name: str = "lightning.qubit",
    ) -> None:
        super().__init__()

        if num_qubits not in [4, 6, 8]:
            raise ValueError("UnifiedPQC supports only 4, 6, or 8 qubits.")

        self.num_qubits = num_qubits
        self.depth = depth

        # Trainable parameters θ[ℓ, i], φ[ℓ, i]
        self.theta = nn.Parameter(torch.randn(depth, num_qubits) * 0.1)
        self.phi = nn.Parameter(torch.randn(depth, num_qubits) * 0.1)

        # PennyLane backend
        self.dev = qml.device(device_name, wires=self.num_qubits)

        # Torch‑compatible QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

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

            # Ring entanglement
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.num_qubits - 1, 0])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

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
#  BaselineB3PQCFusion — Multimodal PQC Fusion Layer (Dynamic M)
# ============================================================

class BaselineB3PQCFusion(nn.Module):
    """
    BaselineB3PQCFusion (dynamic modality count)

    Fusion Steps:
        1. Project each modality embedding: 128 → 16
        2. Stack all active modalities: (B, T, M, 16)
        3. Aggregate across modalities (mean): (B, T, 16)
        4. Project aggregated slot: 16 → 64
        5. Classical → quantum encoding: 64 → Q
        6. Unified PQC → q(t) ∈ ℝ^Q
        7. Project → f(t) ∈ ℝ²⁵⁶ (Transformer input)

    This design:
        - Supports any number of active modalities M ≥ 1
        - Keeps a fixed fused dimension (256) for the transformer
        - Removes the previous "exactly 4 slots" constraint
        - Accepts config.modalities as dict or list
    """

    def __init__(self, config) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Resolve active modalities from config (dict or list)
        # ---------------------------------------------------------
        raw_modalities = getattr(config, "modalities", None)
        if raw_modalities is None:
            raise ValueError("BaselineB3PQCFusion requires a 'modalities' field in config.")

        # Case 1: dict {modality_name: bool}
        if isinstance(raw_modalities, dict):
            self.modalities = [m for m, flag in raw_modalities.items() if flag]

        # Case 2: list ["audio", "video", ...]
        elif isinstance(raw_modalities, list):
            self.modalities = list(raw_modalities)

        else:
            raise ValueError(
                f"Unsupported type for config.modalities in BaselineB3PQCFusion: {type(raw_modalities)}"
            )

        if len(self.modalities) == 0:
            raise ValueError("BaselineB3PQCFusion received an empty modality list.")

        # ---------------------------------------------------------
        # Per‑modality projection: 128 → 16
        # ---------------------------------------------------------
        self.mod_proj = nn.ModuleDict({
            m: nn.Linear(128, 16) for m in self.modalities
        })

        # Aggregation: 16 → 64 (fixed fused slot dimension)
        self.slot_agg = nn.Linear(16, 64)

        # Classical → quantum encoding: 64 → Q
        self.num_qubits = getattr(config, "num_qubits", 4)
        self.angle_encoder = nn.Linear(64, self.num_qubits)

        # Unified PQC
        self.pqc = UnifiedPQC(
            num_qubits=self.num_qubits,
            depth=getattr(config, "depth", 3),
            device_name="lightning.qubit",
        )

        # PQC output → Transformer input: Q → 256
        self.post_proj = nn.Linear(self.num_qubits, 256)

    def _build_slots(self, features: dict) -> torch.Tensor:
        """
        Build modality‑aggregated representation from modality embeddings.

        Parameters
        ----------
        features : dict
            {
                "audio": (B, T, 128) or None,
                "video": (B, T, 128) or None,
                "ecg":   (B, T, 128) or None,
                "eda":   (B, T, 128) or None,
                "text":  (B, T, 128) or None
            }

        Returns
        -------
        Z : Tensor(B, T, 64)
            Aggregated fused representation (fixed 64‑dim).
        """

        slot_list = []

        # Collect projected slots for all active modalities that are present
        for m in self.modalities:
            h_m = features.get(m, None)
            if h_m is None:
                # Modality inactive or missing in this batch
                continue

            # h_m: (B, T, 128) → z_m: (B, T, 16)
            z_m = self.mod_proj[m](h_m)
            slot_list.append(z_m)

        if len(slot_list) == 0:
            raise ValueError("BaselineB3PQCFusion received no active modality features.")

        # Stack slots: (B, T, M, 16)
        stacked = torch.stack(slot_list, dim=2)

        # Aggregate across modalities (mean): (B, T, 16)
        mean_slots = stacked.mean(dim=2)

        # Project aggregated slot: 16 → 64
        Z = self.slot_agg(mean_slots)  # (B, T, 64)

        return Z

    def forward(self, features: dict) -> torch.Tensor:
        """
        PQC fusion forward pass.

        Parameters
        ----------
        features : dict
            Modality embeddings (B, T, 128) for all active modalities.

        Returns
        -------
        fused : (B, T, 256)
            Fused quantum‑classical representation suitable as transformer input.
        """

        # Step 1: Build Z(t) ∈ ℝ⁶⁴ (modality‑aggregated)
        Z = self._build_slots(features)  # (B, T, 64)

        # Step 2: Encode → α(t) ∈ ℝ^Q
        angles = self.angle_encoder(Z)   # (B, T, Q)

        # Step 3: PQC → q(t) ∈ ℝ^Q
        q = self.pqc(angles)            # (B, T, Q)

        # Step 4: Convert float64 → float32 for PyTorch Linear
        q = q.to(torch.float32)

        # Step 5: Project → f(t) ∈ ℝ²⁵⁶
        return self.post_proj(q)        # (B, T, 256)
