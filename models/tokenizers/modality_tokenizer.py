# models/tokenizers/modality_tokenizer.py
"""
ModalityTokenizer
-----------------

Unified tokenizer for ALL methodologies:

    • Baseline B1 (classical fusion)
    • Baseline B2 (classical + projection)
    • Baseline B3 (PQC fusion)
    • QSAL (quantum self-attention layer)

Responsibilities:
    - Accept modality embeddings of shape (B, T, emb_dim)
    - Apply a learnable projection per modality
    - Add a learned modality/type embedding
    - Stack tokens into shape (B, T, M, emb_dim)
    - Support variable modality sets (RECOLA vs SEWA vs synthetic)
    - Maintain fixed modality ordering for reproducibility

This tokenizer is fully aligned with:
    - UnifiedMultimodalDataset (real + synthetic)
    - multimodal_collate_fn
    - MultimodalFeatureExtractors
    - TextEncoder (real + synthetic)
    - PQC fusion + QSAL fusion
"""

import torch
import torch.nn as nn


class ModalityTokenizer(nn.Module):
    """
    Tokenizes modality embeddings into a unified token space.

    Output:
        tokens: Tensor(B, T, M, emb_dim)
    """

    def __init__(
        self,
        emb_dim: int = 128,
        num_modalities: int = 5,
        modality_order=None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Fixed modality order (important for reproducibility)
        # ---------------------------------------------------------
        if modality_order is None:
            modality_order = ["audio", "video", "ecg", "eda", "text"]

        # Only keep the first num_modalities modalities
        self.modality_order = modality_order[:num_modalities]
        self.emb_dim = emb_dim

        # ---------------------------------------------------------
        # One projection per modality (emb_dim → emb_dim)
        # ---------------------------------------------------------
        self.projections = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim) for _ in self.modality_order
        ])

        # ---------------------------------------------------------
        # One learned modality embedding per modality
        # ---------------------------------------------------------
        self.modality_embeddings = nn.Parameter(
            torch.randn(len(self.modality_order), emb_dim)
        )

        self.dropout = nn.Dropout(dropout)

    # -------------------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------------------
    def forward(self, emb_dict):
        """
        Parameters
        ----------
        emb_dict : dict
            {
                "audio": (B, T, emb_dim),
                "video": (B, T, emb_dim),
                "ecg":   (B, T, emb_dim),
                "eda":   (B, T, emb_dim),
                "text":  (B, T, emb_dim) or None
            }

        Returns
        -------
        tokens : Tensor(B, T, M, emb_dim)
        """

        token_list = []

        # Iterate in fixed modality order
        for idx, modality in enumerate(self.modality_order):

            emb = emb_dict.get(modality, None)

            # Skip missing modalities (e.g., RECOLA text, SEWA ecg/eda)
            if emb is None:
                continue

            # emb: (B, T, emb_dim)
            proj = self.projections[idx](emb)

            # Add learned modality/type embedding
            type_emb = self.modality_embeddings[idx].view(1, 1, -1)
            proj = proj + type_emb

            proj = self.dropout(proj)

            token_list.append(proj)

        # Stack into (B, T, M, emb_dim)
        tokens = torch.stack(token_list, dim=2)

        return tokens
