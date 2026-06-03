import torch
import torch.nn as nn


class ModalityTokenizer(nn.Module):
    """
    ModalityTokenizer
    -----------------
    Tokenizes modality-specific embeddings into a unified token space.

    This module is used by:
        - Baseline B1 (classical fusion)
        - Baseline B2/B3
        - QSAL (quantum self-attention layer)

    It accepts a dictionary of modality embeddings:
        {
            "audio": (B, T, 128) or None,
            "video": (B, T, 128) or None,
            "ecg":   (B, T, 128) or None,
            "eda":   (B, T, 128) or None,
            "text":  (B, T, 128) or None
        }

    Responsibilities:
        - Apply a learnable projection per modality (128 → 128)
        - Add a learned modality/type embedding
        - Stack all modality tokens into shape (B, T, M, 128)
        - Support variable modality sets (RECOLA vs SEWA)
        - Maintain fixed modality ordering for reproducibility

    Output:
        tokens: Tensor of shape (B, T, M, 128)
    """

    def __init__(self, emb_dim: int = 128, num_modalities: int = 5, dropout: float = 0.1):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_modalities = num_modalities

        # Fixed modality order (important for reproducibility)
        self.modality_order = ["audio", "video", "ecg", "eda", "text"]

        # One projection per modality: 128 → 128
        self.projections = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim) for _ in range(num_modalities)
        ])

        # One learned modality embedding per modality
        self.modality_embeddings = nn.Parameter(
            torch.randn(num_modalities, emb_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, emb_dict):
        """
        Parameters
        ----------
        emb_dict : dict
            Dictionary mapping modality names to embeddings of shape (B, T, 128).

        Returns
        -------
        tokens : Tensor
            Stacked modality tokens of shape (B, T, M, 128)
        """

        token_list = []
        proj_index = 0

        for key in self.modality_order:
            emb = emb_dict.get(key, None)

            if emb is None:
                continue  # skip missing modalities (e.g., RECOLA has no text)

            # emb: (B, T, 128)
            proj = self.projections[proj_index](emb)  # (B, T, 128)

            # Add modality/type embedding
            type_emb = self.modality_embeddings[proj_index].view(1, 1, -1)
            proj = proj + type_emb

            proj = self.dropout(proj)

            token_list.append(proj)
            proj_index += 1

        # Stack into (B, T, M, 128)
        tokens = torch.stack(token_list, dim=2)

        return tokens
