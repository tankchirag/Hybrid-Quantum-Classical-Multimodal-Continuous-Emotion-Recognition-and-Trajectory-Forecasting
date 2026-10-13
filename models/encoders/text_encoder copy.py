# models/encoders/text_encoder.py
"""
TextEncoder
-----------

Unified text encoder supporting BOTH:

    • REAL mode (SEWA)
        - Input: List[str]
        - Tokenizer → DistilBERT → pooled embedding → (B, 128)
        - Replicated across T → (B, T, 128)

    • REAL mode (RECOLA)
        - Input: None
        - Output: zeros (B, T, 128)

    • SYNTHETIC mode (QSAL debugging)
        - Input: Tensor(B, T, D_text) or List[Tensor]
        - Output: random embeddings (B, T, 128)

This encoder is fully aligned with the unified modality contract:

    raw text → tokenizer → DistilBERT → embeddings → fusion → transformer

It is compatible with:
    - Baseline B1/B2/B3
    - QSAL
    - MultimodalFeatureExtractors
    - UnifiedMultimodalDataset (real + synthetic)
"""

import torch
import torch.nn as nn

try:
    from transformers import DistilBertModel, DistilBertTokenizerFast
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class TextEncoder(nn.Module):
    """
    DistilBERT-based text encoder producing (B, T, 128) embeddings.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    emb_dim : int
        Output embedding dimension (128).
    synthetic : bool
        If True → synthetic mode (QSAL debugging).
        If False → real mode (SEWA raw text).
    target_seq_len : int
        Temporal length T for alignment.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        emb_dim: int = 128,
        synthetic: bool = True,
        target_seq_len: int = 50,
    ):
        super().__init__()

        self.synthetic = synthetic
        self.emb_dim = emb_dim
        self.target_seq_len = target_seq_len

        # ---------------------------------------------------------
        # SYNTHETIC MODE: no tokenizer, no backbone
        # ---------------------------------------------------------
        if synthetic:
            self.tokenizer = None
            self.backbone = None
            self.proj = None
            return

        # ---------------------------------------------------------
        # REAL MODE: load DistilBERT
        # ---------------------------------------------------------
        assert _HF_AVAILABLE, "Install transformers to use DistilBERT."

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.backbone = DistilBertModel.from_pretrained(model_name)

        hidden_size = self.backbone.config.dim  # 768
        self.proj = nn.Linear(hidden_size, emb_dim)

    # -------------------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------------------
    def forward(self, text):
        """
        Parameters
        ----------
        text : List[str] | None | Tensor | List[Tensor]
            REAL mode:
                - SEWA: List[str]
                - RECOLA: None
            SYNTHETIC mode:
                - Tensor(B, T, D_text)
                - List[Tensor(T, D_text)]

        Returns
        -------
        Tensor
            Text embeddings of shape (B, T, 128)
        """

        # =========================================================
        # SYNTHETIC MODE
        # =========================================================
        if self.synthetic:
            # text may be Tensor(B, T, D) or List[Tensor]
            if isinstance(text, torch.Tensor):
                B = text.size(0)
                device = text.device
            elif isinstance(text, (list, tuple)):
                B = len(text)
                device = torch.device("cpu")
            else:
                B = 1
                device = torch.device("cpu")

            # Synthetic embeddings
            return torch.randn(B, self.target_seq_len, self.emb_dim, device=device)

        # =========================================================
        # REAL MODE — RECOLA (no text)
        # =========================================================
        if text is None or (isinstance(text, list) and text[0] is None):
            # RECOLA → no text modality
            B = len(text) if isinstance(text, list) else 1
            return torch.zeros(B, self.target_seq_len, self.emb_dim)

        # =========================================================
        # REAL MODE — SEWA (raw text strings)
        # =========================================================
        if isinstance(text, list) and isinstance(text[0], str):
            # Tokenize raw strings
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            # Move to same device as backbone
            encoded = {k: v.to(next(self.backbone.parameters()).device)
                       for k, v in encoded.items()}

            # DistilBERT forward
            outputs = self.backbone(**encoded)
            hidden_states = outputs.last_hidden_state  # (B, L_tokens, 768)

            # Mean pooling across tokens
            pooled = hidden_states.mean(dim=1)  # (B, 768)

            # Project to 128
            emb = self.proj(pooled)  # (B, 128)

            # Replicate across T
            emb = emb.unsqueeze(1).repeat(1, self.target_seq_len, 1)  # (B, T, 128)

            return emb

        # =========================================================
        # REAL MODE — tokenized input (Dict[str, Tensor])
        # =========================================================
        if isinstance(text, dict):
            encoded = {k: v.to(next(self.backbone.parameters()).device)
                       for k, v in text.items()}

            outputs = self.backbone(**encoded)
            hidden_states = outputs.last_hidden_state

            pooled = hidden_states.mean(dim=1)
            emb = self.proj(pooled)
            emb = emb.unsqueeze(1).repeat(1, self.target_seq_len, 1)

            return emb

        # =========================================================
        # FALLBACK — unexpected type
        # =========================================================
        raise ValueError(
            f"TextEncoder received unsupported input type: {type(text)}"
        )
