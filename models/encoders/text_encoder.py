import torch
import torch.nn as nn

try:
    from transformers import DistilBertModel, DistilBertTokenizerFast
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class TextEncoder(nn.Module):
    """
    TextEncoder
    -----------
    A DistilBERT-based text feature extractor producing fixed-length 128-dimensional
    embeddings aligned to the multimodal sequence length T.

    This module supports two modes:

    synthetic = True:
        - Used during synthetic dataset experiments.
        - Ignores the input text and returns random embeddings of shape (B, T, 128).
        - Ensures the architecture remains identical to the real pipeline.

    synthetic = False:
        - Used for real datasets (e.g., SEWA).
        - Accepts a list of text strings or tokenized inputs.
        - Runs DistilBERT to obtain contextualized token embeddings (768-dim).
        - Applies mean pooling across tokens to obtain a sentence-level embedding.
        - Projects to 128-dim.
        - Replicates or interpolates to match the target sequence length T.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for DistilBERT.
    emb_dim : int
        Output embedding dimension (fixed at 128 for your thesis).
    synthetic : bool
        Whether to operate in synthetic mode.
    target_seq_len : int
        Number of time steps T expected by the fusion/transformer pipeline.

    Input (real mode)
    -----------------
    text : List[str] or Dict[str, Tensor]
        Raw text strings or pre-tokenized inputs.

    Output
    ------
    text_emb : Tensor
        Text embeddings of shape (B, T, 128).
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

        # Synthetic mode: no backbone needed
        if synthetic:
            self.tokenizer = None
            self.backbone = None
            self.proj = None
            return

        # Real mode: load DistilBERT
        assert _HF_AVAILABLE, "Install transformers to use DistilBERT."

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.backbone = DistilBertModel.from_pretrained(model_name)

        hidden_size = self.backbone.config.dim  # 768 for DistilBERT
        self.proj = nn.Linear(hidden_size, emb_dim)

    def forward(self, text):
        """
        Forward pass for text encoding.

        Parameters
        ----------
        text : List[str] or Dict[str, Tensor]
            - Synthetic mode: ignored.
            - Real mode: raw text strings or tokenized inputs.

        Returns
        -------
        Tensor
            Text embeddings of shape (B, T, 128).
        """

        # ---------------------------------------------------------
        # SYNTHETIC MODE: return random embeddings (B, T, 128)
        # ---------------------------------------------------------
        if self.synthetic:
            if isinstance(text, torch.Tensor):
                B = text.size(0)
                device = text.device
            elif isinstance(text, (list, tuple)):
                B = len(text)
                device = torch.device("cpu")
            else:
                B = 1
                device = torch.device("cpu")

            return torch.randn(B, self.target_seq_len, self.emb_dim, device=device)

        # ---------------------------------------------------------
        # REAL MODE: DistilBERT feature extraction
        # ---------------------------------------------------------
        if isinstance(text, list):
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        else:
            encoded = text  # already tokenized

        encoded = {k: v.to(next(self.backbone.parameters()).device) for k, v in encoded.items()}

        outputs = self.backbone(**encoded)
        hidden_states = outputs.last_hidden_state  # (B, L_tokens, 768)

        # mean pooling across tokens
        pooled = hidden_states.mean(dim=1)  # (B, 768)

        # project to 128
        emb = self.proj(pooled)  # (B, 128)

        # replicate across T time steps
        emb = emb.unsqueeze(1).repeat(1, self.target_seq_len, 1)  # (B, T, 128)

        return emb
