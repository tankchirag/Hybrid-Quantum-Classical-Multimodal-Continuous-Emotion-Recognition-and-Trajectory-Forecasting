import torch
import torch.nn as nn

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class AudioEncoder(nn.Module):
    """
    AudioEncoder
    ------------
    A unified audio feature extractor for both synthetic and real datasets.

    This module converts raw audio waveforms into fixed-length 128‑dimensional
    frame‑level embeddings suitable for multimodal fusion and downstream
    Transformer processing.

    Modes
    -----
    synthetic = True:
        - Used during synthetic dataset experiments.
        - Ignores the input audio and returns random embeddings of shape (B, T, 128).
        - Ensures the architecture remains identical to the real pipeline.

    synthetic = False:
        - Used for real datasets (e.g., RECOLA, SEWA).
        - Accepts raw audio waveforms sampled at 16 kHz.
        - Runs wav2vec2‑base to extract contextualized speech representations.
        - Projects wav2vec2 hidden states (768‑dim) to 128‑dim.
        - Interpolates temporal dimension to a fixed sequence length T.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for wav2vec2 (default: "facebook/wav2vec2-base").
    emb_dim : int
        Output embedding dimension (fixed at 128 for your thesis).
    synthetic : bool
        Whether to operate in synthetic mode.
    freeze_backbone : bool
        Whether to freeze wav2vec2 parameters during training.
    target_seq_len : int
        Number of time steps T expected by the fusion/transformer pipeline.
    sample_rate : int
        Expected audio sampling rate (RECOLA uses 16 kHz).

    Input (real mode)
    -----------------
    audio : Tensor or List[Tensor]
        Raw audio waveforms of shape (B, L) or list of 1D tensors.

    Output
    ------
    audio_emb : Tensor
        Frame‑level embeddings of shape (B, T, 128).
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        emb_dim: int = 128,
        synthetic: bool = True,
        freeze_backbone: bool = True,
        target_seq_len: int = 50,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.synthetic = synthetic
        self.emb_dim = emb_dim
        self.target_seq_len = target_seq_len
        self.sample_rate = sample_rate

        # Synthetic mode: no backbone needed
        if synthetic:
            self.backbone = None
            self.proj = None
            return

        # Real mode: load wav2vec2
        assert _HF_AVAILABLE, "Install transformers to use wav2vec2."

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size  # 768 for wav2vec2-base
        self.proj = nn.Linear(hidden_size, emb_dim)

    def forward(self, audio, lengths=None):
        """
        Forward pass for audio encoding.

        Parameters
        ----------
        audio : Tensor or List[Tensor]
            - Synthetic mode: ignored.
            - Real mode: raw audio waveforms (B, L) or list of 1D tensors.

        Returns
        -------
        Tensor
            Audio embeddings of shape (B, T, 128).
        """

        # ---------------------------------------------------------
        # SYNTHETIC MODE: return random embeddings (B, T, 128)
        # ---------------------------------------------------------
        if self.synthetic:
            if isinstance(audio, torch.Tensor):
                B = audio.size(0)
                device = audio.device
            elif isinstance(audio, (list, tuple)) and len(audio) > 0:
                B = len(audio)
                device = audio[0].device if isinstance(audio[0], torch.Tensor) else torch.device("cpu")
            else:
                B = 1
                device = torch.device("cpu")

            return torch.randn(B, self.target_seq_len, self.emb_dim, device=device)

        # ---------------------------------------------------------
        # REAL MODE: wav2vec2 feature extraction
        # ---------------------------------------------------------
        if isinstance(audio, torch.Tensor):
            inputs = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
        else:
            inputs = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )

        input_values = inputs.input_values.to(next(self.backbone.parameters()).device)
        attention_mask = inputs.attention_mask.to(input_values.device)

        # wav2vec2 forward
        with torch.set_grad_enabled(self.backbone.training):
            outputs = self.backbone(input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (B, T_raw, 768)

        # project to 128
        emb = self.proj(hidden_states)  # (B, T_raw, 128)

        # ---------------------------------------------------------
        # TEMPORAL ALIGNMENT: interpolate to fixed T
        # ---------------------------------------------------------
        B, T_raw, D = emb.shape

        if T_raw == self.target_seq_len:
            return emb

        emb = emb.permute(0, 2, 1)  # (B, 128, T_raw)
        emb = torch.nn.functional.interpolate(
            emb,
            size=self.target_seq_len,
            mode="linear",
            align_corners=False,
        )
        emb = emb.permute(0, 2, 1)  # (B, T, 128)

        return emb
