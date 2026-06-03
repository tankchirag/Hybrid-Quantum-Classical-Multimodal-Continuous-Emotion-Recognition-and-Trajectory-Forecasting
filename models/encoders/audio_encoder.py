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
    Unified audio feature extractor for both synthetic and real datasets.

    This module converts raw audio waveforms into fixed-length frame-level
    embeddings suitable for multimodal fusion and downstream Transformer
    processing in your thesis architecture.

    Modes
    -----
    synthetic = True:
        - Used during synthetic dataset experiments.
        - Ignores the input audio and returns random embeddings of shape (B, T, emb_dim).
        - Ensures the architecture (dimensions, interfaces) matches the real pipeline.

    synthetic = False:
        - Used for real datasets (e.g., RECOLA, SEWA).
        - Accepts raw audio waveforms sampled at `sample_rate` (default: 16 kHz).
        - Uses wav2vec2-base to extract contextualized speech representations.
        - Projects wav2vec2 hidden states (768-dim) to `emb_dim`.
        - Interpolates temporal dimension to a fixed sequence length `target_seq_len`.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for wav2vec2
        (default: "facebook/wav2vec2-base").
    emb_dim : int
        Output embedding dimension (e.g., 128 or 256).
    synthetic : bool
        Whether to operate in synthetic mode.
    freeze_backbone : bool
        Whether to freeze wav2vec2 parameters during training.
    target_seq_len : int
        Number of time steps T expected by the fusion/transformer pipeline.
    sample_rate : int
        Expected audio sampling rate (RECOLA/SEWA typically use 16 kHz).

    Input (real mode)
    -----------------
    audio : Tensor or List[Tensor]
        Raw audio waveforms of shape (B, L) or list of 1D tensors.

    Output
    ------
    audio_emb : Tensor
        Frame-level embeddings of shape (B, T, emb_dim).
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

        # ---------------------------------------------------------
        # SYNTHETIC MODE: no backbone needed
        # ---------------------------------------------------------
        if synthetic:
            self.processor = None
            self.backbone = None
            self.proj = None
            return

        # ---------------------------------------------------------
        # REAL MODE: load wav2vec2 backbone + processor
        # ---------------------------------------------------------
        assert _HF_AVAILABLE, "Install `transformers` to use wav2vec2 in real mode."

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
            - Synthetic mode: ignored (only used to infer batch size/device).
            - Real mode: raw audio waveforms (B, L) or list of 1D tensors.

        Returns
        -------
        Tensor
            Audio embeddings of shape (B, T, emb_dim).
        """

        # ---------------------------------------------------------
        # SYNTHETIC MODE: return random embeddings (B, T, emb_dim)
        # ---------------------------------------------------------
        if self.synthetic:
            if isinstance(audio, torch.Tensor):
                B = audio.size(0)
                device = audio.device
            elif isinstance(audio, (list, tuple)) and len(audio) > 0:
                B = len(audio)
                device = (
                    audio[0].device
                    if isinstance(audio[0], torch.Tensor)
                    else torch.device("cpu")
                )
            else:
                B = 1
                device = torch.device("cpu")

            return torch.randn(B, self.target_seq_len, self.emb_dim, device=device)

        # ---------------------------------------------------------
        # REAL MODE: wav2vec2 feature extraction
        # ---------------------------------------------------------

        # 1) Convert audio to list of numpy arrays (HF requirement)
        if isinstance(audio, torch.Tensor):
            # audio: (B, L)
            audio_list = [a.cpu().numpy() for a in audio]
        else:
            # audio: list of tensors
            audio_list = [a.cpu().numpy() for a in audio]

        # 2) Preprocess with Wav2Vec2Processor
        inputs = self.processor(
            audio_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        device = next(self.backbone.parameters()).device
        input_values = inputs.input_values.to(device)

        # 3) Optional attention mask
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 4) Wav2Vec2 forward
        with torch.set_grad_enabled(self.backbone.training):
            if attention_mask is not None:
                outputs = self.backbone(input_values, attention_mask=attention_mask)
            else:
                outputs = self.backbone(input_values)

            hidden_states = outputs.last_hidden_state  # (B, T_raw, 768)

        # 5) Project to emb_dim
        emb = self.proj(hidden_states)  # (B, T_raw, emb_dim)

        # ---------------------------------------------------------
        # 6) TEMPORAL ALIGNMENT: interpolate to fixed T
        # ---------------------------------------------------------
        B, T_raw, D = emb.shape

        if T_raw != self.target_seq_len:
            emb = emb.permute(0, 2, 1)  # (B, emb_dim, T_raw)
            emb = torch.nn.functional.interpolate(
                emb,
                size=self.target_seq_len,
                mode="linear",
                align_corners=False,
            )
            emb = emb.permute(0, 2, 1)  # (B, T, emb_dim)

        return emb
