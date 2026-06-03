import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors import MultimodalFeatureExtractors
from models.tokenizers.modality_tokenizer import ModalityTokenizer
from models.fusion.classical_fusion import ClassicalFusion
from models.temporal.transformer_ed import TransformerEncoderDecoder
from models.heads.recognition_head import RecognitionHead
from models.heads.forecasting_head import ForecastingHead


class BaselineB1(nn.Module):
    """
    BaselineB1
    ----------
    Classical multimodal baseline using:

        1. Multimodal feature extractors
        2. Modality tokenizer
        3. Classical self-attention fusion
        4. Transformer Encoder–Decoder (with VA history)
        5. Recognition head (VA_t)
        6. Forecasting head (VA_future)

    This model follows the thesis pipeline:

        Raw signals
            → Feature Extractors (B, T, 128)
            → Modality Tokenizer (B, T, M, D)
            → Classical Fusion (B, T, 256)
            → TransformerEncoderDecoder(fused, va_hist)
            → enc_out (B, T, 256)
            → dec_out (B, S, 256)
            → Heads

    IMPORTANT:
    ----------
    The TransformerEncoderDecoder requires VA history (B, T, 2).
    For Baseline B1 on synthetic data, VA history is provided by the dataset.
    """

    def __init__(
        self,
        emb_dim: int = 128,
        fusion_dim: int = 64,
        transformer_dim: int = 256,
        num_heads: int = 2,
        num_layers: int = 2,
        target_seq_len: int = 50,
        forecast_horizon: int = 10,
        synthetic: bool = True,
        include_text: bool = True,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # 1. Multimodal Feature Extractors
        # ---------------------------------------------------------
        self.extractors = MultimodalFeatureExtractors(
            emb_dim=emb_dim,
            synthetic=synthetic,
            target_seq_len=target_seq_len,
            include_text=include_text,
        )

        # ---------------------------------------------------------
        # 2. Modality Tokenizer
        # ---------------------------------------------------------
        self.tokenizer = ModalityTokenizer(
            emb_dim=emb_dim,
            num_modalities=5 if include_text else 4,
        )

        # ---------------------------------------------------------
        # 3. Classical Fusion
        # ---------------------------------------------------------
        self.fusion = ClassicalFusion(
            emb_dim=emb_dim,
            fusion_dim=fusion_dim,
            transformer_dim=transformer_dim,
            num_heads=num_heads,
        )

        # ---------------------------------------------------------
        # 4. Transformer Encoder–Decoder (with VA history)
        # ---------------------------------------------------------
        self.temporal_model = TransformerEncoderDecoder(
            d_model=transformer_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            forecast_horizon=forecast_horizon,
        )

        # ---------------------------------------------------------
        # 5. Recognition Head
        # ---------------------------------------------------------
        self.recognition_head = RecognitionHead(
            input_dim=transformer_dim,
            output_dim=2,
        )

        # ---------------------------------------------------------
        # 6. Forecasting Head
        # ---------------------------------------------------------
        self.forecasting_head = ForecastingHead(
            input_dim=transformer_dim,
            output_dim=2,
        )


    def forward(self, batch):
        """
        Forward pass for Baseline B1.

        Parameters
        ----------
        batch : dict
            Must contain:
                - raw modality inputs
                - "va_hist": (B, T, 2)

        Returns
        -------
        dict
            {
                "va_pred":   (B, T, 2),
                "va_future": (B, S, 2)
            }
        """

        # ---------------------------------------------------------
        # 1. Extract modality embeddings (B, T, 128)
        # ---------------------------------------------------------
        emb_dict = self.extractors(batch)

        # ---------------------------------------------------------
        # 2. Tokenize modalities → (B, T, M, D)
        # ---------------------------------------------------------
        tokens = self.tokenizer(emb_dict)

        # ---------------------------------------------------------
        # 3. Classical fusion → (B, T, 256)
        # ---------------------------------------------------------
        fused = self.fusion(tokens)

        # ---------------------------------------------------------
        # 4. Temporal modeling with VA history
        # ---------------------------------------------------------
        va_hist = batch["va_hist"]  # (B, T, 2)
        enc_out, dec_out = self.temporal_model(fused, va_hist)

        # ---------------------------------------------------------
        # 5. Recognition head → (B, T, 2)
        # ---------------------------------------------------------
        va_pred = self.recognition_head(enc_out)

        # ---------------------------------------------------------
        # 6. Forecasting head → (B, S, 2)
        # ---------------------------------------------------------
        va_future = self.forecasting_head(dec_out)

        return {
            "va_pred": va_pred,
            "va_future": va_future,
        }
