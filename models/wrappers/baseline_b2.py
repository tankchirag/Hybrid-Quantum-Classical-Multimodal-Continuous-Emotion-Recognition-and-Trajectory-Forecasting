import torch
import torch.nn as nn

from models.encoders.multimodal_feature_extractors import MultimodalFeatureExtractors
from models.fusion.unified_concat_fusion import UnifiedConcatFusion
from models.temporal.transformer_ed import TransformerEncoderDecoder
from models.heads.recognition_head import RecognitionHead
from models.heads.forecasting_head import ForecastingHead


class BaselineB2(nn.Module):
    """
    BaselineB2
    ----------
    Capacity-controlled multimodal baseline using:

        1. Multimodal feature extractors (shared with all baselines)
        2. UnifiedConcatFusion (per-modality 128→proj_dim projections + concatenation)
        3. Linear projection fused_dim → transformer_dim (256)
        4. Transformer Encoder–Decoder (conditioned on VA history)
        5. Recognition head (frame-wise VA_t)
        6. Forecasting head (future VA trajectory)

    The number of active modalities depends on the dataset:
        - RECOLA: audio, video, ecg, eda
        - SEWA:   audio, video, text
        - Combined: audio, video, ecg, eda, text

    UnifiedConcatFusion computes:
        fused_dim = (#active_modalities) × proj_dim

    We then project:
        fused_dim → transformer_dim (256)
    to keep Transformer capacity identical across baselines.
    """

    def __init__(
        self,
        emb_dim: int = 128,
        proj_dim: int = 64,
        transformer_dim: int = 256,
        num_heads: int = 2,
        num_layers: int = 2,
        target_seq_len: int = 50,
        forecast_horizon: int = 10,
        synthetic: bool = True,
        modalities: dict | None = None,
    ):
        """
        Parameters
        ----------
        emb_dim : int
            Dimensionality of each modality encoder output (default: 128).
        proj_dim : int
            Dimensionality after per-modality projection (default: 64).
        transformer_dim : int
            Hidden size of the shared Transformer (default: 256).
        num_heads : int
            Number of attention heads in the Transformer.
        num_layers : int
            Number of encoder and decoder layers in the Transformer.
        target_seq_len : int
            Number of historical frames used as input.
        forecast_horizon : int
            Number of future frames to forecast.
        synthetic : bool
            Whether to use synthetic dummy data in the feature extractors.
        modalities : dict
            Dictionary of active modalities, e.g.
                {
                    "audio": True/False,
                    "video": True/False,
                    "ecg":   True/False,
                    "eda":   True/False,
                    "text":  True/False,
                }
        """
        super().__init__()

        if modalities is None:
            raise ValueError("BaselineB2: 'modalities' dict must be provided.")

        self.modalities = modalities
        self.target_seq_len = target_seq_len
        self.forecast_horizon = forecast_horizon

        # ---------------------------------------------------------
        # 1. Multimodal Feature Extractors
        # ---------------------------------------------------------
        self.extractors = MultimodalFeatureExtractors(
            emb_dim=emb_dim,
            synthetic=synthetic,
            target_seq_len=target_seq_len,
            include_text=self.modalities["text"],
        )

        # ---------------------------------------------------------
        # 2. Unified Concatenation Fusion
        #    Modality usage is driven by the dataset's modality flags.
        # ---------------------------------------------------------
        self.fusion = UnifiedConcatFusion(
            use_audio=self.modalities["audio"],
            use_video=self.modalities["video"],
            use_ecg=self.modalities["ecg"],
            use_eda=self.modalities["eda"],
            use_text=self.modalities["text"],
            in_dim=emb_dim,
            proj_dim=proj_dim,
            expected_out_dim=None,   # dynamic; no hard check
        )

        # True fused_dim from fusion module (e.g. 192, 256, 320)
        fused_dim = self.fusion.out_dim

        # ---------------------------------------------------------
        # 2.5 Project fused_dim → transformer_dim (256)
        # ---------------------------------------------------------
        self.fused_proj = nn.Linear(fused_dim, transformer_dim)

        # ---------------------------------------------------------
        # 3. Transformer Encoder–Decoder
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
        # 4. Recognition Head
        # ---------------------------------------------------------
        self.recognition_head = RecognitionHead(
            input_dim=transformer_dim,
            output_dim=2,
        )

        # ---------------------------------------------------------
        # 5. Forecasting Head
        # ---------------------------------------------------------
        self.forecasting_head = ForecastingHead(
            input_dim=transformer_dim,
            output_dim=2,
        )

    def forward(self, batch):
        """
        Forward pass for Baseline B2.

        Parameters
        ----------
        batch : dict
            Must contain:
                - raw modality inputs (audio, video, ecg, eda, text/None)
                - "va": Tensor of shape (B, T_va, 2)

        Returns
        -------
        dict
            {
                "va_pred":      (B, T_align, 2),
                "va_future":    (B, S, 2),
                "va_hist":      (B, T_align, 2),
                "va_future_gt": (B, S, 2),
            }
        """

        # ---------------------------------------------------------
        # 0. VA history and future
        # ---------------------------------------------------------
        va_full = batch["va"]          # (B, T_va, 2)
        B, T_va, _ = va_full.shape
        S = self.forecast_horizon

        if S <= T_va:
            va_future_gt = va_full[:, -S:, :]
        else:
            va_future_gt = va_full[:, -1:, :].repeat(1, S, 1)

        # ---------------------------------------------------------
        # 1. Extract modality embeddings
        # ---------------------------------------------------------
        emb_dict = self.extractors(batch)   # each: (B, T, emb_dim) or None

        # ---------------------------------------------------------
        # 2. Fusion → (B, T, fused_dim)
        # ---------------------------------------------------------
        fused_raw = self.fusion(emb_dict)

        # ---------------------------------------------------------
        # 2.5 Project fused_dim → transformer_dim (256)
        # ---------------------------------------------------------
        fused = self.fused_proj(fused_raw)  # (B, T, transformer_dim)

        # ---------------------------------------------------------
        # 3. Temporal alignment
        # ---------------------------------------------------------
        T_fused = fused.size(1)
        T_align = min(T_fused, T_va)

        fused_aligned = fused[:, -T_align:, :]   # (B, T_align, d)
        va_hist = va_full[:, -T_align:, :]       # (B, T_align, 2)

        # ---------------------------------------------------------
        # 4. Transformer
        # ---------------------------------------------------------
        enc_out, dec_out = self.temporal_model(fused_aligned, va_hist)

        # ---------------------------------------------------------
        # 5. Recognition + Forecasting
        # ---------------------------------------------------------
        va_pred = self.recognition_head(enc_out)       # (B, T_align, 2)
        va_future_pred = self.forecasting_head(dec_out)  # (B, S, 2)

        return {
            "va_pred": va_pred,
            "va_future": va_future_pred,
            "va_hist": va_hist,
            "va_future_gt": va_future_gt,
        }
