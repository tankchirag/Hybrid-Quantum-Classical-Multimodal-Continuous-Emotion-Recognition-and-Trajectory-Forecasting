# datasets/unified_multimodal_dataset.py

import torch
from torch.utils.data import Dataset


class UnifiedMultimodalDataset(Dataset):
    """
    UnifiedMultimodalDataset
    ------------------------
    A single dataset class supporting both SEWA and RECOLA.

    This class currently uses DUMMY DATA (placeholder tensors).
    Real loading will be added later when dataset access is available.

    Features:
        - dataset_name switch ("recola" or "sewa")
        - modality flags (audio/video/ecg/eda/text)
        - unified output format for the model
        - consistent raw-signal interface for all baselines + QSAL
        - ready for real data integration

    Output format (raw signals):
        {
            "audio": Tensor(L_audio),
            "video": Tensor(T, 3, H, W),
            "ecg":   Tensor(L_ecg) or None,
            "eda":   Tensor(L_eda) or None,
            "text":  str or None,
            "va":    Tensor(T, 2)
        }
    """

    def __init__(
        self,
        dataset_name: str,
        root_dir: str = "",
        target_seq_len: int = 50,
        forecast_horizon: int = 10,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Dataset selection
        # ---------------------------------------------------------
        assert dataset_name in ["recola", "sewa"], \
            "dataset_name must be 'recola' or 'sewa'"

        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.target_seq_len = target_seq_len
        self.forecast_horizon = forecast_horizon

        # ---------------------------------------------------------
        # Modality flags based on dataset
        # ---------------------------------------------------------
        if dataset_name == "recola":
            self.has_audio = True
            self.has_video = True
            self.has_ecg = True
            self.has_eda = True
            self.has_text = False   # RECOLA has no text

        elif dataset_name == "sewa":
            self.has_audio = True
            self.has_video = True
            self.has_ecg = False
            self.has_eda = False
            self.has_text = True    # SEWA has text

        # ---------------------------------------------------------
        # Placeholder dataset size
        # ---------------------------------------------------------
        self.num_samples = 100

    # -------------------------------------------------------------
    # Required Dataset Methods
    # -------------------------------------------------------------
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a dictionary of raw signals for one sample.
        Currently returns dummy tensors for architecture validation.
        """

        sample = {}

        # ---------------------------------------------------------
        # AUDIO (dummy waveform)
        # ---------------------------------------------------------
        if self.has_audio:
            L_audio = 16000 * 4  # 4 seconds @ 16 kHz
            sample["audio"] = torch.randn(L_audio)

        # ---------------------------------------------------------
        # VIDEO (dummy frames)
        # ---------------------------------------------------------
        if self.has_video:
            T = self.target_seq_len
            sample["video"] = torch.randn(T, 3, 112, 112)

        # ---------------------------------------------------------
        # ECG (dummy or None)
        # ---------------------------------------------------------
        sample["ecg"] = torch.randn(5000) if self.has_ecg else None

        # ---------------------------------------------------------
        # EDA (dummy or None)
        # ---------------------------------------------------------
        sample["eda"] = torch.randn(5000) if self.has_eda else None

        # ---------------------------------------------------------
        # TEXT (dummy or None)
        # ---------------------------------------------------------
        sample["text"] = "hello world" if self.has_text else None

        # ---------------------------------------------------------
        # VA LABELS (dummy)
        # ---------------------------------------------------------
        T = self.target_seq_len
        sample["va"] = torch.randn(T, 2)

        return sample
