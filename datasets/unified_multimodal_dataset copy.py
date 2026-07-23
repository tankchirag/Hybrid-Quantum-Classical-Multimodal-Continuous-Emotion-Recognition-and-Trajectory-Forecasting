# datasets/unified_multimodal_dataset.py

import torch
from torch.utils.data import Dataset


class UnifiedMultimodalDataset(Dataset):
    """
    UnifiedMultimodalDataset
    ------------------------
    A unified dataset wrapper for SEWA and RECOLA.

    This version provides:
        - Automatic modality flags (audio/video/ecg/eda/text)
        - A consistent raw-signal interface for all baselines
        - Dummy tensors for architecture validation
        - Real data can be plugged in later without changing the model

    Output format (raw signals):
        {
            "audio": Tensor(L_audio) or None,
            "video": Tensor(T, 3, H, W) or None,
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

        dataset_name = dataset_name.lower()
        assert dataset_name in ["recola", "sewa"], \
            "dataset_name must be 'recola' or 'sewa'"

        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.target_seq_len = target_seq_len
        self.forecast_horizon = forecast_horizon

        # ---------------------------------------------------------
        # Automatic modality flags (truth table)
        # ---------------------------------------------------------
        if dataset_name == "recola":
            self.modalities = {
                "audio": True,
                "video": True,
                "ecg":   True,
                "eda":   True,
                "text":  False,   # RECOLA has no text
            }

        elif dataset_name == "sewa":
            self.modalities = {
                "audio": True,
                "video": True,
                "ecg":   False,
                "eda":   False,
                "text":  True,    # SEWA has text
            }

        # ---------------------------------------------------------
        # Placeholder dataset size (dummy mode)
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

        # AUDIO
        if self.modalities["audio"]:
            L_audio = 16000 * 4  # 4 seconds @ 16 kHz
            sample["audio"] = torch.randn(L_audio)
        else:
            sample["audio"] = None

        # VIDEO
        if self.modalities["video"]:
            T = self.target_seq_len
            sample["video"] = torch.randn(T, 3, 112, 112)
        else:
            sample["video"] = None

        # ECG
        sample["ecg"] = torch.randn(5000) if self.modalities["ecg"] else None

        # EDA
        sample["eda"] = torch.randn(5000) if self.modalities["eda"] else None

        # TEXT
        sample["text"] = "hello world" if self.modalities["text"] else None

        # VA LABELS
        T = self.target_seq_len
        sample["va"] = torch.randn(T, 2)

        return sample
