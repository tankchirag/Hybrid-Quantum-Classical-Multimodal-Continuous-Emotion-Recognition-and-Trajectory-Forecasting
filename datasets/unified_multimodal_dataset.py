# datasets/unified_multimodal_dataset.py
"""
UnifiedMultimodalDataset
------------------------

A unified multimodal dataset wrapper for RECOLA and SEWA that supports:

    • REAL mode  (dataset_name in {"recola", "sewa"}, synthetic=False)
        - Returns *raw* signals for all available modalities
        - Returns *raw text strings* for SEWA
        - Downstream encoders (wav2vec2, ResNet18, 1D CNNs, DistilBERT)
          perform the actual feature extraction

    • SYNTHETIC mode (dataset_name == "synthetic", synthetic=True)
        - Returns random float32 tensors with correct shapes
        - Used for QSAL debugging and architecture validation
        - Still respects modality flags and temporal dimensions

This design is fully compatible with:

    • Baseline B1 / B2 / B3
    • QSAL (Quantum Self‑Attention Layer)
    • MultimodalFeatureExtractors
    • ModalityTokenizer / QSALFusion
    • PyTorch DataLoader (no None values in synthetic mode)

Key design rules
----------------
1) REAL mode:
       - The dataset returns *raw* data, not embeddings.
       - Encoders are responsible for converting raw signals/text
         into (B, T, emb_dim) feature sequences.
       - This matches the thesis exposé:
             raw → encoder → tokenizer → feature extractor → fusion.

2) SYNTHETIC mode:
       - The dataset returns random float32 tensors with the same
         shapes as the encoder outputs would have.
       - This allows QSAL and baselines to be debugged without
         requiring real RECOLA/SEWA files.
       - No modality returns None; disabled modalities return zeros.

3) Modality flags:
       - For RECOLA:
             audio, video, ecg, eda = True
             text = False
       - For SEWA:
             audio, video, text = True
             ecg, eda = False
       - For synthetic:
             all four core modalities (audio, video, ecg, eda) enabled
             text optional via config.

You will later replace the REAL mode stubs with actual RECOLA/SEWA
loading and preprocessing.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional


class UnifiedMultimodalDataset(Dataset):
    """
    Unified multimodal dataset for RECOLA, SEWA, and synthetic debugging.

    Parameters
    ----------
    dataset_name : str
        One of:
            - "recola"   : real RECOLA dataset (no text)
            - "sewa"     : real SEWA dataset (with text)
            - "synthetic": synthetic debugging dataset
    root_dir : str
        Path to dataset root (used in REAL mode).
    target_seq_len : int
        Number of frames/time steps (T) for temporal alignment.
    forecast_horizon : int
        Number of future steps (S) for forecasting (not used here,
        but kept for consistency with configs).
    synthetic : bool
        If True, operate in synthetic mode (numeric tensors).
        If False, operate in real mode (raw signals/text).
    num_samples : int
        Dataset size (used in synthetic mode; in real mode you will
        typically derive this from the actual dataset).
    """

    def __init__(
        self,
        dataset_name: str,
        root_dir: str = "",
        target_seq_len: int = 50,
        forecast_horizon: int = 10,
        synthetic: bool = False,
        num_samples: int = 100,
    ) -> None:
        super().__init__()

        dataset_name = dataset_name.lower()
        assert dataset_name in ["recola", "sewa", "synthetic"], \
            "dataset_name must be 'recola', 'sewa', or 'synthetic'"

        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.target_seq_len = target_seq_len
        self.forecast_horizon = forecast_horizon
        self.synthetic = synthetic

        # ---------------------------------------------------------
        # Modality flags (truth table)
        # ---------------------------------------------------------
        if dataset_name == "recola":
            # RECOLA: audio, video, ECG, EDA; no text
            self.modalities = {
                "audio": True,
                "video": True,
                "ecg":   True,
                "eda":   True,
                "text":  False,
            }

        elif dataset_name == "sewa":
            # SEWA: audio, video, text; no ECG/EDA
            self.modalities = {
                "audio": True,
                "video": True,
                "ecg":   False,
                "eda":   False,
                "text":  True,
            }

        elif dataset_name == "synthetic":
            # Synthetic: all four core modalities enabled; text optional
            self.modalities = {
                "audio": True,
                "video": True,
                "ecg":   True,
                "eda":   True,
                "text":  True,   # can be toggled via model config
            }

        # ---------------------------------------------------------
        # Dataset size
        # ---------------------------------------------------------
        if dataset_name == "synthetic":
            # Synthetic: user‑defined size
            self.num_samples = num_samples
        else:
            # REAL mode: placeholder; replace with actual dataset length
            # e.g., self.num_samples = len(recola_file_list)
            self.num_samples = num_samples  # temporary stub

    # -------------------------------------------------------------
    # Required Dataset Methods
    # -------------------------------------------------------------
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        In REAL mode, you will later set this to the actual number
        of RECOLA/SEWA samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary of raw signals for one sample.

        REAL mode:
            - audio : raw waveform tensor (L_audio,)
            - video : raw frame tensor (T, 3, H, W)
            - ecg   : raw ECG waveform (L_ecg,) or zeros if disabled
            - eda   : raw EDA waveform (L_eda,) or zeros if disabled
            - text  : raw string (SEWA) or None (RECOLA)
            - va    : raw V/A labels (T, 2) or dummy in synthetic mode

        SYNTHETIC mode:
            - audio : random float32 tensor (L_audio,)
            - video : random float32 tensor (T, 3, 112, 112)
            - ecg   : random float32 tensor (L_ecg,)
            - eda   : random float32 tensor (L_eda,)
            - text  : random float32 tensor (T, D_text) or zeros
            - va    : random float32 tensor (T, 2)

        All tensors are float32 to avoid dtype mismatch in fusion/QSAL.
        """

        if self.synthetic:
            return self._get_synthetic_sample(idx)
        else:
            return self._get_real_sample(idx)

    # -------------------------------------------------------------
    # SYNTHETIC MODE: numeric tensors for all modalities
    # -------------------------------------------------------------
    def _get_synthetic_sample(self, idx: int) -> Dict[str, Any]:
        """
        Synthetic sample generator.

        Used for QSAL debugging and architecture validation.
        Produces random tensors with correct shapes and dtypes.
        """

        sample: Dict[str, Any] = {}

        # Temporal length
        T = self.target_seq_len

        # ---------------------------------------------------------
        # AUDIO (dummy waveform)
        # ---------------------------------------------------------
        L_audio = 16000 * 4  # 4 seconds @ 16 kHz
        if self.modalities["audio"]:
            sample["audio"] = torch.randn(L_audio, dtype=torch.float32)
        else:
            sample["audio"] = torch.zeros(L_audio, dtype=torch.float32)

        # ---------------------------------------------------------
        # VIDEO (dummy frames)
        # ---------------------------------------------------------
        if self.modalities["video"]:
            sample["video"] = torch.randn(T, 3, 112, 112, dtype=torch.float32)
        else:
            sample["video"] = torch.zeros(T, 3, 112, 112, dtype=torch.float32)

        # ---------------------------------------------------------
        # ECG (dummy)
        # ---------------------------------------------------------
        L_ecg = 5000
        if self.modalities["ecg"]:
            sample["ecg"] = torch.randn(L_ecg, dtype=torch.float32)
        else:
            sample["ecg"] = torch.zeros(L_ecg, dtype=torch.float32)

        # ---------------------------------------------------------
        # EDA (dummy)
        # ---------------------------------------------------------
        L_eda = 5000
        if self.modalities["eda"]:
            sample["eda"] = torch.randn(L_eda, dtype=torch.float32)
        else:
            sample["eda"] = torch.zeros(L_eda, dtype=torch.float32)

        # ---------------------------------------------------------
        # TEXT (dummy embedding)
        # In synthetic mode, text is numeric to match QSAL expectations.
        # ---------------------------------------------------------
        D_text = 128
        if self.modalities["text"]:
            sample["text"] = torch.randn(T, D_text, dtype=torch.float32)
        else:
            sample["text"] = torch.zeros(T, D_text, dtype=torch.float32)

        # ---------------------------------------------------------
        # VA LABELS (dummy)
        # ---------------------------------------------------------
        sample["va"] = torch.randn(T, 2, dtype=torch.float32)

        return sample

    # -------------------------------------------------------------
    # REAL MODE: raw signals + raw text strings
    # -------------------------------------------------------------
    def _get_real_sample(self, idx: int) -> Dict[str, Any]:
        """
        Real sample loader.

        This is where you will plug in actual RECOLA/SEWA loading
        and preprocessing. For now, we provide *shape‑correct stubs*
        that represent raw signals and raw text, not embeddings.

        IMPORTANT:
            - audio/video/ecg/eda are raw signals (1D/4D tensors)
            - text is a raw string for SEWA, None for RECOLA
            - encoders (AudioEncoder, VideoEncoder, ECGEncoder, EDAEncoder,
              TextEncoder) will convert these into (B, T, emb_dim).
        """

        sample: Dict[str, Any] = {}

        # Temporal length
        T = self.target_seq_len

        # ---------------------------------------------------------
        # AUDIO (raw waveform stub)
        # Replace with actual RECOLA/SEWA audio loading.
        # ---------------------------------------------------------
        L_audio = 16000 * 4  # 4 seconds @ 16 kHz (example)
        if self.modalities["audio"]:
            # Stub: raw waveform tensor
            sample["audio"] = torch.randn(L_audio, dtype=torch.float32)
        else:
            sample["audio"] = torch.zeros(L_audio, dtype=torch.float32)

        # ---------------------------------------------------------
        # VIDEO (raw frames stub)
        # Replace with actual frame loading (e.g., 25 Hz @ 300 frames).
        # ---------------------------------------------------------
        if self.modalities["video"]:
            sample["video"] = torch.randn(T, 3, 112, 112, dtype=torch.float32)
        else:
            sample["video"] = torch.zeros(T, 3, 112, 112, dtype=torch.float32)

        # ---------------------------------------------------------
        # ECG (raw waveform stub or zeros)
        # ---------------------------------------------------------
        L_ecg = 5000
        if self.modalities["ecg"]:
            sample["ecg"] = torch.randn(L_ecg, dtype=torch.float32)
        else:
            sample["ecg"] = torch.zeros(L_ecg, dtype=torch.float32)

        # ---------------------------------------------------------
        # EDA (raw waveform stub or zeros)
        # ---------------------------------------------------------
        L_eda = 5000
        if self.modalities["eda"]:
            sample["eda"] = torch.randn(L_eda, dtype=torch.float32)
        else:
            sample["eda"] = torch.zeros(L_eda, dtype=torch.float32)

        # ---------------------------------------------------------
        # TEXT (raw string for SEWA, None for RECOLA)
        # ---------------------------------------------------------
        if self.modalities["text"]:
            # SEWA: raw text string (example stub)
            # In practice, you will load the actual transcript here.
            sample["text"] = "dummy sewa transcript for sample {}".format(idx)
        else:
            # RECOLA: no text modality
            sample["text"] = None

        # ---------------------------------------------------------
        # VA LABELS (raw or stub)
        # Replace with actual RECOLA/SEWA V/A labels.
        # ---------------------------------------------------------
        sample["va"] = torch.randn(T, 2, dtype=torch.float32)

        return sample
