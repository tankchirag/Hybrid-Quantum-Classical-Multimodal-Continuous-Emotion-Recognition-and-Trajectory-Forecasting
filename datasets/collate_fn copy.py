# datasets/collate_fn.py
"""
multimodal_collate_fn
---------------------

Unified collate function supporting BOTH:

    • REAL mode (RECOLA / SEWA)
    • SYNTHETIC mode (QSAL debugging)

This function converts a list of per-sample dictionaries into a single
batch dictionary with modality-consistent shapes and types.

Unified Modality Contract
-------------------------
REAL mode:
    audio : Tensor(B, L_audio)
    video : Tensor(B, T, 3, H, W)
    ecg   : Tensor(B, L_ecg)
    eda   : Tensor(B, L_eda)
    text  : List[str]      (SEWA)
            None           (RECOLA)
    va    : Tensor(B, T, 2)

SYNTHETIC mode:
    audio : Tensor(B, L_audio)
    video : Tensor(B, T, 3, H, W)
    ecg   : Tensor(B, L_ecg)
    eda   : Tensor(B, L_eda)
    text  : Tensor(B, T, D_text)
    va    : Tensor(B, T, 2)

This ensures:
    • Baseline B1/B2/B3 receive raw text strings → DistilBERT works
    • QSAL receives numeric tensors → synthetic debugging works
    • No None values in synthetic mode → fusion/QSAL stable
    • All encoders receive correct raw inputs → scientific pipeline preserved
"""

import torch
from typing import List, Dict, Any


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for UnifiedMultimodalDataset.

    Parameters
    ----------
    batch : list of dict
        Each dict contains raw modality data for one sample.

    Returns
    -------
    dict
        {
            "audio": Tensor(B, L_audio)
            "video": Tensor(B, T, 3, H, W)
            "ecg":   Tensor(B, L_ecg)
            "eda":   Tensor(B, L_eda)
            "text":  List[str] or Tensor(B, T, D_text) or None
            "va":    Tensor(B, T, 2)
        }
    """

    # ---------------------------------------------------------
    # AUDIO (always tensor)
    # ---------------------------------------------------------
    audio = torch.stack([item["audio"] for item in batch], dim=0)

    # ---------------------------------------------------------
    # VIDEO (always tensor)
    # ---------------------------------------------------------
    video = torch.stack([item["video"] for item in batch], dim=0)

    # ---------------------------------------------------------
    # ECG (tensor or zeros)
    # ---------------------------------------------------------
    ecg = torch.stack([item["ecg"] for item in batch], dim=0)

    # ---------------------------------------------------------
    # EDA (tensor or zeros)
    # ---------------------------------------------------------
    eda = torch.stack([item["eda"] for item in batch], dim=0)

    # ---------------------------------------------------------
    # TEXT (REAL vs SYNTHETIC)
    # ---------------------------------------------------------
    first_text = batch[0]["text"]

    # REAL MODE (SEWA → raw strings, RECOLA → None)
    if isinstance(first_text, str) or first_text is None:
        # SEWA: List[str]
        # RECOLA: None
        text = [item["text"] for item in batch]

    # SYNTHETIC MODE (numeric tensor)
    else:
        # item["text"] is Tensor(T, D_text)
        text = torch.stack([item["text"] for item in batch], dim=0)

    # ---------------------------------------------------------
    # VA LABELS (always tensor)
    # ---------------------------------------------------------
    va = torch.stack([item["va"] for item in batch], dim=0)

    return {
        "audio": audio,
        "video": video,
        "ecg": ecg,
        "eda": eda,
        "text": text,
        "va": va,
    }
