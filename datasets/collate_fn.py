# datasets/collate_fn.py

import torch

def multimodal_collate_fn(batch):
    """
    Collate function for UnifiedMultimodalDataset.
    Converts a list of per-sample dictionaries into a single batch dictionary.

    Handles:
        - Missing modalities (ECG/EDA/Text)
        - Tensor stacking
        - Text batching (None vs List[str])
        - Consistent output format for the model

    Output:
        {
            "audio": Tensor(B, L_audio),
            "video": Tensor(B, T, 3, H, W),
            "ecg":   Tensor(B, L_ecg) or zeros,
            "eda":   Tensor(B, L_eda) or zeros,
            "text":  None or List[str],
            "va":    Tensor(B, T, 2)
        }
    """

    # ---------------------------------------------------------
    # AUDIO (always present)
    # ---------------------------------------------------------
    audio = torch.stack([item["audio"] for item in batch], dim=0)

    # ---------------------------------------------------------
    # VIDEO (always present)
    # ---------------------------------------------------------
    video = torch.stack([item["video"] for item in batch], dim=0)

    # ---------------------------------------------------------
    # ECG (may be None for SEWA)
    # ---------------------------------------------------------
    ecg = [
        item["ecg"] if item["ecg"] is not None else torch.zeros(5000)
        for item in batch
    ]
    ecg = torch.stack(ecg, dim=0)

    # ---------------------------------------------------------
    # EDA (may be None for SEWA)
    # ---------------------------------------------------------
    eda = [
        item["eda"] if item["eda"] is not None else torch.zeros(5000)
        for item in batch
    ]
    eda = torch.stack(eda, dim=0)

    # ---------------------------------------------------------
    # TEXT (None for RECOLA, List[str] for SEWA)
    # ---------------------------------------------------------
    first_text = batch[0]["text"]

    if first_text is None:
        # RECOLA → no text modality
        text = None
    else:
        # SEWA → list of strings
        text = [item["text"] for item in batch]

    # ---------------------------------------------------------
    # VA LABELS (always present)
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
