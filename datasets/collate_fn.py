# datasets/collate_fn.py

import torch

def multimodal_collate_fn(batch):
    """
    Collate function for UnifiedMultimodalDataset.
    Converts list of samples → batch dictionary.
    """

    # AUDIO
    audio = torch.stack([item["audio"] for item in batch], dim=0)

    # VIDEO
    video = torch.stack([item["video"] for item in batch], dim=0)

    # ECG
    ecg = [
        item["ecg"] if item["ecg"] is not None else torch.zeros(5000)
        for item in batch
    ]
    ecg = torch.stack(ecg, dim=0)

    # EDA
    eda = [
        item["eda"] if item["eda"] is not None else torch.zeros(5000)
        for item in batch
    ]
    eda = torch.stack(eda, dim=0)

    # TEXT
    text = [item["text"] for item in batch]

    # VA
    va = torch.stack([item["va"] for item in batch], dim=0)

    return {
        "audio": audio,
        "video": video,
        "ecg": ecg,
        "eda": eda,
        "text": text,
        "va": va,
    }
