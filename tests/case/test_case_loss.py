from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)

from datasets.loaders.case_loader import (
    CaseDataset,
)

from models.wrappers.baseline_b1 import (
    BaselineB1,
)

from utils.config import (
    load_config,
)

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def main():

    cfg = load_config(
        "configs/case_b1.yaml"
    )

    model = BaselineB1(cfg)

    model.train()

    dataset = CaseDataset(
        DATASET_ROOT,
        split="train",
    )

    sample = dataset[0]

    X = torch.tensor(
        sample["X"],
        dtype=torch.float32,
    )

    y = torch.tensor(
        sample["y"],
        dtype=torch.float32,
    )

    modalities = {

        "ecg": X[:, :, 0:1],
        "bvp": X[:, :, 1:2],
        "gsr": X[:, :, 2:3],
        "rsp": X[:, :, 3:4],
        "skt": X[:, :, 4:5],
        "emg_zygo": X[:, :, 5:6],
        "emg_coru": X[:, :, 6:7],
        "emg_trap": X[:, :, 7:8],
    }

    outputs = model(
        {
            "modalities": modalities
        }
    )

    forecast = outputs[
        "forecasting"
    ]["continuous"]

    pred_valence = forecast[
        "valence"
    ]

    pred_arousal = forecast[
        "arousal"
    ]

    target_valence = y[:, :, 0]

    target_arousal = y[:, :, 1]

    loss_valence = F.mse_loss(
        pred_valence,
        target_valence,
    )

    loss_arousal = F.mse_loss(
        pred_arousal,
        target_arousal,
    )

    loss = (
        loss_valence
        +
        loss_arousal
    )

    print()
    print("VALENCE LOSS")
    print(
        loss_valence.item()
    )

    print()
    print("AROUSAL LOSS")
    print(
        loss_arousal.item()
    )

    print()
    print("TOTAL LOSS")
    print(
        loss.item()
    )

    loss.backward()

    print()
    print(
        "✅ REAL CASE LOSS PASSED"
    )


if __name__ == "__main__":
    main()