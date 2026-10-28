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

    model = BaselineB1(
        cfg
    )

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

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

    #
    # Inspect Forecast Outputs
    #

    print()
    print("FORECAST OUTPUTS")

    forecast = outputs[
        "forecasting"
    ]

    print(
        forecast.keys()
    )

    #
    # Temporary loss
    #

    loss = (
        outputs[
            "decoder_outputs"
        ].mean()
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print()

    print(
        "LOSS:",
        float(loss)
    )

    print()

    print(
        "✅ EPOCH TEST PASSED"
    )


if __name__ == "__main__":
    main()