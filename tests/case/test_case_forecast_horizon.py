from pathlib import Path
import sys
import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)

from utils.config import load_config
from models.wrappers.baseline_b1 import BaselineB1
from datasets.loaders.case_loader import CaseDataset

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def main():

    cfg = load_config(
        "configs/case_b1.yaml"
    )

    model = BaselineB1(cfg)

    model.eval()

    ds = CaseDataset(
        DATASET_ROOT,
        split="train",
    )

    sample = ds[0]

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

    print()
    print("TARGET")
    print(
        tuple(y.shape)
    )

    print()
    print("DECODER")
    print(
        tuple(
            outputs[
                "decoder_outputs"
            ].shape
        )
    )

    forecast = outputs[
        "forecasting"
    ]["continuous"]

    print()
    print("VALENCE")
    print(
        tuple(
            forecast[
                "valence"
            ].shape
        )
    )

    print()
    print("AROUSAL")
    print(
        tuple(
            forecast[
                "arousal"
            ].shape
        )
    )

    print()
    print(
        "✅ HORIZON TEST PASSED"
    )


if __name__ == "__main__":
    main()