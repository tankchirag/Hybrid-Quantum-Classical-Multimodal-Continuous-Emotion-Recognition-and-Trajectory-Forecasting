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
    print("=" * 60)
    print("RECOGNITION")
    print("=" * 60)

    for k, v in outputs[
        "recognition"
    ].items():

        print(
            k,
            type(v).__name__
        )

        if torch.is_tensor(v):

            print(
                k,
                tuple(v.shape)
            )

        elif isinstance(v, dict):

            print()
            print(k)

            for kk, vv in v.items():

                if torch.is_tensor(vv):

                    print(
                        "   ",
                        kk,
                        tuple(vv.shape)
                    )

                else:

                    print(
                        "   ",
                        kk,
                        type(vv).__name__
                    )

        else:

            print(
                k,
                type(v).__name__
            )

    print()
    print("=" * 60)
    print("FORECASTING")
    print("=" * 60)

    for k, v in outputs[
        "forecasting"
    ].items():

        print(
            k,
            type(v).__name__
        )

    print()
    print(
        "✅ HEAD TEST PASSED"
    )


if __name__ == "__main__":
    main()