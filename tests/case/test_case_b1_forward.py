"""
tests/case/test_case_b1_forward.py
==================================

End-to-end CASE → B1 smoke test.

Verifies:

CASE
 ↓
Encoders
 ↓
Tokenizer
 ↓
Fusion
 ↓
Transformer
 ↓
Recognition
 ↓
Forecasting
"""

from pathlib import Path
import sys

import torch

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

from pathlib import Path

print()
print("CONFIG SEARCH")
print("-" * 40)

for p in Path("configs").rglob("*.yaml"):
    print(p)


DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def main():

    print()
    print("=" * 60)
    print("CASE B1 FORWARD TEST")
    print("=" * 60)

    #
    # Config
    #
    cfg = load_config(
        "configs/baseline_b1.yaml"
    )

    #
    # Model
    #
    model = BaselineB1(cfg)

    model.eval()

    #
    # Data
    #
    dataset = CaseDataset(
        DATASET_ROOT,
        split="train",
    )

    sample = dataset[0]

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

    batch = {
        "modalities": modalities,
    }

    #
    # Forward
    #
    with torch.no_grad():

        outputs = model(
            batch
        )

    print()

    print("OUTPUTS")
    print("-" * 60)

    for k, v in outputs.items():

        if torch.is_tensor(v):

            print(
                k,
                tuple(v.shape),
            )

        elif isinstance(v, dict):

            print(
                k,
                "dict",
            )

        else:

            print(
                k,
                type(v).__name__,
            )

    print()
    print("✅ B1 FORWARD PASSED")


if __name__ == "__main__":
    main()