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

from models.encoders.multimodal_feature_extractors_v2 import (
    MultimodalFeatureExtractorsV2,
)

from models.tokenizers.tokenizer import (
    Tokenizer,
)

DATASET_ROOT = Path(
    r"D:\Datasets\CASE\processed"
)


def main():

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

    extractor = (
        MultimodalFeatureExtractorsV2(
            embedding_dim=128,

            modalities={
                k: True
                for k
                in modalities
            },
        )
    )

    embeddings = extractor(
        modalities
    )

    print()
    print("ENCODER OUTPUTS")

    for k, v in embeddings.items():

        print(
            k,
            tuple(v.shape),
        )

    tokenizer = Tokenizer()

    tokens = tokenizer(
        embeddings
    )

    print()

    print(
        "TOKENS:",
        tuple(tokens.shape)
    )


if __name__ == "__main__":
    main()