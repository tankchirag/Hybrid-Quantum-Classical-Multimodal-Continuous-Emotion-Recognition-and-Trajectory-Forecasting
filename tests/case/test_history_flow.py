from torch.utils.data import DataLoader

from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)

from datasets.loaders.collate import (
    collate,
)

from utils.config import load_config

from models.wrappers.baseline_b1 import (
    BaselineB1,
)


def main():

    config = load_config(
        "configs/case_b1.yaml"
    )

    dataset = CaseTrainingDataset(
        dataset_root=
        config["dataset"]["root_dir"],
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate,
    )

    batch = next(iter(loader))

    print()
    print("=" * 80)
    print("CONTINUOUS LABELS")
    print("=" * 80)

    print(
        batch[
            "continuous_labels"
        ].keys()
    )

    print()

    print(
        batch[
            "continuous_labels"
        ][
            "valence"
        ].shape
    )

    print(
        batch[
            "continuous_labels"
        ][
            "arousal"
        ].shape
    )

    model = BaselineB1(
        config
    )

    outputs = model(
        batch
    )

    print()
    print("=" * 80)
    print("MODEL OUTPUT")
    print("=" * 80)

    print(
        "history enabled:",
        outputs[
            "history_enabled"
        ]
    )

    print(
        "history state is None:",
        outputs[
            "history_state"
        ] is None
    )


if __name__ == "__main__":
    main()