from torch.utils.data import DataLoader

from utils.config import load_config

from datasets.wrappers.case_training_dataset import (
    CaseTrainingDataset,
)

from datasets.loaders.collate import (
    collate,
)

from models.wrappers.baseline_b1 import (
    BaselineB1,
)


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def main():

    config = load_config(
        "configs/case_b1.yaml"
    )

    print("=" * 80)
    print("CONFIG")
    print("=" * 80)

    print(
        "history enabled:",
        config["history_encoder"][
            "enabled"
        ]
    )

    print(
        "ratio:",
        config["history_encoder"][
            "ratio"
        ]
    )

    print(
        "transformer dim:",
        config["model"][
            "transformer_dim"
        ]
    )

    print(
        "forecast horizon:",
        config["forecasting"][
            "horizon"
        ]
    )

    dataset = (
        CaseTrainingDataset(
            dataset_root=
            config["dataset"][
                "root_dir"
            ],
            split="train",
        )
    )

    sample = dataset[0]

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)

    print(
        "history valence:",
        sample[
            "continuous_labels"
        ][
            "valence"
        ].shape
    )

    print(
        "forecast valence:",
        sample[
            "forecast_targets"
        ][
            "continuous"
        ][
            "valence"
        ].shape
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate,
    )

    batch = next(
        iter(loader)
    )

    print()

    print(
        "batch history:",
        batch[
            "continuous_labels"
        ][
            "valence"
        ].shape
    )

    model = BaselineB1(
        config
    )

    print()
    print("=" * 80)
    print("MODEL")
    print("=" * 80)

    print(
        "params:",
        count_parameters(
            model
        )
    )

    outputs = model(
        batch
    )

    print()

    print(
        "history enabled:",
        outputs[
            "history_enabled"
        ]
    )

    if (
        outputs[
            "history_state"
        ]
        is not None
    ):

        print(
            "history state:",
            outputs[
                "history_state"
            ].shape
        )

    else:

        print(
            "history state: None"
        )


if __name__ == "__main__":
    main()