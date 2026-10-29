from copy import deepcopy

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


def run_case(
    title,
    cfg,
    dataset,
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate,
        shuffle=False,
    )

    batch = next(iter(loader))

    model = BaselineB1(cfg)

    outputs = model(batch)

    print(
        "Parameters:",
        f"{count_parameters(model):,}"
    )

    print(
        "History Enabled:",
        outputs["history_enabled"]
    )

    if outputs["history_state"] is not None:

        print(
            "History Shape:",
            tuple(
                outputs[
                    "history_state"
                ].shape
            )
        )

    else:

        print(
            "History Shape: None"
        )

    print(
        "Forecast Horizon:",
        batch[
            "forecast_targets"
        ][
            "continuous"
        ][
            "valence"
        ].shape[-1]
    )


def main():

    config = load_config(
        "configs/case_b1.yaml"
    )

    dataset = CaseTrainingDataset(
        dataset_root=
        config["dataset"][
            "root_dir"
        ],
        split="train",
    )

    # ----------------------
    # Baseline
    # ----------------------

    run_case(
        "DEFAULT CONFIG",
        deepcopy(config),
        dataset,
    )

    # ----------------------
    # History OFF
    # ----------------------

    cfg = deepcopy(config)

    cfg[
        "history_encoder"
    ][
        "enabled"
    ] = False

    run_case(
        "HISTORY OFF",
        cfg,
        dataset,
    )

    # ----------------------
    # Ratio 8
    # ----------------------

    cfg = deepcopy(config)

    cfg[
        "history_encoder"
    ][
        "enabled"
    ] = True

    cfg[
        "history_encoder"
    ][
        "ratio"
    ] = 8

    run_case(
        "RATIO 8",
        cfg,
        dataset,
    )

    # ----------------------
    # Smaller Transformer
    # ----------------------

    cfg = deepcopy(config)

    cfg[
        "model"
    ][
        "transformer_dim"
    ] = 128

    run_case(
        "TRANSFORMER 128",
        cfg,
        dataset,
    )


if __name__ == "__main__":
    main()