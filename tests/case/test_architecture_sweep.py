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

    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate,
        shuffle=False,
    )

    batch = next(iter(loader))

    model = BaselineB1(cfg)

    outputs = model(batch)

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        "params:",
        f"{count_parameters(model):,}"
    )

    print(
        "transformer_dim:",
        model.transformer_dim
    )

    print(
        "history_enabled:",
        outputs[
            "history_enabled"
        ]
    )

    if outputs[
        "history_state"
    ] is not None:

        print(
            "history_state:",
            tuple(
                outputs[
                    "history_state"
                ].shape
            )
        )

    else:

        print(
            "history_state: None"
        )

    print(
        "tokens:",
        tuple(
            outputs[
                "tokens"
            ].shape
        )
    )

    print(
        "fused:",
        tuple(
            outputs[
                "fused"
            ].shape
        )
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

    # ==================================================
    # DEFAULT
    # ==================================================

    run_case(
        "DEFAULT",
        deepcopy(config),
        dataset,
    )

    # ==================================================
    # TRANSFORMER 128
    # ==================================================

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

    # ==================================================
    # TRANSFORMER 64
    # ==================================================

    cfg = deepcopy(config)

    cfg[
        "model"
    ][
        "transformer_dim"
    ] = 64

    run_case(
        "TRANSFORMER 64",
        cfg,
        dataset,
    )

    # ==================================================
    # HISTORY OFF
    # ==================================================

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

    # ==================================================
    # SINGLE MODALITY
    # ==================================================

    cfg = deepcopy(config)

    cfg[
        "modalities"
    ] = {

        "ecg": True,

        "bvp": False,
        "gsr": False,
        "rsp": False,
        "skt": False,

        "emg_zygo": False,
        "emg_coru": False,
        "emg_trap": False,
    }

    run_case(
        "ONLY ECG",
        cfg,
        dataset,
    )


if __name__ == "__main__":
    main()