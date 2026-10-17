"""
Change horizon

25 -> 10 -> 40

Verify decoder output shape changes
Verify forecasting output shape changes  

"""


import pathlib
import sys

# ==========================================================
# Project Root
# ==========================================================


PROJECT_ROOT = (
    pathlib.Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from models.history.adapter import (
    HistoryAdapter,
)

from utils.config import load_config
from training.base_training_script import (
    to_namespace,
    build_dataloaders,
)
from models.wrappers.qsal import QSAL
import torch


def run_case(horizon):

    cfg = to_namespace(
        load_config(
            "configs/qsal.yaml"
        )
    )

    cfg.forecasting.horizon = horizon

    train_loader, _ = build_dataloaders(
        cfg,
        torch.device("cpu"),
    )

    batch = next(iter(train_loader))

    model = QSAL(cfg)

    with torch.no_grad():

        outputs = model(batch)

    print(
        f"\nHorizon={horizon}"
    )

    print(
        "decoder:",
        outputs[
            "decoder_outputs"
        ].shape
    )

    print(
        "forecast valence:",
        outputs[
            "forecasting"
        ][
            "continuous"
        ][
            "valence"
        ].shape
    )


run_case(10)
run_case(25)
run_case(40)