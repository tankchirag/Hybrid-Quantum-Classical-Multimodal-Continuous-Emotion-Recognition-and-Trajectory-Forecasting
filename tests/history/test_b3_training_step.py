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


from utils.config import load_config

from training.base_training_script import (
    to_namespace,
    build_dataloaders,
)

from models.wrappers.baseline_b3 import (
    BaselineB3,
)

from training.losses import (
    UnifiedMultiTaskLoss,
)

import torch


cfg = to_namespace(
    load_config(
        "configs/baseline_b3.yaml"
    )
)

train_loader, _ = build_dataloaders(
    cfg,
    torch.device("cpu")
)

batch = next(
    iter(train_loader)
)

model = BaselineB3(cfg)

outputs = model(batch)

loss_fn = (
    UnifiedMultiTaskLoss()
)

losses = loss_fn(

    outputs=outputs,

    recognition_targets=
    batch[
        "recognition_targets"
    ],

    forecast_targets=
    batch[
        "forecast_targets"
    ],
)

print(
    losses["total_loss"]
)

print(
    "\nB3 TRAINING STEP OK"
)