"""
training.train_baseline_b1
==========================

Baseline B1 Training Entry Point

Run

    python -m training.train_baseline_b1
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from utils.config import load_config

from models.wrappers.baseline_b1 import (
    BaselineB1,
)

from training.base_training_script import (
    run_training,
)


def main():

    config = load_config(
        "configs/baseline_b1.yaml"
    )

    run_training(

        config=config,

        model_class=BaselineB1,

        run_dir_name=
        "runs/baseline_b1",
    )


if __name__ == "__main__":
    main()