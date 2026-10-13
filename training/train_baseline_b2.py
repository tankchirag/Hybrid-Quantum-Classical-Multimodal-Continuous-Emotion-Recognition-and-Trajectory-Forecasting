"""
training.train_baseline_b2
==========================

Baseline B2 Training Entry Point

Baseline B2
-----------

Concatenation Fusion Baseline

Run

    python -m training.train_baseline_b2
"""

from __future__ import annotations

from utils.config import load_config

from models.wrappers.baseline_b2 import (
    BaselineB2,
)

from training.base_training_script import (
    run_training,
)


def main() -> None:

    config = load_config(
        "configs/baseline_b2.yaml"
    )

    run_training(

        config=config,

        model_class=
        BaselineB2,

        run_dir_name=
        "runs/baseline_b2",
    )


if __name__ == "__main__":
    main()