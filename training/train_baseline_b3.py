"""
training.train_baseline_b3
==========================

Baseline B3 Training Entry Point

Baseline B3
-----------

Parameterized Quantum Circuit Fusion Baseline

Run

    python -m training.train_baseline_b3
"""

from __future__ import annotations

from utils.config import load_config

from models.wrappers.baseline_b3 import (
    BaselineB3,
)

from training.base_training_script import (
    run_training,
)


def main() -> None:

    config = load_config(
        "configs/baseline_b3.yaml"
    )

    print(
        "\n[INFO] Baseline B3"
    )

    print(
        "[INFO] Quantum Fusion Enabled\n"
    )

    run_training(

        config=config,

        model_class=
        BaselineB3,

        run_dir_name=
        "runs/baseline_b3",
    )


if __name__ == "__main__":
    main()