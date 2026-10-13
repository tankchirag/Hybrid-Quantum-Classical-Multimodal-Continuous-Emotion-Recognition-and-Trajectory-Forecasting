"""
training.train_qsal
===================

QSAL Training Entry Point

QSAL
----

Quantum Self-Attention Learning

Main Thesis Model

Run

    python -m training.train_qsal
"""

from __future__ import annotations

from utils.config import load_config

from models.wrappers.qsal import (
    QSAL,
)

from training.base_training_script import (
    run_training,
)


def main() -> None:

    config = load_config(
        "configs/qsal.yaml"
    )

    print(
        "\n[INFO] QSAL"
    )

    print(
        "[INFO] Thesis Model\n"
    )

    run_training(

        config=config,

        model_class=
        QSAL,

        run_dir_name=
        "runs/qsal",
    )


if __name__ == "__main__":
    main()