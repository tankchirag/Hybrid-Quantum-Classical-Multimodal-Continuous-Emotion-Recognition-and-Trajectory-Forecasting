"""
training.train_case_b2
======================
CASE training entrypoint

Run

python -m training.train_case_b2
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

from models.wrappers.baseline_b2 import (
    BaselineB2,
)

from training.base_training_script import (
    run_training,
)


def main():

    config = load_config(
        "configs/case_b2.yaml"
    )

    run_training(

        config=config,

        model_class=
        BaselineB2,

        run_dir_name=
        "runs/case_b2",
    )


if __name__ == "__main__":
    main()