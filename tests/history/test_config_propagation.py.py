"""
tests.history.test_config_propagation
=====================================

Verify that history configuration values
propagate correctly through the framework.

Checks
------

History Encoder

    ratio
    resolved_dim
    num_layers
    dropout

Dataset

    discrete cardinalities

Forecasting

    horizon

Model

    transformer_dim

Purpose
-------

Catch configuration wiring bugs before
long training runs.
"""

from pathlib import Path
import sys

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from utils.config import (
    load_config,
)

from training.base_training_script import (
    to_namespace,
)

from utils.history_config import (
    resolve_history_dim,
)


def main():

    config = load_config(
        "configs/qsal.yaml"
    )

    config = to_namespace(
        config
    )

    print("\n")
    print("=" * 70)
    print("CONFIG PROPAGATION")
    print("=" * 70)

    # ==================================================
    # Transformer
    # ==================================================

    transformer_dim = (
        config.model.transformer_dim
    )

    print(
        f"Transformer Dim : "
        f"{transformer_dim}"
    )

    # ==================================================
    # History Encoder
    # ==================================================

    history_cfg = (
        config.history_encoder
    )

    resolved_dim = (
        resolve_history_dim(
            config
        )
    )

    print(
        f"History Enabled : "
        f"{history_cfg.enabled}"
    )

    print(
        f"Auto Scale      : "
        f"{history_cfg.auto_scale}"
    )

    print(
        f"Ratio           : "
        f"{history_cfg.ratio}"
    )

    print(
        f"Resolved Dim    : "
        f"{resolved_dim}"
    )

    print(
        f"Num Layers      : "
        f"{history_cfg.num_layers}"
    )

    print(
        f"History Dropout : "
        f"{history_cfg.history_dropout}"
    )

    print(
        f"Injection Point : "
        f"{history_cfg.injection_point}"
    )

    # ==================================================
    # Forecasting
    # ==================================================

    print(
        f"Forecast Horizon : "
        f"{config.forecasting.horizon}"
    )

    # ==================================================
    # Discrete Cardinalities
    # ==================================================

    print("\n")
    print("Discrete Cardinalities")

    cardinalities_cfg = (
        config.dataset
        .discrete_cardinalities
    )

    cardinalities = vars(
        cardinalities_cfg
    )

    for (
        label,
        value,
    ) in cardinalities.items():

        print(
            f"{label:<20}"
            f"{value}"
        )

    # ==================================================
    # Validation
    # ==================================================

    assert (
        resolved_dim
        ==
        transformer_dim
        //
        history_cfg.ratio
    )

    assert (
        config.forecasting.horizon
        > 0
    )

    assert (
        len(cardinalities)
        > 0
    )

    print("\n")
    print(
        "CONFIG PROPAGATION PASS"
    )


if __name__ == "__main__":
    main()