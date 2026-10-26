"""
Tests for DatasetConfig.

Purpose
-------
Validate that dataset YAML files are correctly
loaded and exposed through the strongly-typed
DatasetConfig abstraction.

This test is intentionally independent from the
CASE preprocessing pipeline.

Validation
----------
1. YAML loading
2. Typed config creation
3. Feature extraction
4. Target extraction
5. History settings
6. Forecast settings
7. Harmonization settings
8. Normalization settings
"""

from pathlib import Path

from configs.dataset_config import (
    DatasetConfig,
)


# ==========================================================
# Configuration
# ==========================================================

CONFIG_PATH = Path(
    "configs/datasets/case.yaml"
)


# ==========================================================
# Test
# ==========================================================

def main() -> None:

    print(
        "\n========================================"
    )

    print(
        "DATASET CONFIG TEST"
    )

    print(
        "========================================"
    )

    # ------------------------------------------------------
    # Load
    # ------------------------------------------------------

    config = DatasetConfig.load(
        CONFIG_PATH
    )

    print(
        "\n✅ Config loaded"
    )

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    assert (
        config.dataset.name
        == "case"
    )

    print(
        "✅ Dataset name validated"
    )

    # ------------------------------------------------------
    # Features
    # ------------------------------------------------------

    expected_features = [
        "ecg",
        "bvp",
        "gsr",
        "rsp",
        "skt",
        "emg_zygo",
        "emg_coru",
        "emg_trap",
    ]

    assert (
        config.feature_columns
        == expected_features
    )

    print(
        "✅ Features validated"
    )

    # ------------------------------------------------------
    # Targets
    # ------------------------------------------------------

    expected_targets = [
        "valence",
        "arousal",
    ]

    assert (
        config.target_columns
        == expected_targets
    )

    print(
        "✅ Targets validated"
    )

    # ------------------------------------------------------
    # Sampling
    # ------------------------------------------------------

    assert (
        config.sampling_rate_hz
        == 2
    )

    print(
        "✅ Sampling rate validated"
    )

    # ------------------------------------------------------
    # History
    # ------------------------------------------------------

    assert (
        config.history.enabled
        is True
    )

    assert (
        config.history_steps
        == 120
    )

    print(
        "✅ History configuration validated"
    )

    # ------------------------------------------------------
    # Forecast
    # ------------------------------------------------------

    assert (
        config.forecasting.enabled
        is True
    )

    assert (
        config.forecast_steps
        == 20
    )

    print(
        "✅ Forecast configuration validated"
    )

    # ------------------------------------------------------
    # Normalization
    # ------------------------------------------------------

    assert (
        config.normalization.enabled
        is True
    )

    assert (
        config.normalization.strategy
        == "zscore"
    )

    print(
        "✅ Normalization validated"
    )

    # ------------------------------------------------------
    # Harmonization
    # ------------------------------------------------------

    assert (
        config.harmonization.enabled
        is True
    )

    assert (
        config.harmonization
        .target_sampling_rate_hz
        == 2
    )

    print(
        "✅ Harmonization validated"
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print(
        "\n===== CONFIG SUMMARY ====="
    )

    config.summary()

    print(
        "✅ DATASET CONFIG TEST PASSED"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()