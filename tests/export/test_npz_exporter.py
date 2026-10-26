"""
NPZ Export Contract Test.

Purpose
-------
Validate save -> load -> schema cycle.
"""

from pathlib import Path

import numpy as np

from configs.dataset_config import (
    DatasetConfig,
)

from datasets.export.npz_exporter import (
    NpzExporter,
)


CASE_CONFIG = DatasetConfig.load(
    "configs/datasets/case.yaml"
)


OUTPUT_FILE = Path(
    "artifacts/tests/"
    "case_export_test.npz"
)


def main():

    print(
        "\n====================================="
    )

    print(
        "NPZ EXPORT TEST"
    )

    print(
        "====================================="
    )

    exporter = NpzExporter()

    X = np.random.rand(
        120,
        8,
    ).astype(np.float32)

    y = np.random.rand(
        20,
        2,
    ).astype(np.float32)

    exporter.save(

        OUTPUT_FILE,

        X=X,
        y=y,

        feature_names=
        CASE_CONFIG.feature_columns,

        target_names=
        CASE_CONFIG.target_columns,

        subject_id=1,

        video_id=1,

        dataset_name=
        CASE_CONFIG.dataset.name,

        sampling_rate_hz=
        CASE_CONFIG.sampling_rate_hz,

        history_steps=
        CASE_CONFIG.history_steps,

        forecast_steps=
        CASE_CONFIG.forecast_steps,

        config_dict=
        CASE_CONFIG.raw,
    )

    print(
        "✅ Save successful"
    )

    assert (
        OUTPUT_FILE.exists()
    )

    assert (
        exporter.validate(
            OUTPUT_FILE
        )
    )

    print(
        "✅ Schema validated"
    )

    data = exporter.load(
        OUTPUT_FILE
    )

    assert (
        data["X"].shape
        ==
        (120, 8)
    )

    assert (
        data["y"].shape
        ==
        (20, 2)
    )

    print(
        "✅ Tensor shapes validated"
    )

    print(
        "\nStored Keys:"
    )

    print(
        sorted(
            data.files
        )
    )

    print(
        "\n✅ NPZ EXPORT TEST PASSED"
    )


if __name__ == "__main__":
    main()